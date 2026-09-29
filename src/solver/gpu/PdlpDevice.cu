#include "sovsolve/solver/gpu/PdlpDevice.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sovsolve::solver::gpu {

namespace {

using core::Status;

constexpr unsigned kBlock = 256;
/// Upper bound on the reduction grid. Fixed per problem size, never per run, so
/// the partial-sum order -- and therefore the result -- is reproducible.
constexpr unsigned kMaxReductionBlocks = 1024;

unsigned blocks_for(std::size_t len) {
  return static_cast<unsigned>((len + kBlock - 1) / kBlock);
}

// ---- kernels ---------------------------------------------------------------
//
// Each is the device form of a loop in HostIterationBackend, and the comments
// name which, so the two can be read side by side.

/// Algorithm 2 line 4, and the extrapolation `2x' - x`.
/// `fmin(fmax(v, l), u)` equals `std::clamp(v, l, u)` whenever `l <= u`, which
/// the canonicalizer guarantees; infinite bounds are the finite sentinel
/// `core::INF = 1e20`, so no infinity arithmetic is involved.
__global__ void primal_trial_kernel(std::size_t n, const Real* x, const Real* c,
                                    const Real* kty, const Real* lower,
                                    const Real* upper, Real tau, Real* x_trial,
                                    Real* extrapolated) {
  const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= n) return;
  const Real step = x[j] - tau * (c[j] - kty[j]);
  const Real v = fmin(fmax(step, lower[j]), upper[j]);
  x_trial[j] = v;
  extrapolated[j] = 2.0 * v - x[j];
}

/// Algorithm 2 line 5. Equality rows are unrestricted; inequality rows carry
/// `y <= 0` (FORMULATION.md section 4).
__global__ void dual_trial_kernel(std::size_t m, std::size_t num_equality,
                                  const Real* y, const Real* b, const Real* kext,
                                  Real sigma, Real* y_trial) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const Real step = y[i] + sigma * (b[i] - kext[i]);
  y_trial[i] = i < num_equality ? step : fmin(step, 0.0);
}

/// Algorithm 2 line 6, stage one: per-block partial sums of the three
/// quantities, written to `partials[3 * block + {0, 1, 2}]`.
///
/// `K dx = (K(2x'-x) - Kx) / 2` exactly as on the host, so `dy' K dx` costs no
/// product.
__global__ void metrics_partial_kernel(std::size_t n, std::size_t m, const Real* x,
                                       const Real* x_trial, const Real* y,
                                       const Real* y_trial, const Real* kext,
                                       const Real* kx_current, Real* partials) {
  __shared__ Real s_int[kBlock];
  __shared__ Real s_dx[kBlock];
  __shared__ Real s_dy[kBlock];

  Real interaction = 0.0;
  Real dx_sq = 0.0;
  Real dy_sq = 0.0;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t len = n > m ? n : m;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < len; k += stride) {
    if (k < m) {
      const Real dy = y_trial[k] - y[k];
      interaction += dy * (0.5 * (kext[k] - kx_current[k]));
      dy_sq += dy * dy;
    }
    if (k < n) {
      const Real dx = x_trial[k] - x[k];
      dx_sq += dx * dx;
    }
  }
  s_int[threadIdx.x] = interaction;
  s_dx[threadIdx.x] = dx_sq;
  s_dy[threadIdx.x] = dy_sq;
  __syncthreads();
  for (unsigned half = kBlock / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) {
      s_int[threadIdx.x] += s_int[threadIdx.x + half];
      s_dx[threadIdx.x] += s_dx[threadIdx.x + half];
      s_dy[threadIdx.x] += s_dy[threadIdx.x + half];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[3 * blockIdx.x + 0] = s_int[0];
    partials[3 * blockIdx.x + 1] = s_dx[0];
    partials[3 * blockIdx.x + 2] = s_dy[0];
  }
}

/// Stage two: ONE block sums the partials in a fixed order. Deterministic,
/// where atomics would not be -- see the header.
__global__ void metrics_final_kernel(unsigned blocks, const Real* partials, Real* out) {
  __shared__ Real s[3][kBlock];
  Real a = 0.0;
  Real b = 0.0;
  Real c = 0.0;
  for (unsigned k = threadIdx.x; k < blocks; k += kBlock) {
    a += partials[3 * k + 0];
    b += partials[3 * k + 1];
    c += partials[3 * k + 2];
  }
  s[0][threadIdx.x] = a;
  s[1][threadIdx.x] = b;
  s[2][threadIdx.x] = c;
  __syncthreads();
  for (unsigned half = kBlock / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) {
      s[0][threadIdx.x] += s[0][threadIdx.x + half];
      s[1][threadIdx.x] += s[1][threadIdx.x + half];
      s[2][threadIdx.x] += s[2][threadIdx.x + half];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[0] = s[0][0];
    out[1] = s[1][0];
    out[2] = s[2][0];
  }
}

/// `diff <- z - diff`, `sum <- sum + z` (HostIterationBackend::finish_difference).
__global__ void finish_difference_kernel(std::size_t len, const Real* z, Real* diff,
                                         Real* sum) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= len) return;
  diff[i] = z[i] - diff[i];
  sum[i] += z[i];
}

/// `acc <- acc + w z` (HostIterationBackend::accumulate_average).
__global__ void axpy_kernel(std::size_t len, Real w, const Real* z, Real* acc) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= len) return;
  acc[i] += w * z[i];
}

// ---- Module 31: reflected Halpern ------------------------------------------

/// Stage one of `||v||^2`: per-block partial sums written to the SAME slot
/// layout `metrics_final_kernel` reads (`partials[3 * block + 1]`), so the
/// deterministic fixed-order second stage is reused rather than duplicated.
/// Slots 0 and 2 are zeroed so the final kernel's other two outputs are
/// well-defined, not garbage.
__global__ void sum_squares_partial_kernel(std::size_t len, const Real* v,
                                           Real* partials) {
  __shared__ Real s[kBlock];
  Real acc = 0.0;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < len; k += stride) {
    acc += v[k] * v[k];
  }
  s[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned half = kBlock / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) s[threadIdx.x] += s[threadIdx.x + half];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[3 * blockIdx.x + 0] = 0.0;
    partials[3 * blockIdx.x + 1] = s[0];
    partials[3 * blockIdx.x + 2] = 0.0;
  }
}

/// `v <- v / s`. A DIVISION, not a multiply by `1/s`: the host routine
/// divides, and `v * (1/s)` differs from `v / s` in the last bit often enough
/// to send the two power iterations down different rounding paths.
__global__ void divide_kernel(std::size_t len, Real s, Real* v) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= len) return;
  v[i] /= s;
}

// ---- the controller on the device (HalpernControl.hpp) ---------------------
//
// The resident loop reads the step sizes and `lambda` from the controller's
// state in device memory instead of taking them as launch arguments, because
// the host no longer knows them between checks: a restart can move `omega` and
// reset `k` at any iteration, and only the device sees it happen.

/// `primal_trial_kernel` with `tau = eta / omega` read from device state.
__global__ void resident_primal_kernel(std::size_t n, const Real* x, const Real* c,
                                       const Real* kty, const Real* lower,
                                       const Real* upper, Real eta,
                                       const pdlp::HalpernState* state, Real* x_trial,
                                       Real* extrapolated) {
  const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= n) return;
  const Real tau = eta / state->omega;
  const Real step = x[j] - tau * (c[j] - kty[j]);
  const Real v = fmin(fmax(step, lower[j]), upper[j]);
  x_trial[j] = v;
  extrapolated[j] = 2.0 * v - x[j];
}

/// `dual_trial_kernel` with `sigma = eta * omega` read from device state.
__global__ void resident_dual_kernel(std::size_t m, std::size_t num_equality,
                                     const Real* y, const Real* b, const Real* kext,
                                     Real eta, const pdlp::HalpernState* state,
                                     Real* y_trial) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m) return;
  const Real sigma = eta * state->omega;
  const Real step = y[i] + sigma * (b[i] - kext[i]);
  y_trial[i] = i < num_equality ? step : fmin(step, 0.0);
}

/// `halpern_blend_kernel` with `lambda` from the epoch counter in device state.
__global__ void resident_blend_kernel(std::size_t len, Real gamma,
                                      const pdlp::HalpernState* state, const Real* trial,
                                      Real* current, const Real* base,
                                      const Real* image_trial, Real* image_current,
                                      const Real* image_base) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= len) return;
  const Real lambda = pdlp::halpern_lambda(*state);
  const Real reflected = lambda * (1.0 + gamma);
  const Real pull_back = lambda * gamma;
  const Real anchor = 1.0 - lambda;
  current[i] = reflected * trial[i] - pull_back * current[i] + anchor * base[i];
  image_current[i] =
      reflected * image_trial[i] - pull_back * image_current[i] + anchor * image_base[i];
}

/// ONE thread: `r(z)` from the step's three scalars, then the restart
/// decision. Exactly HostIterationBackend::run_halpern's two lines, from the
/// same header.
__global__ void controller_kernel(pdlp::HalpernParams params, pdlp::HalpernState* state,
                                  const Real* metrics) {
  const Real r = pdlp::halpern_fixed_point_error(params.eta, state->omega, metrics[0],
                                                 metrics[1], metrics[2]);
  // The iteration count advances HERE, on the device, rather than arriving
  // as a launch argument -- so this launch is byte-identical every
  // iteration and a captured graph can replay it.
  state->total += 1;
  (void)pdlp::halpern_observe(params, *state, r, state->total);
}

/// Seeds the device counter at the start of a chunk. Outside any graph.
__global__ void set_total_kernel(pdlp::HalpernState* state, std::uint64_t total) {
  state->total = total;
}

/// Everything a restart needs to know about the restart point `T(z)`, in ONE
/// pass, stage one:
///
///   [0] ||T(z)_x - x_0||^2        distance moved, primal   (PID and HPR-LP)
///   [1] ||T(z)_y - y_0||^2        distance moved, dual
///   [2] ||Pi_D(b - A T(z)_x)||^2  primal infeasibility     (HPR-LP (18))
///   [3] ||c - A' T(z)_y - z||^2   dual infeasibility
///
/// The last two use the images `K T(z)_x` and `K' T(z)_y` already held from
/// the step, so no product. With a non-null `state` it runs ONLY when a
/// restart is pending -- every other iteration each block reads one flag and
/// leaves; with a null `state` (the per-step `restart_at_pdhg_point`) it
/// always runs. The per-element terms come from HalpernControl.hpp, the same
/// functions the host backend uses.
__global__ void restart_partial_kernel(std::size_t n, std::size_t m,
                                       std::size_t num_equality,
                                       const pdlp::HalpernState* state,
                                       const Real* x_trial, const Real* x_anchor,
                                       const Real* y_trial, const Real* y_anchor,
                                       const Real* kx_trial, const Real* kty_trial,
                                       const Real* b, const Real* c, const Real* lower,
                                       const Real* upper, Real* partials) {
  if (state != nullptr && state->restart_pending == 0) return;
  __shared__ Real s[4][kBlock];
  Real acc[4] = {0.0, 0.0, 0.0, 0.0};
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t len = n > m ? n : m;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < len; k += stride) {
    if (k < n) {
      const Real d = x_trial[k] - x_anchor[k];
      acc[0] += d * d;
      const Real v = pdlp::halpern_dual_leftover(c[k] - kty_trial[k], lower[k], upper[k]);
      acc[3] += v * v;
    }
    if (k < m) {
      const Real d = y_trial[k] - y_anchor[k];
      acc[1] += d * d;
      const Real v = pdlp::halpern_primal_violation(kx_trial[k], b[k], k < num_equality);
      acc[2] += v * v;
    }
  }
  for (int q = 0; q < 4; ++q) s[q][threadIdx.x] = acc[q];
  __syncthreads();
  for (unsigned half = kBlock / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) {
      for (int q = 0; q < 4; ++q) s[q][threadIdx.x] += s[q][threadIdx.x + half];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int q = 0; q < 4; ++q) partials[4 * blockIdx.x + q] = s[q][0];
  }
}

/// Stage two: one block sums the four partials in a FIXED order -- the same
/// determinism argument as `metrics_final_kernel`. Guarded like stage one.
__global__ void restart_final_kernel(unsigned blocks, const pdlp::HalpernState* state,
                                     const Real* partials, Real* out) {
  if (state != nullptr && state->restart_pending == 0) return;
  __shared__ Real s[4][kBlock];
  Real acc[4] = {0.0, 0.0, 0.0, 0.0};
  for (unsigned k = threadIdx.x; k < blocks; k += kBlock) {
    for (int q = 0; q < 4; ++q) acc[q] += partials[4 * k + q];
  }
  for (int q = 0; q < 4; ++q) s[q][threadIdx.x] = acc[q];
  __syncthreads();
  for (unsigned half = kBlock / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) {
      for (int q = 0; q < 4; ++q) s[q][threadIdx.x] += s[q][threadIdx.x + half];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int q = 0; q < 4; ++q) out[q] = s[q][0];
  }
}

/// The termination check's five sums over T(z) and its two images, in place
/// (IterationBackend::evaluate_resident). Term by term the host `evaluate` in
/// Pdlp.cpp: `inv_row`/`inv_col` are the preconditioner's inverse factors, or
/// null for the problem as given. Two stages with a fixed-order final sum, as
/// every other reduction here, so the result is bit-for-bit repeatable.
constexpr int kEvalSums = 5;

__global__ void eval_partial_kernel(std::size_t n, std::size_t m, std::size_t num_equality,
                                    const Real* x_trial, const Real* y_trial,
                                    const Real* kx_trial, const Real* kty_trial, const Real* b,
                                    const Real* c, const Real* lower, const Real* upper,
                                    const Real* inv_row, const Real* inv_col, Real* partials) {
  __shared__ Real s[kEvalSums][kBlock];
  Real acc[kEvalSums] = {0.0, 0.0, 0.0, 0.0, 0.0};
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t len = n > m ? n : m;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       k < len; k += stride) {
    if (k < n) {
      const Real raw = c[k] - kty_trial[k];
      const Real lambda = pdlp::halpern_absorbed(raw, lower[k], upper[k]);
      const Real leftover = (raw - lambda) * (inv_col != nullptr ? inv_col[k] : 1.0);
      acc[4] += leftover * leftover;
      if (lambda > 0.0 && lower[k] > -core::INF) {
        acc[2] += lower[k] * lambda;
      } else if (lambda < 0.0 && upper[k] < core::INF) {
        acc[2] += upper[k] * lambda;
      }
      acc[0] += c[k] * x_trial[k];
    }
    if (k < m) {
      const Real v = pdlp::halpern_primal_violation(kx_trial[k], b[k], k < num_equality) *
                     (inv_row != nullptr ? inv_row[k] : 1.0);
      acc[3] += v * v;
      acc[1] += b[k] * y_trial[k];
    }
  }
  for (int q = 0; q < kEvalSums; ++q) s[q][threadIdx.x] = acc[q];
  __syncthreads();
  for (unsigned half = kBlock / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) {
      for (int q = 0; q < kEvalSums; ++q) s[q][threadIdx.x] += s[q][threadIdx.x + half];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int q = 0; q < kEvalSums; ++q) partials[kEvalSums * blockIdx.x + q] = s[q][0];
  }
}

__global__ void eval_final_kernel(unsigned blocks, const Real* partials, Real* out) {
  __shared__ Real s[kEvalSums][kBlock];
  Real acc[kEvalSums] = {0.0, 0.0, 0.0, 0.0, 0.0};
  for (unsigned k = threadIdx.x; k < blocks; k += kBlock) {
    for (int q = 0; q < kEvalSums; ++q) acc[q] += partials[kEvalSums * k + q];
  }
  for (int q = 0; q < kEvalSums; ++q) s[q][threadIdx.x] = acc[q];
  __syncthreads();
  for (unsigned half = kBlock / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) {
      for (int q = 0; q < kEvalSums; ++q) s[q][threadIdx.x] += s[q][threadIdx.x + half];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (int q = 0; q < kEvalSums; ++q) out[q] = s[q][0];
  }
}

/// ONE thread: the controller's half of a restart, when pending -- HPR-LP's
/// rule or the PID, per `params.weight_rule`. `sums` is `restart_final_kernel`'s
/// output for this iteration.
__global__ void restart_controller_kernel(pdlp::HalpernParams params,
                                          pdlp::HalpernState* state, const Real* sums) {
  if (state->restart_pending == 0) return;
  pdlp::halpern_on_restart(params, *state, sqrt(sums[0]), sqrt(sums[1]),
                           sqrt(sums[2]) / (1.0 + params.b_norm),
                           sqrt(sums[3]) / (1.0 + params.c_norm));
}

/// The iterate's half of a restart, when pending: `z <- z_0 <- T(z)`, and the
/// same for both matrix images. Runs AFTER `restart_controller_kernel`, which
/// is why `halpern_on_restart` leaves the flag set.
__global__ void restart_move_kernel(std::size_t n, std::size_t m,
                                    const pdlp::HalpernState* state, const Real* x_trial,
                                    const Real* kty_trial, const Real* y_trial,
                                    const Real* kx_trial, Real* x, Real* kty,
                                    Real* x_anchor, Real* kty_anchor, Real* y, Real* kx,
                                    Real* y_anchor, Real* kx_anchor) {
  if (state->restart_pending == 0) return;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = x_trial[i];
    x_anchor[i] = x_trial[i];
    kty[i] = kty_trial[i];
    kty_anchor[i] = kty_trial[i];
  }
  if (i < m) {
    y[i] = y_trial[i];
    y_anchor[i] = y_trial[i];
    kx[i] = kx_trial[i];
    kx_anchor[i] = kx_trial[i];
  }
}

/// `K T(z)_x = (K(2x' - x) + K x) / 2` -- the trial point's image, recovered
/// from the product already taken (HostIterationBackend::halpern_step).
__global__ void trial_image_kernel(std::size_t m, const Real* kext,
                                   const Real* kx_current, Real* kx_trial) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m) return;
  kx_trial[i] = 0.5 * (kext[i] + kx_current[i]);
}

/// The Halpern blend, applied to a vector AND its matrix image in one pass:
///
///     v   <- reflected * v_trial   - pull_back * v   + anchor * v_0
///     Kv  <- reflected * Kv_trial  - pull_back * Kv  + anchor * Kv_0
///
/// One kernel for both because they are the same line applied to two
/// operands, and fusing them reads `reflected`/`pull_back`/`anchor` once and
/// launches once. Linearity of `K` is what makes the second line exact.
__global__ void halpern_blend_kernel(std::size_t len, Real reflected, Real pull_back,
                                     Real anchor, const Real* trial, Real* current,
                                     const Real* base, const Real* image_trial,
                                     Real* image_current, const Real* image_base) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= len) return;
  current[i] = reflected * trial[i] - pull_back * current[i] + anchor * base[i];
  image_current[i] =
      reflected * image_trial[i] - pull_back * image_current[i] + anchor * image_base[i];
}

}  // namespace

// ---- state -----------------------------------------------------------------

struct DevicePdlpBackend::Impl {
  std::size_t n = 0;
  std::size_t m = 0;
  std::size_t num_equality = 0;

  cusparseHandle_t handle = nullptr;
  cusparseSpMatDescr_t mat_a = nullptr;   ///< A, m x n
  cusparseSpMatDescr_t mat_at = nullptr;  ///< A', n x m -- the CSC read as CSR
  core::Index* a_offsets = nullptr;
  core::Index* a_indices = nullptr;
  Real* a_values = nullptr;
  core::Index* at_offsets = nullptr;
  core::Index* at_indices = nullptr;
  Real* at_values = nullptr;

  // Problem data, uploaded once.
  Real* c = nullptr;
  Real* b = nullptr;
  Real* lower = nullptr;
  Real* upper = nullptr;

  // Hot-path vectors. `x`/`x_trial` and `y`/`y_trial` are SWAPPED on accept,
  // not copied, which is why SpMV descriptors are rebound per call below.
  Real* x = nullptr;
  Real* y = nullptr;
  Real* x_trial = nullptr;
  Real* y_trial = nullptr;
  Real* extrapolated = nullptr;
  Real* kty = nullptr;
  Real* kx_current = nullptr;
  Real* kext = nullptr;
  Real* diff_x = nullptr;
  Real* diff_y = nullptr;
  Real* sum_x = nullptr;
  Real* sum_y = nullptr;
  Real* avg_x = nullptr;
  Real* avg_y = nullptr;

  // Module 31. `x_trial`/`y_trial` above double as `T(z)`.
  Real* kty_trial = nullptr;   ///< `K' T(z)_y`
  Real* kx_trial = nullptr;    ///< `K T(z)_x`
  Real* x_anchor = nullptr;    ///< `z^{n,0}` and its two images
  Real* y_anchor = nullptr;
  Real* kty_anchor = nullptr;
  Real* kx_anchor = nullptr;
  /// The controller's state (HalpernControl.hpp), resident. Read back only at
  /// a termination check.
  pdlp::HalpernState* control = nullptr;
  /// The anchor-distance reduction's output, kept apart from `metrics` so a
  /// restart's reduction can never be mistaken for the step's.
  Real* restart_metrics = nullptr;
  /// Four slots per reduction block for `restart_partial_kernel`.
  Real* restart_partials = nullptr;

  // ---- CUDA graphs over the resident loop --------------------------------
  //
  // The legacy default stream cannot be captured, so the resident loop runs
  // on a stream of its own, and cuSPARSE is pointed at it. It is created
  // BLOCKING (plain cudaStreamCreate): the legacy stream synchronizes with it
  // implicitly, so every existing `cudaMemcpy` on the cold path still waits
  // for the resident work before reading -- no new synchronization points.
  cudaStream_t stream = nullptr;
  bool use_graphs = true;
  /// Set once when a capture fails; the loop then launches directly for the
  /// rest of this backend's life, and `graph_note` says why.
  bool graphs_unavailable = false;
  std::string graph_note;
  /// Two cached executables: one whole check interval (the common case, one
  /// launch per chunk) and one single iteration (for chunks of any other
  /// length, replayed `count` times).
  cudaGraphExec_t graph_chunk = nullptr;
  cudaGraphExec_t graph_single = nullptr;
  std::size_t graph_chunk_length = 0;
  /// What a captured graph baked in. Kernel arguments are recorded by VALUE
  /// at capture -- the parameters, the tracking choice, and every buffer
  /// address -- so a change in any of them makes the executables stale.
  /// `x`/`x_trial` and `y`/`y_trial` are swapped by `accept_trial` on the
  /// averaged path, which is exactly how an address could change under a
  /// cached graph.
  struct GraphKey {
    pdlp::HalpernParams params{};
    bool track = false;
    const Real* x = nullptr;
    const Real* y = nullptr;
    const Real* x_trial = nullptr;
    const Real* y_trial = nullptr;
  } graph_key;
  bool graph_key_set = false;
  std::size_t graph_launches = 0;  ///< launches actually issued, for the A/B
  std::size_t graph_captures = 0;

  void drop_graphs() {
    if (graph_chunk) cudaGraphExecDestroy(graph_chunk);
    if (graph_single) cudaGraphExecDestroy(graph_single);
    graph_chunk = nullptr;
    graph_single = nullptr;
    graph_chunk_length = 0;
  }

  // Cold-path staging, so a cold product never clobbers hot-path state.
  Real* cold_in_n = nullptr;
  Real* cold_out_m = nullptr;
  Real* cold_in_m = nullptr;
  Real* cold_out_n = nullptr;

  // Dense-vector descriptors, rebound to whichever buffer a call needs.
  cusparseDnVecDescr_t vec_n_in = nullptr;
  cusparseDnVecDescr_t vec_m_out = nullptr;
  cusparseDnVecDescr_t vec_m_in = nullptr;
  cusparseDnVecDescr_t vec_n_out = nullptr;
  void* buffer_a = nullptr;
  void* buffer_at = nullptr;

  unsigned reduction_blocks = 1;
  Real* partials = nullptr;
  Real* metrics = nullptr;
  Real* host_metrics = nullptr;  ///< pinned, so the 24-byte copy is a DMA

  // evaluate_resident: its own partials and outputs, the inverse scale
  // factors uploaded once (keyed by the host spans they came from).
  Real* eval_partials = nullptr;
  Real* eval_out = nullptr;
  Real* eval_host = nullptr;  ///< pinned, 5 doubles
  Real* inv_row = nullptr;
  Real* inv_col = nullptr;
  const Real* inv_row_src = nullptr;
  const Real* inv_col_src = nullptr;

  std::size_t hot_products = 0;
  std::size_t bytes = 0;
  Status sticky = Status::Ok();

  void record(cudaError_t err, const char* what) {
    if (err != cudaSuccess && sticky.ok()) {
      sticky = core::make_error(core::ErrorCode::NumericalError,
                                std::string(what) + ": " + cudaGetErrorString(err));
    }
  }
  void record_sparse(cusparseStatus_t st, const char* what) {
    if (st != CUSPARSE_STATUS_SUCCESS && sticky.ok()) {
      sticky = core::make_error(core::ErrorCode::NumericalError,
                                std::string(what) + " failed (cusparse status " +
                                    std::to_string(static_cast<int>(st)) + ")");
    }
  }
  void check_launch(const char* what) { record(cudaGetLastError(), what); }

  /// `out = A in` (length n -> m) or `out = A' in` (m -> n).
  void spmv(bool transpose, const Real* in, Real* out) {
    const Real one = 1.0;
    const Real zero = 0.0;
    if (!transpose) {
      record_sparse(cusparseDnVecSetValues(vec_n_in, const_cast<Real*>(in)), "bind in");
      record_sparse(cusparseDnVecSetValues(vec_m_out, out), "bind out");
      record_sparse(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_a,
                                 vec_n_in, &zero, vec_m_out, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, buffer_a),
                    "cusparseSpMV(A)");
    } else {
      record_sparse(cusparseDnVecSetValues(vec_m_in, const_cast<Real*>(in)), "bind in");
      record_sparse(cusparseDnVecSetValues(vec_n_out, out), "bind out");
      record_sparse(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_at,
                                 vec_m_in, &zero, vec_n_out, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, buffer_at),
                    "cusparseSpMV(A')");
    }
  }

  ~Impl() {
    drop_graphs();
    if (stream) cudaStreamDestroy(stream);
    for (auto d : {vec_n_in, vec_m_out, vec_m_in, vec_n_out}) {
      if (d) cusparseDestroyDnVec(d);
    }
    if (mat_a) cusparseDestroySpMat(mat_a);
    if (mat_at) cusparseDestroySpMat(mat_at);
    if (handle) cusparseDestroy(handle);
    for (void* p : {static_cast<void*>(a_offsets), static_cast<void*>(a_indices),
                    static_cast<void*>(a_values), static_cast<void*>(at_offsets),
                    static_cast<void*>(at_indices), static_cast<void*>(at_values),
                    static_cast<void*>(c), static_cast<void*>(b),
                    static_cast<void*>(lower), static_cast<void*>(upper),
                    static_cast<void*>(x), static_cast<void*>(y),
                    static_cast<void*>(x_trial), static_cast<void*>(y_trial),
                    static_cast<void*>(extrapolated), static_cast<void*>(kty),
                    static_cast<void*>(kx_current), static_cast<void*>(kext),
                    static_cast<void*>(diff_x), static_cast<void*>(diff_y),
                    static_cast<void*>(sum_x), static_cast<void*>(sum_y),
                    static_cast<void*>(avg_x), static_cast<void*>(avg_y),
                    static_cast<void*>(kty_trial), static_cast<void*>(kx_trial),
                    static_cast<void*>(x_anchor), static_cast<void*>(y_anchor),
                    static_cast<void*>(kty_anchor), static_cast<void*>(kx_anchor),
                    static_cast<void*>(control), static_cast<void*>(restart_metrics),
                    static_cast<void*>(restart_partials),
                    static_cast<void*>(cold_in_n), static_cast<void*>(cold_out_m),
                    static_cast<void*>(cold_in_m), static_cast<void*>(cold_out_n),
                    buffer_a, buffer_at, static_cast<void*>(partials),
                    static_cast<void*>(metrics), static_cast<void*>(eval_partials),
                    static_cast<void*>(eval_out), static_cast<void*>(inv_row),
                    static_cast<void*>(inv_col)}) {
      cudaFree(p);
    }
    if (host_metrics) cudaFreeHost(host_metrics);
    if (eval_host) cudaFreeHost(eval_host);
  }
};

// ---- the cold path's MatVec, over the same device matrix ---------------------

namespace {

class ColdMatVec final : public pdlp::MatVec {
 public:
  explicit ColdMatVec(DevicePdlpBackend::Impl* d) : d_(d) {}

  void multiply(core::HostSpan<const Real> x, core::HostSpan<Real> out) override {
    d_->record(cudaMemcpy(d_->cold_in_n, x.data(), d_->n * sizeof(Real),
                          cudaMemcpyHostToDevice),
               "cold upload x");
    d_->spmv(false, d_->cold_in_n, d_->cold_out_m);
    d_->record(cudaMemcpy(out.data(), d_->cold_out_m, d_->m * sizeof(Real),
                          cudaMemcpyDeviceToHost),
               "cold download Kx");
    ++products_;
  }

  void multiply_transpose(core::HostSpan<const Real> y,
                          core::HostSpan<Real> out) override {
    d_->record(cudaMemcpy(d_->cold_in_m, y.data(), d_->m * sizeof(Real),
                          cudaMemcpyHostToDevice),
               "cold upload y");
    d_->spmv(true, d_->cold_in_m, d_->cold_out_n);
    d_->record(cudaMemcpy(out.data(), d_->cold_out_n, d_->n * sizeof(Real),
                          cudaMemcpyDeviceToHost),
               "cold download K'y");
    ++products_;
  }

  [[nodiscard]] std::size_t num_rows() const override { return d_->m; }
  [[nodiscard]] std::size_t num_cols() const override { return d_->n; }

 private:
  DevicePdlpBackend::Impl* d_;
};

}  // namespace

// ---- construction ----------------------------------------------------------

DevicePdlpBackend::DevicePdlpBackend(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)), cold_(std::make_unique<ColdMatVec>(impl_.get())) {}

DevicePdlpBackend::~DevicePdlpBackend() {
  cold_.reset();  // holds a pointer into impl_, so it goes first
}

core::Expected<std::unique_ptr<DevicePdlpBackend>> DevicePdlpBackend::create(
    const model::CanonicalProblem& problem) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    return core::make_error(core::ErrorCode::NotImplemented,
                            "DevicePdlpBackend::create: no CUDA device is available");
  }

  auto d = std::make_unique<Impl>();
  d->n = problem.num_cols();
  d->m = problem.num_rows();
  d->num_equality = problem.num_equality;

  const auto& csr = problem.A.csr;
  const auto& csc = problem.A.csc;
  const std::size_t nnz = csr.values().size();
  if (nnz != csc.values().size()) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "DevicePdlpBackend::create: CSR and CSC disagree on nnz");
  }

  auto fail = [](cudaError_t err, const char* what) -> Status {
    return core::make_error(core::ErrorCode::OutOfMemory,
                            std::string(what) + ": " + cudaGetErrorString(err));
  };
  auto alloc = [&](auto** ptr, std::size_t count, const char* what) -> Status {
    const std::size_t bytes = count * sizeof(**ptr);
    const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(ptr), bytes > 0 ? bytes : 8);
    if (err != cudaSuccess) return fail(err, what);
    d->bytes += bytes;
    const cudaError_t z = cudaMemset(*ptr, 0, bytes > 0 ? bytes : 8);
    if (z != cudaSuccess) return fail(z, what);
    return Status::Ok();
  };
  auto upload = [&](auto** ptr, const auto* src, std::size_t count,
                    const char* what) -> Status {
    Status st = alloc(ptr, count, what);
    if (!st.ok()) return st;
    if (count == 0) return Status::Ok();
    const cudaError_t err =
        cudaMemcpy(*ptr, src, count * sizeof(**ptr), cudaMemcpyHostToDevice);
    return err == cudaSuccess ? Status::Ok() : fail(err, what);
  };

  const std::size_t n = d->n;
  const std::size_t m = d->m;
  Status st = Status::Ok();
  // clang-format off
  if (st.ok()) st = upload(&d->a_offsets, csr.offsets().data(), m + 1, "A offsets");
  if (st.ok()) st = upload(&d->a_indices, csr.indices().data(), nnz, "A indices");
  if (st.ok()) st = upload(&d->a_values, csr.values().data(), nnz, "A values");
  if (st.ok()) st = upload(&d->at_offsets, csc.offsets().data(), n + 1, "A' offsets");
  if (st.ok()) st = upload(&d->at_indices, csc.indices().data(), nnz, "A' indices");
  if (st.ok()) st = upload(&d->at_values, csc.values().data(), nnz, "A' values");
  if (st.ok()) st = upload(&d->c, problem.c.data(), n, "c");
  if (st.ok()) st = upload(&d->b, problem.b.data(), m, "b");
  if (st.ok()) st = upload(&d->lower, problem.col_lower.data(), n, "lower bounds");
  if (st.ok()) st = upload(&d->upper, problem.col_upper.data(), n, "upper bounds");
  for (Real** p : {&d->x, &d->x_trial, &d->extrapolated, &d->kty, &d->diff_x,
                   &d->sum_x, &d->avg_x, &d->cold_in_n, &d->cold_out_n,
                   &d->kty_trial, &d->x_anchor, &d->kty_anchor}) {
    if (st.ok()) st = alloc(p, n, "n-vector");
  }
  for (Real** p : {&d->y, &d->y_trial, &d->kx_current, &d->kext, &d->diff_y,
                   &d->sum_y, &d->avg_y, &d->cold_in_m, &d->cold_out_m,
                   &d->kx_trial, &d->y_anchor, &d->kx_anchor}) {
    if (st.ok()) st = alloc(p, m, "m-vector");
  }
  // clang-format on
  if (!st.ok()) return st.error();

  d->reduction_blocks =
      std::max(1u, std::min(kMaxReductionBlocks, blocks_for(std::max(n, m))));
  st = alloc(&d->partials, 3 * static_cast<std::size_t>(d->reduction_blocks), "partials");
  if (st.ok()) st = alloc(&d->metrics, 3, "metrics");
  if (st.ok()) st = alloc(&d->restart_metrics, 4, "restart metrics");
  if (st.ok()) {
    st = alloc(&d->restart_partials, 4 * static_cast<std::size_t>(d->reduction_blocks),
               "restart partials");
  }
  if (st.ok()) st = alloc(&d->control, 1, "halpern controller state");
  if (!st.ok()) return st.error();
  if (cudaMallocHost(reinterpret_cast<void**>(&d->host_metrics), 3 * sizeof(Real)) !=
      cudaSuccess) {
    return core::make_error(core::ErrorCode::OutOfMemory, "pinned metrics buffer");
  }

  auto sparse = [](cusparseStatus_t s, const char* what) -> Status {
    return s == CUSPARSE_STATUS_SUCCESS
               ? Status::Ok()
               : core::make_error(core::ErrorCode::NumericalError,
                                  std::string(what) + " failed (cusparse status " +
                                      std::to_string(static_cast<int>(s)) + ")");
  };
  const auto mi = static_cast<std::int64_t>(m);
  const auto ni = static_cast<std::int64_t>(n);
  const auto nnzi = static_cast<std::int64_t>(nnz);
  if (cudaStreamCreate(&d->stream) != cudaSuccess) {
    return core::make_error(core::ErrorCode::NotImplemented,
                            "DevicePdlpBackend::create: cudaStreamCreate failed");
  }
  st = sparse(cusparseCreate(&d->handle), "cusparseCreate");
  if (st.ok()) st = sparse(cusparseSetStream(d->handle, d->stream), "cusparseSetStream");
  if (st.ok()) {
    st = sparse(cusparseCreateCsr(&d->mat_a, mi, ni, nnzi, d->a_offsets, d->a_indices,
                                  d->a_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                "cusparseCreateCsr(A)");
  }
  if (st.ok()) {
    st = sparse(cusparseCreateCsr(&d->mat_at, ni, mi, nnzi, d->at_offsets, d->at_indices,
                                  d->at_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                "cusparseCreateCsr(A')");
  }
  if (st.ok()) st = sparse(cusparseCreateDnVec(&d->vec_n_in, ni, d->x, CUDA_R_64F), "vec");
  if (st.ok()) st = sparse(cusparseCreateDnVec(&d->vec_m_out, mi, d->kext, CUDA_R_64F), "vec");
  if (st.ok()) st = sparse(cusparseCreateDnVec(&d->vec_m_in, mi, d->y, CUDA_R_64F), "vec");
  if (st.ok()) st = sparse(cusparseCreateDnVec(&d->vec_n_out, ni, d->kty, CUDA_R_64F), "vec");
  if (!st.ok()) return st.error();

  // Size both workspaces ONCE. Rebinding a vector's values later does not
  // change what the buffer depends on (the matrix and the algorithm).
  const Real one = 1.0;
  const Real zero = 0.0;
  std::size_t bytes_a = 0;
  std::size_t bytes_at = 0;
  st = sparse(cusparseSpMV_bufferSize(d->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                      d->mat_a, d->vec_n_in, &zero, d->vec_m_out,
                                      CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes_a),
              "bufferSize(A)");
  if (st.ok()) {
    st = sparse(cusparseSpMV_bufferSize(d->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                        d->mat_at, d->vec_m_in, &zero, d->vec_n_out,
                                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes_at),
                "bufferSize(A')");
  }
  if (!st.ok()) return st.error();
  if (bytes_a > 0) {
    if (cudaMalloc(&d->buffer_a, bytes_a) != cudaSuccess) {
      return core::make_error(core::ErrorCode::OutOfMemory, "SpMV buffer A");
    }
    d->bytes += bytes_a;
  }
  if (bytes_at > 0) {
    if (cudaMalloc(&d->buffer_at, bytes_at) != cudaSuccess) {
      return core::make_error(core::ErrorCode::OutOfMemory, "SpMV buffer A'");
    }
    d->bytes += bytes_at;
  }

  return std::unique_ptr<DevicePdlpBackend>(new DevicePdlpBackend(std::move(d)));
}

// ---- IterationBackend ------------------------------------------------------

void DevicePdlpBackend::set_iterate(core::HostSpan<const Real> x,
                                    core::HostSpan<const Real> y) {
  Impl& d = *impl_;
  d.record(cudaMemcpy(d.x, x.data(), d.n * sizeof(Real), cudaMemcpyHostToDevice),
           "upload x");
  d.record(cudaMemcpy(d.y, y.data(), d.m * sizeof(Real), cudaMemcpyHostToDevice),
           "upload y");
}

void DevicePdlpBackend::download(pdlp::BackendVector which, core::HostSpan<Real> out) {
  Impl& d = *impl_;
  const Real* src = nullptr;
  std::size_t len = 0;
  switch (which) {
    case pdlp::BackendVector::X: src = d.x; len = d.n; break;
    case pdlp::BackendVector::Y: src = d.y; len = d.m; break;
    case pdlp::BackendVector::AverageX: src = d.avg_x; len = d.n; break;
    case pdlp::BackendVector::AverageY: src = d.avg_y; len = d.m; break;
    case pdlp::BackendVector::IterateSumX: src = d.sum_x; len = d.n; break;
    case pdlp::BackendVector::IterateSumY: src = d.sum_y; len = d.m; break;
    case pdlp::BackendVector::DifferenceX: src = d.diff_x; len = d.n; break;
    case pdlp::BackendVector::DifferenceY: src = d.diff_y; len = d.m; break;
    // Module 31: `T(z)` from the most recent `halpern_step`.
    case pdlp::BackendVector::PdhgX: src = d.x_trial; len = d.n; break;
    case pdlp::BackendVector::PdhgY: src = d.y_trial; len = d.m; break;
  }
  if (len == 0) return;
  d.record(cudaMemcpy(out.data(), src, len * sizeof(Real), cudaMemcpyDeviceToHost),
           "download");
}

void DevicePdlpBackend::begin_step() {
  Impl& d = *impl_;
  d.spmv(true, d.y, d.kty);
  d.spmv(false, d.x, d.kx_current);
  d.hot_products += 2;
}

pdlp::TrialMetrics DevicePdlpBackend::trial(Real tau, Real sigma) {
  Impl& d = *impl_;
  if (d.n > 0) {
    primal_trial_kernel<<<blocks_for(d.n), kBlock>>>(d.n, d.x, d.c, d.kty, d.lower,
                                                     d.upper, tau, d.x_trial,
                                                     d.extrapolated);
    d.check_launch("primal_trial_kernel");
  }
  d.spmv(false, d.extrapolated, d.kext);
  if (d.m > 0) {
    dual_trial_kernel<<<blocks_for(d.m), kBlock>>>(d.m, d.num_equality, d.y, d.b, d.kext,
                                                   sigma, d.y_trial);
    d.check_launch("dual_trial_kernel");
  }
  metrics_partial_kernel<<<d.reduction_blocks, kBlock>>>(
      d.n, d.m, d.x, d.x_trial, d.y, d.y_trial, d.kext, d.kx_current, d.partials);
  d.check_launch("metrics_partial_kernel");
  metrics_final_kernel<<<1, kBlock>>>(d.reduction_blocks, d.partials, d.metrics);
  d.check_launch("metrics_final_kernel");
  ++d.hot_products;

  // The one synchronization in the hot loop, and 24 bytes of it: Algorithm 2's
  // accept/reject is a host decision on these three numbers.
  d.record(cudaMemcpy(d.host_metrics, d.metrics, 3 * sizeof(Real), cudaMemcpyDeviceToHost),
           "download metrics");
  pdlp::TrialMetrics metrics;
  metrics.interaction = d.host_metrics[0];
  metrics.dx_sq = d.host_metrics[1];
  metrics.dy_sq = d.host_metrics[2];
  return metrics;
}

void DevicePdlpBackend::accept_trial() {
  // A pointer swap, not a copy. The SpMV descriptors are rebound on every call,
  // so nothing holds onto the old addresses.
  std::swap(impl_->x, impl_->x_trial);
  std::swap(impl_->y, impl_->y_trial);
}

void DevicePdlpBackend::fixed_step(Real tau, Real sigma) {
  // Equation (3) is Algorithm 2's trial with no acceptance test: `K'y`, the
  // primal step and extrapolation, `K(2x'-x)`, the dual step, then commit.
  Impl& d = *impl_;
  d.spmv(true, d.y, d.kty);
  if (d.n > 0) {
    primal_trial_kernel<<<blocks_for(d.n), kBlock>>>(d.n, d.x, d.c, d.kty, d.lower,
                                                     d.upper, tau, d.x_trial,
                                                     d.extrapolated);
    d.check_launch("primal_trial_kernel");
  }
  d.spmv(false, d.extrapolated, d.kext);
  if (d.m > 0) {
    dual_trial_kernel<<<blocks_for(d.m), kBlock>>>(d.m, d.num_equality, d.y, d.b, d.kext,
                                                   sigma, d.y_trial);
    d.check_launch("dual_trial_kernel");
  }
  d.hot_products += 2;
  accept_trial();
}

void DevicePdlpBackend::snapshot_iterate() {
  Impl& d = *impl_;
  d.record(cudaMemcpyAsync(d.diff_x, d.x, d.n * sizeof(Real), cudaMemcpyDeviceToDevice),
           "snapshot x");
  d.record(cudaMemcpyAsync(d.diff_y, d.y, d.m * sizeof(Real), cudaMemcpyDeviceToDevice),
           "snapshot y");
}

void DevicePdlpBackend::finish_difference() {
  Impl& d = *impl_;
  if (d.n > 0) {
    finish_difference_kernel<<<blocks_for(d.n), kBlock>>>(d.n, d.x, d.diff_x, d.sum_x);
    d.check_launch("finish_difference_kernel(x)");
  }
  if (d.m > 0) {
    finish_difference_kernel<<<blocks_for(d.m), kBlock>>>(d.m, d.y, d.diff_y, d.sum_y);
    d.check_launch("finish_difference_kernel(y)");
  }
}

void DevicePdlpBackend::accumulate_average(Real weight) {
  Impl& d = *impl_;
  if (d.n > 0) {
    axpy_kernel<<<blocks_for(d.n), kBlock>>>(d.n, weight, d.x, d.avg_x);
    d.check_launch("axpy_kernel(x)");
  }
  if (d.m > 0) {
    axpy_kernel<<<blocks_for(d.m), kBlock>>>(d.m, weight, d.y, d.avg_y);
    d.check_launch("axpy_kernel(y)");
  }
}

void DevicePdlpBackend::reset_average() {
  Impl& d = *impl_;
  d.record(cudaMemsetAsync(d.avg_x, 0, d.n * sizeof(Real)), "reset avg x");
  d.record(cudaMemsetAsync(d.avg_y, 0, d.m * sizeof(Real)), "reset avg y");
}

std::optional<Real> DevicePdlpBackend::spectral_norm(std::size_t iterations,
                                                      Real tolerance) {
  Impl& d = *impl_;
  if (d.n == 0) return 0.0;
  // The host routine's seed, built on the host and uploaded ONCE, so both
  // paths start from bit-identical vectors: `1 + 0.1 (j mod 7)`, normalized.
  // The normalization is done here in the same order as the host's
  // `euclidean_norm` so even the first step agrees exactly.
  std::vector<Real> seed(d.n);
  Real norm_sq = 0.0;
  for (std::size_t j = 0; j < d.n; ++j) {
    seed[j] = 1.0 + static_cast<Real>(j % 7) * 0.1;
    norm_sq += seed[j] * seed[j];
  }
  const Real norm = std::sqrt(norm_sq);
  for (std::size_t j = 0; j < d.n; ++j) seed[j] /= norm;

  // The cold path's staging buffers: nothing else is live during start-up,
  // and borrowing them costs no device memory.
  Real* v = d.cold_in_n;
  Real* kv = d.cold_out_m;
  d.record(cudaMemcpy(v, seed.data(), d.n * sizeof(Real), cudaMemcpyHostToDevice),
           "upload power seed");

  Real sigma = 0.0;
  for (std::size_t it = 0; it < iterations; ++it) {
    d.spmv(false, v, kv);
    d.spmv(true, kv, v);
    d.hot_products += 2;
    sum_squares_partial_kernel<<<d.reduction_blocks, kBlock>>>(d.n, v, d.partials);
    d.check_launch("sum_squares_partial_kernel");
    metrics_final_kernel<<<1, kBlock>>>(d.reduction_blocks, d.partials, d.metrics);
    d.check_launch("metrics_final_kernel(power)");
    // One scalar home per power step, against two full vectors on the cold
    // path. The stopping rule needs it on the host.
    d.record(cudaMemcpy(d.host_metrics, d.metrics, 3 * sizeof(Real),
                        cudaMemcpyDeviceToHost),
             "download power norm");
    if (!d.sticky.ok()) return std::nullopt;  // fall back to the host path

    const Real next = std::sqrt(d.host_metrics[1]);
    if (next <= 0.0) return 0.0;
    divide_kernel<<<blocks_for(d.n), kBlock>>>(d.n, next, v);
    d.check_launch("divide_kernel");
    const Real candidate = std::sqrt(next);
    if (sigma > 0.0 && std::fabs(candidate - sigma) <= tolerance * sigma) {
      return candidate;
    }
    sigma = candidate;
  }
  return sigma;
}

// ---- Module 31: reflected Halpern ------------------------------------------
//
// The device form of HostIterationBackend's three Halpern operations, step for
// step and in the same order, so the two can be read side by side and a
// host-vs-device comparison from a shared state is a meaningful test.

void DevicePdlpBackend::begin_halpern() {
  Impl& d = *impl_;
  d.spmv(true, d.y, d.kty);
  d.spmv(false, d.x, d.kx_current);
  d.hot_products += 2;
  const auto copy = [&d](Real* to, const Real* from, std::size_t len, const char* what) {
    d.record(cudaMemcpyAsync(to, from, len * sizeof(Real), cudaMemcpyDeviceToDevice),
             what);
  };
  copy(d.x_anchor, d.x, d.n, "anchor x");
  copy(d.kty_anchor, d.kty, d.n, "anchor K'y");
  copy(d.y_anchor, d.y, d.m, "anchor y");
  copy(d.kx_anchor, d.kx_current, d.m, "anchor Kx");
  // Seed `T(z)` with the start, as on the host: the cold path reads the
  // solution from there, and before the first step there is nothing else.
  copy(d.x_trial, d.x, d.n, "seed T(z) x");
  copy(d.y_trial, d.y, d.m, "seed T(z) y");
  copy(d.kty_trial, d.kty, d.n, "seed K'T(z)");
  copy(d.kx_trial, d.kx_current, d.m, "seed KT(z)");
}

pdlp::TrialMetrics DevicePdlpBackend::halpern_step(Real eta, Real omega, Real gamma,
                                                   Real lambda) {
  Impl& d = *impl_;
  const Real tau = eta / omega;
  const Real sigma = eta * omega;

  // T(z): the same two kernels and one product as Algorithm 2's trial. `K'y`
  // is the cached image, NOT recomputed -- that is the whole saving.
  if (d.n > 0) {
    primal_trial_kernel<<<blocks_for(d.n), kBlock>>>(d.n, d.x, d.c, d.kty, d.lower,
                                                     d.upper, tau, d.x_trial,
                                                     d.extrapolated);
    d.check_launch("primal_trial_kernel");
  }
  d.spmv(false, d.extrapolated, d.kext);  // product 1
  if (d.m > 0) {
    dual_trial_kernel<<<blocks_for(d.m), kBlock>>>(d.m, d.num_equality, d.y, d.b, d.kext,
                                                   sigma, d.y_trial);
    d.check_launch("dual_trial_kernel");
  }

  // The fixed-point error pieces at `z`, BEFORE the blend moves it. The
  // existing reduction computes them unchanged: its `(y' - y) * (Kext - Kx)/2`
  // is the host's `(y - y') * (Kx - K x')` term for term, since
  // `Kx - Kx' = (Kx - Kext)/2`, and both flip together.
  metrics_partial_kernel<<<d.reduction_blocks, kBlock>>>(
      d.n, d.m, d.x, d.x_trial, d.y, d.y_trial, d.kext, d.kx_current, d.partials);
  d.check_launch("metrics_partial_kernel");
  metrics_final_kernel<<<1, kBlock>>>(d.reduction_blocks, d.partials, d.metrics);
  d.check_launch("metrics_final_kernel");

  if (d.m > 0) {
    trial_image_kernel<<<blocks_for(d.m), kBlock>>>(d.m, d.kext, d.kx_current,
                                                    d.kx_trial);
    d.check_launch("trial_image_kernel");
  }
  d.spmv(true, d.y_trial, d.kty_trial);  // product 2
  d.hot_products += 2;

  const Real reflected = lambda * (1.0 + gamma);
  const Real pull_back = lambda * gamma;
  const Real anchor = 1.0 - lambda;
  if (d.n > 0) {
    halpern_blend_kernel<<<blocks_for(d.n), kBlock>>>(
        d.n, reflected, pull_back, anchor, d.x_trial, d.x, d.x_anchor, d.kty_trial,
        d.kty, d.kty_anchor);
    d.check_launch("halpern_blend_kernel(x)");
  }
  if (d.m > 0) {
    halpern_blend_kernel<<<blocks_for(d.m), kBlock>>>(
        d.m, reflected, pull_back, anchor, d.y_trial, d.y, d.y_anchor, d.kx_trial,
        d.kx_current, d.kx_anchor);
    d.check_launch("halpern_blend_kernel(y)");
  }

  // The one synchronization per iteration, 24 bytes -- the same price
  // Algorithm 2's trial already paid, and for the same kind of reason: the
  // restart decision is made on the host from these three numbers. Issued
  // AFTER the blend so the host waits once, for everything.
  d.record(cudaMemcpy(d.host_metrics, d.metrics, 3 * sizeof(Real), cudaMemcpyDeviceToHost),
           "download metrics");
  pdlp::TrialMetrics metrics;
  metrics.interaction = d.host_metrics[0];
  metrics.dx_sq = d.host_metrics[1];
  metrics.dy_sq = d.host_metrics[2];
  return metrics;
}

pdlp::AnchorDistance DevicePdlpBackend::restart_at_pdhg_point() {
  Impl& d = *impl_;
  // The four restart quantities in one pass (restart_partial_kernel), with a
  // null state so it runs unconditionally. Read back: this is the per-step
  // API, whose caller makes the decision on the host.
  restart_partial_kernel<<<d.reduction_blocks, kBlock>>>(
      d.n, d.m, d.num_equality, nullptr, d.x_trial, d.x_anchor, d.y_trial, d.y_anchor,
      d.kx_trial, d.kty_trial, d.b, d.c, d.lower, d.upper, d.restart_partials);
  d.check_launch("restart_partial_kernel");
  restart_final_kernel<<<1, kBlock>>>(d.reduction_blocks, nullptr, d.restart_partials,
                                      d.restart_metrics);
  d.check_launch("restart_final_kernel");
  Real sums[4] = {0.0, 0.0, 0.0, 0.0};
  d.record(cudaMemcpy(sums, d.restart_metrics, 4 * sizeof(Real), cudaMemcpyDeviceToHost),
           "download restart sums");
  pdlp::AnchorDistance moved;
  moved.dx = std::sqrt(sums[0]);
  moved.dy = std::sqrt(sums[1]);
  moved.primal_residual = std::sqrt(sums[2]);
  moved.dual_residual = std::sqrt(sums[3]);

  // Algorithm 2 line 6: the new epoch starts at `T(z)`, and the anchor with
  // it. Device-to-device copies only; zero products.
  const auto copy = [&d](Real* to, const Real* from, std::size_t len, const char* what) {
    d.record(cudaMemcpyAsync(to, from, len * sizeof(Real), cudaMemcpyDeviceToDevice),
             what);
  };
  copy(d.x, d.x_trial, d.n, "restart x");
  copy(d.kty, d.kty_trial, d.n, "restart K'y");
  copy(d.x_anchor, d.x_trial, d.n, "restart anchor x");
  copy(d.kty_anchor, d.kty_trial, d.n, "restart anchor K'y");
  copy(d.y, d.y_trial, d.m, "restart y");
  copy(d.kx_current, d.kx_trial, d.m, "restart Kx");
  copy(d.y_anchor, d.y_trial, d.m, "restart anchor y");
  copy(d.kx_anchor, d.kx_trial, d.m, "restart anchor Kx");
  return moved;
}

// ---- the controller loop, resident ------------------------------------------

void DevicePdlpBackend::write_halpern_state(const pdlp::HalpernState& state) {
  Impl& d = *impl_;
  d.record(cudaMemcpy(d.control, &state, sizeof(state), cudaMemcpyHostToDevice),
           "upload halpern state");
}

pdlp::HalpernState DevicePdlpBackend::read_halpern_state() {
  Impl& d = *impl_;
  pdlp::HalpernState state;
  d.record(cudaMemcpy(&state, d.control, sizeof(state), cudaMemcpyDeviceToHost),
           "download halpern state");
  return state;
}

namespace {

/// ONE resident Halpern iteration, enqueued on `d.stream` -- the unit a CUDA
/// graph captures. Every launch here is a pure function of `params`, `track`
/// and buffer addresses; nothing iteration-specific is passed in (the counter
/// lives in `HalpernState::total`), which is what lets one capture replay.
///
/// In stream order:
///
///   snapshot (if tracking)        async device copies
///   T(z)                          primal kernel, SpMV K(2x'-x), dual kernel
///   r(z) pieces                   the existing two-stage reduction
///   K T(z)_x, K' T(z)_y           image kernel, SpMV K'
///   the blend                     two fused kernels, lambda from state
///   finish_difference (tracking)  two kernels
///   the decision                  controller_kernel, one thread
///   the restart, when pending     anchor distance, PID, move -- each kernel
///                                 reads the flag first and exits if clear
void enqueue_halpern_iteration(DevicePdlpBackend::Impl& d,
                               const pdlp::HalpernParams& params, bool track) {
  const cudaStream_t st = d.stream;
  const unsigned wide = blocks_for(std::max(d.n, d.m));
  if (track) {
    d.record(cudaMemcpyAsync(d.diff_x, d.x, d.n * sizeof(Real),
                             cudaMemcpyDeviceToDevice, st),
             "snapshot x");
    d.record(cudaMemcpyAsync(d.diff_y, d.y, d.m * sizeof(Real),
                             cudaMemcpyDeviceToDevice, st),
             "snapshot y");
  }
  if (d.n > 0) {
    resident_primal_kernel<<<blocks_for(d.n), kBlock, 0, st>>>(
        d.n, d.x, d.c, d.kty, d.lower, d.upper, params.eta, d.control, d.x_trial,
        d.extrapolated);
    d.check_launch("resident_primal_kernel");
  }
  d.spmv(false, d.extrapolated, d.kext);  // product 1, on d.stream via the handle
  if (d.m > 0) {
    resident_dual_kernel<<<blocks_for(d.m), kBlock, 0, st>>>(
        d.m, d.num_equality, d.y, d.b, d.kext, params.eta, d.control, d.y_trial);
    d.check_launch("resident_dual_kernel");
  }
  metrics_partial_kernel<<<d.reduction_blocks, kBlock, 0, st>>>(
      d.n, d.m, d.x, d.x_trial, d.y, d.y_trial, d.kext, d.kx_current, d.partials);
  d.check_launch("metrics_partial_kernel");
  metrics_final_kernel<<<1, kBlock, 0, st>>>(d.reduction_blocks, d.partials, d.metrics);
  d.check_launch("metrics_final_kernel");
  if (d.m > 0) {
    trial_image_kernel<<<blocks_for(d.m), kBlock, 0, st>>>(d.m, d.kext, d.kx_current,
                                                          d.kx_trial);
    d.check_launch("trial_image_kernel");
  }
  d.spmv(true, d.y_trial, d.kty_trial);  // product 2

  // `lambda` comes from the epoch counter BEFORE `controller_kernel`
  // increments it, as on the host.
  if (d.n > 0) {
    resident_blend_kernel<<<blocks_for(d.n), kBlock, 0, st>>>(
        d.n, params.gamma, d.control, d.x_trial, d.x, d.x_anchor, d.kty_trial, d.kty,
        d.kty_anchor);
    d.check_launch("resident_blend_kernel(x)");
  }
  if (d.m > 0) {
    resident_blend_kernel<<<blocks_for(d.m), kBlock, 0, st>>>(
        d.m, params.gamma, d.control, d.y_trial, d.y, d.y_anchor, d.kx_trial,
        d.kx_current, d.kx_anchor);
    d.check_launch("resident_blend_kernel(y)");
  }
  if (track) {
    if (d.n > 0) {
      finish_difference_kernel<<<blocks_for(d.n), kBlock, 0, st>>>(d.n, d.x, d.diff_x,
                                                                   d.sum_x);
      d.check_launch("finish_difference_kernel(x)");
    }
    if (d.m > 0) {
      finish_difference_kernel<<<blocks_for(d.m), kBlock, 0, st>>>(d.m, d.y, d.diff_y,
                                                                   d.sum_y);
      d.check_launch("finish_difference_kernel(y)");
    }
  }

  controller_kernel<<<1, 1, 0, st>>>(params, d.control, d.metrics);
  d.check_launch("controller_kernel");

  restart_partial_kernel<<<d.reduction_blocks, kBlock, 0, st>>>(
      d.n, d.m, d.num_equality, d.control, d.x_trial, d.x_anchor, d.y_trial, d.y_anchor,
      d.kx_trial, d.kty_trial, d.b, d.c, d.lower, d.upper, d.restart_partials);
  d.check_launch("restart_partial_kernel");
  restart_final_kernel<<<1, kBlock, 0, st>>>(d.reduction_blocks, d.control,
                                             d.restart_partials, d.restart_metrics);
  d.check_launch("restart_final_kernel");
  restart_controller_kernel<<<1, 1, 0, st>>>(params, d.control, d.restart_metrics);
  d.check_launch("restart_controller_kernel");
  restart_move_kernel<<<wide, kBlock, 0, st>>>(d.n, d.m, d.control, d.x_trial,
                                               d.kty_trial, d.y_trial, d.kx_trial, d.x,
                                               d.kty, d.x_anchor, d.kty_anchor, d.y,
                                               d.kx_current, d.y_anchor, d.kx_anchor);
  d.check_launch("restart_move_kernel");
}

bool same_params(const pdlp::HalpernParams& a, const pdlp::HalpernParams& b) {
  return a.eta == b.eta && a.gamma == b.gamma && a.sufficient == b.sufficient &&
         a.necessary == b.necessary && a.artificial == b.artificial && a.kp == b.kp &&
         a.ki == b.ki && a.kd == b.kd && a.integral_clamp == b.integral_clamp &&
         a.check_interval == b.check_interval &&
         a.restarts_enabled == b.restarts_enabled && a.weight_update == b.weight_update;
}

/// Captures `iterations` resident iterations into an executable graph.
/// Returns nullptr and fills `why` on any failure, having ended the capture
/// so the stream is usable again. The sticky status is restored: a failed
/// capture is a reason to launch directly, not a solver error.
cudaGraphExec_t capture(DevicePdlpBackend::Impl& d, const pdlp::HalpernParams& params,
                        bool track, std::size_t iterations, std::string& why) {
  const Status before = d.sticky;
  if (cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal) !=
      cudaSuccess) {
    why = "cudaStreamBeginCapture failed";
    (void)cudaGetLastError();
    return nullptr;
  }
  for (std::size_t i = 0; i < iterations; ++i) enqueue_halpern_iteration(d, params, track);
  cudaGraph_t graph = nullptr;
  const cudaError_t ended = cudaStreamEndCapture(d.stream, &graph);
  // Anything recorded during the capture -- a cuSPARSE call that refuses to
  // be captured, say -- belongs to the capture, not to the solve.
  const bool refused = !d.sticky.ok() && before.ok();
  const std::string refusal = refused ? d.sticky.error().format() : std::string();
  d.sticky = before;
  if (ended != cudaSuccess || graph == nullptr || refused) {
    why = "stream capture failed: " +
          (ended != cudaSuccess ? std::string(cudaGetErrorString(ended)) : refusal);
    if (graph) cudaGraphDestroy(graph);
    (void)cudaGetLastError();
    return nullptr;
  }
  cudaGraphExec_t exec = nullptr;
  const cudaError_t inst = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  if (inst != cudaSuccess) {
    why = std::string("cudaGraphInstantiate failed: ") + cudaGetErrorString(inst);
    (void)cudaGetLastError();
    return nullptr;
  }
  return exec;
}

}  // namespace

/// HostIterationBackend::run_halpern on the device, with no host wait.
///
/// WHY GRAPHS. Removing the per-iteration sync measured as a wash on this
/// machine, and the reason was visible in the numbers: an iteration is ~13
/// launches, and at WSL2's ~20 us per launch that is ~0.26 ms of pure launch
/// overhead per iteration, which no amount of asynchrony hides. A captured
/// graph issues the whole sequence as ONE launch; capturing a full check
/// interval makes that one launch per 40 iterations.
///
/// Correctness does not depend on the graph: `use_graphs = false`, a failed
/// capture, and an odd-length chunk all fall back to launching the same
/// `enqueue_halpern_iteration` directly, and the tests compare the two.
void DevicePdlpBackend::run_halpern(std::size_t count, std::uint64_t first_iteration,
                                    const pdlp::HalpernParams& params,
                                    bool track_differences) {
  Impl& d = *impl_;
  if (count == 0) return;
  d.hot_products += 2 * count;

  // Seed the device counter for this chunk. On the resident path it is
  // already `first_iteration` from the previous chunk; setting it anyway
  // makes each chunk self-contained.
  set_total_kernel<<<1, 1, 0, d.stream>>>(d.control, first_iteration);
  d.check_launch("set_total_kernel");

  if (!d.use_graphs || d.graphs_unavailable) {
    for (std::size_t it = 0; it < count; ++it) {
      enqueue_halpern_iteration(d, params, track_differences);
      ++d.graph_launches;  // iterations issued, for comparability
    }
    return;
  }

  const Impl::GraphKey key{params, track_differences, d.x, d.y, d.x_trial, d.y_trial};
  const bool stale = !d.graph_key_set || !same_params(key.params, d.graph_key.params) ||
                     key.track != d.graph_key.track || key.x != d.graph_key.x ||
                     key.y != d.graph_key.y || key.x_trial != d.graph_key.x_trial ||
                     key.y_trial != d.graph_key.y_trial;
  if (stale) {
    d.drop_graphs();
    d.graph_key = key;
    d.graph_key_set = true;
  }

  // The common case: a whole check interval, one launch. The first full chunk
  // of a run is the one that captures it.
  if (d.graph_chunk == nullptr && count > 1 &&
      count == static_cast<std::size_t>(params.check_interval)) {
    d.graph_chunk = capture(d, params, track_differences, count, d.graph_note);
    if (d.graph_chunk == nullptr) {
      d.graphs_unavailable = true;
    } else {
      d.graph_chunk_length = count;
      ++d.graph_captures;
    }
  }
  if (d.graph_chunk != nullptr && count == d.graph_chunk_length) {
    d.record(cudaGraphLaunch(d.graph_chunk, d.stream), "cudaGraphLaunch(chunk)");
    ++d.graph_launches;
    return;
  }

  // Any other length: one captured iteration, replayed.
  if (d.graph_single == nullptr && !d.graphs_unavailable) {
    d.graph_single = capture(d, params, track_differences, 1, d.graph_note);
    if (d.graph_single == nullptr) {
      d.graphs_unavailable = true;
    } else {
      ++d.graph_captures;
    }
  }
  for (std::size_t it = 0; it < count; ++it) {
    if (d.graph_single != nullptr) {
      d.record(cudaGraphLaunch(d.graph_single, d.stream), "cudaGraphLaunch(single)");
    } else {
      enqueue_halpern_iteration(d, params, track_differences);
    }
    ++d.graph_launches;
  }
}

bool DevicePdlpBackend::evaluate_resident(core::HostSpan<const Real> row_scale,
                                          core::HostSpan<const Real> col_scale,
                                          ResidentSums& out) {
  Impl& d = *impl_;
  if (!d.sticky.ok()) return false;

  const auto ensure = [&](Real** p, std::size_t count) {
    if (*p == nullptr) d.record(cudaMalloc(reinterpret_cast<void**>(p), count * sizeof(Real)),
                                "cudaMalloc(evaluate_resident)");
  };
  ensure(&d.eval_partials, kEvalSums * static_cast<std::size_t>(d.reduction_blocks));
  ensure(&d.eval_out, kEvalSums);
  if (d.eval_host == nullptr &&
      cudaMallocHost(reinterpret_cast<void**>(&d.eval_host), kEvalSums * sizeof(Real)) !=
          cudaSuccess) {
    return false;
  }
  // `1/scale`, uploaded the first time and again only if the caller's factors
  // change (they are fixed for a run).
  const auto upload_inverse = [&](core::HostSpan<const Real> scale, std::size_t len, Real** dev,
                                  const Real** src) {
    if (scale.empty()) return;
    if (*dev != nullptr && *src == scale.data()) return;
    ensure(dev, len);
    std::vector<Real> inv(len);
    for (std::size_t k = 0; k < len; ++k) inv[k] = 1.0 / scale[k];
    d.record(cudaMemcpy(*dev, inv.data(), len * sizeof(Real), cudaMemcpyHostToDevice),
             "cudaMemcpy(inverse scale)");
    *src = scale.data();
  };
  upload_inverse(row_scale, d.m, &d.inv_row, &d.inv_row_src);
  upload_inverse(col_scale, d.n, &d.inv_col, &d.inv_col_src);
  if (!d.sticky.ok()) return false;

  // On the resident stream, after the chunk just enqueued.
  eval_partial_kernel<<<d.reduction_blocks, kBlock, 0, d.stream>>>(
      d.n, d.m, d.num_equality, d.x_trial, d.y_trial, d.kx_trial, d.kty_trial, d.b, d.c,
      d.lower, d.upper, row_scale.empty() ? nullptr : d.inv_row,
      col_scale.empty() ? nullptr : d.inv_col, d.eval_partials);
  d.check_launch("eval_partial_kernel");
  eval_final_kernel<<<1, kBlock, 0, d.stream>>>(d.reduction_blocks, d.eval_partials, d.eval_out);
  d.check_launch("eval_final_kernel");
  d.record(cudaMemcpyAsync(d.eval_host, d.eval_out, kEvalSums * sizeof(Real),
                           cudaMemcpyDeviceToHost, d.stream),
           "cudaMemcpyAsync(eval sums)");
  d.record(cudaStreamSynchronize(d.stream), "cudaStreamSynchronize(eval)");
  if (!d.sticky.ok()) return false;

  out.primal_objective = d.eval_host[0];
  out.b_dot_y = d.eval_host[1];
  out.bound_term = d.eval_host[2];
  out.primal_sq = d.eval_host[3];
  out.dual_sq = d.eval_host[4];
  return true;
}

void DevicePdlpBackend::set_use_graphs(bool enabled) { impl_->use_graphs = enabled; }
bool DevicePdlpBackend::graphs_active() const {
  return impl_->use_graphs && !impl_->graphs_unavailable && impl_->graph_captures > 0;
}
const std::string& DevicePdlpBackend::graph_note() const { return impl_->graph_note; }
std::size_t DevicePdlpBackend::graph_launches() const { return impl_->graph_launches; }

core::Status DevicePdlpBackend::status() const { return impl_->sticky; }
std::size_t DevicePdlpBackend::own_products() const { return impl_->hot_products; }
pdlp::MatVec& DevicePdlpBackend::cold_matvec() { return *cold_; }
std::size_t DevicePdlpBackend::device_bytes() const { return impl_->bytes; }

}  // namespace sovsolve::solver::gpu
