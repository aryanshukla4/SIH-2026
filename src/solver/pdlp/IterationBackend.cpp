#include "sovsolve/solver/pdlp/IterationBackend.hpp"

#include <algorithm>
#include <cmath>

namespace sovsolve::solver::pdlp {

namespace {

core::HostSpan<const Real> in(const core::RealVector& v) { return {v.data(), v.size()}; }
core::HostSpan<Real> out(core::RealVector& v) { return {v.data(), v.size()}; }

void zeroed(core::RealVector& v, std::size_t n) {
  v.resize(n);
  v.assign(0.0);
}

void copy_into(const core::RealVector& from, core::HostSpan<Real> to) {
  for (std::size_t i = 0; i < from.size(); ++i) to[i] = from[i];
}

}  // namespace

HostIterationBackend::HostIterationBackend(const model::CanonicalProblem& problem,
                                           MatVec& matvec)
    : problem_(problem),
      matvec_(matvec),
      n_(problem.num_cols()),
      m_(problem.num_rows()) {
  for (auto* v : {&x_, &x_trial_, &extrapolated_, &kt_y_, &x_prev_, &diff_x_, &sum_x_,
                  &avg_x_, &kt_y_trial_, &x_anchor_, &kt_y_anchor_}) {
    zeroed(*v, n_);
  }
  for (auto* v : {&y_, &y_trial_, &k_x_current_, &k_extrapolated_, &k_x_, &diff_y_,
                  &sum_y_, &avg_y_, &k_x_trial_, &y_anchor_, &k_x_anchor_}) {
    zeroed(*v, m_);
  }
}

void HostIterationBackend::set_iterate(core::HostSpan<const Real> x,
                                       core::HostSpan<const Real> y) {
  for (std::size_t j = 0; j < n_; ++j) x_[j] = x[j];
  for (std::size_t i = 0; i < m_; ++i) y_[i] = y[i];
}

void HostIterationBackend::download(BackendVector which, core::HostSpan<Real> dst) {
  switch (which) {
    case BackendVector::X: copy_into(x_, dst); break;
    case BackendVector::Y: copy_into(y_, dst); break;
    case BackendVector::AverageX: copy_into(avg_x_, dst); break;
    case BackendVector::AverageY: copy_into(avg_y_, dst); break;
    case BackendVector::IterateSumX: copy_into(sum_x_, dst); break;
    case BackendVector::IterateSumY: copy_into(sum_y_, dst); break;
    case BackendVector::DifferenceX: copy_into(diff_x_, dst); break;
    case BackendVector::DifferenceY: copy_into(diff_y_, dst); break;
    case BackendVector::PdhgX: copy_into(x_trial_, dst); break;
    case BackendVector::PdhgY: copy_into(y_trial_, dst); break;
  }
}

void HostIterationBackend::begin_step() {
  matvec_.multiply_transpose(in(y_), out(kt_y_));
  matvec_.multiply(in(x_), out(k_x_current_));
}

// The loop bodies below are the pre-refactor Pdlp.cpp code, including the ORDER
// of every accumulation. That is not fussiness: floating-point addition is not
// associative, so reordering a sum changes the last bits, the adaptive step
// then accepts a borderline trial differently, and the run diverges -- which is
// exactly the drift measured between the CPU and cuSPARSE paths in section 24E.
// Keeping the order is what lets "the refactor changed nothing" be checked as
// bit-identical output rather than argued.
TrialMetrics HostIterationBackend::trial(Real tau, Real sigma) {
  // line 4: x' = proj_X(x - tau (c - K'y))
  for (std::size_t j = 0; j < n_; ++j) {
    const Real step = x_[j] - tau * (problem_.c[j] - kt_y_[j]);
    x_trial_[j] = std::clamp(step, problem_.col_lower[j], problem_.col_upper[j]);
    extrapolated_[j] = 2.0 * x_trial_[j] - x_[j];
  }

  // line 5: y' = proj_Y(y + sigma (q - K(2x' - x)))
  matvec_.multiply(in(extrapolated_), out(k_extrapolated_));
  for (std::size_t i = 0; i < m_; ++i) {
    const Real step = y_[i] + sigma * (problem_.b[i] - k_extrapolated_[i]);
    y_trial_[i] = i < problem_.num_equality ? step : std::min(step, 0.0);
  }

  // line 6. `K dx = (K(2x'-x) - Kx) / 2`, so it costs no product of its own.
  TrialMetrics metrics;
  for (std::size_t i = 0; i < m_; ++i) {
    const Real k_dx = 0.5 * (k_extrapolated_[i] - k_x_current_[i]);
    metrics.interaction += (y_trial_[i] - y_[i]) * k_dx;
  }
  for (std::size_t j = 0; j < n_; ++j) {
    const Real d = x_trial_[j] - x_[j];
    metrics.dx_sq += d * d;
  }
  for (std::size_t i = 0; i < m_; ++i) {
    const Real d = y_trial_[i] - y_[i];
    metrics.dy_sq += d * d;
  }
  return metrics;
}

void HostIterationBackend::accept_trial() {
  for (std::size_t j = 0; j < n_; ++j) x_[j] = x_trial_[j];
  for (std::size_t i = 0; i < m_; ++i) y_[i] = y_trial_[i];
}

void HostIterationBackend::fixed_step(Real tau, Real sigma) {
  matvec_.multiply_transpose(in(y_), out(kt_y_));
  for (std::size_t j = 0; j < n_; ++j) {
    x_prev_[j] = x_[j];
    const Real step = x_[j] - tau * (problem_.c[j] - kt_y_[j]);
    x_[j] = std::clamp(step, problem_.col_lower[j], problem_.col_upper[j]);
  }
  // `2x^{k+1} - x^k` is what makes this PDHG rather than Arrow-Hurwicz.
  for (std::size_t j = 0; j < n_; ++j) extrapolated_[j] = 2.0 * x_[j] - x_prev_[j];
  matvec_.multiply(in(extrapolated_), out(k_x_));
  for (std::size_t i = 0; i < m_; ++i) {
    const Real step = y_[i] + sigma * (problem_.b[i] - k_x_[i]);
    y_[i] = i < problem_.num_equality ? step : std::min(step, 0.0);
  }
}

// --------------------------------------------------------------------------
// Module 31: reflected Halpern (cuPDLPx, arXiv 2507.14051)
// --------------------------------------------------------------------------

void HostIterationBackend::begin_halpern() {
  matvec_.multiply_transpose(in(y_), out(kt_y_));
  matvec_.multiply(in(x_), out(k_x_current_));
  for (std::size_t j = 0; j < n_; ++j) {
    x_anchor_[j] = x_[j];
    kt_y_anchor_[j] = kt_y_[j];
  }
  for (std::size_t i = 0; i < m_; ++i) {
    y_anchor_[i] = y_[i];
    k_x_anchor_[i] = k_x_current_[i];
  }
  // `T(z)` is not defined yet, and the cold path reads the solution from
  // there. Seed it with the anchor so a run that terminates before its first
  // step reports the starting point rather than a zero vector.
  for (std::size_t j = 0; j < n_; ++j) x_trial_[j] = x_[j];
  for (std::size_t i = 0; i < m_; ++i) y_trial_[i] = y_[i];
  for (std::size_t j = 0; j < n_; ++j) kt_y_trial_[j] = kt_y_[j];
  for (std::size_t i = 0; i < m_; ++i) k_x_trial_[i] = k_x_current_[i];
}

TrialMetrics HostIterationBackend::halpern_step(Real eta, Real omega, Real gamma,
                                                Real lambda) {
  const Real tau = eta / omega;
  const Real sigma = eta * omega;

  // --- T(z): one PDHG step, equation (3). `K' y` is already cached. -------
  for (std::size_t j = 0; j < n_; ++j) {
    const Real step = x_[j] - tau * (problem_.c[j] - kt_y_[j]);
    x_trial_[j] = std::clamp(step, problem_.col_lower[j], problem_.col_upper[j]);
    extrapolated_[j] = 2.0 * x_trial_[j] - x_[j];
  }
  matvec_.multiply(in(extrapolated_), out(k_extrapolated_));  // product 1
  for (std::size_t i = 0; i < m_; ++i) {
    const Real step = y_[i] + sigma * (problem_.b[i] - k_extrapolated_[i]);
    y_trial_[i] = i < problem_.num_equality ? step : std::min(step, 0.0);
  }

  // --- the fixed-point error at `z`, in the pieces the caller combines ----
  //
  // `dx = x - T(z)_x`, so `K dx = K x - K T(z)_x`, and since
  // `K(2x' - x) = 2 K x' - K x` the image of the trial point comes out of the
  // product already taken: `K x' = (K(2x' - x) + K x) / 2`. Hence
  // `K dx = (K x - K(2x' - x)) / 2`, no second product.
  //
  // Note the SIGN convention: this is `x - T(z)`, the opposite of `trial()`'s
  // `x' - x`. `||.||` and the product of two differences are both invariant
  // under flipping both, so `TrialMetrics` means the same thing either way --
  // but only because BOTH are flipped together.
  TrialMetrics metrics;
  for (std::size_t i = 0; i < m_; ++i) {
    k_x_trial_[i] = 0.5 * (k_extrapolated_[i] + k_x_current_[i]);
  }
  for (std::size_t i = 0; i < m_; ++i) {
    metrics.interaction += (y_[i] - y_trial_[i]) * (k_x_current_[i] - k_x_trial_[i]);
  }
  for (std::size_t j = 0; j < n_; ++j) {
    const Real d = x_[j] - x_trial_[j];
    metrics.dx_sq += d * d;
  }
  for (std::size_t i = 0; i < m_; ++i) {
    const Real d = y_[i] - y_trial_[i];
    metrics.dy_sq += d * d;
  }

  // `K' T(z)_y`, needed for the next iteration's cached `K' y`.
  matvec_.multiply_transpose(in(y_trial_), out(kt_y_trial_));  // product 2

  // --- the blend, applied identically to the iterate and to its images ----
  //
  //     z^{k+1} = lambda [ (1+gamma) T(z) - gamma z ] + (1-lambda) z^{n,0}
  //
  // Linear in `z`, so `K` and `K'` commute with it exactly. `blend` below is
  // that one line; running it on `K x` and `K' y` too is what keeps the
  // iteration at two products.
  const Real reflected = lambda * (1.0 + gamma);
  const Real pull_back = lambda * gamma;
  const Real anchor = 1.0 - lambda;
  const auto blend = [&](Real trial, Real current, Real base) {
    return reflected * trial - pull_back * current + anchor * base;
  };
  for (std::size_t j = 0; j < n_; ++j) {
    const Real next = blend(x_trial_[j], x_[j], x_anchor_[j]);
    kt_y_[j] = blend(kt_y_trial_[j], kt_y_[j], kt_y_anchor_[j]);
    x_[j] = next;
  }
  for (std::size_t i = 0; i < m_; ++i) {
    const Real next = blend(y_trial_[i], y_[i], y_anchor_[i]);
    k_x_current_[i] = blend(k_x_trial_[i], k_x_current_[i], k_x_anchor_[i]);
    y_[i] = next;
  }
  return metrics;
}

AnchorDistance HostIterationBackend::restart_at_pdhg_point() {
  // arXiv 2407.16144 Algorithm 2 line 6: the next epoch starts at `T(z^{n,k})`,
  // NOT at `z^{n,k}`. Both the point and its two matrix images are already
  // sitting in the trial slots from the last `halpern_step`, so this costs no
  // product -- and it is also the only point in the epoch guaranteed to lie
  // inside the box, since reflection extrapolates out of it.
  AnchorDistance moved;
  for (std::size_t j = 0; j < n_; ++j) {
    const Real d = x_trial_[j] - x_anchor_[j];
    moved.dx += d * d;
  }
  for (std::size_t i = 0; i < m_; ++i) {
    const Real d = y_trial_[i] - y_anchor_[i];
    moved.dy += d * d;
  }
  moved.dx = std::sqrt(moved.dx);
  moved.dy = std::sqrt(moved.dy);

  // HPR-LP's safeguard (18) needs both infeasibilities at the restart point
  // `T(z)`. Its images `K T(z)_x` and `K' T(z)_y` are in the trial slots, so
  // this is two elementwise sweeps and no product.
  Real primal_sq = 0.0;
  for (std::size_t i = 0; i < m_; ++i) {
    const Real v = halpern_primal_violation(k_x_trial_[i], problem_.b[i],
                                            i < problem_.num_equality);
    primal_sq += v * v;
  }
  Real dual_sq = 0.0;
  for (std::size_t j = 0; j < n_; ++j) {
    const Real v = halpern_dual_leftover(problem_.c[j] - kt_y_trial_[j],
                                         problem_.col_lower[j], problem_.col_upper[j]);
    dual_sq += v * v;
  }
  moved.primal_residual = std::sqrt(primal_sq);
  moved.dual_residual = std::sqrt(dual_sq);

  for (std::size_t j = 0; j < n_; ++j) {
    x_[j] = x_trial_[j];
    kt_y_[j] = kt_y_trial_[j];
    x_anchor_[j] = x_trial_[j];
    kt_y_anchor_[j] = kt_y_trial_[j];
  }
  for (std::size_t i = 0; i < m_; ++i) {
    y_[i] = y_trial_[i];
    k_x_current_[i] = k_x_trial_[i];
    y_anchor_[i] = y_trial_[i];
    k_x_anchor_[i] = k_x_trial_[i];
  }
  return moved;
}

void HostIterationBackend::run_halpern(std::size_t count, std::uint64_t first_iteration,
                                       const HalpernParams& params,
                                       bool track_differences) {
  // The reference implementation of the controller loop. A device backend
  // fuses all of this into kernels, but the decisions it makes are these ones,
  // from the same header.
  for (std::size_t i = 0; i < count; ++i) {
    if (track_differences) snapshot_iterate();
    const TrialMetrics fp = halpern_step(params.eta, control_.omega, params.gamma,
                                         halpern_lambda(control_));
    if (track_differences) finish_difference();
    const Real r = halpern_fixed_point_error(params.eta, control_.omega, fp.interaction,
                                             fp.dx_sq, fp.dy_sq);
    control_.total = first_iteration + i + 1;
    if (halpern_observe(params, control_, r, control_.total)) {
      const AnchorDistance moved = restart_at_pdhg_point();
      halpern_on_restart(params, control_, moved.dx, moved.dy,
                         moved.primal_residual / (1.0 + params.b_norm),
                         moved.dual_residual / (1.0 + params.c_norm));
    }
  }
}

void HostIterationBackend::snapshot_iterate() {
  for (std::size_t j = 0; j < n_; ++j) diff_x_[j] = x_[j];
  for (std::size_t i = 0; i < m_; ++i) diff_y_[i] = y_[i];
}

void HostIterationBackend::finish_difference() {
  for (std::size_t j = 0; j < n_; ++j) {
    diff_x_[j] = x_[j] - diff_x_[j];
    sum_x_[j] += x_[j];
  }
  for (std::size_t i = 0; i < m_; ++i) {
    diff_y_[i] = y_[i] - diff_y_[i];
    sum_y_[i] += y_[i];
  }
}

void HostIterationBackend::accumulate_average(Real weight) {
  for (std::size_t j = 0; j < n_; ++j) avg_x_[j] += weight * x_[j];
  for (std::size_t i = 0; i < m_; ++i) avg_y_[i] += weight * y_[i];
}

void HostIterationBackend::reset_average() {
  avg_x_.assign(0.0);
  avg_y_.assign(0.0);
}

}  // namespace sovsolve::solver::pdlp
