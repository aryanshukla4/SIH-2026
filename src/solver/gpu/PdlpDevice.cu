#include "sovsolve/solver/gpu/PdlpDevice.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <string>
#include <utility>

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
                    static_cast<void*>(cold_in_n), static_cast<void*>(cold_out_m),
                    static_cast<void*>(cold_in_m), static_cast<void*>(cold_out_n),
                    buffer_a, buffer_at, static_cast<void*>(partials),
                    static_cast<void*>(metrics)}) {
      cudaFree(p);
    }
    if (host_metrics) cudaFreeHost(host_metrics);
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
                   &d->sum_x, &d->avg_x, &d->cold_in_n, &d->cold_out_n}) {
    if (st.ok()) st = alloc(p, n, "n-vector");
  }
  for (Real** p : {&d->y, &d->y_trial, &d->kx_current, &d->kext, &d->diff_y,
                   &d->sum_y, &d->avg_y, &d->cold_in_m, &d->cold_out_m}) {
    if (st.ok()) st = alloc(p, m, "m-vector");
  }
  // clang-format on
  if (!st.ok()) return st.error();

  d->reduction_blocks =
      std::max(1u, std::min(kMaxReductionBlocks, blocks_for(std::max(n, m))));
  st = alloc(&d->partials, 3 * static_cast<std::size_t>(d->reduction_blocks), "partials");
  if (st.ok()) st = alloc(&d->metrics, 3, "metrics");
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
  st = sparse(cusparseCreate(&d->handle), "cusparseCreate");
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

core::Status DevicePdlpBackend::status() const { return impl_->sticky; }
std::size_t DevicePdlpBackend::own_products() const { return impl_->hot_products; }
pdlp::MatVec& DevicePdlpBackend::cold_matvec() { return *cold_; }
std::size_t DevicePdlpBackend::device_bytes() const { return impl_->bytes; }

}  // namespace sovsolve::solver::gpu
