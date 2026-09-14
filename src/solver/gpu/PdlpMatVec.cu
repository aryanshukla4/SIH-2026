#include "sovsolve/solver/gpu/PdlpMatVec.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <chrono>
#include <string>
#include <vector>

namespace sovsolve::solver::gpu {

namespace {

using core::Status;

Status cuda_check(cudaError_t st, const char* what) {
  if (st != cudaSuccess) {
    return core::make_error(core::ErrorCode::NumericalError,
                            std::string(what) + " failed: " + cudaGetErrorString(st));
  }
  return Status::Ok();
}

Status cusparse_check(cusparseStatus_t st, const char* what) {
  if (st != CUSPARSE_STATUS_SUCCESS) {
    return core::make_error(core::ErrorCode::NumericalError,
                            std::string(what) + " failed (cusparse status " +
                                std::to_string(static_cast<int>(st)) + ")");
  }
  return Status::Ok();
}

double now_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace

/// Everything device-side. The matrix halves are uploaded once in `create` and
/// never touched again; only the vectors move per call.
struct CusparseMatVec::Impl {
  std::size_t m = 0;
  std::size_t n = 0;

  cusparseHandle_t handle = nullptr;

  // A as CSR (m x n) and A' as CSR (n x m), the latter being the canonicalizer's
  // CSC read directly -- no transpose is ever computed. A transposed SpMV on a
  // CSR is markedly slower than a second CSR, and we already hold both
  // orientations on the host, so this trades VRAM we have for time we do not.
  core::Index* a_offsets = nullptr;
  core::Index* a_indices = nullptr;
  Real* a_values = nullptr;
  core::Index* at_offsets = nullptr;
  core::Index* at_indices = nullptr;
  Real* at_values = nullptr;

  cusparseSpMatDescr_t mat_a = nullptr;
  cusparseSpMatDescr_t mat_at = nullptr;

  Real* d_in_n = nullptr;   ///< x, for `K x`
  Real* d_out_m = nullptr;  ///< `K x`
  Real* d_in_m = nullptr;   ///< y, for `K' y`
  Real* d_out_n = nullptr;  ///< `K' y`

  cusparseDnVecDescr_t vec_in_n = nullptr;
  cusparseDnVecDescr_t vec_out_m = nullptr;
  cusparseDnVecDescr_t vec_in_m = nullptr;
  cusparseDnVecDescr_t vec_out_n = nullptr;

  void* buffer_a = nullptr;
  void* buffer_at = nullptr;
  std::size_t buffer_a_bytes = 0;
  std::size_t buffer_at_bytes = 0;

  std::size_t matrix_bytes = 0;
  /// OFF by default, and that is a correctness point rather than a
  /// preference. Splitting kernel from transfer needs a
  /// cudaDeviceSynchronize between the phases, and PDLP applies `K` tens
  /// of thousands of times -- at 64k products that is ~128k extra syncs,
  /// which is a large fraction of what the run then appears to cost. So
  /// the breakdown and the wall time CANNOT be measured in the same run.
  /// Default off gives an honest wall time; turn it on to attribute that
  /// time, and read the totals as inflated when you do.
  bool detailed_timing = false;
  double kernel_seconds = 0.0;
  double transfer_seconds = 0.0;

  ~Impl() {
    if (vec_in_n) cusparseDestroyDnVec(vec_in_n);
    if (vec_out_m) cusparseDestroyDnVec(vec_out_m);
    if (vec_in_m) cusparseDestroyDnVec(vec_in_m);
    if (vec_out_n) cusparseDestroyDnVec(vec_out_n);
    if (mat_a) cusparseDestroySpMat(mat_a);
    if (mat_at) cusparseDestroySpMat(mat_at);
    if (handle) cusparseDestroy(handle);
    cudaFree(a_offsets);
    cudaFree(a_indices);
    cudaFree(a_values);
    cudaFree(at_offsets);
    cudaFree(at_indices);
    cudaFree(at_values);
    cudaFree(d_in_n);
    cudaFree(d_out_m);
    cudaFree(d_in_m);
    cudaFree(d_out_n);
    cudaFree(buffer_a);
    cudaFree(buffer_at);
  }
};

CusparseMatVec::CusparseMatVec(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CusparseMatVec::~CusparseMatVec() = default;

std::size_t CusparseMatVec::num_rows() const { return impl_->m; }
std::size_t CusparseMatVec::num_cols() const { return impl_->n; }
std::size_t CusparseMatVec::device_bytes() const {
  return impl_->matrix_bytes + impl_->buffer_a_bytes + impl_->buffer_at_bytes +
         (2 * impl_->n + 2 * impl_->m) * sizeof(Real);
}
double CusparseMatVec::kernel_seconds() const { return impl_->kernel_seconds; }
double CusparseMatVec::transfer_seconds() const { return impl_->transfer_seconds; }

core::Expected<std::unique_ptr<CusparseMatVec>> CusparseMatVec::create(
    const model::CanonicalProblem& problem, bool detailed_timing) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    return core::make_error(core::ErrorCode::NotImplemented,
                            "CusparseMatVec::create: no CUDA device is available");
  }

  auto impl = std::make_unique<Impl>();
  impl->detailed_timing = detailed_timing;
  impl->m = problem.num_rows();
  impl->n = problem.num_cols();

  const auto& csr = problem.A.csr;
  const auto& csc = problem.A.csc;
  const std::size_t nnz = csr.values().size();
  if (nnz != csc.values().size()) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "CusparseMatVec::create: the CSR and CSC orientations "
                            "disagree on the nonzero count");
  }

  // Upload, once. `cudaMemcpy` here and nowhere else for the matrix -- per
  // iteration only vectors move, which is the point of the whole arrangement.
  auto upload = [&](void** dst, const void* src, std::size_t bytes,
                    const char* what) -> Status {
    Status st = cuda_check(cudaMalloc(dst, bytes), what);
    if (!st.ok()) return st;
    impl->matrix_bytes += bytes;
    return cuda_check(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice), what);
  };

  const std::size_t idx_bytes = sizeof(core::Index);
  Status st = upload(reinterpret_cast<void**>(&impl->a_offsets), csr.offsets().data(),
                     (impl->m + 1) * idx_bytes, "cudaMalloc/Memcpy(A offsets)");
  if (!st.ok()) return st.error();
  st = upload(reinterpret_cast<void**>(&impl->a_indices), csr.indices().data(),
              nnz * idx_bytes, "cudaMalloc/Memcpy(A indices)");
  if (!st.ok()) return st.error();
  st = upload(reinterpret_cast<void**>(&impl->a_values), csr.values().data(),
              nnz * sizeof(Real), "cudaMalloc/Memcpy(A values)");
  if (!st.ok()) return st.error();
  st = upload(reinterpret_cast<void**>(&impl->at_offsets), csc.offsets().data(),
              (impl->n + 1) * idx_bytes, "cudaMalloc/Memcpy(A' offsets)");
  if (!st.ok()) return st.error();
  st = upload(reinterpret_cast<void**>(&impl->at_indices), csc.indices().data(),
              nnz * idx_bytes, "cudaMalloc/Memcpy(A' indices)");
  if (!st.ok()) return st.error();
  st = upload(reinterpret_cast<void**>(&impl->at_values), csc.values().data(),
              nnz * sizeof(Real), "cudaMalloc/Memcpy(A' values)");
  if (!st.ok()) return st.error();

  for (auto* p : {&impl->d_in_n, &impl->d_out_n}) {
    st = cuda_check(cudaMalloc(reinterpret_cast<void**>(p), impl->n * sizeof(Real)),
                    "cudaMalloc(vector n)");
    if (!st.ok()) return st.error();
  }
  for (auto* p : {&impl->d_in_m, &impl->d_out_m}) {
    st = cuda_check(cudaMalloc(reinterpret_cast<void**>(p), impl->m * sizeof(Real)),
                    "cudaMalloc(vector m)");
    if (!st.ok()) return st.error();
  }

  st = cusparse_check(cusparseCreate(&impl->handle), "cusparseCreate");
  if (!st.ok()) return st.error();

  const auto mi = static_cast<std::int64_t>(impl->m);
  const auto ni = static_cast<std::int64_t>(impl->n);
  const auto nnzi = static_cast<std::int64_t>(nnz);

  st = cusparse_check(
      cusparseCreateCsr(&impl->mat_a, mi, ni, nnzi, impl->a_offsets, impl->a_indices,
                        impl->a_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
      "cusparseCreateCsr(A)");
  if (!st.ok()) return st.error();
  st = cusparse_check(
      cusparseCreateCsr(&impl->mat_at, ni, mi, nnzi, impl->at_offsets, impl->at_indices,
                        impl->at_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
      "cusparseCreateCsr(A')");
  if (!st.ok()) return st.error();

  st = cusparse_check(cusparseCreateDnVec(&impl->vec_in_n, ni, impl->d_in_n, CUDA_R_64F),
                      "cusparseCreateDnVec(in n)");
  if (!st.ok()) return st.error();
  st = cusparse_check(cusparseCreateDnVec(&impl->vec_out_m, mi, impl->d_out_m, CUDA_R_64F),
                      "cusparseCreateDnVec(out m)");
  if (!st.ok()) return st.error();
  st = cusparse_check(cusparseCreateDnVec(&impl->vec_in_m, mi, impl->d_in_m, CUDA_R_64F),
                      "cusparseCreateDnVec(in m)");
  if (!st.ok()) return st.error();
  st = cusparse_check(cusparseCreateDnVec(&impl->vec_out_n, ni, impl->d_out_n, CUDA_R_64F),
                      "cusparseCreateDnVec(out n)");
  if (!st.ok()) return st.error();

  // Size both workspaces once. cusparseSpMV can allocate internally per call
  // otherwise, which on a loop this tight shows up as pure overhead.
  const Real one = 1.0;
  const Real zero = 0.0;
  st = cusparse_check(
      cusparseSpMV_bufferSize(impl->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                              impl->mat_a, impl->vec_in_n, &zero, impl->vec_out_m,
                              CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                              &impl->buffer_a_bytes),
      "cusparseSpMV_bufferSize(A)");
  if (!st.ok()) return st.error();
  if (impl->buffer_a_bytes > 0) {
    st = cuda_check(cudaMalloc(&impl->buffer_a, impl->buffer_a_bytes),
                    "cudaMalloc(SpMV buffer A)");
    if (!st.ok()) return st.error();
  }
  st = cusparse_check(
      cusparseSpMV_bufferSize(impl->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                              impl->mat_at, impl->vec_in_m, &zero, impl->vec_out_n,
                              CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                              &impl->buffer_at_bytes),
      "cusparseSpMV_bufferSize(A')");
  if (!st.ok()) return st.error();
  if (impl->buffer_at_bytes > 0) {
    st = cuda_check(cudaMalloc(&impl->buffer_at, impl->buffer_at_bytes),
                    "cudaMalloc(SpMV buffer A')");
    if (!st.ok()) return st.error();
  }

  return std::unique_ptr<CusparseMatVec>(new CusparseMatVec(std::move(impl)));
}

void CusparseMatVec::multiply(core::HostSpan<const Real> x, core::HostSpan<Real> out) {
  Impl& d = *impl_;
  const Real one = 1.0;
  const Real zero = 0.0;

  // Timed in three parts, each with its own sync, because the split is the
  // measurement this stage exists to produce -- see the header.
  const bool split = d.detailed_timing;
  const double t0 = split ? now_seconds() : 0.0;
  cudaMemcpy(d.d_in_n, x.data(), d.n * sizeof(Real), cudaMemcpyHostToDevice);
  double t1 = 0.0;
  if (split) {
    cudaDeviceSynchronize();
    t1 = now_seconds();
  }

  cusparseSpMV(d.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d.mat_a, d.vec_in_n,
               &zero, d.vec_out_m, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d.buffer_a);
  double t2 = 0.0;
  if (split) {
    cudaDeviceSynchronize();
    t2 = now_seconds();
  }

  // A blocking D2H copy is itself a synchronization point, so the result is
  // complete here whether or not the phases above were timed.
  cudaMemcpy(out.data(), d.d_out_m, d.m * sizeof(Real), cudaMemcpyDeviceToHost);
  if (split) {
    const double t3 = now_seconds();
    d.transfer_seconds += (t1 - t0) + (t3 - t2);
    d.kernel_seconds += t2 - t1;
  }
  ++products_;
}

void CusparseMatVec::multiply_transpose(core::HostSpan<const Real> y,
                                        core::HostSpan<Real> out) {
  Impl& d = *impl_;
  const Real one = 1.0;
  const Real zero = 0.0;

  const bool split = d.detailed_timing;
  const double t0 = split ? now_seconds() : 0.0;
  cudaMemcpy(d.d_in_m, y.data(), d.m * sizeof(Real), cudaMemcpyHostToDevice);
  double t1 = 0.0;
  if (split) {
    cudaDeviceSynchronize();
    t1 = now_seconds();
  }

  // `mat_at` is n x m, so this is an ordinary NON_TRANSPOSE product against it.
  cusparseSpMV(d.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, d.mat_at, d.vec_in_m,
               &zero, d.vec_out_n, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d.buffer_at);
  double t2 = 0.0;
  if (split) {
    cudaDeviceSynchronize();
    t2 = now_seconds();
  }

  cudaMemcpy(out.data(), d.d_out_n, d.n * sizeof(Real), cudaMemcpyDeviceToHost);
  if (split) {
    const double t3 = now_seconds();
    d.transfer_seconds += (t1 - t0) + (t3 - t2);
    d.kernel_seconds += t2 - t1;
  }
  ++products_;
}

}  // namespace sovsolve::solver::gpu
