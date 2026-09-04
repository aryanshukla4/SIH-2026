#include "sovsolve/solver/gpu/LinearSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "sovsolve/core/DenseMatrix.hpp"

namespace sovsolve::solver::gpu {

namespace {

Status cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return core::make_error(core::ErrorCode::NumericalError,
                            std::string(what) + ": " + cudaGetErrorString(err));
  }
  return Status::Ok();
}

Status cusolver_check(cusolverStatus_t st, const char* what) {
  if (st != CUSOLVER_STATUS_SUCCESS) {
    return core::make_error(core::ErrorCode::NumericalError,
                            std::string(what) + " failed (cusolver status " +
                                std::to_string(static_cast<int>(st)) + ")");
  }
  return Status::Ok();
}

/// Owns every device allocation this solve makes, freed on any exit path
/// (including an error return) without a hand-written cleanup call at each
/// `if (!status.ok())` site.
struct DeviceBuffers {
  double* a = nullptr;
  double* b = nullptr;
  int* ipiv = nullptr;
  int* info = nullptr;
  double* work = nullptr;
  cusolverDnHandle_t handle = nullptr;

  ~DeviceBuffers() {
    if (a) cudaFree(a);
    if (b) cudaFree(b);
    if (ipiv) cudaFree(ipiv);
    if (info) cudaFree(info);
    if (work) cudaFree(work);
    if (handle) cusolverDnDestroy(handle);
  }
};

}  // namespace

Expected<LinearSolveResult> solve(const KktSystem& system,
                                   const SymbolicFactorization* /*symbolic*/,
                                   int /*max_refinement_steps*/) {
  const std::size_t dim = system.matrix.rows();
  if (system.matrix.cols() != dim || system.rhs.size() != dim) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "solve: KktSystem matrix/rhs dimensions are inconsistent");
  }
  if (dim == 0) return LinearSolveResult{};

  // Sparse -> dense. See the header comment: a deliberate stopgap, O(dim^2)
  // memory and O(dim^3) time, not the intended sparse path.
  core::DenseMatrix<Real> dense(dim, dim);
  for (std::size_t j = 0; j < dim; ++j) {
    for (std::size_t i = 0; i < dim; ++i) dense(i, j) = 0.0;
  }
  const auto& csr = system.matrix.csr;
  for (std::size_t i = 0; i < dim; ++i) {
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      dense(i, static_cast<std::size_t>(csr.indices()[k])) = csr.values()[k];
    }
  }

  const int n = static_cast<int>(dim);
  const int lda = static_cast<int>(dense.ld());
  const std::size_t a_elems = static_cast<std::size_t>(lda) * dim;

  DeviceBuffers dev;
  Status st = cusolver_check(cusolverDnCreate(&dev.handle), "cusolverDnCreate");
  if (!st.ok()) return st.error();

  st = cuda_check(cudaMalloc(&dev.a, a_elems * sizeof(double)), "cudaMalloc(A)");
  if (!st.ok()) return st.error();
  st = cuda_check(cudaMalloc(&dev.b, dim * sizeof(double)), "cudaMalloc(b)");
  if (!st.ok()) return st.error();
  st = cuda_check(cudaMalloc(&dev.ipiv, dim * sizeof(int)), "cudaMalloc(ipiv)");
  if (!st.ok()) return st.error();
  st = cuda_check(cudaMalloc(&dev.info, sizeof(int)), "cudaMalloc(info)");
  if (!st.ok()) return st.error();

  st = cuda_check(
      cudaMemcpy(dev.a, dense.data(), a_elems * sizeof(double), cudaMemcpyHostToDevice),
      "cudaMemcpy(A, host->device)");
  if (!st.ok()) return st.error();
  st = cuda_check(cudaMemcpy(dev.b, system.rhs.data(), dim * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(b, host->device)");
  if (!st.ok()) return st.error();

  int lwork = 0;
  st = cusolver_check(cusolverDnDgetrf_bufferSize(dev.handle, n, n, dev.a, lda, &lwork),
                      "cusolverDnDgetrf_bufferSize");
  if (!st.ok()) return st.error();
  st = cuda_check(cudaMalloc(&dev.work, static_cast<std::size_t>(lwork) * sizeof(double)),
                  "cudaMalloc(work)");
  if (!st.ok()) return st.error();

  st = cusolver_check(
      cusolverDnDgetrf(dev.handle, n, n, dev.a, lda, dev.work, dev.ipiv, dev.info),
      "cusolverDnDgetrf");
  if (!st.ok()) return st.error();

  int info_host = 0;
  st = cuda_check(cudaMemcpy(&info_host, dev.info, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(info, device->host)");
  if (!st.ok()) return st.error();
  if (info_host != 0) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "cusolverDnDgetrf: matrix is exactly singular at pivot " +
                                std::to_string(info_host) +
                                " (delta_p/delta_d regularization was not enough)");
  }

  // Pivot growth ratio: max|U_ii| / min|U_ii|, read off the factor now sitting
  // in dev.a (U occupies the diagonal and above -- L's diagonal is implicit
  // 1s, never stored). A strided 2D copy pulls just the `dim` diagonal
  // entries, not the whole factored matrix. This is the caller's only signal
  // that the solve was numerically meaningless despite info==0 -- see the
  // comment on Options::IpmOptions::max_pivot_ratio.
  Real pivot_ratio = 1.0;
  {
    core::RealVector diag(dim);
    st = cuda_check(cudaMemcpy2D(diag.data(), sizeof(double), dev.a,
                                 (static_cast<std::size_t>(lda) + 1) * sizeof(double),
                                 sizeof(double), dim, cudaMemcpyDeviceToHost),
                    "cudaMemcpy2D(diag, device->host)");
    if (!st.ok()) return st.error();
    Real max_abs = 0.0;
    Real min_abs = -1.0;
    for (std::size_t j = 0; j < dim; ++j) {
      const Real mag = std::fabs(diag[j]);
      max_abs = std::max(max_abs, mag);
      if (min_abs < 0.0 || mag < min_abs) min_abs = mag;
    }
    pivot_ratio = min_abs > 0.0 ? max_abs / min_abs : std::numeric_limits<Real>::infinity();
  }

  st = cusolver_check(
      cusolverDnDgetrs(dev.handle, CUBLAS_OP_N, n, 1, dev.a, lda, dev.ipiv, dev.b, n, dev.info),
      "cusolverDnDgetrs");
  if (!st.ok()) return st.error();

  st = cuda_check(cudaMemcpy(&info_host, dev.info, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(info, device->host)");
  if (!st.ok()) return st.error();
  if (info_host != 0) {
    return core::make_error(
        core::ErrorCode::NumericalError,
        "cusolverDnDgetrs: invalid argument " + std::to_string(info_host));
  }

  LinearSolveResult result;
  result.solution = core::RealVector(dim);
  st = cuda_check(
      cudaMemcpy(result.solution.data(), dev.b, dim * sizeof(double), cudaMemcpyDeviceToHost),
      "cudaMemcpy(solution, device->host)");
  if (!st.ok()) return st.error();

  result.refinement_passes = 0;  // deferred -- see the header comment
  result.pivot_ratio = pivot_ratio;
  return std::move(result);
}

}  // namespace sovsolve::solver::gpu
