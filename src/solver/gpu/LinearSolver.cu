#include "sovsolve/solver/gpu/LinearSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include "sovsolve/analysis/MatrixAnalysis.hpp"
#include "sovsolve/core/DenseMatrix.hpp"
#include "sovsolve/solver/gpu/Preconditioner.hpp"
#include "sovsolve/solver/gpu/VectorOps.hpp"

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

Status cublas_check(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    return core::make_error(core::ErrorCode::NumericalError,
                            std::string(what) + " failed (cublas status " +
                                std::to_string(static_cast<int>(st)) + ")");
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

/// Every device allocation this solve needs, reused across calls instead of
/// torn down and rebuilt each time. Measured cause (not guessed): on
/// afiro.mps (dim=60), cusolverDnCreate + 5x cudaMalloc/cudaFree per call
/// accounted for 96% of total solve() wall time across the 35 Newton solves
/// one IPM run makes -- the actual Dgetrf/Dgetrs arithmetic on a matrix this
/// small is a rounding error by comparison. The KKT dimension (n+m) is fixed
/// for the entire lifetime of one solve_problem() call (every IPM iteration
/// re-solves the same size system), so the handle and buffers are recreated
/// only when `dim` actually changes -- once per problem, not once per
/// iteration. Kept as function-local state (not a caller-owned resource
/// threaded through PredictorCorrector/Solve) to avoid changing this
/// function's signature or its call sites' shape for a fix that the
/// eventual sparse/cuDSS path replaces outright (cuDSS's own
/// CUDSS_PHASE_REFACTORIZATION is the same reuse idea, formalized). Not
/// thread-safe -- fine today since the IPM loop is single-threaded
/// (Solve.cu); would need per-thread state if that ever changes.
struct PersistentSolverContext {
  cusolverDnHandle_t handle = nullptr;
  double* a = nullptr;
  double* b = nullptr;
  int* ipiv = nullptr;
  int* info = nullptr;
  double* work = nullptr;
  std::size_t dim = 0;  // 0 means "buffers not (yet validly) sized"

  void free_buffers() {
    if (a) cudaFree(a);
    if (b) cudaFree(b);
    if (ipiv) cudaFree(ipiv);
    if (info) cudaFree(info);
    if (work) cudaFree(work);
    a = b = work = nullptr;
    ipiv = nullptr;
    info = nullptr;
    dim = 0;
  }

  ~PersistentSolverContext() {
    free_buffers();
    if (handle) cusolverDnDestroy(handle);
  }
};

PersistentSolverContext& solver_context() {
  static PersistentSolverContext ctx;
  return ctx;
}

/// Mirrors `PersistentSolverContext` above for the SPD normal-equations path
/// (`solve_spd`) -- same measured motivation (handle creation and device
/// allocation dominate a single small solve's cost), kept as a SEPARATE
/// context because this path's buffers are shaped by (m, n) rather than one
/// `dim`, and it needs a cuBLAS handle in addition to cusolverDn's.
struct PersistentSpdContext {
  cusolverDnHandle_t solver_handle = nullptr;
  cublasHandle_t blas_handle = nullptr;
  double* a = nullptr;           // m x n, unscaled dense A
  double* b = nullptr;           // m x n, A scaled by sqrt(theta) per column
  double* mat = nullptr;         // m x m: A*T*A^T, then the Cholesky factor in place
  double* rhs_dy = nullptr;      // length m: rhs in, dy out
  double* sqrt_theta = nullptr;  // length n
  double* diag_add = nullptr;    // length m
  int* info = nullptr;
  double* work = nullptr;
  int lwork = 0;
  std::size_t m = 0;
  std::size_t n = 0;  // 0 (along with m) means "buffers not (yet validly) sized"

  void free_buffers() {
    if (a) cudaFree(a);
    if (b) cudaFree(b);
    if (mat) cudaFree(mat);
    if (rhs_dy) cudaFree(rhs_dy);
    if (sqrt_theta) cudaFree(sqrt_theta);
    if (diag_add) cudaFree(diag_add);
    if (info) cudaFree(info);
    if (work) cudaFree(work);
    a = b = mat = rhs_dy = sqrt_theta = diag_add = work = nullptr;
    info = nullptr;
    lwork = 0;
    m = n = 0;
  }

  ~PersistentSpdContext() {
    free_buffers();
    if (solver_handle) cusolverDnDestroy(solver_handle);
    if (blas_handle) cublasDestroy(blas_handle);
  }
};

PersistentSpdContext& spd_context() {
  static PersistentSpdContext ctx;
  return ctx;
}

/// Device state for the matrix-free CG solve (`solve_spd_cg`). `A`'s
/// structure AND values are uploaded once per (m, n, nnz) -- unlike `theta`/
/// `diag_add`/`rhs`, which change every Newton solve, `A` itself is fixed for
/// the entire `solve_problem()` call, so this context caches it across every
/// CG solve in the whole IPM run, not just across CG's own iterations.
struct PersistentCgContext {
  cusparseHandle_t sparse_handle = nullptr;
  cublasHandle_t blas_handle = nullptr;

  // Which columns MatrixAnalysis flags as dense -- cached alongside
  // everything else keyed by (m, n, nnz), since it depends only on A's fixed
  // structure. Fed to ic0_build (Preconditioner.hpp) every solve; see that
  // file's header comment for why dense columns are excluded from the
  // preconditioner's fill pattern.
  std::vector<bool> is_dense_column;

  // Device copy of A's structure+values -- CSR for `A*t`, CSC (used directly
  // as CSR-of-A^T) for `A^T*p`. See LinearSolver.hpp's solve_spd_cg doc
  // comment for why both are needed.
  int* csr_offsets = nullptr;    // m+1
  int* csr_indices = nullptr;    // nnz
  double* csr_values = nullptr;  // nnz
  int* csc_offsets = nullptr;    // n+1
  int* csc_indices = nullptr;    // nnz
  double* csc_values = nullptr;  // nnz

  cusparseSpMatDescr_t mat_a = nullptr;         // A, m x n
  cusparseSpMatDescr_t mat_a_transpose = nullptr;  // A^T, n x m (CSC-of-A read as CSR)

  // CG working vectors, all device-resident. Length m unless noted.
  double* x = nullptr;
  double* r = nullptr;
  double* z = nullptr;
  double* p = nullptr;
  double* ap = nullptr;
  double* t = nullptr;  // length n: A^T*p, then theta .* that
  double* theta = nullptr;             // length n
  double* diag_add = nullptr;          // length m
  double* precond_diag_inv = nullptr;  // length m

  cusparseDnVecDescr_t vec_p = nullptr;   // wraps `p` (length m)
  cusparseDnVecDescr_t vec_t = nullptr;   // wraps `t` (length n)
  cusparseDnVecDescr_t vec_ap = nullptr;  // wraps `ap` (length m)

  void* spmv_buffer_at = nullptr;
  void* spmv_buffer_a = nullptr;

  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t nnz = 0;

  void free_buffers() {
    if (mat_a) cusparseDestroySpMat(mat_a);
    if (mat_a_transpose) cusparseDestroySpMat(mat_a_transpose);
    if (vec_p) cusparseDestroyDnVec(vec_p);
    if (vec_t) cusparseDestroyDnVec(vec_t);
    if (vec_ap) cusparseDestroyDnVec(vec_ap);
    mat_a = mat_a_transpose = nullptr;
    vec_p = vec_t = vec_ap = nullptr;

    if (csr_offsets) cudaFree(csr_offsets);
    if (csr_indices) cudaFree(csr_indices);
    if (csr_values) cudaFree(csr_values);
    if (csc_offsets) cudaFree(csc_offsets);
    if (csc_indices) cudaFree(csc_indices);
    if (csc_values) cudaFree(csc_values);
    if (x) cudaFree(x);
    if (r) cudaFree(r);
    if (z) cudaFree(z);
    if (p) cudaFree(p);
    if (ap) cudaFree(ap);
    if (t) cudaFree(t);
    if (theta) cudaFree(theta);
    if (diag_add) cudaFree(diag_add);
    if (precond_diag_inv) cudaFree(precond_diag_inv);
    if (spmv_buffer_at) cudaFree(spmv_buffer_at);
    if (spmv_buffer_a) cudaFree(spmv_buffer_a);
    csr_offsets = csc_offsets = nullptr;
    csr_indices = csc_indices = nullptr;
    csr_values = csc_values = nullptr;
    x = r = z = p = ap = t = theta = diag_add = precond_diag_inv = nullptr;
    spmv_buffer_at = spmv_buffer_a = nullptr;
    m = n = nnz = 0;
  }

  ~PersistentCgContext() {
    free_buffers();
    if (sparse_handle) cusparseDestroy(sparse_handle);
    if (blas_handle) cublasDestroy(blas_handle);
  }
};

PersistentCgContext& cg_context() {
  static PersistentCgContext ctx;
  return ctx;
}

/// Device state for the matrix-free MINRES solve (`solve_minres`). Simpler
/// than the CG context: the augmented matrix is symmetric with BOTH
/// triangles stored explicitly (`build_kkt` inserts `(2,1)=A` and
/// `(1,2)=A^T` as separate entries), so a single CSR descriptor and a single
/// SpMV per iteration compute the full product -- no second (transposed)
/// view needed the way CG's rectangular `A` needs one.
struct PersistentMinresContext {
  cusparseHandle_t sparse_handle = nullptr;
  cublasHandle_t blas_handle = nullptr;

  int* csr_offsets = nullptr;
  int* csr_indices = nullptr;
  double* csr_values = nullptr;
  cusparseSpMatDescr_t mat = nullptr;

  // MINRES working vectors (Paige & Saunders' formulation), all length dim.
  double* x = nullptr;
  double* r1 = nullptr;
  double* r2 = nullptr;
  double* v = nullptr;
  double* w = nullptr;
  double* w1 = nullptr;
  double* w2 = nullptr;
  double* y = nullptr;  // precond(r2), i.e. r2 .* precond_diag_inv
  double* precond_diag_inv = nullptr;

  // vec_v/vec_y wrap `v`/`y` respectively -- the operator apply's input and
  // output. Never bound to `r1`: `r1` holds the previous iteration's Lanczos
  // vector, which the very next line after the operator apply still needs to
  // read, so it must not be reused as the apply's scratch output.
  cusparseDnVecDescr_t vec_v = nullptr;
  cusparseDnVecDescr_t vec_y = nullptr;

  void* spmv_buffer = nullptr;

  std::size_t dim = 0;
  std::size_t nnz = 0;

  void free_buffers() {
    if (mat) cusparseDestroySpMat(mat);
    if (vec_v) cusparseDestroyDnVec(vec_v);
    if (vec_y) cusparseDestroyDnVec(vec_y);
    mat = nullptr;
    vec_v = vec_y = nullptr;

    if (csr_offsets) cudaFree(csr_offsets);
    if (csr_indices) cudaFree(csr_indices);
    if (csr_values) cudaFree(csr_values);
    if (x) cudaFree(x);
    if (r1) cudaFree(r1);
    if (r2) cudaFree(r2);
    if (v) cudaFree(v);
    if (w) cudaFree(w);
    if (w1) cudaFree(w1);
    if (w2) cudaFree(w2);
    if (y) cudaFree(y);
    if (precond_diag_inv) cudaFree(precond_diag_inv);
    if (spmv_buffer) cudaFree(spmv_buffer);
    csr_offsets = nullptr;
    csr_indices = nullptr;
    csr_values = nullptr;
    x = r1 = r2 = v = w = w1 = w2 = y = precond_diag_inv = nullptr;
    spmv_buffer = nullptr;
    dim = nnz = 0;
  }

  ~PersistentMinresContext() {
    free_buffers();
    if (sparse_handle) cusparseDestroy(sparse_handle);
    if (blas_handle) cublasDestroy(blas_handle);
  }
};

PersistentMinresContext& minres_context() {
  static PersistentMinresContext ctx;
  return ctx;
}

}  // namespace

Expected<LinearSolveResult> solve_dense(const KktSystem& system,
                                        const SymbolicFactorization* /*symbolic*/,
                                        int /*max_refinement_steps*/) {
  const std::size_t dim = system.matrix.rows();
  if (system.matrix.cols() != dim || system.rhs.size() != dim) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "solve_dense: KktSystem matrix/rhs dimensions are inconsistent");
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

  PersistentSolverContext& dev = solver_context();

  Status st = Status::Ok();
  if (!dev.handle) {
    st = cusolver_check(cusolverDnCreate(&dev.handle), "cusolverDnCreate");
    if (!st.ok()) return st.error();
  }

  if (dev.dim != dim) {
    dev.free_buffers();  // no-op the first time; drops stale differently-sized buffers otherwise

    st = cuda_check(cudaMalloc(&dev.a, a_elems * sizeof(double)), "cudaMalloc(A)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.b, dim * sizeof(double)), "cudaMalloc(b)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.ipiv, dim * sizeof(int)), "cudaMalloc(ipiv)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.info, sizeof(int)), "cudaMalloc(info)");
    if (!st.ok()) return st.error();

    int lwork = 0;
    st = cusolver_check(cusolverDnDgetrf_bufferSize(dev.handle, n, n, dev.a, lda, &lwork),
                        "cusolverDnDgetrf_bufferSize");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.work, static_cast<std::size_t>(lwork) * sizeof(double)),
                    "cudaMalloc(work)");
    if (!st.ok()) return st.error();

    // Only now, with every allocation for this size confirmed to have
    // succeeded, is it safe to mark the context as sized for `dim` -- a
    // failure above must leave dev.dim at 0 (set by free_buffers) so the
    // next call retries the full allocation instead of trusting partially
    // set-up buffers.
    dev.dim = dim;
  }

  st = cuda_check(
      cudaMemcpy(dev.a, dense.data(), a_elems * sizeof(double), cudaMemcpyHostToDevice),
      "cudaMemcpy(A, host->device)");
  if (!st.ok()) return st.error();
  st = cuda_check(cudaMemcpy(dev.b, system.rhs.data(), dim * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(b, host->device)");
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

Expected<LinearSolveResult> solve_spd_dense(const NormalEquationsSystem& system) {
  if (system.a == nullptr) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "solve_spd_dense: NormalEquationsSystem::a is null");
  }
  const std::size_t m = system.rhs.size();
  const std::size_t n = system.theta.size();
  if (system.a->rows() != m || system.a->cols() != n || system.diag_add.size() != m) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "solve_spd_dense: NormalEquationsSystem dimensions are inconsistent");
  }
  if (m == 0) return LinearSolveResult{};

  // Test-oracle only (see the header comment): densify `system.a` locally.
  // Production code (`solve_spd_cg`) never does this -- it applies `A` via
  // cuSPARSE SpMV instead.
  core::DenseMatrix<Real> dense_a(m, n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < m; ++i) dense_a(i, j) = 0.0;
  }
  {
    const auto& csr = system.a->csr;
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
        dense_a(i, static_cast<std::size_t>(csr.indices()[k])) = csr.values()[k];
      }
    }
  }

  PersistentSpdContext& dev = spd_context();

  Status st = Status::Ok();
  if (!dev.solver_handle) {
    st = cusolver_check(cusolverDnCreate(&dev.solver_handle), "cusolverDnCreate");
    if (!st.ok()) return st.error();
  }
  if (!dev.blas_handle) {
    st = cublas_check(cublasCreate(&dev.blas_handle), "cublasCreate");
    if (!st.ok()) return st.error();
  }

  const int m_i = static_cast<int>(m);
  const int n_i = static_cast<int>(n);
  const int ld = m_i;  // every device buffer below is packed tightly, stride m

  if (dev.m != m || dev.n != n) {
    dev.free_buffers();  // no-op the first time; drops stale differently-sized buffers otherwise

    st = cuda_check(cudaMalloc(&dev.a, m * n * sizeof(double)), "cudaMalloc(spd A)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.b, m * n * sizeof(double)), "cudaMalloc(spd B)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.mat, m * m * sizeof(double)), "cudaMalloc(spd M)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.rhs_dy, m * sizeof(double)), "cudaMalloc(spd rhs)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.sqrt_theta, n * sizeof(double)), "cudaMalloc(spd sqrt_theta)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.diag_add, m * sizeof(double)), "cudaMalloc(spd diag_add)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.info, sizeof(int)), "cudaMalloc(spd info)");
    if (!st.ok()) return st.error();

    int lwork = 0;
    st = cusolver_check(
        cusolverDnDpotrf_bufferSize(dev.solver_handle, CUBLAS_FILL_MODE_LOWER, m_i, dev.mat, ld,
                                    &lwork),
        "cusolverDnDpotrf_bufferSize");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.work, static_cast<std::size_t>(lwork) * sizeof(double)),
                    "cudaMalloc(spd work)");
    if (!st.ok()) return st.error();
    dev.lwork = lwork;

    // Only now, with every allocation for this (m, n) confirmed to have
    // succeeded, is it safe to mark the context as sized -- same
    // partial-failure policy as PersistentSolverContext above.
    dev.m = m;
    dev.n = n;
  }

  // Host -> device: dense A, in ONE cudaMemcpy2D call rather than one memcpy
  // per column (`dense_a` is padded to `dense_a.ld()`, `dev.a`'s stride is
  // exactly `m` -- a per-column loop was measured, in the production path
  // this test oracle mirrors, to cost far more than the transfer itself
  // under WSL2's per-call CUDA overhead; see git history for the measurement
  // that motivated switching to cudaMemcpy2D).
  st = cuda_check(cudaMemcpy2D(dev.a, m * sizeof(double), dense_a.data(),
                               dense_a.ld() * sizeof(double), m * sizeof(double), n,
                               cudaMemcpyHostToDevice),
                  "cudaMemcpy2D(spd A, host->device)");
  if (!st.ok()) return st.error();

  core::RealVector sqrt_theta_host(n);
  for (std::size_t j = 0; j < n; ++j) sqrt_theta_host[j] = std::sqrt(system.theta[j]);
  st = cuda_check(cudaMemcpy(dev.sqrt_theta, sqrt_theta_host.data(), n * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(spd sqrt_theta, host->device)");
  if (!st.ok()) return st.error();

  st = cuda_check(cudaMemcpy(dev.diag_add, system.diag_add.data(), m * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(spd diag_add, host->device)");
  if (!st.ok()) return st.error();

  st = cuda_check(cudaMemcpy(dev.rhs_dy, system.rhs.data(), m * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(spd rhs, host->device)");
  if (!st.ok()) return st.error();

  const double one = 1.0;
  const double zero = 0.0;

  // B = A * diag(sqrt_theta) -- column j of A scaled by sqrt(theta_j), so
  // that B*B^T = A*T*A^T below without ever forming T explicitly.
  st = cublas_check(cublasDdgmm(dev.blas_handle, CUBLAS_SIDE_RIGHT, m_i, n_i, dev.a, ld,
                                dev.sqrt_theta, 1, dev.b, ld),
                    "cublasDdgmm");
  if (!st.ok()) return st.error();

  // M = B*B^T = A*T*A^T.
  st = cublas_check(cublasDsyrk(dev.blas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, m_i, n_i,
                                &one, dev.b, ld, &zero, dev.mat, ld),
                    "cublasDsyrk");
  if (!st.ok()) return st.error();

  // M's diagonal += D_s + delta_d*I. A strided AXPY: the diagonal of an m x m
  // column-major matrix with leading dimension `ld` sits at stride `ld + 1`.
  st = cublas_check(cublasDaxpy(dev.blas_handle, m_i, &one, dev.diag_add, 1, dev.mat, ld + 1),
                    "cublasDaxpy(spd diagonal)");
  if (!st.ok()) return st.error();

  st = cusolver_check(
      cusolverDnDpotrf(dev.solver_handle, CUBLAS_FILL_MODE_LOWER, m_i, dev.mat, ld, dev.work,
                        dev.lwork, dev.info),
      "cusolverDnDpotrf");
  if (!st.ok()) return st.error();

  int info_host = 0;
  st = cuda_check(cudaMemcpy(&info_host, dev.info, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(spd info, device->host)");
  if (!st.ok()) return st.error();
  if (info_host != 0) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "cusolverDnDpotrf: reduced normal-equations matrix is not "
                            "positive definite at leading minor " +
                                std::to_string(info_host) +
                                " (delta_p/delta_d regularization was not enough)");
  }

  // Cholesky's analogue of the LU path's pivot_ratio: max|L_ii| / min|L_ii|
  // off the factor's diagonal, same cheap no-extra-solve breakdown proxy
  // (LinearSolveResult::pivot_ratio's doc comment, LinearSolver.hpp).
  Real pivot_ratio = 1.0;
  {
    core::RealVector diag(m);
    st = cuda_check(cudaMemcpy2D(diag.data(), sizeof(double), dev.mat,
                                 (static_cast<std::size_t>(ld) + 1) * sizeof(double),
                                 sizeof(double), m, cudaMemcpyDeviceToHost),
                    "cudaMemcpy2D(spd diag, device->host)");
    if (!st.ok()) return st.error();
    Real max_abs = 0.0;
    Real min_abs = -1.0;
    for (std::size_t i = 0; i < m; ++i) {
      const Real mag = std::fabs(diag[i]);
      max_abs = std::max(max_abs, mag);
      if (min_abs < 0.0 || mag < min_abs) min_abs = mag;
    }
    pivot_ratio = min_abs > 0.0 ? max_abs / min_abs : std::numeric_limits<Real>::infinity();
  }

  st = cusolver_check(
      cusolverDnDpotrs(dev.solver_handle, CUBLAS_FILL_MODE_LOWER, m_i, 1, dev.mat, ld, dev.rhs_dy,
                        m_i, dev.info),
      "cusolverDnDpotrs");
  if (!st.ok()) return st.error();

  st = cuda_check(cudaMemcpy(&info_host, dev.info, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(spd info, device->host)");
  if (!st.ok()) return st.error();
  if (info_host != 0) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "cusolverDnDpotrs: invalid argument " + std::to_string(info_host));
  }

  LinearSolveResult result;
  result.solution = core::RealVector(m);
  st = cuda_check(
      cudaMemcpy(result.solution.data(), dev.rhs_dy, m * sizeof(double), cudaMemcpyDeviceToHost),
      "cudaMemcpy(spd dy, device->host)");
  if (!st.ok()) return st.error();

  result.refinement_passes = 0;
  result.pivot_ratio = pivot_ratio;
  return std::move(result);
}

Expected<LinearSolveResult> solve_spd_cg(const NormalEquationsSystem& system,
                                          Real cg_tolerance, int cg_max_iterations,
                                          bool try_direct) {
  if (system.a == nullptr) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "solve_spd_cg: NormalEquationsSystem::a is null");
  }
  const std::size_t m = system.rhs.size();
  const std::size_t n = system.theta.size();
  const std::size_t nnz = system.a->nnz();
  if (system.a->rows() != m || system.a->cols() != n || system.diag_add.size() != m) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "solve_spd_cg: NormalEquationsSystem dimensions are inconsistent");
  }
  if (m == 0) return LinearSolveResult{};

  PersistentCgContext& dev = cg_context();

  Status st = Status::Ok();
  if (!dev.sparse_handle) {
    st = cusparse_check(cusparseCreate(&dev.sparse_handle), "cusparseCreate");
    if (!st.ok()) return st.error();
  }
  if (!dev.blas_handle) {
    st = cublas_check(cublasCreate(&dev.blas_handle), "cublasCreate");
    if (!st.ok()) return st.error();
  }

  const auto m_i = static_cast<int64_t>(m);
  const auto n_i = static_cast<int64_t>(n);
  const auto nnz_i = static_cast<int64_t>(nnz);
  const int m32 = static_cast<int>(m);

  if (dev.m != m || dev.n != n || dev.nnz != nnz) {
    dev.free_buffers();  // no-op the first time; drops stale differently-sized state otherwise

    // Depends only on A's fixed structure -- computed once here, reused by
    // ic0_build (Preconditioner.hpp) every solve. detect_duplicates is off:
    // v1's preconditioner doesn't use duplicate-row/column information.
    {
      analysis::AnalysisOptions opts;
      opts.detect_duplicates = false;
      const analysis::MatrixAnalysis mat_analysis = analysis::analyze(*system.a, opts);
      dev.is_dense_column.assign(n, false);
      for (const core::Index j : mat_analysis.dense_columns) {
        dev.is_dense_column[static_cast<std::size_t>(j)] = true;
      }
    }

    st = cuda_check(cudaMalloc(&dev.csr_offsets, (m + 1) * sizeof(int)), "cudaMalloc(cg csr_off)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.csr_indices, nnz * sizeof(int)), "cudaMalloc(cg csr_idx)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.csr_values, nnz * sizeof(double)), "cudaMalloc(cg csr_val)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.csc_offsets, (n + 1) * sizeof(int)), "cudaMalloc(cg csc_off)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.csc_indices, nnz * sizeof(int)), "cudaMalloc(cg csc_idx)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.csc_values, nnz * sizeof(double)), "cudaMalloc(cg csc_val)");
    if (!st.ok()) return st.error();

    st = cuda_check(cudaMalloc(&dev.x, m * sizeof(double)), "cudaMalloc(cg x)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.r, m * sizeof(double)), "cudaMalloc(cg r)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.z, m * sizeof(double)), "cudaMalloc(cg z)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.p, m * sizeof(double)), "cudaMalloc(cg p)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.ap, m * sizeof(double)), "cudaMalloc(cg ap)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.t, n * sizeof(double)), "cudaMalloc(cg t)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.theta, n * sizeof(double)), "cudaMalloc(cg theta)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.diag_add, m * sizeof(double)), "cudaMalloc(cg diag_add)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.precond_diag_inv, m * sizeof(double)),
                    "cudaMalloc(cg precond_diag_inv)");
    if (!st.ok()) return st.error();

    // Upload A's structure AND values ONCE -- unlike theta/diag_add/rhs
    // below, A itself never changes across the Newton solves one
    // solve_problem() call makes (see this context's own doc comment).
    const auto& csr = system.a->csr;
    const auto& csc = system.a->csc;
    st = cuda_check(cudaMemcpy(dev.csr_offsets, csr.offsets().data(), (m + 1) * sizeof(int),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(cg csr_off, host->device)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemcpy(dev.csr_indices, csr.indices().data(), nnz * sizeof(int),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(cg csr_idx, host->device)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemcpy(dev.csr_values, csr.values().data(), nnz * sizeof(double),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(cg csr_val, host->device)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemcpy(dev.csc_offsets, csc.offsets().data(), (n + 1) * sizeof(int),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(cg csc_off, host->device)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemcpy(dev.csc_indices, csc.indices().data(), nnz * sizeof(int),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(cg csc_idx, host->device)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemcpy(dev.csc_values, csc.values().data(), nnz * sizeof(double),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(cg csc_val, host->device)");
    if (!st.ok()) return st.error();

    // mat_a: A itself, m x n -- used for `A * t`.
    st = cusparse_check(
        cusparseCreateCsr(&dev.mat_a, m_i, n_i, nnz_i, dev.csr_offsets, dev.csr_indices,
                          dev.csr_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
        "cusparseCreateCsr(A)");
    if (!st.ok()) return st.error();

    // mat_a_transpose: A^T, n x m -- A's CSC read directly as CSR-of-A^T (a
    // CSC major slice for column j of A lists exactly row j of A^T). Used
    // for `A^T * p`. See SparseMatrix.hpp's header comment: CSC exists
    // specifically so this needs no transpose operation, on GPU or off.
    st = cusparse_check(
        cusparseCreateCsr(&dev.mat_a_transpose, n_i, m_i, nnz_i, dev.csc_offsets, dev.csc_indices,
                          dev.csc_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
        "cusparseCreateCsr(A^T)");
    if (!st.ok()) return st.error();

    st = cusparse_check(cusparseCreateDnVec(&dev.vec_p, m_i, dev.p, CUDA_R_64F),
                        "cusparseCreateDnVec(p)");
    if (!st.ok()) return st.error();
    st = cusparse_check(cusparseCreateDnVec(&dev.vec_t, n_i, dev.t, CUDA_R_64F),
                        "cusparseCreateDnVec(t)");
    if (!st.ok()) return st.error();
    st = cusparse_check(cusparseCreateDnVec(&dev.vec_ap, m_i, dev.ap, CUDA_R_64F),
                        "cusparseCreateDnVec(ap)");
    if (!st.ok()) return st.error();

    const double one = 1.0;
    const double zero = 0.0;
    std::size_t buf_at = 0;
    std::size_t buf_a = 0;
    st = cusparse_check(
        cusparseSpMV_bufferSize(dev.sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                dev.mat_a_transpose, dev.vec_p, &zero, dev.vec_t, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, &buf_at),
        "cusparseSpMV_bufferSize(A^T)");
    if (!st.ok()) return st.error();
    st = cusparse_check(
        cusparseSpMV_bufferSize(dev.sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                dev.mat_a, dev.vec_t, &zero, dev.vec_ap, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, &buf_a),
        "cusparseSpMV_bufferSize(A)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.spmv_buffer_at, buf_at > 0 ? buf_at : 1),
                    "cudaMalloc(cg spmv_buffer_at)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.spmv_buffer_a, buf_a > 0 ? buf_a : 1),
                    "cudaMalloc(cg spmv_buffer_a)");
    if (!st.ok()) return st.error();

    // Only now, with every allocation confirmed to have succeeded, is it
    // safe to mark the context as sized -- same partial-failure policy as
    // every other persistent context in this file.
    dev.m = m;
    dev.n = n;
    dev.nnz = nnz;
  }

  // -- per-solve upload: theta, diag_add, rhs change every Newton solve ------
  st = cuda_check(
      cudaMemcpy(dev.theta, system.theta.data(), n * sizeof(double), cudaMemcpyHostToDevice),
      "cudaMemcpy(cg theta, host->device)");
  if (!st.ok()) return st.error();
  st = cuda_check(cudaMemcpy(dev.diag_add, system.diag_add.data(), m * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(cg diag_add, host->device)");
  if (!st.ok()) return st.error();
  st = cuda_check(
      cudaMemcpy(dev.r, system.rhs.data(), m * sizeof(double), cudaMemcpyHostToDevice),
      "cudaMemcpy(cg rhs->r, host->device)");
  if (!st.ok()) return st.error();
  st = cuda_check(cudaMemset(dev.x, 0, m * sizeof(double)), "cudaMemset(cg x)");
  if (!st.ok()) return st.error();

  // Jacobi preconditioner diagonal, built host-side from system.a's CSR rows
  // (O(nnz), one pass) -- diag_i = diag_add_i + sum_j A_ij^2 * theta_j.
  {
    core::RealVector precond_inv_host(m);
    const auto& csr = system.a->csr;
    for (std::size_t i = 0; i < m; ++i) {
      Real d = system.diag_add[i];
      for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
        const Real aij = csr.values()[k];
        const auto j = static_cast<std::size_t>(csr.indices()[k]);
        d += aij * aij * system.theta[j];
      }
      precond_inv_host[i] = d > 0.0 ? 1.0 / d : 1.0;
    }
    st = cuda_check(cudaMemcpy(dev.precond_diag_inv, precond_inv_host.data(), m * sizeof(double),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(cg precond_diag_inv, host->device)");
    if (!st.ok()) return st.error();
  }

  const double one = 1.0;
  const double zero = 0.0;
  const auto apply_operator = [&]() -> Status {
    // Ap = A*T*A^T*p + diag_add.*p, applied without ever forming a matrix.
    Status inner =
        cusparse_check(cusparseSpMV(dev.sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                    dev.mat_a_transpose, dev.vec_p, &zero, dev.vec_t, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dev.spmv_buffer_at),
                       "cusparseSpMV(A^T*p)");
    if (!inner.ok()) return inner;
    inner = hadamard(dev.t, dev.theta, dev.t, n);
    if (!inner.ok()) return inner;
    inner =
        cusparse_check(cusparseSpMV(dev.sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                    dev.mat_a, dev.vec_t, &zero, dev.vec_ap, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dev.spmv_buffer_a),
                       "cusparseSpMV(A*t)");
    if (!inner.ok()) return inner;
    return hadamard_add(dev.diag_add, dev.p, dev.ap, m);
  };

  // Try IC(0) (Preconditioner.hpp) once per solve; fall back to the plain
  // Jacobi diagonal for this ENTIRE solve if it fails (a structural zero
  // pivot, or simply nothing built yet) -- see that file's doc comment for
  // why this is a real, expected failure mode, not a bug. `use_ic0` is
  // mutable: `precond` below also clears it if `ic0_apply` itself ever
  // fails mid-solve, so the rest of this CG run degrades gracefully instead
  // of erroring out.
  // Preconditioner, strongest first: the exact cuDSS factor (CG then only
  // absorbs the dense columns and rounding), IC(0), then the Jacobi diagonal.
  bool use_direct =
      try_direct && cudss_build(*system.a, system.theta, system.diag_add, dev.is_dense_column).ok();
  bool use_ic0 =
      !use_direct && ic0_build(*system.a, system.theta, system.diag_add, dev.is_dense_column).ok();
  const auto precond = [&](const double* r_in, double* z_out) -> Status {
    if (use_direct) {
      if (cudss_apply(r_in, z_out, m).ok()) return Status::Ok();
      use_direct = false;
      use_ic0 = ic0_build(*system.a, system.theta, system.diag_add, dev.is_dense_column).ok();
    }
    if (use_ic0) {
      Status ic0_status = ic0_apply(r_in, z_out, m);
      if (ic0_status.ok()) return Status::Ok();
      use_ic0 = false;
    }
    return hadamard(r_in, dev.precond_diag_inv, z_out, m);
  };

  double rhs_norm = 0.0;
  st = cublas_check(cublasDnrm2(dev.blas_handle, m32, dev.r, 1, &rhs_norm), "cublasDnrm2(rhs)");
  if (!st.ok()) return st.error();

  bool converged = (rhs_norm == 0.0);  // an all-zero rhs solves to dy=0 trivially
  int iterations = 0;

  if (!converged) {
    st = precond(dev.r, dev.z);
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemcpy(dev.p, dev.z, m * sizeof(double), cudaMemcpyDeviceToDevice),
                    "cudaMemcpy(cg p<-z, device->device)");
    if (!st.ok()) return st.error();

    double rho_old = 0.0;
    st = cublas_check(cublasDdot(dev.blas_handle, m32, dev.r, 1, dev.z, 1, &rho_old),
                      "cublasDdot(rho0)");
    if (!st.ok()) return st.error();

    for (; iterations < cg_max_iterations; ++iterations) {
      st = apply_operator();
      if (!st.ok()) return st.error();

      double p_ap = 0.0;
      st = cublas_check(cublasDdot(dev.blas_handle, m32, dev.p, 1, dev.ap, 1, &p_ap),
                        "cublasDdot(p.Ap)");
      if (!st.ok()) return st.error();
      if (p_ap == 0.0) break;  // breakdown guard -- reported as non-convergence below

      const double alpha = rho_old / p_ap;
      st = cublas_check(cublasDaxpy(dev.blas_handle, m32, &alpha, dev.p, 1, dev.x, 1),
                        "cublasDaxpy(x+=alpha*p)");
      if (!st.ok()) return st.error();
      const double neg_alpha = -alpha;
      st = cublas_check(cublasDaxpy(dev.blas_handle, m32, &neg_alpha, dev.ap, 1, dev.r, 1),
                        "cublasDaxpy(r-=alpha*Ap)");
      if (!st.ok()) return st.error();

      double r_norm = 0.0;
      st = cublas_check(cublasDnrm2(dev.blas_handle, m32, dev.r, 1, &r_norm), "cublasDnrm2(r)");
      if (!st.ok()) return st.error();
      if (r_norm / rhs_norm < cg_tolerance) {
        converged = true;
        ++iterations;
        break;
      }

      st = precond(dev.r, dev.z);
      if (!st.ok()) return st.error();
      double rho_new = 0.0;
      st = cublas_check(cublasDdot(dev.blas_handle, m32, dev.r, 1, dev.z, 1, &rho_new),
                        "cublasDdot(rho_new)");
      if (!st.ok()) return st.error();
      const double beta = rho_new / rho_old;
      st = cublas_check(cublasDscal(dev.blas_handle, m32, &beta, dev.p, 1), "cublasDscal(beta*p)");
      if (!st.ok()) return st.error();
      st = cublas_check(cublasDaxpy(dev.blas_handle, m32, &one, dev.z, 1, dev.p, 1),
                        "cublasDaxpy(p=z+beta*p)");
      if (!st.ok()) return st.error();
      rho_old = rho_new;
    }
  }

  LinearSolveResult result;
  result.solution = core::RealVector(m);
  st = cuda_check(
      cudaMemcpy(result.solution.data(), dev.x, m * sizeof(double), cudaMemcpyDeviceToHost),
      "cudaMemcpy(cg dy, device->host)");
  if (!st.ok()) return st.error();

  result.refinement_passes = 0;
  // See LinearSolveResult::pivot_ratio's doc comment (LinearSolver.hpp) for
  // why +infinity, not a numeric ratio, signals non-convergence here.
  result.pivot_ratio = converged ? 1.0 : std::numeric_limits<Real>::infinity();
  return std::move(result);
}

/// Preconditioned MINRES (Paige & Saunders' original formulation -- the
/// standard reference algorithm for symmetric indefinite systems, not
/// something derived here; correctness is checked against dense LU by the
/// unit cross-check in solver_gpu_algorithms_test.cpp, the same way
/// solve_spd_cg's algebra is checked against solve_spd_dense). Every scalar
/// (alfa, beta, the Givens rotation cs/sn, gamma, phi, phibar, ...) is host
/// arithmetic -- only the vector operations (the operator apply, dot
/// products, axpy/scal/copy) touch the device.
Expected<LinearSolveResult> solve_minres(const KktSystem& system, Real minres_tolerance,
                                          int minres_max_iterations) {
  const std::size_t dim = system.matrix.rows();
  const std::size_t nnz = system.matrix.nnz();
  if (system.matrix.cols() != dim || system.rhs.size() != dim ||
      system.precond_diag.size() != dim) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "solve_minres: KktSystem dimensions are inconsistent");
  }
  if (dim == 0) return LinearSolveResult{};

  PersistentMinresContext& dev = minres_context();

  Status st = Status::Ok();
  if (!dev.sparse_handle) {
    st = cusparse_check(cusparseCreate(&dev.sparse_handle), "cusparseCreate");
    if (!st.ok()) return st.error();
  }
  if (!dev.blas_handle) {
    st = cublas_check(cublasCreate(&dev.blas_handle), "cublasCreate");
    if (!st.ok()) return st.error();
  }

  const auto dim_i = static_cast<int64_t>(dim);
  const auto nnz_i = static_cast<int64_t>(nnz);
  const int dim32 = static_cast<int>(dim);

  if (dev.dim != dim || dev.nnz != nnz) {
    dev.free_buffers();

    st = cuda_check(cudaMalloc(&dev.csr_offsets, (dim + 1) * sizeof(int)),
                    "cudaMalloc(minres csr_off)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.csr_indices, nnz * sizeof(int)), "cudaMalloc(minres csr_idx)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.csr_values, nnz * sizeof(double)),
                    "cudaMalloc(minres csr_val)");
    if (!st.ok()) return st.error();

    st = cuda_check(cudaMalloc(&dev.x, dim * sizeof(double)), "cudaMalloc(minres x)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.r1, dim * sizeof(double)), "cudaMalloc(minres r1)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.r2, dim * sizeof(double)), "cudaMalloc(minres r2)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.v, dim * sizeof(double)), "cudaMalloc(minres v)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.w, dim * sizeof(double)), "cudaMalloc(minres w)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.w1, dim * sizeof(double)), "cudaMalloc(minres w1)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.w2, dim * sizeof(double)), "cudaMalloc(minres w2)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.y, dim * sizeof(double)), "cudaMalloc(minres y)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.precond_diag_inv, dim * sizeof(double)),
                    "cudaMalloc(minres precond_diag_inv)");
    if (!st.ok()) return st.error();

    // The matrix's structure (indices) is fixed for the whole solve_problem()
    // call -- unlike KktBuilder.cu's VALUES, which change every Newton solve
    // (Theta/D_s depend on the current point), so only the values are
    // re-uploaded below, not re-allocated/re-described here.
    st = cuda_check(cudaMemcpy(dev.csr_offsets, system.matrix.csr.offsets().data(),
                               (dim + 1) * sizeof(int), cudaMemcpyHostToDevice),
                    "cudaMemcpy(minres csr_off, host->device)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemcpy(dev.csr_indices, system.matrix.csr.indices().data(),
                               nnz * sizeof(int), cudaMemcpyHostToDevice),
                    "cudaMemcpy(minres csr_idx, host->device)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemcpy(dev.csr_values, system.matrix.csr.values().data(),
                               nnz * sizeof(double), cudaMemcpyHostToDevice),
                    "cudaMemcpy(minres csr_val, host->device)");
    if (!st.ok()) return st.error();

    st = cusparse_check(
        cusparseCreateCsr(&dev.mat, dim_i, dim_i, nnz_i, dev.csr_offsets, dev.csr_indices,
                          dev.csr_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
        "cusparseCreateCsr(minres matrix)");
    if (!st.ok()) return st.error();
    st = cusparse_check(cusparseCreateDnVec(&dev.vec_v, dim_i, dev.v, CUDA_R_64F),
                        "cusparseCreateDnVec(minres v)");
    if (!st.ok()) return st.error();
    st = cusparse_check(cusparseCreateDnVec(&dev.vec_y, dim_i, dev.y, CUDA_R_64F),
                        "cusparseCreateDnVec(minres y)");
    if (!st.ok()) return st.error();

    const double one = 1.0;
    const double zero = 0.0;
    std::size_t buf = 0;
    st = cusparse_check(
        cusparseSpMV_bufferSize(dev.sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                dev.mat, dev.vec_v, &zero, dev.vec_y, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, &buf),
        "cusparseSpMV_bufferSize(minres)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMalloc(&dev.spmv_buffer, buf > 0 ? buf : 1),
                    "cudaMalloc(minres spmv_buffer)");
    if (!st.ok()) return st.error();

    dev.dim = dim;
    dev.nnz = nnz;
  } else {
    // Same matrix shape as a previous call -- structure is unchanged, but
    // the VALUES (Theta/D_s at the current point) are not, so refresh them.
    st = cuda_check(cudaMemcpy(dev.csr_values, system.matrix.csr.values().data(),
                               nnz * sizeof(double), cudaMemcpyHostToDevice),
                    "cudaMemcpy(minres csr_val refresh, host->device)");
    if (!st.ok()) return st.error();
  }

  st = cuda_check(
      cudaMemcpy(dev.precond_diag_inv, system.precond_diag.data(), dim * sizeof(double),
                 cudaMemcpyHostToDevice),
      "cudaMemcpy(minres precond_diag_inv upload, host->device)");
  if (!st.ok()) return st.error();
  // precond_diag holds the diagonal MAGNITUDE (KktBuilder.cu already takes
  // |.|) -- invert it in place on the device via the same hadamard kernel,
  // multiplying by itself's reciprocal is awkward, so reciprocate host-side
  // before upload instead (dim-sized, O(dim), negligible).
  {
    core::RealVector inv_host(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      inv_host[i] = system.precond_diag[i] > 0.0 ? 1.0 / system.precond_diag[i] : 1.0;
    }
    st = cuda_check(cudaMemcpy(dev.precond_diag_inv, inv_host.data(), dim * sizeof(double),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(minres precond_diag_inv, host->device)");
    if (!st.ok()) return st.error();
  }

  st = cuda_check(cudaMemset(dev.x, 0, dim * sizeof(double)), "cudaMemset(minres x)");
  if (!st.ok()) return st.error();
  st = cuda_check(
      cudaMemcpy(dev.r1, system.rhs.data(), dim * sizeof(double), cudaMemcpyHostToDevice),
      "cudaMemcpy(minres rhs->r1, host->device)");
  if (!st.ok()) return st.error();

  st = hadamard(dev.r1, dev.precond_diag_inv, dev.y, dim);
  if (!st.ok()) return st.error();

  double beta1 = 0.0;
  {
    double dot_r1_y = 0.0;
    st = cublas_check(cublasDdot(dev.blas_handle, dim32, dev.r1, 1, dev.y, 1, &dot_r1_y),
                      "cublasDdot(beta1)");
    if (!st.ok()) return st.error();
    beta1 = std::sqrt(std::max(dot_r1_y, 0.0));
  }

  bool converged = (beta1 == 0.0);  // an all-zero rhs solves to x=0 trivially
  int iterations = 0;

  if (!converged) {
    constexpr double kMinresEps = 1e-300;  // guards a Givens division only, not a physical floor

    double oldb = 0.0;
    double beta = beta1;
    double dbar = 0.0;
    double epsln = 0.0;
    double phibar = beta1;
    double cs = -1.0;
    double sn = 0.0;

    st = cuda_check(cudaMemcpy(dev.r2, dev.r1, dim * sizeof(double), cudaMemcpyDeviceToDevice),
                    "cudaMemcpy(minres r2<-r1, device->device)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemset(dev.w, 0, dim * sizeof(double)), "cudaMemset(minres w)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemset(dev.w1, 0, dim * sizeof(double)), "cudaMemset(minres w1)");
    if (!st.ok()) return st.error();
    st = cuda_check(cudaMemset(dev.w2, 0, dim * sizeof(double)), "cudaMemset(minres w2)");
    if (!st.ok()) return st.error();

    const double one = 1.0;
    const double zero = 0.0;

    for (; iterations < minres_max_iterations; ++iterations) {
      // -- Lanczos step: extend the tridiagonal system by one row/column ----
      const double s = 1.0 / beta;
      st = cuda_check(cudaMemcpy(dev.v, dev.y, dim * sizeof(double), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy(minres v<-y, device->device)");
      if (!st.ok()) return st.error();
      st = cublas_check(cublasDscal(dev.blas_handle, dim32, &s, dev.v, 1), "cublasDscal(v)");
      if (!st.ok()) return st.error();

      // y = A*v, written directly into dev.y (never into dev.r1 -- r1 still
      // holds the previous iteration's Lanczos vector, read by name two
      // lines below, and must not be clobbered by this apply).
      st = cusparse_check(
          cusparseSpMV(dev.sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, dev.mat,
                      dev.vec_v, &zero, dev.vec_y, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                      dev.spmv_buffer),
          "cusparseSpMV(minres A*v)");
      if (!st.ok()) return st.error();

      if (iterations >= 1) {
        const double neg_ratio = -(beta / oldb);
        st = cublas_check(cublasDaxpy(dev.blas_handle, dim32, &neg_ratio, dev.r1, 1, dev.y, 1),
                          "cublasDaxpy(minres y-=...*r1)");
        if (!st.ok()) return st.error();
      }

      double alfa = 0.0;
      st = cublas_check(cublasDdot(dev.blas_handle, dim32, dev.v, 1, dev.y, 1, &alfa),
                        "cublasDdot(alfa)");
      if (!st.ok()) return st.error();
      const double neg_alfa_over_beta = -(alfa / beta);
      st = cublas_check(
          cublasDaxpy(dev.blas_handle, dim32, &neg_alfa_over_beta, dev.r2, 1, dev.y, 1),
          "cublasDaxpy(minres y-=...*r2)");
      if (!st.ok()) return st.error();

      st = cuda_check(cudaMemcpy(dev.r1, dev.r2, dim * sizeof(double), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy(minres r1<-r2, device->device)");
      if (!st.ok()) return st.error();
      st = cuda_check(cudaMemcpy(dev.r2, dev.y, dim * sizeof(double), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy(minres r2<-y, device->device)");
      if (!st.ok()) return st.error();
      st = hadamard(dev.r2, dev.precond_diag_inv, dev.y, dim);
      if (!st.ok()) return st.error();

      oldb = beta;
      double dot_r2_y = 0.0;
      st = cublas_check(cublasDdot(dev.blas_handle, dim32, dev.r2, 1, dev.y, 1, &dot_r2_y),
                        "cublasDdot(beta)");
      if (!st.ok()) return st.error();
      beta = std::sqrt(std::max(dot_r2_y, 0.0));

      // -- apply the previous Givens rotation, then compute the next one ----
      const double oldeps = epsln;
      const double delta = cs * dbar + sn * alfa;
      const double gbar = sn * dbar - cs * alfa;
      epsln = sn * beta;
      dbar = -cs * beta;

      double gamma = std::sqrt(gbar * gbar + beta * beta);
      gamma = std::max(gamma, kMinresEps);
      cs = gbar / gamma;
      sn = beta / gamma;
      const double phi = cs * phibar;
      phibar = sn * phibar;

      // -- update the search direction and the solution ---------------------
      const double denom = 1.0 / gamma;
      st = cuda_check(cudaMemcpy(dev.w1, dev.w2, dim * sizeof(double), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy(minres w1<-w2, device->device)");
      if (!st.ok()) return st.error();
      st = cuda_check(cudaMemcpy(dev.w2, dev.w, dim * sizeof(double), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy(minres w2<-w, device->device)");
      if (!st.ok()) return st.error();
      st = cuda_check(cudaMemcpy(dev.w, dev.v, dim * sizeof(double), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy(minres w<-v, device->device)");
      if (!st.ok()) return st.error();
      const double neg_oldeps = -oldeps;
      st = cublas_check(cublasDaxpy(dev.blas_handle, dim32, &neg_oldeps, dev.w1, 1, dev.w, 1),
                        "cublasDaxpy(minres w-=oldeps*w1)");
      if (!st.ok()) return st.error();
      const double neg_delta = -delta;
      st = cublas_check(cublasDaxpy(dev.blas_handle, dim32, &neg_delta, dev.w2, 1, dev.w, 1),
                        "cublasDaxpy(minres w-=delta*w2)");
      if (!st.ok()) return st.error();
      st = cublas_check(cublasDscal(dev.blas_handle, dim32, &denom, dev.w, 1),
                        "cublasDscal(minres w*=denom)");
      if (!st.ok()) return st.error();
      st = cublas_check(cublasDaxpy(dev.blas_handle, dim32, &phi, dev.w, 1, dev.x, 1),
                        "cublasDaxpy(minres x+=phi*w)");
      if (!st.ok()) return st.error();

      // phibar is MINRES's own running estimate of the residual norm --
      // standard practice for this algorithm, not an approximation added
      // here; recomputing the true residual every iteration would cost
      // another full operator apply for no accuracy this algorithm needs.
      if (phibar / beta1 < minres_tolerance) {
        converged = true;
        ++iterations;
        break;
      }
    }
  }

  LinearSolveResult result;
  result.solution = core::RealVector(dim);
  st = cuda_check(
      cudaMemcpy(result.solution.data(), dev.x, dim * sizeof(double), cudaMemcpyDeviceToHost),
      "cudaMemcpy(minres solution, device->host)");
  if (!st.ok()) return st.error();

  result.refinement_passes = 0;
  result.pivot_ratio = converged ? 1.0 : std::numeric_limits<Real>::infinity();
  return std::move(result);
}

}  // namespace sovsolve::solver::gpu
