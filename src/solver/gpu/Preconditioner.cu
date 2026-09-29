#include "sovsolve/solver/gpu/Preconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "sovsolve/solver/SparseLdl.hpp"

#ifdef SOVSOLVE_HAVE_CUDSS
#include <cudss.h>
#endif

namespace sovsolve::solver::gpu {

namespace {

Status cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return core::make_error(core::ErrorCode::NumericalError,
                            std::string(what) + ": " + cudaGetErrorString(err));
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

/// Device state for the IC(0) factor and the two triangular-solve analyses
/// built from it. Mirrors LinearSolver.cu's PersistentCgContext idiom: device
/// allocation and the (expensive) symbolic pattern only redone when the
/// problem's dimensions change; the numeric values, the factorization
/// itself, and both solve analyses are redone every `ic0_build` call, since
/// `theta`/`diag_add` change every Newton solve.
///
/// Two cuSPARSE API generations, deliberately: `cusparseDcsric02` (the
/// incomplete-Cholesky factorization itself) has no generic-API equivalent,
/// so it uses the legacy `cusparseMatDescr_t`/`csric02Info_t`. The
/// triangular solves DO have a generic equivalent (`cusparseSpSV`) and, on
/// the CUDA toolkit this builds against, the legacy `csrsv2` solve family is
/// unavailable at link time -- confirmed by the build, not assumed -- so
/// those two solves use `cusparseSpMatDescr_t`/`cusparseSpSVDescr_t` over
/// the SAME underlying CSR device arrays `descr` describes.
struct PersistentIc0Context {
  cusparseHandle_t handle = nullptr;
  cusparseMatDescr_t descr = nullptr;         // legacy: for cusparseDcsric02 only
  cusparseSpMatDescr_t mat_generic = nullptr;  // generic: for the two SpSV solves
  csric02Info_t ic0_info = nullptr;
  cusparseSpSVDescr_t spsv_lower = nullptr;  // L y = r
  cusparseSpSVDescr_t spsv_upper = nullptr;  // L^T z = y
  // Dense-vector descriptors reused across every apply, per cuSPARSE's own
  // recommended pattern for a fixed matrix solved against changing RHS:
  // analysis binds these once, then `cusparseDnVecSetValues` repoints
  // `vec_lower_x`/`vec_upper_y` to the CALLER's r/z pointers at apply time.
  // `vec_lower_y`/`vec_upper_x` never move -- both are `scratch_y`.
  cusparseDnVecDescr_t vec_lower_x = nullptr;
  cusparseDnVecDescr_t vec_lower_y = nullptr;
  cusparseDnVecDescr_t vec_upper_x = nullptr;
  cusparseDnVecDescr_t vec_upper_y = nullptr;

  int* offsets = nullptr;    // m+1
  int* indices = nullptr;    // nnz_m
  double* values = nullptr;  // nnz_m -- the CSR values, factored in place by csric02

  void* ic0_buffer = nullptr;
  void* spsv_lower_buffer = nullptr;
  void* spsv_upper_buffer = nullptr;
  double* scratch_y = nullptr;  // length m, intermediate between the two solves

  std::size_t m = 0;
  std::size_t nnz_m = 0;
  bool ready = false;  // true once a successful ic0_build has run this call

  void free_device_buffers() {
    if (mat_generic) cusparseDestroySpMat(mat_generic);
    mat_generic = nullptr;
    if (vec_lower_x) cusparseDestroyDnVec(vec_lower_x);
    if (vec_lower_y) cusparseDestroyDnVec(vec_lower_y);
    if (vec_upper_x) cusparseDestroyDnVec(vec_upper_x);
    if (vec_upper_y) cusparseDestroyDnVec(vec_upper_y);
    vec_lower_x = vec_lower_y = vec_upper_x = vec_upper_y = nullptr;
    if (offsets) cudaFree(offsets);
    if (indices) cudaFree(indices);
    if (values) cudaFree(values);
    if (ic0_buffer) cudaFree(ic0_buffer);
    if (spsv_lower_buffer) cudaFree(spsv_lower_buffer);
    if (spsv_upper_buffer) cudaFree(spsv_upper_buffer);
    if (scratch_y) cudaFree(scratch_y);
    offsets = nullptr;
    indices = nullptr;
    values = nullptr;
    ic0_buffer = spsv_lower_buffer = spsv_upper_buffer = nullptr;
    scratch_y = nullptr;
    m = nnz_m = 0;
    ready = false;
  }

  ~PersistentIc0Context() {
    free_device_buffers();
    if (ic0_info) cusparseDestroyCsric02Info(ic0_info);
    if (spsv_lower) cusparseSpSV_destroyDescr(spsv_lower);
    if (spsv_upper) cusparseSpSV_destroyDescr(spsv_upper);
    if (descr) cusparseDestroyMatDescr(descr);
    if (handle) cusparseDestroy(handle);
  }
};

PersistentIc0Context& ic0_context() {
  static PersistentIc0Context ctx;
  return ctx;
}

/// Host-side symbolic + numeric build of M_sparse's lower triangle (including
/// the diagonal), in one pass -- see Preconditioner.hpp's doc comment for why
/// this is O(sum over non-dense columns j of (rows touching j)^2), and why
/// dense columns are excluded from the off-diagonal fill but not from the
/// diagonal. A sparse-accumulator (SPA) pattern: `marker[i']` records the
/// last row `i` that touched position `i'`, so `accum` needs no O(m) reset
/// between rows.
void build_host_pattern(const core::SparseMatrixPair<>& a, const RealVector& theta,
                        const RealVector& diag_add, const std::vector<bool>& is_dense_column,
                        std::vector<int>& offsets, std::vector<int>& indices,
                        std::vector<double>& values) {
  const std::size_t m = a.rows();
  const auto& csr = a.csr;
  const auto& csc = a.csc;

  std::vector<core::Index> marker(m, -1);
  std::vector<Real> accum(m, 0.0);
  std::vector<core::Index> row_fill;

  offsets.assign(m + 1, 0);
  indices.clear();
  values.clear();
  indices.reserve(m * 4);  // a rough guess; grows as needed, never realloc-storms
  values.reserve(m * 4);

  for (std::size_t i = 0; i < m; ++i) {
    row_fill.clear();
    marker[i] = static_cast<core::Index>(i);
    accum[i] = diag_add[i];
    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      const auto j = static_cast<std::size_t>(csr.indices()[k]);
      if (!is_dense_column[j]) continue;
      const Real aij = csr.values()[k];
      accum[i] += theta[j] * aij * aij;
    }
    row_fill.push_back(static_cast<core::Index>(i));

    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      const auto j = static_cast<std::size_t>(csr.indices()[k]);
      if (is_dense_column[j]) continue;
      const Real aij = csr.values()[k];
      const Real tj = theta[j];
      for (auto k2 = csc.slice_begin(j); k2 < csc.slice_end(j); ++k2) {
        const auto i2 = static_cast<std::size_t>(csc.indices()[k2]);
        if (i2 > i) continue;  // lower triangle only
        const Real ai2j = csc.values()[k2];
        if (marker[i2] != static_cast<core::Index>(i)) {
          marker[i2] = static_cast<core::Index>(i);
          accum[i2] = 0.0;
          row_fill.push_back(static_cast<core::Index>(i2));
        }
        accum[i2] += tj * aij * ai2j;
      }
    }

    std::sort(row_fill.begin(), row_fill.end());
    offsets[i + 1] = offsets[i] + static_cast<int>(row_fill.size());
    for (const core::Index idx : row_fill) {
      indices.push_back(static_cast<int>(idx));
      values.push_back(accum[static_cast<std::size_t>(idx)]);
    }
  }
}

}  // namespace

Status ic0_build(const core::SparseMatrixPair<>& a, const RealVector& theta,
                 const RealVector& diag_add, const std::vector<bool>& is_dense_column) {
  const std::size_t m = a.rows();
  PersistentIc0Context& dev = ic0_context();
  dev.ready = false;
  if (m == 0) return Status::Ok();

  Status st = Status::Ok();
  if (!dev.handle) {
    st = cusparse_check(cusparseCreate(&dev.handle), "cusparseCreate(ic0)");
    if (!st.ok()) return st;
  }
  if (!dev.descr) {
    st = cusparse_check(cusparseCreateMatDescr(&dev.descr), "cusparseCreateMatDescr");
    if (!st.ok()) return st;
    cusparseSetMatType(dev.descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(dev.descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(dev.descr, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(dev.descr, CUSPARSE_INDEX_BASE_ZERO);
  }
  if (!dev.ic0_info) {
    st = cusparse_check(cusparseCreateCsric02Info(&dev.ic0_info), "cusparseCreateCsric02Info");
    if (!st.ok()) return st;
  }
  if (!dev.spsv_lower) {
    st = cusparse_check(cusparseSpSV_createDescr(&dev.spsv_lower), "cusparseSpSV_createDescr(L)");
    if (!st.ok()) return st;
  }
  if (!dev.spsv_upper) {
    st = cusparse_check(cusparseSpSV_createDescr(&dev.spsv_upper), "cusparseSpSV_createDescr(U)");
    if (!st.ok()) return st;
  }

  std::vector<int> host_offsets;
  std::vector<int> host_indices;
  std::vector<double> host_values;
  build_host_pattern(a, theta, diag_add, is_dense_column, host_offsets, host_indices,
                     host_values);
  const std::size_t nnz_m = host_indices.size();

  if (dev.m != m || dev.nnz_m != nnz_m) {
    dev.free_device_buffers();

    st = cuda_check(cudaMalloc(&dev.offsets, (m + 1) * sizeof(int)), "cudaMalloc(ic0 offsets)");
    if (!st.ok()) return st;
    st = cuda_check(cudaMalloc(&dev.indices, nnz_m * sizeof(int)), "cudaMalloc(ic0 indices)");
    if (!st.ok()) return st;
    st = cuda_check(cudaMalloc(&dev.values, nnz_m * sizeof(double)), "cudaMalloc(ic0 values)");
    if (!st.ok()) return st;
    st = cuda_check(cudaMalloc(&dev.scratch_y, m * sizeof(double)), "cudaMalloc(ic0 scratch_y)");
    if (!st.ok()) return st;

    // The generic descriptor binds to these exact device pointers, so it is
    // rebuilt whenever they are (the factored VALUES change every call, but
    // that's an in-place update through the same pointer -- no need to
    // rebuild the descriptor for that, only when the pointers themselves do).
    st = cusparse_check(
        cusparseCreateCsr(&dev.mat_generic, static_cast<int64_t>(m), static_cast<int64_t>(m),
                          static_cast<int64_t>(nnz_m), dev.offsets, dev.indices, dev.values,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_64F),
        "cusparseCreateCsr(ic0 generic)");
    if (!st.ok()) return st;
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    st = cusparse_check(
        cusparseSpMatSetAttribute(dev.mat_generic, CUSPARSE_SPMAT_FILL_MODE, &fill_lower,
                                  sizeof(fill_lower)),
        "cusparseSpMatSetAttribute(fill mode)");
    if (!st.ok()) return st;
    cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
    st = cusparse_check(
        cusparseSpMatSetAttribute(dev.mat_generic, CUSPARSE_SPMAT_DIAG_TYPE, &diag_non_unit,
                                  sizeof(diag_non_unit)),
        "cusparseSpMatSetAttribute(diag type)");
    if (!st.ok()) return st;

    // vec_lower_x/vec_upper_y start out pointing at scratch_y as a
    // placeholder -- ic0_apply repoints them to the caller's r/z pointers
    // via cusparseDnVecSetValues before every solve. vec_lower_y/vec_upper_x
    // are scratch_y itself and never move.
    const auto m_i64 = static_cast<int64_t>(m);
    st = cusparse_check(cusparseCreateDnVec(&dev.vec_lower_x, m_i64, dev.scratch_y, CUDA_R_64F),
                        "cusparseCreateDnVec(ic0 lower x)");
    if (!st.ok()) return st;
    st = cusparse_check(cusparseCreateDnVec(&dev.vec_lower_y, m_i64, dev.scratch_y, CUDA_R_64F),
                        "cusparseCreateDnVec(ic0 lower y)");
    if (!st.ok()) return st;
    st = cusparse_check(cusparseCreateDnVec(&dev.vec_upper_x, m_i64, dev.scratch_y, CUDA_R_64F),
                        "cusparseCreateDnVec(ic0 upper x)");
    if (!st.ok()) return st;
    st = cusparse_check(cusparseCreateDnVec(&dev.vec_upper_y, m_i64, dev.scratch_y, CUDA_R_64F),
                        "cusparseCreateDnVec(ic0 upper y)");
    if (!st.ok()) return st;

    dev.m = m;
    dev.nnz_m = nnz_m;
  }

  st = cuda_check(cudaMemcpy(dev.offsets, host_offsets.data(), (m + 1) * sizeof(int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(ic0 offsets, host->device)");
  if (!st.ok()) return st;
  st = cuda_check(cudaMemcpy(dev.indices, host_indices.data(), nnz_m * sizeof(int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(ic0 indices, host->device)");
  if (!st.ok()) return st;
  st = cuda_check(cudaMemcpy(dev.values, host_values.data(), nnz_m * sizeof(double),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(ic0 values, host->device)");
  if (!st.ok()) return st;

  const int m_i = static_cast<int>(m);
  const int nnz_i = static_cast<int>(nnz_m);

  int ic0_buf = 0;
  st = cusparse_check(cusparseDcsric02_bufferSize(dev.handle, m_i, nnz_i, dev.descr, dev.values,
                                                  dev.offsets, dev.indices, dev.ic0_info,
                                                  &ic0_buf),
                      "cusparseDcsric02_bufferSize");
  if (!st.ok()) return st;
  const double sv_alpha = 1.0;
  std::size_t lower_buf = 0;
  st = cusparse_check(
      cusparseSpSV_bufferSize(dev.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &sv_alpha,
                              dev.mat_generic, dev.vec_lower_x, dev.vec_lower_y, CUDA_R_64F,
                              CUSPARSE_SPSV_ALG_DEFAULT, dev.spsv_lower, &lower_buf),
      "cusparseSpSV_bufferSize(L)");
  if (!st.ok()) return st;
  std::size_t upper_buf = 0;
  st = cusparse_check(
      cusparseSpSV_bufferSize(dev.handle, CUSPARSE_OPERATION_TRANSPOSE, &sv_alpha,
                              dev.mat_generic, dev.vec_upper_x, dev.vec_upper_y, CUDA_R_64F,
                              CUSPARSE_SPSV_ALG_DEFAULT, dev.spsv_upper, &upper_buf),
      "cusparseSpSV_bufferSize(U)");
  if (!st.ok()) return st;

  const auto ensure_buffer = [](void*& buf, std::size_t& current_size,
                                std::size_t needed) -> Status {
    if (needed == 0) return Status::Ok();
    if (needed <= current_size && buf != nullptr) return Status::Ok();
    if (buf) cudaFree(buf);
    buf = nullptr;
    Status st2 = cuda_check(cudaMalloc(&buf, needed), "cudaMalloc(ic0/sv buffer)");
    if (!st2.ok()) return st2;
    current_size = needed;
    return Status::Ok();
  };
  static thread_local std::size_t ic0_buf_size = 0;
  static thread_local std::size_t lower_buf_size = 0;
  static thread_local std::size_t upper_buf_size = 0;
  st = ensure_buffer(dev.ic0_buffer, ic0_buf_size, static_cast<std::size_t>(ic0_buf));
  if (!st.ok()) return st;
  st = ensure_buffer(dev.spsv_lower_buffer, lower_buf_size, lower_buf);
  if (!st.ok()) return st;
  st = ensure_buffer(dev.spsv_upper_buffer, upper_buf_size, upper_buf);
  if (!st.ok()) return st;

  st = cusparse_check(
      cusparseDcsric02_analysis(dev.handle, m_i, nnz_i, dev.descr, dev.values, dev.offsets,
                                dev.indices, dev.ic0_info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                dev.ic0_buffer),
      "cusparseDcsric02_analysis");
  if (!st.ok()) return st;

  st = cusparse_check(cusparseDcsric02(dev.handle, m_i, nnz_i, dev.descr, dev.values, dev.offsets,
                                       dev.indices, dev.ic0_info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                       dev.ic0_buffer),
                      "cusparseDcsric02");
  if (!st.ok()) return st;

  int zero_pivot = -1;
  const cusparseStatus_t pivot_status =
      cusparseXcsric02_zeroPivot(dev.handle, dev.ic0_info, &zero_pivot);
  if (pivot_status == CUSPARSE_STATUS_ZERO_PIVOT) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "ic0_build: IC(0) hit a structural zero pivot at row " +
                                std::to_string(zero_pivot) +
                                " -- falling back to Jacobi for this solve");
  }
  st = cusparse_check(pivot_status, "cusparseXcsric02_zeroPivot");
  if (!st.ok()) return st;

  st = cusparse_check(
      cusparseSpSV_analysis(dev.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &sv_alpha,
                            dev.mat_generic, dev.vec_lower_x, dev.vec_lower_y, CUDA_R_64F,
                            CUSPARSE_SPSV_ALG_DEFAULT, dev.spsv_lower, dev.spsv_lower_buffer),
      "cusparseSpSV_analysis(L)");
  if (!st.ok()) return st;
  st = cusparse_check(
      cusparseSpSV_analysis(dev.handle, CUSPARSE_OPERATION_TRANSPOSE, &sv_alpha, dev.mat_generic,
                            dev.vec_upper_x, dev.vec_upper_y, CUDA_R_64F,
                            CUSPARSE_SPSV_ALG_DEFAULT, dev.spsv_upper, dev.spsv_upper_buffer),
      "cusparseSpSV_analysis(U)");
  if (!st.ok()) return st;

  dev.ready = true;
  return Status::Ok();
}

Status ic0_apply(const Real* r_device, Real* z_device, std::size_t m) {
  PersistentIc0Context& dev = ic0_context();
  if (!dev.ready || dev.m != m) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "ic0_apply: called without a successful ic0_build for this size");
  }

  // cusparseSpSV_solve computes op(A)*y = alpha*x -- x is the given RHS,
  // y is what gets solved for. Repoint the two ends that actually change
  // call to call (the middle, scratch_y, never moves); analysis above ran
  // once against these same descriptor OBJECTS, which is what matters, not
  // against these exact pointer values.
  Status st = cusparse_check(
      cusparseDnVecSetValues(dev.vec_lower_x, const_cast<Real*>(r_device)),
      "cusparseDnVecSetValues(lower x)");
  if (!st.ok()) return st;
  st = cusparse_check(cusparseDnVecSetValues(dev.vec_upper_y, z_device),
                      "cusparseDnVecSetValues(upper y)");
  if (!st.ok()) return st;

  const double alpha = 1.0;
  st = cusparse_check(
      cusparseSpSV_solve(dev.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, dev.mat_generic,
                        dev.vec_lower_x, dev.vec_lower_y, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                        dev.spsv_lower),
      "cusparseSpSV_solve(L)");
  if (!st.ok()) return st;

  st = cusparse_check(
      cusparseSpSV_solve(dev.handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, dev.mat_generic,
                        dev.vec_upper_x, dev.vec_upper_y, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
                        dev.spsv_upper),
      "cusparseSpSV_solve(U)");
  return st;
}

// ---------------------------------------------------------------------------
// In-house exact factor on the host. See Preconditioner.hpp.
// ---------------------------------------------------------------------------

namespace {

struct HostLdlContext {
  SparseLdl ldl;
  std::vector<int> lower_offsets, lower_indices;  ///< pattern of the cached analysis
  std::vector<std::size_t> col_ptr, row_idx;      ///< full symmetric pattern
  std::vector<std::size_t> lower_to_full_a, lower_to_full_b;  ///< where each lower entry goes
  std::vector<Real> full_values, work;
  std::size_t m = 0;
  bool ready = false;
};

HostLdlContext& host_ldl_context() {
  static HostLdlContext ctx;
  return ctx;
}

/// SparseLdl pivots at or below this fraction of their starting diagonal are
/// treated as dependent rows (Wright 1999, see SparseLdl.hpp).
constexpr Real kHostPivotTolerance = 1e-14;

}  // namespace

Status host_ldl_build(const core::SparseMatrixPair<>& a, const RealVector& theta,
                      const RealVector& diag_add, const std::vector<bool>& is_dense_column) {
  HostLdlContext& ctx = host_ldl_context();
  ctx.ready = false;
  const std::size_t m = a.rows();
  if (m == 0) return Status::Ok();

  std::vector<int> offsets, indices;
  std::vector<double> values;
  build_host_pattern(a, theta, diag_add, is_dense_column, offsets, indices, values);

  const bool same = ctx.m == m && ctx.lower_offsets == offsets && ctx.lower_indices == indices;
  if (!same) {
    // Lower CSR (row i, columns <= i) -> full symmetric compressed columns,
    // remembering where each lower entry lands (once, or twice off-diagonal).
    std::vector<std::size_t> count(m, 0);
    for (std::size_t i = 0; i < m; ++i) {
      for (int p = offsets[i]; p < offsets[i + 1]; ++p) {
        const auto j = static_cast<std::size_t>(indices[static_cast<std::size_t>(p)]);
        ++count[j];
        if (j != i) ++count[i];
      }
    }
    ctx.col_ptr.assign(m + 1, 0);
    for (std::size_t j = 0; j < m; ++j) ctx.col_ptr[j + 1] = ctx.col_ptr[j] + count[j];
    ctx.row_idx.assign(ctx.col_ptr[m], 0);
    std::vector<std::size_t> next(ctx.col_ptr.begin(), ctx.col_ptr.end() - 1);
    ctx.lower_to_full_a.assign(indices.size(), 0);
    ctx.lower_to_full_b.assign(indices.size(), static_cast<std::size_t>(-1));
    for (std::size_t i = 0; i < m; ++i) {
      for (int p = offsets[i]; p < offsets[i + 1]; ++p) {
        const auto e = static_cast<std::size_t>(p);
        const auto j = static_cast<std::size_t>(indices[e]);
        ctx.row_idx[next[j]] = i;  // entry (i, j) in column j
        ctx.lower_to_full_a[e] = next[j]++;
        if (j != i) {
          ctx.row_idx[next[i]] = j;  // its mirror (j, i) in column i
          ctx.lower_to_full_b[e] = next[i]++;
        }
      }
    }
    Status st = ctx.ldl.analyze(m, ctx.col_ptr, ctx.row_idx);
    if (!st.ok()) return st;
    ctx.lower_offsets = std::move(offsets);
    ctx.lower_indices = std::move(indices);
    ctx.full_values.assign(ctx.row_idx.size(), 0.0);
    ctx.work.assign(m, 0.0);
    ctx.m = m;
  }

  for (std::size_t e = 0; e < values.size(); ++e) {
    ctx.full_values[ctx.lower_to_full_a[e]] = values[e];
    if (ctx.lower_to_full_b[e] != static_cast<std::size_t>(-1)) {
      ctx.full_values[ctx.lower_to_full_b[e]] = values[e];
    }
  }
  Status st = ctx.ldl.factorize(ctx.full_values, kHostPivotTolerance);
  if (!st.ok()) return st;
  ctx.ready = true;
  return Status::Ok();
}

Status host_ldl_apply(const Real* r_device, Real* z_device, std::size_t m) {
  HostLdlContext& ctx = host_ldl_context();
  if (!ctx.ready || ctx.m != m) {
    return core::make_error(core::ErrorCode::NumericalError, "host_ldl_apply: no factor");
  }
  Status st = cuda_check(
      cudaMemcpy(ctx.work.data(), r_device, m * sizeof(double), cudaMemcpyDeviceToHost),
      "cudaMemcpy(host ldl r, device->host)");
  if (!st.ok()) return st;
  ctx.ldl.solve(ctx.work);
  for (std::size_t i = 0; i < m; ++i) {
    if (!std::isfinite(ctx.work[i])) {
      return core::make_error(core::ErrorCode::NumericalError, "host_ldl_apply: non-finite");
    }
  }
  return cuda_check(
      cudaMemcpy(z_device, ctx.work.data(), m * sizeof(double), cudaMemcpyHostToDevice),
      "cudaMemcpy(host ldl z, host->device)");
}

// ---------------------------------------------------------------------------
// cuDSS: the exact factor. See Preconditioner.hpp.
// ---------------------------------------------------------------------------

#ifdef SOVSOLVE_HAVE_CUDSS

namespace {

Status cudss_check(cudssStatus_t st, const char* what) {
  if (st == CUDSS_STATUS_SUCCESS) return Status::Ok();
  return core::make_error(core::ErrorCode::NumericalError,
                          std::string(what) + " failed (cudssStatus " +
                              std::to_string(static_cast<int>(st)) + ")");
}

struct PersistentCudssContext {
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t mat = nullptr;
  cudssMatrix_t x = nullptr;
  cudssMatrix_t b = nullptr;

  int* offsets = nullptr;
  int* indices = nullptr;
  double* values = nullptr;
  double* xbuf = nullptr;
  double* bbuf = nullptr;

  /// The pattern the cached analysis belongs to.
  std::vector<int> host_offsets, host_indices;
  std::size_t m = 0;
  bool ready = false;

  void release() {
    if (mat) cudssMatrixDestroy(mat);
    if (x) cudssMatrixDestroy(x);
    if (b) cudssMatrixDestroy(b);
    if (data && handle) cudssDataDestroy(handle, data);
    mat = x = b = nullptr;
    data = nullptr;
    if (offsets) cudaFree(offsets);
    if (indices) cudaFree(indices);
    if (values) cudaFree(values);
    if (xbuf) cudaFree(xbuf);
    if (bbuf) cudaFree(bbuf);
    offsets = indices = nullptr;
    values = xbuf = bbuf = nullptr;
    host_offsets.clear();
    host_indices.clear();
    m = 0;
    ready = false;
  }

  ~PersistentCudssContext() {
    release();
    if (config) cudssConfigDestroy(config);
    if (handle) cudssDestroy(handle);
  }
};

PersistentCudssContext& cudss_context() {
  static PersistentCudssContext ctx;
  return ctx;
}

}  // namespace

bool cudss_available() noexcept { return true; }

Status cudss_build(const core::SparseMatrixPair<>& a, const RealVector& theta,
                   const RealVector& diag_add, const std::vector<bool>& is_dense_column) {
  PersistentCudssContext& dev = cudss_context();
  dev.ready = false;
  const std::size_t m = a.rows();
  if (m == 0) return Status::Ok();

  std::vector<int> offsets, indices;
  std::vector<double> values;
  build_host_pattern(a, theta, diag_add, is_dense_column, offsets, indices, values);
  const std::size_t nnz = indices.size();

  Status st = Status::Ok();
  if (!dev.handle) {
    st = cudss_check(cudssCreate(&dev.handle), "cudssCreate");
    if (!st.ok()) return st;
    st = cudss_check(cudssConfigCreate(&dev.config), "cudssConfigCreate");
    if (!st.ok()) return st;
    // Measured: without this, the same model gave Optimal on one run and
    // NotConverged on the next (Netlib capri, sctap1). Results must not
    // depend on the run -- the project's determinism rule.
    int deterministic = 1;
    st = cudss_check(cudssConfigSet(dev.config, CUDSS_CONFIG_DETERMINISTIC_MODE, &deterministic,
                                    sizeof(deterministic)),
                     "cudssConfigSet(deterministic)");
    if (!st.ok()) return st;
  }

  const bool same_pattern =
      dev.mat != nullptr && dev.m == m && dev.host_offsets == offsets && dev.host_indices == indices;
  if (!same_pattern) {
    dev.release();
    st = cuda_check(cudaMalloc(&dev.offsets, (m + 1) * sizeof(int)), "cudaMalloc(cudss off)");
    if (!st.ok()) return st;
    st = cuda_check(cudaMalloc(&dev.indices, nnz * sizeof(int)), "cudaMalloc(cudss idx)");
    if (!st.ok()) return st;
    st = cuda_check(cudaMalloc(&dev.values, nnz * sizeof(double)), "cudaMalloc(cudss val)");
    if (!st.ok()) return st;
    st = cuda_check(cudaMalloc(&dev.xbuf, m * sizeof(double)), "cudaMalloc(cudss x)");
    if (!st.ok()) return st;
    st = cuda_check(cudaMalloc(&dev.bbuf, m * sizeof(double)), "cudaMalloc(cudss b)");
    if (!st.ok()) return st;
    st = cuda_check(cudaMemcpy(dev.offsets, offsets.data(), (m + 1) * sizeof(int),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(cudss off)");
    if (!st.ok()) return st;
    st = cuda_check(
        cudaMemcpy(dev.indices, indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy(cudss idx)");
    if (!st.ok()) return st;
    st = cudss_check(cudssDataCreate(dev.handle, &dev.data), "cudssDataCreate");
    if (!st.ok()) return st;
    const auto m64 = static_cast<int64_t>(m);
    st = cudss_check(cudssMatrixCreateCsr(&dev.mat, m64, m64, static_cast<int64_t>(nnz),
                                          dev.offsets, nullptr, dev.indices, dev.values,
                                          CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                                          CUDSS_MTYPE_SPD,
                                          CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO),
                     "cudssMatrixCreateCsr");
    if (!st.ok()) return st;
    st = cudss_check(
        cudssMatrixCreateDn(&dev.x, m64, 1, m64, dev.xbuf, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
        "cudssMatrixCreateDn(x)");
    if (!st.ok()) return st;
    st = cudss_check(
        cudssMatrixCreateDn(&dev.b, m64, 1, m64, dev.bbuf, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
        "cudssMatrixCreateDn(b)");
    if (!st.ok()) return st;
    dev.host_offsets = std::move(offsets);
    dev.host_indices = std::move(indices);
    dev.m = m;
  }

  st = cuda_check(
      cudaMemcpy(dev.values, values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice),
      "cudaMemcpy(cudss val)");
  if (!st.ok()) return st;

  if (!same_pattern) {
    st = cudss_check(cudssExecute(dev.handle, CUDSS_PHASE_ANALYSIS, dev.config, dev.data,
                                  dev.mat, dev.x, dev.b),
                     "cudssExecute(analysis)");
    if (!st.ok()) {
      dev.release();
      return st;
    }
  }
  st = cudss_check(cudssExecute(dev.handle, CUDSS_PHASE_FACTORIZATION, dev.config, dev.data,
                                dev.mat, dev.x, dev.b),
                   "cudssExecute(factorization)");
  if (!st.ok()) return st;

  // A Cholesky that met a non-positive pivot reports it here, not through the
  // status: the factor is unusable, so the caller falls back.
  int info = 0;
  std::size_t written = 0;
  st = cudss_check(cudssDataGet(dev.handle, dev.data, CUDSS_DATA_INFO, &info, sizeof(info),
                                &written),
                   "cudssDataGet(info)");
  if (!st.ok()) return st;
  if (info != 0) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "cudss_build: Cholesky met a non-positive pivot (info " +
                                std::to_string(info) + ")");
  }
  dev.ready = true;
  return Status::Ok();
}

Status cudss_apply(const Real* r_device, Real* z_device, std::size_t m) {
  PersistentCudssContext& dev = cudss_context();
  if (!dev.ready || dev.m != m) {
    return core::make_error(core::ErrorCode::NumericalError, "cudss_apply: no factor");
  }
  Status st = cuda_check(
      cudaMemcpy(dev.bbuf, r_device, m * sizeof(double), cudaMemcpyDeviceToDevice),
      "cudaMemcpy(cudss r->b)");
  if (!st.ok()) return st;
  st = cudss_check(cudssExecute(dev.handle, CUDSS_PHASE_SOLVE, dev.config, dev.data, dev.mat,
                                dev.x, dev.b),
                   "cudssExecute(solve)");
  if (!st.ok()) return st;
  return cuda_check(cudaMemcpy(z_device, dev.xbuf, m * sizeof(double), cudaMemcpyDeviceToDevice),
                    "cudaMemcpy(cudss x->z)");
}

#else  // no cuDSS in this build

bool cudss_available() noexcept { return false; }

Status cudss_build(const core::SparseMatrixPair<>&, const RealVector&, const RealVector&,
                   const std::vector<bool>&) {
  return core::make_error(core::ErrorCode::UnsupportedFeature,
                          "cudss_build: this build does not link cuDSS");
}

Status cudss_apply(const Real*, Real*, std::size_t) {
  return core::make_error(core::ErrorCode::UnsupportedFeature,
                          "cudss_apply: this build does not link cuDSS");
}

#endif  // SOVSOLVE_HAVE_CUDSS

}  // namespace sovsolve::solver::gpu
