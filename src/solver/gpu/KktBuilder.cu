#include "sovsolve/solver/gpu/KktBuilder.hpp"

#include <cstddef>

#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

Status build_kkt(const CanonicalProblem& problem, const SolverState& state,
                  const Residuals& residuals,
                  const analysis::MatrixAnalysis& /*mat_analysis*/, Real delta_p,
                  Real delta_d, KktSystem& out) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;
  const std::size_t m_i = problem.num_inequality_rows();

  if (state.x.size() != n || state.s.size() != m_i || state.y.size() != m ||
      state.z.size() != n || state.v.size() != n || residuals.rp.size() != m ||
      residuals.rd.size() != n) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "build_kkt: SolverState/Residuals size does not match problem");
  }

  // T^-1_j = z_j/(x_j-l_j) + v_j/(u_j-x_j), each term omitted where that
  // bound is infinite. Not floored: on this path delta_p is added to the
  // diagonal directly (KktBuilder.hpp doc comment / FORMULATION.md 10.2).
  core::RealVector theta_inv(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    Real t = 0.0;
    if (core::is_finite_bound(problem.col_lower[j])) {
      t += state.z[j] / safe_gap(state.x[j] - problem.col_lower[j]);
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      t += state.v[j] / safe_gap(problem.col_upper[j] - state.x[j]);
    }
    theta_inv[j] = t;
  }

  const std::size_t dim = n + m;
  core::SparseBuilder builder(dim, dim);

  const auto& Q_csr = problem.Q.csr;
  const auto& A_csr = problem.A.csr;
  const bool has_q = !problem.Q.empty();

  // -- pass 1: count --------------------------------------------------------
  if (has_q) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t k = Q_csr.slice_begin(i); k < Q_csr.slice_end(i); ++k) {
        builder.count(static_cast<core::Index>(i), Q_csr.indices()[k]);
      }
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    builder.count(static_cast<core::Index>(j), static_cast<core::Index>(j));
  }
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = A_csr.slice_begin(i); k < A_csr.slice_end(i); ++k) {
      const auto j = A_csr.indices()[k];
      builder.count(static_cast<core::Index>(n + i), j);  // (2,1) = A
      builder.count(j, static_cast<core::Index>(n + i));  // (1,2) = A^T
    }
  }
  for (std::size_t i = 0; i < m; ++i) {
    builder.count(static_cast<core::Index>(n + i), static_cast<core::Index>(n + i));
  }

  auto alloc_status = builder.allocate();
  if (!alloc_status.ok()) return alloc_status;

  // -- pass 2: insert ---------------------------------------------------------
  // Block (1,1) = -(Q + T^-1 + delta_p I). Q's entries and the diagonal
  // T^-1+delta_p term are inserted as separate triplets at the same (j,j)
  // position where Q has a diagonal entry; SparseBuilder sums duplicates
  // (SparseBuilder.hpp), so they combine into the correct single value.
  if (has_q) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t k = Q_csr.slice_begin(i); k < Q_csr.slice_end(i); ++k) {
        builder.insert(static_cast<core::Index>(i), Q_csr.indices()[k], -Q_csr.values()[k]);
      }
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    builder.insert(static_cast<core::Index>(j), static_cast<core::Index>(j),
                    -(theta_inv[j] + delta_p));
  }

  // Blocks (2,1) = A and (1,2) = A^T, from one CSR pass over A.
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t k = A_csr.slice_begin(i); k < A_csr.slice_end(i); ++k) {
      const auto j = A_csr.indices()[k];
      const Real val = A_csr.values()[k];
      builder.insert(static_cast<core::Index>(n + i), j, val);
      builder.insert(j, static_cast<core::Index>(n + i), val);
    }
  }

  // Block (2,2) = D_s + delta_d I. D_s is 0 on equality rows (no slack);
  // D_s_k = s_k / (-y_I_k) on inequality rows.
  for (std::size_t i = 0; i < m_e; ++i) {
    builder.insert(static_cast<core::Index>(n + i), static_cast<core::Index>(n + i), delta_d);
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    const std::size_t i = m_e + k;
    const Real d_s = state.s[k] / safe_gap(-state.y[i]);
    builder.insert(static_cast<core::Index>(n + i), static_cast<core::Index>(n + i),
                    d_s + delta_d);
  }

  out.matrix = builder.finish();

  // -- right-hand side --------------------------------------------------------
  out.rhs = core::RealVector(dim);
  for (std::size_t j = 0; j < n; ++j) {
    Real rhs1 = residuals.rd[j];
    if (core::is_finite_bound(problem.col_lower[j])) {
      rhs1 += residuals.rxz[j] / safe_gap(state.x[j] - problem.col_lower[j]);
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      rhs1 -= residuals.ruv[j] / safe_gap(problem.col_upper[j] - state.x[j]);
    }
    out.rhs[j] = rhs1;
  }
  for (std::size_t i = 0; i < m_e; ++i) {
    out.rhs[n + i] = -residuals.rp[i];
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    const std::size_t i = m_e + k;
    out.rhs[n + i] = -(residuals.rp[i] - residuals.rsy[k] / safe_gap(-state.y[i]));
  }

  out.descriptor.type = ReductionType::QpAugmentedKkt;
  // Never inverted on this path, so nothing is clamped -- delta_p is added
  // to the diagonal directly instead of flooring T^-1 (see the doc comment).
  out.descriptor.theta_floor_activations = 0;
  out.descriptor.reason =
      "augmented path selected unconditionally: LpNormalEquationsDy needs a "
      "sparse A*Theta*A^T product this pass does not implement";

  return Status::Ok();
}

}  // namespace sovsolve::solver::gpu
