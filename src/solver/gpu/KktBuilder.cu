#include "sovsolve/solver/gpu/KktBuilder.hpp"

#include <cmath>
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

  // Block-Jacobi preconditioner diagonal for solve_minres (LinearSolver.cu)
  // -- the SAME theta_inv/D_s scalars already computed above for the
  // matrix's own diagonal, just exposed instead of only being folded into
  // it. Magnitude, not signed value: the (1,1) block is negative definite,
  // but a Jacobi preconditioner needs a positive scaling.
  out.precond_diag = core::RealVector(dim);
  for (std::size_t j = 0; j < n; ++j) {
    out.precond_diag[j] = std::fabs(theta_inv[j] + delta_p);
  }
  for (std::size_t i = 0; i < m_e; ++i) {
    out.precond_diag[n + i] = delta_d;
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    const std::size_t i = m_e + k;
    out.precond_diag[n + i] = state.s[k] / safe_gap(-state.y[i]) + delta_d;
  }

  return Status::Ok();
}

Status build_normal_equations(const CanonicalProblem& problem, const SolverState& state,
                               const Residuals& residuals, Real delta_p, Real delta_d,
                               NormalEquationsSystem& out) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;
  const std::size_t m_i = problem.num_inequality_rows();

  if (state.x.size() != n || state.s.size() != m_i || state.y.size() != m ||
      state.z.size() != n || state.v.size() != n || residuals.rp.size() != m ||
      residuals.rd.size() != n) {
    return core::make_error(
        core::ErrorCode::DimensionMismatch,
        "build_normal_equations: SolverState/Residuals size does not match problem");
  }

  // T^-1, floored THEN inverted -- unlike build_kkt's augmented path above,
  // which never inverts it. FORMULATION.md 10.1.
  out.theta = core::RealVector(n, 0.0);
  std::size_t floor_activations = 0;
  for (std::size_t j = 0; j < n; ++j) {
    Real t_inv = 0.0;
    if (core::is_finite_bound(problem.col_lower[j])) {
      t_inv += state.z[j] / safe_gap(state.x[j] - problem.col_lower[j]);
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      t_inv += state.v[j] / safe_gap(problem.col_upper[j] - state.x[j]);
    }
    if (t_inv < delta_p) {
      t_inv = delta_p;
      ++floor_activations;
    }
    out.theta[j] = 1.0 / t_inv;
  }

  // rhs1, identical formula to build_kkt's rhs1 above -- both reductions
  // eliminate dz/dv/ds via the same six-block Newton system rows
  // (FORMULATION.md 7), independent of which system dx/dy come out of.
  out.rhs1 = core::RealVector(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    Real r1 = residuals.rd[j];
    if (core::is_finite_bound(problem.col_lower[j])) {
      r1 += residuals.rxz[j] / safe_gap(state.x[j] - problem.col_lower[j]);
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      r1 -= residuals.ruv[j] / safe_gap(problem.col_upper[j] - state.x[j]);
    }
    out.rhs1[j] = r1;
  }

  // rhs2 and D_s + delta_d*I, identical formulas to build_kkt's above.
  core::RealVector rhs2(m, 0.0);
  out.diag_add = core::RealVector(m, 0.0);
  for (std::size_t i = 0; i < m_e; ++i) {
    rhs2[i] = -residuals.rp[i];
    out.diag_add[i] = delta_d;
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    const std::size_t i = m_e + k;
    rhs2[i] = -(residuals.rp[i] - residuals.rsy[k] / safe_gap(-state.y[i]));
    out.diag_add[i] = state.s[k] / safe_gap(-state.y[i]) + delta_d;
  }

  // Non-owning reference to the ALREADY-sparse A -- solve_spd_cg
  // (LinearSolver.cu) applies it via cuSPARSE SpMV, never densifies it. See
  // NormalEquationsSystem::a's doc comment for the lifetime argument.
  out.a = &problem.A;
  const auto& A_csr = problem.A.csr;

  // rhs = rhs2 + A*T*rhs1 (Schur-complement elimination of dx -- see
  // NormalEquationsSystem's doc comment in KktBuilder.hpp for the derivation).
  core::RealVector t_rhs1(n);
  for (std::size_t j = 0; j < n; ++j) t_rhs1[j] = out.theta[j] * out.rhs1[j];

  out.rhs = std::move(rhs2);
  for (std::size_t i = 0; i < m; ++i) {
    Real acc = 0.0;
    for (std::size_t k = A_csr.slice_begin(i); k < A_csr.slice_end(i); ++k) {
      acc += A_csr.values()[k] * t_rhs1[static_cast<std::size_t>(A_csr.indices()[k])];
    }
    out.rhs[i] += acc;
  }

  out.descriptor.type = ReductionType::LpNormalEquationsDy;
  out.descriptor.theta_floor_activations = floor_activations;
  out.descriptor.reason =
      "LP (Q empty), Options::IpmOptions::use_normal_equations enabled: "
      "matrix-free A*T*A^T reduction, solved by CG (FORMULATION.md 10.1)";

  return Status::Ok();
}

}  // namespace sovsolve::solver::gpu
