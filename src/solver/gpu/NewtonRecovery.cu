#include "sovsolve/solver/gpu/NewtonRecovery.hpp"

#include <cstddef>

#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

Status recover_newton_direction(const CanonicalProblem& problem, const KktSystem& system,
                                 const Residuals& residuals,
                                 const RealVector& linear_solution, SolverState& state) {
  if (system.descriptor.type != ReductionType::QpAugmentedKkt) {
    return core::make_error(
        core::ErrorCode::NotImplemented,
        "recover_newton_direction: only ReductionType::QpAugmentedKkt is implemented");
  }

  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;
  const std::size_t m_i = problem.num_inequality_rows();

  if (linear_solution.size() != n + m || residuals.rp.size() != m ||
      residuals.rxz.size() != n || residuals.ruv.size() != n) {
    return core::make_error(
        core::ErrorCode::DimensionMismatch,
        "recover_newton_direction: linear_solution/residuals size does not match problem");
  }

  // On the augmented path linear_solution already IS [dx; dy].
  state.dx = core::RealVector(n);
  state.dy = core::RealVector(m);
  for (std::size_t j = 0; j < n; ++j) state.dx[j] = linear_solution[j];
  for (std::size_t i = 0; i < m; ++i) state.dy[i] = linear_solution[n + i];

  // dz_j = (-rxz_j - z_j dx_j) / (x_j - l_j), finite lower bound only.
  // dv_j = (-ruv_j + v_j dx_j) / (u_j - x_j), finite upper bound only.
  state.dz = core::RealVector(n, 0.0);
  state.dv = core::RealVector(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    if (core::is_finite_bound(problem.col_lower[j])) {
      state.dz[j] = (-residuals.rxz[j] - state.z[j] * state.dx[j]) /
                    safe_gap(state.x[j] - problem.col_lower[j]);
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      state.dv[j] = (-residuals.ruv[j] + state.v[j] * state.dx[j]) /
                    safe_gap(problem.col_upper[j] - state.x[j]);
    }
  }

  // ds_k = -rp_I_k - (A_I dx)_k, over every inequality row.
  state.ds = core::RealVector(m_i);
  const auto& A_csr = problem.A.csr;
  for (std::size_t k = 0; k < m_i; ++k) {
    const std::size_t i = m_e + k;
    Real a_dx = 0.0;
    for (std::size_t idx = A_csr.slice_begin(i); idx < A_csr.slice_end(i); ++idx) {
      a_dx += A_csr.values()[idx] * state.dx[static_cast<std::size_t>(A_csr.indices()[idx])];
    }
    state.ds[k] = -residuals.rp[i] - a_dx;
  }

  return Status::Ok();
}

}  // namespace sovsolve::solver::gpu
