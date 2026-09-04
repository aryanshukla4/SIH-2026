#include "sovsolve/solver/gpu/StepLength.hpp"

#include <algorithm>
#include <cstddef>

namespace sovsolve::solver::gpu {

Status compute_step_lengths(const CanonicalProblem& problem, const SolverState& state,
                             Real eta, Real& alpha_primal, Real& alpha_dual) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;
  const std::size_t m_i = problem.num_inequality_rows();

  if (state.x.size() != n || state.dx.size() != n || state.s.size() != m_i ||
      state.ds.size() != m_i || state.z.size() != n || state.dz.size() != n ||
      state.v.size() != n || state.dv.size() != n || state.y.size() != m ||
      state.dy.size() != m) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "compute_step_lengths: SolverState size does not match problem");
  }

  Real alpha_p_max = 1.0;
  for (std::size_t j = 0; j < n; ++j) {
    if (state.dx[j] < 0.0 && core::is_finite_bound(problem.col_lower[j])) {
      alpha_p_max = std::min(alpha_p_max,
                              (state.x[j] - problem.col_lower[j]) / (-state.dx[j]));
    }
    if (state.dx[j] > 0.0 && core::is_finite_bound(problem.col_upper[j])) {
      alpha_p_max =
          std::min(alpha_p_max, (problem.col_upper[j] - state.x[j]) / state.dx[j]);
    }
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    if (state.ds[k] < 0.0) {
      alpha_p_max = std::min(alpha_p_max, state.s[k] / (-state.ds[k]));
    }
  }

  Real alpha_d_max = 1.0;
  for (std::size_t j = 0; j < n; ++j) {
    if (state.dz[j] < 0.0 && core::is_finite_bound(problem.col_lower[j])) {
      alpha_d_max = std::min(alpha_d_max, state.z[j] / (-state.dz[j]));
    }
    if (state.dv[j] < 0.0 && core::is_finite_bound(problem.col_upper[j])) {
      alpha_d_max = std::min(alpha_d_max, state.v[j] / (-state.dv[j]));
    }
  }
  // Inequality rows only -- equality-row y is never ratio-tested.
  for (std::size_t k = 0; k < m_i; ++k) {
    const std::size_t i = m_e + k;
    if (state.dy[i] > 0.0) {
      alpha_d_max = std::min(alpha_d_max, slack_dual(state.y[i]) / state.dy[i]);
    }
  }

  alpha_primal = eta * alpha_p_max;
  alpha_dual = eta * alpha_d_max;
  return Status::Ok();
}

}  // namespace sovsolve::solver::gpu
