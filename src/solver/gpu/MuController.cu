#include "sovsolve/solver/gpu/MuController.hpp"

#include <cstddef>

namespace sovsolve::solver::gpu {

namespace {

Status check_sizes(const CanonicalProblem& problem, const SolverState& state) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_i = problem.num_inequality_rows();
  if (state.x.size() != n || state.s.size() != m_i || state.y.size() != m ||
      state.z.size() != n || state.v.size() != n) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "MuController: SolverState size does not match problem");
  }
  return Status::Ok();
}

}  // namespace

Status update_mu(const CanonicalProblem& problem, SolverState& state) {
  if (auto st = check_sizes(problem, state); !st.ok()) return st;

  const std::size_t n = problem.num_cols();
  const std::size_t m_e = problem.num_equality;
  const std::size_t m_i = problem.num_inequality_rows();

  Real sum = 0.0;
  std::size_t active_pairs = m_i;
  for (std::size_t j = 0; j < n; ++j) {
    if (core::is_finite_bound(problem.col_lower[j])) {
      sum += (state.x[j] - problem.col_lower[j]) * state.z[j];
      ++active_pairs;
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      sum += (problem.col_upper[j] - state.x[j]) * state.v[j];
      ++active_pairs;
    }
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    sum += -state.s[k] * state.y[m_e + k];
  }

  state.mu = active_pairs > 0 ? sum / static_cast<Real>(active_pairs) : 1.0;
  return Status::Ok();
}

Status compute_mu_at_trial_point(const CanonicalProblem& problem, const SolverState& state,
                                  Real alpha_primal, Real alpha_dual, Real& mu_trial) {
  if (auto st = check_sizes(problem, state); !st.ok()) return st;

  const std::size_t n = problem.num_cols();
  const std::size_t m_e = problem.num_equality;
  const std::size_t m_i = problem.num_inequality_rows();

  if (state.dx.size() != n || state.ds.size() != m_i || state.dy.size() != problem.num_rows() ||
      state.dz.size() != n || state.dv.size() != n) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "compute_mu_at_trial_point: direction field size does not "
                            "match problem");
  }

  Real sum = 0.0;
  std::size_t active_pairs = m_i;
  for (std::size_t j = 0; j < n; ++j) {
    if (core::is_finite_bound(problem.col_lower[j])) {
      const Real x_trial = state.x[j] + alpha_primal * state.dx[j];
      const Real z_trial = state.z[j] + alpha_dual * state.dz[j];
      sum += (x_trial - problem.col_lower[j]) * z_trial;
      ++active_pairs;
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      const Real x_trial = state.x[j] + alpha_primal * state.dx[j];
      const Real v_trial = state.v[j] + alpha_dual * state.dv[j];
      sum += (problem.col_upper[j] - x_trial) * v_trial;
      ++active_pairs;
    }
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    const std::size_t i = m_e + k;
    const Real s_trial = state.s[k] + alpha_primal * state.ds[k];
    const Real y_trial = state.y[i] + alpha_dual * state.dy[i];
    sum += -s_trial * y_trial;
  }

  mu_trial = active_pairs > 0 ? sum / static_cast<Real>(active_pairs) : 1.0;
  return Status::Ok();
}

}  // namespace sovsolve::solver::gpu
