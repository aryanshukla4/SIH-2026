#include "sovsolve/solver/Initializer.hpp"

#include <cstddef>
#include <string>

namespace sovsolve::solver {

namespace {

/// Offset from a single finite bound for an otherwise one-sided column.
/// Arbitrary but conventional -- any positive constant keeps the point
/// strictly interior; 1 is the usual textbook choice.
constexpr Real kUnboundedOffset = 1.0;

}  // namespace

Expected<SolverState> initialize(const CanonicalProblem& problem, const Options& /*options*/) {
  std::size_t bad_index = 0;
  bool bad_is_row = false;
  if (!problem.is_ipm_startable(&bad_index, &bad_is_row)) {
    return core::make_error(
        core::ErrorCode::NumericalError,
        std::string("model is not IPM-startable: ") +
            (bad_is_row ? "row " : "column ") + std::to_string(bad_index));
  }

  SolverState state;
  const std::size_t n = problem.num_cols();
  const std::size_t m_i = problem.num_inequality_rows();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;

  state.x = core::RealVector(n);
  state.z = core::RealVector(n);
  state.v = core::RealVector(n);
  state.s = core::RealVector(m_i, 1.0);
  state.y = core::RealVector(m, 0.0);
  // -y_I > 0 is a strict interiority requirement (FORMULATION.md 3), not
  // just a convenience -- equality-row y is unrestricted and stays 0, but
  // y_I = 0 both violates that contract and makes D_s = s/(-y_I) divide by
  // zero the moment the KKT builder runs on iteration 0.
  for (std::size_t i = m_e; i < m; ++i) state.y[i] = -1.0;

  Real sum_lower = 0.0;
  Real sum_upper = 0.0;
  std::size_t active_pairs = m_i;

  for (std::size_t j = 0; j < n; ++j) {
    const Real lo = problem.col_lower[j];
    const Real hi = problem.col_upper[j];
    const bool has_lower = core::is_finite_bound(lo);
    const bool has_upper = core::is_finite_bound(hi);

    if (has_lower && has_upper) {
      state.x[j] = 0.5 * (lo + hi);
    } else if (has_lower) {
      state.x[j] = lo + kUnboundedOffset;
    } else if (has_upper) {
      state.x[j] = hi - kUnboundedOffset;
    } else {
      state.x[j] = 0.0;
    }

    state.z[j] = has_lower ? 1.0 : 0.0;
    state.v[j] = has_upper ? 1.0 : 0.0;

    if (has_lower) {
      sum_lower += (state.x[j] - lo) * state.z[j];
      ++active_pairs;
    }
    if (has_upper) {
      sum_upper += (hi - state.x[j]) * state.v[j];
      ++active_pairs;
    }
  }

  // mu = [sum((x-l).*z) + sum((u-x).*v) + sum(-s.*y_I)] / active_pair_count
  // (Module 16's formula, FORMULATION.md 6). Guards the degenerate case of a
  // model with no bounds and no inequality rows at all -- unreachable in
  // practice, but division by zero is not a graceful way to find out.
  Real sum_slack = 0.0;
  for (std::size_t k = 0; k < m_i; ++k) {
    sum_slack += -state.s[k] * state.y[m_e + k];
  }
  state.mu = active_pairs > 0
                 ? (sum_lower + sum_upper + sum_slack) / static_cast<Real>(active_pairs)
                 : 1.0;

  return state;
}

}  // namespace sovsolve::solver
