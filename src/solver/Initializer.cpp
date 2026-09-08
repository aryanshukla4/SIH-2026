#include "sovsolve/solver/Initializer.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

namespace sovsolve::solver {

namespace {

/// Offset from a single finite bound for an otherwise one-sided column.
/// Arbitrary but conventional -- any positive constant keeps the point
/// strictly interior; 1 is the usual textbook choice.
constexpr Real kUnboundedOffset = 1.0;

}  // namespace

Expected<SolverState> initialize(const CanonicalProblem& problem, const Options& /*options*/,
                                  const core::RealVector* warm_start) {
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
  state.s = core::RealVector(m_i);
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

    const bool has_hint = warm_start != nullptr && j < warm_start->size() &&
                          core::is_finite_bound((*warm_start)[j]);
    if (has_hint) {
      // A real hint is clamped strictly inside the column's own bounds --
      // never used against a finite bound as-is -- the margin scales with
      // the box's own width so a binary column's [0,1] box (extremely
      // common in Module 22's branch-and-bound) is not distorted by a
      // margin sized for a wide one.
      Real x_hint = (*warm_start)[j];
      if (has_lower && has_upper) {
        const Real margin = std::min({1e-6, 0.001 * (hi - lo), 0.25 * (hi - lo)});
        x_hint = std::min(std::max(x_hint, lo + margin), hi - margin);
      } else if (has_lower) {
        x_hint = std::max(x_hint, lo + kUnboundedOffset * 1e-6);
      } else if (has_upper) {
        x_hint = std::min(x_hint, hi - kUnboundedOffset * 1e-6);
      }
      state.x[j] = x_hint;
    } else if (has_lower && has_upper) {
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

  // `s_k = b_i - (Ax)_i` -- the slack that makes row `i` EXACTLY satisfied by
  // the x just chosen above, rather than a flat 1.0 oblivious to the row's
  // own scale and RHS. On a large, tightly-coupled instance (80bau3b: 2262
  // inequality rows, 9799 columns) a flat slack left the primal residual at
  // ~4e5 from iteration 0 -- not because the bound-midpoint x was a
  // uniformly bad guess, but because s never even tried to match what x
  // already implied, so nearly every row started out "wrong" by whatever
  // its own RHS happened to be. Falls back to the textbook 1.0 only when the
  // implied slack isn't safely positive (the row is already tight or
  // violated at this x) -- interiority must never be sacrificed for a
  // closer-but-boundary-touching start.
  {
    const auto& csr = problem.A.csr;
    for (std::size_t k = 0; k < m_i; ++k) {
      const std::size_t i = m_e + k;
      Real activity = 0.0;
      for (auto t = csr.slice_begin(i); t < csr.slice_end(i); ++t) {
        activity += csr.values()[t] * state.x[static_cast<std::size_t>(csr.indices()[t])];
      }
      const Real implied_slack = problem.b[i] - activity;
      state.s[k] = implied_slack > kUnboundedOffset ? implied_slack : kUnboundedOffset;
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
