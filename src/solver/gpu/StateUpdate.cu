#include "sovsolve/solver/gpu/StateUpdate.hpp"

#include <cstddef>

namespace sovsolve::solver::gpu {

Status apply_step(const CanonicalProblem& problem, SolverState& state) {
  if (state.dx.size() != state.x.size() || state.ds.size() != state.s.size() ||
      state.dy.size() != state.y.size() || state.dz.size() != state.z.size() ||
      state.dv.size() != state.v.size()) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "apply_step: a direction field's size does not match its "
                            "corresponding state field");
  }

  const auto ap = state.alpha_primal;
  const auto ad = state.alpha_dual;
  const std::size_t m_e = problem.num_equality;

  for (std::size_t j = 0; j < state.x.size(); ++j) {
    state.x[j] += ap * state.dx[j];
    // See StepLength.cu / SolverState.hpp::kConvergedFloor: the ratio test
    // may have excluded this bound because it was ALREADY within the floor,
    // letting a larger step meant for other coordinates carry alongside it.
    // That exclusion is only unsafe if it lets x actually cross the bound --
    // so this must fire ONLY on a genuine violation (distance < 0), never on
    // "merely close but still valid". Clamping on "< floor" instead of "< 0"
    // was a real bug: on a wide row with thousands of columns legitimately
    // converging to sit near a bound (confirmed on datt256.mps, a 262144-col
    // binary relaxation with rows up to 57840 nonzeros), it force-resets
    // every one of them to the SAME synthetic bound+floor value every
    // iteration, discarding genuine Newton progress and freezing rp at
    // roughly (columns-near-bound-in-one-row * floor) forever instead of
    // letting it keep shrinking.
    if (core::is_finite_bound(problem.col_lower[j]) && state.x[j] < problem.col_lower[j]) {
      state.x[j] = problem.col_lower[j] + kConvergedFloor;
    }
    if (core::is_finite_bound(problem.col_upper[j]) && state.x[j] > problem.col_upper[j]) {
      state.x[j] = problem.col_upper[j] - kConvergedFloor;
    }

    state.z[j] += ad * state.dz[j];
    if (core::is_finite_bound(problem.col_lower[j]) && state.z[j] <= 0.0) {
      state.z[j] = kConvergedFloor;
    }
    state.v[j] += ad * state.dv[j];
    if (core::is_finite_bound(problem.col_upper[j]) && state.v[j] <= 0.0) {
      state.v[j] = kConvergedFloor;
    }
  }
  for (std::size_t k = 0; k < state.s.size(); ++k) {
    state.s[k] += ap * state.ds[k];
    if (state.s[k] <= 0.0) state.s[k] = kConvergedFloor;
  }
  for (std::size_t i = 0; i < state.y.size(); ++i) {
    state.y[i] += ad * state.dy[i];
    // Inequality rows only (i >= m_e): equality-row y is unrestricted in
    // sign and never ratio-tested, so it is never clamped either.
    if (i >= m_e && slack_dual(state.y[i]) <= 0.0) {
      state.y[i] = -kConvergedFloor;
    }
  }

  return Status::Ok();
}

}  // namespace sovsolve::solver::gpu
