// Module 15: x += ap*dx, s += ap*ds, y += ad*dy, z += ad*dz, v += ad*dv.
// Must not run until Module 19 (Diagnostics) has recorded the iteration that
// produced `state`'s direction fields -- module.txt Module 15.

#ifndef SOVSOLVE_SOLVER_GPU_STATE_UPDATE_HPP
#define SOVSOLVE_SOLVER_GPU_STATE_UPDATE_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Status;
using model::CanonicalProblem;

/// Applies `state.alpha_primal`/`state.alpha_dual` (set by
/// gpu::compute_step_lengths) to `state`'s direction fields in place, then
/// clamps every coordinate's distance from its own bound up to
/// `kConvergedFloor` (StepLength.cu) if the update left it any closer --
/// this is what makes that file's ratio test safe to EXCLUDE an
/// already-converged coordinate from: excluding it stops it throttling
/// every other variable's step, but only if it is then kept from drifting
/// past its bound by a now-larger step meant for everyone else. Never
/// applies to a coordinate whose corresponding bound is infinite (nothing to
/// clamp against) or that genuinely still had room (the clamp is a no-op
/// there, since the ratio test already guaranteed the update stays inside).
[[nodiscard]] Status apply_step(const CanonicalProblem& problem, SolverState& state);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_STATE_UPDATE_HPP
