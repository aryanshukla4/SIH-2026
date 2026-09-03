// Module 14. Separate primal/dual ratio tests against
// `lower < x + ap*dx < upper`, `s + ap*ds > 0`, `z + ad*dz > 0`,
// `v + ad*dv > 0`, `slack_dual(y_I + ad*dy_I) > 0` -- using the
// `slack_dual()` helper (SolverState.hpp), never an inline `-y > 0` test.
// Equality-row `y` is never ratio-tested. `eta` is the safety factor from
// Options::IpmOptions (0.995 default).

#ifndef SOVSOLVE_SOLVER_GPU_STEP_LENGTH_HPP
#define SOVSOLVE_SOLVER_GPU_STEP_LENGTH_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Real;
using core::Status;
using model::CanonicalProblem;

/// STUB: signature only.
[[nodiscard]] Status compute_step_lengths(const CanonicalProblem& problem,
                                           const SolverState& state, Real eta,
                                           Real& alpha_primal, Real& alpha_dual);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_STEP_LENGTH_HPP
