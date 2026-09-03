// Module 13. Uses `system.descriptor` to know what was actually solved: on
// the normal-equations path `dy` comes out of the linear solve and
// `dx, ds, dz, dv` are recovered from it; on the augmented path the full
// direction set comes out directly, and `dy` is not privileged (module.txt
// Module 13: "never assume dy is always the solved direction").

#ifndef SOVSOLVE_SOLVER_GPU_NEWTON_RECOVERY_HPP
#define SOVSOLVE_SOLVER_GPU_NEWTON_RECOVERY_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/KktSystem.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::RealVector;
using core::Status;
using model::CanonicalProblem;

/// STUB: signature only.
[[nodiscard]] Status recover_newton_direction(const CanonicalProblem& problem,
                                               const KktSystem& system,
                                               const RealVector& linear_solution,
                                               SolverState& state);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_NEWTON_RECOVERY_HPP
