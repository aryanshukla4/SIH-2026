// Module 15: x += ap*dx, s += ap*ds, y += ad*dy, z += ad*dz, v += ad*dv.
// Must not run until Module 19 (Diagnostics) has recorded the iteration that
// produced `state`'s direction fields -- module.txt Module 15.

#ifndef SOVSOLVE_SOLVER_GPU_STATE_UPDATE_HPP
#define SOVSOLVE_SOLVER_GPU_STATE_UPDATE_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Status;

/// STUB: signature only.
[[nodiscard]] Status apply_step(SolverState& state);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_STATE_UPDATE_HPP
