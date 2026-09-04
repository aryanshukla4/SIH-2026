// Module 13. Uses `system.descriptor` to know what was actually solved: on
// the normal-equations path `dy` comes out of the linear solve and
// `dx, ds, dz, dv` are recovered from it; on the augmented path the full
// direction set comes out directly, and `dy` is not privileged (module.txt
// Module 13: "never assume dy is always the solved direction").
//
// Only ReductionType::QpAugmentedKkt is implemented -- the only type
// gpu::build_kkt currently produces. LpNormalEquationsDy/QpSchurDy return
// NotImplemented, same scope narrowing as build_kkt itself.
//
// On the augmented path, `linear_solution` (length n+m) already IS
// [dx; dy] -- Module 12 solved for both directly. Recovery here means
// un-eliminating `ds`, `dz`, `dv` via the same three Newton-system rows
// (FORMULATION.md 7) that build_kkt used to eliminate them:
//
//     dz_j = (-rxz_j - z_j dx_j) / (x_j - l_j)     finite lower bound only, else 0
//     dv_j = (-ruv_j + v_j dx_j) / (u_j - x_j)      finite upper bound only, else 0
//     ds_k = -rp_I_k - (A_I dx)_k                   every inequality row

#ifndef SOVSOLVE_SOLVER_GPU_NEWTON_RECOVERY_HPP
#define SOVSOLVE_SOLVER_GPU_NEWTON_RECOVERY_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/KktSystem.hpp"
#include "sovsolve/solver/Residuals.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Real;
using core::RealVector;
using core::Status;
using model::CanonicalProblem;

/// Fills `state.dx, state.ds, state.dy, state.dz, state.dv` from
/// `linear_solution` (Module 12's raw output) and `residuals` (the same
/// Module 7 result `build_kkt` used to form `system`'s right-hand side).
[[nodiscard]] Status recover_newton_direction(const CanonicalProblem& problem,
                                               const KktSystem& system,
                                               const Residuals& residuals,
                                               const RealVector& linear_solution,
                                               SolverState& state);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_NEWTON_RECOVERY_HPP
