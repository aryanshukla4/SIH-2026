// Module 14. Separate primal/dual ratio tests against
// `lower < x + ap*dx < upper`, `s + ap*ds > 0`, `z + ad*dz > 0`,
// `v + ad*dv > 0`, `slack_dual(y_I + ad*dy_I) > 0` -- using the
// `slack_dual()` helper (SolverState.hpp), never an inline `-y > 0` test.
// Equality-row `y` is never ratio-tested. `eta` is the safety factor from
// Options::IpmOptions (0.995 default).
//
// FORMULATION.md 8:
//
//     alpha_p = eta * max { a in (0,1] : l < x+a*dx < u  and  s+a*ds > 0 }
//     alpha_d = eta * max { a in (0,1] : z+a*dz > 0, v+a*dv > 0,
//                                        slack_dual(y_I+a*dy_I) > 0 }
//
// Per-entry ratio, only where the direction moves toward the bound:
//
//     dx_j < 0, finite lower:  (x_j-l_j)/(-dx_j)     dx_j > 0, finite upper:  (u_j-x_j)/dx_j
//     ds_k < 0:                s_k/(-ds_k)
//     dz_j < 0, finite lower:  z_j/(-dz_j)           dv_j < 0, finite upper:  v_j/(-dv_j)
//     dy_i > 0 (inequality k=i-num_equality only):    slack_dual(y_i)/dy_i
//
// alpha_{p,d}_max starts at 1 and is the min over every applicable ratio.

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
