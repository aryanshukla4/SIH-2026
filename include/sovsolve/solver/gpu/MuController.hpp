// Module 16. mu = [sum((x-l).*z) + sum((u-x).*v) + sum(-s.*y_I)] /
// active_pair_count, where active_pair_count counts only structurally
// existing pairs -- never `2n + m_I` unconditionally, or a model with many
// free columns (gas11: 44%) understates mu.
//
// Two functions, both the same averaging formula, evaluated at different
// points: `update_mu` at the CURRENT (x,z,v,s,y); `compute_mu_at_trial_point`
// at a hypothetical x+ap*dx, z+ad*dz, v+ad*dv, s+ap*ds, y_I+ad*dy_I, using
// whatever direction is currently stored in `state`'s dx/ds/dy/dz/dv fields.
// The latter is what Mehrotra's predictor-corrector (PredictorCorrector.hpp)
// uses for mu_aff -- evaluated at the AFFINE direction and its own (eta=1)
// step lengths -- then computes sigma = clamp((mu_aff/mu)^3, 0, 1) itself;
// that clamp is simple enough not to need its own function here.

#ifndef SOVSOLVE_SOLVER_GPU_MU_CONTROLLER_HPP
#define SOVSOLVE_SOLVER_GPU_MU_CONTROLLER_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Real;
using core::Status;
using model::CanonicalProblem;

/// Sets `state.mu` from `state`'s current x, z, v, s, y.
[[nodiscard]] Status update_mu(const CanonicalProblem& problem, SolverState& state);

/// Evaluates the same formula at the trial point implied by `state`'s
/// CURRENT point plus its direction fields, scaled by `alpha_primal`/
/// `alpha_dual`. Does not mutate `state`.
[[nodiscard]] Status compute_mu_at_trial_point(const CanonicalProblem& problem,
                                                const SolverState& state,
                                                Real alpha_primal, Real alpha_dual,
                                                Real& mu_trial);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_MU_CONTROLLER_HPP
