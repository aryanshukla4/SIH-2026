// Module 16. mu = [sum((x-l).*z) + sum((u-x).*v) + sum(-s.*y_I)] /
// active_pair_count, where active_pair_count counts only structurally
// existing pairs -- never `2n + m_I` unconditionally, or a model with many
// free columns (gas11: 44%) understates mu. Also computes mu_aff from the
// affine step and sigma = clamp((mu_aff/mu)^3, 0, 1).

#ifndef SOVSOLVE_SOLVER_GPU_MU_CONTROLLER_HPP
#define SOVSOLVE_SOLVER_GPU_MU_CONTROLLER_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Status;
using model::CanonicalProblem;

/// STUB: signature only.
[[nodiscard]] Status update_mu(const CanonicalProblem& problem, SolverState& state);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_MU_CONTROLLER_HPP
