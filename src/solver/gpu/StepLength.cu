#include "sovsolve/solver/gpu/StepLength.hpp"

namespace sovsolve::solver::gpu {

Status compute_step_lengths(const CanonicalProblem& /*problem*/,
                             const SolverState& /*state*/, Real /*eta*/,
                             Real& alpha_primal, Real& alpha_dual) {
  alpha_primal = 0.0;
  alpha_dual = 0.0;
  return core::make_error(core::ErrorCode::NotImplemented,
                           "compute_step_lengths: ratio test not yet written");
}

}  // namespace sovsolve::solver::gpu
