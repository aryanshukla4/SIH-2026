#include "sovsolve/solver/gpu/NewtonRecovery.hpp"

namespace sovsolve::solver::gpu {

Status recover_newton_direction(const CanonicalProblem& /*problem*/,
                                 const KktSystem& /*system*/,
                                 const RealVector& /*linear_solution*/,
                                 SolverState& /*state*/) {
  return core::make_error(core::ErrorCode::NotImplemented,
                           "recover_newton_direction: not yet written");
}

}  // namespace sovsolve::solver::gpu
