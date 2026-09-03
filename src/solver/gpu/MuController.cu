#include "sovsolve/solver/gpu/MuController.hpp"

namespace sovsolve::solver::gpu {

Status update_mu(const CanonicalProblem& /*problem*/, SolverState& /*state*/) {
  return core::make_error(core::ErrorCode::NotImplemented, "update_mu: not yet written");
}

}  // namespace sovsolve::solver::gpu
