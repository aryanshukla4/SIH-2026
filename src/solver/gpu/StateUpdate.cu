#include "sovsolve/solver/gpu/StateUpdate.hpp"

namespace sovsolve::solver::gpu {

Status apply_step(SolverState& /*state*/) {
  return core::make_error(core::ErrorCode::NotImplemented, "apply_step: not yet written");
}

}  // namespace sovsolve::solver::gpu
