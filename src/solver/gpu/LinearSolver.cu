#include "sovsolve/solver/gpu/LinearSolver.hpp"

namespace sovsolve::solver::gpu {

Expected<LinearSolveResult> solve(const KktSystem& /*system*/,
                                   const SymbolicFactorization& /*symbolic*/,
                                   int /*max_refinement_steps*/) {
  return core::make_error(core::ErrorCode::NotImplemented,
                           "solve: GPU factorization/solve kernels not yet written");
}

}  // namespace sovsolve::solver::gpu
