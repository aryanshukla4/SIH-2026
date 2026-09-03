// Module 12. LP -> SPD/Cholesky-type path; general QP ->
// symmetric-indefinite LDL^T path, chosen from `system.descriptor.type`.
// CUDA numerical libraries (cuSOLVER, cuDSS) are used as PRIMITIVES here --
// the optimization algorithm stays in the solver modules, per the spec's GPU
// boundary text.
//
// Refinement is against the UNREGULARIZED residual (module.txt Module 12,
// FORMULATION.md section 10.3) -- the regularized system is a nearby
// problem, not the one actually being solved.

#ifndef SOVSOLVE_SOLVER_GPU_LINEAR_SOLVER_HPP
#define SOVSOLVE_SOLVER_GPU_LINEAR_SOLVER_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/solver/KktSystem.hpp"
#include "sovsolve/solver/gpu/Ordering.hpp"

namespace sovsolve::solver::gpu {

using core::Expected;
using core::RealVector;
using core::Status;

struct LinearSolveResult {
  RealVector solution;
  std::size_t refinement_passes = 0;
};

/// STUB: no CPU reference path -- see the plan's recorded spec deviation
/// (README.md "Solver core" section). Signature only; always fails.
[[nodiscard]] Expected<LinearSolveResult> solve(const KktSystem& system,
                                                  const SymbolicFactorization& symbolic,
                                                  int max_refinement_steps);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_LINEAR_SOLVER_HPP
