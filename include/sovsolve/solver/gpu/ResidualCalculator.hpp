// Module 7. SpMV + vector ops -- GPU-resident per the spec's GPU-boundary
// text (architecture.txt, "## GPU boundary").

#ifndef SOVSOLVE_SOLVER_GPU_RESIDUAL_CALCULATOR_HPP
#define SOVSOLVE_SOLVER_GPU_RESIDUAL_CALCULATOR_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/Residuals.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Real;
using core::Status;
using model::CanonicalProblem;

/// Computes all six residuals (see Residuals.hpp) against `state` at barrier
/// parameter `mu`.
///
/// Currently executes on the HOST, not a CUDA kernel, even though this
/// translation unit only builds under SOVSOLVE_ENABLE_CUDA. `SolverState` and
/// `CanonicalProblem` are Host-resident (`core::DefaultAllocator`) -- there is
/// no device-memory-resident copy of either yet, and no transfer plumbing to
/// build one. Writing a cuSPARSE SpMV kernel against host pointers would not
/// be more "real" than this: it would need the same host loop underneath, or
/// silently fault on dereferencing an unmapped device pointer. Real GPU
/// residence is a separate, larger design decision -- how much of the
/// problem stays resident across iterations, what a device-side SolverState
/// looks like -- and is deliberately out of scope here. The math itself is
/// correct and tested now; only its execution location is provisional.
[[nodiscard]] Status compute_residuals(const CanonicalProblem& problem,
                                        const SolverState& state, Real mu,
                                        Residuals& out);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_RESIDUAL_CALCULATOR_HPP
