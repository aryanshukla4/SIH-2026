// Module 8: orchestrates one Mehrotra predictor-corrector iteration. Pure
// control flow -- no numerical work of its own. Calls into Module 9 (KKT
// builder), Module 12 (linear solver), Module 13 (Newton recovery) and
// Module 14 (step length), reusing the affine-step factorization for the
// corrector solve where the reduction allows it.

#ifndef SOVSOLVE_SOLVER_PREDICTOR_CORRECTOR_HPP
#define SOVSOLVE_SOLVER_PREDICTOR_CORRECTOR_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/Diagnostics.hpp"
#include "sovsolve/solver/Regularization.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver {

using core::Status;
using model::CanonicalProblem;
using model::Options;

/// Runs one predictor-corrector iteration in place on `state`, recording the
/// result into `record`.
///
/// STUB: does not yet call into the Module 9/12/13/14 GPU modules -- that
/// wiring, and the affine/corrector solves themselves, are the next pass
/// once the GPU kernels exist. This pass only establishes the call boundary.
[[nodiscard]] Status run_iteration(const CanonicalProblem& problem,
                                    const Options& options,
                                    RegularizationController& regularization,
                                    SolverState& state, IterationRecord& record);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_PREDICTOR_CORRECTOR_HPP
