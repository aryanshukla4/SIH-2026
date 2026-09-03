#include "sovsolve/solver/PredictorCorrector.hpp"

namespace sovsolve::solver {

Status run_iteration(const CanonicalProblem& /*problem*/, const Options& /*options*/,
                      RegularizationController& /*regularization*/,
                      SolverState& /*state*/, IterationRecord& /*record*/) {
  return core::make_error(
      core::ErrorCode::NotImplemented,
      "run_iteration: predictor-corrector orchestration not yet wired to "
      "the GPU KKT/linear-solver/step-length modules");
}

}  // namespace sovsolve::solver
