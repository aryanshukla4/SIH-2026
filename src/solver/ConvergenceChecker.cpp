#include "sovsolve/solver/ConvergenceChecker.hpp"

namespace sovsolve::solver {

SolverStatus ConvergenceChecker::check(const Residuals& /*residuals*/,
                                        Real /*objective*/, Real /*dual_objective*/,
                                        std::size_t iteration) {
  if (iteration >= options_.limits.max_iterations) {
    return SolverStatus::MaxIterations;
  }
  return SolverStatus::NotConverged;
}

}  // namespace sovsolve::solver
