#include "sovsolve/solver/ConvergenceChecker.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sovsolve::solver {

namespace {

Real inf_norm(const core::RealVector& v) {
  Real result = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) result = std::max(result, std::fabs(v[i]));
  return result;
}

}  // namespace

ConvergenceChecker::ConvergenceChecker(const Options& options, const CanonicalProblem& problem)
    : options_(options), b_inf_(inf_norm(problem.b)), c_inf_(inf_norm(problem.c)) {}

Real ConvergenceChecker::gap_relative(Real objective, Real dual_objective) noexcept {
  return std::fabs(objective - dual_objective) / (1.0 + std::fabs(objective));
}

SolverStatus ConvergenceChecker::check(const Residuals& residuals, Real objective,
                                        Real dual_objective, std::size_t iteration) {
  if (iteration >= options_.limits.max_iterations) {
    return SolverStatus::MaxIterations;
  }

  const Real primal_rel = primal_residual_relative(residuals.rp_inf);
  const Real dual_rel = dual_residual_relative(residuals.rd_inf);
  const Real gap_rel = gap_relative(objective, dual_objective);

  // NaN compares false against every threshold below, so a poisoned solve
  // would otherwise silently run to MaxIterations instead of reporting
  // what actually happened.
  if (!std::isfinite(primal_rel) || !std::isfinite(dual_rel) || !std::isfinite(gap_rel)) {
    return SolverStatus::NumericalError;
  }

  if (primal_rel < options_.tolerances.primal_feasibility &&
      dual_rel < options_.tolerances.dual_feasibility &&
      gap_rel < options_.tolerances.relative_gap) {
    return SolverStatus::Optimal;
  }

  const Real combined = std::max({primal_rel, dual_rel, gap_rel});
  if (!has_best_ || combined < best_combined_) {
    best_combined_ = combined;
    has_best_ = true;
    stall_count_ = 0;
  } else {
    ++stall_count_;
  }

  return SolverStatus::NotConverged;
}

}  // namespace sovsolve::solver
