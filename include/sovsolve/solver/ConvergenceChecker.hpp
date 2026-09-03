// Module 18: relative-infinity-norm convergence + stall detection.
//
// Every comparison is explicitly NaN-guarded: `NaN < tol` is false in
// IEEE754, so a poisoned residual would otherwise silently run to the
// iteration limit and report MaxIterations on what may be a solvable
// problem. A non-finite residual must produce NumericalError directly
// instead (module.txt Module 18).
//
// INFEASIBLE is never returned from here -- only from a definite verdict in
// the Canonicalizer (an inconsistent empty row or a crossed bound pair).
// Stagnation is evidence a run stopped improving, not evidence about the
// model.

#ifndef SOVSOLVE_SOLVER_CONVERGENCE_CHECKER_HPP
#define SOVSOLVE_SOLVER_CONVERGENCE_CHECKER_HPP

#include <cstddef>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/Residuals.hpp"

namespace sovsolve::solver {

using core::Real;
using core::SolverStatus;
using model::Options;

/// Tracks stall progress across iterations; construct one per solve.
class ConvergenceChecker {
 public:
  explicit ConvergenceChecker(const Options& options) : options_(options) {}

  /// Returns SolverStatus::NotConverged while the solve should continue, or
  /// a terminal status (Optimal, NumericalError, MaxIterations,
  /// NotConverged-on-stall) when it should stop.
  ///
  /// STUB: the NaN-guard and the three relative infinity-norm tests
  /// (FORMULATION.md section 9) are not yet implemented; only the iteration
  /// limit is checked, so callers can wire the loop around this before the
  /// real criteria land.
  [[nodiscard]] SolverStatus check(const Residuals& residuals, Real objective,
                                    Real dual_objective, std::size_t iteration);

 private:
  Options options_;
  Real best_objective_ = 0.0;
  std::size_t stall_count_ = 0;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_CONVERGENCE_CHECKER_HPP
