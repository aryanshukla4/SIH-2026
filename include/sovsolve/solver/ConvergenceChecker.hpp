// Module 18: relative-infinity-norm convergence + stall detection.
//
// Every comparison is explicitly NaN-guarded: `NaN < tol` is false in
// IEEE754, so a poisoned residual would otherwise silently run to the
// iteration limit and report MaxIterations on what may be a solvable
// problem. A non-finite residual must produce NumericalError directly
// instead (docs/spec/module.txt Module 18).
//
// INFEASIBLE is never returned from here -- only from a definite verdict in
// the Canonicalizer (an inconsistent empty row or a crossed bound pair).
// Stagnation is evidence a run stopped improving, not evidence about the
// model.
//
// FORMULATION.md 9:
//
//     ||rp||inf / (1 + ||b||inf)    <  tol_primal
//     ||rd||inf / (1 + ||c||inf)    <  tol_dual
//     |c'x - b'y| / (1 + |c'x|)     <  tol_gap   (OR |c'x - b'y| < tol_abs_gap --
//                                                 see Options.hpp's absolute_gap)
//
// `check()` returns NotConverged for BOTH "still running" and "gave up,
// stalled" -- FORMULATION.md 11 documents NOT_CONVERGED as covering both
// cases, and there is no fourth status for it. A caller distinguishes them
// with `is_stalled()`, checked only once `check()` itself has already
// returned NotConverged: `check() != NotConverged` means stop with that
// terminal status; `check() == NotConverged && is_stalled()` means stop
// anyway, reporting NotConverged; otherwise take another step.

#ifndef SOVSOLVE_SOLVER_CONVERGENCE_CHECKER_HPP
#define SOVSOLVE_SOLVER_CONVERGENCE_CHECKER_HPP

#include <cstddef>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/Residuals.hpp"

namespace sovsolve::solver {

using core::Real;
using core::SolverStatus;
using model::CanonicalProblem;
using model::Options;

/// Tracks stall progress across iterations; construct one per solve.
class ConvergenceChecker {
 public:
  ConvergenceChecker(const Options& options, const CanonicalProblem& problem);

  /// Returns SolverStatus::NotConverged while the solve should continue
  /// (see the file comment for what "continue" means alongside
  /// `is_stalled()`), or a terminal status (Optimal, NumericalError,
  /// MaxIterations) when it should stop outright.
  [[nodiscard]] SolverStatus check(const Residuals& residuals, Real objective,
                                    Real dual_objective, std::size_t iteration);

  /// True once `stall_iterations` consecutive checks produced no
  /// improvement in the worse of primal_rel/dual_rel (NOT gap_rel -- the
  /// duality gap is not a reliable monotone progress signal before y/z/v are
  /// near dual-feasible; see ConvergenceChecker.cpp's doc comment on
  /// `combined`). Only meaningful to consult after `check()` has returned
  /// NotConverged.
  [[nodiscard]] bool is_stalled() const noexcept {
    return stall_count_ >= options_.limits.stall_iterations;
  }

  /// The same normalizations `check()` uses internally, exposed so a caller
  /// building a `Solution::quality` from the same residuals does not
  /// recompute ||b||_inf / ||c||_inf itself.
  [[nodiscard]] Real primal_residual_relative(Real rp_inf) const noexcept {
    return rp_inf / (1.0 + b_inf_);
  }
  [[nodiscard]] Real dual_residual_relative(Real rd_inf) const noexcept {
    return rd_inf / (1.0 + c_inf_);
  }
  [[nodiscard]] static Real gap_relative(Real objective, Real dual_objective) noexcept;

 private:
  Options options_;
  Real b_inf_;
  Real c_inf_;
  Real best_combined_ = 0.0;
  bool has_best_ = false;
  std::size_t stall_count_ = 0;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_CONVERGENCE_CHECKER_HPP
