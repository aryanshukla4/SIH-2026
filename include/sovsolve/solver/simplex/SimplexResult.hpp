// Module 23: what either simplex returns.
//
// One type for both algorithms, because the primal and the dual simplex are
// two iterations over the SAME object -- a basis, the primal values it implies,
// and the duals it implies. They differ in which invariant they hold onto while
// moving (the dual keeps dual feasibility and chases primal feasibility; the
// primal does the reverse) and therefore in which verdicts they can prove. They
// do not differ in what an answer looks like.
//
// Everything here is in CANONICAL space and augmented (w) indexing.
// `SimplexSolution.hpp` maps it to a `model::Solution`, and the existing
// `reconstruct_solution()` maps that back to the original model.

#ifndef SOVSOLVE_SOLVER_SIMPLEX_SIMPLEX_RESULT_HPP
#define SOVSOLVE_SOLVER_SIMPLEX_SIMPLEX_RESULT_HPP

#include <cstddef>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/solver/simplex/Basis.hpp"

namespace sovsolve::solver::simplex {

using core::Real;
using core::SolverStatus;

struct SimplexResult {
  SolverStatus status = SolverStatus::NotConverged;

  /// The final basis, for warm-starting a neighbouring problem -- or for
  /// handing this solve's endpoint to the other algorithm, which is how the
  /// composite path works.
  Basis basis;

  /// Primal values of every augmented variable, length n + m.
  std::vector<Real> x;
  /// Row duals, length m, in the sign convention of FORMULATION.md section 4.
  std::vector<Real> y;
  /// Reduced costs `d = c - Ahat' y`, length n + m.
  std::vector<Real> reduced_cost;

  /// Canonical (minimization) objective at `x`.
  Real objective = 0.0;

  std::size_t iterations = 0;
  std::size_t refactorizations = 0;

  /// Basis columns replaced because the basis turned out to be singular
  /// (LuFactor.hpp's `factorize_repairing`). Reported rather than swallowed: a
  /// nonzero count means the basis handed to, or built by, the iteration was
  /// not the one it thought it had, even though the repair is exact.
  std::size_t basis_repairs = 0;

  /// Dual simplex: pivots avoided because the long-step ratio test chose a
  /// bound flip instead of an entering column. Primal simplex: steps that
  /// moved the entering column across its own range without a basis change.
  /// Zero on a boxed model means the mechanism never fired.
  std::size_t bound_flips = 0;

  /// Primal simplex only: pivots spent minimizing the sum of infeasibilities
  /// before a feasible point existed. Zero when the starting basis was already
  /// primal feasible -- which is exactly the case when the primal is running as
  /// the dual's cleanup phase, and is worth being able to confirm.
  std::size_t phase1_iterations = 0;
};

}  // namespace sovsolve::solver::simplex

#endif  // SOVSOLVE_SOLVER_SIMPLEX_SIMPLEX_RESULT_HPP
