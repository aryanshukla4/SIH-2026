// Crossover: from a point that is optimal to a tolerance -- what cuPDLPx and
// the interior point return -- to a simplex basis at a vertex.
//
// A tolerance-optimal point is not a vertex: many variables sit strictly
// between their bounds, and the objective is right only to the engine's
// stopping tolerance. The classical way back to a basis has three steps
// (Megiddo, "On finding primal- and dual-optimal bases", ORSA J. Computing
// 3(1), 1991; Bixby & Saltzman, "Recovering an optimal LP basis from an
// interior point solution", Operations Research Letters 15(4), 1994):
//
//   1. CRASH. Guess a basis from the point. Variables are ranked by the
//      Andersen-Ye indicator t_j = g_j / (g_j + d_j) -- distance g_j from
//      the nearer bound against that bound's dual d_j (Andersen & Ye,
//      "Combining interior-point and pivoting algorithms for linear
//      programming", Management Science 42(12), 1996) -- and the `m` largest
//      become basic. The LU factorization's repair makes the guess
//      nonsingular whatever it is.
//
//   2. PRIMAL PUSH. The point itself is kept, not snapped to bounds -- that
//      would break A x = b and cost a phase 1. Every nonbasic variable still
//      strictly between its bounds (a SUPERBASIC) is moved, along the
//      direction that keeps A x = b and does not worsen the objective, until
//      it reaches its own bound or a basic variable reaches one first; in the
//      second case the two change places, one product-form LU update. Each
//      push leaves one fewer superbasic and keeps the point primal feasible,
//      so the end is a primal-feasible basic solution: a vertex.
//
//   3. DUAL PUSH. The mirror image, on the dual side. A basic variable whose
//      reduced cost in the point's duals is clearly nonzero belongs on a
//      bound. Along the tableau row of its slot the duals are moved until
//      either its reduced cost reaches zero or a nonbasic one would change
//      sign; in the second case the two change places. The variable is
//      already on the bound it leaves to, so the pivot is primal-degenerate
//      and the point does not move.
//
//   4. CLEANUP. The basis is now primal feasible and close to dual feasible;
//      the dual simplex, warm-started from it, fixes what is left. (Measured:
//      the primal simplex is the worse finisher -- it stalls in Bland's rule
//      on degenerate models, e.g. Netlib truss, which it cannot solve cold
//      either -- while the dual finishes truss in 688 iterations.)
//
// This header provides steps 1 to 3; the caller runs step 4 with the
// existing simplex (ConcurrentSolve.cpp).
//
// Written from the papers' descriptions. Nothing here is taken from another
// solver's source -- see docs/HIGHS-COMPARISON.md.

#ifndef SOVSOLVE_SOLVER_CROSSOVER_HPP
#define SOVSOLVE_SOLVER_CROSSOVER_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/solver/simplex/Basis.hpp"

namespace sovsolve::solver {

struct CrossoverStats {
  /// Nonbasic variables strictly between their bounds after the crash.
  std::size_t superbasic = 0;
  /// Superbasics moved onto one of their own bounds.
  std::size_t pushed_to_bound = 0;
  /// Superbasics that entered the basis, pushing a basic variable out.
  std::size_t pivots = 0;
  /// Columns the LU repair replaced, over the crash and every refactorization.
  std::size_t repairs = 0;
  /// Refactorizations after the first.
  std::size_t refactorizations = 0;
  /// Basic variables with a clearly nonzero reduced cost after the primal
  /// push, and how many of them the dual push moved out of the basis.
  std::size_t dual_superbasic = 0;
  std::size_t dual_pivots = 0;
  /// Wall time of each step: crash with its factorization, primal push, dual push.
  double crash_seconds = 0.0;
  double primal_push_seconds = 0.0;
  double dual_push_seconds = 0.0;
};

/// Steps 1 to 3 above: a crash basis read off `point` (canonical space,
/// with `x`, `y`, `z`, `v` filled in, as every engine's
/// `to_canonical_solution` produces), then the primal and dual pushes. On
/// success every nonbasic variable rests on a bound, so the returned basis
/// names a vertex.
///
/// `time_limit_seconds` bounds the push; past it the call returns an error
/// ("crossover: time limit") and the caller keeps the point it had. Any other
/// error is likewise a reason to keep the point, never a wrong answer.
[[nodiscard]] core::Expected<simplex::Basis> crossover_basis(
    const model::CanonicalProblem& problem, const model::Solution& point,
    core::Real pivot_tolerance, double time_limit_seconds,
    CrossoverStats* stats = nullptr);

/// Step 1 alone: the crash basis, before any repair. Exposed for tests.
[[nodiscard]] simplex::Basis crossover_crash_basis(const model::CanonicalProblem& problem,
                                                   const model::Solution& point);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_CROSSOVER_HPP
