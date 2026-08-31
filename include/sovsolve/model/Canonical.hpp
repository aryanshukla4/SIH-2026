// Canonicalization: faithful model -> solver working form.
//
// Layer two of the two-layer design. Takes a `Problem` as parsed and produces
//
//     minimize    1/2 x'Qx + c'x
//     subject to  A x + s = b
//                 x >= 0
//                 s >= 0
//
// together with the `TransformStack` needed to map a solution back.
//
// This module does not exist in the handoff docs. Its absence is why they
// could assume `Ax <= b, x >= 0` -- the transformation from what files contain
// to what the solver wants was simply never assigned to anyone.
//
// It composes with presolve and scaling: all three push onto the same stack,
// in application order, and recovery replays the whole thing in reverse.

#ifndef SOVSOLVE_MODEL_CANONICAL_HPP
#define SOVSOLVE_MODEL_CANONICAL_HPP

#include <cstddef>

#include "sovsolve/core/SparseMatrix.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/model/Transform.hpp"

namespace sovsolve::model {

/// The solver's working model. Every variable is non-negative, every
/// constraint is an equality with an explicit non-negative slack.
struct CanonicalProblem {
  RealVector c;                    ///< length n_canonical
  core::SparseMatrixPair<> A;      ///< m_canonical x n_canonical
  core::SparseMatrixPair<> Q;      ///< empty for LP
  RealVector b;                    ///< length m_canonical

  /// Constant folded out of the objective by shifts and the sense flip.
  /// Added back during recovery.
  Real obj_offset = 0.0;

  /// True when the original model was a maximization, so the objective was
  /// negated. Recorded on the stack too; duplicated here for convenience.
  bool objective_negated = false;

  [[nodiscard]] std::size_t num_rows() const noexcept { return A.rows(); }
  [[nodiscard]] std::size_t num_cols() const noexcept { return A.cols(); }
};

/// Result of canonicalizing: the working model plus its inverse map.
struct CanonicalResult {
  CanonicalProblem problem;
  TransformStack transforms;
};

// ---------------------------------------------------------------------------
// Transformations applied, and what each records
// ---------------------------------------------------------------------------
//
//   Maximize                 negate c                    NegateObjective
//   x >= l  (l != 0, finite) shift x' = x - l            ShiftVariable
//   x free                   split x = xp - xm           SplitFreeVariable
//   x <= u only              negate x' = -x              NegateVariable
//   x <= u  (with lower)     add row x + t = u           AddBoundRow
//   a'x <= b                 add slack                   AddSlack
//   l <= a'x <= u            bounded slack               BoundedSlack
//   free row (N)             drop                        DropFreeRow
//
// Note on SplitFreeVariable: doubling a column is the standard treatment but
// it worsens conditioning, because xp and xm are perfectly correlated near the
// solution. The record is kept distinguishable so the IPM can later switch to
// native free-variable handling without changing the loader.

/// Canonicalize `problem`. The input is not modified.
[[nodiscard]] core::Expected<CanonicalResult> canonicalize(
    const Problem& problem, const Options& options = {});

/// Map a canonical-space point back to the original model.
///
/// Inverts `transforms` back to front, producing values for the original
/// columns and rows. Handles primal values and duals; see Transform.hpp for
/// why both are designed in from the start.
[[nodiscard]] core::Expected<Solution> recover_solution(
    const Problem& original, const CanonicalProblem& canonical,
    const TransformStack& transforms, const Solution& canonical_solution);

}  // namespace sovsolve::model

#endif  // SOVSOLVE_MODEL_CANONICAL_HPP
