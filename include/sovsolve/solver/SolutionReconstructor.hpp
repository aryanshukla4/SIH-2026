// Module 17: composition point for undoing everything done to the model
// before the solver saw it -- canonicalization, then scaling -- in reverse
// order.
//
// Currently a thin wrapper over model::recover_solution(), which already
// inverts every canonicalization TransformRecord (RemoveFixedVariable,
// RemoveEmptyRow, BoundedSlack, row negation, ...) and performs the single
// sign-normalization at the pipeline's exit point (Canonical.hpp). It does
// NOT yet invert RowScaling/ColumnScaling records, because Scaler (Module 5)
// does not push any yet -- see Scaler.hpp. Once it does, this function is
// where that inversion is added, rather than model::recover_solution, so the
// ingestion layer's own recovery stays independent of the solver-side stack.

#ifndef SOVSOLVE_SOLVER_SOLUTION_RECONSTRUCTOR_HPP
#define SOVSOLVE_SOLVER_SOLUTION_RECONSTRUCTOR_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/model/Transform.hpp"

namespace sovsolve::solver {

using core::Expected;
using model::CanonicalProblem;
using model::Problem;
using model::Solution;
using model::TransformStack;

[[nodiscard]] Expected<Solution> reconstruct_solution(
    const Problem& original, const CanonicalProblem& canonical,
    const TransformStack& transforms, const Solution& canonical_solution);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_SOLUTION_RECONSTRUCTOR_HPP
