// Module 23: pack a dual simplex result into the canonical-space
// `model::Solution` the existing postsolve already consumes.
//
// There is deliberately no new postsolve code anywhere in Module 23.
// `reconstruct_solution()` (SolutionReconstructor.hpp, a thin wrapper over
// `model::recover_solution()`) inverts canonicalization, presolve and scaling
// from values alone -- `x`, `s`, `y`, `z`, `v` -- and never needs a basis. So
// the only thing the simplex has to do differently from the IPM is fill those
// five vectors; everything downstream is shared, already tested, and untouched.
//
// The one thing this file must get right is the sign convention, and it is not
// a choice made here -- it falls out of the formulation (see DualSimplex.hpp):
//
//     d = c - Ahat' y,  with  y = B^-T c_B
//
// An inequality row's logical has zero cost and bounds `[0, +INF)`. Resting at
// its lower bound, dual feasibility requires `d >= 0`, and `d = -y_i`, so
// `y_i <= 0` on every inequality row -- which is exactly FORMULATION.md
// section 4's `-y_I > 0`. Nothing is negated to make that true.

#ifndef SOVSOLVE_SOLVER_SIMPLEX_SIMPLEX_SOLUTION_HPP
#define SOVSOLVE_SOLVER_SIMPLEX_SIMPLEX_SOLUTION_HPP

#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/solver/simplex/SimplexResult.hpp"

namespace sovsolve::solver::simplex {

/// Build the canonical-space `Solution` for `result`, with quality metrics
/// measured against `problem` rather than asserted.
///
/// The metrics are computed the same way the interior-point path computes
/// them, so the two engines' reported numbers mean the same thing and can be
/// compared directly. For a simplex they should all be at rounding level at a
/// vertex; measuring rather than writing zeros is what would catch it if they
/// were not.
[[nodiscard]] model::Solution to_canonical_solution(const model::CanonicalProblem& problem,
                                                    const SimplexResult& result);

}  // namespace sovsolve::solver::simplex

#endif  // SOVSOLVE_SOLVER_SIMPLEX_SIMPLEX_SOLUTION_HPP
