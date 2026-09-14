// Measured quality metrics for a canonical-space solution.
//
// Extracted from simplex/SimplexSolution.cpp when Module 24 (PDLP) needed the
// identical computation. Two engines reporting "primal infeasibility" must
// mean the same thing by it, or the corpus tables comparing them are
// meaningless -- and the way to guarantee that is one implementation, not two
// that agree today.
//
// Everything here is MEASURED against `problem`, never asserted. A simplex at
// a vertex should show rounding-level residuals and a first-order method
// should not; writing zeros because the algorithm "guarantees" them is how a
// broken engine reports perfect quality.

#ifndef SOVSOLVE_SOLVER_SOLUTION_QUALITY_HPP
#define SOVSOLVE_SOLVER_SOLUTION_QUALITY_HPP

#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Solution.hpp"

namespace sovsolve::solver {

/// Fills `solution.quality` from `solution`'s own `x`, `s`, `y`, `z`, `v`.
///
/// Expects all five to be populated and correctly sized for `problem`:
/// `x`, `z`, `v` of length `num_cols()`, `y` of `num_rows()`, and `s` of
/// `num_inequality_rows()` -- element `k` belonging to canonical row
/// `num_equality + k`.
void compute_solution_quality(const model::CanonicalProblem& problem,
                              model::Solution& solution);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_SOLUTION_QUALITY_HPP
