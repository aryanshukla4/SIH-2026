// Module 23: pick a simplex algorithm, and let the two cover for each other.
//
// The dual and the primal simplex fail in different places, which is the
// practical reason for having both rather than the better one:
//
//   * The DUAL needs dual feasibility to start. A column whose reduced cost
//     wants it on a side it has no bound for -- and every free column, which
//     has no bound at all -- has to be given a temporary artificial one first.
//     When that box is still binding at the end and no ray proves
//     unboundedness, the dual has no verdict to give: the point it found is
//     optimal for the BOXED problem, which is not the model.
//
//   * The PRIMAL needs primal feasibility to start, and pays for it with a
//     phase 1 over the sum of infeasibilities. But it has no boxes, treats a
//     free column as an ordinary candidate, and gets unboundedness directly
//     out of its own ratio test.
//
// And the handoff between them is free: the dual's endpoint is feasible for
// its boxed problem, and the true bounds are WIDER than the box, so that same
// point is primal feasible for the model. The primal therefore starts in phase
// 2 -- `SimplexResult::phase1_iterations` is zero, which the composite test
// asserts rather than assumes -- and simply finishes.
//
// Measured on Netlib `greenbea`: the dual alone escalates its box five times
// and reports NotConverged.

#ifndef SOVSOLVE_SOLVER_SIMPLEX_SOLVE_SIMPLEX_HPP
#define SOVSOLVE_SOLVER_SIMPLEX_SOLVE_SIMPLEX_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/simplex/Basis.hpp"
#include "sovsolve/solver/simplex/SimplexResult.hpp"

namespace sovsolve::solver::simplex {

/// Solve `problem` with the simplex algorithm `options.simplex.method` names,
/// applying the primal cleanup described above when the dual ends without a
/// verdict and `options.simplex.primal_cleanup` is set.
///
/// This is the entry point callers should use; `solve_dual_simplex` and
/// `solve_primal_simplex` remain available for exercising one algorithm on its
/// own, which is what the tests do when checking that each produces its own
/// verdicts unaided.
[[nodiscard]] core::Expected<SimplexResult> solve_simplex(
    const model::CanonicalProblem& problem, const model::Options& options,
    const Basis* warm_start = nullptr);

}  // namespace sovsolve::solver::simplex

#endif  // SOVSOLVE_SOLVER_SIMPLEX_SOLVE_SIMPLEX_HPP
