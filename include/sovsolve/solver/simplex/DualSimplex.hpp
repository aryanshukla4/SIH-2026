// Module 23: the bounded-variable dual simplex.
//
// Derived from the definition of the method (Lemke 1954; the bounded-variable
// ratio test, bound flipping and dual pricing as stated in the published
// literature), not ported from any implementation -- see
// docs/HIGHS-COMPARISON.md for why that constraint exists.
//
// --------------------------------------------------------------------------
// Why the signs work out, written once so no call site has to re-derive them
// --------------------------------------------------------------------------
//
// Let slot `r` be the leaving position, holding basic variable `p`, and let
//
//     alpha_j = (B^-1 Ahat)_{r,j} = rho' Ahat_j,   rho = B^-T e_r
//
// be the pivot row. Moving nonbasic `j` by `t` moves `x_B[r]` by `-alpha_j t`.
// There are two infeasibility cases and they are mirror images, so define
//
//     sigma = +1 when x_B[r] > u_p   (must DECREASE)
//     sigma = -1 when x_B[r] < l_p   (must INCREASE)
//     arow_j = sigma * alpha_j
//
// and both collapse to one rule:
//
//     j at lower is eligible iff arow_j > 0
//     j at upper is eligible iff arow_j < 0
//     ratio_j   = d_j / arow_j                    (>= 0 for every eligible j)
//     theta_d   = min ratio over eligible j       (the dual step)
//     t         = delta / arow_q                  (the primal step)
//
// where `delta > 0` is the bound violation. `ratio_j >= 0` because dual
// feasibility means `d_j >= 0` at lower (where `arow_j > 0`) and `d_j <= 0` at
// upper (where `arow_j < 0`) -- the two signs cancel. Dual values move as
// `y += theta_d * sigma * rho`, which is the same statement as
// `d_j -= theta_d * arow_j`.
//
// No eligible `j` at all means the dual objective improves without limit,
// which is a PROOF that the primal is infeasible -- not a stall, not a
// tolerance artifact. That verdict is one this codebase could not previously
// produce from the solver at all (only the canonicalizer could), and it is why
// `SolverStatus::Infeasible` appears here.

#ifndef SOVSOLVE_SOLVER_SIMPLEX_DUAL_SIMPLEX_HPP
#define SOVSOLVE_SOLVER_SIMPLEX_DUAL_SIMPLEX_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include <vector>

#include "sovsolve/solver/simplex/Basis.hpp"
#include "sovsolve/solver/simplex/SimplexResult.hpp"

namespace sovsolve::solver::simplex {

using core::Real;
using core::SolverStatus;

/// Solve `problem` with the dual simplex.
///
/// `warm_start`, when given, must index the same augmented space as `problem`
/// (`n + m` statuses, `m` slots); it is used as the starting basis instead of
/// the all-logical one. A warm start that fails to factorize is not an error:
/// the solve falls back to the logical basis and says so through
/// `refactorizations`, because a caller handing over a neighbouring problem's
/// basis cannot know in advance that a bound change left it singular.
///
/// Returns an `Error` only for a malformed request. Every genuine outcome --
/// optimal, infeasible, unbounded, limit reached -- is a `SolverStatus` on a
/// successful return, because "this model is infeasible" is an answer, not a
/// failure to produce one.
///
/// `costs`, when given, replaces `problem.c` for the whole run -- the cost
/// perturbation of SolveSimplex.cpp. The reported `objective` is then the
/// PERTURBED objective; removing the perturbation is the caller's job.
[[nodiscard]] core::Expected<SimplexResult> solve_dual_simplex(
    const model::CanonicalProblem& problem, const model::Options& options,
    const Basis* warm_start = nullptr, const std::vector<Real>* costs = nullptr);

}  // namespace sovsolve::solver::simplex

#endif  // SOVSOLVE_SOLVER_SIMPLEX_DUAL_SIMPLEX_HPP
