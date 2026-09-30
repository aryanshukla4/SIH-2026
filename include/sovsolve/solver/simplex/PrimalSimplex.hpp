// Module 23: the bounded-variable primal simplex.
//
// The mirror of DualSimplex.hpp, over the same revised-simplex machinery
// (detail/SimplexEngine.hpp). Where the dual holds dual feasibility and chases
// primal feasibility, this holds PRIMAL feasibility and chases dual -- so the
// two choices swap order: entering column first, then the leaving row.
//
// --------------------------------------------------------------------------
// The iteration
// --------------------------------------------------------------------------
//
// Pick a nonbasic `q` whose reduced cost says moving it improves the
// objective:
//
//     at lower bound, may increase:  improving iff d_q < 0,  direction +1
//     at upper bound, may decrease:  improving iff d_q > 0,  direction -1
//     free at zero, may do either:   improving iff d_q != 0, direction -sign(d_q)
//
// Moving `x_q` by `t >= 0` in `direction` moves the basic variables along
//
//     alpha = B^-1 Ahat_q,     dx_B[r] = -alpha[r] * direction
//
// and the ratio test asks how far `t` can go before something hits a bound.
// Three kinds of thing can stop it, and the third is the one a textbook
// standard-form derivation does not have:
//
//   1. a basic variable reaching a bound       -> ordinary pivot
//   2. the ENTERING variable reaching its OWN opposite bound  -> a bound flip,
//      which changes no basis at all
//   3. (phase 1 only) an INFEASIBLE basic variable reaching the bound it is
//      violating, i.e. becoming feasible -- a breakpoint of the phase-1
//      objective rather than a blockage
//
// Nothing stopping it at all, in phase 2, is an UNBOUNDED RAY: `x_q` grows
// forever, no basic variable ever hits a bound, and the objective falls
// without limit. That is a direct certificate -- the primal simplex produces
// it as an ordinary outcome of its own ratio test, where the dual can only
// infer it from a phase-1 artificial bound that refuses to stop binding.
//
// --------------------------------------------------------------------------
// Phase 1: minimize the sum of infeasibilities
// --------------------------------------------------------------------------
//
// The starting basis is generally primal INfeasible -- every equality row's
// logical is basic at `b_i` while fixed at zero. Phase 1 replaces the
// objective with the total bound violation,
//
//     c1[basic r] = -1 if x_B[r] < l,  +1 if x_B[r] > u,  else 0
//
// whose gradient at the current point is exactly "reduce the violations", and
// runs the same iteration on it. Reaching zero means a feasible point, and
// phase 2 begins from that basis. Running out of improving columns while the
// sum is still positive is a PROOF of primal infeasibility: the minimum total
// violation over the whole polytope is positive.
//
// --------------------------------------------------------------------------
// Degeneracy: EXPAND
// --------------------------------------------------------------------------
//
// On a degenerate vertex the textbook ratio test returns zero-length steps,
// and nothing stops a sequence of bases from repeating. The ratio test here is
// the EXPAND procedure (Gill, Murray, Saunders & Wright, Math. Prog. 45,
// 1989): a working feasibility tolerance grows a little every iteration, so
// every step is positive and the objective strictly falls. A leaving variable
// may stay up to that tolerance past its bound until a periodic reset puts
// every nonbasic back. See PrimalSimplex.cpp for the details and
// docs/TUNABLES.md for the measurements.
//
// Derived from the method's definition, not ported -- see
// docs/HIGHS-COMPARISON.md for why that constraint exists.

#ifndef SOVSOLVE_SOLVER_SIMPLEX_PRIMAL_SIMPLEX_HPP
#define SOVSOLVE_SOLVER_SIMPLEX_PRIMAL_SIMPLEX_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/simplex/Basis.hpp"
#include "sovsolve/solver/simplex/SimplexResult.hpp"

namespace sovsolve::solver::simplex {

/// Solve `problem` with the primal simplex.
///
/// `warm_start` is used as the starting basis when it fits and factorizes. A
/// primal-feasible warm start skips phase 1 entirely, which is exactly what
/// happens when this runs as the dual simplex's cleanup: the dual's endpoint
/// is feasible for its artificially bounded problem, and the true bounds are
/// wider, so it is feasible here too. `SimplexResult::phase1_iterations` is
/// zero in that case and is worth checking rather than assuming.
///
/// Returns an `Error` only for a malformed request; every genuine outcome,
/// including `Infeasible` and `Unbounded`, is a `SolverStatus` on a successful
/// return.
[[nodiscard]] core::Expected<SimplexResult> solve_primal_simplex(
    const model::CanonicalProblem& problem, const model::Options& options,
    const Basis* warm_start = nullptr);

}  // namespace sovsolve::solver::simplex

#endif  // SOVSOLVE_SOLVER_SIMPLEX_PRIMAL_SIMPLEX_HPP
