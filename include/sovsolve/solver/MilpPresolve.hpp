// Module 22 sub-component: MILP-specific row tightening.
//
// The existing Presolver (Module 4, Presolver.hpp) operates on
// CanonicalProblem, which drops col_type entirely during canonicalization --
// it was never given a reason to keep it, since Modules 1-21 never needed
// integrality. This reduction genuinely needs to know which columns are
// discrete, so it runs on the raw Problem, once, before Module 22's search
// begins (BranchAndBound.cu) -- a new, clearly scoped piece, not folded into
// Module 4 (docs/spec/module.txt's "no module silently implements another module's
// job").

#ifndef SOVSOLVE_SOLVER_MILP_PRESOLVE_HPP
#define SOVSOLVE_SOLVER_MILP_PRESOLVE_HPP

#include <cstddef>
#include <vector>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Problem.hpp"

namespace sovsolve::solver {

/// One column `eliminate_equality_row_absorbing_singletons` removed from a
/// row, recorded so the CALLER can recover its true value after the whole
/// solve finishes: `col`'s true value is `(b_i - sum_{k != col} a_ik*x_k) /
/// a_ij`, using row `row`'s OTHER (still-present) entries and coefficient
/// `a_ij` against the FINAL solution's `x`. Without this, `col` would be
/// left at whatever arbitrary bound the generic empty-column presolve rule
/// picks for it (correct for the OBJECTIVE, once its cost is folded, but
/// not for reporting a value that actually satisfies the original row).
struct AbsorbingColumnElimination {
  std::size_t row;
  std::size_t col;
  core::Real a_ij;  ///< col's ORIGINAL coefficient in `row`, before removal
  core::Real b_i;   ///< row's ORIGINAL equality right-hand side
};

/// GCD (Chvatal-Gomory rounding) tightening for rows where every nonzero
/// column is discrete (Integer or Binary) with an exact-integer
/// coefficient. Any integer combination `sum a_j*x_j` over such a row is
/// necessarily a multiple of `g = gcd(|a_j|)`, so:
///
///   - a finite row_upper can be rounded DOWN to the nearest multiple of g
///     with no integer point lost -- the LP relaxation shrinks, the integer
///     feasible region does not;
///   - a finite row_lower can be rounded UP, symmetrically;
///   - if that crosses (`row_lower > row_upper`) -- including an equality
///     row whose single bound is not itself a multiple of g -- the model is
///     PROVABLY infeasible: no integer combination can ever land on a
///     non-multiple of g.
///
/// Needs no simplex tableau or LP solve -- read directly off the model's
/// own integer data, which is why it runs once up front rather than waiting
/// for the search to rediscover the same infeasibility one branch at a
/// time. A row with any continuous/semi-continuous column, or a
/// non-integer coefficient (rare but legal in an MPS file), is left
/// untouched -- the "always a multiple of g" property does not hold for it.
///
/// Mutates `problem`'s row bounds in place. Returns `PrimalInfeasible` (via
/// core::make_error) on the provable-infeasibility case above; Ok otherwise.
[[nodiscard]] core::Status tighten_integer_rows(model::Problem& problem);

/// A different, complementary reduction: an EQUALITY row of the form
/// `a_ij*x_j + R = b_i`, where `x_j` is CONTINUOUS (or semi-continuous), a
/// SINGLETON in `A` (this row is its only appearance), and has exactly ONE
/// finite bound -- e.g. a non-negative "deviation" or "slack-like" column a
/// modeler added explicitly, appearing alongside otherwise-integer columns
/// (confirmed on markshare_4_0.mps: every row is exactly this shape, one
/// continuous non-negative column plus ~30 binary columns, and it is
/// specifically what blocked `tighten_integer_rows` and the branch-and-
/// bound cover-cut separation from ever firing there -- both require every
/// column in a row to be discrete).
///
/// Since `x_j` has no OTHER constraint anywhere, it can absorb any value in
/// its own range with no effect elsewhere -- so the row is equivalent to a
/// derived ONE-SIDED inequality purely on `R` (the remaining columns), with
/// `x_j`'s own contribution range folded into the bound. This function
/// removes `x_j`'s single entry from the row and replaces the row's
/// equality bounds with that derived one-sided bound -- turning what
/// canonicalize() will see as an ordinary inequality row over ALL-DISCRETE
/// columns (the case `tighten_integer_rows` and cover-cut separation both
/// need), while `x_j` itself becomes a genuinely empty column with its
/// original cost and bound untouched.
///
/// `x_j` is NOT an independently-chosen variable -- the row forces
/// `x_j = (b_i - R) / a_ij`, so its cost `c_j*x_j` is actually a function of
/// R, not a free term. This function folds that cost into the remaining
/// columns' own costs and an objective constant (`c_j*b_i/a_ij -
/// (c_j/a_ij)*R`, the same substitution `RemoveFreeSingleton`/
/// `RemoveFixedVariable` already apply for their own cases elsewhere), then
/// zeroes `c_j` -- only THEN is `x_j` a genuinely zero-cost empty column,
/// which the EXISTING Presolver (Module 4, Presolver.cpp) is free to fix
/// arbitrarily. Leaving `c_j` un-folded was a real, caught bug: the generic
/// empty-column rule would fix `x_j` at whichever bound minimizes its own
/// isolated cost, completely ignoring what the row actually requires --
/// confirmed on markshare_4_0.mps, which silently reported an unreachable
/// objective of 0.0 against a true optimum of 1.0 before this fold existed.
///
/// Pushes no new transform record and does not itself recover `x_j`'s TRUE
/// value (the empty-column rule still picks an arbitrary point satisfying
/// only `x_j`'s own bounds, not the row) -- every eliminated column is
/// appended to `eliminations` instead, so the CALLER can recompute its real
/// value from the final solution once the whole search finishes (see
/// `AbsorbingColumnElimination`'s doc comment). Without this second step,
/// the reported objective would be right but the reported `x_j` could
/// silently violate the original row.
///
/// Only ever removes ONE absorbing column per row (a row with more than one
/// candidate leaves the extras untouched -- rare, and not the shape this
/// was built for) and only for a row with BOTH bounds finite and equal (a
/// genuine equality); an inequality row with an absorbing column is a
/// separate, rarer case not attempted here. Never fails.
void eliminate_equality_row_absorbing_singletons(
    model::Problem& problem, std::vector<AbsorbingColumnElimination>& eliminations);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_MILP_PRESOLVE_HPP
