// Module 4. Presolver -- safe, reversible reductions on the canonical
// problem, run once between canonicalization and scaling
// (gpu/Solve.cu::solve_problem). module.txt section 4.
//
// v1 scope: empty-row removal, empty-column removal (folding an inert
// column into the existing `RemoveFixedVariable` machinery, or detecting a
// provable unbounded direction), and free-column-singleton substitution --
// all iterated to a fixed point, since eliminating one thing can create
// another ("Re-check empty rows after every reduction pass. This module
// GENERATES them," module.txt section 4). `analysis::MatrixAnalysis`
// already detects everything this consumes (`empty_rows`, `empty_columns`,
// `free_column_singletons()`) -- this module is what ACTS on that, which
// nothing did before it existed.
//
// Deliberately NOT in this pass: singleton-row bound-tightening and
// duplicate-row/column merging. Both are real `module.txt` §4 items, but
// their dual recovery is materially harder than free-column-singleton
// substitution -- tightening a bound changes which of a column's `z`/`v`
// is active in the ORIGINAL space vs. the reduced problem's space, which
// silently corrupts a reported dual if done wrong rather than crashing.
// A free variable carries no `z`/`v` at all, so it has no such ambiguity.

#ifndef SOVSOLVE_SOLVER_PRESOLVER_HPP
#define SOVSOLVE_SOLVER_PRESOLVER_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Transform.hpp"

namespace sovsolve::solver {

using core::Status;
using model::CanonicalProblem;
using model::Options;
using model::TransformStack;

/// Reduces `problem` in place, pushing reversible transforms onto
/// `transforms` -- the SAME stack `canonicalize()` already started (see
/// `Solve.cu`'s call order: canonicalize -> presolve -> scale -> initialize).
/// A no-op when `options.presolve.enabled` is false (module.txt: the
/// Presolver "must not depend on" the canonicalizer's startability contract,
/// and callers may switch it off).
///
/// **Free-column-singleton substitution.** A free variable `x_j` appearing
/// in exactly one row `i` is that row's definition:
///
///     a_ij*x_j + sum_{k!=j} a_ik*x_k = b_i
///     => x_j = (b_i - sum_{k!=j} a_ik*x_k) / a_ij
///
/// so row `i` and column `j` are both removed -- the row is now implied by
/// the columns that remain, not an independent constraint. Dual recovery
/// (new `RemoveFreeSingleton` branch in
/// `Canonicalizer.cpp::recover_solution`): a free variable has `z_j = v_j =
/// 0` always, so stationarity (`c_j + (Qx)_j - (A_j)'y = 0`) determines the
/// dropped row's dual directly, since `A_j` has exactly one nonzero:
/// `y_i = (c_j + (Qx)_j) / a_ij`.
///
/// Substituting `x_j` into the OBJECTIVE is not free, though: `c_j*x_j`
/// becomes `(c_j/a_ij)*(b_i - sum_{k!=j} a_ik*x_k)`, so every other column
/// `k` in row `i` needs `c_k -= (c_j/a_ij)*a_ik`, plus a constant
/// `(c_j/a_ij)*b_i` folded into `obj_offset` -- done at presolve time
/// (`Presolver.cpp`), not deferred to recovery, because the REDUCED
/// problem's own IPM loop optimizes this objective directly; getting it
/// wrong would silently change what the solver actually minimizes, not just
/// the final reported value.
///
/// **Empty column.** `c_j == 0` and absent from `Q`: the column is inert,
/// fixed at whichever finite bound exists (0 if free) and folded into the
/// EXISTING `RemoveFixedVariable` transform (its recovery math already
/// handles "column removed, value is a known constant" correctly -- an
/// empty column fixed at a bound is that case exactly, no new recovery
/// needed). `c_j != 0` with the bound in the improving direction infinite:
/// a PROVABLE unbounded certificate (moving this one variable improves the
/// objective without limit, no other constraint touched) -- returned as
/// `core::ErrorCode::Unbounded`, the same "verdict, not a malfunction"
/// pattern `canonicalize()` already uses for `PrimalInfeasible`.
///
/// **Empty row.** `RemoveEmptyRow` if the row's own bound is satisfied by zero
/// activity -- exact zero for an equality row, `b_i >= 0` for an inequality
/// row's one-sided canonical form (a POSITIVE leftover RHS there is unused
/// slack, not a violation; only equality rows require exact zero); returns
/// `core::ErrorCode::PrimalInfeasible` otherwise -- never regularized away
/// (module.txt is explicit: doing so turns a provable infeasibility into an
/// unexplained NUMERICAL_FAILURE downstream).
///
/// **Singleton-row bound tightening / forcing.** A singleton row `i` (one
/// entry `a_ij`) implies a bound on `x_j` from the row's own bound. Written
/// into `x_j`'s bounds UNCONDITIONALLY -- safe with no new recovery logic at
/// all, since the row stays exactly as it was (it cannot exclude anything
/// the row itself did not already exclude). When the implication collapses
/// the bound to an exact point, that is a genuine fixed variable, removed by
/// reusing `RemoveFixedVariable` exactly like the empty-column case above;
/// otherwise the bound is simply tighter and the row is untouched.
///
/// **Duplicate-column merging.** Two columns `j`/`k` with an identical `A`
/// pattern (same rows, same values -- `analysis::MatrixAnalysis`'s
/// `duplicate_column_candidates` are hash collisions, confirmed here by a
/// direct comparison before anything acts on them) AND identical cost
/// `c_j == c_k` are interchangeable: the objective and every constraint see
/// only `x_j + x_k`, never the two separately, so one variable can stand in
/// for both. The survivor (`j`) keeps its own column, widened to the
/// Minkowski sum of the pair's bounds (`is_finite_bound()`-guarded, so an
/// infinite side stays infinite rather than corrupting arithmetic on the
/// `INF` sentinel); the other (`k`) is dropped. `MergeDuplicateColumn`
/// records `primary = k`, `secondary = j`. Unlike singleton-row tightening,
/// this DOES need new recovery math (`Canonicalizer.cpp::recover_solution`):
/// the reduced problem only ever solves for the combined value, so splitting
/// it back into `x_j` and `x_k` -- and their two separate reduced costs, off
/// one shared stationarity value, since `c_j == c_k` and `A_j == A_k` mean
/// the value itself doesn't distinguish them -- happens entirely on the way
/// out, not here.
///
/// **General singleton-column elimination.** A column `j` that is a
/// singleton (one entry `a_ij`, row `i`) but NOT free -- generalizing
/// free-column-singleton substitution to a column with a real bound, at the
/// cost of restricting it to an INEQUALITY row `i` (canonical one-sided
/// `a_ij*x_j + R + s_i = b_i, s_i >= 0`; the equality-row case would turn
/// row `i` into a derived inequality rather than remove anything, a smaller
/// win left for later). Three cases, by how `c_j` relates to `a_ij`'s sign:
///
///   * `c_j == 0`: the objective is indifferent to `x_j`, so pick whichever
///     bound MINIMIZES `a_ij*x_j` -- maximizing row `i`'s remaining slack for
///     everything else in it (`a_ij>0` -> `l_j`; `a_ij<0` -> `u_j`) -- and
///     skip the column, unresolved, if that bound is infinite (a zero-cost
///     column cannot be unbounded, but this rule still needs a real number
///     to fold).
///   * `c_j != 0`, sign-MATCHED with `a_ij` (`c_j>0,a_ij>0` or
///     `c_j<0,a_ij<0`): the SAME bound is now also the cost-improving one,
///     so it can be fixed there UNCONDITIONALLY -- a pure win, since moving
///     that way never costs row `i` any feasibility either. An infinite
///     bound here IS a second, genuine `Unbounded` certificate, exactly like
///     the empty-column case, just sourced from a row-bounded rather than
///     totally free column.
///   * `c_j != 0`, sign-MISMATCHED: the cost-improving direction for `x_j`
///     actively consumes row `i`'s slack -- genuinely coupled to the rest of
///     the problem, and explicitly left untouched.
///
/// Both resolvable cases fold `b_i -= a_ij*x_j*` (row `i` survives,
/// transformed, not removed -- the SAME fold Phase-1's fixed-column case
/// uses, just against the one row a singleton column touches instead of
/// every row a singleton ROW's fixed column touches) and reuse
/// `RemoveFixedVariable` exactly like every other fold-to-a-constant case
/// above: `x_j`'s value is picked and baked into `b_i` once, at presolve
/// time, so it is a known constant from that point on -- not something
/// recovery re-derives from the row's other columns, and needs no new
/// recovery math at all.
[[nodiscard]] Status presolve(CanonicalProblem& problem, const Options& options,
                              TransformStack& transforms);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_PRESOLVER_HPP
