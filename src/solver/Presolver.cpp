#include "sovsolve/solver/Presolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "sovsolve/analysis/MatrixAnalysis.hpp"
#include "sovsolve/core/SparseBuilder.hpp"

namespace sovsolve::solver {

namespace {

using core::ErrorCode;
using core::Index;
using core::INF;
using core::is_finite_bound;
using core::Real;
using model::TransformKind;
using model::TransformRecord;

constexpr Index kNone = TransformRecord::kIndexNone;

/// Original-index for every CURRENT row/column, rebuilt from the transform
/// stack rather than threaded through the worklist loop by hand -- a later
/// KeepColumn/MapRow record for the same original index always overwrites an
/// earlier one in a straight scan, so re-scanning at the start of every pass
/// naturally picks up whatever the PREVIOUS pass's rebuild last recorded.
struct OriginalIndexMaps {
  std::vector<Index> row;  ///< size = problem.num_rows() at scan time
  std::vector<Index> col;  ///< size = problem.num_cols() at scan time
};

OriginalIndexMaps build_original_index_maps(const TransformStack& transforms, std::size_t m,
                                            std::size_t n) {
  OriginalIndexMaps maps;
  maps.row.assign(m, kNone);
  maps.col.assign(n, kNone);
  for (const auto& rec : transforms.records()) {
    if (rec.secondary == kNone) continue;
    const auto idx = static_cast<std::size_t>(rec.secondary);
    if (rec.kind == TransformKind::MapRow && idx < m) {
      maps.row[idx] = rec.primary;
    } else if (rec.kind == TransformKind::KeepColumn && idx < n) {
      maps.col[idx] = rec.primary;
    }
  }
  return maps;
}

/// True if row `i` (current numbering) touches one of the trailing range
/// columns (`[n - num_range, n)`). Free-column-singleton substitution skips
/// such rows: a ranged row's canonical form is `a'x + t = hi` with `t` the
/// range column, and removing the row would drop `t`'s only defining
/// equation while `t`'s bound survives -- silently changing what the model
/// means. No instance in this project's corpus has a ranged row
/// (`Canonical.hpp`), so this guard is defensive, not load-bearing today.
bool row_touches_range_column(const CanonicalProblem& problem, std::size_t row) {
  if (problem.num_range == 0) return false;
  const std::size_t first_range_col = problem.num_cols() - problem.num_range;
  const auto& csr = problem.A.csr;
  for (auto k = csr.slice_begin(row); k < csr.slice_end(row); ++k) {
    if (static_cast<std::size_t>(csr.indices()[k]) >= first_range_col) return true;
  }
  return false;
}

/// Confirms a `duplicate_column_candidates` hash match by direct comparison
/// of columns `j` and `k`'s CSC slices -- same nnz, and the same (row,
/// value) pairs IN STORED ORDER (the hash itself is order-sensitive over
/// that same stored order, so a real match is always order-preserving too;
/// this is not an independent, weaker check). A collision surviving this is
/// astronomically unlikely but not provably impossible, and Presolver.hpp's
/// contract is "candidates, not confirmations" -- so this always runs before
/// anything acts on a pair.
bool duplicate_columns_confirmed(const core::SparseMatrixPair<>& a, std::size_t j,
                                 std::size_t k) {
  const auto& csc = a.csc;
  if (csc.slice_nnz(j) != csc.slice_nnz(k)) return false;
  auto kj = csc.slice_begin(j);
  auto kk = csc.slice_begin(k);
  for (; kj < csc.slice_end(j); ++kj, ++kk) {
    if (csc.indices()[kj] != csc.indices()[kk]) return false;
    if (csc.values()[kj] != csc.values()[kk]) return false;
  }
  return true;
}

/// Rebuilds `problem` keeping only rows/columns where `keep_row`/`keep_col`
/// is true, remapping `A`, `Q`, `b`, `c`, bounds and `num_equality`
/// accordingly, and pushes updated `MapRow`/`KeepColumn` records (keyed by
/// ORIGINAL index, via `maps`) reflecting each survivor's FINAL position --
/// the same "later record for the same original index wins" mechanism
/// `recover_solution` already relies on for canonicalize()'s own records.
///
/// No RHS folding is needed here: every row this module removes is either
/// genuinely empty (no coefficients to fold) or is a free-column-singleton's
/// row, whose OTHER columns' coefficients simply disappear along with the
/// row (nothing else references them there). The one exception is the
/// OBJECTIVE, which the free-singleton caller already folds into `c`/
/// `obj_offset` directly, before calling this function -- see
/// `Presolver.hpp`'s doc comment and the call site above.
Status rebuild(CanonicalProblem& problem, const std::vector<bool>& keep_row,
              const std::vector<bool>& keep_col, const OriginalIndexMaps& maps,
              TransformStack& transforms) {
  const std::size_t old_m = problem.num_rows();
  const std::size_t old_n = problem.num_cols();

  std::vector<Index> new_row_index(old_m, kNone);
  std::vector<Index> new_col_index(old_n, kNone);
  std::size_t new_m = 0;
  std::size_t new_equality = 0;
  for (std::size_t i = 0; i < old_m; ++i) {
    if (!keep_row[i]) continue;
    new_row_index[i] = static_cast<Index>(new_m++);
    if (i < problem.num_equality) ++new_equality;
  }
  std::size_t new_n = 0;
  for (std::size_t j = 0; j < old_n; ++j) {
    if (!keep_col[j]) continue;
    new_col_index[j] = static_cast<Index>(new_n++);
  }

  for (std::size_t i = 0; i < old_m; ++i) {
    if (new_row_index[i] == kNone) continue;
    transforms.push({TransformKind::MapRow, maps.row[i], new_row_index[i], 0.0, 0.0});
  }
  for (std::size_t j = 0; j < old_n; ++j) {
    if (new_col_index[j] == kNone) continue;
    transforms.push({TransformKind::KeepColumn, maps.col[j], new_col_index[j], 0.0, 0.0});
  }

  core::RealVector new_b(new_m, 0.0);
  core::RealVector new_c(new_n, 0.0);
  core::RealVector new_lower(new_n, 0.0);
  core::RealVector new_upper(new_n, 0.0);
  for (std::size_t i = 0; i < old_m; ++i) {
    if (new_row_index[i] != kNone) new_b[static_cast<std::size_t>(new_row_index[i])] = problem.b[i];
  }
  for (std::size_t j = 0; j < old_n; ++j) {
    if (new_col_index[j] == kNone) continue;
    const auto nj = static_cast<std::size_t>(new_col_index[j]);
    new_c[nj] = problem.c[j];
    new_lower[nj] = problem.col_lower[j];
    new_upper[nj] = problem.col_upper[j];
  }

  const auto& old_csr = problem.A.csr;
  const auto enumerate_a = [&](auto&& emit) {
    for (std::size_t i = 0; i < old_m; ++i) {
      if (new_row_index[i] == kNone) continue;
      for (auto k = old_csr.slice_begin(i); k < old_csr.slice_end(i); ++k) {
        const auto j = static_cast<std::size_t>(old_csr.indices()[k]);
        if (new_col_index[j] == kNone) continue;
        emit(new_row_index[i], new_col_index[j], old_csr.values()[k]);
      }
    }
  };
  core::SparseBuilder a_builder(new_m, new_n);
  enumerate_a([&](Index r, Index c, Real) { a_builder.count(r, c); });
  // Cannot actually overflow here: the rebuilt matrix's nnz never exceeds the
  // original's, which the reader already confirmed fits `Index` at load
  // time -- checked anyway, consistent with never trusting a Status unread.
  Status alloc_status = a_builder.allocate();
  if (!alloc_status.ok()) return alloc_status;
  enumerate_a([&](Index r, Index c, Real v) { a_builder.insert(r, c, v); });

  core::SparseMatrixPair<> new_q;
  if (!problem.Q.empty()) {
    const auto& old_q_csr = problem.Q.csr;
    const auto enumerate_q = [&](auto&& emit) {
      for (std::size_t i = 0; i < old_n; ++i) {
        if (new_col_index[i] == kNone) continue;
        for (auto k = old_q_csr.slice_begin(i); k < old_q_csr.slice_end(i); ++k) {
          const auto j = static_cast<std::size_t>(old_q_csr.indices()[k]);
          if (new_col_index[j] == kNone) continue;
          emit(new_col_index[i], new_col_index[j], old_q_csr.values()[k]);
        }
      }
    };
    core::SparseBuilder q_builder(new_n, new_n);
    enumerate_q([&](Index r, Index c, Real) { q_builder.count(r, c); });
    Status q_alloc_status = q_builder.allocate();
    if (!q_alloc_status.ok()) return q_alloc_status;
    enumerate_q([&](Index r, Index c, Real v) { q_builder.insert(r, c, v); });
    new_q = q_builder.finish();
  }

  problem.A = a_builder.finish();
  problem.Q = std::move(new_q);
  problem.b = std::move(new_b);
  problem.c = std::move(new_c);
  problem.col_lower = std::move(new_lower);
  problem.col_upper = std::move(new_upper);
  problem.num_equality = new_equality;
  // num_range is left unchanged: row_touches_range_column() above guarantees
  // every range column's row is skipped, so no range column is ever removed.
  return Status::Ok();
}

}  // namespace

Status presolve(CanonicalProblem& problem, const Options& options, TransformStack& transforms) {
  if (!options.presolve.enabled) return Status::Ok();

  const Real feas_tol = options.tolerances.bound_violation;
  analysis::AnalysisOptions analysis_opts;
  analysis_opts.detect_duplicates = true;  // Phase 2 (duplicate-column merging) needs this

  for (;;) {
    const std::size_t m = problem.num_rows();
    const std::size_t n = problem.num_cols();
    const OriginalIndexMaps maps = build_original_index_maps(transforms, m, n);

    const analysis::MatrixAnalysis mat_analysis = analysis::analyze(problem.A, analysis_opts);

    std::vector<bool> keep_row(m, true);
    std::vector<bool> keep_col(n, true);
    bool changed = false;

    // -- empty rows -----------------------------------------------------
    //
    // An empty row's activity is identically zero, so feasibility depends on
    // whether 0 satisfies the row -- and that check is NOT the same shape for
    // both row kinds. Equality rows have no slack: `0 = b_i` must hold
    // exactly, so any |b_i| > tol is infeasible. Inequality rows are
    // one-sided in canonical form (`0 + s_i = b_i, s_i >= 0`): 0 satisfies
    // them whenever `b_i >= 0` -- a NEGATIVE b_i is infeasible, but a
    // POSITIVE one is just unused slack, not a violation. Using the
    // equality-style |b_i|>tol check for an inequality row would wrongly
    // report PrimalInfeasible on a perfectly feasible row whenever its
    // leftover RHS happened to be a nonzero positive number -- found while
    // deriving Phase 1 (singleton-row tightening), which creates exactly
    // this shape of leftover RHS on inequality rows routinely.
    for (const Index i_idx : mat_analysis.empty_rows) {
      const auto i = static_cast<std::size_t>(i_idx);
      if (!keep_row[i]) continue;
      const bool is_equality = i < problem.num_equality;
      const bool infeasible =
          is_equality ? std::fabs(problem.b[i]) > feas_tol : problem.b[i] < -feas_tol;
      if (infeasible) {
        return core::make_error(ErrorCode::PrimalInfeasible,
                                "presolve: empty row " + std::to_string(i) +
                                    (is_equality
                                         ? " has a right-hand side that excludes zero"
                                         : " has a negative right-hand side (infeasible slack)"));
      }
      keep_row[i] = false;
      changed = true;
      transforms.push({TransformKind::RemoveEmptyRow, maps.row[i], kNone, 0.0, 0.0});
    }

    // -- empty columns ----------------------------------------------------
    const bool has_q = !problem.Q.empty();
    for (const Index j_idx : mat_analysis.empty_columns) {
      const auto j = static_cast<std::size_t>(j_idx);
      if (!keep_col[j]) continue;

      const bool empty_in_q = !has_q || problem.Q.csc.slice_nnz(j) == 0;

      if (problem.c[j] == 0.0 && empty_in_q) {
        // Inert: contributes nothing to any row, Q, or the objective. Fix at
        // any feasible value -- reuse RemoveFixedVariable, whose recovery
        // (Canonicalizer.cpp) already handles "column removed, value is a
        // known constant" correctly; no new recovery math needed.
        Real fixed_at = 0.0;
        if (is_finite_bound(problem.col_lower[j])) {
          fixed_at = problem.col_lower[j];
        } else if (is_finite_bound(problem.col_upper[j])) {
          fixed_at = problem.col_upper[j];
        }
        keep_col[j] = false;
        changed = true;
        transforms.push(
            {TransformKind::RemoveFixedVariable, maps.col[j], kNone, fixed_at, 0.0});
        continue;
      }

      if (problem.c[j] != 0.0 && empty_in_q) {
        // Minimizing c_j*x_j alone (the column touches nothing else): the
        // improving direction is x_j -> -inf if c_j > 0, or x_j -> +inf if
        // c_j < 0. An infinite bound in that direction is a PROVABLE
        // unbounded certificate, not a heuristic -- no other constraint
        // limits this variable at all.
        const bool unbounded = (problem.c[j] > 0.0 && !is_finite_bound(problem.col_lower[j])) ||
                               (problem.c[j] < 0.0 && !is_finite_bound(problem.col_upper[j]));
        if (unbounded) {
          return core::make_error(ErrorCode::Unbounded,
                                  "presolve: empty column " + std::to_string(j) +
                                      " has nonzero cost and no bound in its improving "
                                      "direction");
        }
        // Bounded in the improving direction: fix there. Folds a real
        // constant into the objective (unlike the zero-cost case above),
        // which RemoveFixedVariable's OWN recovery does not do on its
        // own here -- it recovers x_j and duals, not the constant term --
        // so fold it directly, the same way canonicalize() folds a fixed
        // column's contribution into obj_offset.
        const Real fixed_at = problem.c[j] > 0.0 ? problem.col_lower[j] : problem.col_upper[j];
        problem.obj_offset += problem.c[j] * fixed_at;
        keep_col[j] = false;
        changed = true;
        transforms.push(
            {TransformKind::RemoveFixedVariable, maps.col[j], kNone, fixed_at, 0.0});
      }
      // else: nonzero cost but touches Q -- not an "empty column" in the
      // sense this rule targets (Q coupling means it's not actually inert).
    }

    // -- free-column singletons -------------------------------------------
    if (!has_q) {  // FORMULATION.md-style LP-only scope, same reasoning as
                   // the normal-equations reduction: this substitution's
                   // recovery (Canonicalizer.cpp) assumes (Qx)_j == 0.
      const auto singleton_cols =
          analysis::free_column_singletons(problem.A, problem.col_lower.span(),
                                            problem.col_upper.span());
      const auto& csc = problem.A.csc;
      for (const Index j_idx : singleton_cols) {
        const auto j = static_cast<std::size_t>(j_idx);
        if (!keep_col[j] || csc.slice_nnz(j) != 1) continue;
        const auto i = static_cast<std::size_t>(csc.indices()[csc.slice_begin(j)]);
        if (!keep_row[i] || i >= problem.num_equality) continue;  // equality rows only
        if (row_touches_range_column(problem, i)) continue;

        // Substituting x_j = (b_i - sum_{k!=j} a_ik*x_k) / a_ij into the
        // objective changes what it depends on: c_j*x_j becomes
        // (c_j/a_ij)*(b_i - sum_{k!=j} a_ik*x_k), so every OTHER column k in
        // row i needs c_k -= (c_j/a_ij)*a_ik, and a constant (c_j/a_ij)*b_i
        // is added. This is not optional bookkeeping -- skipping it would
        // silently change the objective the reduced problem's IPM loop
        // actually optimizes, not just the recovery step at the end.
        {
          const Real factor = problem.c[j] / csc.values()[csc.slice_begin(j)];
          const auto& csr = problem.A.csr;
          for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
            const auto col = static_cast<std::size_t>(csr.indices()[k]);
            if (col == j) continue;
            problem.c[col] -= factor * csr.values()[k];
          }
          problem.obj_offset += factor * problem.b[i];
        }

        keep_row[i] = false;
        keep_col[j] = false;
        changed = true;
        transforms.push({TransformKind::RemoveFreeSingleton, maps.row[i], maps.col[j], 0.0, 0.0});
      }
    }

    // -- singleton-row bound tightening / forcing --------------------------
    //
    // A singleton row i (one entry a_ij) implies a bound on x_j from the
    // row's own bound: equality gives an exact value; inequality (canonical
    // one-sided a'x + s = b, s>=0, i.e. a'x <= b) gives a one-sided bound
    // whose direction depends on a_ij's sign. Intersecting with x_j's current
    // bounds and writing the result back UNCONDITIONALLY is always safe and
    // needs no transform record at all: the row is left exactly as-is (it
    // still gets a normal dual via the existing MapRow path next pass), and
    // the tightened bound cannot exclude anything the row didn't already
    // exclude -- any x_j outside the implied range was already infeasible
    // for this row. Only when the tightening COLLAPSES to an exact point
    // does anything structural happen: that's a genuine fixed variable,
    // handled by reusing RemoveFixedVariable exactly like the empty-column
    // case above (fold into every OTHER row containing j, fold the
    // objective, drop the column) -- row i needs no special handling itself,
    // since removing its only column empties it, and the NEXT pass's
    // empty-row block (above, now correctly handling both row kinds) finds
    // and removes it then.
    if (!has_q) {
      for (const Index i_idx : mat_analysis.singleton_rows) {
        const auto i = static_cast<std::size_t>(i_idx);
        if (!keep_row[i]) continue;
        const auto& csr = problem.A.csr;
        if (csr.slice_nnz(i) != 1) continue;  // defensive; matches free-singleton's own guard
        const auto k = csr.slice_begin(i);
        const auto j = static_cast<std::size_t>(csr.indices()[k]);
        if (!keep_col[j]) continue;
        const Real a_ij = csr.values()[k];

        Real implied_lower = -INF;
        Real implied_upper = INF;
        if (i < problem.num_equality) {
          const Real value = problem.b[i] / a_ij;
          implied_lower = value;
          implied_upper = value;
        } else if (a_ij > 0.0) {
          implied_upper = problem.b[i] / a_ij;
        } else {
          implied_lower = problem.b[i] / a_ij;  // dividing by a negative flips the direction
        }

        const Real new_lower = std::max(problem.col_lower[j], implied_lower);
        const Real new_upper = std::min(problem.col_upper[j], implied_upper);
        if (new_lower > new_upper + feas_tol) {
          return core::make_error(ErrorCode::PrimalInfeasible,
                                  "presolve: singleton row " + std::to_string(i) +
                                      " implies an infeasible bound on column " +
                                      std::to_string(j));
        }
        problem.col_lower[j] = new_lower;
        problem.col_upper[j] = new_upper;

        if (new_upper - new_lower > feas_tol) continue;  // tightened, not fixed -- done, safely

        // Fold into EVERY row containing j, INCLUDING row i itself -- found
        // via direct tracing (not assumed): row i's own leftover RHS must
        // become `b_i - a_ij*fixed_at`, which is exactly zero when
        // `fixed_at` came from row i's own implied value (the equality case,
        // or an inequality row whose own bound was what triggered the
        // collapse), and a genuine, correctly-signed residual slack
        // otherwise (an inequality row that collapsed because ANOTHER
        // row/bound was the binding one). Skipping row i here (an earlier,
        // wrong version of this code did) leaves row i with its ORIGINAL,
        // un-folded b -- so once column j is removed, row i reads as
        // "empty with the original nonzero RHS" and gets wrongly reported
        // PrimalInfeasible on the very next pass, even though row i is
        // actually satisfied exactly by the fix.
        const Real fixed_at = new_lower;
        const auto& csc = problem.A.csc;
        for (auto k2 = csc.slice_begin(j); k2 < csc.slice_end(j); ++k2) {
          const auto row = static_cast<std::size_t>(csc.indices()[k2]);
          problem.b[row] -= csc.values()[k2] * fixed_at;
        }
        problem.obj_offset += problem.c[j] * fixed_at;
        keep_col[j] = false;
        changed = true;
        transforms.push({TransformKind::RemoveFixedVariable, maps.col[j], kNone, fixed_at, 0.0});
      }
    }

    // -- duplicate-column merging ------------------------------------------
    //
    // A hash match in `duplicate_column_candidates` means columns `first`
    // and `second` have the identical `A` pattern AND values -- confirmed
    // directly below regardless, per the analyzer's own "candidates, not
    // confirmations" contract. Identical `A` is not quite enough on its own:
    // two columns can share every constraint coefficient yet cost differently,
    // in which case they are NOT interchangeable (merging them would silently
    // change which one the optimizer prefers) -- `c_j == c_k` is this rule's
    // OWN safety condition, since nothing about the hash covers the objective
    // row at all.
    //
    // The survivor's widened bounds can never be infeasible on their own
    // (`l_j<=u_j` and `l_k<=u_k` individually already guarantee
    // `l_j+l_k<=u_j+u_k`), unlike singleton-row tightening's two independently
    // implied bounds above -- so there is no infeasibility path here, only a
    // merge.
    if (!has_q) {
      for (const auto& candidate : mat_analysis.duplicate_column_candidates) {
        const auto j = static_cast<std::size_t>(candidate.first);   // survivor
        const auto k = static_cast<std::size_t>(candidate.second);  // dropped
        if (!keep_col[j] || !keep_col[k]) continue;
        if (problem.c[j] != problem.c[k]) continue;
        if (!duplicate_columns_confirmed(problem.A, j, k)) continue;

        const bool lower_finite = is_finite_bound(problem.col_lower[j]) &&
                                  is_finite_bound(problem.col_lower[k]);
        const bool upper_finite = is_finite_bound(problem.col_upper[j]) &&
                                  is_finite_bound(problem.col_upper[k]);
        problem.col_lower[j] =
            lower_finite ? problem.col_lower[j] + problem.col_lower[k] : -INF;
        problem.col_upper[j] =
            upper_finite ? problem.col_upper[j] + problem.col_upper[k] : INF;

        keep_col[k] = false;
        changed = true;
        transforms.push(
            {TransformKind::MergeDuplicateColumn, maps.col[k], maps.col[j], 0.0, 0.0});
      }
    }

    // -- general singleton-column elimination (inequality rows only) -------
    //
    // A singleton column `j` (one entry `a_ij`, row `i`) that is NOT free --
    // generalizing free-column-singleton substitution, which only fires on a
    // totally free column in an EQUALITY row, to a column with a real bound
    // in an INEQUALITY row's one-sided form `a_ij*x_j + R + s_i = b_i,
    // s_i>=0`. `c_j == 0`: the objective doesn't care, so pick whichever
    // bound MINIMIZES `a_ij*x_j` -- maximizing row `i`'s remaining slack for
    // everything else in it -- skipping the column, unresolved, if that
    // bound is infinite (no cost pressure means no unboundedness either, but
    // this rule still needs a real number). `c_j != 0`, sign-MATCHED with
    // `a_ij` (`c_j>0,a_ij>0` or `c_j<0,a_ij<0`): that SAME slack-maximizing
    // bound is now ALSO the cost-improving one, so it can be fixed there
    // UNCONDITIONALLY -- a pure win, never costing row `i` any feasibility;
    // an infinite bound here is a second, genuine Unbounded certificate,
    // same reasoning as the empty-column case, just row-bounded rather than
    // totally free. Sign-MISMATCHED: `x_j`'s cost-improving direction
    // actively consumes row `i`'s slack -- genuinely coupled to the rest of
    // the problem, left untouched. Either resolvable case folds
    // `b_i -= a_ij*x_j*` (row `i` survives, transformed, not removed) and
    // reuses `RemoveFixedVariable`: `x_j`'s value is picked and baked into
    // `b_i` once, here, so it is a known constant from this point on, not
    // something recovery re-derives -- no new recovery math needed.
    if (!has_q) {
      const auto& csc = problem.A.csc;
      for (const Index j_idx : mat_analysis.singleton_columns) {
        const auto j = static_cast<std::size_t>(j_idx);
        if (!keep_col[j] || csc.slice_nnz(j) != 1) continue;
        const auto k = csc.slice_begin(j);
        const auto i = static_cast<std::size_t>(csc.indices()[k]);
        if (!keep_row[i] || i < problem.num_equality) continue;  // inequality rows only
        if (row_touches_range_column(problem, i)) continue;
        const Real a_ij = csc.values()[k];

        Real fixed_at = 0.0;
        if (problem.c[j] == 0.0) {
          const Real bound = a_ij > 0.0 ? problem.col_lower[j] : problem.col_upper[j];
          if (!is_finite_bound(bound)) continue;  // no cost pressure -- just leave it
          fixed_at = bound;
        } else {
          const bool sign_matched = (problem.c[j] > 0.0 && a_ij > 0.0) ||
                                    (problem.c[j] < 0.0 && a_ij < 0.0);
          if (!sign_matched) continue;  // genuinely coupled to the rest of the row -- leave it
          const Real bound = problem.c[j] > 0.0 ? problem.col_lower[j] : problem.col_upper[j];
          if (!is_finite_bound(bound)) {
            return core::make_error(ErrorCode::Unbounded,
                                    "presolve: singleton column " + std::to_string(j) +
                                        " has nonzero cost aligned with row " +
                                        std::to_string(i) + "'s slack direction and no "
                                        "bound there");
          }
          fixed_at = bound;
        }

        problem.b[i] -= a_ij * fixed_at;
        problem.obj_offset += problem.c[j] * fixed_at;
        keep_col[j] = false;
        changed = true;
        transforms.push({TransformKind::RemoveFixedVariable, maps.col[j], kNone, fixed_at, 0.0});
      }
    }

    if (!changed) return Status::Ok();
    Status st = rebuild(problem, keep_row, keep_col, maps, transforms);
    if (!st.ok()) return st;
  }
}

}  // namespace sovsolve::solver
