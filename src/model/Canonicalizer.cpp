// Canonicalization and solution recovery.
//
// See Canonical.hpp for the target form and the startability contract. The
// transform stack recorded here is the same stack presolve and scaling push
// onto, so recovery replays all three in reverse.
//
// The bounded-variable form makes this much smaller than a `x >= 0` form would:
// bounds pass through untouched, so there is no shifting, no reflection and no
// free-variable splitting. The only column operation is substituting out a
// fixed column, and the only row operations are negating a `>=` row, dropping a
// vacuous one, and permuting equalities to the front.

#include "sovsolve/model/Canonical.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/core/Types.hpp"

namespace sovsolve::model {
namespace {

using core::ErrorCode;
using core::Index;
using core::INF;
using core::is_finite_bound;
using core::Real;

constexpr Index kNone = TransformRecord::kIndexNone;

/// What becomes of one original row.
enum class RowFate : std::uint8_t {
  Equality,    ///< a'x = b, no slack
  Inequality,  ///< a'x + s = b, s >= 0 (possibly with a finite upper bound)
  Dropped,     ///< vacuous (both bounds infinite) or empty and consistent
};

struct RowPlan {
  RowFate fate = RowFate::Equality;
  Index canonical = kNone;   ///< canonical row index, kNone when dropped
  Index slack = kNone;       ///< index within `s`, kNone unless Inequality
  bool negate = false;       ///< `>=` rows are negated so a `+s` slack fits
  bool was_empty = false;    ///< dropped because no kept column touched it
  bool was_free = false;     ///< dropped because both bounds were infinite
  Real rhs = 0.0;
  /// Ranged rows only: the bounded column `t` added for this row, and its
  /// width `hi - lo`. A ranged row becomes an EQUALITY carrying `t`, not an
  /// inequality with a two-sided slack.
  Index range_col = kNone;
  Real range_width = 0.0;
};

/// y <- M * t, using the CSR view.
void multiply(const core::SparseMatrixPair<>& M, const std::vector<Real>& t,
              std::vector<Real>& y) {
  const auto off = M.csr.offsets();
  const auto idx = M.csr.indices();
  const auto val = M.csr.values();
  y.assign(M.rows(), 0.0);
  for (std::size_t i = 0; i < M.rows(); ++i) {
    Real acc = 0.0;
    for (auto k = static_cast<std::size_t>(off[i]);
         k < static_cast<std::size_t>(off[i + 1]); ++k) {
      acc += val[k] * t[static_cast<std::size_t>(idx[k])];
    }
    y[i] = acc;
  }
}

}  // namespace

// ---------------------------------------------------------------------------

Real CanonicalProblem::objective(core::HostSpan<const Real> x) const noexcept {
  Real linear = 0.0;
  for (std::size_t j = 0; j < c.size(); ++j) linear += c[j] * x[j];

  Real quad = 0.0;
  if (!Q.empty()) {
    const auto off = Q.csr.offsets();
    const auto idx = Q.csr.indices();
    const auto val = Q.csr.values();
    for (std::size_t i = 0; i < Q.rows(); ++i) {
      Real row = 0.0;
      for (auto k = static_cast<std::size_t>(off[i]);
           k < static_cast<std::size_t>(off[i + 1]); ++k) {
        row += val[k] * x[static_cast<std::size_t>(idx[k])];
      }
      quad += x[i] * row;
    }
    quad *= 0.5;
  }
  return quad + linear;
}

bool CanonicalProblem::validate() const noexcept {
  if (c.size() != num_cols()) return false;
  if (b.size() != num_rows()) return false;
  if (col_lower.size() != num_cols() || col_upper.size() != num_cols()) {
    return false;
  }
  if (num_equality > num_rows()) return false;
  if (num_range > num_cols()) return false;
  if (!A.csr.validate() || !A.csc.validate()) return false;
  if (!Q.empty() && (Q.rows() != num_cols() || Q.cols() != num_cols())) {
    return false;
  }
  return true;
}

bool CanonicalProblem::is_ipm_startable(std::size_t* bad_index,
                                        bool* bad_is_row) const noexcept {
  const auto fail = [&](std::size_t idx, bool is_row) {
    if (bad_index != nullptr) *bad_index = idx;
    if (bad_is_row != nullptr) *bad_is_row = is_row;
    return false;
  };

  // A fixed column admits no strictly interior point: x-l > 0 and u-x > 0 sum
  // to u-l, which is zero.
  for (std::size_t j = 0; j < num_cols(); ++j) {
    if (col_lower[j] == col_upper[j]) return fail(j, false);
    if (col_lower[j] > col_upper[j]) return fail(j, false);
  }

  // An all-zero row makes that row and column of A*Theta*A' identically zero
  // for every Theta -- an exact zero pivot, not an ill-conditioning.
  const auto off = A.csr.offsets();
  for (std::size_t i = 0; i < num_rows(); ++i) {
    if (off[i + 1] == off[i]) return fail(i, true);
  }
  return true;
}

// ---------------------------------------------------------------------------

core::Expected<CanonicalResult> canonicalize(const Problem& problem,
                                             const Options& options) {
  const std::size_t m0 = problem.num_rows();
  const std::size_t n0 = problem.num_cols();
  const Real feas_tol = options.tolerances.bound_violation;

  CanonicalResult out;
  auto& stack = out.transforms;
  stack.original_rows = m0;
  stack.original_cols = n0;

  const bool maximize = problem.sense == core::ObjSense::Maximize;
  const Real sense_sign = maximize ? -1.0 : 1.0;
  if (maximize) {
    stack.push({TransformKind::NegateObjective, kNone, kNone, 0.0, 0.0});
  }

  // -- columns: keep, or substitute out as fixed ---------------------------
  //
  // `t` holds the value of every substituted column and zero elsewhere, so the
  // whole substitution reduces to the three products A*t, Q*t and c'*t.

  std::vector<Index> col_map(n0, kNone);
  std::vector<Real> t(n0, 0.0);
  std::size_t n_kept = 0;

  for (std::size_t j = 0; j < n0; ++j) {
    const Real lo = problem.col_lower[j];
    const Real hi = problem.col_upper[j];
    if (lo > hi) {
      return core::make_error(
          ErrorCode::InconsistentBounds,
          "column " + std::to_string(j) + " has lower bound above upper bound");
    }
    if (is_finite_bound(lo) && is_finite_bound(hi) && lo == hi) {
      t[j] = lo;  // fixed; column disappears
      continue;
    }
    col_map[j] = static_cast<Index>(n_kept++);
  }

  std::vector<Real> at;
  multiply(problem.A, t, at);
  std::vector<Real> qt(n0, 0.0);
  if (!problem.Q.empty()) multiply(problem.Q, t, qt);

  // -- rows ----------------------------------------------------------------
  //
  // Emptiness is counted over the KEPT columns, because substituting fixed
  // columns out can empty a row that had entries before. This is the same
  // effect that makes presolve create new empty rows, appearing here first.

  std::vector<std::size_t> kept_nnz(m0, 0);
  {
    const auto off = problem.A.csr.offsets();
    const auto idx = problem.A.csr.indices();
    for (std::size_t i = 0; i < m0; ++i) {
      std::size_t count = 0;
      for (auto k = static_cast<std::size_t>(off[i]);
           k < static_cast<std::size_t>(off[i + 1]); ++k) {
        if (col_map[static_cast<std::size_t>(idx[k])] != kNone) ++count;
      }
      kept_nnz[i] = count;
    }
  }

  std::vector<RowPlan> rows(m0);
  std::size_t n_equality = 0;
  std::size_t n_inequality = 0;
  std::size_t n_range = 0;

  for (std::size_t i = 0; i < m0; ++i) {
    const Real raw_lo = problem.row_lower[i];
    const Real raw_hi = problem.row_upper[i];
    if (raw_lo > raw_hi) {
      return core::make_error(
          ErrorCode::InconsistentBounds,
          "row " + std::to_string(i) + " has lower bound above upper bound");
    }
    auto& r = rows[i];

    const bool lo_finite = is_finite_bound(raw_lo);
    const bool hi_finite = is_finite_bound(raw_hi);

    // The fixed columns' contribution moves to the right-hand side.
    const Real lo = lo_finite ? raw_lo - at[i] : -INF;
    const Real hi = hi_finite ? raw_hi - at[i] : INF;

    if (!lo_finite && !hi_finite) {
      r.fate = RowFate::Dropped;
      r.was_free = true;
      continue;
    }

    if (kept_nnz[i] == 0) {
      // The row's activity is identically zero, so the model is feasible in
      // this row only if zero lies within the adjusted bounds. This is a
      // verdict, not a numerical difficulty: passing it on would surface as an
      // unexplained factorization breakdown instead of INFEASIBLE.
      const bool zero_below = lo_finite && lo > feas_tol;
      const bool zero_above = hi_finite && hi < -feas_tol;
      if (zero_below || zero_above) {
        return core::make_error(
            ErrorCode::PrimalInfeasible,
            "row " + std::to_string(i) +
                " has no entries outside fixed columns and a right-hand side "
                "that excludes zero");
      }
      r.fate = RowFate::Dropped;
      r.was_empty = true;
      continue;
    }

    if (lo_finite && hi_finite && raw_lo == raw_hi) {
      r.fate = RowFate::Equality;
      r.rhs = hi;
      ++n_equality;
    } else if (hi_finite && !lo_finite) {
      r.fate = RowFate::Inequality;
      r.rhs = hi;
      ++n_inequality;
    } else if (lo_finite && !hi_finite) {
      // `>=`: negate the row so a non-negative slack fits `a'x + s = b`.
      r.fate = RowFate::Inequality;
      r.negate = true;
      r.rhs = -lo;
      ++n_inequality;
    } else {
      // Ranged: `a'x + t = hi` with `0 <= t <= hi - lo`, an EQUALITY carrying a
      // bounded column. Not an inequality with a two-sided slack -- that would
      // need a second dual and a matching residual, step-length test and mu
      // term across six downstream modules, where a bounded column needs
      // nothing new at all.
      r.fate = RowFate::Equality;
      r.rhs = hi;
      r.range_width = hi - lo;
      r.range_col = 0;  // real index assigned once n_kept is final
      ++n_equality;
      ++n_range;
    }
  }

  // -- canonical row order: equalities first, then inequalities ------------
  //
  // Contiguity is what lets A_E and A_I be row ranges of one matrix rather than
  // two matrices, so A*x and A'*y stay single kernel calls.

  {
    std::size_t next_eq = 0;
    std::size_t next_ineq = n_equality;
    std::size_t next_slack = 0;
    for (std::size_t i = 0; i < m0; ++i) {
      auto& r = rows[i];
      switch (r.fate) {
        case RowFate::Equality:
          r.canonical = static_cast<Index>(next_eq++);
          break;
        case RowFate::Inequality:
          r.canonical = static_cast<Index>(next_ineq++);
          r.slack = static_cast<Index>(next_slack++);
          break;
        case RowFate::Dropped:
          break;
      }
    }
    // Range columns are appended after every kept original column, so the
    // original columns keep indices `[0, n_kept)` and recovery stays a direct
    // lookup.
    std::size_t next_range = n_kept;
    for (std::size_t i = 0; i < m0; ++i) {
      if (rows[i].range_col == kNone) continue;
      rows[i].range_col = static_cast<Index>(next_range++);
    }
  }

  const std::size_t n_canon_rows = n_equality + n_inequality;
  const std::size_t n_canon_cols = n_kept + n_range;

  // -- record the stack ----------------------------------------------------
  //
  // Records are keyed by the ORIGINAL index in `primary`. Every column pushes
  // exactly one record and every row at least one, so recovery is a lookup
  // rather than a positional walk -- a walk breaks the moment one entry pushes
  // nothing, because everything after it then shifts.

  for (std::size_t j = 0; j < n0; ++j) {
    const auto oj = static_cast<Index>(j);
    if (col_map[j] == kNone) {
      stack.push({TransformKind::RemoveFixedVariable, oj, kNone, t[j], 0.0});
    } else {
      stack.push({TransformKind::KeepColumn, oj, col_map[j], 0.0, 0.0});
    }
  }

  for (std::size_t i = 0; i < m0; ++i) {
    const auto& r = rows[i];
    const auto oi = static_cast<Index>(i);
    if (r.fate == RowFate::Dropped) {
      stack.push({r.was_free ? TransformKind::DropFreeRow
                             : TransformKind::RemoveEmptyRow,
                  oi, kNone, 0.0, 0.0});
      continue;
    }
    stack.push({TransformKind::MapRow, oi, r.canonical, 0.0, 0.0});
    if (r.negate) stack.push({TransformKind::NegateRow, oi, kNone, 0.0, 0.0});
    if (r.fate == RowFate::Inequality) {
      stack.push({TransformKind::AddSlack, oi, r.slack, r.rhs, 0.0});
    } else if (r.range_col != kNone) {
      stack.push({TransformKind::BoundedSlack, oi, r.range_col, r.rhs,
                  r.range_width});
    }
  }

  // -- objective under the substitution ------------------------------------
  //
  // Splitting x into kept (k) and fixed (f = t_f):
  //
  //     1/2 x'Qx + c'x  =  1/2 x_k'Q_kk x_k + (c_k + (Qt)_k)'x_k
  //                        + 1/2 t'Qt + c't
  //
  // with the sense flip applied so the offset accumulates in canonical
  // (minimization) space.

  auto& cp = out.problem;
  cp.objective_negated = maximize;
  cp.num_equality = n_equality;

  cp.num_range = n_range;
  cp.c = core::RealVector(n_canon_cols, 0.0);
  cp.col_lower = core::RealVector(n_canon_cols, 0.0);
  cp.col_upper = core::RealVector(n_canon_cols, 0.0);

  Real offset = 0.0;
  for (std::size_t j = 0; j < n0; ++j) {
    if (col_map[j] == kNone) {
      offset += sense_sign * (0.5 * qt[j] * t[j] + problem.c[j] * t[j]);
      continue;
    }
    const auto cj = static_cast<std::size_t>(col_map[j]);
    cp.c[cj] = sense_sign * (problem.c[j] + qt[j]);
    cp.col_lower[cj] = problem.col_lower[j];
    cp.col_upper[cj] = problem.col_upper[j];
  }
  cp.obj_offset = offset;

  // -- right-hand sides and slack bounds -----------------------------------

  cp.b = core::RealVector(n_canon_rows, 0.0);
  for (std::size_t i = 0; i < m0; ++i) {
    const auto& r = rows[i];
    if (r.fate == RowFate::Dropped) continue;
    cp.b[static_cast<std::size_t>(r.canonical)] = r.rhs;
    if (r.range_col != kNone) {
      const auto rc = static_cast<std::size_t>(r.range_col);
      cp.col_lower[rc] = 0.0;
      cp.col_upper[rc] = r.range_width;
    }
  }

  // -- constraint matrix ---------------------------------------------------
  //
  // Enumerated twice through one lambda -- once to count, once to fill -- so
  // SparseBuilder sizes exactly with no reallocation. Slacks are NOT columns
  // here: the slack block would be exactly the identity, and the reduced system
  // never needs it formed.

  const auto enumerate = [&](auto&& emit) {
    const auto off = problem.A.csr.offsets();
    const auto idx = problem.A.csr.indices();
    const auto val = problem.A.csr.values();

    for (std::size_t i = 0; i < m0; ++i) {
      const auto& r = rows[i];
      if (r.fate == RowFate::Dropped) continue;
      const Real row_sign = r.negate ? -1.0 : 1.0;
      for (auto k = static_cast<std::size_t>(off[i]);
           k < static_cast<std::size_t>(off[i + 1]); ++k) {
        const auto j = static_cast<std::size_t>(idx[k]);
        if (col_map[j] == kNone) continue;  // folded into the right-hand side
        emit(r.canonical, col_map[j], row_sign * val[k]);
      }
      if (r.range_col != kNone) emit(r.canonical, r.range_col, 1.0);
    }
  };

  core::SparseBuilder builder(n_canon_rows, n_canon_cols);
  enumerate([&](Index r, Index c, Real) { builder.count(r, c); });
  if (auto st = builder.allocate(); !st.ok()) return st.error();
  enumerate([&](Index r, Index c, Real v) { builder.insert(r, c, v); });
  cp.A = builder.finish(true, 0.0);

  // -- quadratic objective, restricted to the kept columns -----------------

  if (!problem.Q.empty()) {
    const auto off = problem.Q.csr.offsets();
    const auto idx = problem.Q.csr.indices();
    const auto val = problem.Q.csr.values();

    const auto enumerate_q = [&](auto&& emit) {
      for (std::size_t i = 0; i < problem.Q.rows(); ++i) {
        if (col_map[i] == kNone) continue;
        for (auto k = static_cast<std::size_t>(off[i]);
             k < static_cast<std::size_t>(off[i + 1]); ++k) {
          const auto j = static_cast<std::size_t>(idx[k]);
          if (col_map[j] == kNone) continue;
          emit(col_map[i], col_map[j], sense_sign * val[k]);
        }
      }
    };

    core::SparseBuilder qb(n_canon_cols, n_canon_cols);
    enumerate_q([&](Index r, Index c, Real) { qb.count(r, c); });
    if (auto st = qb.allocate(); !st.ok()) return st.error();
    enumerate_q([&](Index r, Index c, Real v) { qb.insert(r, c, v); });
    cp.Q = qb.finish(true, 0.0);
  }

  return out;
}

// ---------------------------------------------------------------------------

core::Expected<Solution> recover_solution(const Problem& original,
                                          const CanonicalProblem& canonical,
                                          const TransformStack& transforms,
                                          const Solution& canonical_solution) {
  const std::size_t m0 = original.num_rows();
  const std::size_t n0 = original.num_cols();

  if (transforms.original_rows != m0 || transforms.original_cols != n0) {
    return core::make_error(ErrorCode::DimensionMismatch,
                            "transform stack does not describe this problem");
  }

  // Rebuild the maps from the recorded stack rather than recomputing them.
  // Replaying the record is what makes this the *inverse* of canonicalize()
  // instead of a parallel implementation that can drift away from it.
  std::vector<Index> col_map(n0, kNone);
  std::vector<Real> fixed_value(n0, 0.0);
  std::vector<Index> row_map(m0, kNone);
  std::vector<bool> row_negated(m0, false);

  // Presolver's free-column-singleton substitution (Presolver.cpp): row
  // `primary` and column `secondary` were removed TOGETHER, so neither ever
  // receives a MapRow/KeepColumn record -- row_map/col_map stay kNone for
  // both, same as any other dropped row/column. Recovered separately below,
  // once every kept/fixed column's `x` is known. `is_free_singleton_col`
  // marks the column side so the generic `any_fixed` bound-dual branch
  // (below) skips it -- a free variable has z_j = v_j = 0 always, not the
  // stationarity-derived split that branch computes for a genuinely FIXED
  // column.
  std::vector<Index> free_singleton_col_of_row(m0, kNone);
  std::vector<bool> is_free_singleton_col(n0, false);

  // Presolver's duplicate-column merge (Presolver.cpp): `k` (dropped) never
  // gets a KeepColumn record either, same reasoning as a free singleton's
  // column above, so `is_merged_dropped_col` marks it for the same kind of
  // skip in the generic bound-dual branch below -- its dual is NOT the
  // generic "must be at one of its two bounds" split that branch assumes,
  // since a merged pair's survivor can land strictly inside its own box.
  // `merge_pairs` (dropped `k`, surviving `j`, both ORIGINAL indices) drives
  // the primal split and dual recovery further down.
  std::vector<bool> is_merged_dropped_col(n0, false);
  std::vector<std::pair<Index, Index>> merge_pairs;
  // Which of {j at l_j, j at u_j, k at l_k, k at u_k, neither} the primal
  // split below picked, in lockstep with `merge_pairs` -- computed once
  // there and reused for the dual split further down, instead of
  // re-deriving it from comparing floats (`x_j == l_j` is not reliably
  // exact after `y - u_k`-style arithmetic; the branch taken IS exact).
  enum class MergeBranch { kJLower, kJUpper, kKLower, kKUpper, kNeither };
  std::vector<MergeBranch> merge_branch;

  // Scaling (Scaler, Module 5) pushes onto this SAME stack, keyed by
  // CANONICAL row/column index rather than original index -- it runs after
  // canonicalization, on the canonicalized problem. Default 1.0 (identity)
  // covers both a Scaler that never ran and a canonical index a record
  // didn't reach.
  std::vector<Real> col_scale(canonical.num_cols(), 1.0);
  std::vector<Real> row_scale(canonical.num_rows(), 1.0);

  for (const auto& rec : transforms.records()) {
    if (rec.primary == kNone) continue;
    const auto k = static_cast<std::size_t>(rec.primary);
    switch (rec.kind) {
      case TransformKind::KeepColumn:
        if (k < n0) col_map[k] = rec.secondary;
        break;
      case TransformKind::RemoveFixedVariable:
        // canonicalize() itself may have ALREADY pushed a KeepColumn record
        // for this same original index (it keeps every column IT doesn't
        // fix, with no way to know Presolver will fix this one later), and
        // rebuild() never pushes a competing record for a column it drops --
        // so without this, "last record wins" would leave col_map[k] at that
        // STALE canonical index, which the dual-recovery loop below reads as
        // "still a normal kept column" and pulls its z/v from whatever
        // (unrelated) canonical column that stale index now names, instead
        // of taking the stationarity-based path meant for a substituted one.
        if (k < n0) {
          fixed_value[k] = rec.value;
          col_map[k] = kNone;
        }
        break;
      case TransformKind::MapRow:
        if (k < m0) row_map[k] = rec.secondary;
        break;
      case TransformKind::NegateRow:
        if (k < m0) row_negated[k] = true;
        break;
      case TransformKind::ColumnScaling:
        if (k < col_scale.size()) col_scale[k] = rec.value;
        break;
      case TransformKind::RowScaling:
        if (k < row_scale.size()) row_scale[k] = rec.value;
        break;
      case TransformKind::RemoveFreeSingleton:
        if (k < m0) free_singleton_col_of_row[k] = rec.secondary;
        // Same stale-KeepColumn hazard as RemoveFixedVariable above, for the
        // dropped free column (`rec.secondary`, not `k` -- `k` is the ROW).
        if (rec.secondary != kNone && static_cast<std::size_t>(rec.secondary) < n0) {
          is_free_singleton_col[static_cast<std::size_t>(rec.secondary)] = true;
          col_map[static_cast<std::size_t>(rec.secondary)] = kNone;
        }
        break;
      case TransformKind::MergeDuplicateColumn:
        // `k` = rec.primary (dropped), `j` = rec.secondary (survivor). Same
        // stale-KeepColumn hazard as RemoveFixedVariable above, for `k`.
        if (k < n0) {
          is_merged_dropped_col[k] = true;
          col_map[k] = kNone;
        }
        if (rec.secondary != kNone) merge_pairs.emplace_back(rec.primary, rec.secondary);
        break;
      default:
        break;
    }
  }

  Solution s;
  s.status = canonical_solution.status;
  s.iterations = canonical_solution.iterations;
  s.solve_time_seconds = canonical_solution.solve_time_seconds;
  s.from_best_iterate = canonical_solution.from_best_iterate;
  s.quality = canonical_solution.quality;

  // Unscale into true canonical-space values -- x = col_scale * x', y =
  // row_scale * y', z = z'/col_scale, v = v'/col_scale (derived from
  // stationarity in the scaled system matching stationarity in the
  // unscaled one; see Scaler.hpp). All four stay the raw canonical_solution
  // values when Scaler never ran, since col_scale/row_scale default to 1.0.
  //
  // The OBJECTIVE below deliberately uses canonical_solution.x directly
  // (still scaled), not this unscaled xc: the objective is scale-invariant
  // only for a MATCHED pair, and canonical.c/canonical.Q are the scaled
  // coefficients Scaler left in place. Pairing scaled c/Q with unscaled x
  // here would silently reintroduce the scale factor into the reported
  // objective.
  core::RealVector xc(canonical_solution.x.size());
  for (std::size_t j = 0; j < xc.size(); ++j) {
    xc[j] = canonical_solution.x[j] * (j < col_scale.size() ? col_scale[j] : 1.0);
  }
  core::RealVector yc(canonical_solution.y.size());
  for (std::size_t i = 0; i < yc.size(); ++i) {
    yc[i] = canonical_solution.y[i] * (i < row_scale.size() ? row_scale[i] : 1.0);
  }
  core::RealVector zc(canonical_solution.z.size());
  for (std::size_t j = 0; j < zc.size(); ++j) {
    const Real cs = j < col_scale.size() && col_scale[j] != 0.0 ? col_scale[j] : 1.0;
    zc[j] = canonical_solution.z[j] / cs;
  }
  core::RealVector vc(canonical_solution.v.size());
  for (std::size_t j = 0; j < vc.size(); ++j) {
    const Real cs = j < col_scale.size() && col_scale[j] != 0.0 ? col_scale[j] : 1.0;
    vc[j] = canonical_solution.v[j] / cs;
  }

  const auto at = [](const core::RealVector& v, Index i) -> Real {
    if (i == kNone) return 0.0;
    const auto k = static_cast<std::size_t>(i);
    return k < v.size() ? v[k] : 0.0;
  };

  // -- primal --------------------------------------------------------------
  s.x = core::RealVector(n0, 0.0);
  for (std::size_t j = 0; j < n0; ++j) {
    s.x[j] = col_map[j] == kNone ? fixed_value[j] : at(xc, col_map[j]);
  }

  // -- duplicate-column primal split (Presolver.cpp) ------------------------
  //
  // `k` (dropped) and `j` (survivor) are ORIGINAL indices for a
  // pattern-and-cost-identical pair Presolver merged into one column with
  // Minkowski-summed bounds. `s.x[j]` above holds their COMBINED value `y`
  // (the reduced problem's own answer for the widened column), not `j`'s
  // true value -- split here, before anything downstream treats it as `j`'s
  // own. Deterministically prefers `j`'s own lower bound, falling back to
  // `j`'s upper, then to `k`'s bounds, and finally (only when EVERY one of
  // the four bounds is infinite -- both columns fully free) an arbitrary
  // even split -- so it always lands inside both original boxes without the
  // `INF` sentinel ever entering arithmetic (`is_finite_bound()` guards every
  // step).
  for (const auto& [k_idx, j_idx] : merge_pairs) {
    const auto k = static_cast<std::size_t>(k_idx);
    const auto j = static_cast<std::size_t>(j_idx);
    const Real y = s.x[j];
    const Real l_j = original.col_lower[j];
    const Real u_j = original.col_upper[j];
    const Real l_k = original.col_lower[k];
    const Real u_k = original.col_upper[k];

    Real x_j;
    MergeBranch branch;
    if (is_finite_bound(l_j)) {
      x_j = l_j;
      branch = MergeBranch::kJLower;
      if (is_finite_bound(u_k) && y - u_k > x_j) {
        x_j = y - u_k;
        branch = MergeBranch::kKUpper;
      }
    } else if (is_finite_bound(u_j)) {
      x_j = u_j;
      branch = MergeBranch::kJUpper;
      if (is_finite_bound(l_k) && y - l_k < x_j) {
        x_j = y - l_k;
        branch = MergeBranch::kKLower;
      }
    } else if (is_finite_bound(l_k)) {
      x_j = y - l_k;
      branch = MergeBranch::kKLower;
    } else if (is_finite_bound(u_k)) {
      x_j = y - u_k;
      branch = MergeBranch::kKUpper;
    } else {
      x_j = y / 2.0;  // both columns fully free: arbitrary split, no bound ever active
      branch = MergeBranch::kNeither;
    }
    s.x[j] = x_j;
    s.x[k] = y - x_j;
    merge_branch.push_back(branch);
  }

  // -- free-column-singleton primal recovery (Presolver.cpp) ----------------
  //
  // Row `i` and column `j` were removed TOGETHER (`i`/`j` here are already
  // ORIGINAL indices -- `free_singleton_col_of_row` is keyed that way, see
  // the scan loop above): `x_j` was never solved by the reduced problem, so
  // the primal loop just above left it at `fixed_value`'s default, 0.0 --
  // wrong. Presolver restricts this transform to EQUALITY rows only (an
  // inequality row's canonical form has an implicit slack this formula does
  // not account for), so the ORIGINAL row's equation is exactly
  // `row_lower[i] == row_upper[i] == sum_k a_ik*x_k` with no slack term,
  // and solving it for the one unknown gives
  //
  //     x_j = (row_lower[i] - sum_{k!=j} a_ik*x_k) / a_ij
  //
  // using row i's OTHER coefficients against the x_k values already
  // recovered above -- every other column in row i is necessarily kept or
  // fixed (by canonicalize() OR by Presolver reusing RemoveFixedVariable),
  // never another free singleton: row i is claimed by at most one such pair.
  for (std::size_t i = 0; i < m0; ++i) {
    if (free_singleton_col_of_row[i] == kNone) continue;
    const auto j = static_cast<std::size_t>(free_singleton_col_of_row[i]);

    const auto off = original.A.csr.offsets();
    const auto idx = original.A.csr.indices();
    const auto val = original.A.csr.values();

    Real a_ij = 0.0;
    Real other_terms = 0.0;
    for (auto k = static_cast<std::size_t>(off[i]); k < static_cast<std::size_t>(off[i + 1]);
         ++k) {
      const auto col = static_cast<std::size_t>(idx[k]);
      if (col == j) {
        a_ij = val[k];
      } else {
        other_terms += val[k] * s.x[col];
      }
    }

    s.x[j] = (original.row_lower[i] - other_terms) / a_ij;
  }

  // -- row activity and slacks in original terms ---------------------------
  s.s = core::RealVector(m0, 0.0);
  {
    const auto off = original.A.csr.offsets();
    const auto idx = original.A.csr.indices();
    const auto val = original.A.csr.values();
    for (std::size_t i = 0; i < m0; ++i) {
      Real act = 0.0;
      for (auto k = static_cast<std::size_t>(off[i]);
           k < static_cast<std::size_t>(off[i + 1]); ++k) {
        act += val[k] * s.x[static_cast<std::size_t>(idx[k])];
      }
      // Distance to whichever bound is finite, so a reported slack means what
      // it meant in the original model.
      if (is_finite_bound(original.row_upper[i])) {
        s.s[i] = original.row_upper[i] - act;
      } else if (is_finite_bound(original.row_lower[i])) {
        s.s[i] = act - original.row_lower[i];
      } else {
        s.s[i] = 0.0;
      }
    }
  }

  // -- row duals -----------------------------------------------------------
  //
  // A dropped row has no dual. A negated row has its dual negated back, which
  // is exactly what turns the internally non-positive dual of the negated `<=`
  // row into the non-negative dual a `>=` row is reported with. No further
  // normalization: the convention already matches HiGHS (see Solution.hpp).
  s.y = core::RealVector(m0, 0.0);
  for (std::size_t i = 0; i < m0; ++i) {
    const Real raw = at(yc, row_map[i]);
    s.y[i] = row_negated[i] ? -raw : raw;
  }

  // A free-column-singleton row has no MapRow record (dropped, like any
  // other removed row), so the loop above leaves it at 0 -- overwritten here
  // via stationarity, `y_i = (c_j + (Qx)_j) / a_ij`. Presolver only ever
  // emits this transform when column j has no entries in Q, so `(Qx)_j` is
  // exactly 0 (guaranteed by the producer's precondition, not approximated
  // here) -- see the primal-recovery comment above for `a_ij`.
  for (std::size_t i = 0; i < m0; ++i) {
    if (free_singleton_col_of_row[i] == kNone) continue;
    const auto j = static_cast<std::size_t>(free_singleton_col_of_row[i]);
    const auto& csr = original.A.csr;
    Real a_ij = 0.0;
    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      if (static_cast<std::size_t>(csr.indices()[k]) == j) {
        a_ij = csr.values()[k];
        break;
      }
    }
    s.y[i] = original.c[j] / a_ij;
  }

  // -- bound duals ---------------------------------------------------------
  //
  // Kept columns carry theirs through unchanged: bounds were never transformed,
  // so neither were their multipliers. A substituted column has no dual in the
  // canonical solution at all, so its reduced cost is reconstructed from the
  // stationarity condition against the ORIGINAL data,
  //
  //     d_j = sigma*(c_j + (Qx)_j) - (A'y)_j
  //
  // and split into the non-negative pair. A fixed column may sit at either
  // bound, so the sign of d_j decides which side holds it.
  s.z = core::RealVector(n0, 0.0);
  s.v = core::RealVector(n0, 0.0);

  const Real sense_sign = original.sense == core::ObjSense::Maximize ? -1.0 : 1.0;
  bool any_fixed = false;
  for (std::size_t j = 0; j < n0; ++j) {
    if (col_map[j] == kNone) {
      any_fixed = true;
      continue;
    }
    s.z[j] = at(zc, col_map[j]);
    s.v[j] = at(vc, col_map[j]);
  }

  if (any_fixed) {
    // A'y over the original matrix, accumulated from the CSR view so the scan
    // stays row-major; a CSC walk would be the natural shape but this avoids
    // depending on both orientations being present.
    std::vector<Real> aty(n0, 0.0);
    {
      const auto off = original.A.csr.offsets();
      const auto idx = original.A.csr.indices();
      const auto val = original.A.csr.values();
      for (std::size_t i = 0; i < m0; ++i) {
        const Real yi = s.y[i];
        if (yi == 0.0) continue;
        for (auto k = static_cast<std::size_t>(off[i]);
             k < static_cast<std::size_t>(off[i + 1]); ++k) {
          aty[static_cast<std::size_t>(idx[k])] += val[k] * yi;
        }
      }
    }
    std::vector<Real> qx(n0, 0.0);
    if (!original.Q.empty()) {
      std::vector<Real> xv(n0, 0.0);
      for (std::size_t j = 0; j < n0; ++j) xv[j] = s.x[j];
      multiply(original.Q, xv, qx);
    }
    for (std::size_t j = 0; j < n0; ++j) {
      if (col_map[j] != kNone) continue;
      // Free (unbounded) columns keep z_j = v_j = 0 -- there is no bound for
      // either to be active against, unlike a genuinely FIXED column, which
      // is what the stationarity split below assumes.
      if (is_free_singleton_col[j]) continue;
      // A merged pair's DROPPED column is handled below, off the SURVIVOR's
      // stationarity value -- not this generic split, which assumes (unlike
      // a merged pair) that landing off of ITS bound is impossible.
      if (is_merged_dropped_col[j]) continue;
      const Real d = sense_sign * (original.c[j] + qx[j]) - aty[j];
      if (d >= 0.0) {
        s.z[j] = d;
      } else {
        s.v[j] = -d;
      }
    }

    // -- duplicate-column dual split (Presolver.cpp) -----------------------
    //
    // `c_j == c_k` and `A_j == A_k` exactly (Presolver only merges confirmed
    // duplicates), so the stationarity value is the SAME for both columns --
    // computed once, from `j`'s own data, using the SAME `aty`/`qx` built
    // above (survivor `j` is a normal kept column, so `aty[j]`/`qx[j]` were
    // already accumulated over the full original matrix, not skipped by the
    // `col_map[j] == kNone` guard above). What differs between the two is
    // only WHICH bound (if either) each one's own split value actually
    // landed on -- decided by `merge_branch`, in lockstep with `merge_pairs`,
    // from the primal split above, not by re-comparing floats here.
    for (std::size_t idx = 0; idx < merge_pairs.size(); ++idx) {
      const auto k = static_cast<std::size_t>(merge_pairs[idx].first);
      const auto j = static_cast<std::size_t>(merge_pairs[idx].second);
      const Real d = sense_sign * (original.c[j] + qx[j]) - aty[j];

      s.z[j] = 0.0;
      s.v[j] = 0.0;
      s.z[k] = 0.0;
      s.v[k] = 0.0;
      switch (merge_branch[idx]) {
        case MergeBranch::kJLower:
          s.z[j] = d >= 0.0 ? d : 0.0;
          break;
        case MergeBranch::kJUpper:
          s.v[j] = d < 0.0 ? -d : 0.0;
          break;
        case MergeBranch::kKLower:
          s.z[k] = d >= 0.0 ? d : 0.0;
          break;
        case MergeBranch::kKUpper:
          s.v[k] = d < 0.0 ? -d : 0.0;
          break;
        case MergeBranch::kNeither:
          break;  // neither column is at a bound of its own -- both stay 0
      }
    }
  }

  // -- objective, back in the original sense -------------------------------
  {
    // canonical_solution.x, NOT xc: see the comment where xc is built above.
    Real obj = canonical.objective(canonical_solution.x.span());
    obj += canonical.obj_offset;
    if (canonical.objective_negated) obj = -obj;
    s.objective = obj + original.obj_constant;
  }

  // Bound violations are measured against the ORIGINAL bounds.
  Real worst = 0.0;
  for (std::size_t j = 0; j < n0; ++j) {
    if (is_finite_bound(original.col_lower[j])) {
      worst = std::max(worst, original.col_lower[j] - s.x[j]);
    }
    if (is_finite_bound(original.col_upper[j])) {
      worst = std::max(worst, s.x[j] - original.col_upper[j]);
    }
  }
  s.quality.max_bound_violation = std::max(worst, Real{0.0});

  return s;
}

}  // namespace sovsolve::model
