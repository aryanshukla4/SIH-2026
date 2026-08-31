// Canonicalization and solution recovery.
//
// See Canonical.hpp for the target form and the affine-map framing. The
// transform stack recorded here is the same stack presolve and scaling push
// onto, so recovery replays all three in reverse.

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

/// How one original column becomes one or two canonical columns.
struct ColumnPlan {
  Index primary = kNone;    ///< canonical column for x'
  Index partner = kNone;    ///< canonical column for the negative part, if split
  Real scale = 1.0;         ///< d in x = d*x' + t
  Real shift = 0.0;         ///< t
  bool split = false;       ///< free variable, x = xp - xm
  bool bounded = false;     ///< needs a bound row
  Real width = 0.0;         ///< upper - lower, when bounded
  Index bound_row = kNone;
  Index bound_col = kNone;  ///< the auxiliary t of the bound row
};

/// How one original row becomes one canonical row.
struct RowPlan {
  Index canonical = kNone;
  bool negate = false;      ///< `>=` rows are negated so a `+s` slack fits
  bool has_slack = false;   ///< false for equality rows
  Index slack_col = kNone;
  Real rhs = 0.0;
  bool slack_bounded = false;   ///< ranged row: the slack has its own upper bound
  Real slack_width = 0.0;
  Index slack_bound_row = kNone;
  Index slack_bound_col = kNone;
};

/// y <- A * t, using the CSR view.
void multiply(const core::SparseMatrixPair<>& A, const std::vector<Real>& t,
              std::vector<Real>& y) {
  const auto off = A.csr.offsets();
  const auto idx = A.csr.indices();
  const auto val = A.csr.values();
  y.assign(A.rows(), 0.0);
  for (std::size_t i = 0; i < A.rows(); ++i) {
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
  if (num_structural + num_slack + num_bound != num_cols()) return false;
  if (c.size() != num_cols()) return false;
  if (b.size() != num_rows()) return false;
  if (!A.csr.validate() || !A.csc.validate()) return false;
  if (!Q.empty() && (Q.rows() != num_cols() || Q.cols() != num_cols())) {
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------

core::Expected<CanonicalResult> canonicalize(const Problem& problem,
                                             const Options& options) {
  (void)options;

  const std::size_t m0 = problem.num_rows();
  const std::size_t n0 = problem.num_cols();

  CanonicalResult out;
  auto& stack = out.transforms;
  stack.original_rows = m0;
  stack.original_cols = n0;

  const bool maximize = problem.sense == core::ObjSense::Maximize;
  const Real sense_sign = maximize ? -1.0 : 1.0;
  if (maximize) {
    stack.push({TransformKind::NegateObjective, kNone, kNone, 0.0, 0.0});
  }

  // -- plan the columns ----------------------------------------------------
  //
  // Four shapes, each an instance of x = d*x' + t (free variables excepted).

  std::vector<ColumnPlan> cols(n0);
  std::size_t n_structural = 0;
  std::size_t n_bound_rows = 0;

  for (std::size_t j = 0; j < n0; ++j) {
    const Real lo = problem.col_lower[j];
    const Real hi = problem.col_upper[j];
    if (lo > hi) {
      return core::make_error(
          ErrorCode::InconsistentBounds,
          "column " + std::to_string(j) + " has lower bound above upper bound");
    }
    auto& p = cols[j];

    const bool lo_finite = is_finite_bound(lo);
    const bool hi_finite = is_finite_bound(hi);

    if (lo_finite && hi_finite) {
      // Boxed (or fixed, when the width is zero). Shift to zero, then the
      // remaining upper bound becomes a constraint row.
      p.scale = 1.0;
      p.shift = lo;
      p.bounded = true;
      p.width = hi - lo;
      ++n_bound_rows;
    } else if (lo_finite) {
      p.scale = 1.0;
      p.shift = lo;
    } else if (hi_finite) {
      // Upper bound only: reflect, so x' = u - x is non-negative.
      p.scale = -1.0;
      p.shift = hi;
    } else {
      // Free. The one case outside the affine map.
      p.split = true;
      p.scale = 1.0;
      p.shift = 0.0;
    }

    p.primary = static_cast<Index>(n_structural++);
    if (p.split) p.partner = static_cast<Index>(n_structural++);
  }

  // -- plan the rows -------------------------------------------------------

  std::vector<RowPlan> rows(m0);
  std::size_t n_slack = 0;

  for (std::size_t i = 0; i < m0; ++i) {
    const Real lo = problem.row_lower[i];
    const Real hi = problem.row_upper[i];
    if (lo > hi) {
      return core::make_error(
          ErrorCode::InconsistentBounds,
          "row " + std::to_string(i) + " has lower bound above upper bound");
    }
    auto& r = rows[i];
    r.canonical = static_cast<Index>(i);

    const bool lo_finite = is_finite_bound(lo);
    const bool hi_finite = is_finite_bound(hi);

    if (lo_finite && hi_finite && lo == hi) {
      // Equality: no slack. This is the case the handoff form could not
      // express, and it is the majority of rows in real instances.
      r.rhs = hi;
    } else if (lo_finite && hi_finite) {
      // Ranged: a'x + s = u with 0 <= s <= u - l, so the slack needs its own
      // bound row.
      r.has_slack = true;
      r.rhs = hi;
      r.slack_bounded = true;
      r.slack_width = hi - lo;
      ++n_slack;
      ++n_bound_rows;
    } else if (hi_finite) {
      r.has_slack = true;
      r.rhs = hi;
      ++n_slack;
    } else if (lo_finite) {
      // `>=`: negate so that a non-negative slack fits the `+s` form.
      r.negate = true;
      r.has_slack = true;
      r.rhs = -lo;
      ++n_slack;
    } else {
      // Free row. The reader drops these, so reaching here means a model built
      // programmatically; keep it as a trivially satisfied equality by giving
      // it an unbounded slack.
      r.has_slack = true;
      r.rhs = 0.0;
      ++n_slack;
    }
  }

  // -- assign the slack and bound blocks -----------------------------------

  const std::size_t slack_begin = n_structural;
  const std::size_t bound_begin = slack_begin + n_slack;
  const std::size_t n_canon_cols = bound_begin + n_bound_rows;
  const std::size_t n_canon_rows = m0 + n_bound_rows;

  {
    std::size_t next_slack = slack_begin;
    std::size_t next_bound_row = m0;
    std::size_t next_bound_col = bound_begin;

    for (std::size_t i = 0; i < m0; ++i) {
      auto& r = rows[i];
      if (!r.has_slack) continue;
      r.slack_col = static_cast<Index>(next_slack++);
      if (r.slack_bounded) {
        r.slack_bound_row = static_cast<Index>(next_bound_row++);
        r.slack_bound_col = static_cast<Index>(next_bound_col++);
      }
    }
    for (std::size_t j = 0; j < n0; ++j) {
      auto& p = cols[j];
      if (!p.bounded) continue;
      p.bound_row = static_cast<Index>(next_bound_row++);
      p.bound_col = static_cast<Index>(next_bound_col++);
    }
  }

  // -- record the stack ----------------------------------------------------
  //
  // Recorded in application order; recovery replays it backwards.

  // `primary` is the ORIGINAL index and `secondary` the canonical one, so
  // recovery is a direct lookup rather than a positional walk. Every column
  // pushes exactly one record -- including untransformed ones, whose canonical
  // index is otherwise unknowable, since it depends on how many free variables
  // were split ahead of them.
  for (std::size_t j = 0; j < n0; ++j) {
    const auto& p = cols[j];
    const auto oj = static_cast<Index>(j);
    if (p.split) {
      // The partner is always the next canonical column, by construction.
      stack.push({TransformKind::SplitFreeVariable, oj, p.primary, 0.0, 0.0});
    } else if (p.scale < 0.0) {
      stack.push({TransformKind::NegateVariable, oj, p.primary, p.shift, 0.0});
    } else {
      stack.push({TransformKind::ShiftVariable, oj, p.primary, p.shift, 0.0});
    }
    if (p.bounded) {
      stack.push({TransformKind::AddBoundRow, oj, p.bound_row, p.width, 0.0});
    }
  }
  for (std::size_t i = 0; i < m0; ++i) {
    const auto& r = rows[i];
    const auto oi = static_cast<Index>(i);
    if (r.negate) stack.push({TransformKind::NegateRow, oi, kNone, 0.0, 0.0});
    if (r.has_slack) {
      stack.push({r.slack_bounded ? TransformKind::BoundedSlack
                                  : TransformKind::AddSlack,
                  oi, r.slack_col, r.rhs, r.slack_width});
    }
  }

  // -- objective under the affine map --------------------------------------
  //
  //     c_new  = D (Q t + c)
  //     offset = 1/2 t'Q t + c't
  //
  // with the sense flip applied to c and Q first, so the offset accumulates in
  // canonical (minimization) space.

  std::vector<Real> shift(n0, 0.0);
  for (std::size_t j = 0; j < n0; ++j) shift[j] = cols[j].shift;

  std::vector<Real> qt(n0, 0.0);
  if (!problem.Q.empty()) multiply(problem.Q, shift, qt);

  auto& cp = out.problem;
  cp.objective_negated = maximize;
  cp.num_structural = n_structural;
  cp.num_slack = n_slack;
  cp.num_bound = n_bound_rows;

  cp.c = core::RealVector(n_canon_cols, 0.0);
  Real offset = 0.0;
  for (std::size_t j = 0; j < n0; ++j) {
    const auto& p = cols[j];
    const Real cj = sense_sign * problem.c[j];
    const Real qtj = sense_sign * qt[j];

    offset += 0.5 * qtj * p.shift + cj * p.shift;

    const Real cnew = p.scale * (qtj + cj);
    cp.c[static_cast<std::size_t>(p.primary)] = cnew;
    if (p.split) cp.c[static_cast<std::size_t>(p.partner)] = -cnew;
  }
  cp.obj_offset = offset;

  // -- row right-hand sides, shifted by A*t --------------------------------

  std::vector<Real> at;
  multiply(problem.A, shift, at);

  cp.b = core::RealVector(n_canon_rows, 0.0);
  for (std::size_t i = 0; i < m0; ++i) {
    const auto& r = rows[i];
    const Real adjusted = r.negate ? (r.rhs + at[i]) : (r.rhs - at[i]);
    cp.b[static_cast<std::size_t>(r.canonical)] = adjusted;
  }
  for (std::size_t i = 0; i < m0; ++i) {
    const auto& r = rows[i];
    if (r.slack_bounded) {
      cp.b[static_cast<std::size_t>(r.slack_bound_row)] = r.slack_width;
    }
  }
  for (std::size_t j = 0; j < n0; ++j) {
    const auto& p = cols[j];
    if (p.bounded) cp.b[static_cast<std::size_t>(p.bound_row)] = p.width;
  }

  // -- constraint matrix ---------------------------------------------------
  //
  // Enumerated twice through one lambda -- once to count, once to fill -- so
  // SparseBuilder can size exactly with no reallocation.

  const auto enumerate = [&](auto&& emit) {
    const auto off = problem.A.csr.offsets();
    const auto idx = problem.A.csr.indices();
    const auto val = problem.A.csr.values();

    for (std::size_t i = 0; i < m0; ++i) {
      const auto& r = rows[i];
      const Real row_sign = r.negate ? -1.0 : 1.0;
      for (auto k = static_cast<std::size_t>(off[i]);
           k < static_cast<std::size_t>(off[i + 1]); ++k) {
        const auto j = static_cast<std::size_t>(idx[k]);
        const auto& p = cols[j];
        const Real v = row_sign * val[k] * p.scale;
        emit(r.canonical, p.primary, v);
        if (p.split) emit(r.canonical, p.partner, -v);
      }
      if (r.has_slack) emit(r.canonical, r.slack_col, 1.0);
      if (r.slack_bounded) {
        emit(r.slack_bound_row, r.slack_col, 1.0);
        emit(r.slack_bound_row, r.slack_bound_col, 1.0);
      }
    }
    for (std::size_t j = 0; j < n0; ++j) {
      const auto& p = cols[j];
      if (!p.bounded) continue;
      emit(p.bound_row, p.primary, 1.0);
      emit(p.bound_row, p.bound_col, 1.0);
    }
  };

  core::SparseBuilder builder(n_canon_rows, n_canon_cols);
  enumerate([&](Index r, Index c, Real) { builder.count(r, c); });
  if (auto st = builder.allocate(); !st.ok()) return st.error();
  enumerate([&](Index r, Index c, Real v) { builder.insert(r, c, v); });
  cp.A = builder.finish(true, 0.0);

  // -- quadratic objective under the affine map ----------------------------
  //
  //     Q_new[i][j] = d_i d_j Q[i][j]
  //
  // and a split column duplicates both its row and its column with a sign
  // flip, since xp and xm enter the quadratic form with opposite signs.

  if (!problem.Q.empty()) {
    const auto off = problem.Q.csr.offsets();
    const auto idx = problem.Q.csr.indices();
    const auto val = problem.Q.csr.values();

    const auto enumerate_q = [&](auto&& emit) {
      for (std::size_t i = 0; i < problem.Q.rows(); ++i) {
        const auto& pi = cols[i];
        for (auto k = static_cast<std::size_t>(off[i]);
             k < static_cast<std::size_t>(off[i + 1]); ++k) {
          const auto j = static_cast<std::size_t>(idx[k]);
          const auto& pj = cols[j];
          const Real v = sense_sign * val[k] * pi.scale * pj.scale;

          emit(pi.primary, pj.primary, v);
          if (pj.split) emit(pi.primary, pj.partner, -v);
          if (pi.split) emit(pi.partner, pj.primary, -v);
          if (pi.split && pj.split) emit(pi.partner, pj.partner, v);
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

  // Rebuild the column and row plans from the stack. Replaying the record is
  // what makes this the *inverse* of canonicalize() rather than a
  // reimplementation that can drift away from it.
  struct ColInv {
    Index primary = kNone;
    Index partner = kNone;
    Real scale = 1.0;
    Real shift = 0.0;
    bool split = false;
    Index bound_row = kNone;
  };
  std::vector<ColInv> cinv(n0);
  std::vector<bool> row_negated(m0, false);

  for (const auto& rec : transforms.records()) {
    if (rec.primary == kNone) continue;
    const auto k = static_cast<std::size_t>(rec.primary);
    switch (rec.kind) {
      case TransformKind::SplitFreeVariable:
        if (k >= n0) break;
        cinv[k].primary = rec.secondary;
        cinv[k].partner = rec.secondary + 1;  // assigned contiguously
        cinv[k].split = true;
        break;
      case TransformKind::NegateVariable:
        if (k >= n0) break;
        cinv[k].primary = rec.secondary;
        cinv[k].scale = -1.0;
        cinv[k].shift = rec.value;
        break;
      case TransformKind::ShiftVariable:
        if (k >= n0) break;
        cinv[k].primary = rec.secondary;
        cinv[k].scale = 1.0;
        cinv[k].shift = rec.value;
        break;
      case TransformKind::AddBoundRow:
        if (k >= n0) break;
        cinv[k].bound_row = rec.secondary;
        break;
      case TransformKind::NegateRow:
        if (k < m0) row_negated[k] = true;
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

  const auto& xc = canonical_solution.x;

  // -- primal --------------------------------------------------------------
  s.x = core::RealVector(n0, 0.0);
  for (std::size_t j = 0; j < n0; ++j) {
    const auto& p = cinv[j];
    const auto pj = static_cast<std::size_t>(p.primary);
    Real v = pj < xc.size() ? xc[pj] : 0.0;
    if (p.split) {
      const auto qj = static_cast<std::size_t>(p.partner);
      v -= (qj < xc.size() ? xc[qj] : 0.0);
    }
    // Invert x = d*x' + t.
    s.x[j] = p.scale * v + p.shift;
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
      // Distance to whichever bound is finite, so the reported slack means the
      // same thing it meant in the original model.
      if (is_finite_bound(original.row_upper[i])) {
        s.s[i] = original.row_upper[i] - act;
      } else if (is_finite_bound(original.row_lower[i])) {
        s.s[i] = act - original.row_lower[i];
      } else {
        s.s[i] = 0.0;
      }
    }
  }

  // -- duals ---------------------------------------------------------------
  //
  // A negated row flips the sign of its dual. A bound row carries what was
  // originally a bound multiplier, so its dual belongs in the column's reduced
  // cost rather than among the row duals.
  const auto& yc = canonical_solution.y;
  const auto& zc = canonical_solution.z;

  s.y = core::RealVector(m0, 0.0);
  for (std::size_t i = 0; i < m0; ++i) {
    const Real raw = i < yc.size() ? yc[i] : 0.0;
    s.y[i] = row_negated[i] ? -raw : raw;
  }

  s.z = core::RealVector(n0, 0.0);
  for (std::size_t j = 0; j < n0; ++j) {
    const auto& p = cinv[j];
    const auto pj = static_cast<std::size_t>(p.primary);
    Real z = pj < zc.size() ? zc[pj] : 0.0;
    if (p.bound_row != kNone) {
      const auto br = static_cast<std::size_t>(p.bound_row);
      z += br < yc.size() ? yc[br] : 0.0;
    }
    // A reflected column reverses the sense of its reduced cost.
    s.z[j] = p.scale * z;
    // A split column stands for a free variable, whose reduced cost is zero at
    // optimality.
    if (p.split) s.z[j] = 0.0;
  }

  s.w = core::RealVector(m0, 0.0);
  for (std::size_t i = 0; i < m0; ++i) {
    const Real raw = i < canonical_solution.w.size() ? canonical_solution.w[i] : 0.0;
    s.w[i] = raw;
  }

  // -- objective, back in the original sense -------------------------------
  {
    Real obj = canonical.objective(xc.span());
    obj += canonical.obj_offset;
    if (canonical.objective_negated) obj = -obj;
    s.objective = obj + original.obj_constant;
  }

  // Bound violations are measured against the ORIGINAL bounds: recovery
  // through shifts and reflections can introduce small violations that were
  // not present in the canonical point.
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
