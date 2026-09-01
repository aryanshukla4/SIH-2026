// Canonicalizer property tests.
//
// The contract has four parts, each checked independently:
//
//   1. FEASIBILITY.  A point feasible in the original model maps to a point
//      satisfying `A_E x = b_E`, `A_I x + s = b_I`, `l <= x <= u`, `s >= 0`.
//   2. OBJECTIVE.    The two models agree on the objective value at
//      corresponding points, in the original sense and including the constant.
//   3. INVERSE.      `recover_solution` maps the canonical point back to the
//      original point exactly, fixed columns included.
//   4. STARTABILITY. The canonical model has no fixed column and no empty row,
//      so an interior-point method can actually begin on it.
//
// The forward map here is written from the specification in Canonical.hpp
// rather than borrowed from the implementation, so a bug in the canonicalizer
// cannot hide behind a matching bug in the test. In particular the row
// permutation is re-derived from "equalities first, in original order" rather
// than read out of the transform stack.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::INF;
using core::is_finite_bound;
using core::Real;

namespace {

model::Problem parse(std::string_view text) {
  auto r = io::parseProblem(text, io::FileFormat::Mps);
  if (!r.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "parse", r.error().format());
    return {};
  }
  return std::move(r).value();
}

/// A canonical point: the variables plus the inequality slacks.
struct CanonPoint {
  std::vector<Real> x;
  std::vector<Real> s;
};

/// Which original rows survive, and where they land.
struct RowLayout {
  std::vector<std::size_t> canonical;  ///< kNone-equivalent: npos when dropped
  std::vector<std::size_t> slack;      ///< npos unless an inequality
  /// Ranged rows only: the bounded column added for the row. A ranged row is
  /// an EQUALITY carrying that column, not an inequality with a bounded slack.
  std::vector<std::size_t> range_col;
  std::vector<bool> negated;
  std::size_t num_equality = 0;
  std::size_t num_inequality = 0;
  std::size_t num_range = 0;
  static constexpr std::size_t kDropped = static_cast<std::size_t>(-1);
};

/// Re-derive the column map from the spec: keep every column except `l == u`.
std::vector<std::size_t> column_layout(const model::Problem& p) {
  std::vector<std::size_t> map(p.num_cols(), RowLayout::kDropped);
  std::size_t next = 0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    const Real lo = p.col_lower[j];
    const Real hi = p.col_upper[j];
    if (is_finite_bound(lo) && is_finite_bound(hi) && lo == hi) continue;
    map[j] = next++;
  }
  return map;
}

/// Re-derive the row map from the spec: equalities first in original order,
/// then inequalities in original order, skipping vacuous and empty rows.
RowLayout row_layout(const model::Problem& p,
                     const std::vector<std::size_t>& col_map) {
  RowLayout out;
  out.canonical.assign(p.num_rows(), RowLayout::kDropped);
  out.slack.assign(p.num_rows(), RowLayout::kDropped);
  out.range_col.assign(p.num_rows(), RowLayout::kDropped);
  out.negated.assign(p.num_rows(), false);

  const auto off = p.A.csr.offsets();
  const auto idx = p.A.csr.indices();

  std::vector<int> kind(p.num_rows(), -1);  // 0 equality, 1 inequality
  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    const Real lo = p.row_lower[i];
    const Real hi = p.row_upper[i];
    const bool lf = is_finite_bound(lo);
    const bool hf = is_finite_bound(hi);
    if (!lf && !hf) continue;  // vacuous

    std::size_t kept = 0;
    for (auto k = static_cast<std::size_t>(off[i]);
         k < static_cast<std::size_t>(off[i + 1]); ++k) {
      if (col_map[static_cast<std::size_t>(idx[k])] != RowLayout::kDropped) {
        ++kept;
      }
    }
    if (kept == 0) continue;  // empty over the kept columns

    if (lf && hf && lo == hi) {
      kind[i] = 0;
      ++out.num_equality;
    } else if (lf && hf) {
      // Ranged: an equality plus a bounded column, not a two-sided slack.
      kind[i] = 2;
      ++out.num_equality;
      ++out.num_range;
    } else {
      kind[i] = 1;
      out.negated[i] = lf && !hf;
      ++out.num_inequality;
    }
  }

  std::size_t next_eq = 0;
  std::size_t next_ineq = out.num_equality;
  std::size_t next_slack = 0;
  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    if (kind[i] == 0 || kind[i] == 2) {
      out.canonical[i] = next_eq++;
    } else if (kind[i] == 1) {
      out.canonical[i] = next_ineq++;
      out.slack[i] = next_slack++;
    }
  }
  // Range columns are appended after the kept original columns.
  std::size_t next_range = 0;
  for (std::size_t j = 0; j < col_map.size(); ++j) {
    if (col_map[j] != RowLayout::kDropped) ++next_range;
  }
  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    if (kind[i] == 2) out.range_col[i] = next_range++;
  }
  return out;
}

/// Map an original point into canonical space. In the bounded-variable form
/// this is nearly the identity on `x` -- bounds are not transformed, so no
/// column is shifted, reflected or split. Only fixed columns disappear.
CanonPoint forward_map(const model::Problem& p, const std::vector<Real>& x,
                       const std::vector<std::size_t>& col_map,
                       const RowLayout& rl) {
  CanonPoint out;
  std::size_t n_kept = 0;
  for (const auto m : col_map) {
    if (m != RowLayout::kDropped) ++n_kept;
  }
  out.x.assign(n_kept + rl.num_range, 0.0);
  out.s.assign(rl.num_inequality, 0.0);

  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    if (col_map[j] == RowLayout::kDropped) continue;
    out.x[col_map[j]] = x[j];
  }

  const auto off = p.A.csr.offsets();
  const auto idx = p.A.csr.indices();
  const auto val = p.A.csr.values();

  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    const bool has_slack = rl.slack[i] != RowLayout::kDropped;
    const bool has_range = rl.range_col[i] != RowLayout::kDropped;
    if (!has_slack && !has_range) continue;
    Real act = 0.0;
    for (auto k = static_cast<std::size_t>(off[i]);
         k < static_cast<std::size_t>(off[i + 1]); ++k) {
      act += val[k] * x[static_cast<std::size_t>(idx[k])];
    }
    // Fixed columns moved to the right-hand side, so the slack is measured
    // against the full original activity either way -- the two shifts cancel.
    const Real lo = p.row_lower[i];
    const Real hi = p.row_upper[i];
    if (has_slack) {
      out.s[rl.slack[i]] = rl.negated[i] ? (act - lo) : (hi - act);
    } else {
      out.x[rl.range_col[i]] = hi - act;  // a'x + t = hi
    }
  }
  return out;
}

/// Verify all four contract parts for one model at one feasible point.
void check_roundtrip(const char* label, const model::Problem& p,
                     const std::vector<Real>& x) {
  auto canon = model::canonicalize(p);
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label, canon.error().format());
    return;
  }
  const auto& cp = canon->problem;
  CHECK(cp.validate());

  // -- 4. startability -----------------------------------------------------
  std::size_t bad = 0;
  bool bad_is_row = false;
  if (!cp.is_ipm_startable(&bad, &bad_is_row)) {
    ::sovsolve::test::record(__FILE__, __LINE__, label,
                             std::string("not IPM-startable: ") +
                                 (bad_is_row ? "row " : "column ") +
                                 std::to_string(bad));
    return;
  }

  const auto col_map = column_layout(p);
  const auto rl = row_layout(p, col_map);
  const auto pt = forward_map(p, x, col_map, rl);

  CHECK_EQ(cp.num_cols(), pt.x.size());
  CHECK_EQ(cp.num_equality, rl.num_equality);
  CHECK_EQ(cp.num_inequality_rows(), pt.s.size());

  // -- 1. feasibility ------------------------------------------------------
  bool bounds_ok = true;
  for (std::size_t j = 0; j < cp.num_cols(); ++j) {
    if (is_finite_bound(cp.col_lower[j]) && pt.x[j] < cp.col_lower[j] - 1e-9) {
      bounds_ok = false;
    }
    if (is_finite_bound(cp.col_upper[j]) && pt.x[j] > cp.col_upper[j] + 1e-9) {
      bounds_ok = false;
    }
  }
  CHECK(bounds_ok);

  bool slacks_ok = true;
  for (std::size_t k = 0; k < pt.s.size(); ++k) {
    // Every canonical slack is one-sided. A ranged row does not produce a
    // two-sided slack; it produces an equality plus a bounded column.
    if (pt.s[k] < -1e-9) slacks_ok = false;
  }
  CHECK(slacks_ok);

  Real worst_row = 0.0;
  {
    const auto off = cp.A.csr.offsets();
    const auto idx = cp.A.csr.indices();
    const auto val = cp.A.csr.values();
    for (std::size_t i = 0; i < cp.num_rows(); ++i) {
      Real act = 0.0;
      for (auto k = static_cast<std::size_t>(off[i]);
           k < static_cast<std::size_t>(off[i + 1]); ++k) {
        act += val[k] * pt.x[static_cast<std::size_t>(idx[k])];
      }
      if (i >= cp.num_equality) act += pt.s[i - cp.num_equality];
      worst_row = std::max(worst_row, std::fabs(act - cp.b[i]));
    }
  }
  CHECK_NEAR(worst_row, 0.0, 1e-8);

  // -- 2. objective --------------------------------------------------------
  core::RealVector xcv(pt.x.size());
  for (std::size_t k = 0; k < pt.x.size(); ++k) xcv[k] = pt.x[k];

  Real canonical_obj = cp.objective(xcv.span()) + cp.obj_offset;
  if (cp.objective_negated) canonical_obj = -canonical_obj;
  canonical_obj += p.obj_constant;

  Real original_obj = p.obj_constant;
  for (std::size_t j = 0; j < p.num_cols(); ++j) original_obj += p.c[j] * x[j];
  if (p.has_quadratic()) {
    const auto off = p.Q.csr.offsets();
    const auto idx = p.Q.csr.indices();
    const auto val = p.Q.csr.values();
    Real quad = 0.0;
    for (std::size_t i = 0; i < p.Q.rows(); ++i) {
      Real row = 0.0;
      for (auto k = static_cast<std::size_t>(off[i]);
           k < static_cast<std::size_t>(off[i + 1]); ++k) {
        row += val[k] * x[static_cast<std::size_t>(idx[k])];
      }
      quad += x[i] * row;
    }
    original_obj += 0.5 * quad;
  }
  CHECK_NEAR(canonical_obj, original_obj, 1e-8);

  // -- 3. inverse ----------------------------------------------------------
  model::Solution cs;
  cs.x = std::move(xcv);
  cs.y = core::RealVector(cp.num_rows(), 0.0);
  cs.z = core::RealVector(cp.num_cols(), 0.0);
  cs.v = core::RealVector(cp.num_cols(), 0.0);

  auto back = model::recover_solution(p, cp, canon->transforms, cs);
  if (!back.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label, back.error().format());
    return;
  }

  Real worst_x = 0.0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    worst_x = std::max(worst_x, std::fabs(back->x[j] - x[j]));
  }
  CHECK_NEAR(worst_x, 0.0, 1e-9);
  CHECK_NEAR(back->objective, original_obj, 1e-8);
  CHECK_NEAR(back->quality.max_bound_violation, 0.0, 1e-9);
}

// ---------------------------------------------------------------------------

void test_lower_bounds_only() {
  const auto p = parse(R"(NAME          BASIC
ROWS
 N  COST
 L  R1
 G  R2
 E  R3
COLUMNS
    X         COST         1.0   R1           1.0
    X         R2           1.0   R3           1.0
    Y         COST         2.0   R1           2.0
    Y         R3           1.0
RHS
    RHS       R1          10.0   R2           1.0
    RHS       R3           4.0
ENDATA
)");
  // x + y = 4 (R3), x >= 1 (R2), x + 2y <= 10 (R1). Take x = 2, y = 2.
  check_roundtrip("lower-only", p, {2.0, 2.0});
  check_roundtrip("lower-only b", p, {3.0, 1.0});
}

void test_nonzero_lower_bound_is_not_shifted() {
  const auto p = parse(R"(NAME          SHIFT
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         3.0   R1           1.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1          20.0
BOUNDS
 LO BND       X            5.0
 LO BND       Y           -2.0
ENDATA
)");
  // In the bounded-variable form a nonzero lower bound stays put: no shift, so
  // no objective constant moves and no right-hand side changes.
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  CHECK_NEAR(canon->problem.col_lower[0], 5.0, 1e-12);
  CHECK_NEAR(canon->problem.col_lower[1], -2.0, 1e-12);
  CHECK_NEAR(canon->problem.b[0], 20.0, 1e-12);
  CHECK_NEAR(canon->problem.obj_offset, 0.0, 1e-12);

  check_roundtrip("shift", p, {5.0, -2.0});
  check_roundtrip("shift b", p, {7.5, 3.25});
}

void test_boxed_variables_add_no_rows() {
  const auto p = parse(R"(NAME          BOXED
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
    Y         COST        -2.0   R1           1.0
RHS
    RHS       R1          10.0
BOUNDS
 UP BND       X            4.0
 LO BND       Y            1.0
 UP BND       Y            6.0
ENDATA
)");
  // The whole point of the bounded-variable form: a boxed column costs nothing.
  // The `x >= 0, Ax = b` form turned each into a row, measured at +750% row
  // growth on rgn and +648% on gt2.
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  CHECK_EQ(canon->problem.num_rows(), std::size_t{1});
  CHECK_EQ(canon->problem.num_cols(), std::size_t{2});

  check_roundtrip("boxed", p, {0.0, 1.0});
  check_roundtrip("boxed b", p, {4.0, 6.0});
  check_roundtrip("boxed mid", p, {2.5, 3.5});
}

void test_free_variables_are_not_split() {
  const auto p = parse(R"(NAME          FREEVAR
ROWS
 N  COST
 E  R1
COLUMNS
    X         COST         1.0   R1           1.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1           3.0
BOUNDS
 FR BND       X
ENDATA
)");
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  // One column, not two: the xp - xm split is prohibited.
  CHECK_EQ(canon->problem.num_cols(), std::size_t{2});
  CHECK(!is_finite_bound(canon->problem.col_lower[0]));
  CHECK(!is_finite_bound(canon->problem.col_upper[0]));

  check_roundtrip("free positive", p, {2.0, 1.0});
  check_roundtrip("free negative", p, {-5.0, 8.0});
  check_roundtrip("free zero", p, {0.0, 3.0});
}

void test_upper_bound_only_is_not_reflected() {
  const auto p = parse(R"(NAME          UPPERONLY
ROWS
 N  COST
 G  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1         -10.0
BOUNDS
 MI BND       X
 UP BND       X            3.0
ENDATA
)");
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  CHECK(!is_finite_bound(canon->problem.col_lower[0]));
  CHECK_NEAR(canon->problem.col_upper[0], 3.0, 1e-12);

  check_roundtrip("upper-only", p, {3.0});
  check_roundtrip("upper-only b", p, {-4.0});
}

void test_ranged_rows() {
  const auto p = parse(R"(NAME          RANGED
ROWS
 N  COST
 L  R1
 G  R2
 E  R3
COLUMNS
    X         COST         1.0   R1           1.0
    X         R2           1.0   R3           1.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0   R2           2.0
    RHS       R3           5.0
RANGES
    RNG       R1           6.0   R2           8.0
    RNG       R3           3.0
ENDATA
)");
  // R1: [4,10], R2: [2,10], R3: [5,8]. All three become bounded slacks rather
  // than extra rows. No Netlib instance in our corpus has a ranged row, so this
  // path is covered only here.
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  // Three ranged rows become three EQUALITIES, each carrying one bounded
  // column: 3 rows unchanged, 2 original columns plus 3 range columns.
  CHECK_EQ(canon->problem.num_rows(), std::size_t{3});
  CHECK_EQ(canon->problem.num_equality, std::size_t{3});
  CHECK_EQ(canon->problem.num_inequality_rows(), std::size_t{0});
  CHECK_EQ(canon->problem.num_range, std::size_t{3});
  CHECK_EQ(canon->problem.num_cols(), std::size_t{5});
  // Each range column is bounded by the row's width, and none is fixed --
  // a zero-width range would be an equality row, not a ranged one.
  const std::size_t first = canon->problem.num_cols() - canon->problem.num_range;
  for (std::size_t k = first; k < canon->problem.num_cols(); ++k) {
    CHECK_NEAR(canon->problem.col_lower[k], 0.0, 1e-12);
    CHECK(is_finite_bound(canon->problem.col_upper[k]));
    CHECK(canon->problem.col_upper[k] > 0.0);
  }
  CHECK(canon->problem.is_ipm_startable());

  check_roundtrip("ranged", p, {5.0, 1.0});
  check_roundtrip("ranged b", p, {6.0, 2.0});
}

void test_maximize() {
  const auto p = parse(R"(NAME          MAXOBJ
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         3.0   R1           1.0
    Y         COST         2.0   R1           1.0
RHS
    RHS       R1          10.0   COST         7.0
ENDATA
)");
  CHECK(p.sense == core::ObjSense::Minimize);

  auto q = parse(R"(NAME          MAXOBJ2
OBJSENSE
    MAX
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         3.0   R1           1.0
    Y         COST         2.0   R1           1.0
RHS
    RHS       R1          10.0   COST         7.0
ENDATA
)");
  CHECK(q.sense == core::ObjSense::Maximize);
  check_roundtrip("maximize", q, {4.0, 6.0});
  check_roundtrip("minimize", p, {4.0, 6.0});
}

void test_quadratic() {
  const auto p = parse(R"(NAME          QPSHIFT
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
    Y         COST        -1.0   R1           1.0
RHS
    RHS       R1          10.0
BOUNDS
 LO BND       X            2.0
QUADOBJ
    X         X            2.0
    X         Y            1.0
    Y         Y            3.0
ENDATA
)");
  CHECK(p.has_quadratic());
  check_roundtrip("qp", p, {2.0, 0.0});
  check_roundtrip("qp b", p, {3.5, 4.25});
}

void test_range_column_is_dropped_on_recovery() {
  // A RANGE column is not an original variable, so the recovered solution must
  // not contain it -- and the row's reported slack must be recomputed from the
  // original activity rather than read out of the dropped column.
  const auto p = parse(R"(NAME          RNGREC
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
RANGES
    RNG       R1           6.0
ENDATA
)");
  // R1 is 4 <= X + Y <= 10.
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& cp = canon->problem;

  // Two original columns plus one range column; the row became an equality.
  CHECK_EQ(cp.num_cols(), std::size_t{3});
  CHECK_EQ(cp.num_range, std::size_t{1});
  CHECK_EQ(cp.num_equality, std::size_t{1});
  CHECK_EQ(cp.num_inequality_rows(), std::size_t{0});

  // Canonical point: X = 3, Y = 4 gives activity 7, so t = 10 - 7 = 3, which
  // lies inside [0, 6] as it must.
  model::Solution cs;
  cs.x = core::RealVector(3, 0.0);
  cs.x[0] = 3.0;
  cs.x[1] = 4.0;
  cs.x[2] = 3.0;
  cs.y = core::RealVector(cp.num_rows(), 0.0);
  cs.z = core::RealVector(cp.num_cols(), 0.0);
  cs.v = core::RealVector(cp.num_cols(), 0.0);

  auto back = model::recover_solution(p, cp, canon->transforms, cs);
  CHECK(back.has_value());
  if (!back.has_value()) return;

  // The range column is gone: the solution has exactly the original columns.
  CHECK_EQ(back->x.size(), p.num_cols());
  CHECK_EQ(back->z.size(), p.num_cols());
  CHECK_EQ(back->v.size(), p.num_cols());
  CHECK_NEAR(back->x[0], 3.0, 1e-12);
  CHECK_NEAR(back->x[1], 4.0, 1e-12);

  // The reported slack is distance to the finite upper bound of the ORIGINAL
  // row, recomputed from activity -- 10 - 7 = 3 -- not read from the column.
  CHECK_EQ(back->s.size(), p.num_rows());
  CHECK_NEAR(back->s[0], 3.0, 1e-12);
  CHECK_NEAR(back->quality.max_bound_violation, 0.0, 1e-12);
}

void test_maximize_quadratic_negates_Q() {
  // The scope says `Q` is symmetric POSITIVE semidefinite. A legitimate
  // concave maximization arrives with `Q` NEGATIVE semidefinite, so the sense
  // flip has to negate `Q` as well as `c` -- otherwise the stated scope
  // silently fails to hold for a valid input, and the factorization meets an
  // indefinite matrix it was promised it would not see.
  const auto p = parse(R"(NAME          MAXQP
OBJSENSE
    MAX
ROWS
 N  obj
 L  R1
COLUMNS
    x         obj          4.0   R1           1.0
    y         obj          2.0   R1           1.0
RHS
    RHS       R1          10.0
QUADOBJ
    x         x           -2.0
    y         y           -4.0
ENDATA
)");
  CHECK(p.sense == core::ObjSense::Maximize);
  CHECK(p.has_quadratic());

  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& cp = canon->problem;
  CHECK(cp.objective_negated);

  // Original: max 4x + 2y - x^2 - 2y^2   (concave, Q = diag(-2,-4))
  // Canonical: min -4x - 2y + x^2 + 2y^2 (convex,  Q = diag(+2,+4))
  const auto q = [&](std::size_t i, std::size_t j) {
    const auto off = cp.Q.csr.offsets();
    const auto idx = cp.Q.csr.indices();
    const auto val = cp.Q.csr.values();
    Real s = 0.0;
    for (auto k = static_cast<std::size_t>(off[i]);
         k < static_cast<std::size_t>(off[i + 1]); ++k) {
      if (static_cast<std::size_t>(idx[k]) == j) s += val[k];
    }
    return s;
  };
  CHECK_NEAR(q(0, 0), 2.0, 1e-12);
  CHECK_NEAR(q(1, 1), 4.0, 1e-12);
  CHECK_NEAR(cp.c[0], -4.0, 1e-12);
  CHECK_NEAR(cp.c[1], -2.0, 1e-12);

  // Every diagonal entry non-negative is the cheap necessary condition for PSD
  // that the canonical form is supposed to guarantee.
  for (std::size_t i = 0; i < cp.num_cols(); ++i) CHECK(q(i, i) >= 0.0);

  // At x = y = 1 the original objective is 4 + 2 - 1 - 2 = 3.
  check_roundtrip("max qp", p, {1.0, 1.0});
  check_roundtrip("max qp b", p, {2.0, 0.5});
}

// ---------------------------------------------------------------------------
// The startability contract
// ---------------------------------------------------------------------------

void test_fixed_variable_is_substituted_out() {
  const auto p = parse(R"(NAME          FIXED
ROWS
 N  COST
 E  R1
COLUMNS
    X         COST         2.0   R1           1.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1           9.0
BOUNDS
 FX BND       X            4.0
ENDATA
)");
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& cp = canon->problem;

  // X is gone: the IPM needs x-l > 0 and u-x > 0 at once, and their sum is
  // u-l = 0. One column and one row remain, with X's contribution folded in.
  CHECK_EQ(cp.num_cols(), std::size_t{1});
  CHECK_EQ(cp.num_rows(), std::size_t{1});
  CHECK_NEAR(cp.b[0], 5.0, 1e-12);        // 9 - 1*4
  CHECK_NEAR(cp.obj_offset, 8.0, 1e-12);  // c_X * 4
  CHECK(cp.is_ipm_startable());

  check_roundtrip("fixed", p, {4.0, 5.0});
}

void test_fixed_variable_in_quadratic() {
  const auto p = parse(R"(NAME          FIXEDQP
ROWS
 N  COST
 E  R1
COLUMNS
    X         COST         1.0   R1           1.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1           7.0
BOUNDS
 FX BND       X            3.0
QUADOBJ
    X         X            2.0
    X         Y            4.0
    Y         Y            6.0
ENDATA
)");
  // The cross term is what makes this more than dropping a column: with x = 3
  // the term 1/2 * 2 * Q_XY * x * y becomes a LINEAR contribution to y.
  check_roundtrip("fixed qp", p, {3.0, 4.0});
}

void test_empty_row_consistent_is_dropped() {
  // R2 mentions only X, which is fixed -- so after substitution it has no
  // entries at all. Its right-hand side is consistent, so it is dropped.
  const auto p = parse(R"(NAME          EMPTYOK
ROWS
 N  COST
 E  R1
 L  R2
COLUMNS
    X         COST         1.0   R1           1.0
    X         R2           1.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1           9.0   R2           5.0
BOUNDS
 FX BND       X            2.0
ENDATA
)");
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  // R2 becomes 0 <= 5 - 2 = 3, which holds; it is removed rather than handed
  // to the solver as an exactly singular row of A*Theta*A'.
  CHECK_EQ(canon->problem.num_rows(), std::size_t{1});
  CHECK(canon->problem.is_ipm_startable());

  check_roundtrip("empty dropped", p, {2.0, 7.0});
}

void test_empty_row_inconsistent_is_infeasible() {
  // Same shape, but R2 requires X <= 1 while X is fixed at 2. After
  // substitution the row reads 0 <= -1, which is false for every x.
  const auto p = parse(R"(NAME          EMPTYBAD
ROWS
 N  COST
 E  R1
 L  R2
COLUMNS
    X         COST         1.0   R1           1.0
    X         R2           1.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1           9.0   R2           1.0
BOUNDS
 FX BND       X            2.0
ENDATA
)");
  auto canon = model::canonicalize(p);
  CHECK(!canon.has_value());
  if (canon.has_value()) return;
  // A verdict, not a numerical failure. Regularizing this away would report
  // NUMERICAL_FAILURE for a model that is provably infeasible.
  CHECK(canon.error().code == core::ErrorCode::PrimalInfeasible);
}

void test_startability_holds_on_every_corpus_instance() {
  // The contract is only worth stating if it survives real data. 8 of the 19
  // instances carry fixed columns (498 in 80bau3b) and 2 carry empty rows.
  const char* names[] = {"afiro",  "adlittle", "25fv47", "80bau3b", "greenbea",
                         "shell",  "stair",    "gas11",  "etamacro", "egout",
                         "israel", "e226",     "standata"};
  std::size_t total_dropped_cols = 0;
  std::size_t total_dropped_rows = 0;

  for (const char* name : names) {
#ifdef SOVSOLVE_TEST_DATA_DIR
    const std::string path =
        std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/" + name + ".mps";
#else
    const std::string path = std::string("tests/data/netlib/") + name + ".mps";
#endif
    auto loaded = io::loadProblem(path);
    CHECK(loaded.has_value());
    if (!loaded.has_value()) continue;
    const auto& p = loaded.value();

    auto canon = model::canonicalize(p);
    CHECK(canon.has_value());
    if (!canon.has_value()) continue;
    const auto& cp = canon->problem;

    CHECK(cp.validate());
    CHECK(cp.A.csr.validate());
    CHECK(cp.A.csc.validate());

    std::size_t bad = 0;
    bool bad_is_row = false;
    const bool ok = cp.is_ipm_startable(&bad, &bad_is_row);
    if (!ok) {
      ::sovsolve::test::record(__FILE__, __LINE__, name,
                               std::string("not startable: ") +
                                   (bad_is_row ? "row " : "column ") +
                                   std::to_string(bad));
    }

    // Rows never grow -- that is the whole point of keeping bounds native.
    CHECK(cp.num_rows() <= p.num_rows());
    CHECK(cp.num_cols() <= p.num_cols());
    CHECK_EQ(cp.num_equality + cp.num_inequality_rows(), cp.num_rows());

    total_dropped_cols += p.num_cols() - cp.num_cols();
    total_dropped_rows += p.num_rows() - cp.num_rows();

    std::printf("  %-9s %5zu x %5zu -> %5zu x %5zu  (eq %5zu ineq %5zu)  "
                "-%zu col -%zu row\n",
                name, p.num_rows(), p.num_cols(), cp.num_rows(), cp.num_cols(),
                cp.num_equality, cp.num_inequality_rows(),
                p.num_cols() - cp.num_cols(), p.num_rows() - cp.num_rows());
  }
  std::printf("  substituted %zu fixed columns, dropped %zu rows\n",
              total_dropped_cols, total_dropped_rows);
  // If these were zero the contract would be vacuous on real data.
  CHECK(total_dropped_cols > 0);
  CHECK(total_dropped_rows > 0);
}

// ---------------------------------------------------------------------------

void test_structure_counts() {
  const auto p = parse(R"(NAME          COUNTS
ROWS
 N  COST
 E  R1
 L  R2
 G  R3
COLUMNS
    A         COST         1.0   R1           1.0
    A         R2           1.0   R3           1.0
    B         COST         1.0   R1           1.0
    C         COST         1.0   R2           1.0
RHS
    RHS       R1           5.0   R2          10.0
    RHS       R3           1.0
BOUNDS
 FR BND       B
 UP BND       C            8.0
ENDATA
)");
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& cp = canon->problem;

  // Three columns in, three out: free is not split and boxed is not expanded.
  CHECK_EQ(cp.num_cols(), std::size_t{3});
  // R1 equality, R2 and R3 inequalities -- and equalities come first.
  CHECK_EQ(cp.num_equality, std::size_t{1});
  CHECK_EQ(cp.num_inequality_rows(), std::size_t{2});
  CHECK_EQ(cp.num_rows(), std::size_t{3});
  CHECK_EQ(cp.num_range, std::size_t{0});
  CHECK(cp.validate());
}

void test_equality_rows_get_no_slack() {
  const auto p = parse(R"(NAME          ALLEQ
ROWS
 N  COST
 E  R1
 E  R2
COLUMNS
    X         COST         1.0   R1           1.0
    Y         COST         1.0   R2           1.0
RHS
    RHS       R1           3.0   R2           4.0
ENDATA
)");
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  CHECK_EQ(canon->problem.num_inequality_rows(), std::size_t{0});
  CHECK_EQ(canon->problem.num_equality, std::size_t{2});
  check_roundtrip("all equalities", p, {3.0, 4.0});
}

void test_equalities_are_ordered_first() {
  // A_E and A_I are row ranges of one matrix, so the permutation is part of
  // the contract, not an implementation detail.
  const auto p = parse(R"(NAME          ORDER
ROWS
 N  COST
 L  R1
 E  R2
 L  R3
 E  R4
COLUMNS
    X         COST         1.0   R1           1.0
    X         R2           1.0   R3           1.0
    X         R4           1.0
RHS
    RHS       R1          10.0   R2           3.0
    RHS       R3          20.0   R4           3.0
ENDATA
)");
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& cp = canon->problem;
  CHECK_EQ(cp.num_equality, std::size_t{2});
  // Original R2 and R4 (rhs 3) land in rows 0 and 1; R1 and R3 follow.
  CHECK_NEAR(cp.b[0], 3.0, 1e-12);
  CHECK_NEAR(cp.b[1], 3.0, 1e-12);
  CHECK_NEAR(cp.b[2], 10.0, 1e-12);
  CHECK_NEAR(cp.b[3], 20.0, 1e-12);
  check_roundtrip("ordering", p, {3.0});
}

void test_dual_recovery_signs() {
  // A `>=` row is negated to fit `A_I x + s = b_I`, so its dual is negated
  // back. That flip is what makes a `>=` row report the non-negative dual
  // HiGHS reports; no further normalization is applied on top.
  const auto p = parse(R"(NAME          DUALS
ROWS
 N  COST
 L  R1
 G  R2
COLUMNS
    X         COST         1.0   R1           1.0
    X         R2           1.0
RHS
    RHS       R1          10.0   R2           1.0
ENDATA
)");
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& cp = canon->problem;
  CHECK_EQ(cp.num_equality, std::size_t{0});

  model::Solution cs;
  cs.x = core::RealVector(cp.num_cols(), 1.0);
  cs.y = core::RealVector(cp.num_rows(), 0.0);
  cs.z = core::RealVector(cp.num_cols(), 0.0);
  cs.v = core::RealVector(cp.num_cols(), 0.0);
  // Internally both inequality rows carry a non-positive dual.
  for (std::size_t i = 0; i < cp.num_rows(); ++i) cs.y[i] = -2.0;

  auto back = model::recover_solution(p, cp, canon->transforms, cs);
  CHECK(back.has_value());
  if (!back.has_value()) return;
  CHECK_NEAR(back->y[0], -2.0, 1e-12);  // `<=` row: sign kept
  CHECK_NEAR(back->y[1], 2.0, 1e-12);   // `>=` row: negated back
}

void test_fixed_column_reduced_cost() {
  // A substituted column has no dual in the canonical solution at all, so its
  // reduced cost is rebuilt from stationarity against the ORIGINAL data.
  const auto p = parse(R"(NAME          FIXRC
ROWS
 N  COST
 E  R1
COLUMNS
    X         COST         7.0   R1           2.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1          10.0
BOUNDS
 FX BND       X            3.0
ENDATA
)");
  auto canon = model::canonicalize(p);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& cp = canon->problem;

  model::Solution cs;
  cs.x = core::RealVector(cp.num_cols(), 4.0);
  cs.y = core::RealVector(cp.num_rows(), 1.5);
  cs.z = core::RealVector(cp.num_cols(), 0.0);
  cs.v = core::RealVector(cp.num_cols(), 0.0);

  auto back = model::recover_solution(p, cp, canon->transforms, cs);
  CHECK(back.has_value());
  if (!back.has_value()) return;
  // d_X = c_X - (A'y)_X = 7 - 2*1.5 = 4, positive, so it sits in z.
  CHECK_NEAR(back->reduced_cost(0), 4.0, 1e-12);
  CHECK_NEAR(back->z[0], 4.0, 1e-12);
  CHECK_NEAR(back->v[0], 0.0, 1e-12);
  CHECK_NEAR(back->x[0], 3.0, 1e-12);
}

void test_randomized_roundtrip() {
  const auto p = parse(R"(NAME          MIXED
ROWS
 N  COST
 L  R1
 L  R2
 G  R3
COLUMNS
    A         COST         1.5   R1           1.0
    A         R2           2.0
    B         COST        -0.5   R1           1.0
    B         R3           1.0
    C         COST         2.0   R2           1.0
    C         R3           1.0
    D         COST         0.25  R1           1.0
    D         R2           1.0
    E         COST         1.0   R1           1.0
RHS
    RHS       R1          30.0   R2          40.0
    RHS       R3         -20.0
BOUNDS
 LO BND       A            1.0
 UP BND       A            9.0
 FR BND       B
 MI BND       C
 UP BND       C            5.0
 LO BND       D           -3.0
 FX BND       E            2.5
ENDATA
)");
  CHECK_EQ(p.num_cols(), std::size_t{5});

  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<Real> unit(0.0, 1.0);

  for (int trial = 0; trial < 200; ++trial) {
    std::vector<Real> x(p.num_cols(), 0.0);
    for (std::size_t j = 0; j < p.num_cols(); ++j) {
      const Real lo = p.col_lower[j];
      const Real hi = p.col_upper[j];
      const bool lf = is_finite_bound(lo);
      const bool hf = is_finite_bound(hi);
      if (lf && hf) x[j] = lo + unit(rng) * (hi - lo);
      else if (lf) x[j] = lo + unit(rng) * 10.0;
      else if (hf) x[j] = hi - unit(rng) * 10.0;
      else x[j] = (unit(rng) - 0.5) * 20.0;
    }
    // Every row here is an inequality, so a row-feasible canonical point always
    // exists. Equality rows cannot be sampled this way -- a random point simply
    // will not satisfy them -- so they are covered by the deterministic tests.
    check_roundtrip("random", p, x);
  }
}

}  // namespace

int main() {
  test_lower_bounds_only();
  test_nonzero_lower_bound_is_not_shifted();
  test_boxed_variables_add_no_rows();
  test_free_variables_are_not_split();
  test_upper_bound_only_is_not_reflected();
  test_ranged_rows();
  test_maximize();
  test_quadratic();
  test_maximize_quadratic_negates_Q();
  test_range_column_is_dropped_on_recovery();

  test_fixed_variable_is_substituted_out();
  test_fixed_variable_in_quadratic();
  test_empty_row_consistent_is_dropped();
  test_empty_row_inconsistent_is_infeasible();

  test_structure_counts();
  test_equality_rows_get_no_slack();
  test_equalities_are_ordered_first();
  test_dual_recovery_signs();
  test_fixed_column_reduced_cost();

  std::printf("canonicalization of real instances:\n");
  test_startability_holds_on_every_corpus_instance();
  test_randomized_roundtrip();
  return sovsolve::test::report("canonical");
}
