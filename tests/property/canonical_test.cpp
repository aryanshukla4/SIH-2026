// Canonicalizer property tests.
//
// The contract has three parts, and each is checked independently:
//
//   1. FEASIBILITY.  A point feasible in the original model maps to a point
//      satisfying `Ax = b, x >= 0` in the canonical model.
//   2. OBJECTIVE.    The two models agree on the objective value at
//      corresponding points, in the original sense and including the constant.
//   3. INVERSE.      `recover_solution` maps the canonical point back to the
//      original point exactly.
//
// The forward map used here is written from the specification in
// Canonical.hpp rather than borrowed from the implementation, so a bug in the
// canonicalizer cannot hide behind a matching bug in the test.

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

/// Map an original point to the canonical space, following Canonical.hpp:
///
///     boxed / lower-only :  x' = x - l        (and boxed adds t = w - x')
///     upper-only         :  x' = u - x
///     free               :  x = xp - xm
///
/// then slacks for inequality rows and auxiliaries for bound rows.
std::vector<Real> forward_map(const model::Problem& p,
                              const model::CanonicalProblem& cp,
                              const std::vector<Real>& x) {
  std::vector<Real> xc(cp.num_cols(), 0.0);

  std::size_t next = 0;
  std::size_t next_bound_col = cp.bound_begin();
  std::vector<std::size_t> bound_col_of(p.num_cols(), 0);
  std::vector<bool> has_bound(p.num_cols(), false);

  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    const Real lo = p.col_lower[j];
    const Real hi = p.col_upper[j];
    const bool lf = is_finite_bound(lo);
    const bool hf = is_finite_bound(hi);

    if (lf && hf) {
      xc[next] = x[j] - lo;
      has_bound[j] = true;
      bound_col_of[j] = 0;  // filled in the second bound pass below
      ++next;
    } else if (lf) {
      xc[next++] = x[j] - lo;
    } else if (hf) {
      xc[next++] = hi - x[j];
    } else {
      // Split: put the whole magnitude in whichever half keeps both parts
      // non-negative.
      xc[next] = x[j] > 0 ? x[j] : 0.0;
      xc[next + 1] = x[j] < 0 ? -x[j] : 0.0;
      next += 2;
    }
  }

  // Slacks, in row order over inequality rows only.
  std::size_t next_slack = cp.slack_begin();
  std::vector<std::size_t> slack_of(p.num_rows(), 0);
  std::vector<bool> has_slack(p.num_rows(), false);
  std::vector<bool> ranged(p.num_rows(), false);

  const auto off = p.A.csr.offsets();
  const auto idx = p.A.csr.indices();
  const auto val = p.A.csr.values();

  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    const Real lo = p.row_lower[i];
    const Real hi = p.row_upper[i];
    const bool lf = is_finite_bound(lo);
    const bool hf = is_finite_bound(hi);
    if (lf && hf && lo == hi) continue;  // equality: no slack

    Real act = 0.0;
    for (auto k = static_cast<std::size_t>(off[i]);
         k < static_cast<std::size_t>(off[i + 1]); ++k) {
      act += val[k] * x[static_cast<std::size_t>(idx[k])];
    }

    has_slack[i] = true;
    slack_of[i] = next_slack;
    if (lf && hf) {
      xc[next_slack] = hi - act;   // a'x + s = u
      ranged[i] = true;
    } else if (hf) {
      xc[next_slack] = hi - act;
    } else if (lf) {
      xc[next_slack] = act - lo;   // row negated: -a'x + s = -l
    } else {
      xc[next_slack] = 0.0;
    }
    ++next_slack;
  }

  // Bound-row auxiliaries: ranged-row slacks first, then boxed columns --
  // matching the assignment order in canonicalize().
  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    if (!ranged[i]) continue;
    const Real width = p.row_upper[i] - p.row_lower[i];
    xc[next_bound_col++] = width - xc[slack_of[i]];
  }
  std::size_t col_cursor = 0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    const bool lf = is_finite_bound(p.col_lower[j]);
    const bool hf = is_finite_bound(p.col_upper[j]);
    const std::size_t primary = col_cursor;
    col_cursor += (!lf && !hf) ? 2 : 1;
    if (!(lf && hf)) continue;
    const Real width = p.col_upper[j] - p.col_lower[j];
    xc[next_bound_col++] = width - xc[primary];
  }

  return xc;
}

/// Verify all three contract parts for one model at one feasible point.
void check_roundtrip(const char* label, const model::Problem& p,
                     const std::vector<Real>& x) {
  auto canon = model::canonicalize(p);
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label,
                             canon.error().format());
    return;
  }
  const auto& cp = canon->problem;
  CHECK(cp.validate());

  const auto xc = forward_map(p, cp, x);

  // -- 1. feasibility ------------------------------------------------------
  bool nonneg = true;
  for (const Real v : xc) {
    if (v < -1e-9) nonneg = false;
  }
  CHECK(nonneg);

  Real worst_row = 0.0;
  {
    const auto off = cp.A.csr.offsets();
    const auto idx = cp.A.csr.indices();
    const auto val = cp.A.csr.values();
    for (std::size_t i = 0; i < cp.num_rows(); ++i) {
      Real act = 0.0;
      for (auto k = static_cast<std::size_t>(off[i]);
           k < static_cast<std::size_t>(off[i + 1]); ++k) {
        act += val[k] * xc[static_cast<std::size_t>(idx[k])];
      }
      worst_row = std::max(worst_row, std::fabs(act - cp.b[i]));
    }
  }
  CHECK_NEAR(worst_row, 0.0, 1e-8);

  // -- 2. objective --------------------------------------------------------
  core::RealVector xcv(xc.size());
  for (std::size_t k = 0; k < xc.size(); ++k) xcv[k] = xc[k];

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
  cs.w = core::RealVector(cp.num_rows(), 0.0);

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
  // The simplest shape: every column x >= 0, mixed row senses.
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

void test_shifted_lower_bound() {
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
  // Nonzero lower bounds, one of them negative -- the shift has to move the
  // objective constant as well as the row bounds.
  check_roundtrip("shift", p, {5.0, -2.0});
  check_roundtrip("shift b", p, {7.5, 3.25});
}

void test_boxed_variables() {
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
  // Each boxed column costs one extra row and one extra column.
  check_roundtrip("boxed", p, {0.0, 1.0});
  check_roundtrip("boxed b", p, {4.0, 6.0});
  check_roundtrip("boxed mid", p, {2.5, 3.5});
}

void test_free_variables() {
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
  // A free column splits into xp - xm, so both signs must round-trip.
  check_roundtrip("free positive", p, {2.0, 1.0});
  check_roundtrip("free negative", p, {-5.0, 8.0});
  check_roundtrip("free zero", p, {0.0, 3.0});
}

void test_upper_bound_only() {
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
  // Reflected column: x' = u - x. The reduced cost reverses sense with it.
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
  // R1: [4,10], R2: [2,10], R3: [5,8]. Ranged slacks get their own bound rows.
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
  // The sense flip and the objective constant must both survive: the reported
  // objective is in the ORIGINAL sense.
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
  // A shifted column in a QP moves the linear term (c += Q*t) as well as the
  // constant -- the cross term is what makes this more than a translation.
  check_roundtrip("qp shift", p, {2.0, 0.0});
  check_roundtrip("qp shift b", p, {3.5, 4.25});
}

void test_fixed_variable() {
  const auto p = parse(R"(NAME          FIXED
ROWS
 N  COST
 E  R1
COLUMNS
    X         COST         1.0   R1           1.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1           9.0
BOUNDS
 FX BND       X            4.0
ENDATA
)");
  check_roundtrip("fixed", p, {4.0, 5.0});
}

void test_structure_counts() {
  // The column layout is part of the contract -- the IPM reads `s` as the
  // slack block of `x`, so the block sizes have to be right.
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

  // A: lower-only -> 1 column.  B: free -> 2.  C: boxed [0,8] -> 1 + bound row.
  CHECK_EQ(cp.num_structural, std::size_t{4});
  // R1 is an equality (no slack); R2 and R3 are inequalities.
  CHECK_EQ(cp.num_slack, std::size_t{2});
  // One bound row, for C.
  CHECK_EQ(cp.num_bound, std::size_t{1});
  CHECK_EQ(cp.num_rows(), std::size_t{4});   // 3 original + 1 bound
  CHECK_EQ(cp.num_cols(), std::size_t{7});   // 4 + 2 + 1
  CHECK(cp.validate());
}

void test_equality_rows_get_no_slack() {
  // The handoff form `Ax + s = b, s >= 0` cannot express an equality row.
  // Appending slack columns only to inequality rows is how that is resolved,
  // and it matters: 516 of 821 rows in Netlib 25fv47 are equalities.
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
  CHECK_EQ(canon->problem.num_slack, std::size_t{0});
  CHECK_EQ(canon->problem.num_cols(), std::size_t{2});
  check_roundtrip("all equalities", p, {3.0, 4.0});
}

void test_real_instances() {
  // Structural checks on real models. No feasible point is known without
  // solving, so this verifies the canonical model is well-formed and that its
  // size grows the way the transforms predict.
  const char* names[] = {"afiro", "adlittle"};
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

    // Every original column contributes at least one canonical column.
    CHECK(cp.num_structural >= p.num_cols());
    // Slacks never outnumber rows.
    CHECK(cp.num_slack <= p.num_rows());
    CHECK_EQ(cp.num_rows(), p.num_rows() + cp.num_bound);

    // The canonical matrix holds the original entries plus one per slack and
    // two per bound row.
    CHECK(cp.A.nnz() >= p.nnz());

    std::printf("  %-10s %4zu x %4zu -> %4zu x %4zu  (struct %zu slack %zu "
                "bound %zu)  nnz %zu -> %zu\n",
                name, p.num_rows(), p.num_cols(), cp.num_rows(), cp.num_cols(),
                cp.num_structural, cp.num_slack, cp.num_bound, p.nnz(),
                cp.A.nnz());
  }
}

void test_randomized_roundtrip() {
  // Random feasible points over a model exercising every column shape at once.
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
ENDATA
)");
  CHECK_EQ(p.num_cols(), std::size_t{4});

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
    // Every row here is an inequality, so the forward map can always build a
    // row-feasible canonical point by computing slacks from the actual
    // activity. Equality rows cannot work that way -- a random point simply
    // will not satisfy them -- so they are covered by the deterministic tests
    // above, where feasible points are chosen by hand.
    check_roundtrip("random", p, x);
  }
}

}  // namespace

int main() {
  test_lower_bounds_only();
  test_shifted_lower_bound();
  test_boxed_variables();
  test_free_variables();
  test_upper_bound_only();
  test_ranged_rows();
  test_maximize();
  test_quadratic();
  test_fixed_variable();
  test_structure_counts();
  test_equality_rows_get_no_slack();
  std::printf("canonicalization of real instances:\n");
  test_real_instances();
  test_randomized_roundtrip();
  return sovsolve::test::report("canonical");
}
