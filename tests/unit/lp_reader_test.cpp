// CPLEX LP format reader tests.
//
// There is no LP benchmark corpus -- every set the problem statement names
// ships as MPS -- so the strongest available test is EQUIVALENCE: each model is
// written twice, once in LP and once in MPS, and both must load to structurally
// identical `Problem`s. That checks the LP reader against a reader already
// verified on 19 Netlib instances, rather than against my own expectations.
//
// The dialect table in docs/LP-FORMAT-NOTES.md is covered case by case below.

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::INF;
using core::is_finite_bound;
using core::Real;

namespace {

core::Expected<model::Problem> load_lp(std::string_view text) {
  return io::parseProblem(text, io::FileFormat::Lp);
}

model::Problem lp(std::string_view text) {
  auto r = load_lp(text);
  if (!r.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "lp parse", r.error().format());
    return {};
  }
  return std::move(r).value();
}

model::Problem mps(std::string_view text) {
  auto r = io::parseProblem(text, io::FileFormat::Mps);
  if (!r.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "mps parse", r.error().format());
    return {};
  }
  return std::move(r).value();
}

/// Guard before indexing. A failed parse yields a default-constructed
/// `Problem`, and reading `row_lower[0]` off that is a segfault that hides
/// whatever the real parse error was -- so every test that indexes checks the
/// shape first and bails with a readable message instead.
bool shaped(const char* label, const model::Problem& p, std::size_t rows,
            std::size_t cols) {
  if (p.num_rows() == rows && p.num_cols() == cols) return true;
  ::sovsolve::test::record(__FILE__, __LINE__, label,
                           "expected " + std::to_string(rows) + "x" +
                               std::to_string(cols) + ", got " +
                               std::to_string(p.num_rows()) + "x" +
                               std::to_string(p.num_cols()));
  return false;
}

/// Dense coefficient lookup, so equivalence does not depend on storage order.
Real coeff(const model::Problem& p, std::size_t row, std::size_t col) {
  const auto off = p.A.csr.offsets();
  const auto idx = p.A.csr.indices();
  const auto val = p.A.csr.values();
  Real sum = 0.0;
  for (auto k = static_cast<std::size_t>(off[row]);
       k < static_cast<std::size_t>(off[row + 1]); ++k) {
    if (static_cast<std::size_t>(idx[k]) == col) sum += val[k];
  }
  return sum;
}

/// Assert two models are structurally the same, matching by NAME rather than
/// by index -- LP discovers columns in expression order and MPS in COLUMNS
/// order, so the two need not agree on numbering.
void check_equivalent(const char* label, const model::Problem& a,
                      const model::Problem& b) {
  if (a.num_rows() != b.num_rows() || a.num_cols() != b.num_cols()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label,
                             "dimension mismatch: " +
                                 std::to_string(a.num_rows()) + "x" +
                                 std::to_string(a.num_cols()) + " vs " +
                                 std::to_string(b.num_rows()) + "x" +
                                 std::to_string(b.num_cols()));
    return;
  }
  CHECK(a.sense == b.sense);
  CHECK_NEAR(a.obj_constant, b.obj_constant, 1e-12);
  CHECK_EQ(a.nnz(), b.nnz());

  const auto find = [](const core::NameArena& arena,
                       std::string_view name) -> std::size_t {
    for (std::size_t i = 0; i < arena.size(); ++i) {
      if (arena[i] == name) return i;
    }
    return static_cast<std::size_t>(-1);
  };

  std::vector<std::size_t> col_of(a.num_cols(), 0);
  bool mapped = true;
  for (std::size_t j = 0; j < a.num_cols(); ++j) {
    const std::size_t k = find(b.col_names, a.col_names[j]);
    if (k == static_cast<std::size_t>(-1)) {
      ::sovsolve::test::record(__FILE__, __LINE__, label,
                               "column " + std::string(a.col_names[j]) +
                                   " missing from the MPS model");
      mapped = false;
      break;
    }
    col_of[j] = k;
  }
  if (!mapped) return;

  std::vector<std::size_t> row_of(a.num_rows(), 0);
  for (std::size_t i = 0; i < a.num_rows(); ++i) {
    const std::size_t k = find(b.row_names, a.row_names[i]);
    if (k == static_cast<std::size_t>(-1)) {
      ::sovsolve::test::record(__FILE__, __LINE__, label,
                               "row " + std::string(a.row_names[i]) +
                                   " missing from the MPS model");
      return;
    }
    row_of[i] = k;
  }

  for (std::size_t j = 0; j < a.num_cols(); ++j) {
    const std::size_t k = col_of[j];
    CHECK_NEAR(a.c[j], b.c[k], 1e-12);
    CHECK_NEAR(a.col_lower[j], b.col_lower[k], 1e-12);
    CHECK_NEAR(a.col_upper[j], b.col_upper[k], 1e-12);
    CHECK(a.col_type[j] == b.col_type[k]);
  }
  for (std::size_t i = 0; i < a.num_rows(); ++i) {
    const std::size_t r = row_of[i];
    CHECK_NEAR(a.row_lower[i], b.row_lower[r], 1e-12);
    CHECK_NEAR(a.row_upper[i], b.row_upper[r], 1e-12);
    for (std::size_t j = 0; j < a.num_cols(); ++j) {
      CHECK_NEAR(coeff(a, i, j), coeff(b, r, col_of[j]), 1e-12);
    }
  }
}

// ---------------------------------------------------------------------------

void test_minimal() {
  const auto p = lp(R"(Minimize
 obj: 3 x + 2 y
Subject To
 c1: x + y <= 10
 c2: x - y >= 2
End
)");
  if (!shaped("minimal", p, 2, 2)) return;
  CHECK(p.sense == core::ObjSense::Minimize);
  CHECK_NEAR(p.c[0], 3.0, 1e-12);
  CHECK_NEAR(p.c[1], 2.0, 1e-12);
  CHECK_NEAR(coeff(p, 0, 0), 1.0, 1e-12);
  CHECK_NEAR(coeff(p, 1, 1), -1.0, 1e-12);
  // Default column bounds are [0, +inf), the same as MPS.
  CHECK_NEAR(p.col_lower[0], 0.0, 1e-12);
  CHECK(!is_finite_bound(p.col_upper[0]));
  CHECK_NEAR(p.row_upper[0], 10.0, 1e-12);
  CHECK(!is_finite_bound(p.row_lower[0]));
  CHECK_NEAR(p.row_lower[1], 2.0, 1e-12);
}

void test_unnamed_constraints_do_not_merge() {
  // The bug this guards: with variables allowed on the right-hand side, the
  // `x` of the second constraint gets folded into the first one's RHS, and the
  // model differs from the file with no error raised.
  const auto p = lp(R"(Minimize
 x + y
Subject To
 x + y <= 5
 x - y >= 1
 y <= 3
End
)");
  if (!shaped("unnamed", p, 3, 2)) return;
  CHECK_NEAR(p.row_upper[0], 5.0, 1e-12);
  CHECK_NEAR(p.row_lower[1], 1.0, 1e-12);
  CHECK_NEAR(p.row_upper[2], 3.0, 1e-12);
  CHECK_NEAR(coeff(p, 0, 0), 1.0, 1e-12);
  CHECK_NEAR(coeff(p, 1, 1), -1.0, 1e-12);
  CHECK_NEAR(coeff(p, 2, 0), 0.0, 1e-12);
}

void test_labelled_constraint_is_not_eaten_as_a_variable() {
  // Same failure in the other direction: `c2` is a label, not a variable, and
  // consuming it would create a phantom column.
  const auto p = lp(R"(Minimize
 obj: x
Subject To
 c1: x <= 5
 c2: x >= 1
End
)");
  if (!shaped("labelled", p, 2, 1)) return;
  CHECK(p.col_names[0] == "x");
}

void test_variables_on_the_right_are_refused() {
  auto r = load_lp(R"(Minimize
 obj: x
Subject To
 c1: x <= y
End
)");
  CHECK(!r.has_value());
  if (r.has_value()) return;
  CHECK(r.error().code == core::ErrorCode::UnsupportedFeature);
}

void test_comments_and_line_wrapping() {
  const auto p = lp(R"(\ a comment before everything
Minimize
 obj: 3 x
      + 2 y     \ trailing comment
      - 1 z
Subject To
 c1: x + y
     + z <= 10
End
)");
  if (!shaped("comments", p, 1, 3)) return;
  CHECK_NEAR(p.c[2], -1.0, 1e-12);
  CHECK_NEAR(coeff(p, 0, 2), 1.0, 1e-12);
}

void test_coefficient_spellings() {
  // `3x`, `3 x` and `3 * x` are the same term; `2e3` is a number while `2 e3`
  // is a coefficient on a variable named e3.
  const auto p = lp(R"(Minimize
 obj: 3x + 4 y + 5 * z + 2e3 w + 1.5e-2 v
Subject To
 c1: x + y + z + w + v <= 1
End
)");
  if (!shaped("spellings", p, 1, 5)) return;
  CHECK_NEAR(p.c[0], 3.0, 1e-12);
  CHECK_NEAR(p.c[1], 4.0, 1e-12);
  CHECK_NEAR(p.c[2], 5.0, 1e-12);
  CHECK_NEAR(p.c[3], 2000.0, 1e-12);
  CHECK_NEAR(p.c[4], 0.015, 1e-15);
}

void test_exponent_is_not_confused_with_a_variable() {
  // `3 e5` must be a coefficient of 3 on a variable named e5, not the number
  // 300000 -- an exponent needs digits after it to be an exponent.
  const auto p = lp(R"(Minimize
 obj: 3 e5
Subject To
 c1: e5 <= 1
End
)");
  if (!shaped("exponent", p, 1, 1)) return;
  CHECK(p.col_names[0] == "e5");
  CHECK_NEAR(p.c[0], 3.0, 1e-12);
}

void test_objective_constant_has_no_sign_trap() {
  // MPS negates an RHS entry on the objective row. LP writes the constant
  // directly, so the two readers must NOT agree by accident here.
  const auto p = lp(R"(Minimize
 obj: 3 x + 7
Subject To
 c1: x <= 5
End
)");
  CHECK_NEAR(p.obj_constant, 7.0, 1e-12);

  const auto q = mps(R"(NAME          OBJC
ROWS
 N  obj
 L  c1
COLUMNS
    x         obj          3.0   c1           1.0
RHS
    RHS       c1           5.0   obj         -7.0
ENDATA
)");
  CHECK_NEAR(q.obj_constant, 7.0, 1e-12);
  check_equivalent("objective constant", p, q);
}

void test_maximize() {
  const auto p = lp(R"(Maximize
 obj: 3 x + 2 y
Subject To
 c1: x + y <= 4
End
)");
  CHECK(p.sense == core::ObjSense::Maximize);
  const auto q = lp(R"(max
 obj: 3 x + 2 y
Subject To
 c1: x + y <= 4
End
)");
  CHECK(q.sense == core::ObjSense::Maximize);
}

void test_section_keyword_spellings() {
  for (std::string_view st : {"Subject To", "subject to", "such that", "st",
                              "s.t.", "ST"}) {
    const std::string text = std::string("Minimize\n obj: x\n") +
                             std::string(st) + "\n c1: x <= 5\nEnd\n";
    const auto p = lp(text);
    (void)shaped("section spelling", p, 1, 1);
  }
}

void test_bounds_all_forms() {
  const auto p = lp(R"(Minimize
 obj: a + b + c + d + e + f
Subject To
 r1: a + b + c + d + e + f <= 100
Bounds
 a >= 2
 b <= 7
 3 <= c <= 9
 d = 4
 e free
 -inf <= f <= 5
End
)");
  if (!shaped("bounds", p, 1, 6)) return;
  CHECK_NEAR(p.col_lower[0], 2.0, 1e-12);
  CHECK(!is_finite_bound(p.col_upper[0]));
  CHECK_NEAR(p.col_lower[1], 0.0, 1e-12);
  CHECK_NEAR(p.col_upper[1], 7.0, 1e-12);
  CHECK_NEAR(p.col_lower[2], 3.0, 1e-12);
  CHECK_NEAR(p.col_upper[2], 9.0, 1e-12);
  CHECK_NEAR(p.col_lower[3], 4.0, 1e-12);
  CHECK_NEAR(p.col_upper[3], 4.0, 1e-12);
  CHECK(!is_finite_bound(p.col_lower[4]));
  CHECK(!is_finite_bound(p.col_upper[4]));
  CHECK(!is_finite_bound(p.col_lower[5]));
  CHECK_NEAR(p.col_upper[5], 5.0, 1e-12);
}

void test_integrality() {
  const auto p = lp(R"(Minimize
 obj: x + y + z
Subject To
 c1: x + y + z <= 10
Bounds
 x <= 8
General
 x
Binary
 y
End
)");
  if (!shaped("integrality", p, 1, 3)) return;
  CHECK(p.col_type[0] == model::VarType::Integer);
  CHECK(p.col_type[1] == model::VarType::Binary);
  CHECK(p.col_type[2] == model::VarType::Continuous);
  // Binary implies [0, 1].
  CHECK_NEAR(p.col_lower[1], 0.0, 1e-12);
  CHECK_NEAR(p.col_upper[1], 1.0, 1e-12);
  CHECK(p.has_discrete());
  CHECK_EQ(p.num_discrete(), std::size_t{2});
}

void test_ranged_constraint_gurobi_form() {
  // CPLEX rejects this; Gurobi accepts it, and so do we.
  const auto p = lp(R"(Minimize
 obj: x + y
Subject To
 c1: -5 <= x + y <= 10
End
)");
  if (!shaped("ranged", p, 1, 2)) return;
  CHECK_NEAR(p.row_lower[0], -5.0, 1e-12);
  CHECK_NEAR(p.row_upper[0], 10.0, 1e-12);
}

void test_constant_inside_a_ranged_constraint_shifts_both_sides() {
  // `-5 <= x + y + 3 <= 10` constrains `x + y` to [-8, 7]. Getting only one
  // side right is the easy mistake.
  const auto p = lp(R"(Minimize
 obj: x + y
Subject To
 c1: -5 <= x + y + 3 <= 10
End
)");
  if (!shaped("ranged constant", p, 1, 2)) return;
  CHECK_NEAR(p.row_lower[0], -8.0, 1e-12);
  CHECK_NEAR(p.row_upper[0], 7.0, 1e-12);
}

void test_constant_on_the_left_moves_right() {
  const auto p = lp(R"(Minimize
 obj: x
Subject To
 c1: x + 3 <= 10
End
)");
  if (!shaped("lhs constant", p, 1, 1)) return;
  CHECK_NEAR(p.row_upper[0], 7.0, 1e-12);
}

void test_duplicate_terms_are_summed() {
  const auto p = lp(R"(Minimize
 obj: x + 2 x
Subject To
 c1: x + 3 x <= 8
End
)");
  if (!shaped("duplicates", p, 1, 1)) return;
  CHECK_NEAR(p.c[0], 3.0, 1e-12);
  CHECK_NEAR(coeff(p, 0, 0), 4.0, 1e-12);
}

void test_quadratic_objective_half_convention() {
  // `[ 2 x^2 + 4 x * y ] / 2` means the objective term is x^2 + 2xy.
  // The model stores 1/2 x'Qx, so x'Qx must equal 2x^2 + 4xy, which with Q
  // full symmetric means Q_xx = 2 and Q_xy = Q_yx = 2.
  //
  // This is the same factor-of-two trap QUADOBJ has: getting it wrong scales
  // every QP objective by two, and nothing else in the pipeline notices.
  const auto p = lp(R"(Minimize
 obj: [ 2 x ^ 2 + 4 x * y ] / 2
Subject To
 c1: x + y <= 10
End
)");
  if (!shaped("quadratic", p, 1, 2)) return;
  CHECK(p.has_quadratic());
  CHECK_EQ(p.Q.rows(), std::size_t{2});

  const auto q = [&](std::size_t i, std::size_t j) {
    const auto off = p.Q.csr.offsets();
    const auto idx = p.Q.csr.indices();
    const auto val = p.Q.csr.values();
    Real s = 0.0;
    for (auto k = static_cast<std::size_t>(off[i]);
         k < static_cast<std::size_t>(off[i + 1]); ++k) {
      if (static_cast<std::size_t>(idx[k]) == j) s += val[k];
    }
    return s;
  };
  CHECK_NEAR(q(0, 0), 2.0, 1e-12);
  CHECK_NEAR(q(0, 1), 2.0, 1e-12);
  CHECK_NEAR(q(1, 0), 2.0, 1e-12);
  CHECK_NEAR(q(1, 1), 0.0, 1e-12);

  // Hand-check the objective at x = 3, y = 1: 1/2 x'Qx
  //   = 1/2 (2*9 + 2*3*1 + 2*1*3) = 1/2 (18 + 12) = 15
  // and directly from the file: x^2 + 2xy = 9 + 6 = 15.
  const Real xv[2] = {3.0, 1.0};
  Real quad = 0.0;
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 2; ++j) quad += xv[i] * q(i, j) * xv[j];
  }
  CHECK_NEAR(0.5 * quad, 15.0, 1e-12);
}

void test_quadratic_without_divisor() {
  // With no `/2`, the bracket is read as the objective contribution itself,
  // so `[ 2 x^2 ]` means 2x^2 and Q_xx must be 4.
  const auto p = lp(R"(Minimize
 obj: [ 2 x ^ 2 ]
Subject To
 c1: x <= 10
End
)");
  if (!shaped("quadratic no divisor", p, 1, 1)) return;
  CHECK(p.has_quadratic());
  const auto off = p.Q.csr.offsets();
  const auto val = p.Q.csr.values();
  CHECK_EQ(static_cast<std::size_t>(off[1] - off[0]), std::size_t{1});
  CHECK_NEAR(val[0], 4.0, 1e-12);
}

void test_unsupported_sections_are_reported_not_skipped() {
  for (std::string_view section : {"SOS", "Semi-Continuous"}) {
    const std::string text = std::string("Minimize\n obj: x\nSubject To\n") +
                             " c1: x <= 5\n" + std::string(section) +
                             "\n s1: x\nEnd\n";
    auto r = load_lp(text);
    CHECK(!r.has_value());
    if (r.has_value()) continue;
    CHECK(r.error().code == core::ErrorCode::UnsupportedFeature);
  }
}

void test_missing_objective_section_is_an_error() {
  auto r = load_lp(R"(Subject To
 c1: x <= 5
End
)");
  CHECK(!r.has_value());
  if (r.has_value()) return;
  CHECK(r.error().code == core::ErrorCode::ParseError);
}

void test_duplicate_row_name_is_an_error() {
  auto r = load_lp(R"(Minimize
 obj: x
Subject To
 c1: x <= 5
 c1: x >= 1
End
)");
  CHECK(!r.has_value());
  if (r.has_value()) return;
  CHECK(r.error().code == core::ErrorCode::DuplicateName);
}

// ---------------------------------------------------------------------------
// Equivalence against the MPS reader
// ---------------------------------------------------------------------------

void test_equivalence_mixed_model() {
  const auto a = lp(R"(Minimize
 COST: 1.5 A - 0.5 B + 2 C + 0.25 D
Subject To
 R1: A + B + D <= 30
 R2: 2 A + C + D <= 40
 R3: B + C >= -20
 R4: A + C = 7
Bounds
 1 <= A <= 9
 B free
 -inf <= C <= 5
 D >= -3
End
)");
  const auto b = mps(R"(NAME          MIXED
ROWS
 N  COST
 L  R1
 L  R2
 G  R3
 E  R4
COLUMNS
    A         COST         1.5   R1           1.0
    A         R2           2.0   R4           1.0
    B         COST        -0.5   R1           1.0
    B         R3           1.0
    C         COST         2.0   R2           1.0
    C         R3           1.0   R4           1.0
    D         COST         0.25  R1           1.0
    D         R2           1.0
RHS
    RHS       R1          30.0   R2          40.0
    RHS       R3         -20.0   R4           7.0
BOUNDS
 LO BND       A            1.0
 UP BND       A            9.0
 FR BND       B
 MI BND       C
 UP BND       C            5.0
 LO BND       D           -3.0
ENDATA
)");
  check_equivalent("mixed model", a, b);
}

void test_equivalence_integrality() {
  const auto a = lp(R"(Maximize
 profit: 5 x + 4 y + 3 z
Subject To
 cap: 2 x + 3 y + z <= 5
 lab: 4 x + y + 2 z <= 11
Bounds
 x <= 10
 z <= 1
General
 x
Binary
 z
End
)");
  const auto b = mps(R"(NAME          INTEG
OBJSENSE
    MAX
ROWS
 N  profit
 L  cap
 L  lab
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    x         profit       5.0   cap          2.0
    x         lab          4.0
    MARKER                 'MARKER'                 'INTEND'
    y         profit       4.0   cap          3.0
    y         lab          1.0
    MARKER                 'MARKER'                 'INTORG'
    z         profit       3.0   cap          1.0
    z         lab          2.0
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       cap          5.0   lab         11.0
BOUNDS
 UP BND       x           10.0
 BV BND       z
ENDATA
)");
  // MPS `BV` gives Binary; LP `Binary` gives Binary. `General` gives Integer,
  // and MPS INTORG/INTEND with a finite upper bound gives Integer too.
  check_equivalent("integrality", a, b);
}

void test_equivalence_quadratic() {
  const auto a = lp(R"(Minimize
 obj: x - y + [ 2 x ^ 2 + 2 x * y + 3 y ^ 2 ] / 2
Subject To
 R1: x + y <= 10
Bounds
 x >= 2
End
)");
  const auto b = mps(R"(NAME          QPEQ
ROWS
 N  obj
 L  R1
COLUMNS
    x         obj          1.0   R1           1.0
    y         obj         -1.0   R1           1.0
RHS
    RHS       R1          10.0
BOUNDS
 LO BND       x            2.0
QUADOBJ
    x         x            2.0
    x         y            1.0
    y         y            3.0
ENDATA
)");
  check_equivalent("quadratic", a, b);

  // And the stored Q must actually match entry for entry, not just in nnz.
  CHECK(a.has_quadratic());
  CHECK(b.has_quadratic());
  CHECK_EQ(a.Q.nnz(), b.Q.nnz());
}

}  // namespace

int main() {
  test_minimal();
  test_unnamed_constraints_do_not_merge();
  test_labelled_constraint_is_not_eaten_as_a_variable();
  test_variables_on_the_right_are_refused();
  test_comments_and_line_wrapping();
  test_coefficient_spellings();
  test_exponent_is_not_confused_with_a_variable();
  test_objective_constant_has_no_sign_trap();
  test_maximize();
  test_section_keyword_spellings();
  test_bounds_all_forms();
  test_integrality();
  test_ranged_constraint_gurobi_form();
  test_constant_inside_a_ranged_constraint_shifts_both_sides();
  test_constant_on_the_left_moves_right();
  test_duplicate_terms_are_summed();
  test_quadratic_objective_half_convention();
  test_quadratic_without_divisor();
  test_unsupported_sections_are_reported_not_skipped();
  test_missing_objective_section_is_an_error();
  test_duplicate_row_name_is_an_error();

  test_equivalence_mixed_model();
  test_equivalence_integrality();
  test_equivalence_quadratic();
  return sovsolve::test::report("lp_reader");
}
