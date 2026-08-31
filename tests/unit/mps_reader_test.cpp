// MPS reader tests.
//
// Every case here is either a hand-verifiable model or one of the traps
// documented in docs/MPS-FORMAT-NOTES.md. The traps matter more than the happy
// path: each one produces a *plausible but wrong* model rather than a parse
// failure, so nothing but a test catches them.

#include <string>
#include <string_view>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::INF;
using core::ObjSense;
using core::VarType;

namespace {

model::Problem must_parse(std::string_view text) {
  auto result = io::parseProblem(text, io::FileFormat::Mps);
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "parse succeeded",
                             result.error().format());
    return model::Problem{};
  }
  return std::move(result).value();
}

/// Dense lookup into the CSR view, for readable assertions.
double at(const model::Problem& p, std::size_t row, std::size_t col) {
  const auto off = p.A.csr.offsets();
  const auto idx = p.A.csr.indices();
  const auto val = p.A.csr.values();
  for (auto k = static_cast<std::size_t>(off[row]);
       k < static_cast<std::size_t>(off[row + 1]); ++k) {
    if (static_cast<std::size_t>(idx[k]) == col) return val[k];
  }
  return 0.0;
}

// ---------------------------------------------------------------------------

// The canonical small MPS example. Small enough to verify entirely by hand,
// which is the point: 3 rows, 3 columns, 6 nonzeros.
constexpr std::string_view kTestLp = R"(NAME          TESTLP
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
COLUMNS
    XONE      COST         1.0   LIM1         1.0
    XONE      LIM2         1.0
    YTWO      COST         2.0   LIM1         1.0
    YTWO      MYEQN       -1.0
    ZTHREE    COST         3.0   LIM2         1.0
    ZTHREE    MYEQN        1.0
RHS
    RHS       LIM1         4.0   LIM2         1.0
    RHS       MYEQN        7.0
BOUNDS
 UP BND       XONE         4.0
 LO BND       YTWO        -1.0
 UP BND       YTWO         1.0
ENDATA
)";

void test_basic_model() {
  const auto p = must_parse(kTestLp);

  CHECK(p.problem_name == "TESTLP");
  CHECK(p.objective_row_name == "COST");
  CHECK_EQ(p.num_rows(), std::size_t{3});
  CHECK_EQ(p.num_cols(), std::size_t{3});
  CHECK_EQ(p.nnz(), std::size_t{6});
  CHECK(p.sense == ObjSense::Minimize);
  CHECK(p.type() == core::ProblemType::LP);
  CHECK(p.validate());

  // Objective row entries become c, not matrix entries.
  CHECK_NEAR(p.c[0], 1.0, 1e-15);
  CHECK_NEAR(p.c[1], 2.0, 1e-15);
  CHECK_NEAR(p.c[2], 3.0, 1e-15);
  CHECK_NEAR(p.obj_constant, 0.0, 1e-15);

  // A, by hand:  LIM1 [1 1 0] / LIM2 [1 0 1] / MYEQN [0 -1 1]
  CHECK_NEAR(at(p, 0, 0), 1.0, 1e-15);
  CHECK_NEAR(at(p, 0, 1), 1.0, 1e-15);
  CHECK_NEAR(at(p, 0, 2), 0.0, 1e-15);
  CHECK_NEAR(at(p, 1, 0), 1.0, 1e-15);
  CHECK_NEAR(at(p, 1, 2), 1.0, 1e-15);
  CHECK_NEAR(at(p, 2, 1), -1.0, 1e-15);
  CHECK_NEAR(at(p, 2, 2), 1.0, 1e-15);

  // Row senses become bounds.
  CHECK(core::is_neg_infinite(p.row_lower[0]));   // L
  CHECK_NEAR(p.row_upper[0], 4.0, 1e-15);
  CHECK_NEAR(p.row_lower[1], 1.0, 1e-15);         // G
  CHECK(core::is_pos_infinite(p.row_upper[1]));
  CHECK_NEAR(p.row_lower[2], 7.0, 1e-15);         // E
  CHECK_NEAR(p.row_upper[2], 7.0, 1e-15);

  CHECK_NEAR(p.col_lower[0], 0.0, 1e-15);
  CHECK_NEAR(p.col_upper[0], 4.0, 1e-15);
  CHECK_NEAR(p.col_lower[1], -1.0, 1e-15);
  CHECK_NEAR(p.col_upper[1], 1.0, 1e-15);
  CHECK_NEAR(p.col_lower[2], 0.0, 1e-15);
  CHECK(core::is_pos_infinite(p.col_upper[2]));

  CHECK(p.row_names[0] == "LIM1");
  CHECK(p.row_names[2] == "MYEQN");
  CHECK(p.col_names[0] == "XONE");
  CHECK(p.col_names[2] == "ZTHREE");
}

void test_csr_csc_agree() {
  const auto p = must_parse(kTestLp);

  // The two orientations must describe the same matrix. Sum each column from
  // the CSC and compare against the same column summed from the CSR.
  const auto coff = p.A.csc.offsets();
  const auto cidx = p.A.csc.indices();
  const auto cval = p.A.csc.values();

  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    for (auto k = static_cast<std::size_t>(coff[j]);
         k < static_cast<std::size_t>(coff[j + 1]); ++k) {
      const auto i = static_cast<std::size_t>(cidx[k]);
      CHECK_NEAR(cval[k], at(p, i, j), 1e-15);
    }
  }

  // Invariant I2: strictly increasing indices within every slice.
  CHECK(p.A.csr.validate());
  CHECK(p.A.csc.validate());

  // Column 1 (YTWO) touches rows 0 and 2, in that order.
  CHECK_EQ(p.A.csc.slice_nnz(1), std::size_t{2});
  CHECK_EQ(cidx[static_cast<std::size_t>(coff[1])], core::Index{0});
  CHECK_EQ(cidx[static_cast<std::size_t>(coff[1]) + 1], core::Index{2});
}

void test_objective_constant_is_negated() {
  // The trap: an RHS entry on the objective row is the NEGATED constant.
  // Near-universal convention, rarely documented. Getting it wrong shifts
  // every objective value the solver ever reports.
  const auto p = must_parse(R"(NAME          OBJC
ROWS
 N  COST
 G  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1           2.0   COST         5.0
ENDATA
)");
  CHECK_NEAR(p.obj_constant, -5.0, 1e-15);
  CHECK_NEAR(p.row_lower[0], 2.0, 1e-15);
}

void test_ranges_all_four_cases() {
  // The sign of the range matters ONLY for E rows. Applying |R| uniformly is
  // the most common RANGES bug.
  const auto p = must_parse(R"(NAME          RNG
ROWS
 N  COST
 L  RL
 G  RG
 E  REPOS
 E  RENEG
COLUMNS
    X         COST         1.0   RL           1.0
    X         RG           1.0   REPOS        1.0
    X         RENEG        1.0
RHS
    RHS       RL          10.0   RG          10.0
    RHS       REPOS       10.0   RENEG       10.0
RANGES
    RNG       RL           4.0   RG           4.0
    RNG       REPOS        4.0   RENEG       -4.0
ENDATA
)");
  CHECK_EQ(p.num_rows(), std::size_t{4});

  // L with rhs b, range R  ->  [b - |R|, b]
  CHECK_NEAR(p.row_lower[0], 6.0, 1e-15);
  CHECK_NEAR(p.row_upper[0], 10.0, 1e-15);

  // G with rhs b, range R  ->  [b, b + |R|]
  CHECK_NEAR(p.row_lower[1], 10.0, 1e-15);
  CHECK_NEAR(p.row_upper[1], 14.0, 1e-15);

  // E with R >= 0  ->  [b, b + R]
  CHECK_NEAR(p.row_lower[2], 10.0, 1e-15);
  CHECK_NEAR(p.row_upper[2], 14.0, 1e-15);

  // E with R < 0   ->  [b + R, b]     <- the case that differs
  CHECK_NEAR(p.row_lower[3], 6.0, 1e-15);
  CHECK_NEAR(p.row_upper[3], 10.0, 1e-15);
}

void test_bounds_keys() {
  const auto p = must_parse(R"(NAME          BND
ROWS
 N  COST
 G  R1
COLUMNS
    A         COST         1.0   R1           1.0
    B         COST         1.0   R1           1.0
    C         COST         1.0   R1           1.0
    D         COST         1.0   R1           1.0
    E         COST         1.0   R1           1.0
    F         COST         1.0   R1           1.0
BOUNDS
 FR BND       A
 MI BND       B
 FX BND       C            3.0
 BV BND       D
 UI BND       E            9.0
 PL BND       F
ENDATA
)");
  CHECK(core::is_neg_infinite(p.col_lower[0]));   // FR
  CHECK(core::is_pos_infinite(p.col_upper[0]));

  CHECK(core::is_neg_infinite(p.col_lower[1]));   // MI leaves upper alone
  CHECK(core::is_pos_infinite(p.col_upper[1]));

  CHECK_NEAR(p.col_lower[2], 3.0, 1e-15);         // FX
  CHECK_NEAR(p.col_upper[2], 3.0, 1e-15);

  CHECK_NEAR(p.col_lower[3], 0.0, 1e-15);         // BV
  CHECK_NEAR(p.col_upper[3], 1.0, 1e-15);
  CHECK(p.col_type[3] == VarType::Binary);

  CHECK_NEAR(p.col_upper[4], 9.0, 1e-15);         // UI
  CHECK(p.col_type[4] == VarType::Integer);

  CHECK(core::is_pos_infinite(p.col_upper[5]));   // PL
}

void test_negative_upper_bound_quirk() {
  // UP with a negative value and no prior LO. The default lower bound of 0
  // would give the empty interval [0, -5]. CPLEX/Gurobi/HiGHS free the lower
  // bound instead; readers disagree, so this is an explicit option.
  constexpr std::string_view text = R"(NAME          NEGUP
ROWS
 N  COST
 G  R1
COLUMNS
    X         COST         1.0   R1           1.0
BOUNDS
 UP BND       X           -5.0
ENDATA
)";
  const auto p = must_parse(text);
  CHECK(core::is_neg_infinite(p.col_lower[0]));
  CHECK_NEAR(p.col_upper[0], -5.0, 1e-15);
  CHECK(p.validate());  // would fail if lower stayed at 0

  // With the option off, the lower bound stays at 0 and the model is
  // rejected as inconsistent rather than silently solved.
  model::ReaderOptions strict;
  strict.negative_upper_implies_free_lower = false;
  auto rejected = io::parseProblem(text, io::FileFormat::Mps, strict);
  CHECK(!rejected.has_value());
  if (!rejected.has_value()) {
    CHECK(rejected.error().code == core::ErrorCode::InconsistentBounds);
  }
}

void test_integer_markers() {
  const auto p = must_parse(R"(NAME          MILP
ROWS
 N  COST
 L  R1
COLUMNS
    CONT1     COST         1.0   R1           1.0
    MARKER                 'MARKER'                 'INTORG'
    INT1      COST         2.0   R1           1.0
    INT2      COST         3.0   R1           1.0
    MARKER                 'MARKER'                 'INTEND'
    CONT2     COST         4.0   R1           1.0
RHS
    RHS       R1          10.0
ENDATA
)");
  CHECK_EQ(p.num_cols(), std::size_t{4});
  CHECK(p.col_type[0] == VarType::Continuous);
  CHECK(p.col_type[1] == VarType::Integer);
  CHECK(p.col_type[2] == VarType::Integer);
  CHECK(p.col_type[3] == VarType::Continuous);

  CHECK_EQ(p.num_discrete(), std::size_t{2});
  CHECK(p.has_discrete());
  // Without integrality on Problem, this would be indistinguishable from an LP.
  CHECK(p.type() == core::ProblemType::MILP);
}

void test_duplicate_entries_are_summed() {
  // MPS permits a repeated (row, column) pair and requires the values to add.
  // Keeping the last value instead silently changes the model.
  const auto p = must_parse(R"(NAME          DUP
ROWS
 N  COST
 L  R1
COLUMNS
    X         R1           1.0   COST         2.0
    X         R1           3.0   COST         5.0
RHS
    RHS       R1          10.0
ENDATA
)");
  CHECK_EQ(p.nnz(), std::size_t{1});
  CHECK_NEAR(at(p, 0, 0), 4.0, 1e-15);   // 1 + 3
  CHECK_NEAR(p.c[0], 7.0, 1e-15);        // 2 + 5
  CHECK(p.A.csr.validate());             // duplicates would break invariant I2
}

void test_free_rows_dropped() {
  // The FIRST N row is the objective; later N rows are free rows carrying
  // alternative objectives and must be dropped, not parsed as constraints.
  const auto p = must_parse(R"(NAME          FREEROW
ROWS
 N  COST
 N  ALTOBJ
 L  R1
COLUMNS
    X         COST         1.0   ALTOBJ       9.0
    X         R1           1.0
RHS
    RHS       R1           5.0
ENDATA
)");
  CHECK_EQ(p.num_rows(), std::size_t{1});
  CHECK(p.row_names[0] == "R1");
  CHECK_NEAR(p.c[0], 1.0, 1e-15);   // from COST, not ALTOBJ
  CHECK_EQ(p.nnz(), std::size_t{1});
}

void test_maximize() {
  const auto p = must_parse(R"(NAME          MAXP
OBJSENSE
    MAX
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1           5.0
ENDATA
)");
  CHECK(p.sense == ObjSense::Maximize);
  // The objective is stored as written; the sense flip belongs to the
  // canonicalizer, so the faithful model stays faithful.
  CHECK_NEAR(p.c[0], 1.0, 1e-15);
}

void test_quadratic_objective() {
  // QUADOBJ lists one triangle. An off-diagonal entry stands for both (i,j)
  // and (j,i); getting this wrong scales the quadratic objective by 2.
  const auto p = must_parse(R"(NAME          QPTEST
ROWS
 N  COST
 G  R1
COLUMNS
    X         COST         1.0   R1           1.0
    Y         COST         1.0   R1           1.0
RHS
    RHS       R1           1.0
QUADOBJ
    X         X            2.0
    X         Y            1.0
    Y         Y            4.0
ENDATA
)");
  CHECK(p.has_quadratic());
  CHECK(p.type() == core::ProblemType::QP);
  // 3 declared entries expand to 4: the off-diagonal is mirrored.
  CHECK_EQ(p.Q.nnz(), std::size_t{4});
  CHECK(p.validate());  // also checks Q symmetry

  const auto off = p.Q.csr.offsets();
  const auto idx = p.Q.csr.indices();
  const auto val = p.Q.csr.values();
  auto q = [&](std::size_t i, std::size_t j) {
    for (auto k = static_cast<std::size_t>(off[i]);
         k < static_cast<std::size_t>(off[i + 1]); ++k) {
      if (static_cast<std::size_t>(idx[k]) == j) return val[k];
    }
    return 0.0;
  };
  CHECK_NEAR(q(0, 0), 2.0, 1e-15);
  CHECK_NEAR(q(0, 1), 1.0, 1e-15);
  CHECK_NEAR(q(1, 0), 1.0, 1e-15);   // mirrored
  CHECK_NEAR(q(1, 1), 4.0, 1e-15);
}

void test_fortran_d_exponent() {
  // Old Netlib instances were produced by Fortran writers and use D exponents,
  // which std::from_chars rejects outright.
  const auto p = must_parse(R"(NAME          DEXP
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST      1.5D+02   R1        2.5D-01
RHS
    RHS       R1        1.0D+01
ENDATA
)");
  CHECK_NEAR(p.c[0], 150.0, 1e-12);
  CHECK_NEAR(at(p, 0, 0), 0.25, 1e-12);
  CHECK_NEAR(p.row_upper[0], 10.0, 1e-12);
}

void test_errors_are_located() {
  // A parse failure must say which line and which section, or it is useless on
  // a large file.
  auto bad = io::parseProblem(R"(NAME          BAD
ROWS
 N  COST
 Q  R1
ENDATA
)", io::FileFormat::Mps);
  CHECK(!bad.has_value());
  if (!bad.has_value()) {
    const auto& e = bad.error();
    CHECK(e.code == core::ErrorCode::ParseError);
    CHECK(e.line.has_value());
    CHECK(e.line.value_or(0) == 4);
    CHECK(e.section.value_or("") == "ROWS");
    CHECK(e.format().find("ROWS") != std::string::npos);
  }

  auto undef = io::parseProblem(R"(NAME          BAD2
ROWS
 N  COST
 L  R1
COLUMNS
    X         NOSUCHROW    1.0
ENDATA
)", io::FileFormat::Mps);
  CHECK(!undef.has_value());
  if (!undef.has_value()) {
    CHECK(undef.error().code == core::ErrorCode::UndefinedName);
  }

  auto noobj = io::parseProblem(R"(NAME          BAD3
ROWS
 L  R1
ENDATA
)", io::FileFormat::Mps);
  CHECK(!noobj.has_value());
}

void test_sos_is_refused_not_ignored() {
  // Silently dropping an SOS set changes the model and produces a confidently
  // wrong answer, which is worse than refusing the file.
  auto result = io::parseProblem(R"(NAME          SOSTEST
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1           1.0
SOS
 S1 SOS       SET1         1
ENDATA
)", io::FileFormat::Mps);
  CHECK(!result.has_value());
  if (!result.has_value()) {
    CHECK(result.error().code == core::ErrorCode::UnsupportedFeature);
  }
}

void test_empty_and_comment_lines() {
  const auto p = must_parse(R"(* a comment in column 1
NAME          CMT
* another comment
ROWS
 N  COST
 L  R1

COLUMNS
    X         COST         1.0   R1           1.0
RHS
    RHS       R1           5.0
ENDATA
)");
  CHECK_EQ(p.num_rows(), std::size_t{1});
  CHECK_EQ(p.num_cols(), std::size_t{1});
  CHECK(p.validate());
}

}  // namespace

int main() {
  test_basic_model();
  test_csr_csc_agree();
  test_objective_constant_is_negated();
  test_ranges_all_four_cases();
  test_bounds_keys();
  test_negative_upper_bound_quirk();
  test_integer_markers();
  test_duplicate_entries_are_summed();
  test_free_rows_dropped();
  test_maximize();
  test_quadratic_objective();
  test_fortran_d_exponent();
  test_errors_are_located();
  test_sos_is_refused_not_ignored();
  test_empty_and_comment_lines();
  return sovsolve::test::report("mps_reader");
}
