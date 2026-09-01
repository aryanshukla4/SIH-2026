// QPLIB reader tests.
//
// The anchor is the worked example from Furini et al., "QPLIB: a library of
// quadratic programming instances" (RAL-P-2017-003, Appendix B). It is
// reproduced here VERBATIM, and the model it encodes is stated in the paper in
// closed form, so every field can be checked against a published expectation
// rather than against my reading of the format.
//
// That matters more here than for MPS or LP. QPLIB is POSITIONAL -- it has no
// section keywords at all -- so a reader that loses its place does not raise a
// syntax error, it produces a plausible model with everything after the slip
// shifted by one field. The tests below therefore check values, not just that
// the parse succeeded.

#include <cmath>
#include <string>
#include <string_view>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::INF;
using core::is_finite_bound;
using core::Real;

namespace {

core::Expected<model::Problem> load(std::string_view text) {
  return io::parseProblem(text, io::FileFormat::Qplib);
}

model::Problem qplib(std::string_view text) {
  auto r = load(text);
  if (!r.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "qplib parse",
                             r.error().format());
    return {};
  }
  return std::move(r).value();
}

Real at(const model::Problem& p, std::size_t row, std::size_t col) {
  const auto off = p.A.csr.offsets();
  const auto idx = p.A.csr.indices();
  const auto val = p.A.csr.values();
  Real s = 0.0;
  for (auto k = static_cast<std::size_t>(off[row]);
       k < static_cast<std::size_t>(off[row + 1]); ++k) {
    if (static_cast<std::size_t>(idx[k]) == col) s += val[k];
  }
  return s;
}

Real q_at(const model::Problem& p, std::size_t i, std::size_t j) {
  if (p.Q.empty()) return 0.0;
  const auto off = p.Q.csr.offsets();
  const auto idx = p.Q.csr.indices();
  const auto val = p.Q.csr.values();
  Real s = 0.0;
  for (auto k = static_cast<std::size_t>(off[i]);
       k < static_cast<std::size_t>(off[i + 1]); ++k) {
    if (static_cast<std::size_t>(idx[k]) == j) s += val[k];
  }
  return s;
}

// The paper's example, verbatim including its annotations. The model is
//
//   min  x1^2 + x2^2 + x3^2 - x1x2 - x2x3 - 0.2x1 - 0.4x2 - 0.2x3
//   s.t. 1 <= x1 + x2,  1 <= x1 + x3,  0 <= x1 <= 1,  0 <= x2 <= 2,
//        x3 binary
//
// with Hessian  Q0 = [ 2 -1  0 ; -1  2 -1 ;  0 -1  2 ].
constexpr std::string_view kPaperExample = R"(! ---------------
! example problem
! ---------------
MIPBAND   # problem name
QML       # problem is a mixed-integer quadratic program
Minimize  # minimize the objective function
3         # variables
2         # general linear constraints
5         # nonzeros in lower triangle of Q^0
1 1 2.0   5 lines row & column index & value of nonzero in lower triangle Q^0
2 1 -1.0  |
2 2 2.0   |
3 2 -1.0  |
3 3 2.0   |
-0.2      default value for entries in b_0
1         # non default entries in b_0
2 -0.4    1 line of index & value of non-default values in b_0
0.0       value of q^0
4         # nonzeros in vectors b^i (i=1,...,m)
1 1 1.0   4 lines constraint, index & value of nonzero in b^i (i=1,...,m)
1 2 1.0   |
2 1 1.0   |
2 3 1.0   |
1.0E+20   infinity
1.0       default value for entries in c_l
0         # non default entries in c_l
1.0E+20   default value for entries in c_u
0         # non default entries in c_u
0.0       default value for entries in l
0         # non default entries in l
1.0       default value for entries in u
1         # non default entries in u
2 2.0     1 line of non-default indices and values in u
0         default variable type is continuous
1         # non default variable types
3 2       variable 3 is binary
1.0       default value for initial values for x
0         # non default entries in x
0.0       default value for initial values for y
0         # non default entries in y
0.0       default value for initial values for z
0         # non default entries in z
0         # non default names for variables
0         # non default names for constraints
)";

void test_paper_example() {
  const auto p = qplib(kPaperExample);
  if (p.num_cols() != 3 || p.num_rows() != 2) {
    ::sovsolve::test::record(__FILE__, __LINE__, "paper example",
                             "expected 2x3, got " +
                                 std::to_string(p.num_rows()) + "x" +
                                 std::to_string(p.num_cols()));
    return;
  }
  CHECK(p.problem_name == "MIPBAND");
  CHECK(p.sense == core::ObjSense::Minimize);

  // Hessian: the lower triangle in the file mirrors to full symmetric at FULL
  // value. Halving off-diagonals is the classic error, and it would show up
  // here as Q(0,1) = -0.5.
  CHECK_NEAR(q_at(p, 0, 0), 2.0, 1e-15);
  CHECK_NEAR(q_at(p, 1, 1), 2.0, 1e-15);
  CHECK_NEAR(q_at(p, 2, 2), 2.0, 1e-15);
  CHECK_NEAR(q_at(p, 0, 1), -1.0, 1e-15);
  CHECK_NEAR(q_at(p, 1, 0), -1.0, 1e-15);
  CHECK_NEAR(q_at(p, 1, 2), -1.0, 1e-15);
  CHECK_NEAR(q_at(p, 2, 1), -1.0, 1e-15);
  CHECK_NEAR(q_at(p, 0, 2), 0.0, 1e-15);
  CHECK_EQ(p.Q.nnz(), std::size_t{7});  // 3 diagonal + 2 pairs

  // Linear objective: default -0.2 with index 2 overridden to -0.4.
  CHECK_NEAR(p.c[0], -0.2, 1e-15);
  CHECK_NEAR(p.c[1], -0.4, 1e-15);
  CHECK_NEAR(p.c[2], -0.2, 1e-15);
  CHECK_NEAR(p.obj_constant, 0.0, 1e-15);

  // Constraints: x1 + x2 and x1 + x3, both >= 1.
  CHECK_NEAR(at(p, 0, 0), 1.0, 1e-15);
  CHECK_NEAR(at(p, 0, 1), 1.0, 1e-15);
  CHECK_NEAR(at(p, 0, 2), 0.0, 1e-15);
  CHECK_NEAR(at(p, 1, 0), 1.0, 1e-15);
  CHECK_NEAR(at(p, 1, 1), 0.0, 1e-15);
  CHECK_NEAR(at(p, 1, 2), 1.0, 1e-15);
  CHECK_EQ(p.nnz(), std::size_t{4});

  for (std::size_t i = 0; i < 2; ++i) {
    CHECK_NEAR(p.row_lower[i], 1.0, 1e-15);
    // 1.0E+20 is the file's declared infinity, so the row is one-sided.
    CHECK(!is_finite_bound(p.row_upper[i]));
  }

  // Bounds: 0 <= x1 <= 1, 0 <= x2 <= 2, and x3 binary -- whose [0,1] comes
  // from the type, overriding the default upper bound of 1.
  CHECK_NEAR(p.col_lower[0], 0.0, 1e-15);
  CHECK_NEAR(p.col_upper[0], 1.0, 1e-15);
  CHECK_NEAR(p.col_lower[1], 0.0, 1e-15);
  CHECK_NEAR(p.col_upper[1], 2.0, 1e-15);
  CHECK_NEAR(p.col_lower[2], 0.0, 1e-15);
  CHECK_NEAR(p.col_upper[2], 1.0, 1e-15);

  CHECK(p.col_type[0] == model::VarType::Continuous);
  CHECK(p.col_type[1] == model::VarType::Continuous);
  CHECK(p.col_type[2] == model::VarType::Binary);
  CHECK(p.has_discrete());
  CHECK_EQ(p.num_discrete(), std::size_t{1});

  // Default names are the variable's own index, as a decimal string.
  CHECK(p.col_names[0] == "1");
  CHECK(p.col_names[2] == "3");

  // The objective at x = (1, 1, 1), computed from the paper's closed form:
  //   1 + 1 + 1 - 1 - 1 - 0.2 - 0.4 - 0.2 = 0.2
  // and from the stored data as 1/2 x'Qx + c'x + q0.
  const Real xv[3] = {1.0, 1.0, 1.0};
  Real quad = 0.0;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) quad += xv[i] * q_at(p, i, j) * xv[j];
  }
  Real obj = 0.5 * quad + p.obj_constant;
  for (std::size_t j = 0; j < 3; ++j) obj += p.c[j] * xv[j];
  CHECK_NEAR(obj, 0.2, 1e-12);
}

void test_positional_slip_is_caught_not_absorbed() {
  // The format's real hazard: no section keywords, so a wrong count shifts
  // every later field. Claiming one Hessian entry too many consumes the line
  // that should have been the linear-objective default, and everything after
  // slides. The reader must fail rather than return a plausible model.
  std::string bad(kPaperExample);
  const auto pos = bad.find("5         # nonzeros in lower triangle");
  CHECK(pos != std::string::npos);
  if (pos == std::string::npos) return;
  bad[pos] = '6';

  auto r = load(bad);
  CHECK(!r.has_value());
  if (r.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "positional slip",
                             "an over-long Hessian count was absorbed silently");
    return;
  }
  // And the diagnostic must name what was being read when it went wrong.
  CHECK(r.error().section.has_value());
}

void test_quadratic_constraints_are_refused() {
  // `**Q` carries quadratic constraints, which Problem does not model. Dropping
  // the quadratic terms would leave a valid-looking but strictly easier model.
  std::string qcp(kPaperExample);
  const auto pos = qcp.find("QML");
  CHECK(pos != std::string::npos);
  if (pos == std::string::npos) return;
  qcp.replace(pos, 3, "QMQ");

  auto r = load(qcp);
  CHECK(!r.has_value());
  if (r.has_value()) return;
  CHECK(r.error().code == core::ErrorCode::UnsupportedFeature);
}

void test_linear_objective_omits_the_hessian_section() {
  // Note 3: type `L**` has no Q0 section at all. A reader that looks for one
  // would consume the linear-objective default as a nonzero count.
  const auto p = qplib(R"(LINOBJ    # problem name
LCL       # linear objective, continuous variables, linear constraints
Minimize
2
1
3.0       default value for entries in b_0
1
2 5.0
1.5       value of q^0
2         # nonzeros in b^i
1 1 1.0
1 2 1.0
1.0E+20
0.0       default c_l
0
10.0      default c_u
0
0.0       default l
0
1.0E+20   default u
0
1.0       default x0
0
0.0       default y0
0
0.0       default z0
0
0
0
)");
  if (p.num_cols() != 2 || p.num_rows() != 1) {
    ::sovsolve::test::record(__FILE__, __LINE__, "linear objective",
                             "expected 1x2, got " +
                                 std::to_string(p.num_rows()) + "x" +
                                 std::to_string(p.num_cols()));
    return;
  }
  CHECK(!p.has_quadratic());
  CHECK_NEAR(p.c[0], 3.0, 1e-15);
  CHECK_NEAR(p.c[1], 5.0, 1e-15);
  CHECK_NEAR(p.obj_constant, 1.5, 1e-15);
  CHECK_NEAR(p.row_lower[0], 0.0, 1e-15);
  CHECK_NEAR(p.row_upper[0], 10.0, 1e-15);
  CHECK(!is_finite_bound(p.col_upper[0]));
}

void test_maximize_is_carried() {
  std::string mx(kPaperExample);
  const auto pos = mx.find("Minimize");
  CHECK(pos != std::string::npos);
  if (pos == std::string::npos) return;
  mx.replace(pos, 8, "Maximize");
  const auto p = qplib(mx);
  CHECK(p.sense == core::ObjSense::Maximize);
}

void test_infinity_is_taken_from_the_file() {
  // The file declares its own infinity; a bound at or beyond it is infinite
  // whatever its magnitude. Here it is 1e6, well under our own 1e20 threshold,
  // so a reader that used a hardcoded constant would report a finite 1e6 bound.
  std::string small(kPaperExample);
  auto replace_all = [&](std::string_view from, std::string_view to) {
    std::size_t at = 0;
    while ((at = small.find(from, at)) != std::string::npos) {
      small.replace(at, from.size(), to);
      at += to.size();
    }
  };
  replace_all("1.0E+20", "1.0E+06");

  const auto p = qplib(small);
  if (p.num_rows() != 2) return;
  CHECK(!is_finite_bound(p.row_upper[0]));
  CHECK(!is_finite_bound(p.row_upper[1]));
}

void test_truncated_file_reports_where() {
  const std::string_view text = kPaperExample;
  auto r = load(text.substr(0, text.size() / 2));
  CHECK(!r.has_value());
  if (r.has_value()) return;
  CHECK(r.error().code == core::ErrorCode::ParseError);
  CHECK(r.error().line.has_value());
}

void test_index_out_of_range_is_rejected() {
  std::string bad(kPaperExample);
  const auto pos = bad.find("3 2       variable 3 is binary");
  CHECK(pos != std::string::npos);
  if (pos == std::string::npos) return;
  bad.replace(pos, 3, "9 2");
  auto r = load(bad);
  CHECK(!r.has_value());
  if (r.has_value()) return;
  CHECK(r.error().code == core::ErrorCode::ParseError);
}

}  // namespace

int main() {
  test_paper_example();
  test_positional_slip_is_caught_not_absorbed();
  test_quadratic_constraints_are_refused();
  test_linear_objective_omits_the_hessian_section();
  test_maximize_is_carried();
  test_infinity_is_taken_from_the_file();
  test_truncated_file_reports_where();
  test_index_out_of_range_is_rejected();
  return sovsolve::test::report("qplib_reader");
}
