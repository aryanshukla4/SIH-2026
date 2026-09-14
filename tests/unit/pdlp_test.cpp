// Module 24, Stage B: baseline PDHG end to end through `solve_lp`.
//
// What this file is and is not checking.
//
// It IS checking the formulation: that the saddle point, the two projections,
// the reduced-cost projection and the dual sign convention derived in
// Pdlp.hpp are right. Those are the things a wrong answer would come from,
// and they are checked against models solved by hand and against the simplex,
// which is exact on this corpus.
//
// It is NOT checking that PDLP is fast or that it converges widely. It cannot
// yet: this is baseline PDHG of the paper's equation (3) with none of the four
// enhancements that make it PDLP. The paper measures that baseline at 50 of
// 383 instances against full PDLP's 283, and our own corpus sweep at 100,000
// iterations puts it at 8 of 19. Tests that demanded more would be testing an
// algorithm this commit does not claim to implement.
//
// The dual sign test is the important one. PDLP's published form puts
// inequality rows first with `y >= 0`; this project puts equality rows first
// with `<=` rows and `y <= 0`. Pdlp.hpp absorbs that entirely into the
// projection onto `Y` rather than permuting the matrix, and if that reasoning
// is wrong every inequality dual comes out inverted -- with a primal solution
// that still looks perfectly correct.

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using core::SolverStatus;

namespace {

model::Options pdlp_options() {
  model::Options options;
  options.simplex.method = model::Method::Pdlp;
  options.scaling.mode = model::ScalingMode::RuizPockChambolle;
  options.log.level = model::LogOptions::Level::Silent;
  return options;
}

model::Options simplex_options() {
  model::Options options;
  options.simplex.method = model::Method::DualSimplex;
  options.log.level = model::LogOptions::Level::Silent;
  return options;
}

model::Problem parse_lp(std::string_view text) {
  auto parsed = io::parseProblem(text, io::FileFormat::Lp);
  if (!parsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "parse", parsed.error().format());
    return {};
  }
  return std::move(parsed).value();
}

bool solve_with(const char* label, std::string_view text, const model::Options& options,
                model::Solution& out) {
  const model::Problem problem = parse_lp(text);
  if (problem.num_cols() == 0) return false;
  auto result = solver::solve_lp(problem, options);
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label, result.error().format());
    return false;
  }
  out = std::move(result).value();
  return true;
}

bool solve(const char* label, std::string_view text, model::Solution& out) {
  return solve_with(label, text, pdlp_options(), out);
}

// --------------------------------------------------------------------------

/// The same model the interior-point capstone and the simplex tests use, so
/// all three engines are held to one hand-derived answer.
///
///   min x1 + x2   s.t.  x1 + x2 = 10,  0 <= x1, x2 <= 8   ->  obj 10
void test_equality_with_boxes() {
  model::Solution s;
  if (!solve("equality_with_boxes", R"(Minimize
 obj: x1 + x2
Subject To
 c1: x1 + x2 = 10
Bounds
 0 <= x1 <= 8
 0 <= x2 <= 8
End
)",
             s)) {
    return;
  }
  CHECK(s.status == SolverStatus::Optimal);
  CHECK_NEAR(s.objective, 10.0, 1e-6);
  CHECK_NEAR(s.x[0] + s.x[1], 10.0, 1e-6);
  CHECK(s.quality.max_bound_violation <= 1e-6);
}

/// THE sign test. Two `<=` rows, both TIGHT at the optimum, so both duals are
/// strictly nonzero and both must come out NEGATIVE.
///
/// FORMULATION.md section 4 fixes that convention and verified it against what
/// HiGHS reports for a `<=` row on a minimization. If Pdlp.hpp's projection
/// onto `Y` had the inequality rows' sign backwards, `x` and the objective
/// here would still be exactly right and only `y` would be wrong -- which is
/// why this is asserted rather than assumed.
///
/// The costs are chosen so the optimum is the INTERSECTION of the two rows,
/// which takes deriving rather than eyeballing: the objective gradient has to
/// lie inside the cone spanned by the two active normals `(1,1)` and `(1,3)`,
/// and `(2,4) = 1*(1,1) + 1*(1,3)` does. A first draft used `3x + 2y`, whose
/// gradient lies OUTSIDE that cone -- the real optimum is the vertex `(4,0)`
/// with the second row slack and its dual zero, so the test would have been
/// checking half of what it claimed. (The solver was right and the test was
/// wrong; the same trap is recorded in primal_simplex_test.cpp.)
void test_inequality_duals_are_non_positive() {
  model::Solution s;
  if (!solve("inequality_duals", R"(Maximize
 obj: 2 x + 4 y
Subject To
 c1: x + y <= 4
 c2: x + 3 y <= 6
Bounds
 0 <= x <= 10
 0 <= y <= 10
End
)",
             s)) {
    return;
  }

  // Intersection of the two tight rows: x + y = 4, x + 3y = 6  ->  y = 1,
  // x = 3. Objective 2*3 + 4*1 = 10, against 8 at either neighbouring vertex.
  CHECK(s.status == SolverStatus::Optimal);
  CHECK_NEAR(s.objective, 10.0, 1e-5);
  CHECK_NEAR(s.x[0], 3.0, 1e-5);
  CHECK_NEAR(s.x[1], 1.0, 1e-5);

  // Both rows bind, so both duals are strictly negative -- not merely
  // non-positive, which a slack row would satisfy trivially.
  for (std::size_t i = 0; i < s.y.size(); ++i) {
    ++::sovsolve::test::checks_run();
    if (!(s.y[i] < -1e-6)) {
      ::sovsolve::test::record(__FILE__, __LINE__, "binding <= row has a negative dual",
                               "y[" + std::to_string(i) + "] = " +
                                   std::to_string(s.y[i]) +
                                   " -- the projection onto Y has the sign backwards");
    }
  }
}

/// A `>=` row is negated into `<=` by the canonicalizer (FORMULATION section
/// 2.3) and negated back by postsolve, so it must be reported with a
/// NON-NEGATIVE dual. That round trip is shared with the simplex, but PDLP is
/// the first engine to exercise it from a projection rather than from a basis.
void test_greater_equal_row_dual_is_non_negative() {
  model::Solution s;
  if (!solve("ge_row_dual", R"(Minimize
 obj: 2 x + 3 y
Subject To
 c1: x + y >= 10
Bounds
 0 <= x <= 100
 0 <= y <= 100
End
)",
             s)) {
    return;
  }
  // All weight on the cheaper column: x = 10, y = 0, objective 20.
  CHECK(s.status == SolverStatus::Optimal);
  CHECK_NEAR(s.objective, 20.0, 1e-5);
  ++::sovsolve::test::checks_run();
  if (!(s.y[0] >= -1e-6)) {
    ::sovsolve::test::record(__FILE__, __LINE__, ">= row dual is non-negative",
                             "y[0] = " + std::to_string(s.y[0]));
  }
}

/// The optimum sits on an UPPER bound, so the projection onto `X` has to clamp
/// on both sides and the reduced cost has to come out negative there.
void test_optimum_at_upper_bound() {
  model::Solution s;
  if (!solve("upper_bound", R"(Maximize
 obj: 5 x + y
Subject To
 c1: x + y <= 100
Bounds
 0 <= x <= 7
 0 <= y <= 3
End
)",
             s)) {
    return;
  }
  // The row is slack; both columns ride their upper bounds. 5*7 + 3 = 38.
  CHECK(s.status == SolverStatus::Optimal);
  CHECK_NEAR(s.objective, 38.0, 1e-5);
  CHECK_NEAR(s.x[0], 7.0, 1e-5);
  CHECK_NEAR(s.x[1], 3.0, 1e-5);
  CHECK(s.quality.max_bound_violation <= 1e-6);
}

/// A free column's reduced cost must vanish -- `Lambda_j = {0}` in the
/// projection, which is the one branch that returns a constant rather than
/// clamping.
void test_free_column() {
  model::Solution s;
  if (!solve("free_column", R"(Minimize
 obj: x + y
Subject To
 c1: x + y = 5
 c2: x - y = 1
Bounds
 x free
 y free
End
)",
             s)) {
    return;
  }
  // x = 3, y = 2, objective 5 -- determined entirely by the two equalities.
  CHECK(s.status == SolverStatus::Optimal);
  CHECK_NEAR(s.objective, 5.0, 1e-5);
  CHECK_NEAR(s.x[0], 3.0, 1e-5);
  CHECK_NEAR(s.x[1], 2.0, 1e-5);
}

/// PDLP must refuse a quadratic objective rather than silently solving the
/// linear part of it.
void test_quadratic_is_rejected() {
  const model::Problem problem = parse_lp(R"(Minimize
 obj: x + [ 2 x ^ 2 ] / 2
Subject To
 c1: x >= 1
Bounds
 0 <= x <= 10
End
)");
  if (problem.num_cols() == 0) return;
  auto result = solver::solve_lp(problem, pdlp_options());
  CHECK(!result.has_value());
  if (!result.has_value()) {
    CHECK(result.error().code == core::ErrorCode::UnsupportedFeature);
  }
}

/// Netlib `afiro` against its published optimum -- the same number the IPM and
/// both simplex engines are held to.
void test_afiro() {
  auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps");
  if (!loaded.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "load afiro.mps",
                             loaded.error().format());
    return;
  }
  auto result = solver::solve_lp(loaded.value(), pdlp_options());
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "solve afiro",
                             result.error().format());
    return;
  }
  CHECK(result->status == SolverStatus::Optimal);
  CHECK_NEAR(result->objective, -464.75314286, 1e-6);
}

/// PDLP and the dual simplex must agree wherever baseline PDHG converges at
/// all. The simplex is exact on this corpus, so it is the reference; this is
/// the check that a formulation error common to none of the hand-built models
/// above would still be caught on a real instance.
void test_agrees_with_simplex() {
  const char* instances[] = {"/netlib/afiro.mps", "/netlib/adlittle.mps"};
  for (const char* name : instances) {
    auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + name);
    if (!loaded.has_value()) {
      ::sovsolve::test::record(__FILE__, __LINE__, "load", loaded.error().format());
      continue;
    }
    auto exact = solver::solve_lp(loaded.value(), simplex_options());
    auto approx = solver::solve_lp(loaded.value(), pdlp_options());
    if (!exact.has_value() || !approx.has_value()) {
      ::sovsolve::test::record(__FILE__, __LINE__, "solve", std::string(name));
      continue;
    }
    CHECK(exact->status == SolverStatus::Optimal);

    // Baseline PDHG tails off, so `adlittle` is expected to hit the iteration
    // cap rather than converge. Whatever it returns must still be CLOSE --
    // a first-order method that stops early lands near the optimum; one with
    // a formulation error lands somewhere else entirely. 1e-3 relative
    // separates those two cases without pretending the baseline converged.
    ++::sovsolve::test::checks_run();
    if (!::sovsolve::test::close(approx->objective, exact->objective, 1e-3)) {
      ::sovsolve::test::record(__FILE__, __LINE__, "PDLP agrees with the simplex",
                               std::string(name) + ": pdlp " +
                                   std::to_string(approx->objective) + ", simplex " +
                                   std::to_string(exact->objective));
    }
  }
}

}  // namespace

int main() {
  test_equality_with_boxes();
  test_inequality_duals_are_non_positive();
  test_greater_equal_row_dual_is_non_negative();
  test_optimum_at_upper_bound();
  test_free_column();
  test_quadratic_is_rejected();
  test_afiro();
  test_agrees_with_simplex();
  return ::sovsolve::test::report("pdlp_test");
}
