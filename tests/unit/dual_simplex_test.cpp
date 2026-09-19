// Module 23, Stage 2: the dual simplex, end to end through `solve_lp`.
//
// Three of these tests assert outcomes this project could not produce at all
// before Module 23:
//
//   * `Infeasible` from the SOLVER (only the canonicalizer could return it,
//     and only for a model infeasible by inspection),
//   * `Unbounded` at all (`ConvergenceChecker` never returns it -- README.md
//     records `gas11` running away to -7.5e10 as an architectural gap),
//   * an exact vertex optimum, where the IPM approaches one asymptotically.
//
// Objectives are checked against values derived by hand from each model, not
// against whatever the solver happens to print.

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

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

/// Solve and report the failure in place rather than returning a sentinel the
/// caller has to remember to check.
bool solve(const char* label, std::string_view text, model::Solution& out) {
  const model::Problem problem = parse_lp(text);
  if (problem.num_cols() == 0) return false;
  auto result = solver::solve_lp(problem, simplex_options());
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label, result.error().format());
    return false;
  }
  out = std::move(result).value();
  return true;
}

// --------------------------------------------------------------------------

/// The same model the interior-point capstone test uses, so the two engines
/// are checked against the same hand-derived answer.
///
///   min x1 + x2   s.t.  x1 + x2 = 10,  0 <= x1, x2 <= 8   ->  obj 10
void test_equality_with_boxes() {
  model::Solution solution;
  if (!solve("equality_with_boxes", R"(Minimize
 obj: x1 + x2
Subject To
 c1: x1 + x2 = 10
Bounds
 0 <= x1 <= 8
 0 <= x2 <= 8
End
)",
             solution)) {
    return;
  }

  CHECK(solution.status == SolverStatus::Optimal);
  CHECK_NEAR(solution.objective, 10.0, 1e-9);
  CHECK_NEAR(solution.x[0] + solution.x[1], 10.0, 1e-9);
  CHECK(solution.quality.max_bound_violation <= 1e-9);
  CHECK(solution.quality.primal_infeasibility <= 1e-9);
  CHECK(solution.quality.dual_infeasibility <= 1e-9);
  CHECK(solution.quality.relative_gap <= 1e-9);
}

/// A maximization whose optimum sits on an upper bound, which is the case a
/// solver that only ever rests variables at their lower bound gets wrong.
///
///   max 3a + 5b  s.t.  a <= 4, 2b <= 12, 3a + 2b <= 18,  a, b >= 0
/// The textbook answer is a = 2, b = 6, objective 36.
void test_vertex_at_upper_bound() {
  model::Solution solution;
  if (!solve("vertex_at_upper_bound", R"(Maximize
 obj: 3 a + 5 b
Subject To
 c1: a <= 4
 c2: 2 b <= 12
 c3: 3 a + 2 b <= 18
End
)",
             solution)) {
    return;
  }

  CHECK(solution.status == SolverStatus::Optimal);
  CHECK_NEAR(solution.objective, 36.0, 1e-9);
  CHECK_NEAR(solution.x[0], 2.0, 1e-7);
  CHECK_NEAR(solution.x[1], 6.0, 1e-7);
}

/// A model with no feasible point, whose contradiction is not visible in any
/// single row.
///
///   c1: x + 2y >= 10
///   c2: 2x + y >= 10      adding these gives 3(x + y) >= 20
///   c3: x + y  <= 3       which contradicts it
///
/// Every row has two entries (no singleton-row rule fires) and the two columns
/// have different coefficient patterns (no duplicate-column merge), so
/// presolve cannot reach this by inspection -- the dual simplex has to prove
/// it, by reaching a row whose infeasibility no column can absorb.
///
/// The deliberately naive version of this test (`x + y >= 10` against
/// `x + y <= 4`) does NOT exercise the solver: `x` and `y` are then duplicate
/// columns, presolve merges them, both rows become singletons, and the verdict
/// comes from presolve before a single pivot runs.
void test_infeasible_is_proved() {
  model::Solution solution;
  if (!solve("infeasible", R"(Minimize
 obj: 3 x + 5 y
Subject To
 c1: x + 2 y >= 10
 c2: 2 x + y >= 10
 c3: x + y <= 3
End
)",
             solution)) {
    return;
  }
  CHECK(solution.status == SolverStatus::Infeasible);
}

/// An unbounded model whose ray needs two variables moving together.
///
///   min -x - y   s.t.  x - y <= 1,  -x + y <= 1,  x, y >= 0
///
/// Along `x = y = t` both rows stay satisfied forever while the objective runs
/// to minus infinity. Neither column is empty and the two patterns differ, so
/// presolve's empty-column unboundedness rule cannot fire and the solver has
/// to produce the ray itself.
void test_unbounded_is_detected() {
  model::Solution solution;
  if (!solve("unbounded", R"(Minimize
 obj: - x - y
Subject To
 c1: x - y <= 1
 c2: - x + y <= 1
End
)",
             solution)) {
    return;
  }
  CHECK(solution.status == SolverStatus::Unbounded);
}

/// The same two verdicts when presolve DOES catch them by inspection. They
/// must arrive as a `SolverStatus` on a successful return, not as an error
/// code the caller has to decode -- "this model is infeasible" is an answer.
void test_presolve_verdicts_are_statuses() {
  model::Solution infeasible;
  if (solve("presolve infeasible", R"(Minimize
 obj: x + y
Subject To
 c1: x + y >= 10
 c2: x + y <= 4
End
)",
            infeasible)) {
    CHECK(infeasible.status == SolverStatus::Infeasible);
  }

  model::Solution unbounded;
  if (solve("presolve unbounded", R"(Minimize
 obj: - x
Subject To
 c1: y <= 5
End
)",
            unbounded)) {
    CHECK(unbounded.status == SolverStatus::Unbounded);
  }
}

/// A free variable has no bound to rest on, so it is dual feasible only at a
/// reduced cost of exactly zero. Phase 1 has to box it before the iteration
/// can start; if the box leaks into the answer the optimum comes out wrong.
///
///   min 2p + q  s.t.  p + q = 6,  p >= 0,  q free
/// q free with cost 1 and p costing 2 means all of it goes to q: obj 6.
void test_free_variable() {
  model::Solution solution;
  if (!solve("free_variable", R"(Minimize
 obj: 2 p + q
Subject To
 c1: p + q = 6
Bounds
 q free
End
)",
             solution)) {
    return;
  }

  CHECK(solution.status == SolverStatus::Optimal);
  CHECK_NEAR(solution.objective, 6.0, 1e-7);
  CHECK_NEAR(solution.x[0], 0.0, 1e-7);
  CHECK_NEAR(solution.x[1], 6.0, 1e-7);
}

/// The sign convention, asserted rather than assumed. FORMULATION.md section 4
/// fixes it for a minimization: a `<=` row's dual is non-positive, a `>=`
/// row's is non-negative, and a column resting on its lower bound has a
/// non-negative reduced cost. An earlier draft of that document claimed the
/// opposite, so this is checked directly rather than trusted.
void test_dual_signs() {
  model::Solution solution;
  if (!solve("dual_signs", R"(Minimize
 obj: 2 x + 3 y
Subject To
 c1: x + y >= 4
 c2: x <= 10
End
)",
             solution)) {
    return;
  }

  CHECK(solution.status == SolverStatus::Optimal);
  // Cheapest way to satisfy x + y >= 4 is all x: obj = 8.
  CHECK_NEAR(solution.objective, 8.0, 1e-9);

  CHECK(solution.y.size() == 2);
  if (solution.y.size() == 2) {
    CHECK(solution.y[0] >= -1e-9);  // `>=` row
    CHECK(solution.y[1] <= 1e-9);   // `<=` row
  }
  // y rests on its lower bound at 0, so its reduced cost must be >= 0.
  CHECK(solution.reduced_cost(1) >= -1e-9);
}

/// Degenerate: more constraints bind at the optimum than there are variables,
/// so the ratio test repeatedly produces zero-length steps. The requirement is
/// that it TERMINATES with the right answer, which is the property a naive
/// implementation loses first.
///
///   min x + y  s.t.  x + y >= 2, x >= 1, y >= 1, x + 2y >= 3
/// Every row is tight at (1, 1); objective 2.
void test_degenerate_terminates() {
  model::Solution solution;
  if (!solve("degenerate", R"(Minimize
 obj: x + y
Subject To
 c1: x + y >= 2
 c2: x >= 1
 c3: y >= 1
 c4: x + 2 y >= 3
End
)",
             solution)) {
    return;
  }

  CHECK(solution.status == SolverStatus::Optimal);
  CHECK_NEAR(solution.objective, 2.0, 1e-9);
  CHECK_NEAR(solution.x[0], 1.0, 1e-7);
  CHECK_NEAR(solution.x[1], 1.0, 1e-7);
}

/// Bound flipping: many boxed columns whose reduced costs all cross zero at
/// nearby ratios is exactly the structure the long-step ratio test exists for.
/// Checked for the right answer with the test on and off -- a ratio test that
/// is fast and wrong is worse than the textbook one.
void test_bound_flipping_agrees_with_textbook() {
  const std::string text = R"(Minimize
 obj: x1 + 2 x2 + 3 x3 + 4 x4 + 5 x5
Subject To
 c1: x1 + x2 + x3 + x4 + x5 >= 7
 c2: x1 + 2 x2 + x3 >= 3
Bounds
 0 <= x1 <= 2
 0 <= x2 <= 2
 0 <= x3 <= 2
 0 <= x4 <= 2
 0 <= x5 <= 2
End
)";

  const model::Problem problem = parse_lp(text);
  if (problem.num_cols() == 0) return;

  model::Options with_flips = simplex_options();
  model::Options without_flips = simplex_options();
  without_flips.simplex.bound_flipping = false;

  auto a = solver::solve_lp(problem, with_flips);
  auto b = solver::solve_lp(problem, without_flips);
  if (!a.has_value() || !b.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "bound flipping solve",
                             !a.has_value() ? a.error().format() : b.error().format());
    return;
  }

  CHECK(a->status == SolverStatus::Optimal);
  CHECK(b->status == SolverStatus::Optimal);
  // Cheapest 7 units: x1,x2,x3 at 2 costs 1*2+2*2+3*2 = 12 and already
  // satisfies c2; the remaining 1 unit goes to x4 at 4. Total 16.
  CHECK_NEAR(a->objective, 16.0, 1e-9);
  CHECK_NEAR(b->objective, 16.0, 1e-9);
}

/// The real thing: Netlib `afiro`, against its published optimum. This is the
/// instance README.md's corpus table uses as the interior-point path's
/// reference, so the two engines are being held to the same number.
void test_afiro() {
  auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps");
  if (!loaded.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "load afiro.mps",
                             loaded.error().format());
    return;
  }

  auto result = solver::solve_lp(loaded.value(), simplex_options());
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "solve afiro",
                             result.error().format());
    return;
  }

  CHECK(result->status == SolverStatus::Optimal);
  // Netlib's published optimum for afiro.
  CHECK_NEAR(result->objective, -464.75314286, 1e-6);
  CHECK(result->quality.max_bound_violation <= 1e-7);
  CHECK(result->quality.primal_infeasibility <= 1e-8);
  CHECK(result->quality.dual_infeasibility <= 1e-8);
}


/// REGRESSION: a FEASIBLE model the bound-flipping ratio test called
/// infeasible, found by milp_test's enumeration oracle (seed 23).
///
/// This is the LP relaxation of that seed's integer program. Row r0 can be met
/// only at its boundary: over the variable box its left-hand side reaches 12 at
/// most, and only at x0 = 1, x2 = 4, x3 = 1 -- the single feasible point. So on
/// the first iteration the boxed columns' flips close r0's violation EXACTLY.
///
/// On the integer data that cancels to zero and the last column enters. After
/// geometric scaling it cancels to ~1e-16, and the test `reduction < remaining`
/// flipped the last column too, left no candidate to enter, and reached the
/// "nothing blocks the dual step" exit -- reporting Infeasible, with a
/// certificate that fails arXiv 2102.04592 (50) at every tolerance up to 0.1.
/// The primal simplex, with no such exit, found the optimum.
///
/// The fix passes a breakpoint only while the violation stays open by more than
/// the primal feasibility tolerance -- the threshold `choose_leaving` already
/// uses to call a row violated at all.
///
/// Caveat recorded rather than hidden: this reproduces through the solver's own
/// default scaling, which is what produces the residual. A change to the
/// scaler could make it pass without exercising the exit; the checks below at
/// least pin that the answer is right either way.
void test_flips_that_close_the_violation_exactly() {
  model::Solution solution;
  if (!solve("exact flips", R"(Maximize
 obj: - 6 x0 + 1 x1 - 1 x2 + 1 x3 + 2 x4
Subject To
 r0: + 3 x0 - 5 x1 + 3 x2 - 3 x3 - 2 x4 >= 12
 r1: - 2 x0 + 1 x1 - 3 x2 + 0 x3 - 3 x4 <= 5
 r2: - 2 x0 + 1 x1 - 3 x2 - 5 x3 - 3 x4 <= -4
Bounds
 -1 <= x0 <= 1
 0 <= x1 <= 3
 0 <= x2 <= 4
 1 <= x3 <= 5
 0 <= x4 <= 2
End
)",
             solution)) {
    return;
  }
  CHECK(solution.status == SolverStatus::Optimal);
  CHECK_NEAR(solution.objective, -9.0, 1e-7);
  if (solution.x.size() == 5) {
    CHECK_NEAR(solution.x[0], 1.0, 1e-7);
    CHECK_NEAR(solution.x[2], 4.0, 1e-7);
    CHECK_NEAR(solution.x[3], 1.0, 1e-7);
  }
}

// --------------------------------------------------------------------------
// Cost perturbation (Koberstein thesis section 6.3.1, SolveSimplex.cpp)
// --------------------------------------------------------------------------

model::Options perturbation_options(bool on) {
  model::Options o = simplex_options();
  o.simplex.cost_perturbation = on;
  return o;
}

bool solve_with(const char* label, std::string_view text, const model::Options& o,
                model::Solution& out) {
  const model::Problem problem = parse_lp(text);
  if (problem.num_cols() == 0) return false;
  auto result = solver::solve_lp(problem, o);
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label, result.error().format());
    return false;
  }
  out = std::move(result).value();
  return true;
}

/// A 5x5 assignment problem with costs drawn from {1, 2, 3, 4}: four distinct
/// values over 25 columns, so the thesis's test (fewer than n/4 distinct) fires.
/// Assignment LPs are integral, so the LP optimum is the best permutation --
/// 6, found by enumerating all 120.
///
/// The check that matters is EXACT: the dual runs on perturbed costs, and if
/// the perturbation were not removed the reported objective would be 6 plus a
/// few 1e-5s. Tolerance 1e-9 catches that; 1e-4 would not.
const char* kDegenerateAssignment = R"(Minimize
 obj: + 3 x00 + 2 x01 + 4 x02 + 1 x03 + 1 x04 + 1 x10 + 3 x11 + 1 x12 + 2 x13 + 1 x14 + 1 x20 + 4 x21 + 4 x22 + 1 x23 + 2 x24 + 1 x30 + 4 x31 + 1 x32 + 1 x33 + 2 x34 + 1 x40 + 4 x41 + 1 x42 + 2 x43 + 1 x44
Subject To
 r0: + x00 + x01 + x02 + x03 + x04 = 1
 r1: + x10 + x11 + x12 + x13 + x14 = 1
 r2: + x20 + x21 + x22 + x23 + x24 = 1
 r3: + x30 + x31 + x32 + x33 + x34 = 1
 r4: + x40 + x41 + x42 + x43 + x44 = 1
 c0: + x00 + x10 + x20 + x30 + x40 = 1
 c1: + x01 + x11 + x21 + x31 + x41 = 1
 c2: + x02 + x12 + x22 + x32 + x42 = 1
 c3: + x03 + x13 + x23 + x33 + x43 = 1
 c4: + x04 + x14 + x24 + x34 + x44 = 1
Bounds
 0 <= x00 <= 1
 0 <= x01 <= 1
 0 <= x02 <= 1
 0 <= x03 <= 1
 0 <= x04 <= 1
 0 <= x10 <= 1
 0 <= x11 <= 1
 0 <= x12 <= 1
 0 <= x13 <= 1
 0 <= x14 <= 1
 0 <= x20 <= 1
 0 <= x21 <= 1
 0 <= x22 <= 1
 0 <= x23 <= 1
 0 <= x24 <= 1
 0 <= x30 <= 1
 0 <= x31 <= 1
 0 <= x32 <= 1
 0 <= x33 <= 1
 0 <= x34 <= 1
 0 <= x40 <= 1
 0 <= x41 <= 1
 0 <= x42 <= 1
 0 <= x43 <= 1
 0 <= x44 <= 1
End
)";

void test_perturbation_is_removed_before_reporting() {
  for (bool on : {false, true}) {
    model::Solution sol;
    if (!solve_with("assignment", kDegenerateAssignment, perturbation_options(on), sol)) {
      return;
    }
    CHECK(sol.status == SolverStatus::Optimal);
    CHECK_NEAR(sol.objective, 6.0, 1e-9);
  }
}

/// A Farkas certificate involves A, b and the bounds -- never the costs -- so
/// an Infeasible verdict reached on perturbed costs stands. Five columns of
/// equal cost trigger the perturbation.
void test_perturbation_keeps_infeasible() {
  model::Solution sol;
  if (!solve_with("infeasible", R"(Minimize
 obj: x1 + x2 + x3 + x4 + x5
Subject To
 lo: x1 + x2 + x3 + x4 + x5 >= 10
 hi: x1 + x2 + x3 + x4 + x5 <= 3
Bounds
 0 <= x1 <= 10
 0 <= x2 <= 10
 0 <= x3 <= 10
 0 <= x4 <= 10
 0 <= x5 <= 10
End
)",
                  perturbation_options(true), sol)) {
    return;
  }
  CHECK(sol.status == SolverStatus::Infeasible);
}

/// Unboundedness DOES depend on the costs, so it must be confirmed on the true
/// ones. Equal costs trigger the perturbation; x3 escapes along the row.
void test_perturbation_confirms_unbounded() {
  model::Solution sol;
  if (!solve_with("unbounded", R"(Minimize
 obj: - x1 - x2 - x3 - x4 - x5
Subject To
 r: x1 + x2 - x3 - x4 - x5 <= 1
End
)",
                  perturbation_options(true), sol)) {
    return;
  }
  CHECK(sol.status == SolverStatus::Unbounded);
}

/// Distinct costs: the thesis's test does not fire, and the solve is the SAME
/// solve -- same objective and the same pivot count -- with the switch on or
/// off. Netlib showed exactly this on every instance that does not qualify.
void test_perturbation_leaves_distinct_costs_alone() {
  const char* model_text = R"(Minimize
 obj: 3 x + 5 y + 7 z + 11 w
Subject To
 a: x + y + z + w >= 4
 b: 2 x + y - z >= 1
 c: x + 3 w <= 9
Bounds
 0 <= x <= 5
 0 <= y <= 5
 0 <= z <= 5
 0 <= w <= 5
End
)";
  model::Solution off;
  model::Solution on;
  if (!solve_with("distinct off", model_text, perturbation_options(false), off)) return;
  if (!solve_with("distinct on", model_text, perturbation_options(true), on)) return;
  CHECK(off.status == SolverStatus::Optimal);
  CHECK(on.status == SolverStatus::Optimal);
  CHECK_NEAR(on.objective, off.objective, 0.0);
  CHECK_EQ(on.iterations, off.iterations);
}
}  // namespace

int main() {
  test_equality_with_boxes();
  test_vertex_at_upper_bound();
  test_infeasible_is_proved();
  test_unbounded_is_detected();
  test_presolve_verdicts_are_statuses();
  test_free_variable();
  test_dual_signs();
  test_degenerate_terminates();
  test_bound_flipping_agrees_with_textbook();
  test_flips_that_close_the_violation_exactly();
  test_perturbation_is_removed_before_reporting();
  test_perturbation_keeps_infeasible();
  test_perturbation_confirms_unbounded();
  test_perturbation_leaves_distinct_costs_alone();
  test_afiro();
  return sovsolve::test::report("dual_simplex");
}
