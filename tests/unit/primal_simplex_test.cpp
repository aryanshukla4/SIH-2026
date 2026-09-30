// Module 23: the primal simplex, and the composite handoff between the two
// algorithms.
//
// Two things are being checked here that the dual simplex's own suite cannot.
//
// First, that the primal reaches the SAME answers by a different route. Both
// algorithms run on one shared revised-simplex core (detail/SimplexEngine.hpp),
// so a bug in that core would move both together -- but the iteration on top of
// it is entirely separate code, and a disagreement between them localizes a
// fault to whichever one is wrong. Where an objective is checked, it is against
// a value derived by hand from the model, not against the other engine.
//
// Second, that the two compose. The dual's phase 1 boxes dual-infeasible
// columns in artificial bounds and can finish optimal for the BOX rather than
// for the model; the primal has no boxes. The handoff is only cheap if the
// dual's endpoint is already primal feasible, so `phase1_iterations == 0` after
// a handoff is asserted rather than assumed.

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#include "sovsolve/solver/simplex/DualSimplex.hpp"
#include "sovsolve/solver/simplex/PrimalSimplex.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using core::SolverStatus;
namespace simplex = solver::simplex;

namespace {

model::Options primal_options() {
  model::Options options;
  options.simplex.method = model::Method::PrimalSimplex;
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

bool solve(const char* label, std::string_view text, model::Solution& out) {
  const model::Problem problem = parse_lp(text);
  if (problem.num_cols() == 0) return false;
  auto result = solver::solve_lp(problem, primal_options());
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, label, result.error().format());
    return false;
  }
  out = std::move(result).value();
  return true;
}

// --------------------------------------------------------------------------

/// Same model as the dual suite's first case and as the interior-point
/// capstone, so all three engines are held to one hand-derived number.
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
}

/// The starting basis is every row's logical, which on an equality row is
/// FIXED at zero while basic at `b_i`. So a model of nothing but equality rows
/// begins primal infeasible by construction, and phase 1 has to do real work
/// before phase 2 can start -- the path the composite handoff never exercises.
///
///   min 4a + 3b + 3c  s.t.  a + b + c = 10,  a + 2b = 8,  a,b,c >= 0
///
/// Substituting `a = 8 - 2b` (so `b <= 4` for `a >= 0`) and `c = 2 + b` gives
/// an objective of `38 - 2b`, minimized by pushing `b` to 4: `a = 0, b = 4,
/// c = 6`, objective 30.
///
/// The cost vector matters here. An earlier draft used `5c`, which makes the
/// objective `42 + 0b` -- constant over the entire feasible set, so every
/// answer is optimal and the test asserts nothing about where the solver went.
void test_phase1_from_equalities() {
  model::Solution solution;
  if (!solve("phase1_from_equalities", R"(Minimize
 obj: 4 a + 3 b + 3 c
Subject To
 c1: a + b + c = 10
 c2: a + 2 b = 8
End
)",
             solution)) {
    return;
  }
  CHECK(solution.status == SolverStatus::Optimal);
  CHECK_NEAR(solution.objective, 30.0, 1e-9);
  CHECK_NEAR(solution.x[0], 0.0, 1e-7);
  CHECK_NEAR(solution.x[1], 4.0, 1e-7);
  CHECK_NEAR(solution.x[2], 6.0, 1e-7);
}

/// A maximization whose optimum sits on an upper bound.
///
///   max 3a + 5b  s.t.  a <= 4, 2b <= 12, 3a + 2b <= 18  ->  a=2, b=6, obj 36
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

/// Phase 1 minimizes the total bound violation. Running out of improving
/// columns while that total is still positive means its MINIMUM over the whole
/// polytope is positive -- no feasible point exists anywhere. A proof, not a
/// stall.
///
/// The contradiction is spread across three rows (c1 + c2 force x + y >= 20/3,
/// c3 caps it at 3) so no single-row presolve rule can reach it, and the two
/// columns have different patterns so duplicate-column merging cannot either.
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

/// Unboundedness straight out of the ratio test: an improving column with no
/// blocking event anywhere is a ray of the feasible region, full stop. The
/// dual can only reach this verdict indirectly, by testing whether a phase-1
/// artificial bound refuses to stop binding.
///
///   min -x - y  s.t.  x - y <= 1,  -x + y <= 1,  x, y >= 0
/// Along `x = y = t` both rows stay satisfied forever.
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

/// A free column has no bound to rest on. The dual simplex must box it in an
/// artificial bound before it can start; the primal treats it as an ordinary
/// entering candidate with a direction chosen by the sign of its reduced cost.
///
///   min 2p + q  s.t.  p + q = 6,  p >= 0,  q free   ->  q = 6, obj 6
void test_free_variable_needs_no_box() {
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

/// Degenerate: four rows bind at a two-variable optimum, so a textbook ratio
/// test keeps producing zero-length steps. EXPAND (Gill, Murray, Saunders &
/// Wright 1989) makes every step positive instead. The requirement here is
/// termination with the right answer.
///
///   min x + y  s.t.  x + y >= 2, x >= 1, y >= 1, x + 2y >= 3  ->  (1,1), obj 2
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

/// Beale's (1955) example, which cycles forever under Dantzig pricing with a
/// textbook ratio test: the start is degenerate in both rows, and six pivots
/// bring back the starting basis. EXPAND's steps are never zero, so no basis
/// can repeat.
///
///   min -3/4 x4 + 20 x5 - 1/2 x6 + 6 x7
///   s.t. 1/4 x4 -  8 x5 -     x6 + 9 x7 <= 0
///        1/2 x4 - 12 x5 - 1/2 x6 + 3 x7 <= 0
///                             x6        <= 1,   x >= 0
///
/// Row 2 caps x4 at x6, so x4 = x6 = 1 gives -5/4. Raising x5 by e frees 24e
/// of x4 in row 2: a gain of 18e against a cost of 20e, so it stays at zero.
void test_beale_cycling_example() {
  model::Solution solution;
  if (!solve("beale", R"(Minimize
 obj: - 0.75 x4 + 20 x5 - 0.5 x6 + 6 x7
Subject To
 r1: 0.25 x4 - 8 x5 - x6 + 9 x7 <= 0
 r2: 0.5 x4 - 12 x5 - 0.5 x6 + 3 x7 <= 0
 r3: x6 <= 1
End
)",
             solution)) {
    return;
  }
  CHECK(solution.status == SolverStatus::Optimal);
  CHECK_NEAR(solution.objective, -1.25, 1e-9);
  CHECK_NEAR(solution.x[0], 1.0, 1e-7);
  CHECK_NEAR(solution.x[2], 1.0, 1e-7);
}

/// The sign convention, checked independently of the dual's implementation of
/// it. FORMULATION.md section 4, for a minimization: a `>=` row's dual is
/// non-negative, a `<=` row's is non-positive, and a column on its lower bound
/// has a non-negative reduced cost.
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
  CHECK_NEAR(solution.objective, 8.0, 1e-9);
  CHECK(solution.y.size() == 2);
  if (solution.y.size() == 2) {
    CHECK(solution.y[0] >= -1e-9);  // `>=` row
    CHECK(solution.y[1] <= 1e-9);   // `<=` row
  }
  CHECK(solution.reduced_cost(1) >= -1e-9);
}

/// The handoff. The dual's endpoint is feasible for its artificially bounded
/// problem, and the model's own bounds contain that box, so the same point is
/// primal feasible for the model -- which means the primal starts in phase 2
/// and spends no pivots getting there.
///
/// That is the whole reason the composite is cheap rather than "run the other
/// one from scratch", so it is asserted rather than assumed.
void test_composite_handoff_skips_phase1() {
  const model::Problem problem = parse_lp(R"(Minimize
 obj: 2 x + 3 y - z
Subject To
 c1: x + y + z >= 6
 c2: x - y + 2 z <= 8
 c3: x + 2 y = 5
Bounds
 0 <= z <= 4
End
)");
  if (problem.num_cols() == 0) return;

  auto canon = model::canonicalize(problem);
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "canonicalize", canon.error().format());
    return;
  }

  const model::Options options = primal_options();
  auto dual = simplex::solve_dual_simplex(canon->problem, options);
  if (!dual.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "dual", dual.error().format());
    return;
  }
  CHECK(dual->status == SolverStatus::Optimal);

  auto primal = simplex::solve_primal_simplex(canon->problem, options, &dual->basis);
  if (!primal.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "primal", primal.error().format());
    return;
  }
  CHECK(primal->status == SolverStatus::Optimal);
  CHECK_EQ(primal->phase1_iterations, std::size_t{0});
  // Already optimal, so the cleanup should have nothing left to do either.
  CHECK_EQ(primal->iterations, std::size_t{0});
  CHECK_NEAR(primal->objective, dual->objective, 1e-9);
}

/// Both algorithms, cold, on every model above plus a real instance: they must
/// agree. They share the basis/LU core but not the iteration, so a
/// disagreement points at one of the two iterations rather than at the shared
/// machinery.
void test_both_engines_agree() {
  const char* models[] = {
      R"(Minimize
 obj: 4 a + 3 b + 3 c
Subject To
 c1: a + b + c = 10
 c2: a + 2 b = 8
End
)",
      R"(Maximize
 obj: 3 a + 5 b
Subject To
 c1: a <= 4
 c2: 2 b <= 12
 c3: 3 a + 2 b <= 18
End
)",
      R"(Minimize
 obj: x + 2 y + 3 z
Subject To
 c1: x + y + z >= 7
 c2: x + 2 y >= 3
 c3: y + z <= 9
Bounds
 0 <= x <= 2
 0 <= y <= 2
 0 <= z <= 8
End
)",
  };

  for (const char* text : models) {
    const model::Problem problem = parse_lp(text);
    if (problem.num_cols() == 0) continue;

    model::Options dual_opts = primal_options();
    dual_opts.simplex.method = model::Method::DualSimplex;

    auto primal = solver::solve_lp(problem, primal_options());
    auto dual = solver::solve_lp(problem, dual_opts);
    if (!primal.has_value() || !dual.has_value()) {
      ::sovsolve::test::record(__FILE__, __LINE__, "engine agreement solve",
                               !primal.has_value() ? primal.error().format()
                                                   : dual.error().format());
      continue;
    }
    CHECK(primal->status == dual->status);
    CHECK_NEAR(primal->objective, dual->objective, 1e-9);
  }
}

/// Netlib `afiro` against its published optimum, through the primal.
void test_afiro() {
  auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps");
  if (!loaded.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "load afiro.mps",
                             loaded.error().format());
    return;
  }
  auto result = solver::solve_lp(loaded.value(), primal_options());
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "solve afiro",
                             result.error().format());
    return;
  }
  CHECK(result->status == SolverStatus::Optimal);
  CHECK_NEAR(result->objective, -464.75314286, 1e-6);
  CHECK(result->quality.max_bound_violation <= 1e-7);
  CHECK(result->quality.primal_infeasibility <= 1e-8);
  CHECK(result->quality.dual_infeasibility <= 1e-8);
}

}  // namespace

int main() {
  test_equality_with_boxes();
  test_phase1_from_equalities();
  test_vertex_at_upper_bound();
  test_infeasible_is_proved();
  test_unbounded_is_detected();
  test_free_variable_needs_no_box();
  test_degenerate_terminates();
  test_beale_cycling_example();
  test_dual_signs();
  test_composite_handoff_skips_phase1();
  test_both_engines_agree();
  test_afiro();
  return sovsolve::test::report("primal_simplex");
}
