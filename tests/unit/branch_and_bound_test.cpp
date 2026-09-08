// Behavioural tests for Module 22 (branch-and-bound over MILP, see
// module.txt and BranchAndBound.hpp).
//
// A separate executable, registered only when SOVSOLVE_ENABLE_CUDA is on --
// same reason as solver_gpu_algorithms_test: BranchAndBound.cu lives in
// sovsolve_solver_gpu, calling solve_problem per node, which does not exist
// in a host-only build.

#include <cmath>
#include <cstddef>

#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/solver/gpu/BranchAndBound.hpp"
#include "sovsolve/solver/gpu/Solve.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)

namespace {

// min -5*x1 - 4*x2  s.t.  6*x1 + 5*x2 <= 10,  x1,x2 in {0,1}.
//
// Hand-derived optimum: LP relaxation picks x1=1, x2=0.8 (obj -8.2), and
// branching on x2 explores exactly the four integer corners feasibility
// allows -- (1,0)->-5, (0,1)->-4, (0,0)->0, (1,1) infeasible (6+5=11>10).
// The true MILP optimum is x1=1, x2=0, objective -5. Confirmed independently
// via the CLI (tools/solve) before this test was written, matching to
// 1e-8: status=Optimal, objective=-4.9999999987, best_bound equal to it.
model::Problem make_tiny_knapsack() {
  core::SparseBuilder builder(1, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 6.0);
  builder.insert(0, 1, 5.0);

  model::Problem problem;
  problem.sense = core::ObjSense::Minimize;
  problem.c = core::RealVector(2);
  problem.c[0] = -5.0;
  problem.c[1] = -4.0;
  problem.A = builder.finish();
  problem.row_lower = core::RealVector(1, -core::INF);
  problem.row_upper = core::RealVector(1, 10.0);
  problem.col_lower = core::RealVector(2, 0.0);
  problem.col_upper = core::RealVector(2, 1.0);
  problem.col_type = {core::VarType::Binary, core::VarType::Binary};
  return problem;
}

// Same knapsack, but tagged VarType::Integer with [0,1] bounds instead of
// VarType::Binary -- exactly how an MPS file declares a 0/1 variable via
// `INTORG ... UP bnd x 1` rather than the explicit `BV` bound type
// (MpsReader.cpp; markshare_4_0.mps uses this exact declaration for all 30
// of its variables). Cover-cut eligibility originally checked
// `col_type == Binary` alone, which silently rejected every column here --
// a real, caught bug (BranchAndBound.cu's `is_binary_like`) that meant
// cover-cut separation never fired on markshare_4_0.mps at all.
model::Problem make_tiny_knapsack_integer_typed() {
  auto problem = make_tiny_knapsack();
  problem.col_type = {core::VarType::Integer, core::VarType::Integer};
  return problem;
}

void test_branch_and_bound_accepts_integer_typed_zero_one_columns() {
  const auto problem = make_tiny_knapsack_integer_typed();

  model::Options options;
  auto result = solver::gpu::solve(problem, options);
  CHECK(result.has_value());
  if (!result.has_value()) return;

  CHECK(result->status == core::SolverStatus::Optimal);
  CHECK_NEAR(result->objective, -5.0, 1e-4);
  CHECK_NEAR(result->best_bound, result->objective, 1e-4);
  CHECK_NEAR(result->x[0], 1.0, 1e-4);
  CHECK_NEAR(result->x[1], 0.0, 1e-4);
}

// x1 + x2 = 10, 0 <= x1,x2 <= 8, min x1 + x2 -- purely continuous, no
// discrete columns. Same shape solve_gpu_algorithms_test's own boxed-
// equality fixture uses, kept local so this file has no cross-file
// dependency on that one.
model::Problem make_continuous_problem() {
  core::SparseBuilder builder(1, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 1.0);

  model::Problem problem;
  problem.sense = core::ObjSense::Minimize;
  problem.c = core::RealVector(2);
  problem.c[0] = 1.0;
  problem.c[1] = 1.0;
  problem.A = builder.finish();
  problem.row_lower = core::RealVector(1, 10.0);
  problem.row_upper = core::RealVector(1, 10.0);
  problem.col_lower = core::RealVector(2, 0.0);
  problem.col_upper = core::RealVector(2, 8.0);
  problem.col_type = {core::VarType::Continuous, core::VarType::Continuous};
  return problem;
}

// min -9*x1 - 6*x2 - 4*x3  s.t.  10*x1 + 6*x2 + 4*x3 <= 15,  x1,x2,x3 in
// {0,1}. Deliberately two branching levels deep (unlike the two-variable
// fixture above) so a warm start actually gets exercised across a
// GRANDCHILD node -- one whose own warm-start hint came from a parent that
// was itself warm-started, not straight from the root.
//
// Hand-derived optimum, exhaustively checked against all 8 integer points:
// root LP relaxation packs x2,x3 fully (ratio 1.0 each) then x1 partially
// (x1=0.5, obj=-14.5); branching x1<=0 gives (0,1,1)->-10 (integral
// immediately); branching x1>=1 fixes x1=1 (capacity 5 left for x2,x3),
// whose OWN relaxation is again fractional (x2=0.833, obj=-14) and branches
// again: x2<=0 gives (1,0,1)->-13 (integral), x2>=1 needs weight 6 against
// remaining capacity 5 -> infeasible. Best of all integer-feasible points:
// -13 at (1,0,1) (checked against every other feasible combination by hand:
// (0,1,1)=-10, (1,0,0)=-9, (0,1,0)=-6, (0,0,1)=-4, (0,0,0)=0; (1,1,0) and
// (1,1,1) both exceed capacity 15).
model::Problem make_two_level_knapsack() {
  core::SparseBuilder builder(1, 3);
  builder.count(0, 0);
  builder.count(0, 1);
  builder.count(0, 2);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 10.0);
  builder.insert(0, 1, 6.0);
  builder.insert(0, 2, 4.0);

  model::Problem problem;
  problem.sense = core::ObjSense::Minimize;
  problem.c = core::RealVector(3);
  problem.c[0] = -9.0;
  problem.c[1] = -6.0;
  problem.c[2] = -4.0;
  problem.A = builder.finish();
  problem.row_lower = core::RealVector(1, -core::INF);
  problem.row_upper = core::RealVector(1, 15.0);
  problem.col_lower = core::RealVector(3, 0.0);
  problem.col_upper = core::RealVector(3, 1.0);
  problem.col_type = {core::VarType::Binary, core::VarType::Binary, core::VarType::Binary};
  return problem;
}

// min s  s.t.  s + 2*x1 = 5,  x1 in {0,1},  s continuous >= 0 (no upper
// bound) -- markshare_4_0.mps's exact row shape in miniature: one
// absorbing continuous "deviation" column plus otherwise-discrete columns
// in an equality row (MilpPresolve.hpp's
// eliminate_equality_row_absorbing_singletons).
//
// Hand-derived optimum: s = 5 - 2*x1, minimized by maximizing x1 (its
// coefficient in the row is positive) -- x1=1 gives s=3 (objective 3),
// x1=0 gives s=5 (worse). This is exactly the class of case that exposed a
// real bug during development: without folding s's cost into x1's own
// cost when s is eliminated from the row, the solver treated x1 as
// costless and reported an unreachable objective of 0 (s "fixed" at its
// own lower bound, 0, ignoring that the row requires s=5-2*x1 -- 0 would
// need x1=2.5, impossible for a binary variable).
model::Problem make_absorbing_singleton_problem() {
  core::SparseBuilder builder(1, 2);
  builder.count(0, 0);  // s
  builder.count(0, 1);  // x1
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(0, 1, 2.0);

  model::Problem problem;
  problem.sense = core::ObjSense::Minimize;
  problem.c = core::RealVector(2);
  problem.c[0] = 1.0;  // s
  problem.c[1] = 0.0;  // x1 -- costless on its own; its true cost is entirely via s
  problem.A = builder.finish();
  problem.row_lower = core::RealVector(1, 5.0);
  problem.row_upper = core::RealVector(1, 5.0);
  problem.col_lower = core::RealVector(2);
  problem.col_upper = core::RealVector(2);
  problem.col_lower[0] = 0.0;
  problem.col_upper[0] = core::INF;  // s: lower-only
  problem.col_lower[1] = 0.0;
  problem.col_upper[1] = 1.0;
  problem.col_type = {core::VarType::Continuous, core::VarType::Binary};
  return problem;
}

void test_branch_and_bound_folds_absorbing_singleton_cost_correctly() {
  const auto problem = make_absorbing_singleton_problem();

  model::Options options;
  auto result = solver::gpu::solve(problem, options);
  CHECK(result.has_value());
  if (!result.has_value()) return;

  CHECK(result->status == core::SolverStatus::Optimal);
  CHECK_NEAR(result->objective, 3.0, 1e-4);
  CHECK_NEAR(result->best_bound, result->objective, 1e-4);
  CHECK_EQ(result->x.size(), std::size_t{2});
  CHECK_NEAR(result->x[1], 1.0, 1e-4);  // x1 = 1
  // s's TRUE value from the row equation (5 - 2*1 = 3), not an arbitrary
  // bound-fixed value (0) that would silently violate the original row.
  CHECK_NEAR(result->x[0], 3.0, 1e-4);
}

void test_branch_and_bound_warm_start_survives_two_branching_levels() {
  const auto problem = make_two_level_knapsack();

  model::Options options;
  auto result = solver::gpu::solve(problem, options);
  CHECK(result.has_value());
  if (!result.has_value()) return;

  CHECK(result->status == core::SolverStatus::Optimal);
  CHECK_NEAR(result->objective, -13.0, 1e-4);
  CHECK_NEAR(result->best_bound, result->objective, 1e-4);
  CHECK_EQ(result->x.size(), std::size_t{3});
  CHECK_NEAR(result->x[0], 1.0, 1e-4);
  CHECK_NEAR(result->x[1], 0.0, 1e-4);
  CHECK_NEAR(result->x[2], 1.0, 1e-4);
  // Not asserting an exact/minimum node count: GCD tightening fires here
  // too (gcd(10,6,4)=2 tightens the row's capacity 15 -> 14), which can
  // change the relaxation enough to need fewer nodes than the hand-traced,
  // untightened tree above -- a real improvement, not a reason to pin the
  // node count down. The answer itself (checked above) is what must be
  // exact.
  CHECK(result->nodes_explored > 0);
}

void test_branch_and_bound_solves_tiny_knapsack_to_proven_optimal() {
  const auto problem = make_tiny_knapsack();

  model::Options options;
  auto result = solver::gpu::solve(problem, options);
  CHECK(result.has_value());
  if (!result.has_value()) return;

  CHECK(result->status == core::SolverStatus::Optimal);
  CHECK(!result->from_best_iterate);
  CHECK_NEAR(result->objective, -5.0, 1e-4);
  // Proof of optimality: the incumbent and the best remaining relaxation
  // bound met exactly (BranchAndBound.cu's best-first termination).
  CHECK_NEAR(result->best_bound, result->objective, 1e-4);
  CHECK_EQ(result->x.size(), std::size_t{2});
  CHECK_NEAR(result->x[0], 1.0, 1e-4);
  CHECK_NEAR(result->x[1], 0.0, 1e-4);
  CHECK(result->nodes_explored > 0);
}

void test_branch_and_bound_dispatch_matches_solve_problem_on_continuous_model() {
  // solve() must behave EXACTLY like solve_problem() for a model with no
  // discrete columns -- this is the whole point of gating branch-and-bound
  // on has_discrete() in Solve(), rather than something that could
  // accidentally change LP/QP behaviour.
  const auto problem = make_continuous_problem();
  model::Options options;

  auto via_dispatch = solver::gpu::solve(problem, options);
  auto direct = solver::gpu::solve_problem(problem, options);
  CHECK(via_dispatch.has_value());
  CHECK(direct.has_value());
  if (!via_dispatch.has_value() || !direct.has_value()) return;

  CHECK(via_dispatch->status == direct->status);
  CHECK_NEAR(via_dispatch->objective, direct->objective, 1e-9);
  CHECK_EQ(via_dispatch->nodes_explored, std::size_t{0});
}

void test_branch_and_bound_rejects_semi_continuous_columns() {
  auto problem = make_tiny_knapsack();
  problem.col_type[0] = core::VarType::SemiContinuous;

  model::Options options;
  auto result = solver::gpu::solve(problem, options);
  CHECK(!result.has_value());
  if (result.has_value()) return;
  CHECK(result.error().code == core::ErrorCode::UnsupportedFeature);
}

}  // namespace

int main() {
  test_branch_and_bound_accepts_integer_typed_zero_one_columns();
  test_branch_and_bound_folds_absorbing_singleton_cost_correctly();
  test_branch_and_bound_warm_start_survives_two_branching_levels();
  test_branch_and_bound_solves_tiny_knapsack_to_proven_optimal();
  test_branch_and_bound_dispatch_matches_solve_problem_on_continuous_model();
  test_branch_and_bound_rejects_semi_continuous_columns();
  return sovsolve::test::report("branch_and_bound");
}
