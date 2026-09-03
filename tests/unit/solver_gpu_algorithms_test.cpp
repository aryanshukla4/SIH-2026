// Behavioural tests for the GPU-boundary Residual Calculator (Module 7).
//
// A separate executable, registered only when SOVSOLVE_ENABLE_CUDA is on:
// gpu::compute_residuals lives in sovsolve_solver_gpu, which does not exist
// in a host-only build (src/solver/CMakeLists.txt). The math it tests
// currently executes on the host regardless of that -- see the header
// comment on gpu/ResidualCalculator.hpp for why -- but the function only
// links where the GPU library is actually built, so this suite only runs on
// the "cuda" preset.

#include <cstddef>

#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/solver/Initializer.hpp"
#include "sovsolve/solver/gpu/ResidualCalculator.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)

namespace {

// x1 + x2 = 10, 0 <= x1,x2 <= 8, min x1 + x2. Equality row, both columns
// boxed, no free variables.
model::Problem make_boxed_equality_problem() {
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
  problem.row_names.add("c1");
  problem.col_names.add("x1");
  problem.col_names.add("x2");
  problem.objective_row_name = "obj";
  return problem;
}

// x1 + x2 <= 5, x1 >= 0, x2 free. Inequality row, one lower-only column, one
// free column -- exercises the slack term, the A'y term with a nonzero y,
// and the "no barrier term" path for a free column.
model::Problem make_inequality_free_var_problem() {
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
  problem.c[1] = 2.0;
  problem.A = builder.finish();
  problem.row_lower = core::RealVector(1, -core::INF);
  problem.row_upper = core::RealVector(1, 5.0);
  problem.col_lower = core::RealVector(2);
  problem.col_lower[0] = 0.0;
  problem.col_lower[1] = -core::INF;
  problem.col_upper = core::RealVector(2, core::INF);
  problem.col_type = {core::VarType::Continuous, core::VarType::Continuous};
  problem.row_names.add("c1");
  problem.col_names.add("x1");
  problem.col_names.add("x2");
  problem.objective_row_name = "obj";
  return problem;
}

void test_residuals_on_boxed_equality_problem() {
  const auto problem = make_boxed_equality_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;

  model::Options options;
  auto state = solver::initialize(canon->problem, options);
  CHECK(state.has_value());
  if (!state.has_value()) return;

  solver::Residuals residuals;
  const auto status =
      solver::gpu::compute_residuals(canon->problem, *state, state->mu, residuals);
  CHECK(status.ok());

  // x0 = [4,4] does not satisfy x1+x2=10 -- the initializer is interior, not
  // feasible: rp = (4+4) - 10 = -2.
  CHECK_NEAR(residuals.rp_inf, 2.0, 1e-12);
  // rd_j = 0 + 1 - 0 - 1 + 1 = 1 for both columns.
  CHECK_NEAR(residuals.rd_inf, 1.0, 1e-12);
  // mu was seeded from exactly this point, so complementarity is exact.
  CHECK_NEAR(residuals.complementarity_inf, 0.0, 1e-12);
}

void test_residuals_on_inequality_free_var_problem() {
  const auto problem = make_inequality_free_var_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;

  model::Options options;
  auto state = solver::initialize(canon->problem, options);
  CHECK(state.has_value());
  if (!state.has_value()) return;

  CHECK_EQ(state->num_inequality_rows(), std::size_t{1});

  solver::Residuals residuals;
  const auto status =
      solver::gpu::compute_residuals(canon->problem, *state, state->mu, residuals);
  CHECK(status.ok());

  // x1=1, x2=0, s=1: rp = (1+0) + 1 - 5 = -3.
  CHECK_NEAR(residuals.rp_inf, 3.0, 1e-12);
  // rd_0 = 1 - z1 = 0 ; rd_1 = 2 - 0 (x2 free, no barrier terms) = 2.
  CHECK_NEAR(residuals.rd_inf, 2.0, 1e-12);
  // mu0 = 0.5; rxz_0 = (1-0)*1 - 0.5 = 0.5; rsy_0 = -1*0 - 0.5 = -0.5.
  CHECK_NEAR(residuals.complementarity_inf, 0.5, 1e-12);
}

void test_residuals_rejects_mismatched_state_size() {
  const auto problem = make_boxed_equality_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;

  solver::SolverState wrong_size;
  wrong_size.x = core::RealVector(1, 0.0);  // the problem has 2 columns
  wrong_size.s = core::RealVector(0);
  wrong_size.y = core::RealVector(1, 0.0);
  wrong_size.z = core::RealVector(1, 0.0);
  wrong_size.v = core::RealVector(1, 0.0);

  solver::Residuals residuals;
  const auto status =
      solver::gpu::compute_residuals(canon->problem, wrong_size, 1.0, residuals);
  CHECK(!status.ok());
}

}  // namespace

int main() {
  test_residuals_on_boxed_equality_problem();
  test_residuals_on_inequality_free_var_problem();
  test_residuals_rejects_mismatched_state_size();
  return sovsolve::test::report("solver_gpu_algorithms");
}
