// Behavioural tests for the new solver-facing types (SolverState,
// ReductionDescriptor, KktSystem, Residuals, Diagnostics) and the host
// modules that are fully implemented this pass (Initializer, Regularization,
// ConvergenceChecker's iteration-limit check, Diagnostics export).
//
// Most GPU-boundary modules have no algorithm bodies yet -- see the STUB
// notes on each header -- so there is nothing to test behaviourally there;
// header_compile_check.cpp covers that they at least compile once
// SOVSOLVE_ENABLE_CUDA is on. The exception is Module 7 (Residuals), which
// IS implemented now (see gpu/ResidualCalculator.hpp for why it currently
// runs on the host) -- its tests live in solver_gpu_algorithms_test.cpp
// instead, since that function only links in a CUDA-enabled build.

#include <cstddef>
#include <string>

#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/solver/ConvergenceChecker.hpp"
#include "sovsolve/solver/Diagnostics.hpp"
#include "sovsolve/solver/Initializer.hpp"
#include "sovsolve/solver/KktSystem.hpp"
#include "sovsolve/solver/Regularization.hpp"
#include "sovsolve/solver/Residuals.hpp"
#include "sovsolve/solver/SolverState.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)

namespace {

void test_solver_state_default_sizes() {
  solver::SolverState state;
  CHECK_EQ(state.num_cols(), std::size_t{0});
  CHECK_EQ(state.num_rows(), std::size_t{0});
  CHECK_EQ(state.num_inequality_rows(), std::size_t{0});
}

void test_slack_dual_helper() {
  // slack_dual(y) = -y -- sign tests should always read through this helper
  // rather than an inline negation (module.txt Module 14).
  CHECK_NEAR(solver::slack_dual(3.0), -3.0, 1e-15);
  CHECK_NEAR(solver::slack_dual(-2.0), 2.0, 1e-15);
}

void test_reduction_descriptor_default() {
  solver::ReductionDescriptor d;
  CHECK(d.type == solver::ReductionType::LpNormalEquationsDy);
  CHECK_EQ(d.theta_floor_activations, std::size_t{0});
  CHECK(d.reason.empty());
}

void test_kkt_system_default_constructs() {
  solver::KktSystem system;
  CHECK_EQ(system.matrix.rows(), std::size_t{0});
  CHECK_EQ(system.rhs.size(), std::size_t{0});
}

void test_residuals_default_constructs() {
  solver::Residuals r;
  CHECK_EQ(r.rp.size(), std::size_t{0});
  CHECK_NEAR(r.rp_inf, 0.0, 1e-15);
}

void test_diagnostics_records_and_exports() {
  solver::Diagnostics diag;
  CHECK(diag.empty());

  solver::IterationRecord r;
  r.iteration = 0;
  r.objective = -464.75;
  r.mu = 1.0;
  diag.record(r);

  r.iteration = 1;
  r.objective = -464.75;
  r.mu = 0.1;
  diag.record(r);

  CHECK_EQ(diag.size(), std::size_t{2});
  CHECK(!diag.empty());

  const auto csv = diag.to_csv();
  CHECK(csv.find("iteration") != std::string::npos);
  CHECK(csv.find("-464.75") != std::string::npos);

  const auto json = diag.to_json();
  CHECK(json.front() == '[');
  CHECK(json.back() == ']');
  CHECK(json.find("\"iteration\":0") != std::string::npos);
  CHECK(json.find("\"iteration\":1") != std::string::npos);
}

void test_regularization_escalate_and_decay() {
  model::Options options;
  options.ipm.primal_regularization_floor = 1e-8;
  options.ipm.dual_regularization_floor = 1e-8;
  options.ipm.regularization_escalation = 100.0;
  options.ipm.regularization_decay = 10.0;
  options.ipm.delta_max = 1e-2;

  solver::RegularizationController reg(options);
  CHECK_NEAR(reg.delta_p(), 1e-8, 1e-9);

  CHECK(reg.escalate());
  CHECK_NEAR(reg.delta_p(), 1e-6, 1e-9);  // 1e-8 * 100
  CHECK_EQ(reg.escalation_events(), std::size_t{1});

  reg.decay();
  CHECK_NEAR(reg.delta_p(), 1e-7, 1e-9);  // 1e-6 / 10

  // Escalate to the ceiling and confirm it refuses to go further.
  for (int i = 0; i < 10; ++i) {
    [[maybe_unused]] const bool escalated = reg.escalate();
  }
  CHECK_NEAR(reg.delta_p(), 1e-2, 1e-9);
  CHECK(!reg.escalate());
}

// x1 + x2 = 10, 0 <= x1,x2 <= 8, min x1 + x2. Built via the real
// canonicalizer rather than a hand-assembled CanonicalProblem, so the
// Initializer test runs against exactly what the ingestion layer produces.
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

void test_initializer_produces_interior_point() {
  const auto problem = make_boxed_equality_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;

  model::Options options;
  auto state = solver::initialize(canon->problem, options);
  CHECK(state.has_value());
  if (!state.has_value()) return;

  CHECK_EQ(state->num_cols(), std::size_t{2});
  CHECK_EQ(state->num_rows(), std::size_t{1});
  CHECK_EQ(state->num_inequality_rows(), std::size_t{0});

  // Both columns are boxed [0,8]: midpoint.
  CHECK_NEAR(state->x[0], 4.0, 1e-12);
  CHECK_NEAR(state->x[1], 4.0, 1e-12);
  CHECK_NEAR(state->z[0], 1.0, 1e-12);
  CHECK_NEAR(state->v[0], 1.0, 1e-12);
  CHECK_NEAR(state->y[0], 0.0, 1e-12);

  // mu0 = [(4-0)*1 + (8-4)*1] * 2 columns / 4 active pairs = 4.0.
  CHECK_NEAR(state->mu, 4.0, 1e-12);
}

void test_initializer_rejects_non_startable_model() {
  // Hand-built, deliberately violating the startability contract (l == u)
  // rather than going through the canonicalizer, which would never produce
  // this -- this is testing Initializer's own re-check, not the
  // canonicalizer's.
  model::CanonicalProblem bad;
  core::SparseBuilder builder(1, 1);
  builder.count(0, 0);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  bad.A = builder.finish();
  bad.c = core::RealVector(1, 1.0);
  bad.b = core::RealVector(1, 5.0);
  bad.col_lower = core::RealVector(1, 3.0);
  bad.col_upper = core::RealVector(1, 3.0);
  bad.num_equality = 1;

  model::Options options;
  auto state = solver::initialize(bad, options);
  CHECK(!state.has_value());
}

void test_convergence_checker_reports_max_iterations() {
  model::Options options;
  options.limits.max_iterations = 5;
  solver::ConvergenceChecker checker(options);

  solver::Residuals r;
  CHECK(checker.check(r, 0.0, 0.0, 3) == core::SolverStatus::NotConverged);
  CHECK(checker.check(r, 0.0, 0.0, 5) == core::SolverStatus::MaxIterations);
}

}  // namespace

int main() {
  test_solver_state_default_sizes();
  test_slack_dual_helper();
  test_reduction_descriptor_default();
  test_kkt_system_default_constructs();
  test_residuals_default_constructs();
  test_diagnostics_records_and_exports();
  test_regularization_escalate_and_decay();
  test_initializer_produces_interior_point();
  test_initializer_rejects_non_startable_model();
  test_convergence_checker_reports_max_iterations();
  return sovsolve::test::report("solver_types");
}
