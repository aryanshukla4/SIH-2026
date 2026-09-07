// Behavioural tests for the GPU-boundary Residual Calculator (Module 7),
// KKT Builder (Module 9), Linear Solver (Module 12), Newton Direction
// Recovery (Module 13), Step Length (Module 14), State Update (Module 15),
// Mu Controller (Module 16) and the Predictor-Corrector orchestrator
// (Module 8, gpu/PredictorCorrector.hpp).
//
// A separate executable, registered only when SOVSOLVE_ENABLE_CUDA is on:
// these functions live in sovsolve_solver_gpu, which does not exist in a
// host-only build (src/solver/CMakeLists.txt), and `solve` actually launches
// real cusolverDn calls on the GPU -- these tests only run where a CUDA
// device is present. compute_residuals currently executes on the host
// regardless of that -- see the header comment on gpu/ResidualCalculator.hpp
// for why -- and `solve` is dense (gpu/LinearSolver.hpp explains the
// stopgap), not the intended sparse path.

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "sovsolve/analysis/MatrixAnalysis.hpp"
#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/solver/Diagnostics.hpp"
#include "sovsolve/solver/Initializer.hpp"
#include "sovsolve/solver/Regularization.hpp"
#include "sovsolve/solver/gpu/KktBuilder.hpp"
#include "sovsolve/solver/gpu/LinearSolver.hpp"
#include "sovsolve/solver/gpu/MuController.hpp"
#include "sovsolve/solver/gpu/NewtonRecovery.hpp"
#include "sovsolve/solver/gpu/PredictorCorrector.hpp"
#include "sovsolve/solver/gpu/ResidualCalculator.hpp"
#include "sovsolve/solver/gpu/Solve.hpp"
#include "sovsolve/solver/gpu/StateUpdate.hpp"
#include "sovsolve/solver/gpu/StepLength.hpp"
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

  // x1=1, x2=0; s is now chosen (Initializer.cpp) to exactly satisfy the
  // row given that x: s = b - (x1+x2) = 5 - 1 = 4, so rp = (1+0)+4-5 = 0.
  CHECK_NEAR(residuals.rp_inf, 0.0, 1e-12);
  // y0 = -1 (interiority: -y_I > 0), so A'y = [-1,-1].
  // rd_0 = 1 - (-1) - z1 = 1+1-1 = 1 ; rd_1 = 2 - (-1) - 0 = 3.
  CHECK_NEAR(residuals.rd_inf, 3.0, 1e-12);
  // mu0 = [(1-0)*1 + (-4*-1)] / 2 = (1+4)/2 = 2.5 -- seeded from exactly this
  // point, but x1*z1=1 and -s*y=4 are no longer equal to each other (the old
  // flat s=1 made every pair's own product coincidentally match mu; the
  // slack-matching s here does not), so each pair's OWN residual against the
  // shared average is nonzero: rxz_0 = 1-2.5 = -1.5, rsy_0 = 4-2.5 = 1.5.
  CHECK_NEAR(residuals.complementarity_inf, 1.5, 1e-12);
}

/// Linear scan within one row's slice -- these test matrices are tiny (3x3),
/// so there is no need for anything cleverer.
double matrix_entry(const core::CsrMatrix<>& m, std::size_t row, std::size_t col) {
  for (std::size_t k = m.slice_begin(row); k < m.slice_end(row); ++k) {
    if (static_cast<std::size_t>(m.indices()[k]) == col) return m.values()[k];
  }
  return 0.0;
}

void test_kkt_builder_augmented_system() {
  const auto problem = make_inequality_free_var_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& canonical = canon->problem;

  model::Options options;
  auto state = solver::initialize(canonical, options);
  CHECK(state.has_value());
  if (!state.has_value()) return;

  solver::Residuals residuals;
  CHECK(solver::gpu::compute_residuals(canonical, *state, state->mu, residuals).ok());

  analysis::MatrixAnalysis mat_analysis;  // unused by build_kkt this pass
  solver::KktSystem system;
  const double delta_p = 0.01;
  const double delta_d = 0.02;
  const auto status = solver::gpu::build_kkt(canonical, *state, residuals, mat_analysis,
                                              delta_p, delta_d, system);
  CHECK(status.ok());

  CHECK(system.descriptor.type == solver::ReductionType::QpAugmentedKkt);
  CHECK(!system.descriptor.reason.empty());

  // dim = n + m = 2 + 1 = 3. nnz = 2 (T^-1+delta_p diagonal, Q empty) +
  // 2 (A block) + 2 (A^T block) + 1 (D_s+delta_d diagonal, m_e=0) = 7.
  CHECK_EQ(system.matrix.rows(), std::size_t{3});
  CHECK_EQ(system.matrix.cols(), std::size_t{3});
  CHECK_EQ(system.matrix.nnz(), std::size_t{7});

  const auto& csr = system.matrix.csr;
  // Block (1,1): T^-1 = [1.0, 0.0] (x1 lower-only at its heuristic start,
  // x2 free) -> diag = -(T^-1 + delta_p) = [-1.01, -0.01]; Q empty, so no
  // off-diagonal (0,1)/(1,0) entries.
  CHECK_NEAR(matrix_entry(csr, 0, 0), -1.01, 1e-12);
  CHECK_NEAR(matrix_entry(csr, 1, 1), -0.01, 1e-12);
  CHECK_NEAR(matrix_entry(csr, 0, 1), 0.0, 1e-12);
  // Blocks (2,1) = A and (1,2) = A^T: row0 of A is [1,1].
  CHECK_NEAR(matrix_entry(csr, 2, 0), 1.0, 1e-12);
  CHECK_NEAR(matrix_entry(csr, 2, 1), 1.0, 1e-12);
  CHECK_NEAR(matrix_entry(csr, 0, 2), 1.0, 1e-12);
  CHECK_NEAR(matrix_entry(csr, 1, 2), 1.0, 1e-12);
  // Block (2,2): D_s = s/(-y_I) = 4/1 = 4.0 (m_e=0, so no equality-row
  // entry; s=4 -- see test_residuals_on_inequality_free_var_problem above
  // for the slack-matching derivation) -> 4.0 + delta_d = 4.02.
  CHECK_NEAR(matrix_entry(csr, 2, 2), 4.02, 1e-12);

  // rhs1_j = rd_j + rxz_j/(x_j-l_j) - ruv_j/(u_j-x_j); rxz_0 = -1.5, x2 has
  // no bounds so contributes nothing to rhs1_1.
  // rhs1_0 = rd_0 + rxz_0/(x_0-l_0) = 1 + (-1.5)/(1-0) = -0.5.
  // rhs1_1 = rd_1 = 3 (no bound terms at all for a free column).
  // rhs2 (one inequality row) = -(rp_0 + rsy_0/y_0) = -(0 + 1.5/-1) = 1.5.
  CHECK_EQ(system.rhs.size(), std::size_t{3});
  CHECK_NEAR(system.rhs[0], -0.5, 1e-12);
  CHECK_NEAR(system.rhs[1], 3.0, 1e-12);
  CHECK_NEAR(system.rhs[2], 1.5, 1e-12);
}

void test_linear_solver_diagonal_system() {
  // A pure LinearSolver test, independent of build_kkt/Initializer: a
  // diagonal system whose answer is obvious by inspection, to catch a gross
  // error (transposed indices, wrong cuSOLVER routine) that a residual check
  // alone might not make obvious to a human reading a failure.
  core::SparseBuilder builder(3, 3);
  builder.count(0, 0);
  builder.count(1, 1);
  builder.count(2, 2);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 2.0);
  builder.insert(1, 1, 3.0);
  builder.insert(2, 2, 4.0);

  solver::KktSystem system;
  system.matrix = builder.finish();
  system.rhs = core::RealVector(3);
  system.rhs[0] = 4.0;
  system.rhs[1] = 9.0;
  system.rhs[2] = 16.0;

  auto result = solver::gpu::solve_dense(system, nullptr, 0);
  CHECK(result.has_value());
  if (!result.has_value()) return;

  CHECK_EQ(result->solution.size(), std::size_t{3});
  CHECK_NEAR(result->solution[0], 2.0, 1e-9);
  CHECK_NEAR(result->solution[1], 3.0, 1e-9);
  CHECK_NEAR(result->solution[2], 4.0, 1e-9);
  CHECK_EQ(result->refinement_passes, std::size_t{0});  // deferred this pass
}

void test_linear_solver_on_augmented_kkt_system() {
  const auto problem = make_inequality_free_var_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& canonical = canon->problem;

  model::Options options;
  auto state = solver::initialize(canonical, options);
  CHECK(state.has_value());
  if (!state.has_value()) return;

  solver::Residuals residuals;
  CHECK(solver::gpu::compute_residuals(canonical, *state, state->mu, residuals).ok());

  analysis::MatrixAnalysis mat_analysis;
  solver::KktSystem system;
  CHECK(solver::gpu::build_kkt(canonical, *state, residuals, mat_analysis, 0.01, 0.02, system)
            .ok());

  auto result = solver::gpu::solve_dense(system, nullptr, 0);
  CHECK(result.has_value());
  if (!result.has_value()) return;

  CHECK_EQ(result->solution.size(), system.rhs.size());

  // Verify K*x == rhs to solver precision instead of hand-solving this 3x3
  // system independently: that validates the actual invariant a linear
  // solve must satisfy, rather than comparing against a separately computed
  // answer that could itself carry an arithmetic slip.
  double max_residual = 0.0;
  const auto& csr = system.matrix.csr;
  for (std::size_t i = 0; i < system.matrix.rows(); ++i) {
    double row_value = 0.0;
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      row_value +=
          csr.values()[k] * result->solution[static_cast<std::size_t>(csr.indices()[k])];
    }
    max_residual = std::max(max_residual, std::fabs(row_value - system.rhs[i]));
  }
  CHECK(max_residual < 1e-8);
}

void test_minres_matches_dense_on_augmented_kkt_system() {
  // Direct correctness cross-check for solve_minres (LinearSolver.cu): no
  // independent reference exists for the matrix-free Lanczos/Givens algebra
  // otherwise, so this solves the SAME augmented system both ways and
  // requires them to agree, the same discipline used for
  // test_solve_problem_normal_equations_matches_augmented.
  const auto problem = make_inequality_free_var_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& canonical = canon->problem;

  model::Options options;
  auto state = solver::initialize(canonical, options);
  CHECK(state.has_value());
  if (!state.has_value()) return;

  solver::Residuals residuals;
  CHECK(solver::gpu::compute_residuals(canonical, *state, state->mu, residuals).ok());

  analysis::MatrixAnalysis mat_analysis;
  solver::KktSystem system;
  CHECK(solver::gpu::build_kkt(canonical, *state, residuals, mat_analysis, 0.01, 0.02, system)
            .ok());

  auto dense_result = solver::gpu::solve_dense(system, nullptr, 0);
  CHECK(dense_result.has_value());
  if (!dense_result.has_value()) return;

  auto minres_result = solver::gpu::solve_minres(system, 1e-10, 500);
  CHECK(minres_result.has_value());
  if (!minres_result.has_value()) return;

  CHECK_EQ(minres_result->solution.size(), dense_result->solution.size());
  for (std::size_t i = 0; i < dense_result->solution.size(); ++i) {
    CHECK_NEAR(minres_result->solution[i], dense_result->solution[i], 1e-6);
  }

  // Also check MINRES's own answer satisfies K*x == rhs directly, the same
  // invariant test_linear_solver_on_augmented_kkt_system already applies to
  // the dense solve -- agreement with solve_dense is necessary but this is
  // the more fundamental check.
  double max_residual = 0.0;
  const auto& csr = system.matrix.csr;
  for (std::size_t i = 0; i < system.matrix.rows(); ++i) {
    double row_value = 0.0;
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      row_value +=
          csr.values()[k] * minres_result->solution[static_cast<std::size_t>(csr.indices()[k])];
    }
    max_residual = std::max(max_residual, std::fabs(row_value - system.rhs[i]));
  }
  CHECK(max_residual < 1e-6);
}

void test_cg_matches_dense_on_normal_equations() {
  // Same discipline as test_minres_matches_dense_on_augmented_kkt_system,
  // for solve_spd_cg against solve_spd_dense.
  const auto problem = make_inequality_free_var_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& canonical = canon->problem;

  model::Options options;
  auto state = solver::initialize(canonical, options);
  CHECK(state.has_value());
  if (!state.has_value()) return;

  solver::Residuals residuals;
  CHECK(solver::gpu::compute_residuals(canonical, *state, state->mu, residuals).ok());

  solver::gpu::NormalEquationsSystem ne_system;
  CHECK(solver::gpu::build_normal_equations(canonical, *state, residuals, 0.01, 0.02, ne_system)
            .ok());

  auto dense_result = solver::gpu::solve_spd_dense(ne_system);
  CHECK(dense_result.has_value());
  if (!dense_result.has_value()) return;

  auto cg_result = solver::gpu::solve_spd_cg(ne_system, 1e-10, 500);
  CHECK(cg_result.has_value());
  if (!cg_result.has_value()) return;

  CHECK_EQ(cg_result->solution.size(), dense_result->solution.size());
  for (std::size_t i = 0; i < dense_result->solution.size(); ++i) {
    CHECK_NEAR(cg_result->solution[i], dense_result->solution[i], 1e-6);
  }
}

void test_newton_recovery_satisfies_newton_system() {
  // End-to-end: Initializer -> Residuals -> KKT builder -> Linear solve ->
  // Newton recovery, then verify the recovered directions against the
  // ORIGINAL six-block Newton system (FORMULATION.md 7) -- not by
  // re-deriving the same formulas recover_newton_direction uses, which
  // would just check the implementation against itself.
  const auto problem = make_inequality_free_var_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& canonical = canon->problem;

  model::Options options;
  auto state = solver::initialize(canonical, options);
  CHECK(state.has_value());
  if (!state.has_value()) return;

  solver::Residuals residuals;
  CHECK(solver::gpu::compute_residuals(canonical, *state, state->mu, residuals).ok());

  const double delta_p = 0.01;
  const double delta_d = 0.02;
  analysis::MatrixAnalysis mat_analysis;
  solver::KktSystem system;
  CHECK(solver::gpu::build_kkt(canonical, *state, residuals, mat_analysis, delta_p, delta_d,
                                system)
            .ok());

  auto linear_result = solver::gpu::solve_dense(system, nullptr, 0);
  CHECK(linear_result.has_value());
  if (!linear_result.has_value()) return;

  CHECK(solver::gpu::recover_newton_direction(canonical, system, residuals,
                                               linear_result->solution, *state)
            .ok());

  const std::size_t n = canonical.num_cols();
  const std::size_t m = canonical.num_rows();
  const std::size_t m_e = canonical.num_equality;
  const std::size_t m_i = canonical.num_inequality_rows();

  CHECK_EQ(state->dx.size(), n);
  CHECK_EQ(state->dy.size(), m);
  CHECK_EQ(state->ds.size(), m_i);
  CHECK_EQ(state->dz.size(), n);
  CHECK_EQ(state->dv.size(), n);

  const double tol = 1e-6;

  // Q dx - A'dy - dz + dv == -rd - delta_p*dx, NOT == -rd exactly. The
  // "-delta_p*dx" term is not slack in the test: delta_p regularizes the
  // (1,1) block build_kkt assembled, so (dx,dy) solve the REGULARIZED
  // system, not the true Newton system, and this is precisely how far one
  // is from the other -- FORMULATION.md 10.1's iterative-refinement
  // requirement exists to correct exactly this gap, which gpu::solve does
  // not implement yet (LinearSolver.hpp). Derivation: the solved (1,1) row
  // gives -(Q+T^-1)dx + A'dy = rhs1 + delta_p*dx, and substituting that into
  // the (regularization-independent) identity "Qdx-A'dy-dz+dv == (Q+T^-1)dx
  // - A'dy + [rxz/X_L - ruv/(U-X)]" collapses the rhs1 cross-terms exactly,
  // leaving only the extra -delta_p*dx.
  const auto& A_csc = canonical.A.csc;
  for (std::size_t j = 0; j < n; ++j) {
    double a_t_dy = 0.0;
    for (std::size_t k = A_csc.slice_begin(j); k < A_csc.slice_end(j); ++k) {
      a_t_dy += A_csc.values()[k] * state->dy[static_cast<std::size_t>(A_csc.indices()[k])];
    }
    CHECK_NEAR(-a_t_dy - state->dz[j] + state->dv[j],
               -residuals.rd[j] - delta_p * state->dx[j], tol);
  }

  // A_I dx + ds == -rp_I, over every inequality row.
  const auto& A_csr = canonical.A.csr;
  for (std::size_t k = 0; k < m_i; ++k) {
    const std::size_t i = m_e + k;
    double a_dx = 0.0;
    for (std::size_t idx = A_csr.slice_begin(i); idx < A_csr.slice_end(i); ++idx) {
      a_dx += A_csr.values()[idx] * state->dx[static_cast<std::size_t>(A_csr.indices()[idx])];
    }
    CHECK_NEAR(a_dx + state->ds[k], -residuals.rp[i], tol);
  }

  // Z dx + X_L dz == -rxz (finite lower only); -V dx + (U-X) dv == -ruv
  // (finite upper only).
  for (std::size_t j = 0; j < n; ++j) {
    if (core::is_finite_bound(canonical.col_lower[j])) {
      const double lhs =
          state->z[j] * state->dx[j] + (state->x[j] - canonical.col_lower[j]) * state->dz[j];
      CHECK_NEAR(lhs, -residuals.rxz[j], tol);
    }
    if (core::is_finite_bound(canonical.col_upper[j])) {
      const double lhs = -state->v[j] * state->dx[j] +
                          (canonical.col_upper[j] - state->x[j]) * state->dv[j];
      CHECK_NEAR(lhs, -residuals.ruv[j], tol);
    }
  }

  // -Y_I ds - S dy_I == -rsy - delta_d*y_I*dy_I, by the same regularization
  // argument as the rd check above, applied to the (2,2) block's delta_d
  // instead of the (1,1) block's delta_p. Derivation: the solved (2,*) row
  // gives A dx + D_s dy = rhs2 - delta_d*dy (D_s = s/(-y_I) unregularized);
  // substituting ds = -rp_I - A*dx (the row-3 recovery formula, unaffected
  // by regularization) into "-Y_I ds - S dy_I" and simplifying with
  // rhs2 = -(rp_I + rsy/y_I) leaves exactly "-rsy - delta_d*y_I*dy_I".
  for (std::size_t k = 0; k < m_i; ++k) {
    const std::size_t i = m_e + k;
    const double lhs = -state->y[i] * state->ds[k] - state->s[k] * state->dy[i];
    CHECK_NEAR(lhs, -residuals.rsy[k] - delta_d * state->y[i] * state->dy[i], tol);
  }
}

// Hand-built CanonicalProblem + SolverState, independent of the
// Initializer/Residuals/KktBuilder chain, so the ratio-test formulas can be
// checked against numbers chosen to make every branch (finite-lower,
// finite-upper, free column, slack, and the equality-row-y exclusion) land
// on a specific, obviously-computed ratio.
model::CanonicalProblem make_step_length_test_state(solver::SolverState& state) {
  model::CanonicalProblem problem;
  core::SparseBuilder builder(2, 2);
  builder.count(0, 0);
  builder.count(1, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 1.0);
  builder.insert(1, 1, 1.0);
  problem.A = builder.finish();
  problem.num_equality = 1;  // row0 equality, row1 inequality
  problem.col_lower = core::RealVector(2);
  problem.col_lower[0] = 0.0;
  problem.col_lower[1] = -core::INF;  // col1 free
  problem.col_upper = core::RealVector(2);
  problem.col_upper[0] = 10.0;
  problem.col_upper[1] = core::INF;
  problem.c = core::RealVector(2, 0.0);
  problem.b = core::RealVector(2, 0.0);

  state.x = core::RealVector(2);
  state.x[0] = 5.0;
  state.x[1] = 0.0;
  state.dx = core::RealVector(2);
  state.dx[0] = -10.0;  // ratio to lower bound: (5-0)/10 = 0.5
  state.dx[1] = 5.0;    // free column: no bound, ratio ignored regardless of sign

  state.s = core::RealVector(1);
  state.s[0] = 2.0;
  state.ds = core::RealVector(1);
  state.ds[0] = -8.0;  // ratio: 2/8 = 0.25 -- the tightest primal constraint

  state.z = core::RealVector(2);
  state.z[0] = 3.0;
  state.z[1] = 0.0;
  state.dz = core::RealVector(2);
  state.dz[0] = -1.0;  // ratio: 3/1 = 3.0 -- not binding
  state.dz[1] = 0.0;

  state.v = core::RealVector(2);
  state.v[0] = 1.0;
  state.v[1] = 0.0;
  state.dv = core::RealVector(2);
  state.dv[0] = -0.5;  // ratio: 1/0.5 = 2.0 -- not binding
  state.dv[1] = 0.0;

  state.y = core::RealVector(2);
  state.y[0] = 0.5;   // equality row: unrestricted sign, must be EXCLUDED from the ratio test
  state.y[1] = -4.0;  // inequality row: slack_dual = 4.0
  state.dy = core::RealVector(2);
  state.dy[0] = 10.0;  // if wrongly included, this huge positive dy on an
                       // unrestricted-sign y would produce a spurious tiny
                       // ratio (0.5/10 = 0.05) and the test below would catch it
  state.dy[1] = 6.0;   // ratio: slack_dual(4.0)/6.0 = 0.6667 -- the tightest dual constraint

  return problem;
}

void test_step_length_ratio_test() {
  solver::SolverState state;
  const auto problem = make_step_length_test_state(state);

  double alpha_primal = 0.0;
  double alpha_dual = 0.0;
  const auto status =
      solver::gpu::compute_step_lengths(problem, state, 0.9, alpha_primal, alpha_dual);
  CHECK(status.ok());

  CHECK_NEAR(alpha_primal, 0.9 * 0.25, 1e-12);
  CHECK_NEAR(alpha_dual, 0.9 * (4.0 / 6.0), 1e-12);
}

void test_step_length_rejects_mismatched_state_size() {
  const model::CanonicalProblem problem;  // n=0, m=0
  solver::SolverState state;
  state.x = core::RealVector(1, 0.0);  // mismatched against problem's n=0
  state.dx = core::RealVector(1, 0.0);

  double alpha_primal = 0.0;
  double alpha_dual = 0.0;
  const auto status =
      solver::gpu::compute_step_lengths(problem, state, 0.9, alpha_primal, alpha_dual);
  CHECK(!status.ok());
}

void test_state_update_applies_step() {
  solver::SolverState state;
  const auto problem = make_step_length_test_state(state);  // reuse the same numbers
  state.alpha_primal = 0.25;
  state.alpha_dual = 0.5;

  CHECK(solver::gpu::apply_step(problem, state).ok());

  CHECK_NEAR(state.x[0], 5.0 + 0.25 * -10.0, 1e-12);   // 2.5
  CHECK_NEAR(state.x[1], 0.0 + 0.25 * 5.0, 1e-12);     // 1.25
  // s[0] lands EXACTLY on 0.0 (2.0 + 0.25*-8.0) -- below kConvergedFloor
  // (SolverState.hpp), so apply_step clamps it back up rather than leaving
  // a slack sitting exactly at its own bound (a real, not hypothetical,
  // case: this is precisely what StepLength.cu's ratio test is designed to
  // tolerate rather than let throttle every other coordinate's step).
  CHECK_NEAR(state.s[0], solver::kConvergedFloor, 1e-12);
  CHECK_NEAR(state.z[0], 3.0 + 0.5 * -1.0, 1e-12);     // 2.5
  CHECK_NEAR(state.v[0], 1.0 + 0.5 * -0.5, 1e-12);     // 0.75
  CHECK_NEAR(state.y[0], 0.5 + 0.5 * 10.0, 1e-12);     // 5.5
  CHECK_NEAR(state.y[1], -4.0 + 0.5 * 6.0, 1e-12);     // -1.0
}

void test_state_update_rejects_mismatched_direction_size() {
  const model::CanonicalProblem problem;  // n=0, m=0 -- irrelevant, size check trips first
  solver::SolverState state;
  state.x = core::RealVector(2, 0.0);
  state.dx = core::RealVector(1, 0.0);  // mismatched

  const auto status = solver::gpu::apply_step(problem, state);
  CHECK(!status.ok());
}

void test_predictor_corrector_run_iteration_reduces_primal_residual() {
  // The strongest black-box check available without duplicating
  // run_iteration's internal derivation in the test: a genuine Newton step
  // toward feasibility should shrink the primal residual. Verified on the
  // boxed-equality problem, whose Initializer starting point is interior
  // but deliberately infeasible (solver_types_test.cpp).
  const auto problem = make_boxed_equality_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& canonical = canon->problem;

  model::Options options;
  auto state = solver::initialize(canonical, options);
  CHECK(state.has_value());
  if (!state.has_value()) return;

  solver::Residuals residuals_before;
  CHECK(solver::gpu::compute_residuals(canonical, *state, state->mu, residuals_before).ok());

  solver::RegularizationController regularization(options);
  solver::IterationRecord record;
  const auto status =
      solver::gpu::run_iteration(canonical, options, regularization, *state, record);
  CHECK(status.ok());
  if (!status.ok()) return;

  // predictor_corrector defaults to true: the affine phase must have run.
  CHECK_EQ(state->dx_aff.size(), canonical.num_cols());

  CHECK(record.sigma >= 0.0);
  CHECK(record.sigma <= 1.0);
  CHECK(record.alpha_primal > 0.0);
  CHECK(record.alpha_primal <= options.ipm.eta + 1e-12);
  CHECK(record.alpha_dual > 0.0);
  CHECK(record.alpha_dual <= options.ipm.eta + 1e-12);

  CHECK(solver::gpu::update_mu(canonical, *state).ok());
  solver::Residuals residuals_after;
  CHECK(solver::gpu::compute_residuals(canonical, *state, state->mu, residuals_after).ok());

  CHECK(residuals_after.rp_inf < residuals_before.rp_inf);
}

void test_predictor_corrector_fixed_sigma_path() {
  const auto problem = make_boxed_equality_problem();
  auto canon = model::canonicalize(problem);
  CHECK(canon.has_value());
  if (!canon.has_value()) return;
  const auto& canonical = canon->problem;

  model::Options options;
  options.ipm.predictor_corrector = false;
  options.ipm.sigma = 0.3;

  auto state = solver::initialize(canonical, options);
  CHECK(state.has_value());
  if (!state.has_value()) return;

  solver::RegularizationController regularization(options);
  solver::IterationRecord record;
  const auto status =
      solver::gpu::run_iteration(canonical, options, regularization, *state, record);
  CHECK(status.ok());
  if (!status.ok()) return;

  CHECK_NEAR(record.sigma, 0.3, 1e-12);
  // The affine phase was skipped -- Initializer never sets dx_aff, and
  // run_iteration should not have touched it on this path.
  CHECK_EQ(state->dx_aff.size(), std::size_t{0});
}

void test_solve_problem_end_to_end() {
  // The capstone: canonicalize -> initialize -> iterate to convergence ->
  // reconstruct, all through the public solve_problem() entry point, on
  // min x1+x2 s.t. x1+x2=10, 0<=x1,x2<=8. Every feasible point on that line
  // has the same objective (10), so this checks feasibility and the
  // objective value rather than a specific x1/x2 split.
  const auto problem = make_boxed_equality_problem();

  model::Options options;
  auto result = solver::gpu::solve_problem(problem, options);
  CHECK(result.has_value());
  if (!result.has_value()) return;

  CHECK(result->status == core::SolverStatus::Optimal);
  CHECK(!result->from_best_iterate);  // reached Optimal cleanly, not a stall/limit snapshot
  CHECK_NEAR(result->objective, 10.0, 1e-5);
  CHECK_EQ(result->x.size(), std::size_t{2});
  CHECK_NEAR(result->x[0] + result->x[1], 10.0, 1e-5);
  CHECK(result->x[0] >= -1e-6 && result->x[0] <= 8.0 + 1e-6);
  CHECK(result->x[1] >= -1e-6 && result->x[1] <= 8.0 + 1e-6);
}

void test_solve_problem_normal_equations_matches_augmented() {
  // Direct correctness cross-check for the normal-equations reduction
  // (Options::IpmOptions::use_normal_equations, KktBuilder.cu's
  // build_normal_equations + LinearSolver.cu's solve_spd): the derived
  // Schur-complement RHS/dx formulas have no independent reference to check
  // against otherwise, so this solves the SAME problem via both reductions
  // and requires them to agree, rather than trusting the algebra blind.
  const auto problem = make_boxed_equality_problem();

  model::Options augmented_options;
  auto augmented = solver::gpu::solve_problem(problem, augmented_options);
  CHECK(augmented.has_value());
  if (!augmented.has_value()) return;
  CHECK(augmented->status == core::SolverStatus::Optimal);

  model::Options normal_eq_options;
  normal_eq_options.ipm.use_normal_equations = true;
  auto normal_eq = solver::gpu::solve_problem(problem, normal_eq_options);
  CHECK(normal_eq.has_value());
  if (!normal_eq.has_value()) return;
  CHECK(normal_eq->status == core::SolverStatus::Optimal);

  CHECK_NEAR(normal_eq->objective, augmented->objective, 1e-6);
  CHECK_EQ(normal_eq->x.size(), augmented->x.size());
  for (std::size_t j = 0; j < augmented->x.size(); ++j) {
    CHECK_NEAR(normal_eq->x[j], augmented->x[j], 1e-5);
  }
  CHECK_EQ(normal_eq->y.size(), augmented->y.size());
  for (std::size_t i = 0; i < augmented->y.size(); ++i) {
    CHECK_NEAR(normal_eq->y[i], augmented->y[i], 1e-5);
  }
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
  test_kkt_builder_augmented_system();
  test_linear_solver_diagonal_system();
  test_linear_solver_on_augmented_kkt_system();
  test_minres_matches_dense_on_augmented_kkt_system();
  test_cg_matches_dense_on_normal_equations();
  test_newton_recovery_satisfies_newton_system();
  test_step_length_ratio_test();
  test_step_length_rejects_mismatched_state_size();
  test_state_update_applies_step();
  test_state_update_rejects_mismatched_direction_size();
  test_predictor_corrector_run_iteration_reduces_primal_residual();
  test_predictor_corrector_fixed_sigma_path();
  test_solve_problem_end_to_end();
  test_solve_problem_normal_equations_matches_augmented();
  test_residuals_rejects_mismatched_state_size();
  return sovsolve::test::report("solver_gpu_algorithms");
}
