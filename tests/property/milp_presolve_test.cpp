// Tests for Module 22's GCD/knapsack row tightening (MilpPresolve.hpp).
// Host-only -- MilpPresolve.cpp lives in sovsolve_solver, no CUDA needed.

#include <cmath>

#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/solver/MilpPresolve.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)

namespace {

// One row, coefficients [2, 4, 6] over three Integer columns -- gcd = 2.
model::Problem make_gcd_row_problem(core::Real row_upper, core::Real row_lower) {
  core::SparseBuilder builder(1, 3);
  builder.count(0, 0);
  builder.count(0, 1);
  builder.count(0, 2);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 2.0);
  builder.insert(0, 1, 4.0);
  builder.insert(0, 2, 6.0);

  model::Problem problem;
  problem.sense = core::ObjSense::Minimize;
  problem.c = core::RealVector(3, 0.0);
  problem.A = builder.finish();
  problem.row_lower = core::RealVector(1, row_lower);
  problem.row_upper = core::RealVector(1, row_upper);
  problem.col_lower = core::RealVector(3, 0.0);
  problem.col_upper = core::RealVector(3, 10.0);
  problem.col_type = {core::VarType::Integer, core::VarType::Integer, core::VarType::Integer};
  return problem;
}

void test_tighten_rounds_upper_bound_down_to_nearest_multiple_of_gcd() {
  auto problem = make_gcd_row_problem(/*row_upper=*/11.0, /*row_lower=*/-core::INF);
  const auto status = solver::tighten_integer_rows(problem);
  CHECK(status.ok());
  // gcd(2,4,6)=2; 11 is not a multiple of 2, so it rounds down to 10 -- no
  // integer combination of these columns can ever reach an odd total, so
  // nothing is lost.
  CHECK_NEAR(problem.row_upper[0], 10.0, 1e-9);
}

void test_tighten_rounds_lower_bound_up_to_nearest_multiple_of_gcd() {
  auto problem = make_gcd_row_problem(/*row_upper=*/core::INF, /*row_lower=*/7.0);
  const auto status = solver::tighten_integer_rows(problem);
  CHECK(status.ok());
  CHECK_NEAR(problem.row_lower[0], 8.0, 1e-9);
}

void test_tighten_detects_infeasible_equality_row() {
  // Equality row with both bounds at 7 -- not a multiple of gcd=2, so no
  // integer combination of these three columns can ever equal it.
  auto problem = make_gcd_row_problem(/*row_upper=*/7.0, /*row_lower=*/7.0);
  const auto status = solver::tighten_integer_rows(problem);
  CHECK(!status.ok());
  if (status.ok()) return;
  CHECK(status.error().code == core::ErrorCode::PrimalInfeasible);
}

void test_tighten_leaves_feasible_equality_row_untouched() {
  // 12 IS a multiple of gcd=2 -- both bounds already sit exactly on a valid
  // multiple, so tightening should be a no-op (not accidentally shift a
  // genuinely reachable value).
  auto problem = make_gcd_row_problem(/*row_upper=*/12.0, /*row_lower=*/12.0);
  const auto status = solver::tighten_integer_rows(problem);
  CHECK(status.ok());
  CHECK_NEAR(problem.row_upper[0], 12.0, 1e-9);
  CHECK_NEAR(problem.row_lower[0], 12.0, 1e-9);
}

void test_tighten_skips_row_with_a_continuous_column() {
  auto problem = make_gcd_row_problem(/*row_upper=*/11.0, /*row_lower=*/-core::INF);
  problem.col_type[1] = core::VarType::Continuous;  // breaks the "always a multiple of g" property
  const auto status = solver::tighten_integer_rows(problem);
  CHECK(status.ok());
  CHECK_NEAR(problem.row_upper[0], 11.0, 1e-9);  // untouched
}

// One equality row: coeff*s + 2*x1 + 4*x2 = 20. `s` is continuous,
// non-negative, unbounded above, and a singleton (only appears here) --
// exactly markshare_4_0.mps's row shape (one absorbing deviation column
// plus otherwise-discrete columns).
model::Problem make_absorbing_row_problem(core::Real s_coeff) {
  core::SparseBuilder builder(1, 3);
  builder.count(0, 0);  // s
  builder.count(0, 1);  // x1
  builder.count(0, 2);  // x2
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, s_coeff);
  builder.insert(0, 1, 2.0);
  builder.insert(0, 2, 4.0);

  model::Problem problem;
  problem.sense = core::ObjSense::Minimize;
  problem.c = core::RealVector(3, 0.0);
  problem.c[0] = 1.0;  // s costs something, like markshare's deviation columns
  problem.A = builder.finish();
  problem.row_lower = core::RealVector(1, 20.0);
  problem.row_upper = core::RealVector(1, 20.0);
  problem.col_lower = core::RealVector(3);
  problem.col_upper = core::RealVector(3);
  problem.col_lower[0] = 0.0;
  problem.col_upper[0] = core::INF;  // s: lower-only
  problem.col_lower[1] = 0.0;
  problem.col_upper[1] = 10.0;
  problem.col_lower[2] = 0.0;
  problem.col_upper[2] = 10.0;
  problem.col_type = {core::VarType::Continuous, core::VarType::Integer, core::VarType::Integer};
  return problem;
}

void test_eliminate_absorbing_singleton_positive_coefficient_derives_upper_bound() {
  // 3*s + 2*x1 + 4*x2 = 20, s >= 0 -- s can grow without limit, so
  // 2*x1+4*x2 = 20 - 3*s has no lower limit but can never exceed 20 (s's
  // own lower bound, 0, is the tightest the subtraction ever gets).
  auto problem = make_absorbing_row_problem(3.0);
  std::vector<solver::AbsorbingColumnElimination> elims;
  solver::eliminate_equality_row_absorbing_singletons(problem, elims);

  CHECK_NEAR(problem.row_upper[0], 20.0, 1e-9);
  CHECK(!core::is_finite_bound(problem.row_lower[0]));
  // s's column is now empty -- its only entry was removed.
  CHECK_EQ(problem.A.csc.slice_nnz(0), std::size_t{0});
  CHECK_EQ(problem.A.csc.slice_nnz(1), std::size_t{1});
  CHECK_EQ(problem.A.csc.slice_nnz(2), std::size_t{1});

  // Cost fold: c_j*x_j = c_j*b_i/a_ij - (c_j/a_ij)*R. Here c_j=1, a_ij=3,
  // b_i=20 -> fold_factor=1/3: c[0] must be zeroed, obj_constant gains
  // 20/3, and x1/x2's own (originally zero) costs pick up -(1/3)*their
  // coefficient -- this is the exact fold whose ABSENCE let
  // markshare_4_0.mps silently report an unreachable objective (see
  // MilpPresolve.hpp's doc comment).
  CHECK_NEAR(problem.c[0], 0.0, 1e-9);
  CHECK_NEAR(problem.obj_constant, 20.0 / 3.0, 1e-9);
  CHECK_NEAR(problem.c[1], -2.0 / 3.0, 1e-9);
  CHECK_NEAR(problem.c[2], -4.0 / 3.0, 1e-9);

  CHECK_EQ(elims.size(), std::size_t{1});
  if (elims.size() == 1) {
    CHECK_EQ(elims[0].row, std::size_t{0});
    CHECK_EQ(elims[0].col, std::size_t{0});
    CHECK_NEAR(elims[0].a_ij, 3.0, 1e-9);
    CHECK_NEAR(elims[0].b_i, 20.0, 1e-9);
  }
}

void test_eliminate_absorbing_singleton_negative_coefficient_derives_lower_bound() {
  // -3*s + 2*x1 + 4*x2 = 20, s >= 0 -- now 2*x1+4*x2 = 20 + 3*s can only
  // grow from 20 upward as s increases, so the derived bound flips to a
  // LOWER bound instead of an upper one.
  auto problem = make_absorbing_row_problem(-3.0);
  std::vector<solver::AbsorbingColumnElimination> elims;
  solver::eliminate_equality_row_absorbing_singletons(problem, elims);

  CHECK_NEAR(problem.row_lower[0], 20.0, 1e-9);
  CHECK(!core::is_finite_bound(problem.row_upper[0]));
  CHECK_EQ(problem.A.csc.slice_nnz(0), std::size_t{0});

  // fold_factor = c_j/a_ij = 1/-3.
  CHECK_NEAR(problem.c[0], 0.0, 1e-9);
  CHECK_NEAR(problem.obj_constant, -20.0 / 3.0, 1e-9);
  CHECK_NEAR(problem.c[1], 2.0 / 3.0, 1e-9);
  CHECK_NEAR(problem.c[2], 4.0 / 3.0, 1e-9);
}

void test_eliminate_absorbing_singleton_skips_boxed_column() {
  // Same shape, but s is now BOXED (both bounds finite) -- outside this
  // rule's scope (would need the two-sided ranged-row machinery this pass
  // does not build). Row must be left completely untouched.
  auto problem = make_absorbing_row_problem(3.0);
  problem.col_upper[0] = 50.0;  // s is now [0, 50], not lower-only
  std::vector<solver::AbsorbingColumnElimination> elims;
  solver::eliminate_equality_row_absorbing_singletons(problem, elims);

  CHECK_NEAR(problem.row_lower[0], 20.0, 1e-9);
  CHECK_NEAR(problem.row_upper[0], 20.0, 1e-9);
  CHECK_EQ(problem.A.csc.slice_nnz(0), std::size_t{1});  // s's entry survives
}

void test_tighten_skips_row_with_coprime_coefficients() {
  // 6 and 5 share no common factor -- the tiny-knapsack fixture used
  // elsewhere in this session's MILP tests -- so nothing should change.
  core::SparseBuilder builder(1, 2);
  builder.count(0, 0);
  builder.count(0, 1);
  CHECK(builder.allocate().ok());
  builder.insert(0, 0, 6.0);
  builder.insert(0, 1, 5.0);

  model::Problem problem;
  problem.sense = core::ObjSense::Minimize;
  problem.c = core::RealVector(2, 0.0);
  problem.A = builder.finish();
  problem.row_lower = core::RealVector(1, -core::INF);
  problem.row_upper = core::RealVector(1, 10.0);
  problem.col_lower = core::RealVector(2, 0.0);
  problem.col_upper = core::RealVector(2, 1.0);
  problem.col_type = {core::VarType::Binary, core::VarType::Binary};

  const auto status = solver::tighten_integer_rows(problem);
  CHECK(status.ok());
  CHECK_NEAR(problem.row_upper[0], 10.0, 1e-9);
}

}  // namespace

int main() {
  test_eliminate_absorbing_singleton_positive_coefficient_derives_upper_bound();
  test_eliminate_absorbing_singleton_negative_coefficient_derives_lower_bound();
  test_eliminate_absorbing_singleton_skips_boxed_column();
  test_tighten_rounds_upper_bound_down_to_nearest_multiple_of_gcd();
  test_tighten_rounds_lower_bound_up_to_nearest_multiple_of_gcd();
  test_tighten_detects_infeasible_equality_row();
  test_tighten_leaves_feasible_equality_row_untouched();
  test_tighten_skips_row_with_a_continuous_column();
  test_tighten_skips_row_with_coprime_coefficients();
  return sovsolve::test::report("milp_presolve");
}
