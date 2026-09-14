// Module 25 stage 4: the homogeneous algorithm end to end.
//
// THE POINT OF THIS MODULE IS THE SECOND AND THIRD TESTS, not the first.
//
// Solving a solvable LP is table stakes -- three other engines here already do
// it, two of them better. The reason the embedding was built is the gap
// README.md has recorded since Version 4: the interior-point family
// STRUCTURALLY CANNOT report `Infeasible` or `Unbounded`. With no feasible
// point there is no interior to follow, the iterates diverge, and the only
// honest outcome is `MaxIterations` -- `gas11` running away to -7.5e10 is
// exactly that.
//
// So the tests that matter are the ones where the answer is a VERDICT rather
// than a vector, and they are written against models small enough that the
// right answer is not in dispute.
//
// Tolerances here are loose compared with the simplex tests, deliberately. This
// is an interior-point method solving with matrix-free CG on `A Theta A'`,
// whose condition number grows like `1/mu^2` (FORMULATION.md section 10.1); it
// returns an interior point near the optimum, not an exact vertex. A test that
// demanded 1e-12 would be testing the linear algebra's luck.

#include <cmath>
#include <cstddef>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/HomogeneousSolve.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using core::SolverStatus;
using solver::HomogeneousVerdict;
using solver::HsdResult;

namespace {

bool solve_text(const char* text, HsdResult& out) {
  auto parsed = io::parseProblem(text, io::FileFormat::Lp);
  if (!parsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "parse", parsed.error().format());
    return false;
  }
  model::Options options;
  options.log.level = model::LogOptions::Level::Silent;
  auto canon = model::canonicalize(parsed.value(), options);
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "canonicalize", canon.error().format());
    return false;
  }
  auto result = solver::solve_hsd(canon->problem, options);
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "solve_hsd", result.error().format());
    return false;
  }
  out = std::move(result.value());
  return true;
}

// -------------------------------------------------------------------------

/// A bounded LP whose optimum is worked out by hand: the vertices of
/// `x + 2y <= 4`, `3x + y <= 6`, `x, y >= 0` are (0,0), (2,0), (0,2) and
/// (8/5, 6/5); minimizing `-x - y` gives -14/5 at the last of them.
void test_optimal() {
  HsdResult r;
  if (!solve_text(R"(Minimize
 obj: -x - y
Subject To
 c1: x + 2 y <= 4
 c2: 3 x + y <= 6
End
)",
                  r)) {
    return;
  }
  CHECK(r.status == SolverStatus::Optimal);
  CHECK(r.verdict == HomogeneousVerdict::Optimal);
  CHECK_NEAR(r.objective, -2.8, 1e-6);
  CHECK_NEAR(r.x[0], 1.6, 1e-5);
  CHECK_NEAR(r.x[1], 1.2, 1e-5);

  // tau is what says "this is a solution, not a ray". It must have stayed well
  // clear of zero, and kappa must have gone to it.
  CHECK(r.tau > 1e-3);
  CHECK(r.kappa < 1e-6 * r.tau);

  // [AA] section 1.5's cost claim, on a real run rather than a stub: one border
  // solve plus a predictor and a corrector, per iteration.
  CHECK(r.iterations > 0);
  CHECK_EQ(r.kkt_solves, 3 * r.iterations);
}

/// THE REASON THIS MODULE EXISTS, part one. `x >= 5` and `x <= 3` cannot both
/// hold. The direct interior-point path cannot say so; this must.
void test_primal_infeasible() {
  HsdResult r;
  if (!solve_text(R"(Minimize
 obj: x
Subject To
 c1: x >= 5
 c2: x <= 3
End
)",
                  r)) {
    return;
  }
  CHECK(r.status == SolverStatus::Infeasible);
  CHECK(r.verdict == HomogeneousVerdict::PrimalInfeasible);

  // tau collapsed and kappa did not -- [AA] Theorem 2. The verdict is only
  // meaningful because of that, so it is asserted rather than left implied.
  CHECK(r.tau < 1e-6 * std::fmax(1.0, r.kappa));
  CHECK(r.kappa > 0.0);
}

/// THE REASON THIS MODULE EXISTS, part two. `min -x` over `x >= 1` with no
/// upper bound runs away, so the problem is dual infeasible -- [AA] Theorem 3.
void test_dual_infeasible() {
  HsdResult r;
  if (!solve_text(R"(Minimize
 obj: -x
Subject To
 c1: x + 0 y >= 1
 c2: y >= 0
End
)",
                  r)) {
    return;
  }
  CHECK(r.status == SolverStatus::Unbounded);
  CHECK(r.verdict == HomogeneousVerdict::DualInfeasible);
  CHECK(r.tau < 1e-6 * std::fmax(1.0, r.kappa));
}

/// Equality rows, finite bounds on both sides, and a free column, all at once
/// -- the shapes `homogeneous_newton_test` checks the algebra of, here exercised
/// through an actual solve where every one of them has to survive the loop.
void test_mixed_shapes() {
  HsdResult r;
  if (!solve_text(R"(Minimize
 obj: 2 x + 3 y + z
Subject To
 e1: x + y + z = 6
 c1: x - y <= 2
Bounds
 0 <= x <= 4
 1 <= y <= 5
 z free
End
)",
                  r)) {
    return;
  }
  CHECK(r.status == SolverStatus::Optimal);

  // Feasibility of the returned point, checked directly rather than trusted.
  CHECK_NEAR(r.x[0] + r.x[1] + r.x[2], 6.0, 1e-5);
  CHECK(r.x[0] - r.x[1] <= 2.0 + 1e-5);
  CHECK(r.x[0] >= -1e-6);
  CHECK(r.x[0] <= 4.0 + 1e-6);
  CHECK(r.x[1] >= 1.0 - 1e-6);
  CHECK(r.x[1] <= 5.0 + 1e-6);

  // z is free and carries the negative cost, so the objective is driven by
  // pushing z down; x and y go to their cheapest feasible corner.
  CHECK(r.progress.rho_p < 1e-6);
  CHECK(r.progress.rho_d < 1e-6);
}

/// A run that is cut off must say so rather than reporting the iterate it
/// happened to be holding as optimal.
void test_iteration_limit_is_reported() {
  auto parsed = io::parseProblem(R"(Minimize
 obj: -x - y
Subject To
 c1: x + 2 y <= 4
 c2: 3 x + y <= 6
End
)",
                                 io::FileFormat::Lp);
  if (!parsed.has_value()) return;
  model::Options options;
  options.log.level = model::LogOptions::Level::Silent;
  options.hsd.max_iterations = 2;
  auto canon = model::canonicalize(parsed.value(), options);
  if (!canon.has_value()) return;
  auto result = solver::solve_hsd(canon->problem, options);
  CHECK(result.has_value());
  if (!result.has_value()) return;
  CHECK(result->status != SolverStatus::Optimal);
  CHECK_EQ(result->iterations, std::size_t{2});
}

/// The canonical-space Solution must carry the same numbers, through the same
/// shared quality routine the other three engines use.
void test_canonical_solution() {
  auto parsed = io::parseProblem(R"(Minimize
 obj: -x - y
Subject To
 c1: x + 2 y <= 4
 c2: 3 x + y <= 6
End
)",
                                 io::FileFormat::Lp);
  if (!parsed.has_value()) return;
  model::Options options;
  options.log.level = model::LogOptions::Level::Silent;
  auto canon = model::canonicalize(parsed.value(), options);
  if (!canon.has_value()) return;
  auto result = solver::solve_hsd(canon->problem, options);
  if (!result.has_value()) return;

  const model::Solution s = solver::to_canonical_solution(canon->problem, *result);
  CHECK(s.status == SolverStatus::Optimal);
  CHECK_NEAR(s.objective, -2.8, 1e-6);
  CHECK(s.quality.primal_infeasibility < 1e-6);
  CHECK(s.quality.dual_infeasibility < 1e-6);
  CHECK(s.quality.max_bound_violation < 1e-6);
  CHECK_EQ(s.x.size(), canon->problem.num_cols());
  CHECK_EQ(s.y.size(), canon->problem.num_rows());
}

}  // namespace

int main() {
  test_optimal();
  test_primal_infeasible();
  test_dual_infeasible();
  test_mixed_shapes();
  test_iteration_limit_is_reported();
  test_canonical_solution();
  return ::sovsolve::test::report("homogeneous_solve_test");
}
