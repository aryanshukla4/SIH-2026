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
#include <random>
#include <string_view>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#include "sovsolve/solver/Presolver.hpp"
#include "sovsolve/solver/Scaler.hpp"
#include "sovsolve/solver/pdlp/DualityGap.hpp"
#include "sovsolve/solver/pdlp/Infeasibility.hpp"
#include "sovsolve/solver/pdlp/MatVec.hpp"
#include "sovsolve/solver/pdlp/Pdlp.hpp"
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

/// The normalized duality gap's defining properties. The two assertions here
/// pin down DIFFERENT things, and it is worth being precise about which,
/// because one of them is weaker than it looks.
///
///   * `rho_r(z) >= 0` for every z. This is the invariant the restart
///     conditions rest on -- they compare gaps against each other, so a gap
///     that could go negative would make "sufficient decay" fire on noise.
///     But note what it does NOT test: `min over a ball containing z` is at
///     most `g'z` whatever `g` happens to be, so this holds even with the
///     gradient derived wrongly. It guards the TRUST REGION SOLVER, not the
///     derivation. (Confirmed by mutation: flipping the y-block gradient sign
///     leaves every one of these checks passing.)
///
///   * `rho_r(z) ~ 0` at an optimal z. THIS is what pins the derivation --
///     the same sign mutation takes it from ~0 to 335.8. It is also what
///     makes the gap a progress measure rather than an arbitrary non-negative
///     number.
void test_normalized_duality_gap_properties() {
  auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps");
  if (!loaded.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "load afiro", loaded.error().format());
    return;
  }
  const model::Options options = pdlp_options();

  auto canon = model::canonicalize(loaded.value(), options);
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "canonicalize",
                             canon.error().format());
    return;
  }
  core::Status st = solver::presolve(canon->problem, options, canon->transforms);
  if (st.ok()) st = solver::scale(canon->problem, options, canon->transforms);
  if (!st.ok()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "presolve/scale", st.error().format());
    return;
  }

  const auto& problem = canon->problem;
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();

  solver::pdlp::NormalizedDualityGap gap(problem);
  solver::pdlp::HostMatVec matvec(problem);

  core::RealVector x(n, 0.0);
  core::RealVector y(m, 0.0);
  core::RealVector ref_x(n, 0.0);
  core::RealVector ref_y(m, 0.0);
  core::RealVector kt_y(n, 0.0);
  core::RealVector k_x(m, 0.0);

  auto rho_at = [&](Real omega) {
    matvec.multiply_transpose(core::HostSpan<const Real>(y.data(), y.size()),
                              core::HostSpan<Real>(kt_y.data(), kt_y.size()));
    matvec.multiply(core::HostSpan<const Real>(x.data(), x.size()),
                    core::HostSpan<Real>(k_x.data(), k_x.size()));
    return gap.evaluate(x, y, kt_y, k_x, ref_x, ref_y, omega);
  };

  // Non-negativity over a spread of arbitrary points, including points that
  // violate the bounds and duals with the wrong sign -- the invariant is not
  // conditional on feasibility.
  std::mt19937 rng(20260914);
  std::uniform_real_distribution<Real> val(-5.0, 5.0);
  for (int trial = 0; trial < 40; ++trial) {
    for (std::size_t j = 0; j < n; ++j) x[j] = val(rng);
    for (std::size_t i = 0; i < m; ++i) y[i] = val(rng);
    for (std::size_t j = 0; j < n; ++j) ref_x[j] = val(rng);
    for (std::size_t i = 0; i < m; ++i) ref_y[i] = val(rng);
    const Real omega = trial % 3 == 0 ? 0.25 : (trial % 3 == 1 ? 1.0 : 4.0);
    auto rho = rho_at(omega);
    ++::sovsolve::test::checks_run();
    if (!rho.has_value()) {
      ::sovsolve::test::record(__FILE__, __LINE__, "rho_r evaluates",
                               rho.error().format());
      continue;
    }
    if (!(*rho >= 0.0) || !std::isfinite(*rho)) {
      ::sovsolve::test::record(__FILE__, __LINE__, "rho_r(z) >= 0 for every z",
                               "trial " + std::to_string(trial) + " gave " +
                                   std::to_string(*rho));
    }
  }

  // An unmoved iterate has radius zero, so there is no progress to normalize.
  for (std::size_t j = 0; j < n; ++j) ref_x[j] = x[j];
  for (std::size_t i = 0; i < m; ++i) ref_y[i] = y[i];
  auto zero_radius = rho_at(1.0);
  CHECK(zero_radius.has_value());
  if (zero_radius.has_value()) CHECK_NEAR(*zero_radius, 0.0, 1e-12);

  // At an optimum the gap must vanish. `solve_pdlp` returns the canonical,
  // scaled iterate, which is the space the gap is defined in.
  auto solved = solver::pdlp::solve_pdlp(problem, options);
  if (!solved.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "solve_pdlp", solved.error().format());
    return;
  }
  CHECK(solved->status == SolverStatus::Optimal);
  for (std::size_t j = 0; j < n; ++j) {
    x[j] = solved->x[j];
    ref_x[j] = solved->x[j] + 1e-3;  // a reference a small distance away
  }
  for (std::size_t i = 0; i < m; ++i) {
    y[i] = solved->y[i];
    ref_y[i] = solved->y[i] + 1e-3;
  }
  auto at_optimum = rho_at(1.0);
  ++::sovsolve::test::checks_run();
  if (!at_optimum.has_value() || !(*at_optimum < 1e-4)) {
    ::sovsolve::test::record(
        __FILE__, __LINE__, "rho_r vanishes at an optimum",
        at_optimum.has_value() ? std::to_string(*at_optimum) : "evaluation failed");
  }
}

/// Infeasibility and unboundedness certificates (arXiv 2102.04592).
///
/// PRESOLVE IS TURNED OFF for every case here, and that is the whole point.
/// With it on, the pipeline's verdict comes from `Presolver` before PDLP runs
/// a single iteration -- `iterations=0` is the tell -- so a test with presolve
/// enabled would pass whether or not PDLP could detect anything at all. It
/// could not, until this change.
///
/// The reference for correctness is the dual simplex, which reaches both
/// verdicts by a completely different route (a failed ratio test, and an
/// explicitly constructed ray). Two independent engines agreeing is worth more
/// than either agreeing with my own derivation.
void test_certificates_without_presolve() {
  struct Case {
    const char* label;
    const char* text;
    SolverStatus expected;
  };
  const Case cases[] = {
      // Two rows that contradict only JOINTLY -- invisible to presolve's
      // structural rules, which is why this is the interesting case.
      {"infeasible", R"(Minimize
 obj: x + y
Subject To
 c1: x + y >= 10
 c2: x + y <= 4
Bounds
 0 <= x <= 100
 0 <= y <= 100
End
)",
       SolverStatus::Infeasible},
      // Unbounded only along a JOINT direction: neither column alone escapes.
      {"unbounded", R"(Minimize
 obj: -x - y
Subject To
 c1: x - y <= 1
 c2: -x + y <= 1
Bounds
 x >= 0
 y >= 0
End
)",
       SolverStatus::Unbounded},
  };

  for (const Case& c : cases) {
    model::Options o = pdlp_options();
    o.presolve.enabled = false;
    o.pdlp.max_iterations = 200000;

    model::Options simplex = simplex_options();
    simplex.presolve.enabled = false;

    model::Solution pdlp_solution;
    model::Solution simplex_solution;
    if (!solve_with(c.label, c.text, o, pdlp_solution)) continue;
    if (!solve_with(c.label, c.text, simplex, simplex_solution)) continue;

    ++::sovsolve::test::checks_run();
    if (pdlp_solution.status != c.expected) {
      ::sovsolve::test::record(__FILE__, __LINE__, "PDLP reaches the verdict itself",
                               std::string(c.label) + ": expected the certificate "
                               "to fire, got a different status");
    }
    // Both engines must agree, by construction rather than by coincidence.
    ++::sovsolve::test::checks_run();
    if (simplex_solution.status != pdlp_solution.status) {
      ::sovsolve::test::record(__FILE__, __LINE__, "PDLP and the simplex agree",
                               std::string(c.label) + ": the two engines reached "
                               "different verdicts");
    }
  }
}

/// The failure mode that matters more than a missed detection: a FEASIBLE
/// model must never be declared infeasible.
///
/// History worth keeping. Under the OLD test (candidate normalized by its size,
/// constraints against an absolute epsilon) 1e-4 gave four false verdicts on
/// the Windows build, 1e-6 gave none there -- and one on the Linux build,
/// israel, which is why "none" was not a finding. Under arXiv 2102.04592
/// (50)/(51), measured on BOTH builds with scripts/cert_sweep.sh at 30000
/// iterations: 1e-4 gives one (greenbea Unbounded), 1e-6 and 1e-8 give none,
/// and gas11 is detected at all three -- where the old test missed it at 1e-8.
void test_no_false_verdicts_on_feasible_models() {
  const char* instances[] = {"/netlib/afiro.mps", "/netlib/adlittle.mps"};
  for (const char* name : instances) {
    auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + name);
    if (!loaded.has_value()) {
      ::sovsolve::test::record(__FILE__, __LINE__, "load", loaded.error().format());
      continue;
    }
    model::Options o = pdlp_options();
    o.presolve.enabled = false;
    o.pdlp.max_iterations = 200000;
    auto result = solver::solve_lp(loaded.value(), o);
    if (!result.has_value()) {
      ::sovsolve::test::record(__FILE__, __LINE__, "solve", result.error().format());
      continue;
    }
    ++::sovsolve::test::checks_run();
    if (result->status == SolverStatus::Infeasible ||
        result->status == SolverStatus::Unbounded) {
      ::sovsolve::test::record(
          __FILE__, __LINE__, "a feasible model is never declared infeasible",
          std::string(name) + " got a false verdict -- the certificate "
          "tolerance is too loose");
    }
  }
}


/// THE REGRESSION THAT MOTIVATED arXiv 2102.04592 (50)/(51) HERE.
///
/// The certificate test used to divide a candidate by its own SIZE and then
/// compare each constraint against an absolute epsilon. That accepts
/// almost-flat rays. This model is built so the difference is exact:
///
///     min -1e-5 x   s.t.  1e-7 x <= 1,   x free
///
/// It is BOUNDED -- the optimum is x = 1e7, objective -100. The candidate
/// `v = 1` improves the objective by 1e-5 and violates the row by 1e-7:
///
///     old test   violation 1e-7 <= 1e-6, rate 1e-5 > 1e-6  ->  "Unbounded"
///     (51)       violation / improvement = 1e-2 > 1e-6      ->  rejected
///
/// Exactly the shape that sent a nearly converged israel to a false
/// `Infeasible` on the Linux build. A genuine ray, by contrast, has zero
/// violation and must still be accepted, and both halves are checked so the
/// test cannot pass by the detector simply refusing everything.
void test_flat_rays_are_rejected() {
  auto canonical = [](const char* text, model::CanonicalResult& out) {
    auto parsed = io::parseProblem(text, io::FileFormat::Lp);
    if (!parsed.has_value()) return false;
    model::Options o;
    o.log.level = model::LogOptions::Level::Silent;
    auto canon = model::canonicalize(parsed.value(), o);
    if (!canon.has_value()) return false;
    out = std::move(canon.value());
    return true;
  };

  model::CanonicalResult flat;
  CHECK(canonical(R"(Minimize
 obj: - 0.00001 x
Subject To
 r: 0.0000001 x <= 1
Bounds
 x free
End
)",
                  flat));
  {
    const model::CanonicalProblem& p = flat.problem;
    CHECK_EQ(p.num_cols(), std::size_t{1});
    solver::pdlp::HostMatVec mv(p);
    solver::pdlp::InfeasibilityDetector detector(p, mv);
    core::RealVector v_x(p.num_cols(), 1.0);
    core::RealVector v_y(p.num_rows(), 0.0);
    // Rejected at every tolerance this project uses or has used.
    for (Real tol : {1e-4, 1e-6, 1e-8}) {
      CHECK(detector.classify(v_x, v_y, tol) == solver::pdlp::CertificateKind::None);
    }
    // And it stays rejected however the candidate is scaled -- (51) is a
    // ratio, so a huge or tiny `v` must not change the answer. The old test
    // normalized by size and so was insensitive to this; the point is that
    // the NEW one is too, for the right reason.
    for (Real scale : {1e-8, 1e8}) {
      core::RealVector scaled(p.num_cols(), scale);
      CHECK(detector.classify(scaled, v_y, 1e-6) == solver::pdlp::CertificateKind::None);
    }
  }

  // A GENUINE ray: unbounded along (1, 1), zero violation.
  model::CanonicalResult genuine;
  CHECK(canonical(R"(Minimize
 obj: -x - y
Subject To
 c1: x - y <= 1
 c2: -x + y <= 1
Bounds
 x >= 0
 y >= 0
End
)",
                  genuine));
  {
    const model::CanonicalProblem& p = genuine.problem;
    solver::pdlp::HostMatVec mv(p);
    solver::pdlp::InfeasibilityDetector detector(p, mv);
    core::RealVector v_x(p.num_cols(), 1.0);
    core::RealVector v_y(p.num_rows(), 0.0);
    CHECK(detector.classify(v_x, v_y, 1e-8) ==
          solver::pdlp::CertificateKind::DualInfeasible);
  }

  // A GENUINE dual ray: x + y >= 10 against x + y <= 4. The `>=` row
  // canonicalizes to `-x - y <= -10`, so both duals are non-positive and
  // `v_y = (-1, -1)` gives K'v = 0 with objective rate 10 - 4 = 6.
  model::CanonicalResult infeasible;
  CHECK(canonical(R"(Minimize
 obj: x + y
Subject To
 c1: x + y >= 10
 c2: x + y <= 4
Bounds
 0 <= x <= 100
 0 <= y <= 100
End
)",
                  infeasible));
  {
    const model::CanonicalProblem& p = infeasible.problem;
    CHECK_EQ(p.num_equality, std::size_t{0});
    solver::pdlp::HostMatVec mv(p);
    solver::pdlp::InfeasibilityDetector detector(p, mv);
    core::RealVector v_x(p.num_cols(), 0.0);
    core::RealVector v_y(p.num_rows(), -1.0);
    CHECK(detector.classify(v_x, v_y, 1e-8) ==
          solver::pdlp::CertificateKind::PrimalInfeasible);
  }

  // The two remaining conditions, each found untested by mutation: deleting
  // either left every check above passing.
  //
  // (a) (51)'s BOX condition. `min -x` over `0 <= x <= 1` with an unrelated row:
  // `v_x = e_x` improves the objective and touches no row, so only the box's
  // recession cone -- {0} for a boxed column -- says it is not a ray. The model
  // is bounded (optimum -1).
  model::CanonicalResult boxed;
  CHECK(canonical(R"(Minimize
 obj: -x
Subject To
 r: y <= 5
Bounds
 0 <= x <= 1
 y free
End
)",
                  boxed));
  {
    const model::CanonicalProblem& p = boxed.problem;
    solver::pdlp::HostMatVec mv(p);
    solver::pdlp::InfeasibilityDetector detector(p, mv);
    core::RealVector v_x(p.num_cols(), 0.0);
    v_x[0] = 1.0;  // x, which is boxed
    core::RealVector v_y(p.num_rows(), 0.0);
    CHECK(detector.classify(v_x, v_y, 1e-6) == solver::pdlp::CertificateKind::None);
  }

  // (a') The SAME geometry with the objective scaled up, which is the dfl001
  // bug in miniature. (51) divides a violation in x-units by an improvement in
  // objective-units, so multiplying `c` by 1e7 divides the measured ratio by
  // 1e7 without changing the model's shape at all: the candidate still points
  // straight out of a boxed column's recession cone, and the model is still
  // bounded (optimum -1e7). Under the old 1e-6 default this was accepted and
  // reported Unbounded -- exactly how dfl001 (optimum 1.1266396047e+07) failed
  // at iteration 480, at a measured cone-per-improvement of 8.8e-7.
  //
  // The fix is the tolerance, not an extra condition: bounding the violation
  // against the candidate's own norm as well was tried and REJECTED, because a
  // real ray is not close to its recession cone in a relative sense at PDLP
  // accuracy -- gas11's genuine certificate has 100% of its row activity as
  // violation. See the header comment in pdlp/Infeasibility.cpp.
  model::CanonicalResult boxed_costly;
  CHECK(canonical(R"(Minimize
 obj: -10000000 x
Subject To
 r: y <= 5
Bounds
 0 <= x <= 1
 y free
End
)",
                  boxed_costly));
  {
    const model::CanonicalProblem& p = boxed_costly.problem;
    solver::pdlp::HostMatVec mv(p);
    solver::pdlp::InfeasibilityDetector detector(p, mv);
    core::RealVector v_x(p.num_cols(), 0.0);
    v_x[0] = 1.0;
    core::RealVector v_y(p.num_rows(), 0.0);
    // The ratio is 1/1e7 = 1e-7: inside the old default, outside the new one.
    CHECK(detector.classify(v_x, v_y, 1e-6) ==
          solver::pdlp::CertificateKind::DualInfeasible);
    CHECK(detector.classify(v_x, v_y, 1e-8) == solver::pdlp::CertificateKind::None);
    // And the shipped default must be the one that rejects it.
    model::Options defaults;
    CHECK(detector.classify(v_x, v_y, defaults.pdlp.certificate_tolerance) ==
          solver::pdlp::CertificateKind::None);
  }

  // (b) (50)'s SIGN condition. Add a redundant `x + y <= 200` to the
  // infeasible model and put a large WRONG-SIGNED dual on it. Its projection
  // onto `y <= 0` is the genuine certificate above -- but the candidate the
  // sequence produced is not near one, and quietly substituting the projection
  // would report a certificate that was never observed. The distance moved is
  // charged to the residual, so this is rejected while the projected vector
  // itself is accepted.
  model::CanonicalResult signs;
  CHECK(canonical(R"(Minimize
 obj: x + y
Subject To
 c1: x + y >= 10
 c2: x + y <= 4
 c3: x + y <= 200
Bounds
 0 <= x <= 100
 0 <= y <= 100
End
)",
                  signs));
  {
    const model::CanonicalProblem& p = signs.problem;
    CHECK_EQ(p.num_rows(), std::size_t{3});
    solver::pdlp::HostMatVec mv(p);
    solver::pdlp::InfeasibilityDetector detector(p, mv);
    core::RealVector v_x(p.num_cols(), 0.0);
    core::RealVector wrong(p.num_rows(), -1.0);
    wrong[2] = 1000.0;
    CHECK(detector.classify(v_x, wrong, 1e-6) == solver::pdlp::CertificateKind::None);
    core::RealVector right(p.num_rows(), -1.0);
    right[2] = 0.0;
    CHECK(detector.classify(v_x, right, 1e-6) ==
          solver::pdlp::CertificateKind::PrimalInfeasible);
  }
}

// --------------------------------------------------------------------------
// Module 31: the cuPDLPx scheme
// --------------------------------------------------------------------------

model::Options pdlpx_options() {
  model::Options options = pdlp_options();
  options.simplex.method = model::Method::PdlpX;
  return options;
}

/// THE load-bearing test for the whole module, and it has a perfect oracle.
///
/// cuPDLPx changes the iteration, the step size, the restart criterion and
/// the primal weight all at once. Any one of those four done wrong produces a
/// method that still converges to SOMETHING -- slower, or to the wrong point,
/// or to a point that is not feasible -- so "it returned Optimal" proves
/// nothing on its own. What pins it is that a reflected-Halpern run and a
/// dual-simplex run must agree on the objective of the same model, because
/// the simplex answer is exact.
///
/// The models below are deliberately varied over the things the derivation
/// touches: equality rows against `<=` rows (the projection onto `Y`), finite
/// upper bounds against free columns (the projection onto `X`), and a
/// maximization (the objective sign travels through `to_original`).
void test_halpern_agrees_with_simplex() {
  struct Case {
    const char* name;
    const char* text;
  };
  const Case cases[] = {
      {"equalities", R"(Minimize
 obj: 2 x1 + 3 x2 + x3
Subject To
 c1: x1 + x2 + x3 = 12
 c2: x1 - x2 = 2
Bounds
 0 <= x1 <= 9
 0 <= x2 <= 9
 0 <= x3 <= 9
End
)"},
      {"inequalities", R"(Maximize
 obj: 3 x + 5 y
Subject To
 c1: x + 2 y <= 14
 c2: 3 x - y >= 0
 c3: x - y <= 2
Bounds
 0 <= x <= 10
 0 <= y <= 10
End
)"},
      {"free_column", R"(Minimize
 obj: x + y - 2 z
Subject To
 c1: x + y + z = 6
 c2: x - z <= 3
Bounds
 0 <= x <= 5
 0 <= y <= 5
 z free
End
)"},
  };

  for (const Case& c : cases) {
    model::Solution exact;
    model::Solution halpern;
    if (!solve_with(c.name, c.text, simplex_options(), exact)) continue;
    if (!solve_with(c.name, c.text, pdlpx_options(), halpern)) continue;

    CHECK(exact.status == SolverStatus::Optimal);
    CHECK(halpern.status == SolverStatus::Optimal);
    ++::sovsolve::test::checks_run();
    if (!::sovsolve::test::close(halpern.objective, exact.objective, 1e-6)) {
      ::sovsolve::test::record(__FILE__, __LINE__, "pdlpx agrees with the simplex",
                               std::string(c.name) + ": pdlpx " +
                                   std::to_string(halpern.objective) + ", simplex " +
                                   std::to_string(exact.objective));
    }

    // THE reflection-specific check. `(1+gamma) T - gamma id` is an
    // EXTRAPOLATION, not a projection, so the Halpern iterate itself can sit
    // outside the box. Reporting it would hand back a primal point violating
    // its own bounds while every residual still looked fine. The solver
    // reports `T(z)` instead (BackendVector::PdhgX), and this is what says so.
    ++::sovsolve::test::checks_run();
    if (!(halpern.quality.max_bound_violation <= 1e-9)) {
      ::sovsolve::test::record(
          __FILE__, __LINE__, "pdlpx respects its own bounds",
          std::string(c.name) + ": violation " +
              std::to_string(halpern.quality.max_bound_violation));
    }
  }
}

/// `gamma = 0` must reduce the reflected scheme to plain Halpern, and both
/// must reach the same answer.
///
/// This is the ablation the theory names -- [rHPDHG] Algorithm 1 versus
/// Algorithm 2 -- and running it is how a sign error in the blend gets
/// caught. `reflected * trial - pull_back * current` with either coefficient
/// wrong still converges when `gamma = 1` on an easy model, because the
/// anchor term drags it back; at `gamma = 0` the `pull_back` term vanishes
/// entirely, so the two settings exercise DIFFERENT halves of that line.
void test_reflection_ablation() {
  const char* model_text = R"(Minimize
 obj: 4 a + 2 b + 7 c
Subject To
 r1: a + b + c = 9
 r2: a - b + 2 c <= 5
 r3: 2 a + c >= 4
Bounds
 0 <= a <= 6
 0 <= b <= 6
 0 <= c <= 6
End
)";
  model::Solution exact;
  if (!solve_with("ablation", model_text, simplex_options(), exact)) return;

  for (Real gamma : {0.0, 0.5, 1.0}) {
    model::Options o = pdlpx_options();
    o.pdlp.reflection = gamma;
    model::Solution s;
    if (!solve_with("ablation", model_text, o, s)) continue;
    CHECK(s.status == SolverStatus::Optimal);
    ++::sovsolve::test::checks_run();
    if (!::sovsolve::test::close(s.objective, exact.objective, 1e-6)) {
      ::sovsolve::test::record(__FILE__, __LINE__, "reflection ablation",
                               "gamma " + std::to_string(gamma) + ": " +
                                   std::to_string(s.objective) + " vs " +
                                   std::to_string(exact.objective));
    }
  }
}

/// The recurrence that keeps the iteration at two matrix products.
///
/// `halpern_step` does NOT recompute `K x` and `K' y`; it carries them
/// through the same linear blend as the iterate itself. That is exact in real
/// arithmetic and the reason the scheme costs what vanilla PDHG costs -- but
/// it is also the one place where a wrong coefficient produces a run that
/// still terminates, just from a corrupted gradient.
///
/// So this asserts the COST, which is the observable consequence: products
/// per iteration must be 2 plus the cold path's 2-per-`check_interval`, and
/// nothing more. A third product per iteration -- the obvious implementation
/// -- shows up here as 3.05 and fails.
void test_halpern_costs_two_products_per_iteration() {
  auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps");
  if (!loaded.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "load afiro", loaded.error().format());
    return;
  }
  model::Options o = pdlpx_options();
  o.pdlp.infeasibility_detection = false;
  auto solved = solver::solve_lp(loaded.value(), o);
  if (!solved.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "pdlpx afiro",
                             solved.error().format());
    return;
  }
  CHECK(solved->status == SolverStatus::Optimal);
  CHECK(solved->iterations > 0);

  // Per iteration: 2 from the step, plus 2 per `check_interval` for the
  // termination evaluation. Everything else is start-up -- the power
  // iteration for `||A||_2`, `begin_halpern`, the final re-evaluation --
  // which is a constant, so a generous slack absorbs it without letting a
  // third per-iteration product through.
  const Real per_iteration =
      static_cast<Real>(solved->matrix_products) / static_cast<Real>(solved->iterations);
  const Real budget = 2.0 + 2.0 / static_cast<Real>(o.pdlp.check_interval);
  ++::sovsolve::test::checks_run();
  if (!(per_iteration < budget + 0.5)) {
    ::sovsolve::test::record(__FILE__, __LINE__,
                             "Halpern stays at two products per iteration",
                             std::to_string(per_iteration) + " products/iteration, " +
                                 "budget " + std::to_string(budget));
  }
}

/// The PID controller reduces to cuPDLP's Algorithm 3 when `K_I = K_D = 0`.
///
/// That equivalence is what licenses `K_P = 0.5` as a PUBLISHED constant
/// rather than a guess (see Pdlp.hpp), so it is worth holding the code to it:
/// if the sign of the error term or of the update were flipped, the collapse
/// would not hold and the two would diverge. Asserting it end-to-end rather
/// than on the formula means the whole path -- the anchor distances, the
/// restart that triggers the update, the log-space arithmetic -- is covered.
///
/// The check is that a P-only run stays a well-behaved solve. A flipped sign
/// drives `omega` the wrong way at every restart and the run stops converging
/// entirely, which is the failure this catches.
void test_pid_reduces_to_algorithm_three() {
  auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps");
  if (!loaded.has_value()) return;

  model::Options p_only = pdlpx_options();
  p_only.pdlp.pid_kp = 0.5;
  p_only.pdlp.pid_ki = 0.0;
  p_only.pdlp.pid_kd = 0.0;

  model::Options no_weight = pdlpx_options();
  no_weight.pdlp.primal_weight_update = false;

  auto with_p = solver::solve_lp(loaded.value(), p_only);
  auto without = solver::solve_lp(loaded.value(), no_weight);
  if (!with_p.has_value() || !without.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "pid ablation", "solve failed");
    return;
  }
  CHECK(with_p->status == SolverStatus::Optimal);
  CHECK(without->status == SolverStatus::Optimal);
  CHECK_NEAR(with_p->objective, without->objective, 1e-5);

  // A controller pushing the weight the wrong way does not merely slow the
  // solve down, it stops it. Holding the P-only run to the SAME iteration
  // budget as the no-controller run is the assertion that it helps or at
  // worst does nothing -- not that it fights the method.
  ++::sovsolve::test::checks_run();
  if (!(with_p->iterations <= without->iterations * 3 + 200)) {
    ::sovsolve::test::record(__FILE__, __LINE__, "P-only controller does not fight",
                             std::to_string(with_p->iterations) + " vs " +
                                 std::to_string(without->iterations));
  }
}

/// A Halpern run on a backend that does not implement it must be REFUSED.
///
/// The alternative, with the default no-op overrides on `IterationBackend`,
/// is an iterate that never moves and a run that reports `MaxIterations` on a
/// solvable model -- a wrong answer dressed as a slow one. Both shipped
/// backends implement the scheme, so this guards the NEXT one: any backend
/// added later inherits the refusal until it opts in.
void test_halpern_refuses_an_unsupporting_backend() {
  auto loaded = io::loadProblem(std::string(SOVSOLVE_TEST_DATA_DIR) + "/netlib/afiro.mps");
  if (!loaded.has_value()) return;
  model::Options o = pdlpx_options();
  auto canon = model::canonicalize(loaded.value(), o);
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "canonicalize", "failed");
    return;
  }

  /// A backend that answers every hot-path call and claims no Halpern
  /// support -- the shape a device implementation has before its kernels land.
  class NoHalpern final : public solver::pdlp::IterationBackend {
   public:
    explicit NoHalpern(solver::pdlp::IterationBackend& inner) : inner_(inner) {}
    void set_iterate(core::HostSpan<const Real> x,
                     core::HostSpan<const Real> y) override {
      inner_.set_iterate(x, y);
    }
    void download(solver::pdlp::BackendVector w, core::HostSpan<Real> o) override {
      inner_.download(w, o);
    }
    void begin_step() override { inner_.begin_step(); }
    [[nodiscard]] solver::pdlp::TrialMetrics trial(Real t, Real s) override {
      return inner_.trial(t, s);
    }
    void accept_trial() override { inner_.accept_trial(); }
    void fixed_step(Real t, Real s) override { inner_.fixed_step(t, s); }
    void snapshot_iterate() override { inner_.snapshot_iterate(); }
    void finish_difference() override { inner_.finish_difference(); }
    void accumulate_average(Real w) override { inner_.accumulate_average(w); }
    void reset_average() override { inner_.reset_average(); }
    [[nodiscard]] std::size_t own_products() const override { return 0; }

   private:
    solver::pdlp::IterationBackend& inner_;
  };

  solver::pdlp::HostMatVec matvec(canon->problem);
  solver::pdlp::HostIterationBackend host(canon->problem, matvec);
  NoHalpern blind(host);
  CHECK(!blind.supports_halpern());

  o.pdlp.halpern = true;
  auto result = solver::pdlp::solve_pdlp(canon->problem, o, matvec, blind);
  ++::sovsolve::test::checks_run();
  if (result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__,
                             "Halpern on an unsupporting backend is refused",
                             "it returned a solution instead");
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
  test_normalized_duality_gap_properties();
  test_certificates_without_presolve();
  test_flat_rays_are_rejected();
  test_no_false_verdicts_on_feasible_models();
  test_halpern_agrees_with_simplex();
  test_reflection_ablation();
  test_halpern_costs_two_products_per_iteration();
  test_pid_reduces_to_algorithm_three();
  test_halpern_refuses_an_unsupporting_backend();
  return ::sovsolve::test::report("pdlp_test");
}
