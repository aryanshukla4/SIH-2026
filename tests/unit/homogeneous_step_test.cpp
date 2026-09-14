// Module 25 stage 2: the homogeneous algorithm's iteration control.
//
// THE LOAD-BEARING TEST IS THE STANDARD-FORM SPECIALIZATION.
//
// Every formula in HomogeneousStep.hpp is transcribed from Andersen & Andersen
// (2000), which states the homogeneous algorithm for STANDARD FORM only --
// `Ax = b, x >= 0`. Four of them had to be generalized to our canonical form
// `l <= x <= u` with equality and `<=` rows. A generalization is only
// trustworthy if it collapses back onto the published formula when the extra
// structure is removed, so the first test below builds an instance that IS in
// standard form after canonicalization and checks each generalized formula
// against [AA]'s own equation, written out here by hand.
//
// That check is worth more than a convergence test would be. A sign or a
// missing term in, say, the ratio test does not crash -- it produces a step
// that is merely somewhat wrong, on an engine that already only reaches
// Optimal on 6-7 of 19 instances, where one more slow instance looks like
// nothing at all.
//
// The second thing these tests do is prove the ONE coupling that standard form
// cannot see: `tau` scales the bounds, so `x - l tau` moves when `tau` moves
// even if `x` does not. With `l = 0` that term is invisible. The test named
// for it constructs a direction where it is the only thing that binds, so
// deleting it changes a finite step into an unbounded one.

#include <cmath>
#include <cstddef>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/Homogeneous.hpp"
#include "sovsolve/solver/HomogeneousStep.hpp"
#include "sovsolve/solver/SolverState.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using solver::HomogeneousParameters;
using solver::HomogeneousProgress;
using solver::HomogeneousReference;
using solver::HomogeneousResiduals;
using solver::HomogeneousTermination;
using solver::SolverState;

namespace {

bool parse_into(const char* text, model::CanonicalResult& out) {
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
  out = std::move(canon.value());
  return true;
}

/// Standard form after canonicalization: equality rows only, every column
/// `x_j >= 0` with no upper bound. This is exactly [AA] section 1.2's `(P)`.
bool build_standard_form(model::CanonicalResult& out) {
  return parse_into(R"(Minimize
 obj: 2 x + 3 y + z
Subject To
 e1: x + y = 4
 e2: y + z = 3
End
)",
                    out);
}

/// Equality and inequality rows; columns bounded below only, above only, on
/// both sides, and free. Every finite-bound branch is reachable.
bool build_general(model::CanonicalResult& out) {
  return parse_into(R"(Minimize
 obj: 2 x + 3 y - z + 4 w
Subject To
 e1: x + y + z = 6
 c1: x - y <= 3
 c2: y + w <= 8
Bounds
 0 <= x <= 5
 1 <= y <= 9
 z free
 w >= 2
End
)",
                    out);
}

/// A strictly feasible-looking iterate: every nonnegative quantity is > 0.
/// Not a solution; these functions read the iterate and a direction, nothing
/// else.
SolverState make_state(const model::CanonicalProblem& p, Real tau, Real kappa) {
  SolverState s;
  const std::size_t n = p.num_cols();
  const std::size_t m = p.num_rows();
  const std::size_t m_i = p.num_inequality_rows();
  s.x = core::RealVector(n);
  s.z = core::RealVector(n);
  s.v = core::RealVector(n);
  s.y = core::RealVector(m);
  s.s = core::RealVector(m_i);
  for (std::size_t j = 0; j < n; ++j) {
    const Real l = p.col_lower[j];
    const Real u = p.col_upper[j];
    const bool has_l = core::is_finite_bound(l);
    const bool has_u = core::is_finite_bound(u);
    if (has_l && has_u) {
      s.x[j] = 0.5 * (l + u) * tau;
    } else if (has_l) {
      s.x[j] = (l + 1.0 + 0.25 * static_cast<Real>(j % 3)) * tau;
    } else if (has_u) {
      s.x[j] = (u - 1.0 - 0.25 * static_cast<Real>(j % 3)) * tau;
    } else {
      s.x[j] = 0.5 + 0.25 * static_cast<Real>(j % 5);
    }
    s.z[j] = 0.3 + 0.1 * static_cast<Real>(j % 3);
    s.v[j] = 0.2 + 0.15 * static_cast<Real>(j % 4);
  }
  for (std::size_t i = 0; i < m; ++i) s.y[i] = -0.4 - 0.1 * static_cast<Real>(i % 3);
  for (std::size_t k = 0; k < m_i; ++k) s.s[k] = 0.7 + 0.2 * static_cast<Real>(k % 2);
  s.tau = tau;
  s.kappa = kappa;
  return s;
}

/// Zero every direction, so a test can switch on exactly one component and
/// know that whatever binds, binds because of that component.
void clear_direction(SolverState& s) {
  const std::size_t n = s.x.size();
  const std::size_t m = s.y.size();
  const std::size_t m_i = s.s.size();
  s.dx = core::RealVector(n, 0.0);
  s.dz = core::RealVector(n, 0.0);
  s.dv = core::RealVector(n, 0.0);
  s.dy = core::RealVector(m, 0.0);
  s.ds = core::RealVector(m_i, 0.0);
  s.dtau = 0.0;
  s.dkappa = 0.0;
}

// --------------------------------------------------------------------------

/// Guards the fixtures. If `build_standard_form` ever stops producing standard
/// form -- a presolve change adding a slack, say -- the specialization test
/// below would silently start comparing two general-form formulas against each
/// other and pass for the wrong reason.
void test_the_fixtures_are_what_they_claim() {
  model::CanonicalResult canon;
  if (!build_standard_form(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  CHECK(p.num_rows() > 0);
  CHECK_EQ(p.num_inequality_rows(), std::size_t{0});
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    CHECK(core::is_finite_bound(p.col_lower[j]));
    CHECK_NEAR(p.col_lower[j], 0.0, 0.0);
    CHECK(!core::is_finite_bound(p.col_upper[j]));
  }

  model::CanonicalResult general;
  if (!build_general(general)) return;
  const model::CanonicalProblem& g = general.problem;
  CHECK(g.num_inequality_rows() > 0);
  std::size_t both = 0, lower_only = 0, upper_only = 0, free_cols = 0, positive_l = 0;
  for (std::size_t j = 0; j < g.num_cols(); ++j) {
    const bool has_l = core::is_finite_bound(g.col_lower[j]);
    const bool has_u = core::is_finite_bound(g.col_upper[j]);
    if (has_l && has_u) ++both;
    if (has_l && !has_u) ++lower_only;
    if (!has_l && has_u) ++upper_only;
    if (!has_l && !has_u) ++free_cols;
    if (has_l && g.col_lower[j] > 0.0) ++positive_l;
  }
  CHECK(both > 0);
  CHECK(lower_only > 0);
  CHECK(free_cols > 0);
  // The `l tau` coupling test below needs a column whose lower bound is
  // NONZERO. At `l = 0` the term it is proving vanishes.
  CHECK(positive_l > 0);
  (void)upper_only;
}

/// [AA] (1.22), (1.21), (1.24) and `n + 1`, each written out by hand for
/// standard form and compared against the generalized implementation.
void test_standard_form_recovers_the_published_formulas() {
  model::CanonicalResult canon;
  if (!build_standard_form(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  const std::size_t n = p.num_cols();

  // --- [AA] (1.22): (x, tau, y, s, kappa) := (e, 1, 0, e, 1). ---------------
  SolverState start;
  CHECK(solver::homogeneous_starting_point(p, start).ok());
  for (std::size_t j = 0; j < n; ++j) {
    CHECK_NEAR(start.x[j], 1.0, 0.0);  // e
    CHECK_NEAR(start.z[j], 1.0, 0.0);  // the `s := e` of (1.22)
    CHECK_NEAR(start.v[j], 0.0, 0.0);  // no upper bound, so no pair at all
  }
  for (std::size_t i = 0; i < p.num_rows(); ++i) CHECK_NEAR(start.y[i], 0.0, 0.0);
  CHECK_NEAR(start.tau, 1.0, 0.0);
  CHECK_NEAR(start.kappa, 1.0, 0.0);
  // Every pair product is 1, and there are n + 1 of them, so mu is exactly 1.
  CHECK_NEAR(start.mu, 1.0, 1e-15);

  // --- [AA]'s `n + 1`. -----------------------------------------------------
  CHECK_EQ(solver::homogeneous_pair_count(p), n + 1);

  // --- [AA] (1.21), the standard-form ratio test. ---------------------------
  SolverState s = make_state(p, 1.3, 0.7);
  clear_direction(s);
  for (std::size_t j = 0; j < n; ++j) {
    s.dx[j] = (j % 2 == 0) ? -0.4 : 0.6;
    s.dz[j] = (j % 3 == 0) ? -0.25 : 0.1;
  }
  s.dtau = -0.2;
  s.dkappa = 0.5;

  // argmax over alpha >= 0 of { (x; tau; s; kappa) + alpha d >= 0 }, with no
  // bound terms because l = 0 and u = inf.
  Real oracle = solver::kUnboundedStep;
  auto bind = [&oracle](Real value, Real step) {
    if (step < 0.0) oracle = std::fmin(oracle, -value / step);
  };
  for (std::size_t j = 0; j < n; ++j) {
    bind(s.x[j], s.dx[j]);
    bind(s.z[j], s.dz[j]);
  }
  bind(s.tau, s.dtau);
  bind(s.kappa, s.dkappa);
  CHECK_NEAR(solver::homogeneous_alpha_max(p, s, /*affine=*/false), oracle, 1e-14);

  // --- [AA] (1.24): |c'x - b'y| / (tau + |b'y|). ----------------------------
  HomogeneousResiduals residuals;
  CHECK(solver::compute_homogeneous_residuals(p, s, 0.1, residuals).ok());
  HomogeneousReference reference;
  reference.rp_norm_0 = 10.0;
  reference.rd_norm_0 = 10.0;
  reference.rg_0 = 10.0;
  reference.mu_0 = 1.0;
  HomogeneousProgress progress;
  CHECK(solver::homogeneous_progress(p, s, residuals, reference, progress).ok());

  Real cx = 0.0;
  for (std::size_t j = 0; j < n; ++j) cx += p.c[j] * s.x[j];
  Real by = 0.0;
  for (std::size_t i = 0; i < p.num_rows(); ++i) by += p.b[i] * s.y[i];
  CHECK_NEAR(progress.rho_a, std::fabs(cx - by) / (s.tau + std::fabs(by)), 1e-14);
}

/// [AA] (1.12), by its closed form, including where the `beta_1` cap takes over.
void test_gamma_matches_equation_1_12() {
  HomogeneousParameters params;  // beta1 = 0.1

  auto expected = [&](Real a) {
    const Real slack = 1.0 - a;
    return slack * slack * std::fmin(slack, params.beta1);
  };

  for (Real a : {0.0, 0.05, 0.25, 0.5, 0.85, 0.9, 0.95, 0.99, 1.0}) {
    CHECK_NEAR(solver::homogeneous_gamma(a, params), expected(a), 1e-15);
  }

  // Below alpha = 0.9 the slack exceeds beta1, so the cap binds and gamma is
  // beta1 * (1-a)^2 -- quadratic, not cubic. Above it, the cubic takes over.
  CHECK_NEAR(solver::homogeneous_gamma(0.5, params), 0.1 * 0.25, 1e-15);
  CHECK_NEAR(solver::homogeneous_gamma(0.95, params), 0.05 * 0.05 * 0.05, 1e-15);

  // A full affine step means no centering is wanted at all, and no step means
  // the cap is in force at its largest.
  CHECK_NEAR(solver::homogeneous_gamma(1.0, params), 0.0, 0.0);
  CHECK_NEAR(solver::homogeneous_gamma(0.0, params), 0.1, 1e-15);

  // Monotone decreasing in alpha_max: a longer affine step must never ask for
  // MORE centering.
  Real previous = solver::homogeneous_gamma(0.0, params);
  for (int k = 1; k <= 100; ++k) {
    const Real g = solver::homogeneous_gamma(0.01 * static_cast<Real>(k), params);
    CHECK(g <= previous + 1e-15);
    previous = g;
  }

  // Out-of-range input is clamped, not propagated: [AA] takes the argmax over
  // [0, 1], and an unbounded ratio test hands back kUnboundedStep.
  CHECK_NEAR(solver::homogeneous_gamma(solver::kUnboundedStep, params), 0.0, 0.0);
  CHECK_NEAR(solver::homogeneous_gamma(-3.0, params), 0.1, 1e-15);
}

/// THE COUPLING STANDARD FORM CANNOT SEE.
///
/// `tau` scales the bounds, so the quantity `x - l tau` has direction
/// `dx - l dtau`. Here `dx` is identically zero and only `dtau` is nonzero, so
/// the step is limited BY THAT TERM ALONE. Drop it and the ratio test reports
/// no binding constraint at all.
void test_tau_moves_the_bounds() {
  model::CanonicalResult canon;
  if (!build_general(canon)) return;
  const model::CanonicalProblem& p = canon.problem;

  SolverState s = make_state(p, 1.0, 0.5);
  clear_direction(s);
  s.dtau = 1.0;  // tau grows, so `l tau` rises to meet a stationary x

  // By hand: the only decreasing quantities are `x_j - l_j tau` over columns
  // with a finite, nonzero lower bound, each moving at rate `-l_j`.
  Real oracle = solver::kUnboundedStep;
  std::size_t contributors = 0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    const Real l = p.col_lower[j];
    if (!core::is_finite_bound(l) || l <= 0.0) continue;
    oracle = std::fmin(oracle, (s.x[j] - l * s.tau) / l);
    ++contributors;
  }
  CHECK(contributors > 0);
  CHECK(oracle < solver::kUnboundedStep);
  CHECK_NEAR(solver::homogeneous_alpha_max(p, s, /*affine=*/false), oracle, 1e-14);

  // The mutation this test exists to catch: with the `l dtau` term removed,
  // every direction above is zero and nothing binds. Assert the two answers
  // are genuinely different, so the check above cannot pass by coincidence.
  CHECK(oracle < 0.5 * solver::kUnboundedStep);

  // The same coupling on the other side, with the sign reversed: shrinking tau
  // pulls `u tau` down onto a stationary x.
  SolverState t = make_state(p, 1.0, 0.5);
  clear_direction(t);
  t.dtau = -1.0;
  Real upper_oracle = solver::kUnboundedStep;
  std::size_t upper_contributors = 0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    const Real u = p.col_upper[j];
    if (!core::is_finite_bound(u) || u <= 0.0) continue;
    upper_oracle = std::fmin(upper_oracle, (u * t.tau - t.x[j]) / u);
    ++upper_contributors;
  }
  CHECK(upper_contributors > 0);
  // `tau` itself is also decreasing now, so it competes.
  upper_oracle = std::fmin(upper_oracle, t.tau);
  CHECK_NEAR(solver::homogeneous_alpha_max(p, t, /*affine=*/false), upper_oracle, 1e-14);
}

/// The affine flag must read the affine direction and nothing else. Getting
/// this wrong would silently feed the corrected step into [AA] (1.12), which
/// expects the PURE NEWTON step.
void test_affine_and_corrected_directions_are_distinct() {
  model::CanonicalResult canon;
  if (!build_general(canon)) return;
  const model::CanonicalProblem& p = canon.problem;

  SolverState s = make_state(p, 1.0, 0.5);
  clear_direction(s);
  const std::size_t n = p.num_cols();
  const std::size_t m = p.num_rows();
  const std::size_t m_i = p.num_inequality_rows();
  s.dx_aff = core::RealVector(n, 0.0);
  s.dz_aff = core::RealVector(n, 0.0);
  s.dv_aff = core::RealVector(n, 0.0);
  s.dy_aff = core::RealVector(m, 0.0);
  s.ds_aff = core::RealVector(m_i, 0.0);
  s.dtau_aff = -0.5;   // affine: tau falls fast
  s.dkappa_aff = 0.0;
  s.dtau = -0.1;       // corrected: tau falls slowly

  const Real affine = solver::homogeneous_alpha_max(p, s, /*affine=*/true);
  const Real corrected = solver::homogeneous_alpha_max(p, s, /*affine=*/false);
  CHECK(affine < corrected);

  // `dtau` is the only nonzero component in either direction, so EVERY binding
  // quantity moves at a rate proportional to it and the two steps must be
  // exactly inversely proportional. That pins both answers without having to
  // work out which column binds, and it fails if either call reads the wrong
  // direction: mixing them would make one side's rate 0.5 and the other's 0.1
  // on the same quantity.
  CHECK_NEAR(affine * 0.5, corrected * 0.1, 1e-12);
  CHECK(affine > 0.0);
}

/// [AA] (1.20) and section 1.4.3.
void test_step_size_respects_beta3_and_the_centrality_condition() {
  model::CanonicalResult canon;
  if (!build_general(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  HomogeneousParameters params;

  SolverState s = make_state(p, 1.0, 0.5);
  clear_direction(s);
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    s.dx[j] = (j % 2 == 0) ? -0.15 : 0.1;
    s.dz[j] = -0.05;
  }
  s.dtau = -0.05;
  s.dkappa = -0.1;

  const Real alpha_max = solver::homogeneous_alpha_max(p, s, false);
  const Real alpha = solver::homogeneous_step_size(p, s, false, params);
  CHECK(alpha > 0.0);
  // Never past the boundary, and never past 1.
  CHECK(alpha <= std::fmin(params.beta3 * alpha_max, 1.0) + 1e-15);
  CHECK(alpha <= 1.0 + 1e-15);

  // The accepted step must actually satisfy (1.20). Recomputed here from the
  // condition as printed, not from the implementation's internals.
  const auto pairs = static_cast<Real>(solver::homogeneous_pair_count(p));
  Real gap = 0.0;
  auto products = [&](Real a, auto&& sink) {
    for (std::size_t j = 0; j < p.num_cols(); ++j) {
      if (core::is_finite_bound(p.col_lower[j])) {
        const Real l = p.col_lower[j];
        sink((s.x[j] - l * s.tau + a * (s.dx[j] - l * s.dtau)) *
             (s.z[j] + a * s.dz[j]));
      }
      if (core::is_finite_bound(p.col_upper[j])) {
        const Real u = p.col_upper[j];
        sink((u * s.tau - s.x[j] + a * (u * s.dtau - s.dx[j])) *
             (s.v[j] + a * s.dv[j]));
      }
    }
    for (std::size_t k = 0; k < p.num_inequality_rows(); ++k) {
      const std::size_t i = p.num_equality + k;
      sink((s.s[k] + a * s.ds[k]) * (-s.y[i] - a * s.dy[i]));
    }
    sink((s.tau + a * s.dtau) * (s.kappa + a * s.dkappa));
  };
  products(alpha, [&gap](Real product) { gap += product; });
  const Real floor_value = params.beta2 * gap / pairs;
  products(alpha, [&](Real product) { CHECK(product >= floor_value - 1e-18); });

  // Every quantity stays strictly positive at the accepted step -- the
  // precondition for the next iteration to exist at all.
  products(alpha, [](Real product) { CHECK(product > 0.0); });
}

/// A direction that immediately destroys centrality must be rejected all the
/// way down rather than accepted at some tiny step, so the caller sees a
/// stall instead of grinding out its iteration budget.
void test_step_size_reports_a_stall_as_zero() {
  model::CanonicalResult canon;
  if (!build_general(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  HomogeneousParameters params;

  SolverState s = make_state(p, 1.0, 1e-16);
  clear_direction(s);
  // Drive kappa to the boundary while every other pair is left alone: the
  // (tau, kappa) product collapses relative to the rest and (1.20) can never
  // be met.
  s.dkappa = -1.0;
  params.beta2 = 0.5;  // a floor no lopsided iterate can satisfy
  CHECK_NEAR(solver::homogeneous_step_size(p, s, false, params), 0.0, 0.0);
}

/// [AA] section 1.4.5, one branch at a time.
void test_termination_branches() {
  HomogeneousParameters params;
  HomogeneousReference reference;
  reference.mu_0 = 1.0;

  SolverState s;
  s.tau = 1.0;
  s.kappa = 1e-14;
  s.mu = 1e-9;

  HomogeneousProgress progress;

  // Nothing met.
  progress = {1e-2, 1e-2, 1e-2, 1e-2};
  CHECK(solver::check_homogeneous_termination(s, progress, reference, 0.0, params) ==
        HomogeneousTermination::Continue);

  // First test: primal, dual and objective measures all within tolerance.
  progress = {1e-9, 1e-9, 1e-2, 1e-11};
  CHECK(solver::check_homogeneous_termination(s, progress, reference, 0.0, params) ==
        HomogeneousTermination::Optimal);

  // Feasible but the objective measure is not there yet: keep going.
  progress = {1e-9, 1e-9, 1e-2, 1e-4};
  CHECK(solver::check_homogeneous_termination(s, progress, reference, 0.0, params) ==
        HomogeneousTermination::Continue);

  // Second test: the embedding is satisfied, including its gap row, and tau
  // has collapsed relative to kappa.
  SolverState collapsed;
  collapsed.tau = 1e-13;
  collapsed.kappa = 2.0;
  collapsed.mu = 1e-3;
  progress = {1e-9, 1e-9, 1e-9, 1e-4};
  CHECK(solver::check_homogeneous_termination(collapsed, progress, reference, 0.0,
                                              params) ==
        HomogeneousTermination::Infeasible);

  // The gap row is part of that test, not decoration. With rho_G alone out of
  // tolerance the verdict must not be reported: kappa lives in that row, and
  // kappa is the certificate.
  progress = {1e-9, 1e-9, 1e-2, 1e-4};
  CHECK(solver::check_homogeneous_termination(collapsed, progress, reference, 0.0,
                                              params) !=
        HomogeneousTermination::Infeasible);

  // Third test: mu collapsed against its starting value with tau below
  // rho_I * min(1, kappa), and no residual evidence at all.
  SolverState illposed;
  illposed.tau = 1e-13;
  illposed.kappa = 2.0;
  illposed.mu = 1e-11;
  progress = {1e-1, 1e-1, 1e-1, 1e-1};
  CHECK(solver::check_homogeneous_termination(illposed, progress, reference, 0.0,
                                              params) ==
        HomogeneousTermination::IllPosed);

  // `min(1, kappa)` and not `max`: with kappa BELOW tau's threshold the third
  // test must not fire, even though mu has collapsed. This is the one place
  // the two infeasibility tests differ, so it is checked directly.
  SolverState small_kappa;
  small_kappa.tau = 1e-13;
  small_kappa.kappa = 1e-14;
  small_kappa.mu = 1e-11;
  CHECK(solver::check_homogeneous_termination(small_kappa, progress, reference, 0.0,
                                              params) !=
        HomogeneousTermination::IllPosed);

  // The late-stage relaxation: tolerances missed by less than a factor of 100,
  // tau well clear of kappa, and a long step on the previous iteration.
  SolverState fast;
  fast.tau = 1.0;
  fast.kappa = 1e-6;
  fast.mu = 1e-9;
  progress = {1e-7, 1e-7, 1e-2, 1e-9};
  CHECK(solver::check_homogeneous_termination(fast, progress, reference, 0.95, params) ==
        HomogeneousTermination::Optimal);
  // Without the long step there is no evidence of fast convergence, so the
  // relaxation is not available.
  CHECK(solver::check_homogeneous_termination(fast, progress, reference, 0.1, params) ==
        HomogeneousTermination::Continue);
}

/// The starting point must be strictly interior on the general form too --
/// that is the only property the algorithm actually requires of it.
void test_starting_point_is_strictly_interior() {
  model::CanonicalResult canon;
  if (!build_general(canon)) return;
  const model::CanonicalProblem& p = canon.problem;

  SolverState s;
  CHECK(solver::homogeneous_starting_point(p, s).ok());
  CHECK(s.tau > 0.0);
  CHECK(s.kappa > 0.0);
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    if (core::is_finite_bound(p.col_lower[j])) {
      CHECK(s.x[j] - p.col_lower[j] * s.tau > 0.0);
      CHECK(s.z[j] > 0.0);
    } else {
      CHECK_NEAR(s.z[j], 0.0, 0.0);  // no pair, so no dual
    }
    if (core::is_finite_bound(p.col_upper[j])) {
      CHECK(p.col_upper[j] * s.tau - s.x[j] > 0.0);
      CHECK(s.v[j] > 0.0);
    } else {
      CHECK_NEAR(s.v[j], 0.0, 0.0);
    }
  }
  for (std::size_t k = 0; k < p.num_inequality_rows(); ++k) {
    CHECK(s.s[k] > 0.0);
    // The pair is `(s, -y_I)`. [AA] (1.22)'s `y := 0` puts this product ON the
    // boundary rather than inside it -- the one place standard form's absence
    // of inequality rows hides a requirement. Both sides must be positive.
    CHECK(-s.y[p.num_equality + k] > 0.0);
  }
  for (std::size_t i = 0; i < p.num_equality; ++i) {
    CHECK_NEAR(s.y[i], 0.0, 0.0);  // unrestricted, no pair: (1.22)'s 0 stands
  }
  CHECK(s.mu > 0.0);

  // And a positive step must exist from it, or the first iteration is already
  // stuck.
  clear_direction(s);
  s.dtau = -0.01;
  HomogeneousParameters params;
  CHECK(solver::homogeneous_step_size(p, s, false, params) > 0.0);
}

}  // namespace

int main() {
  test_the_fixtures_are_what_they_claim();
  test_standard_form_recovers_the_published_formulas();
  test_gamma_matches_equation_1_12();
  test_tau_moves_the_bounds();
  test_affine_and_corrected_directions_are_distinct();
  test_step_size_respects_beta3_and_the_centrality_condition();
  test_step_size_reports_a_stall_as_zero();
  test_termination_branches();
  test_starting_point_is_strictly_interior();
  return ::sovsolve::test::report("homogeneous_step_test");
}
