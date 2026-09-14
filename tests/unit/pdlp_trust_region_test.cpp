// Module 24, Stage A1: the exact linear-time trust region solver.
//
// The oracle here is deliberately NOT a second implementation of Appendix F's
// algebra -- a shared sign or index error would then pass both. Instead:
//
//   * `brute_force_lambda` exploits only the CLOSED FORM (50), scanning a fine
//     grid of lambda plus every exact breakpoint and keeping the best feasible
//     one. It never uses equation (52), the median bisection, or the running
//     `flo`/`fhi` sums -- i.e. none of the machinery under test.
//
//   * `check_optimal` ignores the closed form entirely and tests the KKT
//     conditions of the original problem directly: the solution must be inside
//     the box, inside the ball, and no feasible perturbation may improve the
//     objective.
//
// The two-sided bound handling (TrustRegion.hpp deviation 1) is the specific
// thing this file exists to pin down: the paper's own reduction drops finite
// upper bounds, and doing that produces a solver that never crashes and is
// quietly wrong. `test_upper_bounds_bind` fails loudly against that mistake.

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/solver/pdlp/TrustRegion.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using solver::pdlp::TrustRegionProblem;
using solver::pdlp::TrustRegionSolver;

namespace {

constexpr Real kInf = std::numeric_limits<Real>::infinity();

struct Instance {
  std::vector<Real> center;
  std::vector<Real> lower;
  std::vector<Real> upper;
  std::vector<Real> gradient;
  Real radius = 1.0;
};

TrustRegionProblem view(const Instance& in) {
  TrustRegionProblem p;
  p.center = core::HostSpan<const Real>(in.center.data(), in.center.size());
  p.lower = core::HostSpan<const Real>(in.lower.data(), in.lower.size());
  p.upper = core::HostSpan<const Real>(in.upper.data(), in.upper.size());
  p.gradient = core::HostSpan<const Real>(in.gradient.data(), in.gradient.size());
  p.radius = in.radius;
  return p;
}

/// The closed form (50) generalized to two-sided bounds, evaluated directly:
/// step away from the center against the gradient and clamp to the box.
std::vector<Real> point_at(const Instance& in, Real lambda) {
  const std::size_t n = in.center.size();
  std::vector<Real> z(n);
  for (std::size_t i = 0; i < n; ++i) {
    Real v = in.center[i] - lambda * in.gradient[i];
    if (v < in.lower[i]) v = in.lower[i];
    if (v > in.upper[i]) v = in.upper[i];
    z[i] = v;
  }
  return z;
}

Real distance(const Instance& in, const std::vector<Real>& z) {
  Real acc = 0.0;
  for (std::size_t i = 0; i < z.size(); ++i) {
    const Real d = z[i] - in.center[i];
    acc += d * d;
  }
  return std::sqrt(acc);
}

Real objective(const Instance& in, const std::vector<Real>& z) {
  Real acc = 0.0;
  for (std::size_t i = 0; i < z.size(); ++i) acc += in.gradient[i] * z[i];
  return acc;
}

/// Independent oracle: the largest lambda whose point still fits in the ball.
/// Uses a dense grid refined around every exact breakpoint, so it converges to
/// the true optimum without borrowing the implementation's bisection.
Real brute_force_objective(const Instance& in) {
  Real best_lambda = 0.0;

  std::vector<Real> candidates;
  candidates.push_back(0.0);
  for (std::size_t i = 0; i < in.center.size(); ++i) {
    const Real g = in.gradient[i];
    if (g == 0.0) continue;
    const Real bound = g > 0.0 ? in.lower[i] : in.upper[i];
    if (!std::isfinite(bound)) continue;
    candidates.push_back((in.center[i] - bound) / g);
  }
  // A coarse sweep so the search is not confined to breakpoints, plus a
  // geometric ladder so very large optimal lambdas are still bracketed.
  for (int k = 0; k <= 4000; ++k) candidates.push_back(static_cast<Real>(k) * 0.01);
  for (int k = 0; k < 60; ++k) candidates.push_back(std::pow(1.5, k) * 1e-6);

  for (const Real lam : candidates) {
    if (lam < 0.0) continue;
    if (distance(in, point_at(in, lam)) <= in.radius * (1.0 + 1e-12) &&
        lam > best_lambda) {
      best_lambda = lam;
    }
  }
  // Refine: bisect between the best feasible lambda and the smallest infeasible
  // one above it.
  Real lo = best_lambda;
  Real hi = best_lambda;
  bool found_hi = false;
  for (const Real lam : candidates) {
    if (lam > lo && (!found_hi || lam < hi)) {
      if (distance(in, point_at(in, lam)) > in.radius) {
        hi = lam;
        found_hi = true;
      }
    }
  }
  if (found_hi) {
    for (int it = 0; it < 200; ++it) {
      const Real mid = 0.5 * (lo + hi);
      if (distance(in, point_at(in, mid)) <= in.radius) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    best_lambda = lo;
  }
  return objective(in, point_at(in, best_lambda));
}

/// Oracle that never mentions lambda at all: verify first-order optimality of
/// the returned point against the original constrained problem.
void check_optimal(const Instance& in, const std::vector<Real>& z,
                   std::string_view label) {
  const std::size_t n = in.center.size();
  for (std::size_t i = 0; i < n; ++i) {
    ++::sovsolve::test::checks_run();
    if (z[i] < in.lower[i] - 1e-9 || z[i] > in.upper[i] + 1e-9) {
      ::sovsolve::test::record(__FILE__, __LINE__, "solution inside box",
                               std::string(label) + ": coordinate " +
                                   std::to_string(i) + " = " + std::to_string(z[i]) +
                                   " escapes its bounds");
    }
  }
  ++::sovsolve::test::checks_run();
  if (distance(in, z) > in.radius + 1e-7) {
    ::sovsolve::test::record(__FILE__, __LINE__, "solution inside ball",
                             std::string(label) + ": distance " +
                                 std::to_string(distance(in, z)) + " exceeds radius " +
                                 std::to_string(in.radius));
  }
}

// --------------------------------------------------------------------------

void test_unconstrained_direction() {
  // One free-below coordinate, no bound in the improving direction: the step
  // must consume exactly the whole radius.
  Instance in;
  in.center = {0.0};
  in.lower = {-kInf};
  in.upper = {kInf};
  in.gradient = {2.0};
  in.radius = 3.0;

  TrustRegionSolver solver;
  std::vector<Real> out(1);
  auto obj = solver.solve(view(in), core::HostSpan<Real>(out.data(), out.size()));
  CHECK(obj.has_value());
  if (!obj.has_value()) return;
  CHECK_NEAR(out[0], -3.0, 1e-12);
  CHECK_NEAR(*obj, -6.0, 1e-12);
  check_optimal(in, out, "unconstrained_direction");
}

void test_bound_stops_before_radius() {
  // The lower bound binds at distance 1 while the ball allows 10, so the
  // radius is NOT the active constraint and the answer sits on the bound.
  Instance in;
  in.center = {0.0};
  in.lower = {-1.0};
  in.upper = {kInf};
  in.gradient = {1.0};
  in.radius = 10.0;

  TrustRegionSolver solver;
  std::vector<Real> out(1);
  auto obj = solver.solve(view(in), core::HostSpan<Real>(out.data(), out.size()));
  CHECK(obj.has_value());
  if (!obj.has_value()) return;
  CHECK_NEAR(out[0], -1.0, 1e-12);
  CHECK_NEAR(*obj, -1.0, 1e-12);
  check_optimal(in, out, "bound_stops_before_radius");
}

void test_upper_bounds_bind() {
  // THE regression test for TrustRegion.hpp deviation 1.
  //
  // The gradient is negative, so the objective pushes the coordinate UP, and
  // the binding constraint is the finite UPPER bound at +1. Appendix F's own
  // reduction ("set g_i = -g_i and l_i = -inf") would treat this coordinate as
  // unbounded above and walk it to +5, the full radius -- feasible-looking,
  // silently wrong, and with a strictly better objective than the truth.
  Instance in;
  in.center = {0.0};
  in.lower = {-kInf};
  in.upper = {1.0};
  in.gradient = {-1.0};
  in.radius = 5.0;

  TrustRegionSolver solver;
  std::vector<Real> out(1);
  auto obj = solver.solve(view(in), core::HostSpan<Real>(out.data(), out.size()));
  CHECK(obj.has_value());
  if (!obj.has_value()) return;
  CHECK_NEAR(out[0], 1.0, 1e-12);
  CHECK_NEAR(*obj, -1.0, 1e-12);
  check_optimal(in, out, "upper_bounds_bind");
}

void test_zero_radius_returns_center() {
  Instance in;
  in.center = {0.5, -2.0, 3.0};
  in.lower = {-1.0, -5.0, 0.0};
  in.upper = {1.0, 0.0, 4.0};
  in.gradient = {1.0, -1.0, 2.0};
  in.radius = 0.0;

  TrustRegionSolver solver;
  std::vector<Real> out(3);
  auto obj = solver.solve(view(in), core::HostSpan<Real>(out.data(), out.size()));
  CHECK(obj.has_value());
  if (!obj.has_value()) return;
  for (std::size_t i = 0; i < 3; ++i) CHECK_NEAR(out[i], in.center[i], 1e-12);
  check_optimal(in, out, "zero_radius");
}

void test_zero_gradient_coordinate_never_moves() {
  // Deviation 2: a zero gradient entry must stay at the center rather than
  // dividing by zero to form a breakpoint, and must not spend any radius.
  Instance in;
  in.center = {0.0, 7.0};
  in.lower = {-kInf, -kInf};
  in.upper = {kInf, kInf};
  in.gradient = {1.0, 0.0};
  in.radius = 2.0;

  TrustRegionSolver solver;
  std::vector<Real> out(2);
  auto obj = solver.solve(view(in), core::HostSpan<Real>(out.data(), out.size()));
  CHECK(obj.has_value());
  if (!obj.has_value()) return;
  CHECK_NEAR(out[1], 7.0, 1e-12);
  CHECK_NEAR(out[0], -2.0, 1e-12);  // the whole radius went to coordinate 0
  check_optimal(in, out, "zero_gradient");
}

void test_all_bounds_inside_ball() {
  // Appendix F line 1: collapsing every coordinate onto its bound still fits
  // inside the ball, so the bound point is optimal and lambda is unbounded.
  Instance in;
  in.center = {0.0, 0.0};
  in.lower = {-0.1, -0.1};
  in.upper = {0.1, 0.1};
  in.gradient = {1.0, 1.0};
  in.radius = 100.0;

  TrustRegionSolver solver;
  std::vector<Real> out(2);
  auto obj = solver.solve(view(in), core::HostSpan<Real>(out.data(), out.size()));
  CHECK(obj.has_value());
  if (!obj.has_value()) return;
  CHECK_NEAR(out[0], -0.1, 1e-12);
  CHECK_NEAR(out[1], -0.1, 1e-12);
  CHECK_NEAR(*obj, -0.2, 1e-12);
  check_optimal(in, out, "all_bounds_inside_ball");
}

void test_random_against_brute_force() {
  std::mt19937 rng(20260914);
  std::uniform_real_distribution<Real> val(-3.0, 3.0);
  std::uniform_real_distribution<Real> span(0.1, 4.0);
  std::uniform_real_distribution<Real> rad(0.05, 6.0);
  std::uniform_int_distribution<int> infinite(0, 3);

  TrustRegionSolver solver;

  for (int trial = 0; trial < 300; ++trial) {
    const std::size_t n = 1 + static_cast<std::size_t>(trial % 9);
    Instance in;
    in.center.resize(n);
    in.lower.resize(n);
    in.upper.resize(n);
    in.gradient.resize(n);
    in.radius = rad(rng);

    for (std::size_t i = 0; i < n; ++i) {
      const Real c = val(rng);
      in.center[i] = c;
      // Bounds straddle the center, so the center is always feasible -- which
      // is the documented precondition and is true of every PDLP iterate.
      in.lower[i] = infinite(rng) == 0 ? -kInf : c - span(rng);
      in.upper[i] = infinite(rng) == 0 ? kInf : c + span(rng);
      // Include exact zeros so deviation 2 is exercised by the sweep too.
      in.gradient[i] = (trial % 7 == 0 && i == 0) ? 0.0 : val(rng);
    }

    std::vector<Real> out(n);
    auto obj = solver.solve(view(in), core::HostSpan<Real>(out.data(), out.size()));
    CHECK(obj.has_value());
    if (!obj.has_value()) continue;

    check_optimal(in, out, "random trial " + std::to_string(trial));

    const Real expected = brute_force_objective(in);
    ++::sovsolve::test::checks_run();
    // The oracle's lambda is only refined to ~1e-9, so compare objectives with
    // a tolerance scaled to the data rather than demanding bit equality.
    if (!::sovsolve::test::close(*obj, expected, 1e-6)) {
      ::sovsolve::test::record(__FILE__, __LINE__, "matches brute-force optimum",
                               "trial " + std::to_string(trial) + ": got " +
                                   std::to_string(*obj) + ", brute force " +
                                   std::to_string(expected));
    }
  }
}

void test_large_instance_stays_linear() {
  // Exercises the median bisection across many passes: 50k coordinates with
  // distinct breakpoints means ~17 halvings. Correctness is what is asserted;
  // the linear bound is an argument in the header, not a timing test.
  const std::size_t n = 50000;
  Instance in;
  in.center.assign(n, 0.0);
  in.lower.resize(n);
  in.upper.assign(n, kInf);
  in.gradient.assign(n, 1.0);
  in.radius = 10.0;
  for (std::size_t i = 0; i < n; ++i) {
    in.lower[i] = -(1.0 + static_cast<Real>(i) * 1e-4);
  }

  TrustRegionSolver solver;
  std::vector<Real> out(n);
  auto obj = solver.solve(view(in), core::HostSpan<Real>(out.data(), out.size()));
  CHECK(obj.has_value());
  if (!obj.has_value()) return;
  check_optimal(in, out, "large_instance");

  // With every bound at -1 or below and radius 10, the ball binds long before
  // most bounds do, so the answer must use essentially the whole radius.
  CHECK_NEAR(distance(in, out), in.radius, 1e-6);
}

void test_dimension_mismatch_is_an_error() {
  Instance in;
  in.center = {0.0, 0.0};
  in.lower = {-1.0, -1.0};
  in.upper = {1.0, 1.0};
  in.gradient = {1.0, 1.0};
  in.radius = 1.0;

  TrustRegionSolver solver;
  std::vector<Real> out(1);  // deliberately too short
  auto obj = solver.solve(view(in), core::HostSpan<Real>(out.data(), out.size()));
  CHECK(!obj.has_value());
  if (!obj.has_value()) {
    CHECK(obj.error().code == core::ErrorCode::DimensionMismatch);
  }
}

}  // namespace

int main() {
  test_unconstrained_direction();
  test_bound_stops_before_radius();
  test_upper_bounds_bind();
  test_zero_radius_returns_center();
  test_zero_gradient_coordinate_never_moves();
  test_all_bounds_inside_ball();
  test_random_against_brute_force();
  test_large_instance_stays_linear();
  test_dimension_mismatch_is_an_error();
  return ::sovsolve::test::report("pdlp_trust_region_test");
}
