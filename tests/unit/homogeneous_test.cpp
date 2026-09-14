// Module 25: the homogeneous self-dual embedding.
//
// THE LOAD-BEARING TEST IS THE EQUIVALENCE AT tau = 1.
//
// The embedding homogenizes every constant by `tau`, so at `tau = 1` it must
// reduce to the direct formulation of FORMULATION.md sections 5-6 exactly --
// not approximately, not up to a scale, but term for term. That gives a sign
// error in the homogenization somewhere to fail LOUDLY and immediately,
// instead of surfacing later as an interior-point run that merely converges
// badly on an engine which already only reaches 6-7 of 19 instances. A weak
// signal is no signal.
//
// The expected values are written here straight from FORMULATION sections 5
// and 6, not obtained by calling the direct implementation. Two
// implementations of the same formula can share a sign error; a formula
// transcribed from the specification and one written from the derivation
// cannot, unless the specification itself is wrong -- and that is a different
// bug worth finding too.
//
// Host-only, which is the point of putting this module in `sovsolve_solver`
// rather than beside the IPM it serves: the equivalence is checkable in the
// default `release` preset with no CUDA toolkit present.

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/Homogeneous.hpp"
#include "sovsolve/solver/SolverState.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using solver::HomogeneousResiduals;
using solver::HomogeneousVerdict;
using solver::SolverState;

namespace {

/// A small canonical problem with equality rows, inequality rows, finite
/// bounds on one side, both sides, and neither -- so every branch of the
/// finite-bound guards is exercised rather than only the common one.
bool build(model::CanonicalResult& out) {
  auto parsed = io::parseProblem(R"(Minimize
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
                                 io::FileFormat::Lp);
  if (!parsed.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "parse", parsed.error().format());
    return false;
  }
  model::Options options;
  options.log.level = model::LogOptions::Level::Silent;
  auto canon = model::canonicalize(parsed.value(), options);
  if (!canon.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "canonicalize",
                             canon.error().format());
    return false;
  }
  out = std::move(canon.value());
  return true;
}

/// A deterministic strictly-interior-ish point. Not a solution -- the
/// residuals are being compared, not converged.
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
    s.x[j] = 0.5 + 0.25 * static_cast<Real>(j % 5);
    s.z[j] = 0.3 + 0.1 * static_cast<Real>(j % 3);
    s.v[j] = 0.2 + 0.15 * static_cast<Real>(j % 4);
  }
  for (std::size_t i = 0; i < m; ++i) s.y[i] = -0.4 - 0.1 * static_cast<Real>(i % 3);
  for (std::size_t k = 0; k < m_i; ++k) s.s[k] = 0.7 + 0.2 * static_cast<Real>(k % 2);
  s.tau = tau;
  s.kappa = kappa;
  return s;
}

// --------------------------------------------------------------------------

/// FORMULATION.md section 5, transcribed. This is the oracle.
void test_residuals_reduce_to_the_direct_form_at_tau_one() {
  model::CanonicalResult canon;
  if (!build(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  const std::size_t n = p.num_cols();
  const std::size_t m = p.num_rows();
  const std::size_t m_e = p.num_equality;
  const std::size_t m_i = p.num_inequality_rows();

  const Real mu = 0.125;
  SolverState state = make_state(p, 1.0, 0.0);

  HomogeneousResiduals hsd;
  core::Status st = solver::compute_homogeneous_residuals(p, state, mu, hsd);
  CHECK(st.ok());
  if (!st.ok()) return;

  // rp_E = A_E x - b_E ;  rp_I = A_I x + s - b_I
  const auto& csr = p.A.csr;
  for (std::size_t i = 0; i < m; ++i) {
    Real ax = 0.0;
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      ax += csr.values()[k] * state.x[static_cast<std::size_t>(csr.indices()[k])];
    }
    Real expected = ax - p.b[i];
    if (i >= m_e) expected += state.s[i - m_e];
    CHECK_NEAR(hsd.rp[i], expected, 1e-12);
  }

  // rd = c - A'y - z + v   (Q empty here)
  const auto& csc = p.A.csc;
  for (std::size_t j = 0; j < n; ++j) {
    Real aty = 0.0;
    for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      aty += csc.values()[k] * state.y[static_cast<std::size_t>(csc.indices()[k])];
    }
    CHECK_NEAR(hsd.rd[j], p.c[j] - aty - state.z[j] + state.v[j], 1e-12);
  }

  // rxz = (x-l).*z - mu, omitted where the lower bound is infinite;
  // ruv = (u-x).*v - mu, omitted where the upper bound is infinite.
  for (std::size_t j = 0; j < n; ++j) {
    const Real expect_xz = core::is_finite_bound(p.col_lower[j])
                               ? (state.x[j] - p.col_lower[j]) * state.z[j] - mu
                               : 0.0;
    const Real expect_uv = core::is_finite_bound(p.col_upper[j])
                               ? (p.col_upper[j] - state.x[j]) * state.v[j] - mu
                               : 0.0;
    CHECK_NEAR(hsd.rxz[j], expect_xz, 1e-12);
    CHECK_NEAR(hsd.ruv[j], expect_uv, 1e-12);
  }

  // rsy = -s.*y_I - mu
  for (std::size_t k = 0; k < m_i; ++k) {
    CHECK_NEAR(hsd.rsy[k], -state.s[k] * state.y[m_e + k] - mu, 1e-12);
  }
}

/// The model must actually contain the cases the guards are for, or the test
/// above passes while proving much less than it looks like it proves.
void test_the_fixture_exercises_every_bound_case() {
  model::CanonicalResult canon;
  if (!build(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  bool has_both = false;
  bool has_lower_only = false;
  bool has_neither = false;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    const bool lo = core::is_finite_bound(p.col_lower[j]);
    const bool hi = core::is_finite_bound(p.col_upper[j]);
    if (lo && hi) has_both = true;
    if (lo && !hi) has_lower_only = true;
    if (!lo && !hi) has_neither = true;
  }
  CHECK(has_both);
  CHECK(has_lower_only);
  CHECK(has_neither);
  CHECK(p.num_equality > 0);
  CHECK(p.num_inequality_rows() > 0);
}

/// FORMULATION.md section 6, plus the embedding's own pair (section 13.4).
void test_mu_matches_the_direct_definition_at_tau_one() {
  model::CanonicalResult canon;
  if (!build(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  SolverState state = make_state(p, 1.0, 0.0);

  Real total = 0.0;
  std::size_t pairs = 0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    if (core::is_finite_bound(p.col_lower[j])) {
      total += (state.x[j] - p.col_lower[j]) * state.z[j];
      ++pairs;
    }
    if (core::is_finite_bound(p.col_upper[j])) {
      total += (p.col_upper[j] - state.x[j]) * state.v[j];
      ++pairs;
    }
  }
  for (std::size_t k = 0; k < p.num_inequality_rows(); ++k) {
    total += -state.s[k] * state.y[p.num_equality + k];
    ++pairs;
  }
  // kappa = 0 contributes nothing to the numerator but still counts as a pair.
  const Real expected = total / static_cast<Real>(pairs + 1);
  CHECK_NEAR(solver::homogeneous_mu(p, state), expected, 1e-12);
}

/// The gap row has no counterpart in the direct formulation, so it is checked
/// against its own definition -- including that an infinite bound contributes
/// nothing rather than a NaN.
void test_gap_row() {
  model::CanonicalResult canon;
  if (!build(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  SolverState state = make_state(p, 1.0, 0.75);

  HomogeneousResiduals hsd;
  core::Status st = solver::compute_homogeneous_residuals(p, state, 0.1, hsd);
  CHECK(st.ok());
  if (!st.ok()) return;

  Real cx = 0.0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) cx += p.c[j] * state.x[j];
  Real by = 0.0;
  for (std::size_t i = 0; i < p.num_rows(); ++i) by += p.b[i] * state.y[i];
  Real lz = 0.0;
  Real uv = 0.0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    if (core::is_finite_bound(p.col_lower[j])) lz += p.col_lower[j] * state.z[j];
    if (core::is_finite_bound(p.col_upper[j])) uv += p.col_upper[j] * state.v[j];
  }
  CHECK_NEAR(hsd.rg, cx - by - lz + uv + state.kappa, 1e-12);
  CHECK(std::isfinite(hsd.rg));
  CHECK_NEAR(hsd.rtk, state.tau * state.kappa - 0.1, 1e-12);
}

/// Homogenizing by tau must be exactly a SCALING of the constants, so doubling
/// tau and the iterate together leaves every residual doubled -- the
/// embedding's defining property, and a check that no constant was left
/// un-homogenized.
void test_residuals_are_homogeneous() {
  model::CanonicalResult canon;
  if (!build(canon)) return;
  const model::CanonicalProblem& p = canon.problem;

  SolverState a = make_state(p, 1.0, 0.5);
  SolverState b = make_state(p, 2.0, 1.0);
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    b.x[j] = a.x[j] * 2.0;
    b.z[j] = a.z[j] * 2.0;
    b.v[j] = a.v[j] * 2.0;
  }
  for (std::size_t i = 0; i < p.num_rows(); ++i) b.y[i] = a.y[i] * 2.0;
  for (std::size_t k = 0; k < p.num_inequality_rows(); ++k) b.s[k] = a.s[k] * 2.0;

  HomogeneousResiduals ra;
  HomogeneousResiduals rb;
  // mu is quadratic in the iterate, so it scales by 4 alongside the
  // complementarity products it is subtracted from.
  if (!solver::compute_homogeneous_residuals(p, a, 0.25, ra).ok()) return;
  if (!solver::compute_homogeneous_residuals(p, b, 1.0, rb).ok()) return;

  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    CHECK_NEAR(rb.rp[i], 2.0 * ra.rp[i], 1e-12);
  }
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    CHECK_NEAR(rb.rd[j], 2.0 * ra.rd[j], 1e-12);
    CHECK_NEAR(rb.rxz[j], 4.0 * ra.rxz[j], 1e-12);
    CHECK_NEAR(rb.ruv[j], 4.0 * ra.ruv[j], 1e-12);
  }
  CHECK_NEAR(rb.rg, 2.0 * ra.rg, 1e-12);
}

/// Section 13.2's reading of the final iterate.
void test_classification() {
  model::CanonicalResult canon;
  if (!build(canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  constexpr Real kFloor = 1e-6;

  // tau healthy -> optimal.
  SolverState optimal = make_state(p, 1.0, 1e-12);
  CHECK(solver::classify_homogeneous(p, optimal, kFloor) ==
        HomogeneousVerdict::Optimal);

  // tau collapsed with a strictly positive dual objective -> a dual ray, so
  // the primal has no feasible point. `y` is driven positive on the equality
  // row, whose `b` is positive, to make `b'y` dominate.
  SolverState infeasible = make_state(p, 1e-14, 1.0);
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    infeasible.x[j] = 0.0;
    infeasible.z[j] = 0.0;
    infeasible.v[j] = 0.0;
  }
  for (std::size_t i = 0; i < p.num_rows(); ++i) {
    infeasible.y[i] = p.b[i] > 0.0 ? 1.0 : 0.0;
  }
  CHECK(solver::classify_homogeneous(p, infeasible, kFloor) ==
        HomogeneousVerdict::PrimalInfeasible);

  // tau collapsed with a strictly negative primal objective -> a primal ray,
  // so the primal is unbounded.
  SolverState unbounded = make_state(p, 1e-14, 1.0);
  for (std::size_t i = 0; i < p.num_rows(); ++i) unbounded.y[i] = 0.0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    unbounded.z[j] = 0.0;
    unbounded.v[j] = 0.0;
    unbounded.x[j] = p.c[j] > 0.0 ? -1.0 : (p.c[j] < 0.0 ? 1.0 : 0.0);
  }
  CHECK(solver::classify_homogeneous(p, unbounded, kFloor) ==
        HomogeneousVerdict::DualInfeasible);

  // tau collapsed with neither objective decisive -> say so, do not guess.
  SolverState stalled = make_state(p, 1e-14, 1.0);
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    stalled.x[j] = 0.0;
    stalled.z[j] = 0.0;
    stalled.v[j] = 0.0;
  }
  for (std::size_t i = 0; i < p.num_rows(); ++i) stalled.y[i] = 0.0;
  CHECK(solver::classify_homogeneous(p, stalled, kFloor) ==
        HomogeneousVerdict::Indeterminate);
}

/// Recovery is a division, and it must be refused rather than performed when
/// `tau` has collapsed -- dividing a ray by a number near zero produces a
/// very large vector that looks like an answer.
void test_recovery() {
  model::CanonicalResult canon;
  if (!build(canon)) return;
  const model::CanonicalProblem& p = canon.problem;

  SolverState state = make_state(p, 4.0, 2.0);
  const SolverState before = make_state(p, 4.0, 2.0);
  CHECK(solver::recover_from_homogeneous(state).ok());
  CHECK_NEAR(state.tau, 1.0, 1e-15);
  CHECK_NEAR(state.kappa, before.kappa / 4.0, 1e-12);
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    CHECK_NEAR(state.x[j], before.x[j] / 4.0, 1e-12);
    CHECK_NEAR(state.z[j], before.z[j] / 4.0, 1e-12);
  }

  SolverState collapsed = make_state(p, 0.0, 1.0);
  CHECK(!solver::recover_from_homogeneous(collapsed).ok());
}

}  // namespace

int main() {
  test_the_fixture_exercises_every_bound_case();
  test_residuals_reduce_to_the_direct_form_at_tau_one();
  test_mu_matches_the_direct_definition_at_tau_one();
  test_gap_row();
  test_residuals_are_homogeneous();
  test_classification();
  test_recovery();
  return ::sovsolve::test::report("homogeneous_test");
}
