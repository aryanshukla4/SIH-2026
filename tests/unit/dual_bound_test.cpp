// Module 27: valid dual bounds (Neumaier-Shcherbina primal-bound-shift, as
// presented in Steffy & Wolter section 2).
//
// THE TEST THAT MATTERS IS `test_the_uncorrected_bound_is_actually_invalid`.
//
// Everything else here checks that the corrected bound is right. That one
// checks the correction is NECESSARY -- it takes an approximate dual, evaluates
// the dual objective directly the way a naive branch-and-bound would, and shows
// the result EXCEEDS the true optimum. A bound that exceeds the optimum prunes
// the node containing it. So this is not a test about accuracy; it is a test
// that the unsound thing is genuinely unsound, and without it the rest of this
// module could be deleted and every other test here would still pass.
//
// Validity is one-sided and that asymmetry drives the assertions: for a
// minimization, a valid bound may be arbitrarily LOOSE (far below the optimum)
// and is still correct, but may never be HIGH by even one ulp. So the checks
// below are `bound <= optimum`, never `bound ~= optimum`.

#include <cmath>
#include <cstddef>
#include <limits>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/DualBound.hpp"
#include "sovsolve/solver/simplex/SimplexSolution.hpp"
#include "sovsolve/solver/simplex/SolveSimplex.hpp"
#include "tests/TestMain.hpp"

using namespace sovsolve;  // NOLINT(build/namespaces)
using core::Real;
using solver::DualBound;

namespace {

bool canonicalize(const char* text, model::CanonicalResult& out) {
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

/// The exact answer, from the engine that gets 19/19 on the corpus.
bool exact_solution(const model::CanonicalProblem& p, model::Solution& out) {
  model::Options options;
  options.log.level = model::LogOptions::Level::Silent;
  auto result = solver::simplex::solve_simplex(p, options);
  if (!result.has_value()) {
    ::sovsolve::test::record(__FILE__, __LINE__, "simplex", result.error().format());
    return false;
  }
  out = solver::simplex::to_canonical_solution(p, *result);
  return out.status == core::SolverStatus::Optimal;
}

/// The dual objective as a naive caller would compute it: no repair, just
/// `b'y + l'z - u'v` at whatever point the solver handed back. This is the
/// quantity that is NOT a bound.
Real raw_dual_objective(const model::CanonicalProblem& p, const core::RealVector& y,
                        const core::RealVector& z, const core::RealVector& v) {
  Real value = 0.0;
  for (std::size_t i = 0; i < p.num_rows(); ++i) value += p.b[i] * y[i];
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    if (core::is_finite_bound(p.col_lower[j])) value += p.col_lower[j] * z[j];
    if (core::is_finite_bound(p.col_upper[j])) value -= p.col_upper[j] * v[j];
  }
  return value;
}

/// A deterministic perturbation, standing in for "a first-order method stopped
/// early". Scaled so it is a realistic 1e-3-ish dual error, not noise.
void perturb(core::RealVector& vec, unsigned seed, Real scale) {
  for (std::size_t i = 0; i < vec.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    const Real unit = static_cast<Real>((seed >> 8) % 2000u) / 1000.0 - 1.0;
    vec[i] += scale * unit;
  }
}

core::RealVector copy_of(const core::RealVector& v) {
  core::RealVector out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = v[i];
  return out;
}

/// Bounded on both sides everywhere, so primal-bound-shift applies in full.
const char* kBoxed = R"(Minimize
 obj: -x - 2 y + 3 z
Subject To
 c1: x + y + z <= 10
 c2: x - y <= 3
 e1: x + 2 y - z = 4
Bounds
 0 <= x <= 6
 0 <= y <= 5
 0 <= z <= 4
End
)";

// -------------------------------------------------------------------------

/// THE ONE THAT MATTERS. An approximate dual, used directly, produces a number
/// ABOVE the true minimum -- which would prune the optimum away.
void test_the_uncorrected_bound_is_actually_invalid() {
  model::CanonicalResult canon;
  if (!canonicalize(kBoxed, canon)) return;
  const model::CanonicalProblem& p = canon.problem;

  model::Solution exact;
  if (!exact_solution(p, exact)) return;
  const Real optimum = exact.objective;

  // Search a range of perturbations for one that breaks the naive bound. It
  // does not take much: any dual error in the wrong direction does it.
  bool found_invalid_raw = false;
  bool corrected_always_valid = true;
  Real worst_raw_excess = 0.0;

  for (unsigned seed = 1; seed <= 40; ++seed) {
    core::RealVector y = copy_of(exact.y);
    core::RealVector z = copy_of(exact.z);
    core::RealVector v = copy_of(exact.v);
    perturb(y, seed, 1e-2);
    perturb(z, seed + 977u, 1e-2);
    perturb(v, seed + 5077u, 1e-2);

    const Real raw = raw_dual_objective(p, y, z, v);
    if (raw > optimum + 1e-9) {
      found_invalid_raw = true;
      worst_raw_excess = std::fmax(worst_raw_excess, raw - optimum);
    }

    DualBound bound;
    CHECK(solver::compute_dual_bound(p, y, z, v, bound).ok());
    CHECK(bound.finite);
    // The whole point: never above the optimum, at any perturbation.
    if (bound.bound > optimum + 1e-9) corrected_always_valid = false;
  }

  // If this fails the test is not exercising anything -- it means the naive
  // bound happened to stay valid and the corrected one has nothing to prove.
  CHECK(found_invalid_raw);
  CHECK(worst_raw_excess > 1e-6);
  CHECK(corrected_always_valid);
}

/// Validity is the only hard requirement, so it gets its own sweep at several
/// magnitudes of dual error -- including large ones, where a first-order method
/// stopped very early would be.
void test_validity_across_error_magnitudes() {
  model::CanonicalResult canon;
  if (!canonicalize(kBoxed, canon)) return;
  const model::CanonicalProblem& p = canon.problem;

  model::Solution exact;
  if (!exact_solution(p, exact)) return;
  const Real optimum = exact.objective;

  for (Real scale : {1e-8, 1e-5, 1e-3, 1e-1, 1.0}) {
    for (unsigned seed = 1; seed <= 12; ++seed) {
      core::RealVector y = copy_of(exact.y);
      core::RealVector z = copy_of(exact.z);
      core::RealVector v = copy_of(exact.v);
      perturb(y, seed, scale);
      perturb(z, seed + 31u, scale);
      perturb(v, seed + 61u, scale);

      DualBound bound;
      CHECK(solver::compute_dual_bound(p, y, z, v, bound).ok());
      CHECK(bound.finite);
      CHECK(bound.bound <= optimum + 1e-9);
    }
  }
}

/// The bound tightens toward the optimum as the dual error shrinks, and the
/// penalty [SW] Proposition 2.1 predicts shrinks with it. A bound that is valid
/// but never tightens would be useless for pruning.
void test_the_bound_tightens_as_the_dual_improves() {
  model::CanonicalResult canon;
  if (!canonicalize(kBoxed, canon)) return;
  const model::CanonicalProblem& p = canon.problem;

  model::Solution exact;
  if (!exact_solution(p, exact)) return;
  const Real optimum = exact.objective;

  Real previous_gap = std::numeric_limits<Real>::infinity();
  for (Real scale : {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}) {
    core::RealVector y = copy_of(exact.y);
    core::RealVector z = copy_of(exact.z);
    core::RealVector v = copy_of(exact.v);
    perturb(y, 7u, scale);
    perturb(z, 7u, scale);
    perturb(v, 7u, scale);

    DualBound bound;
    CHECK(solver::compute_dual_bound(p, y, z, v, bound).ok());
    const Real gap = optimum - bound.bound;
    CHECK(gap >= -1e-9);
    CHECK(gap < previous_gap);
    previous_gap = gap;
  }
  CHECK(previous_gap < 1e-2);

  // An EXACT dual must certify the optimum itself, to rounding.
  DualBound tight;
  CHECK(solver::compute_dual_bound(p, exact.y, exact.z, exact.v, tight).ok());
  CHECK(tight.finite);
  CHECK(tight.bound <= optimum + 1e-9);
  CHECK_NEAR(tight.bound, optimum, 1e-7);
  CHECK(tight.residual_inf < 1e-7);
}

/// [SW] section 2's stated limitation, which is why they wrote a second
/// algorithm: a free column cannot be repaired, because repairing means putting
/// weight on a bound dual that does not exist.
void test_a_free_column_makes_the_bound_infinite() {
  model::CanonicalResult canon;
  if (!canonicalize(R"(Minimize
 obj: x + y
Subject To
 c1: x + y >= 2
Bounds
 0 <= x <= 5
 y free
End
)",
                    canon)) {
    return;
  }
  const model::CanonicalProblem& p = canon.problem;

  std::size_t free_columns = 0;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    if (!core::is_finite_bound(p.col_lower[j]) && !core::is_finite_bound(p.col_upper[j])) {
      ++free_columns;
    }
  }
  CHECK(free_columns > 0);

  // A dual with a residual on the free column: unrepairable.
  core::RealVector y(p.num_rows(), -0.5);
  core::RealVector z(p.num_cols(), 0.0);
  core::RealVector v(p.num_cols(), 0.0);

  DualBound bound;
  CHECK(solver::compute_dual_bound(p, y, z, v, bound).ok());

  // UNCONDITIONAL. An earlier version guarded these behind `if (!bound.finite)`,
  // which meant a defect that wrongly reported the bound as finite made the
  // test SKIP its own assertions and pass. Mutation testing found that: silently
  // contributing zero for a missing bound left all 301 checks green. The
  // residual on the free column here is deterministic (the row dual is fixed
  // above, so `r` is a fixed nonzero), so there is nothing to guard against.
  CHECK(!bound.finite);
  CHECK(bound.unbounded_columns > 0);
  // Minus infinity, so a caller that ignores `finite` and prunes anyway prunes
  // NOTHING rather than pruning the optimum. The failure mode is chosen.
  CHECK(bound.bound == -std::numeric_limits<Real>::infinity());
  CHECK(bound.residual_inf > 0.0);
}

/// Sign constraints are part of dual feasibility. An inequality row's dual must
/// be non-positive in our convention, and a first-order method will hand back
/// small positive values; clamping them has to happen BEFORE the residual is
/// taken, or the clamp itself goes uncorrected and the bound is not valid.
void test_sign_violations_are_projected_before_repair() {
  model::CanonicalResult canon;
  if (!canonicalize(kBoxed, canon)) return;
  const model::CanonicalProblem& p = canon.problem;

  model::Solution exact;
  if (!exact_solution(p, exact)) return;
  const Real optimum = exact.objective;

  core::RealVector y = copy_of(exact.y);
  core::RealVector z = copy_of(exact.z);
  core::RealVector v = copy_of(exact.v);
  // Push every inequality dual the WRONG way, and both bound duals negative.
  for (std::size_t i = p.num_equality; i < p.num_rows(); ++i) y[i] += 0.5;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    z[j] -= 0.25;
    v[j] -= 0.25;
  }

  DualBound bound;
  CHECK(solver::compute_dual_bound(p, y, z, v, bound).ok());
  CHECK(bound.finite);
  CHECK(bound.bound <= optimum + 1e-9);
}

/// The sign clamp on inequality-row duals is LOAD-BEARING, and this test exists
/// because a weaker one missed it.
///
/// Weak duality needs `y_I <= 0` in our convention: for a row `a'x <= b_i`,
/// multiplying by a NON-POSITIVE `y_i` gives `y_i (a'x) >= y_i b_i`, which is
/// the direction that makes `b'y` a lower bound. Flip the sign and the
/// inequality flips too, so `y_i b_i` OVERSTATES `y_i (a'x)` and the bound rises
/// above the optimum.
///
/// The overstatement is `y_i * slack_i`, so it is invisible on a row that is
/// TIGHT at the optimum -- which is why the earlier perturbation-based test
/// missed this: it happened to load the tight rows. This one deliberately finds
/// a row with slack and puts a large positive dual on exactly that row.
void test_a_positive_dual_on_a_slack_row_must_not_break_the_bound() {
  model::CanonicalResult canon;
  if (!canonicalize(kBoxed, canon)) return;
  const model::CanonicalProblem& p = canon.problem;

  model::Solution exact;
  if (!exact_solution(p, exact)) return;
  const Real optimum = exact.objective;

  const auto& csr = p.A.csr;
  std::size_t slack_rows = 0;
  for (std::size_t i = p.num_equality; i < p.num_rows(); ++i) {
    Real activity = 0.0;
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      activity += csr.values()[k] * exact.x[static_cast<std::size_t>(csr.indices()[k])];
    }
    const Real slack = p.b[i] - activity;
    if (slack <= 1e-7) continue;  // tight row: nothing to expose
    ++slack_rows;

    for (Real magnitude : {0.1, 1.0, 10.0, 100.0}) {
      core::RealVector y = copy_of(exact.y);
      core::RealVector z = copy_of(exact.z);
      core::RealVector v = copy_of(exact.v);
      y[i] = magnitude;  // the wrong sign, on the row where it shows

      DualBound bound;
      CHECK(solver::compute_dual_bound(p, y, z, v, bound).ok());
      if (bound.finite) CHECK(bound.bound <= optimum + 1e-9);
    }
  }

  // The fixture must actually contain a slack row, or this test proves nothing.
  CHECK(slack_rows > 0);

  // THE ADVERSARIAL CASE, designed rather than searched for.
  //
  // Above, the correction penalty happens to swamp the sign violation, so the
  // bound stays valid even unclamped -- which is luck, not soundness. The
  // violation is worth `y_i * slack_i` while the penalty scales with the
  // variable bound WIDTHS, so the way to expose it is a row with enormous slack
  // and variables pinned in a narrow box.
  //
  // `x <= 1000` is redundant against `x <= 1` and `0 <= x <= 1`. Put `y = +1`
  // on that redundant row: the residual is then repaired onto the upper-bound
  // dual, costing about 1, while `b'y` collects 1000. The bound lands near +999
  // against a true optimum of -1. Unclamped, this module would tell a
  // branch-and-bound to prune every node whose incumbent beat 999 -- which is
  // all of them.
  model::CanonicalResult redundant;
  if (!canonicalize(R"(Minimize
 obj: -x
Subject To
 tight: x <= 1
 slackrow: x <= 1000
Bounds
 0 <= x <= 1
End
)",
                    redundant)) {
    return;
  }
  const model::CanonicalProblem& rp = redundant.problem;
  model::Solution rexact;
  if (!exact_solution(rp, rexact)) return;
  CHECK_NEAR(rexact.objective, -1.0, 1e-9);
  CHECK(rp.num_inequality_rows() >= 2);

  core::RealVector ry(rp.num_rows(), 0.0);
  core::RealVector rz(rp.num_cols(), 0.0);
  core::RealVector rv(rp.num_cols(), 0.0);
  // The wrong sign, on the row carrying ~1000 of slack.
  ry[rp.num_rows() - 1] = 1.0;

  DualBound rbound;
  CHECK(solver::compute_dual_bound(rp, ry, rz, rv, rbound).ok());
  CHECK(rbound.finite);
  CHECK(rbound.bound <= rexact.objective + 1e-9);
}

/// Standard form -- all rows equalities, `l = 0`, `u = inf` -- is where [SW]
/// section 2 is stated. With no upper bounds the `u'v` term must vanish
/// entirely, and the bound reduces to `b'y` since `l = 0` kills `l'z` too.
void test_standard_form_reduces_to_the_published_expression() {
  model::CanonicalResult canon;
  if (!canonicalize(R"(Minimize
 obj: 2 x + 3 y + z
Subject To
 e1: x + y = 4
 e2: y + z = 3
End
)",
                    canon)) {
    return;
  }
  const model::CanonicalProblem& p = canon.problem;
  for (std::size_t j = 0; j < p.num_cols(); ++j) {
    CHECK_NEAR(p.col_lower[j], 0.0, 0.0);
    CHECK(!core::is_finite_bound(p.col_upper[j]));
  }
  CHECK_EQ(p.num_inequality_rows(), std::size_t{0});

  model::Solution exact;
  if (!exact_solution(p, exact)) return;

  core::RealVector y = copy_of(exact.y);
  perturb(y, 3u, 1e-3);
  core::RealVector z(p.num_cols(), 0.0);
  core::RealVector v(p.num_cols(), 0.0);

  DualBound bound;
  CHECK(solver::compute_dual_bound(p, y, z, v, bound).ok());

  // With `l = 0` the shift contributes nothing to the objective, so the bound
  // is exactly `b'y` -- and `r-` must have landed on an absent upper bound
  // wherever the residual was negative, which is the standard form's known
  // weakness rather than a defect here.
  Real by = 0.0;
  for (std::size_t i = 0; i < p.num_rows(); ++i) by += p.b[i] * y[i];
  if (bound.finite) {
    CHECK_NEAR(bound.bound, by, 1e-12);
    CHECK_NEAR(bound.correction_penalty, 0.0, 1e-12);
    CHECK(bound.bound <= exact.objective + 1e-9);
  } else {
    CHECK(bound.unbounded_columns > 0);
  }
}

void test_dimension_mismatch_is_refused() {
  model::CanonicalResult canon;
  if (!canonicalize(kBoxed, canon)) return;
  const model::CanonicalProblem& p = canon.problem;
  core::RealVector y(p.num_rows() + 1, 0.0);
  core::RealVector z(p.num_cols(), 0.0);
  core::RealVector v(p.num_cols(), 0.0);
  DualBound bound;
  CHECK(!solver::compute_dual_bound(p, y, z, v, bound).ok());
}

}  // namespace

int main() {
  test_the_uncorrected_bound_is_actually_invalid();
  test_validity_across_error_magnitudes();
  test_the_bound_tightens_as_the_dual_improves();
  test_a_free_column_makes_the_bound_infinite();
  test_sign_violations_are_projected_before_repair();
  test_a_positive_dual_on_a_slack_row_must_not_break_the_bound();
  test_standard_form_reduces_to_the_published_expression();
  test_dimension_mismatch_is_refused();
  return ::sovsolve::test::report("dual_bound_test");
}
