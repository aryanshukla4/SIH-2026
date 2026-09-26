#include "sovsolve/solver/pdlp/Infeasibility.hpp"

#include <cmath>

namespace sovsolve::solver::pdlp {

namespace {

using core::is_finite_bound;

core::HostSpan<const Real> in(const core::RealVector& v) {
  return core::HostSpan<const Real>(v.data(), v.size());
}

core::HostSpan<Real> out(core::RealVector& v) {
  return core::HostSpan<Real>(v.data(), v.size());
}

}  // namespace

InfeasibilityDetector::InfeasibilityDetector(const model::CanonicalProblem& problem,
                                             MatVec& matvec)
    : problem_(&problem),
      matvec_(&matvec),
      m_(problem.num_rows()),
      n_(problem.num_cols()) {
  scaled_x_.resize(n_);
  scaled_x_.assign(0.0);
  scaled_y_.resize(m_);
  scaled_y_.assign(0.0);
  kt_v_.resize(n_);
  kt_v_.assign(0.0);
  k_v_.resize(m_);
  k_v_.assign(0.0);
}

// ---------------------------------------------------------------------------
// The acceptance test is arXiv 2102.04592 section 6, equations (50) and (51):
// an EPSILON-APPROXIMATE certificate. Both measure the candidate's constraint
// violation PER UNIT OF OBJECTIVE IMPROVEMENT -- the ray is effectively scaled
// so that its objective rate is exactly 1, and only then compared with epsilon.
// The paper notes this parallels the criterion SCS uses.
//
// This replaced a test that normalized the candidate by its own SIZE
// (`max(||v_x||, ||v_y||)`) and then compared each constraint against an
// absolute epsilon. That accepts almost-flat rays: a direction whose objective
// improvement is 1e-5 of its length and whose violations are 1e-7 passes it,
// where (50) sees a violation of 1e-2 per unit of improvement and rejects it.
// A nearly converged run on a FEASIBLE model produces exactly such vectors --
// tiny, directionless differences of iterates -- and on the Linux build that
// was enough to declare israel (optimum -896644.8) infeasible at iteration
// 7600, with the iterate at -896636. The paper had the right definition all
// along; this file had not used it.
//
// Both tests are homogeneous of degree zero in the candidate, so no
// normalization is needed and none is done: `z^{k+1} - z^k`, `z^k / k` and the
// normalized average can be passed as they are.
//
// Norm: Euclidean, as in the reference (its section 1.4).
//
// WHAT THIS TEST CANNOT DO, and a rejected fix, because the reasoning looks
// right and is wrong. (50)/(51) divide a violation measured in VARIABLE units
// by an improvement measured in OBJECTIVE units, so epsilon is dimensionless
// only while `c` (respectively `b`) is O(1). Scale the objective up and
// `epsilon * improvement` becomes a large absolute allowance. dfl001 (feasible,
// optimum 1.1266396047e+07) was reported Unbounded at iteration 480 on exactly
// that: a candidate with `||v_x|| = 5.63` carrying a cone violation of 3.25 --
// 58% of its own length pointing OUT of the box's recession cone -- passed at
// `tol = 1e-6` because `-c'x` was 3.67e6, making the allowance 3.67.
//
// The obvious repair is to bound each violation a SECOND time against the norm
// of the vector it was carved out of: the cone violation against `||v_x||`, the
// row violation against `||A v_x||`. Dimensionally exact, same units on both
// sides. It was implemented, and it is wrong, and gas11 is why. gas11 IS
// unbounded, and the candidate that certifies it measures
//
//     cone/||v_x|| = 0.36        row/||A v_x|| = 1.000
//
// -- a full 100% of its row activity is violation, because the candidate is
// `z^{k+1} - z^k`, the direction of TRAVEL, which on an unbounded model is
// dominated by the ray but still carries the whole transient. A real ray is not
// close to its own recession cone in a relative sense at PDLP accuracy, so that
// test rejects the true certificates along with the false one. It was removed.
//
// What actually separates them is the MARGIN inside (51), and the separation is
// nearly four orders of magnitude wide:
//
//     dfl001   cone/improvement = 8.8e-7    bogus
//     gas11    cone/improvement = 9.2e-11   real
//
// So the discrimination lives in `certificate_tolerance`, whose default moved
// from 1e-6 to 1e-8 for this reason -- see the measurement table at that option
// in Options.hpp. This file's tests are the reference's, unmodified.
// ---------------------------------------------------------------------------

namespace {

/// A rate within this many units of its own terms' magnitude is cancellation,
/// not signal. Not part of (50)/(51): it guards the one case where their ratio
/// cannot -- a candidate whose violation is EXACTLY zero (every column boxed
/// and every row an equality, say) passes (50) for any positive objective, and
/// a positive objective of 1e-18 built from terms of size 1 is rounding.
constexpr Real kSignificance = 1e-12;

/// `violation <= tolerance * scale`, named so the direction of every one of
/// these comparisons reads the same way. `scale` is always the objective rate,
/// which `kSignificance` has already established is strictly positive.
[[nodiscard]] bool within(Real violation, Real tolerance, Real scale) noexcept {
  return violation <= tolerance * scale;
}

}  // namespace

bool InfeasibilityDetector::is_dual_ray(const core::RealVector& v_y, Real tolerance) {
  // (50) requires `y` in its cone -- `y >= 0` in the reference's form (47),
  // which is `y_I <= 0` on our `<=` rows once the row is negated, and free on
  // equality rows. The difference of iterates need not satisfy that exactly,
  // so the candidate is PROJECTED onto the cone and the distance moved is
  // charged to the residual. Rejecting outright would discard the fastest of
  // the three sequences over rounding; ignoring the distance would not be (50).
  Real sign_violation_sq = 0.0;
  for (std::size_t i = 0; i < m_; ++i) {
    Real y = v_y[i];
    if (i >= problem_->num_equality && y > 0.0) {
      sign_violation_sq += y * y;
      y = 0.0;
    }
    scaled_y_[i] = y;
  }

  matvec_->multiply_transpose(in(scaled_y_), out(kt_v_));

  // The objective rate `b'y + l'r+ - u'r-` in the reference's notation, with
  // `r` the closest point to `-K'y` that keeps every bound term finite -- the
  // same choice the reference makes, `r = proj_Cv(...)`. Whatever of `-K'y`
  // that projection had to remove is `r + A'y`, the residual (50) bounds.
  Real rate = 0.0;
  Real magnitude = 0.0;
  for (std::size_t i = 0; i < m_; ++i) {
    const Real term = problem_->b[i] * scaled_y_[i];
    rate += term;
    magnitude += std::fabs(term);
  }

  Real cone_violation_sq = 0.0;
  for (std::size_t j = 0; j < n_; ++j) {
    const Real dlambda = -kt_v_[j];
    const bool has_lower = is_finite_bound(problem_->col_lower[j]);
    const bool has_upper = is_finite_bound(problem_->col_upper[j]);

    // Projection onto Lambda_j, which depends only on which bounds are finite:
    // {0} for a free column, R+ with only a lower bound, R- with only an upper
    // bound, R with both.
    Real r = dlambda;
    if (!has_lower && !has_upper) {
      r = 0.0;
    } else if (!has_upper) {
      r = std::fmax(dlambda, 0.0);
    } else if (!has_lower) {
      r = std::fmin(dlambda, 0.0);
    }
    const Real removed = dlambda - r;
    cone_violation_sq += removed * removed;

    Real term = 0.0;
    if (r > 0.0) {
      term = problem_->col_lower[j] * r;
    } else if (r < 0.0) {
      term = problem_->col_upper[j] * r;
    }
    rate += term;
    magnitude += std::fabs(term);
  }

  // (50): a positive objective, and violation at most epsilon PER UNIT of it.
  if (!(rate > kSignificance * magnitude) || !std::isfinite(rate)) return false;
  const Real residual = std::sqrt(cone_violation_sq + sign_violation_sq);
  return within(residual, tolerance, rate);
}

bool InfeasibilityDetector::is_primal_ray(const core::RealVector& v_x, Real tolerance) {
  // (51), three conditions, each scaled by the improvement `-c'x`:
  //   c'x < 0
  //   ||x - proj_Cv(x)||  / (-c'x) <= epsilon   (x in the box's recession cone)
  //   ||Ax - proj(Ax)||   / (-c'x) <= epsilon   (rows never violated along it)
  // Kept as TWO separate tests, as the reference states them, rather than
  // pooled into one residual.
  Real rate = 0.0;
  Real magnitude = 0.0;
  Real cone_violation_sq = 0.0;
  for (std::size_t j = 0; j < n_; ++j) {
    const bool has_lower = is_finite_bound(problem_->col_lower[j]);
    const bool has_upper = is_finite_bound(problem_->col_upper[j]);
    const Real d = v_x[j];

    // Recession cone of [l_j, u_j]: {0} if both finite, R+ with only a lower
    // bound, R- with only an upper bound, R if free.
    Real projected = d;
    if (has_lower && has_upper) {
      projected = 0.0;
    } else if (has_lower) {
      projected = std::fmax(d, 0.0);
    } else if (has_upper) {
      projected = std::fmin(d, 0.0);
    }
    const Real removed = d - projected;
    cone_violation_sq += removed * removed;

    const Real term = problem_->c[j] * d;
    rate += term;
    magnitude += std::fabs(term);
  }

  const Real improvement = -rate;
  if (!(improvement > kSignificance * magnitude) || !std::isfinite(improvement)) {
    return false;
  }
  const Real cone_violation = std::sqrt(cone_violation_sq);
  if (!within(cone_violation, tolerance, improvement)) return false;

  matvec_->multiply(in(v_x), out(k_v_));

  // Rows: equality rows may not move at all; `<=` rows may not increase.
  Real row_violation_sq = 0.0;
  for (std::size_t i = 0; i < m_; ++i) {
    const Real a = k_v_[i];
    const Real violation = i < problem_->num_equality ? a : std::fmax(a, 0.0);
    row_violation_sq += violation * violation;
  }
  return within(std::sqrt(row_violation_sq), tolerance, improvement);
}

CertificateKind InfeasibilityDetector::classify(const core::RealVector& v_x,
                                                const core::RealVector& v_y,
                                                Real tolerance) {
  // Proposition 4 makes these mutually exclusive on a well-posed model -- a
  // problem cannot be both primal infeasible and primal unbounded -- but they
  // are tested independently rather than assumed so, and the primal-
  // infeasibility test goes first because it is the verdict a caller is more
  // often asking about.
  if (is_dual_ray(v_y, tolerance)) return CertificateKind::PrimalInfeasible;
  if (is_primal_ray(v_x, tolerance)) return CertificateKind::DualInfeasible;
  return CertificateKind::None;
}

}  // namespace sovsolve::solver::pdlp
