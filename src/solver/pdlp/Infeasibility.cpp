#include "sovsolve/solver/pdlp/Infeasibility.hpp"

#include <cmath>

namespace sovsolve::solver::pdlp {

namespace {

using core::is_finite_bound;

[[nodiscard]] Real inf_norm(const core::RealVector& v) {
  Real worst = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) worst = std::fmax(worst, std::fabs(v[i]));
  return worst;
}

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

bool InfeasibilityDetector::is_dual_ray(const core::RealVector& v_y, Real tolerance) {
  // (a) recession cone of `Y`: free on equality rows, `<= 0` on inequality
  // rows. Checked before the matrix product, because it is O(m) and the
  // product is not.
  for (std::size_t i = problem_->num_equality; i < m_; ++i) {
    if (v_y[i] > tolerance) return false;
  }

  matvec_->multiply_transpose(in(v_y), out(kt_v_));

  // `dlambda = -K' v_y`, and the objective rate accumulates as we go.
  Real rate = 0.0;
  for (std::size_t i = 0; i < m_; ++i) rate += problem_->b[i] * v_y[i];

  Real scale = std::fabs(rate);
  for (std::size_t j = 0; j < n_; ++j) {
    const Real dlambda = -kt_v_[j];
    const bool has_lower = is_finite_bound(problem_->col_lower[j]);
    const bool has_upper = is_finite_bound(problem_->col_upper[j]);

    // (b) `dlambda` must lie in `Lambda_j`, which depends only on which
    // bounds are finite.
    if (!has_lower && !has_upper) {
      // Free column: `Lambda_j = {0}`.
      if (std::fabs(dlambda) > tolerance) return false;
      continue;
    }
    if (!has_lower && dlambda > tolerance) return false;   // Lambda_j = R-
    if (!has_upper && dlambda < -tolerance) return false;  // Lambda_j = R+

    // (c) the rate's bound terms, split on the DIRECTION's sign. Condition
    // (b) has already ruled out the combinations that would multiply an
    // infinite bound by a nonzero coefficient.
    if (dlambda > 0.0 && has_lower) {
      const Real term = problem_->col_lower[j] * dlambda;
      rate += term;
      scale = std::fmax(scale, std::fabs(term));
    } else if (dlambda < 0.0 && has_upper) {
      const Real term = problem_->col_upper[j] * dlambda;
      rate += term;
      scale = std::fmax(scale, std::fabs(term));
    }
  }

  // (c) strictly positive, by a margin relative to the magnitudes that went
  // into it -- a rate of 1e-14 built from terms of size 1e6 is rounding.
  return rate > tolerance * (1.0 + scale);
}

bool InfeasibilityDetector::is_primal_ray(const core::RealVector& v_x, Real tolerance) {
  // (a) recession cone of the box, and (c)'s objective rate, in one sweep.
  Real rate = 0.0;
  Real scale = 0.0;
  for (std::size_t j = 0; j < n_; ++j) {
    const bool has_lower = is_finite_bound(problem_->col_lower[j]);
    const bool has_upper = is_finite_bound(problem_->col_upper[j]);
    const Real d = v_x[j];
    if (has_lower && has_upper) {
      // A boxed column cannot move forever in either direction.
      if (std::fabs(d) > tolerance) return false;
    } else if (has_lower) {
      if (d < -tolerance) return false;  // may only increase
    } else if (has_upper) {
      if (d > tolerance) return false;  // may only decrease
    }
    const Real term = problem_->c[j] * d;
    rate += term;
    scale = std::fmax(scale, std::fabs(term));
  }

  matvec_->multiply(in(v_x), out(k_v_));

  // (b) equality rows must not move at all; `<=` rows must not increase.
  for (std::size_t i = 0; i < m_; ++i) {
    if (i < problem_->num_equality) {
      if (std::fabs(k_v_[i]) > tolerance) return false;
    } else if (k_v_[i] > tolerance) {
      return false;
    }
  }

  // (c) strictly decreasing, by the same relative margin.
  return rate < -tolerance * (1.0 + scale);
}

CertificateKind InfeasibilityDetector::classify(const core::RealVector& v_x,
                                                const core::RealVector& v_y,
                                                Real tolerance) {
  // Normalize the whole candidate by one factor, not each block separately:
  // `(v_x, v_y)` is a single direction and rescaling its halves independently
  // would change which certificate it is.
  const Real magnitude = std::fmax(inf_norm(v_x), inf_norm(v_y));
  if (!(magnitude > 0.0) || !std::isfinite(magnitude)) return CertificateKind::None;

  for (std::size_t j = 0; j < n_; ++j) scaled_x_[j] = v_x[j] / magnitude;
  for (std::size_t i = 0; i < m_; ++i) scaled_y_[i] = v_y[i] / magnitude;

  // Proposition 4 makes these mutually exclusive on a well-posed model -- a
  // problem cannot be both primal infeasible and primal unbounded -- but they
  // are tested independently rather than assumed so, and the primal-
  // infeasibility test goes first because it is the verdict a caller is more
  // often asking about.
  if (is_dual_ray(scaled_y_, tolerance)) return CertificateKind::PrimalInfeasible;
  if (is_primal_ray(scaled_x_, tolerance)) return CertificateKind::DualInfeasible;
  return CertificateKind::None;
}

}  // namespace sovsolve::solver::pdlp
