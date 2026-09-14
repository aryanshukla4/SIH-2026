#include "sovsolve/solver/pdlp/DualityGap.hpp"

#include <cmath>
#include <limits>

namespace sovsolve::solver::pdlp {

namespace {

constexpr Real kInfinity = std::numeric_limits<Real>::infinity();

}  // namespace

NormalizedDualityGap::NormalizedDualityGap(const model::CanonicalProblem& problem)
    : problem_(&problem), m_(problem.num_rows()), n_(problem.num_cols()) {
  const std::size_t total = n_ + m_;
  center_.resize(total);
  center_.assign(0.0);
  lower_.resize(total);
  lower_.assign(0.0);
  upper_.resize(total);
  upper_.assign(0.0);
  gradient_.resize(total);
  gradient_.assign(0.0);
  solution_.resize(total);
  solution_.assign(0.0);
}

core::Expected<Real> NormalizedDualityGap::evaluate(
    const core::RealVector& x, const core::RealVector& y,
    const core::RealVector& kt_y, const core::RealVector& k_x,
    const core::RealVector& x_ref, const core::RealVector& y_ref, Real omega) {
  if (!(omega > 0.0) || !std::isfinite(omega)) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "NormalizedDualityGap: primal weight must be finite "
                            "and positive");
  }

  const Real root = std::sqrt(omega);

  // `r = ||z - z_ref||_omega`, which is the Euclidean distance once the
  // change of variables below is applied.
  Real radius_sq = 0.0;
  for (std::size_t j = 0; j < n_; ++j) {
    const Real d = x[j] - x_ref[j];
    radius_sq += omega * d * d;
  }
  for (std::size_t i = 0; i < m_; ++i) {
    const Real d = y[i] - y_ref[i];
    radius_sq += d * d / omega;
  }
  const Real radius = std::sqrt(radius_sq);
  if (!(radius > 0.0)) return 0.0;  // unmoved since the reference point

  // ---- x-block, in primed coordinates x' = sqrt(omega) x ------------------
  for (std::size_t j = 0; j < n_; ++j) {
    center_[j] = x[j] * root;
    const Real lo = problem_->col_lower[j];
    const Real hi = problem_->col_upper[j];
    lower_[j] = core::is_finite_bound(lo) ? lo * root : -kInfinity;
    upper_[j] = core::is_finite_bound(hi) ? hi * root : kInfinity;
    // `g_x = c - K'y`; dividing by `root` is the substitution's effect on a
    // linear coefficient (the variable was multiplied by it).
    gradient_[j] = (problem_->c[j] - kt_y[j]) / root;
  }

  // ---- y-block, in primed coordinates y' = y / sqrt(omega) ----------------
  for (std::size_t i = 0; i < m_; ++i) {
    const std::size_t slot = n_ + i;
    center_[slot] = y[i] / root;
    // proj_Y: equality rows free, inequality rows `y_i <= 0`. Scaling by a
    // positive factor leaves an upper bound of exactly zero at zero.
    lower_[slot] = -kInfinity;
    upper_[slot] = i < problem_->num_equality ? kInfinity : 0.0;
    // `g_y = Kx - q`, multiplied by `root` for the same reason.
    gradient_[slot] = (k_x[i] - problem_->b[i]) * root;
  }

  TrustRegionProblem tr;
  tr.center = core::HostSpan<const Real>(center_.data(), center_.size());
  tr.lower = core::HostSpan<const Real>(lower_.data(), lower_.size());
  tr.upper = core::HostSpan<const Real>(upper_.data(), upper_.size());
  tr.gradient = core::HostSpan<const Real>(gradient_.data(), gradient_.size());
  tr.radius = radius;

  auto minimum =
      solver_.solve(tr, core::HostSpan<Real>(solution_.data(), solution_.size()));
  if (!minimum.has_value()) return minimum.error();

  // `g'z = c'x - q'y` identically (the cross terms cancel), so this is the
  // same quantity the derivation in the header calls the constant, computed
  // the cheap way rather than as two more dot products.
  Real g_dot_z = 0.0;
  for (std::size_t j = 0; j < n_; ++j) g_dot_z += gradient_[j] * center_[j];
  for (std::size_t i = 0; i < m_; ++i) {
    g_dot_z += gradient_[n_ + i] * center_[n_ + i];
  }

  // `z` is inside its own ball, so the minimum is at most `g'z` and the
  // difference is non-negative BY CONSTRUCTION -- see DualityGap.hpp.
  //
  // Which is exactly why it is reported as an error rather than clamped away.
  // A silent `max(gap, 0)` would turn a sign error in the gradient, or a trust
  // region solve that returned a point worse than its own centre, into a
  // plausible-looking zero -- and a zero gap reads as "converged" to the
  // restart logic. Only rounding-level negatives are absorbed, scaled to the
  // magnitudes actually involved.
  const Real difference = g_dot_z - *minimum;
  const Real scale = std::fabs(g_dot_z) + std::fabs(*minimum);
  if (difference < -1e-9 * (1.0 + scale)) {
    return core::make_error(
        core::ErrorCode::NumericalError,
        "NormalizedDualityGap: rho_r came out negative, which the derivation "
        "makes impossible -- the trust region minimum is worse than the centre "
        "that lies inside its own ball");
  }
  const Real gap = difference > 0.0 ? difference / radius : 0.0;
  return gap;
}

}  // namespace sovsolve::solver::pdlp
