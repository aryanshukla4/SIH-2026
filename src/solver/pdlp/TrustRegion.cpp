#include "sovsolve/solver/pdlp/TrustRegion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sovsolve::solver::pdlp {

namespace {

using core::ErrorCode;

[[nodiscard]] bool is_finite(Real v) noexcept { return std::isfinite(v); }

/// `zhat(lambda)` for one coordinate, in REFLECTED space (slope > 0, so the
/// coordinate walks down from `z` and stops at `l`). `reach` is `l - z <= 0`.
[[nodiscard]] Real displacement(Real lambda, Real slope, Real reach) noexcept {
  const Real free_move = -lambda * slope;  // <= 0
  return free_move > reach ? free_move : reach;
}

}  // namespace

Expected<Real> TrustRegionSolver::solve(const TrustRegionProblem& problem,
                                        core::HostSpan<Real> solution) {
  const std::size_t n = problem.center.size();
  if (problem.lower.size() != n || problem.upper.size() != n ||
      problem.gradient.size() != n || solution.size() != n) {
    return core::make_error(ErrorCode::DimensionMismatch,
                            "TrustRegionSolver::solve: center, lower, upper, gradient "
                            "and solution must all have the same length");
  }
  if (!(problem.radius >= 0.0) || !is_finite(problem.radius)) {
    return core::make_error(ErrorCode::NumericalError,
                            "TrustRegionSolver::solve: radius must be finite and "
                            "non-negative");
  }

  // The center is always feasible and is the answer when nothing can move.
  for (std::size_t i = 0; i < n; ++i) solution[i] = problem.center[i];
  if (n == 0) return 0.0;

  // ---- reduce to `slope > 0` with a single binding LOWER bound -------------
  //
  // See TrustRegion.hpp deviation 1: a negative gradient entry is REFLECTED
  // (so its binding bound becomes `-upper`), never given an infinite bound.
  sign_.assign(n, 1.0);
  slope_.assign(n, 0.0);
  reach_.assign(n, 0.0);
  breakpoint_.assign(n, 0.0);
  active_.clear();

  Real fhi = 0.0;  ///< sum of slope^2 over coordinates that never stop
  Real flo = 0.0;  ///< sum of reach^2 over coordinates already at their bound

  for (std::size_t i = 0; i < n; ++i) {
    const Real g = problem.gradient[i];
    if (g == 0.0) continue;  // deviation 2: this coordinate never moves

    const Real s = g > 0.0 ? 1.0 : -1.0;
    const Real bound = g > 0.0 ? problem.lower[i] : problem.upper[i];
    sign_[i] = s;
    slope_[i] = std::fabs(g);

    if (!is_finite(bound)) {
      // Nothing stops this coordinate; it contributes `lambda^2 * slope^2`
      // at every lambda, so it is never in the active set.
      breakpoint_[i] = std::numeric_limits<Real>::infinity();
      reach_[i] = -std::numeric_limits<Real>::infinity();
      fhi += slope_[i] * slope_[i];
      continue;
    }

    // In reflected coordinates the center is `s*z` and the bound is `s*bound`,
    // so `reach = s*bound - s*z = s*(bound - z)`, which is <= 0 because the
    // center is feasible.
    const Real reach = s * (bound - problem.center[i]);
    reach_[i] = reach > 0.0 ? 0.0 : reach;  // clamp: feasible center, guard FP
    const Real lambda_i = -reach_[i] / slope_[i];
    breakpoint_[i] = lambda_i;

    if (lambda_i <= 0.0) {
      // Already sitting on its bound: constant contribution forever.
      flo += reach_[i] * reach_[i];
    } else {
      active_.push_back(i);
    }
  }

  const Real r2 = problem.radius * problem.radius;

  // Appendix F line 1: if collapsing EVERY coordinate onto its bound still
  // fits inside the ball, that point is optimal and lambda is unbounded.
  // `flo` is that squared distance once the active set is also collapsed.
  {
    Real full = flo;
    bool reachable = fhi == 0.0;  // an unstoppable coordinate makes this infinite
    if (reachable) {
      for (const std::size_t i : active_) full += reach_[i] * reach_[i];
      if (full <= r2) {
        Real objective = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          if (problem.gradient[i] != 0.0) {
            solution[i] = problem.center[i] + sign_[i] * reach_[i];
          }
          objective += problem.gradient[i] * solution[i];
        }
        return objective;
      }
    }
  }

  // ---- Appendix F lines 2-17: median bisection on the breakpoints ----------
  //
  // Invariants, maintained exactly as stated in the paper:
  //   active_ == { i : lambda_lo < breakpoint_[i] < lambda_hi }
  //   for lambda in that open interval,
  //     ||zhat(lambda) - z||^2 = flo + lambda^2 * fhi
  //                              + sum_{i in active_} displacement_i(lambda)^2
  //   ||zhat(lambda_lo) - z||^2 <= r2 <= ||zhat(lambda_hi) - z||^2
  while (!active_.empty()) {
    const std::size_t mid = active_.size() / 2;
    std::nth_element(active_.begin(), active_.begin() + static_cast<std::ptrdiff_t>(mid),
                     active_.end(), [this](std::size_t a, std::size_t b) {
                       return breakpoint_[a] < breakpoint_[b];
                     });
    const Real lambda_mid = breakpoint_[active_[mid]];

    Real fmid = flo + fhi * lambda_mid * lambda_mid;
    for (const std::size_t i : active_) {
      const Real d = displacement(lambda_mid, slope_[i], reach_[i]);
      fmid += d * d;
    }

    if (fmid < r2) {
      // The optimal lambda is past lambda_mid: everything that has already
      // stopped by now folds into the constant term.
      for (const std::size_t i : active_) {
        if (breakpoint_[i] <= lambda_mid) flo += reach_[i] * reach_[i];
      }
      active_.erase(std::remove_if(active_.begin(), active_.end(),
                                   [this, lambda_mid](std::size_t i) {
                                     return breakpoint_[i] <= lambda_mid;
                                   }),
                    active_.end());
    } else {
      // The optimal lambda is before lambda_mid: everything still moving at
      // lambda_mid is still moving at the optimum, so it joins the quadratic
      // term.
      for (const std::size_t i : active_) {
        if (breakpoint_[i] >= lambda_mid) fhi += slope_[i] * slope_[i];
      }
      active_.erase(std::remove_if(active_.begin(), active_.end(),
                                   [this, lambda_mid](std::size_t i) {
                                     return breakpoint_[i] >= lambda_mid;
                                   }),
                    active_.end());
    }
  }

  // ---- Appendix F line 18: solve r^2 = flo + lambda^2 * fhi ----------------
  //
  // `fhi > 0` here: fhi == 0 would mean every coordinate is clamped, i.e. the
  // fully-collapsed point, which the early return above already handled. The
  // guard is for floating-point near-equality at that boundary, not for a
  // case the algebra allows.
  Real lambda = 0.0;
  if (fhi > 0.0) {
    const Real numerator = r2 - flo;
    lambda = numerator > 0.0 ? std::sqrt(numerator / fhi) : 0.0;
  }

  Real objective = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (problem.gradient[i] != 0.0) {
      solution[i] =
          problem.center[i] + sign_[i] * displacement(lambda, slope_[i], reach_[i]);
    }
    objective += problem.gradient[i] * solution[i];
  }
  return objective;
}

}  // namespace sovsolve::solver::pdlp
