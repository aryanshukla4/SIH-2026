// Module 24: the normalized duality gap, PDLP's restart criterion.
//
// Derived from Applegate et al., arXiv 2105.12715 (section 3, and section 6.3
// for the linear-time evaluation), as used by PDLP section 3.2.
//
// --------------------------------------------------------------------------
// What it is, and why the ordinary duality gap will not do
// --------------------------------------------------------------------------
//
// The plain duality gap `max_{zhat in Z} {L(x, yhat) - L(xhat, y)}` is the
// natural progress measure for a saddle point problem, and it is useless for
// an LP: `Z` is unbounded, so the supremum is `+infinity` at every point that
// is not already optimal. Restricting the maximization to a BALL of radius `r`
// around the current iterate and dividing by `r` fixes that:
//
//     rho_r(z) = (1/r) * max{ L(x, yhat) - L(xhat, y) : zhat in Z,
//                                                      ||zhat - z||_omega <= r }
//
// This is finite everywhere, and it is zero exactly at an optimal `z`
// (2105.12715 section 3), so it is a valid measure of "how far along are we"
// that can be compared across restarts. PDLP evaluates it at radius
// `r = ||z - z_ref||_omega`, the distance travelled since the last restart.
//
// --------------------------------------------------------------------------
// For an LP it is a trust region problem with a LINEAR objective
// --------------------------------------------------------------------------
//
// Expanding `L(x, y) = c'x - y'Kx + q'y` and collecting terms:
//
//     L(x, yhat) - L(xhat, y)
//         = c'x - yhat'Kx + q'yhat - c'xhat + y'Kxhat - q'y
//         = (c'x - q'y)  +  yhat'(q - Kx)  -  xhat'(c - K'y)
//
// so maximizing it over the ball is MINIMIZING `g' zhat` with
//
//     g = ( c - K'y ,  Kx - q )        stacked as (x-block, y-block)
//
// which is exactly 2105.12715 equation (54), and is solvable exactly in
// `O(n + m)` by pdlp/TrustRegion.hpp. Hence
//
//     rho_r(z) = [ (c'x - q'y) - min_{zhat} g'zhat ] / r
//
// Two facts worth keeping, because they are what the tests assert:
//
//   * `g'z = c'x - q'y` identically -- the cross terms `x'K'y` and `y'Kx`
//     cancel. Since `z` itself is in the ball, the minimum is at most `g'z`,
//     so **`rho_r(z) >= 0` always**. A negative value is a bug, not a hard
//     instance.
//   * At `zhat = z` the bracketed expression is exactly zero, which is the
//     same statement.
//
// --------------------------------------------------------------------------
// The omega-norm, and why the trust region solver does not know about it
// --------------------------------------------------------------------------
//
// The ball is measured in the primal-weighted norm
// `||z||_omega = sqrt(omega||x||^2 + ||y||^2/omega)`, while TrustRegion.hpp
// solves in the plain Euclidean norm. The two are related by a diagonal change
// of variables, not by a different algorithm:
//
//     x' = sqrt(omega) * x,    y' = y / sqrt(omega)
//       =>  ||(x', y')||_2 = ||(x, y)||_omega
//
// so the problem is built in primed coordinates -- centre, bounds and gradient
// each scaled accordingly -- solved there, and the resulting objective value
// used directly, since `g'zhat` is invariant under the substitution. The
// bounds keep their meaning because `sqrt(omega) > 0` preserves order, and an
// upper bound of exactly `0` on an inequality row's dual stays `0`.

#ifndef SOVSOLVE_SOLVER_PDLP_DUALITY_GAP_HPP
#define SOVSOLVE_SOLVER_PDLP_DUALITY_GAP_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/pdlp/TrustRegion.hpp"

namespace sovsolve::solver::pdlp {

using core::Real;

/// Evaluates `rho_r` for one problem, reusing its workspace across calls --
/// PDLP asks for three of these per restart check, on `(n + m)`-length data,
/// for the whole solve.
class NormalizedDualityGap {
 public:
  explicit NormalizedDualityGap(const model::CanonicalProblem& problem);

  /// `rho_r(z)` at `z = (x, y)` with `r = ||z - z_ref||_omega`.
  ///
  /// `kt_y` and `k_x` must already hold `K'y` and `Kx` AT `z` -- they are
  /// passed in rather than computed because the caller has usually just
  /// computed them, and they are the expensive part.
  ///
  /// Returns `0` when `r` is zero: the iterate has not moved since the
  /// reference point, so there is no progress to normalize and the ratio is
  /// `0/0`. Reporting zero makes the restart test read "no progress", which
  /// is what an unmoved iterate means.
  [[nodiscard]] core::Expected<Real> evaluate(const core::RealVector& x,
                                              const core::RealVector& y,
                                              const core::RealVector& kt_y,
                                              const core::RealVector& k_x,
                                              const core::RealVector& x_ref,
                                              const core::RealVector& y_ref,
                                              Real omega);

 private:
  const model::CanonicalProblem* problem_;
  std::size_t m_;
  std::size_t n_;

  TrustRegionSolver solver_;
  core::RealVector center_;
  core::RealVector lower_;
  core::RealVector upper_;
  core::RealVector gradient_;
  core::RealVector solution_;
};

}  // namespace sovsolve::solver::pdlp

#endif  // SOVSOLVE_SOLVER_PDLP_DUALITY_GAP_HPP
