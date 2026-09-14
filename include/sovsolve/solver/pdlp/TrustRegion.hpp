// Module 24: exact linear-time solver for a trust region problem with a LINEAR
// objective, Euclidean ball, and two-sided variable bounds.
//
//     minimize    g' zhat
//     subject to  l <= zhat <= u
//                 ||zhat - z||_2 <= r
//
// This is the load-bearing subroutine of PDLP's adaptive restart scheme. The
// restart criteria are stated in terms of the NORMALIZED DUALITY GAP
//
//     rho_r(z) = (1/r) * max{ L(x, yhat) - L(xhat, y) : ||zhat - z|| <= r }
//
// and for a linear program that maximization is exactly the problem above (its
// objective is linear in `zhat` once `K'y` and `Kx` are known). So one
// `rho_r` evaluation costs one SpMV pair plus one call to this routine --
// which is why PDLP can afford to check restarts at all, and why it checks
// them every 40 iterations rather than every iteration.
//
// Derived from the published definition: Applegate, Diaz, Hinder, Lu, Lubin,
// O'Donoghue, Schudy, "Faster First-Order Primal-Dual Methods for Linear
// Programming using Restarts and Sharpness" (arXiv 2105.12715), Section 6.3.1
// and Appendix F. Nothing here is read from or ported out of another solver --
// see docs/HIGHS-COMPARISON.md for why that constraint exists.
//
// --------------------------------------------------------------------------
// Why this is O(n) and not a generic QP
// --------------------------------------------------------------------------
//
// The optimal point has a CLOSED FORM in one scalar. With `g > 0` (arranged
// below) the objective pushes every coordinate down until it either exhausts
// the ball or hits its bound, so
//
//     zhat(lambda) = max(z - lambda*g, l)                              (50)
//
// for some lambda in [0, inf), and the whole problem collapses to
//
//     maximize lambda  subject to  ||zhat(lambda) - z|| <= r           (51)
//
// Let `lambda_i = (z_i - l_i) / g_i` be the value of lambda at which
// coordinate `i` reaches its bound. Coordinates split cleanly at any lambda:
// those already clamped contribute a constant, those still moving contribute
// `lambda^2 g_i^2`. Hence
//
//     ||zhat(lambda) - z||^2 =  sum_{lambda_i <= lambda} (l_i - z_i)^2
//                            +  lambda^2 * sum_{lambda_i > lambda} g_i^2   (52)
//
// Evaluating (52) at the MEDIAN of the still-undecided `lambda_i` tells us
// which half of them the optimal lambda lies past, and the other half can be
// folded into the two running sums and never looked at again. Halving the
// active set each pass gives f(N) = O(N) + f(N/2) = O(N).
//
// --------------------------------------------------------------------------
// TWO DEVIATIONS from the paper, both required here, neither stated there
// --------------------------------------------------------------------------
//
// 1. TWO-SIDED BOUNDS. Appendix F states (49) with a LOWER bound only,
//    `l <= zhat`, and reduces to `g > 0` with:
//
//        "If g_i < 0 then setting g_i = -g_i and l_i = -inf will create an
//         equivalent problem satisfying g_i > 0."
//
//    That silently assumes nothing stops the coordinate going up. PDLP's
//    feasible set is a BOX, `l <= x <= u` with `u` finite in general (our
//    canonical form has finite upper bounds on most columns), so dropping to
//    `-inf` would ignore a real constraint.
//
//    The correct reduction is to REFLECT rather than discard. For `g_i < 0`
//    substitute `w_i = -zhat_i`:
//
//        objective  g_i * zhat_i  =  (-g_i) * w_i      with -g_i > 0
//        bound      zhat_i <= u_i <=>  w_i >= -u_i
//        center     z_i           ->   -z_i
//
//    so the reflected coordinate has `g' = -g_i > 0` and `l' = -u_i`, and the
//    Euclidean norm is invariant under a coordinate reflection. The opposite
//    bound genuinely cannot bind afterwards: with `g' > 0` and a feasible
//    center `l <= z <= u`, `z - lambda*g'` only ever DECREASES, so it moves
//    away from the upper bound it started below.
//
//    Getting this wrong is silent. There is no crash and no assertion --
//    just a `rho_r` that ignores every finite upper bound, hence restarts
//    fired at the wrong times, hence a solver that merely converges badly.
//
// 2. ZERO GRADIENT ENTRIES. `lambda_i = (z_i - l_i)/g_i` divides by `g_i`.
//    A coordinate with `g_i == 0` does not move at all (it contributes
//    nothing to the objective, and staying at the center spends none of the
//    ball's radius), so it is excluded from the active set entirely rather
//    than being handed an infinite breakpoint.
//
// --------------------------------------------------------------------------
// One honest deviation in COMPLEXITY
// --------------------------------------------------------------------------
//
// Appendix F cites a worst-case-linear median (median-of-medians) to get
// O(N) worst case. This uses `std::nth_element`, which is linear on AVERAGE
// (introselect) but not guaranteed linear in the worst case. The recurrence
// and the halving argument are unchanged; only the worst-case bound is. That
// is a deliberate trade -- median-of-medians has a large constant and this
// routine runs once per restart check, not once per iteration -- and it is
// recorded here rather than left for a reader to discover.

#ifndef SOVSOLVE_SOLVER_PDLP_TRUST_REGION_HPP
#define SOVSOLVE_SOLVER_PDLP_TRUST_REGION_HPP

#include <cstddef>
#include <vector>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"

namespace sovsolve::solver::pdlp {

using core::Expected;
using core::Real;

/// The problem data. All five spans must have the same length; `radius` must
/// be finite and non-negative, and the center must satisfy `lower <= center
/// <= upper` (it is an iterate of the algorithm, so it always does).
///
/// An infinite `lower[i]` or `upper[i]` is allowed and means what it says:
/// that coordinate is never stopped on that side.
struct TrustRegionProblem {
  core::HostSpan<const Real> center;    ///< z
  core::HostSpan<const Real> lower;     ///< l, may contain -infinity
  core::HostSpan<const Real> upper;     ///< u, may contain +infinity
  core::HostSpan<const Real> gradient;  ///< g
  Real radius = 0.0;                    ///< r >= 0
};

/// Solves the problem exactly. The minimizer is written to `solution`, which
/// must be the same length as the problem's spans; the return value is the
/// optimal objective `g' zhat`, which is what `rho_r` actually consumes.
///
/// Reusable scratch lives in `TrustRegionSolver` rather than being allocated
/// per call, because PDLP calls this on (n + m)-length data on a fixed
/// schedule for the whole solve.
class TrustRegionSolver {
 public:
  [[nodiscard]] Expected<Real> solve(const TrustRegionProblem& problem,
                                     core::HostSpan<Real> solution);

 private:
  /// Active set of coordinate indices, i.e. Appendix F's `I`. Holds indices
  /// into the caller's arrays; the reflection of section 1 above is applied
  /// on the fly via `sign_`, never by copying the data.
  std::vector<std::size_t> active_;
  /// `+1` if the coordinate is used as given, `-1` if reflected.
  std::vector<Real> sign_;
  /// Breakpoint `lambda_i` per coordinate, in reflected coordinates.
  std::vector<Real> breakpoint_;
  /// `|g_i|`, i.e. the gradient after reflection, so it is strictly positive
  /// for every coordinate in `active_`.
  std::vector<Real> slope_;
  /// Distance from the center to the binding bound, in reflected coordinates:
  /// `l'_i - z'_i`, which is `<= 0`.
  std::vector<Real> reach_;
};

}  // namespace sovsolve::solver::pdlp

#endif  // SOVSOLVE_SOLVER_PDLP_TRUST_REGION_HPP
