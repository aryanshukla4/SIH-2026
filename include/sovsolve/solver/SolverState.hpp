// The interior-point iterate.
//
// Module 6 (Initializer) constructs one from a CanonicalProblem; every later
// module reads and updates it in place. Strict interiority is the contract:
// x-l > 0 and u-x > 0 wherever that bound is finite, s > 0, z > 0, v > 0, and
// slack_dual(y_I) > 0 on inequality rows. Equality-row y is unrestricted in
// sign. See docs/FORMULATION.md sections 3-4.

#ifndef SOVSOLVE_SOLVER_SOLVER_STATE_HPP
#define SOVSOLVE_SOLVER_SOLVER_STATE_HPP

#include <algorithm>
#include <cstddef>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"

namespace sovsolve::solver {

using core::Real;
using core::RealVector;

struct SolverState {
  RealVector x;  ///< length n
  RealVector s;  ///< length m_I, inequality-row slacks
  RealVector y;  ///< length m, row duals (equality rows unrestricted)
  RealVector z;  ///< length n, lower-bound duals
  RealVector v;  ///< length n, upper-bound duals

  Real mu = 0.0;

  // -- Mehrotra predictor-corrector scratch (Module 8) ---------------------

  /// Affine ("predictor") direction, computed first each iteration.
  RealVector dx_aff, ds_aff, dy_aff, dz_aff, dv_aff;
  Real mu_aff = 0.0;
  Real sigma = 0.0;

  /// Corrector direction -- the one actually applied. Kept separate from the
  /// affine fields rather than overwriting them in place: Module 15 (State
  /// Updater) must not apply a direction until Module 19 (Diagnostics) has
  /// recorded the iteration that produced it.
  RealVector dx, ds, dy, dz, dv;

  Real alpha_primal = 0.0;
  Real alpha_dual = 0.0;

  [[nodiscard]] std::size_t num_cols() const noexcept { return x.size(); }
  [[nodiscard]] std::size_t num_inequality_rows() const noexcept { return s.size(); }
  [[nodiscard]] std::size_t num_rows() const noexcept { return y.size(); }
};

/// `w_s = -y_I` is not stored -- see Solution.hpp. Read the inequality block
/// of `y` through this helper so a sign test always reads positively; writing
/// `-y_i > 0` inline is easy to get backwards (module.txt Module 14).
[[nodiscard]] inline Real slack_dual(Real y_i) noexcept { return -y_i; }

/// Floors a complementarity gap (x-l, u-x, or -y_I) away from exact 0.0
/// before it's used as a divisor (KktBuilder.cu, NewtonRecovery.cu). The
/// ratio test keeps these strictly positive in exact arithmetic, but as mu
/// approaches double precision's noise floor (confirmed on shell.mps: mu hit
/// 9.66e-12 immediately before the crash), `x + alpha*dx` can round to
/// EXACTLY the bound, turning z/(x-l) into a genuine 0/0 = NaN that silently
/// poisons the whole KKT system. 1e-30 is far below any gap a well-scaled
/// problem produces legitimately, so this only ever engages at the precision
/// floor itself.
[[nodiscard]] inline Real safe_gap(Real g) noexcept { return std::max(g, Real{1e-30}); }

/// Below this distance from its own bound, a coordinate (`x-l`, `u-x`, `s`,
/// `z`, `v`, or `slack_dual(y_I)`) is treated as already converged to it,
/// not as still approaching it: StepLength.cu's ratio test excludes such a
/// coordinate from constraining `alpha_{primal,dual}` (an already-converged
/// coordinate has nothing left to contribute, and its own ratio would
/// otherwise shrink toward the noise floor and throttle EVERY other
/// coordinate's step right along with it -- found via direct tracing on
/// `80bau3b`, where one row's slack decayed past 1e-100 over ~20 iterations
/// while its dual kept growing sensibly, yet kept dragging `alpha_primal`
/// down with it). `apply_step` (StateUpdate.cu) then clamps back up to this
/// same floor, which is what makes excluding it from the ratio test safe: a
/// larger step meant for everyone else can never push an already-parked
/// coordinate past its bound. Two orders of magnitude under
/// `Tolerances::bound_violation`'s default (1e-9, Options.hpp), and twenty
/// orders above `safe_gap`'s divide-by-zero floor (1e-30) -- this engages
/// long before a gap is small enough to need THAT protection, which is the
/// point: it stops the runaway shrinkage `safe_gap` merely survives.
inline constexpr Real kConvergedFloor = 1e-12;

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_SOLVER_STATE_HPP
