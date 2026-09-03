// The interior-point iterate.
//
// Module 6 (Initializer) constructs one from a CanonicalProblem; every later
// module reads and updates it in place. Strict interiority is the contract:
// x-l > 0 and u-x > 0 wherever that bound is finite, s > 0, z > 0, v > 0, and
// slack_dual(y_I) > 0 on inequality rows. Equality-row y is unrestricted in
// sign. See docs/FORMULATION.md sections 3-4.

#ifndef SOVSOLVE_SOLVER_SOLVER_STATE_HPP
#define SOVSOLVE_SOLVER_SOLVER_STATE_HPP

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

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_SOLVER_STATE_HPP
