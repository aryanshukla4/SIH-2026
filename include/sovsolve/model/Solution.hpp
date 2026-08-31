// A solution, expressed in the ORIGINAL problem's variables.
//
// The distinction matters. The solver works on a canonicalized, presolved,
// scaled model whose variables do not correspond one-to-one with the user's.
// A `Solution` is what comes back *after* the transform stack has been
// inverted -- so `x[j]` is the value of the column named `col_names[j]` in the
// file that was loaded, and nothing else.
//
// Duals follow the sign convention in docs/FORMULATION.md section 4: `w >= 0`
// is carried explicitly rather than eliminated as `w = -y`.

#ifndef SOVSOLVE_MODEL_SOLUTION_HPP
#define SOVSOLVE_MODEL_SOLUTION_HPP

#include <cstddef>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"

namespace sovsolve::model {

using core::Real;
using core::RealVector;
using core::SolverStatus;

/// Residual and gap measures at the returned point, all **relative** and in the
/// **infinity norm** -- see FORMULATION.md section 9.
struct SolutionQuality {
  Real primal_infeasibility = 0.0;  ///< ||rp||_inf / (1 + ||b||_inf)
  Real dual_infeasibility = 0.0;    ///< ||rd||_inf / (1 + ||c||_inf)
  Real relative_gap = 0.0;          ///< |c'x - b'y| / (1 + |c'x|)
  Real complementarity = 0.0;       ///< mu = (x'z + s'w) / (n + m)

  /// Largest violation of an original bound, after inverting the transform
  /// stack. Recovery through shifts and splits can introduce small violations
  /// that were not present in the working model, so this is measured against
  /// the original problem, not the canonical one.
  Real max_bound_violation = 0.0;
};

struct Solution {
  SolverStatus status = SolverStatus::NotConverged;

  /// Objective in the ORIGINAL sense, including `obj_constant`. If the model
  /// was a maximization, this is the maximized value -- the internal sign flip
  /// has already been inverted.
  Real objective = 0.0;

  RealVector x;  ///< length n, original columns
  RealVector s;  ///< length m, row slacks

  /// Duals. `y` is free in sign; `z` and `w` are non-negative.
  RealVector y;  ///< length m, row duals
  RealVector z;  ///< length n, reduced costs / duals for x >= 0
  RealVector w;  ///< length m, duals for s >= 0

  SolutionQuality quality;

  std::size_t iterations = 0;
  double solve_time_seconds = 0.0;

  /// True when the returned point came from the best-iterate snapshot rather
  /// than the final iterate -- i.e. the solve stalled or hit a limit. Callers
  /// reporting results should say so.
  bool from_best_iterate = false;
};

}  // namespace sovsolve::model

#endif  // SOVSOLVE_MODEL_SOLUTION_HPP
