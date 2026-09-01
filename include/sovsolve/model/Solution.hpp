// A solution, expressed in the ORIGINAL problem's variables.
//
// The distinction matters. The solver works on a canonicalized, presolved,
// scaled model whose variables do not correspond one-to-one with the user's.
// A `Solution` is what comes back *after* the transform stack has been
// inverted -- so `x[j]` is the value of the column named `col_names[j]` in the
// file that was loaded, and nothing else.
//
// Duals follow the team convention in docs/FORMULATION.md section 4: the
// inequality-slack dual is eliminated as `w_s = -y_I` and never stored, since
// it is not independent and a stored copy can drift from `-y_I` under rounding.
//
// That convention was checked against HiGHS and already matches what it, CPLEX
// and Gurobi report, so NO global sign normalization is applied here. For a
// minimization: `<=` rows get a non-positive dual, `>=` rows a non-negative
// one, and a variable at its lower bound a non-negative reduced cost.
//
// The one sign flip `recover_solution()` performs is local: the canonicalizer
// negates `>=` rows to fit `A_I x + s = b_I`, and those duals are negated back.
// That flip is what produces the positive `>=` dual above. A second, global
// flip on top would invert every inequality dual.

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
  /// mu, averaged over the ACTIVE complementarity pairs only: `(x-l)'z` over
  /// columns with a finite lower bound, `(u-x)'v` over those with a finite
  /// upper bound, and `-s'y_I` over the inequality rows. Pairs belonging to an
  /// infinite bound do not exist and must not be counted, or a model with many
  /// free columns reports a mu far below the truth.
  Real complementarity = 0.0;

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

  /// Duals. `y` is free in sign; `z` and `v` are non-negative.
  RealVector y;  ///< length m, row duals, in the standard reporting sign
  RealVector z;  ///< length n, duals for the lower bounds `x >= l`
  RealVector v;  ///< length n, duals for the upper bounds `x <= u`

  /// Reduced cost of column `j`, the quantity most callers mean by "the dual
  /// of a variable". Both parts are non-negative and at most one is nonzero at
  /// a solution, so the difference carries the sign.
  [[nodiscard]] Real reduced_cost(std::size_t j) const noexcept {
    return z[j] - v[j];
  }

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
