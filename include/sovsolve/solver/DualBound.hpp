// Module 27: a VALID dual bound from an approximate dual solution.
//
// THE PROBLEM THIS EXISTS TO SOLVE. Branch-and-bound prunes a node by comparing
// its dual bound against the incumbent. PDLP and the interior-point methods
// return an APPROXIMATELY optimal point, and an approximate dual objective is
// NOT a bound: if it is optimistic by even 1e-9, the node holding the true
// optimum can be pruned, and the search returns a wrong answer with no error
// anywhere. That is not a tolerance issue, it is a soundness issue -- which is
// why this module has to exist before PDLP can feed the MILP search at all.
//
// SOURCE. Neumaier and Shcherbina (2004), as presented in
//
//   D. E. Steffy and K. Wolter, "Valid Linear Programming Bounds for Exact
//   Mixed-Integer Programming", Zuse Institute Berlin -- section 2, which the
//   authors name the PRIMAL-BOUND-SHIFT method.
//
// cited below as [SW]. Nothing here is derived. [SW] states it for
//
//     Primal: max c'x  s.t.  Ax <= b,  l <= x <= u
//     Dual:   min b'y - l'z_l + u'z_u  s.t.  A'y - z_l + z_u = c,  y,z_l,z_u >= 0
//
// with the correction: given approximate `(y~, z~_l, z~_u) >= 0`, take the
// residual `r = c - A'y~ + z~_l - z~_u` and set
//
//     (y, z_l, z_u) = (y~, z~_l + r+, z~_u + r-)
//
// where `r+ = max(r, 0)` and `r- = max(-r, 0)`. The equality constraint is then
// satisfied EXACTLY, the sign constraints still hold because `r+, r- >= 0`, and
// so `b'y - l'z_l + u'z_u` is a valid bound. [SW] Proposition 2.1 gives the
// price: the corrected bound is `-l'r+ + u'r-` away from the approximate one.
//
// TRANSCRIBED TO OUR FORM, which is a minimization with mixed row senses:
//
//     Primal: min c'x  s.t.  A_E x = b_E,  A_I x <= b_I,  l <= x <= u
//     Dual:   max b'y + l'z - u'v  s.t.  A'y + z - v = c,  z,v >= 0,
//             y_E free,  y_I <= 0
//
// which is FORMULATION.md section 13.2's dual objective, already derived and
// tested there. Weak duality makes any dual-feasible `(y, z, v)` a valid LOWER
// bound on the minimum -- which is the direction branch-and-bound prunes on.
// The correction is the same shift: `z = z~ + r+`, `v = v~ + r-` for
// `r = c - A'y~ - z~ + v~`. Standard form (`l = 0, u = inf`, all rows
// equalities) recovers [SW] section 2 verbatim, and a test asserts it.
//
// THE ONE PRECONDITION, and it is why [SW] wrote a second algorithm.
// `l'z` needs `l_j` FINITE wherever `z_j > 0`, and `u'v` needs `u_j` finite
// wherever `v_j > 0`. A free column with a nonzero residual therefore drives
// the bound to minus infinity -- valid, but useless. [SW]: "if some variable
// bounds are very large or missing then it could produce weak or infinite
// bounds. We found that in our test set, 31 out of 59 problems were missing at
// least some variable bounds." Ours is worse in places: `gas11` is 44% free
// columns.
//
// So `DualBound::finite` is a first-class outcome, not an error. A caller that
// gets `false` must NOT prune -- it has no bound, and treating a default-
// constructed `-inf` as a real one is exactly the unsound pruning this module
// exists to prevent.
//
// WHAT IS DELIBERATELY NOT IMPLEMENTED. [SW]'s own contribution is
// PROJECT-AND-SHIFT (their section 3), which drops the finite-bound
// requirement by projecting onto the affine hull of the dual polyhedron and
// then shifting toward its relative interior. It needs an exact LU
// factorization, exact rational arithmetic throughout, and an S-interior point
// obtained by solving one or two auxiliary LPs in a setup phase. That is a
// large piece of machinery aimed at EXACT rational MIP, which this project is
// not; and [SW] section 2 says of the simple method that "the strength and
// simplicity of computing this bound suggests that it will be an excellent
// choice when tight primal variable bounds are available." Refinery planning
// models -- PS 26119's domain -- bound essentially everything physically:
// throughputs, blend fractions, tank levels. The simple method is the right
// first implementation here, and project-and-shift is the named fallback for
// free-column-heavy models.
//
// FLOATING POINT. [SW] computes in exact rational arithmetic. We do not, so
// the bound below is valid up to the rounding of its own arithmetic -- a few
// ulps, against a correction term that is normally far larger. Neumaier and
// Shcherbina's own remedy is directed rounding or interval arithmetic, noted
// here as the route to a genuinely certified bound rather than a safe one.

#ifndef SOVSOLVE_SOLVER_DUAL_BOUND_HPP
#define SOVSOLVE_SOLVER_DUAL_BOUND_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"

namespace sovsolve::solver {

using core::Real;

struct DualBound {
  /// A valid LOWER bound on the primal minimum, when `finite`.
  ///
  /// Never read this without checking `finite` first. When the correction needs
  /// an infinite bound it is left at negative infinity, which is technically a
  /// valid lower bound and completely useless for pruning -- and a caller that
  /// prunes on it will silently discard optimal solutions.
  Real bound = 0.0;

  /// False when some column needed a bound that is infinite. See the header.
  bool finite = false;

  /// Columns whose residual pointed at a missing bound. Zero when `finite`.
  /// Reported rather than just flagged, because on a model where this is a
  /// handful of columns the fix is to bound them, and on a model where it is
  /// half of them the fix is project-and-shift.
  std::size_t unbounded_columns = 0;

  /// [SW] Proposition 2.1: how much the correction cost, `l'r+ - u'r-` in our
  /// signs. The gap between the approximate dual objective and the valid one.
  ///
  /// This is the number that says whether the bound is USEFUL as opposed to
  /// merely correct. A first-order method stopped early gives a large residual
  /// and therefore a loose bound; the same method run longer gives a tighter
  /// one. Watching this is how a caller decides how long to run PDLP per node.
  Real correction_penalty = 0.0;

  /// `||r||_inf`, the dual-feasibility violation that had to be repaired.
  Real residual_inf = 0.0;
};

/// Corrects `(y, z, v)` to exact dual feasibility and returns the bound it
/// certifies.
///
/// The inputs are ADJUSTED COPIES, not required to be feasible: `z` and `v` are
/// clamped at zero and `y` is clamped to `y_I <= 0` first, because the sign
/// constraints are part of dual feasibility and a first-order method will
/// violate them slightly. Clamping happens BEFORE the residual is computed, so
/// the repair accounts for it.
[[nodiscard]] core::Status compute_dual_bound(const model::CanonicalProblem& problem,
                                              const core::RealVector& y,
                                              const core::RealVector& z,
                                              const core::RealVector& v,
                                              DualBound& out);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_DUAL_BOUND_HPP
