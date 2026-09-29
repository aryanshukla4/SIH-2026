// Module 24: infeasibility and unboundedness certificates for PDLP.
//
// Derived from Applegate, Diaz, Hinder, Lu, Lubin, O'Donoghue, Schudy,
// "Infeasibility detection with primal-dual hybrid gradient for large-scale
// linear programming" (arXiv 2102.04592), as referenced by the PDLP paper.
//
// --------------------------------------------------------------------------
// The idea: a diverging first-order method is not failing, it is pointing
// --------------------------------------------------------------------------
//
// Run PDHG on an infeasible LP and the iterates run away. That looks like a
// broken solve and is not: section 4.1 of the reference shows the direction
// they run away IN is an infeasibility certificate. Formally, all three of
//
//     z^{k+1} - z^k        (difference of iterates)
//     z^k / k              (normalized iterates)
//     2/(k+1) * zbar^k     (normalized average)
//
// converge to the same point `v = (v_x, v_y)`, the INFIMAL DISPLACEMENT
// VECTOR of the PDHG operator, and Proposition 4 states: the primal is
// infeasible **if and only if** `v_y` is nonzero, in which case `v_y` is a
// certificate; the dual is infeasible if and only if `v_x` is nonzero, in
// which case `v_x` is one.
//
// So detection costs nothing but bookkeeping: accumulate three vectors, and
// periodically test them. The reference tracks all three deliberately rather
// than just the difference of iterates -- the difference converges at
// `O(1/sqrt(k))` while the other two manage `O(1/k)`, and it explicitly notes
// that ADMM codes relying on the difference alone are leaving that on the
// table.
//
// This is a genuinely different mechanism from the two already in this
// project, and the three are worth contrasting because they fail differently:
//
//   Presolve      structural, by inspection of the data. Catches only what is
//                 visible in the matrix -- a row with identically zero
//                 activity and a right-hand side excluding zero. Two rows that
//                 contradict each other only jointly are invisible to it.
//   Dual simplex  combinatorial. `collect_candidates()` returning false IS the
//                 proof: no column can absorb the dual step, so the dual
//                 objective improves without limit. Exact, finite-time, and
//                 free -- the certificate is a row of `B^-1` that the ratio
//                 test already computed.
//   Here          analytic and ASYMPTOTIC. There is no event to be told by;
//                 `v` is a limit, so this can only ever say "the conditions
//                 hold to within a tolerance", and it has to be asked.
//
// --------------------------------------------------------------------------
// The certificates, DERIVED for this project's bounded canonical form
// --------------------------------------------------------------------------
//
// The reference states its conditions for STANDARD form (`Ax = b, x >= 0`):
// primal infeasible iff `A'v_y >= 0` and `b'v_y < 0`; dual infeasible iff
// `Av_x = 0, v_x >= 0, c'v_x < 0` (Lemma 3 and Proposition 4). Our canonical
// form is bounded and has mixed row senses, so those do not transcribe -- the
// dual ray picks up finite-bound terms and the primal ray has to lie in the
// box's recession cone. Both are re-derived below from LP duality.
//
// PRIMAL INFEASIBILITY -- a dual ray. The dual is
//
//     max  q'y + l'lambda^+ + u'lambda^-   s.t.  c - K'y = lambda,
//                                                y in Y,  lambda in Lambda
//
// (see Pdlp.hpp for why that upper-bound sign is `+` and the paper's is not.)
// A ray is a direction along which the dual stays feasible and its objective
// increases without bound. Taking the direction `v_y` and the `lambda` it
// forces, `dlambda = -K' v_y`:
//
//   (a) `v_y` lies in the recession cone of `Y`: free on equality rows,
//       `v_y_i <= 0` on inequality rows.
//   (b) `dlambda` lies in `Lambda`, which is already a cone, so the recession
//       cone is itself: `{0}` for a free column, `R-` when only the upper
//       bound is finite, `R+` when only the lower is, `R` when both are.
//   (c) the objective's asymptotic RATE along the ray is positive:
//
//           q'v_y + sum_{dlambda_j > 0} l_j dlambda_j
//                 + sum_{dlambda_j < 0} u_j dlambda_j   >  0
//
//       Far enough along the ray, `sign(lambda_j + t*dlambda_j)` is
//       `sign(dlambda_j)`, which is why the split is taken on the DIRECTION
//       rather than on any particular point. Condition (b) also guarantees
//       every term above is finite: an infinite `l_j` forces `dlambda_j <= 0`
//       and an infinite `u_j` forces `dlambda_j >= 0`, so the infinite bound
//       never multiplies a nonzero coefficient.
//
// An unbounded dual objective proves the primal has no feasible point. That is
// Farkas' lemma, and it is the same object the dual simplex produces -- just
// found by watching a limit instead of by a pivot that failed.
//
// DUAL INFEASIBILITY (i.e. the primal is unbounded) -- a primal ray. A
// direction `v_x` that stays feasible forever and decreases the objective:
//
//   (a) `v_x` lies in the recession cone of the box `[l, u]`: zero where both
//       bounds are finite, `>= 0` where only the lower is, `<= 0` where only
//       the upper is, free where neither is.
//   (b) `K v_x` lies in the rows' recession cone: exactly zero on equality
//       rows, `<= 0` on `<=` rows (moving along the ray may not increase a
//       row's activity).
//   (c) `c'v_x < 0`.
//
// --------------------------------------------------------------------------
// Why the tolerance is deliberately strict
// --------------------------------------------------------------------------
//
// `v` is approached, never reached, so every condition above is tested with a
// margin. Choosing that margin is a real decision and it is asymmetric:
//
//   too strict   -> a genuinely infeasible model is reported as
//                   `MaxIterations`. Unhelpful, but honest.
//   too loose    -> a FEASIBLE model that is merely converging slowly gets
//                   declared infeasible. That is a confidently wrong answer,
//                   and this project has already been burned by exactly that
//                   failure -- `greenbea` reporting a false `Infeasible` from
//                   the dual simplex's artificially-bounded phase 1
//                   (docs/spec/module.txt section 23, bug 3).
//
// So the default is strict, and where the conditions do not hold the engine
// stays silent and lets the iteration limit speak.
//
// WHAT "WITHIN A TOLERANCE" MEANS is not a free choice, and getting it wrong
// is what produced a false verdict here. The reference defines it (section 6,
// equations (50) and (51)): constraint violation PER UNIT OF OBJECTIVE
// IMPROVEMENT must be at most epsilon. This file originally normalized the
// candidate by its own SIZE instead, which accepts rays that are nearly flat
// -- exactly what a nearly converged run on a FEASIBLE model produces -- and
// the Linux build duly declared israel infeasible at 7600 iterations. Measured
// on both builds after the fix (scripts/cert_sweep.sh): 1e-4 gives one false
// verdict, 1e-6 and 1e-8 none, and gas11 is detected at all three.

#ifndef SOVSOLVE_SOLVER_PDLP_INFEASIBILITY_HPP
#define SOVSOLVE_SOLVER_PDLP_INFEASIBILITY_HPP

#include <cstddef>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/pdlp/MatVec.hpp"

namespace sovsolve::solver::pdlp {

using core::Real;

/// What a candidate direction was found to prove, if anything.
enum class CertificateKind : std::uint8_t {
  None,
  /// `v_y` is a dual ray: the dual objective is unbounded, so the PRIMAL has
  /// no feasible point.
  PrimalInfeasible,
  /// `v_x` is a primal ray: the primal objective decreases without limit along
  /// a feasible direction, so the primal is UNBOUNDED.
  DualInfeasible,
};

/// Tests candidate directions against the certificate conditions above.
///
/// Holds its own workspace because PDLP asks it about three candidates on a
/// fixed schedule for the whole solve.
class InfeasibilityDetector {
 public:
  InfeasibilityDetector(const model::CanonicalProblem& problem, MatVec& matvec);

  /// Tests `(v_x, v_y)` as both kinds of certificate.
  ///
  /// The acceptance test is arXiv 2102.04592 section 6, (50) and (51): an
  /// EPSILON-APPROXIMATE certificate, measuring constraint violation PER UNIT
  /// OF OBJECTIVE IMPROVEMENT (Euclidean norm). Both are ratios, homogeneous of
  /// degree zero in the candidate, so callers pass `z^{k+1} - z^k`, `z^k / k`
  /// or the normalized average unscaled, and no normalization is done here.
  ///
  /// It replaced a test that normalized by the candidate's SIZE and compared
  /// constraints to an absolute epsilon -- which accepted almost-flat rays, and
  /// declared israel (feasible) infeasible on the Linux build. See the .cpp.
  ///
  /// Costs one `K'` product (for the primal-infeasibility test) and one `K`
  /// product (for the dual-infeasibility test).
  [[nodiscard]] CertificateKind classify(const core::RealVector& v_x,
                                         const core::RealVector& v_y,
                                         Real tolerance);

 private:
  [[nodiscard]] bool is_dual_ray(const core::RealVector& v_y, Real tolerance);
  [[nodiscard]] bool is_primal_ray(const core::RealVector& v_x, Real tolerance);

  const model::CanonicalProblem* problem_;
  MatVec* matvec_;
  std::size_t m_;
  std::size_t n_;

  core::RealVector scaled_x_;
  core::RealVector scaled_y_;  ///< `v_y` projected onto its sign cone
  core::RealVector kt_v_;  ///< `K' v_y`
  core::RealVector k_v_;   ///< `K v_x`
};

}  // namespace sovsolve::solver::pdlp

#endif  // SOVSOLVE_SOLVER_PDLP_INFEASIBILITY_HPP
