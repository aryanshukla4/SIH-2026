// Module 25 stage 2: the homogeneous algorithm's iteration control.
//
// SOURCE. Everything in this header is transcribed from
//
//   E. D. Andersen and K. D. Andersen, "The MOSEK interior point optimizer for
//   linear programming: an implementation of the homogeneous algorithm", in
//   High Performance Optimization, Kluwer, 2000, pp. 197-232.
//
// cited below as [AA] with its own equation numbers. Nothing here is derived.
// Where a formula had to be generalized, the generalization is marked
// GENERALIZED, the standard-form specialization that recovers [AA] verbatim is
// stated next to it, and a test asserts that specialization.
//
// [AA] states the whole algorithm for STANDARD FORM -- section 1.2: "For
// simplicity we will work with the LP problem in standard form", `Ax = b,
// x >= 0`. Our canonical form is `l <= x <= u` with equality and `<=` rows
// (model/Canonical.hpp). The gap that opens is confined to ONE place, the
// border column of the Newton system, which is why the Newton solve is NOT in
// this header. See the note at the end of it.
//
// What IS here, and is unaffected by that gap, because each depends only on
// the iterate and a direction and not on how the direction was produced:
//
//   - the algorithmic parameters,                [AA] Table 1.1
//   - the centering heuristic,                   [AA] (1.12)
//   - the ratio test and step size,              [AA] (1.20), (1.21)
//   - the starting point,                        [AA] (1.22)
//   - the stopping criteria,                     [AA] section 1.4.5, (1.24)
//
// WHY THE COMPOSITE VIEW. [AA] section 1.4.2 writes `x := (x; tau)` and
// `s := (s; kappa)` and from then on treats the embedding as an ordinary
// primal-dual system with `n+1` complementarity pairs. We do the same, except
// that the pair list is the one from FORMULATION.md section 6 rather than
// standard form's single `(x, s)`:
//
//     (x - l tau, z)     (u tau - x, v)     (s, -y_I)     (tau, kappa)
//
// so `n+1` becomes `active_pair_count + 1`. Every formula below that mentions
// `n+1` in [AA] uses that count instead, and that substitution is the ONLY
// change made to the step size and the stopping criteria.

#ifndef SOVSOLVE_SOLVER_HOMOGENEOUS_STEP_HPP
#define SOVSOLVE_SOLVER_HOMOGENEOUS_STEP_HPP

#include <cstddef>
#include <cstdint>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/Homogeneous.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver {

using core::Real;

/// [AA] Table 1.1, verbatim. These are MOSEK's shipped defaults, and the paper
/// records that they "have not been changed during the computational testing",
/// so they are a measured set rather than a starting guess.
struct HomogeneousParameters {
  /// `beta_1`, the cap in the centering heuristic (1.12). [AA] Table 1.1: 0.1.
  Real beta1 = 0.1;

  /// `beta_2`, the centrality floor in the step-size acceptance test (1.20).
  /// [AA] Table 1.1: 1.0e-8.
  Real beta2 = 1.0e-8;

  /// `beta_3`, the fraction-to-boundary factor: `alpha := min(beta_3 alpha_max, 1)`.
  /// [AA] Table 1.1: 0.9999.
  ///
  /// This is NOT `IpmOptions::eta` (0.995), which the direct path uses for the
  /// same job. The two are kept separate deliberately: `eta` is what every
  /// measured interior-point result in README.md was produced with, and
  /// adopting [AA]'s value on the direct path would silently change them.
  Real beta3 = 0.9999;

  /// Relative primal infeasibility tolerance `rho_bar_P`. [AA] Table 1.1: 1.0e-8.
  Real rho_p = 1.0e-8;
  /// Relative dual infeasibility tolerance `rho_bar_D`. [AA] Table 1.1: 1.0e-8.
  Real rho_d = 1.0e-8;
  /// Relative objective-gap tolerance `rho_bar_A`. [AA] Table 1.1: 1.0e-10.
  Real rho_a = 1.0e-10;
  /// `rho_bar_mu`, the floor on `mu / mu_0` in the ill-posed test.
  /// [AA] Table 1.1: 1.0e-10.
  Real rho_mu = 1.0e-10;
  /// `rho_bar_I`, the threshold at which `tau` counts as collapsed.
  /// [AA] Table 1.1: 1.0e-10.
  Real rho_i = 1.0e-10;

  /// `rho_bar_G`, the relative tolerance on the gap-row residual.
  ///
  /// NOT SUPPLIED BY [AA]. It appears in the infeasibility stopping criterion
  /// of section 1.4.5 but is absent from Table 1.1, which lists only
  /// `rho_bar_P`, `rho_bar_D`, `rho_bar_A`, `rho_bar_mu` and `rho_bar_I`. The
  /// value below matches its two siblings in that same criterion
  /// (`rho_bar_P` and `rho_bar_D`, both 1.0e-8) rather than being invented from
  /// nothing -- but it is the one number on this struct with no citation.
  Real rho_g = 1.0e-8;

  /// [AA] section 1.4.5's late-stage relaxation: if the feasibility tolerances
  /// relaxed by this factor are met, `tau >= 1000 kappa`, and fast convergence
  /// has set in, terminate anyway. Basis identification (section 1.7) is what
  /// makes MOSEK want tight tolerances in the first place, and the paper takes
  /// this escape when they turn out to be unreachable.
  Real late_relaxation = 100.0;
  /// The `tau^k >= 1000 kappa^k` of that same paragraph.
  Real late_tau_kappa_ratio = 1000.0;
  /// [AA]: "Whenever the step size is greater than 0.9 in the pure Newton step,
  /// then it is assumed that the point of fast convergence has been reached."
  Real fast_convergence_alpha = 0.9;
};

/// [AA] (1.12) and the line following it: from the maximal step in the PURE
/// NEWTON (affine-scaling) direction, produce the centering parameter
///
///     gamma := (1 - alpha_max)^2 * min(1 - alpha_max, beta_1)
///
/// after which [AA] sets `eta := 1 - gamma`.
///
/// The cubic shape is the point. `gamma` is "a heuristic estimate for the
/// possible reduction in the complementary gap" ([AA] 1.4.1): a long affine
/// step (`alpha_max -> 1`) drives it to zero and so takes nearly the pure
/// Newton direction, while a short one backs off toward the central path.
/// `beta_1` caps it because [AA] 1.4.1 notes that a small `gamma` is wanted for
/// fast residual reduction, but "a small value of gamma might cause convergence
/// problems".
///
/// This is NOT Mehrotra's `sigma = (mu_aff/mu)^3` used on the direct path
/// (FORMULATION.md section 7). [AA] chooses the homogeneous variant
/// deliberately: its (1.10) and (1.11) show the residuals and the complementary
/// gap fall at the same rate `eta` only when `eta = 1 - gamma`, which ties
/// `gamma` to the step rather than to a ratio of gaps.
[[nodiscard]] Real homogeneous_gamma(Real alpha_max, const HomogeneousParameters& params);

/// The number of complementarity pairs the embedding actually has: the
/// `active_pair_count` of FORMULATION.md section 6 -- one per finite lower
/// bound, one per finite upper bound, one per inequality row -- plus one for
/// `(tau, kappa)`. This is [AA]'s `n + 1`.
[[nodiscard]] std::size_t homogeneous_pair_count(const model::CanonicalProblem& problem);

/// Sentinel returned by `homogeneous_alpha_max` when the direction is
/// nonnegative in every coordinate and no ratio test binds.
inline constexpr Real kUnboundedStep = 1e30;

/// [AA] (1.21): the maximal step keeping every complementarity pair nonnegative,
///
///     alpha_max := argmax_{alpha >= 0} { (x; tau; s; kappa) + alpha d >= 0 }.
///
/// GENERALIZED. [AA]'s positive quantities are `(x; tau)` and `(s; kappa)`
/// because standard form has `x >= 0`. Ours are the four pairs in this header's
/// preamble, so the quantity tested in place of `x` is `x - l tau`, whose
/// direction is `dx - l dtau`: `tau` moves inside the bound itself, which is
/// what homogenizing `l -> l tau` means, and dropping that term would let a
/// step violate a bound while every coordinate of `dx` looked safe.
///
/// Standard form is `l = 0, u = inf`: then `x - l tau` is `x` with direction
/// `dx`, and the `u` pair does not exist -- (1.21) verbatim.
///
/// `affine` selects which stored direction is read: `true` for the
/// affine-scaling direction (`SolverState::dx_aff` and friends, whose step is
/// the input to `homogeneous_gamma`), `false` for the final corrected one.
[[nodiscard]] Real homogeneous_alpha_max(const model::CanonicalProblem& problem,
                                         const SolverState& state, bool affine);

/// [AA] (1.20) and section 1.4.3: take `alpha := min(beta_3 alpha_max, 1)`, then
/// reduce it until the trial point satisfies the centrality condition
///
///     (pair product)_j  >=  beta_2 * (total gap) / (pair count)   for every j.
///
/// [AA]: "The condition (1.20) prevents the iterates from converging to the
/// boundary prematurely. Furthermore, if all the iterates satisfy the condition
/// (1.20), then they converge towards a strictly complementary solution [23]."
///
/// That last clause is load-bearing, not decoration. [AA] Theorem 2 -- `tau* > 0`
/// if and only if the problem is feasible -- is stated for a STRICTLY
/// complementary solution, and so is Theorem 3, which is what turns a collapsed
/// `tau` into a named verdict. The verdict this whole module exists to produce
/// is only trustworthy if (1.20) held along the way. Skipping it does not make
/// the solver slower; it makes the answer unsound.
///
/// The step halves on each rejection. Returns the accepted step, which is 0 if
/// no positive step satisfies the condition.
[[nodiscard]] Real homogeneous_step_size(const model::CanonicalProblem& problem,
                                         const SolverState& state, bool affine,
                                         const HomogeneousParameters& params);

/// [AA] (1.22): `(x, tau, y, s, kappa) := (e, 1, 0, e, 1)`.
///
/// GENERALIZED, and the generalization is forced rather than chosen: `x := e`
/// is not available to us, because `e` need not lie strictly inside
/// `l tau <= x <= u tau`, and a strictly positive starting point is a
/// precondition of the whole method. So `x` goes to the midpoint of its own
/// bound interval where both bounds are finite, one unit inside a lone finite
/// bound, and to `e` where the column is free.
///
/// Standard form is `l = 0, u = inf`, which hits the "lone finite bound" case
/// and gives `x = l + 1 = 1 = e` -- [AA] (1.22) exactly.
///
/// The second forced change is `y`. [AA] sets `y := 0`, which is admissible
/// only because standard form has no inequality rows and therefore no
/// `(s, -y_I)` complementarity pair. Ours does, and at `y_I = 0` that pair's
/// product is exactly zero -- on the boundary, not near it -- so the centrality
/// condition (1.20) fails at every positive step and the solve stalls on
/// iteration one. Inequality rows start at `y_I = -1`; equality rows, which are
/// unrestricted and have no pair, stay at 0. A model with no inequality rows
/// therefore still gets `y = 0`. See the comment at the assignment.
///
/// [AA] section 1.4.4 also gives a more elaborate starting point (1.23), costing
/// one extra iteration, and reports it as an improvement. It is NOT implemented
/// here: (1.23) is defined by two solves of the Newton system, which is the one
/// piece this module does not have. [AA] on (1.22): "for most problems the
/// starting point (1.22) works well."
[[nodiscard]] core::Status homogeneous_starting_point(
    const model::CanonicalProblem& problem, SolverState& state);

/// What [AA] section 1.4.5's tests say about the current iterate.
enum class HomogeneousTermination : std::uint8_t {
  /// No test fired.
  Continue,
  /// `rho_P <= rho_bar_P`, `rho_D <= rho_bar_D` and `rho_A <= rho_bar_A`. Divide
  /// the iterate through by `tau` (`recover_from_homogeneous`) to get the
  /// solution to the original problem.
  Optimal,
  /// A feasible point of the embedding was found with `tau` collapsed:
  /// `rho_P`, `rho_D`, `rho_G` all within tolerance and
  /// `tau <= rho_bar_I max(1, kappa)`. The model is primal or dual infeasible;
  /// `classify_homogeneous` says which.
  Infeasible,
  /// `mu <= rho_bar_mu mu_0` and `tau <= rho_bar_I min(1, kappa)`.
  ///
  /// [AA] reports this as infeasible too. It is kept as its own outcome here
  /// because the evidence behind it is weaker: the embedding's own residuals
  /// were never driven down, so what is actually known is that the iteration
  /// collapsed -- not that a certificate was found. Folding it into
  /// `Infeasible` would report a verdict at a strength the iterate does not
  /// support, and the caller can always choose to treat the two alike.
  IllPosed,
};

/// The reference quantities [AA] section 1.4.5 measures progress against: the
/// residual norms AT THE STARTING POINT, and the starting `mu`. Captured once
/// per solve and never recomputed, because every `rho` below is a RELATIVE
/// reduction from where the run began.
struct HomogeneousReference {
  Real rp_norm_0 = 0.0;
  Real rd_norm_0 = 0.0;
  Real rg_0 = 0.0;
  Real mu_0 = 0.0;
};

/// The four measures of [AA] section 1.4.5 and (1.24).
struct HomogeneousProgress {
  Real rho_p = 0.0;  ///< `||r_P|| / max(1, ||r_P^0||)`
  Real rho_d = 0.0;  ///< `||r_D|| / max(1, ||r_D^0||)`
  Real rho_g = 0.0;  ///< `|r_G| / max(1, |r_G^0|)`

  /// [AA] (1.24): `|c'x - b'y| / (tau + |b'y|)`, which "measures the number of
  /// significant digits in the objective value".
  ///
  /// GENERALIZED: `b'y` becomes the full dual objective `b'y + l'z - u'v`, the
  /// same substitution `classify_homogeneous` and FORMULATION.md section 13.2
  /// already make. Standard form has `l = 0, u = inf`, so `l'z - u'v` vanishes
  /// and this is (1.24) verbatim.
  Real rho_a = 0.0;
};

[[nodiscard]] core::Status homogeneous_progress(const model::CanonicalProblem& problem,
                                                const SolverState& state,
                                                const HomogeneousResiduals& residuals,
                                                const HomogeneousReference& reference,
                                                HomogeneousProgress& out);

/// [AA] section 1.4.5. `last_alpha` is the previous iteration's accepted step
/// and is read only by the late-stage relaxation; pass 0 to disable that branch.
[[nodiscard]] HomogeneousTermination check_homogeneous_termination(
    const SolverState& state, const HomogeneousProgress& progress,
    const HomogeneousReference& reference, Real last_alpha,
    const HomogeneousParameters& params);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_HOMOGENEOUS_STEP_HPP
