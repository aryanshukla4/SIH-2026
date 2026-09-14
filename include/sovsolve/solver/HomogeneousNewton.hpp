// Module 25 stage 3: the bordered Newton solve for the homogeneous embedding.
//
// This is the piece FORMULATION.md section 13.3a listed as OPEN, and it is now
// closed -- not by finding a paper, but by a change of variables that turns the
// published standard-form system into ours exactly.
//
// THE PROBLEM. Andersen & Andersen (2000) -- cited as [AA], see
// HomogeneousStep.hpp -- give the bordered Newton system at (1.25)-(1.29), but
// for STANDARD FORM only (`Ax = b, x >= 0`, their section 1.2). Homogenizing
// sends `l -> l tau` and `u -> u tau`, so with FINITE bounds `tau` appears
// inside the complementarity rows `(x - l tau).*z` and `(u tau - x).*v`.
// Eliminating `dz` and `dv` then leaves `dtau` terms in BOTH the dual block row
// and the gap row -- and they are not the same terms. In standard form both
// vanish (`l = 0` multiplies the first by zero, `u = inf` means the second pair
// does not exist), which is why [AA]'s border is the clean `-c`.
//
// THE RESOLUTION, and why this is transcription rather than derivation. The
// shift
//
//     x = l tau + s1,     s2 = (u - l) tau - s1,     s1, s2 >= 0
//
// carries our bounded embedding onto [AA]'s (HLF) EXACTLY, with
//
//     s1 = x - l tau      s2 = u tau - x      y2 = -v      sigma = (z, v)
//
// and, on inequality rows, `sigma_I = -y_I` already being standard form. So the
// bounded system is not a new system; it is [AA]'s in different coordinates.
// Verified numerically against a dense Jacobian in both directions (residuals
// agree to 1e-15, including the gap row, which satisfies
// `T3 = -R3 - l'R2` -- the transformed gap row differs from ours by a multiple
// of the dual row, not at all). The `c'l tau` terms that appear in the shifted
// objective cancel against the shifted right-hand side, which is why our gap
// row `c'x - b'y - l'z + u'v + kappa` comes out of the transformation
// unchanged.
//
// That correspondence is also a TEST ORACLE, which is the real reason to state
// it here: `homogeneous_newton_test` builds the shifted standard-form instance,
// runs [AA]'s system on it verbatim, and requires the mapped-back direction to
// match this module's. It covers exactly the cases [AA] does not -- a column
// with an infinite bound cannot be shifted, but it also contributes no bound
// term, so its border entry is already [AA]'s `-c_j`.
//
// WHAT COMES OUT.
//
//     Theta^-1 = Z/(x - l tau) + V/(u tau - x)      <- section 10.1's diagonal
//     D_s      = S/Sigma_I                          <- on inequality rows only
//     h_x      = Theta_l l + Theta_u u - c          <- border COLUMN, x block
//     g_x      = Theta_l l + Theta_u u + c          <- border ROW,    x block
//     w        = l' Theta_l l + u' Theta_u u        <- extra trailing term
//
// writing `Theta_l = Z/(x - l tau)` and `Theta_u = V/(u tau - x)` for the two
// halves of `Theta^-1`. The system is
//
//     [ -Theta^-1   A'    h_x        ] [ dx   ]   [ rd_hat          ]
//     [   A         D_s  -b          ] [ dy   ] = [ rp_hat          ]
//     [   g_x'     -b'   -(w+kappa/t)] [ dtau ]   [ rg_hat - rtk/t  ]
//
// At `l = 0, u = inf`: `h_x = -c`, `g_x = +c`, `w = 0`, and the third row is
// the negative of [AA] (1.26)'s -- the same equation. Measured with finite
// bounds on a random instance: `||g + h|| = 7.14` and `w = 17.6`, so the border
// row genuinely is NOT the negated border column. FORMULATION.md section 13.3
// used to claim it was; that was our error and it is corrected there.
//
// WHAT IS UNCHANGED, which is the point of the whole arrangement: `K` is the
// SAME matrix as section 10.2's augmented KKT, so `KktBuilder`, `Ordering`,
// `Preconditioner` and `LinearSolver` need no changes. Inequality rows were
// checked separately and touch only `K`'s (2,2) block, never the border.
//
// HOST-ONLY, like the rest of Module 25, and for the same reason: the solve is
// expressed against an INJECTED `KktSolver` rather than against any particular
// backend, so the oracle test above runs in the default `release` preset with a
// dense factorization and no CUDA toolkit present.

#ifndef SOVSOLVE_SOLVER_HOMOGENEOUS_NEWTON_HPP
#define SOVSOLVE_SOLVER_HOMOGENEOUS_NEWTON_HPP

#include <cstddef>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver {

using core::Real;

/// The iterate-dependent pieces of the bordered system. Rebuilt once per
/// iteration; every one of them is a diagonal or a vector, so this is O(n + m)
/// and touches the matrix not at all.
struct HomogeneousBorder {
  /// `Z/(x - l tau) + V/(u tau - x)`, length n. Identical in form to section
  /// 10.1's `T^-1` -- the only difference is that `tau` scales the bounds.
  core::RealVector theta_inv;

  /// `S/Sigma_I` on inequality rows, length m_I. `K`'s (2,2) block.
  core::RealVector d_slack;

  /// Border COLUMN, x block: `Theta_l l + Theta_u u - c`, length n.
  core::RealVector h_x;

  /// Border ROW, x block: `Theta_l l + Theta_u u + c`, length n.
  ///
  /// Not `-h_x`. The two differ by `2c` whenever a finite bound is present,
  /// because the gap row picks up `+l' Theta_l` and `+u' Theta_u` from
  /// eliminating `dz` and `dv` while the dual row picks up the same terms with
  /// the opposite sign on `c`. They coincide only at `l = 0, u = inf`.
  core::RealVector g_x;

  /// `l' Theta_l l + u' Theta_u u`, a nonnegative scalar with no counterpart in
  /// [AA] -- it is zero in standard form.
  Real w = 0.0;

  /// The bordered system's trailing entry, `-(w + kappa/tau)`.
  Real trailing = 0.0;
};

/// Builds `HomogeneousBorder` at `state`. Fails if `tau` is not positive or a
/// complementarity quantity has left the interior, because every entry above
/// divides by one of them.
[[nodiscard]] core::Status compute_homogeneous_border(
    const model::CanonicalProblem& problem, const SolverState& state,
    HomogeneousBorder& out);

/// The Newton right-hand side, block by block, in the UN-eliminated variables.
///
/// Laid out to mirror `HomogeneousResiduals` -- a plain Newton step passes the
/// negated residuals, and the predictor and corrector pass modified versions of
/// the same blocks -- but kept a distinct type, because a residual is a
/// measurement and a right-hand side is an input, and conflating them is how a
/// corrector silently ends up solving the predictor's system.
struct HomogeneousNewtonRhs {
  core::RealVector rp;   ///< length m,   primal rows
  core::RealVector rd;   ///< length n,   dual rows
  Real rg = 0.0;         ///<             the self-dual gap row
  core::RealVector rxz;  ///< length n,   `(x - l tau).*z` rows
  core::RealVector ruv;  ///< length n,   `(u tau - x).*v` rows
  core::RealVector rsy;  ///< length m_I, `s.*(-y_I)` rows
  Real rtk = 0.0;        ///<             the `tau*kappa` row
};

/// Solves `K (dx; dy) = (rhs_x; rhs_y)` for
///
///     K = [ -Theta^-1   A'  ]
///         [   A         D_s ]
///
/// which is EXACTLY section 10.2's augmented KKT matrix with `Q = 0`. The
/// interface exists so this module can be exercised host-only against a dense
/// factorization while production injects the real backend -- the same
/// arrangement, and for the same reason, as `pdlp::MatVec`.
///
/// Implementations must not retain the spans; they are caller-owned buffers
/// reused across the two solves of every iteration.
class KktSolver {
 public:
  KktSolver() = default;
  KktSolver(const KktSolver&) = delete;
  KktSolver& operator=(const KktSolver&) = delete;
  virtual ~KktSolver() = default;

  [[nodiscard]] virtual core::Status solve(core::HostSpan<const Real> rhs_x,
                                           core::HostSpan<const Real> rhs_y,
                                           core::HostSpan<Real> dx,
                                           core::HostSpan<Real> dy) = 0;

  /// Total solves performed. Two per iteration is the expected figure and the
  /// thing an A/B against the direct path has to account for -- see the note on
  /// `solve_homogeneous_newton` about why it is two and not three.
  [[nodiscard]] std::size_t solves() const noexcept { return solves_; }
  void reset_solves() noexcept { solves_ = 0; }

 protected:
  std::size_t solves_ = 0;
};

/// Reusable buffers, so an iteration allocates nothing.
struct HomogeneousNewtonWorkspace {
  core::RealVector rd_hat, rp_hat;
  core::RealVector u_x, u_y;  ///< the per-right-hand-side solve
  core::RealVector p_x, p_y;  ///< the once-per-iteration solve, `K (p;q) = h`
  bool p_valid = false;       ///< set by `refresh_border_solve`
};

/// Solves `K (p; q) = (h_x; -b)`, the part of the bordered solve that depends
/// only on the iterate.
///
/// SEPARATE FROM THE STEP ON PURPOSE. [AA] section 1.5: *"even though the
/// system (1.25) has to be solved for different right-hand sides, the system
/// (1.28) is only solved once in each iteration. Therefore, the main
/// computational cost associated with the homogeneous algorithm compared to the
/// primal-dual algorithm is the additional solution of a linear equation system
/// of the form (1.28)."* Call this ONCE per iteration, then
/// `solve_homogeneous_newton` for the predictor and again for the corrector:
/// the embedding costs one extra solve per ITERATION, not per direction.
///
/// Folding it into the step would double that cost and the tests would still
/// pass, which is why it is a separate entry point rather than a cached detail.
[[nodiscard]] core::Status refresh_border_solve(const model::CanonicalProblem& problem,
                                                const HomogeneousBorder& border,
                                                KktSolver& solver,
                                                HomogeneousNewtonWorkspace& work);

/// One Newton direction. Writes `dx, ds, dy, dz, dv, dtau, dkappa` (the
/// `_aff` variants when `affine`) into `state`.
///
/// `refresh_border_solve` must have been called for this iterate first;
/// otherwise this returns an error rather than silently solving with a stale
/// `(p; q)`, which would be a wrong direction that still converges slowly
/// enough to look like a tuning problem.
[[nodiscard]] core::Status solve_homogeneous_newton(
    const model::CanonicalProblem& problem, const SolverState& state,
    const HomogeneousBorder& border, const HomogeneousNewtonRhs& rhs,
    KktSolver& solver, HomogeneousNewtonWorkspace& work, bool affine,
    SolverState& out);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_HOMOGENEOUS_NEWTON_HPP
