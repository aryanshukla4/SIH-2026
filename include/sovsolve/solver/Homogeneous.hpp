// Module 25: the homogeneous self-dual embedding.
//
// Full derivation in docs/FORMULATION.md section 13. In brief: the direct
// formulation of sections 5-11 cannot report `Infeasible` or `Unbounded` at
// all, because with no feasible point there is no interior to follow. The
// embedding replaces the problem with an always-feasible one whose solution
// answers the question either way, by homogenizing every constant with a
// scalar `tau` and carrying the duality gap as a scalar `kappa`:
//
//     A_E x  - b_E tau                   = 0
//     A_I x  + s - b_I tau               = 0            s >= 0
//     A' y   + z - v - c tau             = 0            z, v >= 0
//     c'x    - b'y - l'z + u'v + kappa   = 0
//                l tau <= x <= u tau
//
// with the complementarity pairs of section 6 plus `(tau, kappa)`.
//
// This header carries the pieces that are independent of the Newton solve --
// residuals, the complementarity measure, and the classification of the final
// iterate. They are separated out because each is checkable on its own, and
// because ONE of them has an exact test available: at `tau = 1, kappa = 0` the
// embedded residuals must reproduce the direct ones bit for bit, since that is
// precisely the point at which the two formulations coincide. A sign error in
// the homogenization shows up there immediately rather than as a solve that
// merely converges badly.
//
// HOST-ONLY, deliberately, even though the IPM it serves is not. Every
// operation here is a sparse matrix-vector product or a vector reduction on
// data the caller already holds, so putting it in `sovsolve_solver` keeps it
// testable in the default `release` preset with no CUDA toolkit -- which is
// where the `tau = 1` equivalence test above actually runs.

#ifndef SOVSOLVE_SOLVER_HOMOGENEOUS_HPP
#define SOVSOLVE_SOLVER_HOMOGENEOUS_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver {

using core::Real;

/// The embedding's residuals. The first five mirror the direct formulation's
/// (FORMULATION.md section 5) with every constant scaled by `tau`; the last
/// two exist only here.
struct HomogeneousResiduals {
  core::RealVector rp;   ///< `A x + [0; s] - b tau`, length m
  core::RealVector rd;   ///< `Q x + c tau - A'y - z + v`, length n
  core::RealVector rxz;  ///< `(x - l tau).*z - mu`, zero where `l` infinite
  core::RealVector ruv;  ///< `(u tau - x).*v - mu`, zero where `u` infinite
  core::RealVector rsy;  ///< `-s.*y_I - mu`, length m_I

  /// The self-dual GAP row: `c'x - b'y - l'z + u'v + kappa`.
  ///
  /// This is the equation that makes the system self-dual, and it is the one
  /// with no counterpart in the direct formulation. `l'z` skips columns with
  /// an infinite lower bound and `u'v` those with an infinite upper bound --
  /// an infinite bound has no complementarity pair, so it contributes no term,
  /// exactly as in section 5.
  Real rg = 0.0;

  /// `tau*kappa - mu`, the embedding's own complementarity pair.
  Real rtk = 0.0;

  Real rp_inf = 0.0;
  Real rd_inf = 0.0;
  Real complementarity_inf = 0.0;
};

/// Computes the embedding's residuals at `state`, using `state.tau` and
/// `state.kappa`.
[[nodiscard]] core::Status compute_homogeneous_residuals(
    const model::CanonicalProblem& problem, const SolverState& state, Real mu,
    HomogeneousResiduals& out);

/// The complementarity measure, with `(tau, kappa)` counted as one more active
/// pair (FORMULATION.md section 13.4).
///
/// `active_pair_count` still counts only pairs that exist -- one per finite
/// lower bound, one per finite upper bound, one per inequality row -- and the
/// embedding adds exactly one to both the numerator and the denominator.
[[nodiscard]] Real homogeneous_mu(const model::CanonicalProblem& problem,
                                  const SolverState& state);

/// What the final iterate says about the model.
enum class HomogeneousVerdict : std::uint8_t {
  /// `tau` is bounded away from zero: the model has an optimal solution, and
  /// dividing the iterate by `tau` recovers it.
  Optimal,
  /// `tau -> 0` with `b'y + l'z - u'v > 0`: a dual ray, so the primal has no
  /// feasible point.
  PrimalInfeasible,
  /// `tau -> 0` with `c'x < 0`: a primal ray, so the primal is unbounded.
  DualInfeasible,
  /// `tau` has collapsed but neither objective term is decisive. Reported
  /// rather than guessed at -- see the implementation for why this case is
  /// kept distinct instead of being folded into one of the two above.
  Indeterminate,
};

/// Classifies the iterate. `tau_floor` is the ratio below which `tau` counts
/// as collapsed, measured relative to `kappa` rather than absolutely: both
/// scale together, so an absolute threshold means different things on
/// different models.
[[nodiscard]] HomogeneousVerdict classify_homogeneous(
    const model::CanonicalProblem& problem, const SolverState& state,
    Real tau_floor);

/// Divides the iterate through by `tau`, in place, turning a solution of the
/// embedding into a solution of the original problem.
///
/// Only meaningful when the verdict is `Optimal`; on a certificate the iterate
/// is already what the caller wants and dividing by a `tau` near zero would
/// amplify it into nonsense.
[[nodiscard]] core::Status recover_from_homogeneous(SolverState& state);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_HOMOGENEOUS_HPP
