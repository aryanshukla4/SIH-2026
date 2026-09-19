// Module 24: PDLP -- primal-dual hybrid gradient for linear programming.
//
// Derived from the published definition: Applegate, Diaz, Hinder, Lu, Lubin,
// O'Donoghue, Schudy, "Practical Large-Scale Linear Programming using
// Primal-Dual Hybrid Gradient" (NeurIPS 2021). The restart machinery's
// normalized duality gap comes from the same authors' arXiv 2105.12715, and
// its trust region subroutine is in pdlp/TrustRegion.hpp. Nothing here is read
// from or ported out of another solver -- see docs/HIGHS-COMPARISON.md.
//
// This file currently implements the BASELINE PDHG of the paper's equation
// (3), plus preconditioning and the termination criteria. The four
// enhancements that make it PDLP rather than PDHG -- adaptive step size
// (Algorithm 2), adaptive restarts (section 3.2), primal weight updates
// (Algorithm 3), and infeasibility certificates -- land as separate measured
// commits. The paper's own numbers say why that ordering matters: baseline
// PDHG solves 50 of 383 instances where full PDLP solves 283, so a weak
// result here is the expected checkpoint, not a defect.
//
// --------------------------------------------------------------------------
// The saddle point, and why no sign conversion happens at this boundary
// --------------------------------------------------------------------------
//
// The paper's form (1) is `min c'x s.t. Gx >= h, Ax = b, l <= x <= u`, with
// the saddle point (2)
//
//     min_{x in X} max_{y in Y}  L(x, y) = c'x - y'Kx + q'y
//
// This project's canonical form (docs/FORMULATION.md section 2) is
//
//     min c'x  s.t.  A_E x = b_E,  A_I x + s = b_I, s >= 0,  l <= x <= u
//
// and FORMULATION section 4 already states its Lagrangian:
//
//     L = 1/2 xQx + cx - y(Ax + s - b) - z(x - l) - v(u - x)
//
// For an LP (`Q` empty), with the bound terms absorbed into the feasible set
// `X` and the slack folded into the row's sense, that is
//
//     L(x, y) = c'x - y'Ax + b'y
//
// which is the paper's (2) EXACTLY, with `K = A` and `q = b`. No rearranging,
// no negation, no permutation of rows.
//
// The one place the two forms differ is the dual's feasible set, and that
// difference is carried there rather than in the matrix:
//
//     X = { x : col_lower <= x <= col_upper }
//     Y = { y : y_i free       for i <  num_equality
//               y_i <= 0       for i >= num_equality }
//
// `y_i <= 0` on an inequality row is forced, not chosen. The row is
// `A_I x + s = b_I` with `s >= 0`; FORMULATION section 4's stationarity gives
// `dL/ds = -y - w_s = 0`, so `w_s = -y_I`, and `w_s >= 0` is required because
// it multiplies the constraint `s >= 0`. Hence `y_I <= 0`.
//
// That is the SAME convention the rest of this project already uses --
// FORMULATION section 4 verified it against what HiGHS reports for a `<=` row
// on a minimization, and the simplex asserts it in its own tests. Keeping it
// means `y` leaves this module in exactly the form `recover_solution()`
// already expects, and no global sign normalization is applied anywhere. An
// earlier draft of FORMULATION claimed the opposite convention; applying it
// would invert every inequality dual, and the warning there applies verbatim
// here.
//
// The paper puts inequality rows FIRST (`y_{1:m1} >= 0`, because its
// inequalities are `>=`); this project puts equality rows first and its
// inequalities are `<=`. Both differences are absorbed by the definition of
// `Y` above. Permuting the matrix to match the paper would copy the largest
// object in the problem in order to remove one branch from the projection.
//
// --------------------------------------------------------------------------
// The iteration
// --------------------------------------------------------------------------
//
// Paper equation (3), specialized to the above:
//
//     x^{k+1} = proj_X( x^k - tau (c - K' y^k) )
//     y^{k+1} = proj_Y( y^k + sigma (q - K (2 x^{k+1} - x^k)) )
//
// The gradients are read straight off `L`: `dL/dx = c - K'y` (minimized, so
// the primal steps against it) and `dL/dy = q - Kx` (maximized, so the dual
// steps along it, at the extrapolated point `2x^{k+1} - x^k`).
//
// Step sizes are reparameterized by (4) into a magnitude and a balance:
//
//     tau = eta / omega,   sigma = omega * eta
//
// so that `tau*sigma = eta^2` is what convergence depends on (`tau sigma
// ||K||^2 <= 1`) while `omega` -- the "primal weight" -- only shifts emphasis
// between the primal and dual iterates. Baseline PDHG fixes
// `eta = 0.9/||K||_2` with `||K||_2` from power iteration, and `omega = 1`.
// PDLP proper needs neither: Algorithm 1 initializes from `1/||K||_inf`, which
// is a single sweep, and then adapts.
//
// --------------------------------------------------------------------------
// Termination (paper section 4.1, equations 6a-6c)
// --------------------------------------------------------------------------
//
// With `lambda = proj_Lambda(c - K'y)` the reduced costs, where `Lambda_j`
// depends only on which of column `j`'s bounds are finite:
//
//     l = -inf, u = +inf   ->  {0}    a free column's reduced cost must vanish
//     l = -inf, u finite   ->  R-
//     l finite, u = +inf   ->  R+
//     otherwise            ->  R      both bounds can absorb any sign
//
//     (6a) |dual_obj - c'x|            <= eps (1 + |dual_obj| + |c'x|)
//     (6b) ||primal residual||_2       <= eps (1 + ||q||_2)
//     (6c) ||c - K'y - lambda||_2      <= eps (1 + ||c||_2)
//
//     dual_obj = q'y + sum_{l_j finite} l_j lambda_j^+
//                    + sum_{u_j finite} u_j lambda_j^-
//
// NOTE THE SIGN ON THE UPPER-BOUND TERM. The paper's problem (1) writes the
// dual objective as `q'y + l'lambda^+ - u'lambda^-`, while its notation
// section defines `v^-_i = min{0, v_i}` -- a NON-POSITIVE quantity. Those two
// statements are inconsistent, and taking them together gives the wrong sign.
// The smallest counterexample settles it:
//
//     min -x   over   0 <= x <= 1,  no rows.
//     Optimum x = 1, objective -1. With no rows, lambda = c = -1,
//     so lambda^+ = 0 and lambda^- = -1.
//
//         paper as printed:  0 + l*0 - u*(-1) = +1     wrong
//         with `+ u'lambda^-`:  0 + l*0 + u*(-1) = -1  correct
//
// So either the paper means `v^-` to be the non-negative magnitude of the
// negative part (the other common convention, which contradicts its own
// notation section), or the minus sign is a slip. Either way the formula that
// satisfies strong duality is the one above, and it is what this code
// computes -- derived from FORMULATION.md section 4's Lagrangian rather than
// transcribed:
//
//     L = c'x - y'(Ax + s - b) - z'(x - l) - v'(u - x)
//     stationarity:  c - K'y - z + v = 0   =>   z - v = lambda
//     dual objective = q'y + l'z - u'v,  with z = lambda^+, v = -lambda^-
//                    = q'y + l'lambda^+ + u'lambda^-
//
// The guards on `l_j`/`u_j` finite are not cosmetic: an infinite bound always
// pairs with a zero `lambda_j` by the projection above, and `inf * 0` is NaN.
//
// The primal residual is the equality rows' `(Kx - q)_i` together with the
// inequality rows' one-sided violation `max((Kx - q)_i, 0)` -- our `<=` rows
// are only violated upward. This is the paper's `(Ax - b; (h - Gx)^+)` with
// `G = -A_I`, `h = -b_I` substituted.
//
// Where these are evaluated matters and is a deliberate choice. The paper
// evaluates on the PRESOLVED, PRECONDITIONED instance. This evaluates on the
// canonical problem as the solver holds it, which is scaled -- so the
// tolerance is a scaled-space tolerance, and the honest unscaled residuals
// are computed once at the end by the same path the simplex uses. Reporting a
// scaled residual as if it were the real one is the failure this avoids.

#ifndef SOVSOLVE_SOLVER_PDLP_PDLP_HPP
#define SOVSOLVE_SOLVER_PDLP_PDLP_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/pdlp/IterationBackend.hpp"
#include "sovsolve/solver/pdlp/MatVec.hpp"

namespace sovsolve::solver::pdlp {

using core::Real;

/// What a PDLP run produces. Deliberately shaped like `simplex::SimplexResult`
/// so `LpSolve.cpp` treats the three engines the same way, minus everything
/// basis-shaped: a first-order method has no basis and never lands on a
/// vertex, which is exactly why crossover to the simplex is the natural
/// follow-on rather than an optimization.
struct PdlpResult {
  core::SolverStatus status = core::SolverStatus::NotConverged;

  core::RealVector x;  ///< length num_cols, in the SCALED canonical space
  core::RealVector y;  ///< length num_rows, sign convention as above
  /// `proj_Lambda(c - K'y)`: the reduced costs, split into `z`/`v` downstream.
  core::RealVector reduced_cost;

  Real objective = 0.0;

  std::size_t iterations = 0;
  /// `K` and `K'` products applied, counted separately. One "KKT pass" in the
  /// paper's sense is two of these.
  std::size_t matrix_products = 0;
  /// Outer-loop restarts performed (paper section 3.2).
  std::size_t restarts = 0;
  /// Trial steps the adaptive rule rejected (Algorithm 2 line 8 failing).
  /// Each cost one extra `K` product, so this is the price paid for the rule;
  /// comparing it against the iteration saving is how the rule is judged.
  std::size_t step_rejections = 0;

  /// The three quantities (6a)-(6c) test, at the returned iterate. Kept so a
  /// non-converged run says HOW far it got rather than only that it stopped.
  Real relative_duality_gap = 0.0;
  Real relative_primal_residual = 0.0;
  Real relative_dual_residual = 0.0;
};

/// Runs PDLP on an already canonicalized, presolved and scaled problem.
///
/// The problem must be an LP: `Q` non-empty is rejected with
/// `UnsupportedFeature` rather than silently ignored.
[[nodiscard]] core::Expected<PdlpResult> solve_pdlp(
    const model::CanonicalProblem& problem, const model::Options& options);

/// The same, with `K` applied by a caller-supplied implementation.
///
/// This is the seam the GPU backend enters through. PDLP touches the matrix
/// ONLY as `K x` and `K' y` -- there is no factorization, no ordering, no
/// preconditioner to port -- so a cuSPARSE `MatVec` is the entire GPU story for
/// this engine, and everything else in the loop is untouched vector arithmetic.
///
/// The backend lives in `sovsolve_solver_gpu` and is INJECTED here rather than
/// selected here, because the `gpu -> solver` library edge is one-way
/// (src/solver/CMakeLists.txt) and must stay that way.
[[nodiscard]] core::Expected<PdlpResult> solve_pdlp(
    const model::CanonicalProblem& problem, const model::Options& options,
    MatVec& matvec);

/// The same, with the ITERATE also held by a caller-supplied implementation
/// (IterationBackend.hpp) -- the device-resident path. `matvec` still serves the
/// cold path (termination, restarts, certificates, once per `check_interval`);
/// `backend` serves every iteration in between.
[[nodiscard]] core::Expected<PdlpResult> solve_pdlp(
    const model::CanonicalProblem& problem, const model::Options& options,
    MatVec& matvec, IterationBackend& backend);

}  // namespace sovsolve::solver::pdlp

#endif  // SOVSOLVE_SOLVER_PDLP_PDLP_HPP
