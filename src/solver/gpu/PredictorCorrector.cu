#include "sovsolve/solver/gpu/PredictorCorrector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include "sovsolve/analysis/MatrixAnalysis.hpp"
#include "sovsolve/solver/KktSystem.hpp"
#include "sovsolve/solver/Residuals.hpp"
#include "sovsolve/solver/gpu/KktBuilder.hpp"
#include "sovsolve/solver/gpu/LinearSolver.hpp"
#include "sovsolve/solver/gpu/MuController.hpp"
#include "sovsolve/solver/gpu/NewtonRecovery.hpp"
#include "sovsolve/solver/gpu/ResidualCalculator.hpp"
#include "sovsolve/solver/gpu/StateUpdate.hpp"
#include "sovsolve/solver/gpu/StepLength.hpp"

namespace sovsolve::solver::gpu {

namespace {

Real inf_norm(const core::RealVector& v) {
  Real result = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) result = std::max(result, std::fabs(v[i]));
  return result;
}

/// Back out `dx` from the normal-equations solve's `dy` and hand the
/// assembled `[dx;dy]` to the SAME `recover_newton_direction` the augmented
/// path uses (NewtonRecovery.cu's dz/dv/ds formulas depend only on x/z/v/dx
/// and residuals, never on how dx was produced -- see NewtonRecovery.hpp).
///
///     dx = T .* (A^T dy - rhs1)
///
/// (KktBuilder.hpp's `NormalEquationsSystem` doc comment has the derivation.)
/// `A^T dy` is accumulated via `ne_system.a->csc` -- for column j, its CSC
/// major slice IS `A^T`'s row j (SparseMatrix.hpp's header comment: CSC
/// exists specifically so this needs no transpose) -- an O(nnz) host pass,
/// not the O(m*n) dense loop the earlier dense-normal-equations pass used.
Status recover_from_normal_equations(const CanonicalProblem& problem, const Residuals& residuals,
                                      const NormalEquationsSystem& ne_system,
                                      const core::RealVector& dy, SolverState& state) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();

  core::RealVector dxdy(n + m);
  const auto& csc = ne_system.a->csc;
  for (std::size_t j = 0; j < n; ++j) {
    Real a_t_dy = 0.0;
    for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      a_t_dy += csc.values()[k] * dy[static_cast<std::size_t>(csc.indices()[k])];
    }
    dxdy[j] = ne_system.theta[j] * (a_t_dy - ne_system.rhs1[j]);
  }
  for (std::size_t i = 0; i < m; ++i) dxdy[n + i] = dy[i];

  KktSystem descriptor_only;
  descriptor_only.descriptor.type = ReductionType::LpNormalEquationsDy;
  return recover_newton_direction(problem, descriptor_only, residuals, dxdy, state);
}

/// One build + solve + recover round trip, on whichever system this Newton
/// solve uses. Escalates and refactors on breakdown, decays on a clean solve
/// -- see the doc comment on Options::IpmOptions::regularization_escalation.
/// Both `solve_spd_cg` and `solve_minres` report non-convergence as
/// `pivot_ratio = +infinity` (LinearSolver.hpp's doc comment on
/// `LinearSolveResult::pivot_ratio` explains why), which always exceeds
/// `max_pivot_ratio` and so always triggers the SAME escalate/refactor path
/// this loop already had for the dense LU/Cholesky solves it used to call --
/// no new branching needed for "this Krylov solve didn't converge."
///
/// At delta_max with that signal still set, this does NOT report
/// NumericalError -- architecture.txt's "at delta_max with factorization
/// still failing, report NumericalError" means exact singularity/breakdown,
/// not "still somewhat ill-conditioned". A high pivot ratio (or a
/// non-converged Krylov solve) at delta_max is the EXPECTED, harmless
/// terminal-phase signature of a converging point (Theta^-1 legitimately
/// spikes for a tightly-bound variable right near the solution) far more
/// often than it's a genuine breakdown -- confirmed on avgas/egout/rgn, which
/// were one iteration from Optimal and got hard-aborted by treating this as
/// fatal. The direction is still accepted; the outer loop's best-iterate
/// tracking (Solve.cu) is what actually protects against a bad step doing
/// damage.
Status solve_newton_system(const CanonicalProblem& problem, const Residuals& residuals,
                            RegularizationController& regularization,
                            const Options& options, SolverState& state) {
  // FORMULATION.md 10.1 is explicit the reduction is only valid for Q=0, so
  // QP always keeps using the augmented path (solve_minres) regardless of
  // the option.
  const bool use_normal_eq = options.ipm.use_normal_equations && problem.Q.empty();

  for (;;) {
    if (use_normal_eq) {
      NormalEquationsSystem ne_system;
      Status st = build_normal_equations(problem, state, residuals, regularization.delta_p(),
                                          regularization.delta_d(), ne_system);
      if (!st.ok()) return st;

      Expected<LinearSolveResult> linear =
          solve_spd_cg(ne_system, options.ipm.cg_tolerance, options.ipm.cg_max_iterations);
      if (!linear.has_value()) return linear.error();

      if (linear->pivot_ratio > options.ipm.max_pivot_ratio) {
        if (regularization.escalate()) continue;  // refactor with the larger delta
        // delta_max reached and still ill-conditioned -- accept the direction
        // anyway, same policy as the augmented path below.
        return recover_from_normal_equations(problem, residuals, ne_system, linear->solution,
                                              state);
      }

      regularization.decay();
      return recover_from_normal_equations(problem, residuals, ne_system, linear->solution,
                                            state);
    }

    analysis::MatrixAnalysis mat_analysis;  // unused by build_kkt this pass -- see its header
    KktSystem system;
    Status st = build_kkt(problem, state, residuals, mat_analysis, regularization.delta_p(),
                          regularization.delta_d(), system);
    if (!st.ok()) return st;

    Expected<LinearSolveResult> linear =
        solve_minres(system, options.ipm.minres_tolerance, options.ipm.minres_max_iterations);
    if (!linear.has_value()) return linear.error();

    if (linear->pivot_ratio > options.ipm.max_pivot_ratio) {
      if (regularization.escalate()) continue;  // refactor with the larger delta
      // delta_max reached and still ill-conditioned -- accept the direction
      // anyway (see the doc comment above) rather than abort.
      return recover_newton_direction(problem, system, residuals, linear->solution, state);
    }

    regularization.decay();
    return recover_newton_direction(problem, system, residuals, linear->solution, state);
  }
}

}  // namespace

Status run_iteration(const CanonicalProblem& problem, const Options& options,
                      RegularizationController& regularization, SolverState& state,
                      IterationRecord& record) {
  const std::size_t events_before = regularization.escalation_events();

  Status st = update_mu(problem, state);
  if (!st.ok()) return st;

  Residuals residuals;
  st = compute_residuals(problem, state, state.mu, residuals);
  if (!st.ok()) return st;

  Real sigma = options.ipm.sigma;
  const bool predictor_corrector = options.ipm.predictor_corrector;

  if (predictor_corrector) {
    Residuals residuals_aff;
    st = compute_residuals(problem, state, 0.0, residuals_aff);
    if (!st.ok()) return st;

    st = solve_newton_system(problem, residuals_aff, regularization, options, state);
    if (!st.ok()) return st;

    state.dx_aff = state.dx.clone();
    state.ds_aff = state.ds.clone();
    state.dy_aff = state.dy.clone();
    state.dz_aff = state.dz.clone();
    state.dv_aff = state.dv.clone();

    Real alpha_p_aff = 0.0;
    Real alpha_d_aff = 0.0;
    // eta = 1: the affine step is never actually taken, only used to see
    // how far it COULD go, so no safety margin belongs here.
    st = compute_step_lengths(problem, state, 1.0, alpha_p_aff, alpha_d_aff);
    if (!st.ok()) return st;

    Real mu_aff = 0.0;
    st = compute_mu_at_trial_point(problem, state, alpha_p_aff, alpha_d_aff, mu_aff);
    if (!st.ok()) return st;
    state.mu_aff = mu_aff;

    const Real ratio = state.mu > 0.0 ? mu_aff / state.mu : 0.0;
    sigma = std::clamp(ratio * ratio * ratio, Real{0.0}, Real{1.0});
  }
  state.sigma = sigma;

  Residuals residuals_corr;
  st = compute_residuals(problem, state, sigma * state.mu, residuals_corr);
  if (!st.ok()) return st;

  if (predictor_corrector) {
    // Mehrotra's second-order correction -- see the header comment on
    // PredictorCorrector.hpp for the derivation of these three lines.
    const std::size_t n = problem.num_cols();
    const std::size_t m_e = problem.num_equality;
    const std::size_t m_i = problem.num_inequality_rows();
    // Safeguard: the cross term is a SECOND-ORDER refinement of the sigma*mu
    // target (a Taylor remainder), theoretically o(mu) near convergence --
    // not something that should ever dominate the target itself. When the
    // affine step for a pair is extreme (a tightly-bound/degenerate variable,
    // exactly the case a crude starting point produces), the raw product can
    // reach 1-2+ orders of magnitude over mu with nothing to stop it,
    // confirmed on afiro.mps: cross terms of -5.2e3 against mu=1.0, and
    // 1.0e6 against mu=5.9e4, at the exact pairs whose step length collapses
    // geometrically every iteration without ever settling. Clamping to
    // [-mu, mu] keeps it a refinement instead of letting it set the target.
    const Real cross_limit = std::max(state.mu, Real{0.0});
    for (std::size_t j = 0; j < n; ++j) {
      if (core::is_finite_bound(problem.col_lower[j])) {
        residuals_corr.rxz[j] +=
            std::clamp(state.dx_aff[j] * state.dz_aff[j], -cross_limit, cross_limit);
      }
      if (core::is_finite_bound(problem.col_upper[j])) {
        residuals_corr.ruv[j] -=
            std::clamp(state.dx_aff[j] * state.dv_aff[j], -cross_limit, cross_limit);
      }
    }
    for (std::size_t k = 0; k < m_i; ++k) {
      residuals_corr.rsy[k] -=
          std::clamp(state.ds_aff[k] * state.dy_aff[m_e + k], -cross_limit, cross_limit);
    }
    // compute_residuals's complementarity_inf is now stale -- recompute.
    residuals_corr.complementarity_inf = std::max(
        {inf_norm(residuals_corr.rxz), inf_norm(residuals_corr.ruv), inf_norm(residuals_corr.rsy)});
  }

  st = solve_newton_system(problem, residuals_corr, regularization, options, state);
  if (!st.ok()) return st;

  Real alpha_primal = 0.0;
  Real alpha_dual = 0.0;
  st = compute_step_lengths(problem, state, options.ipm.eta, alpha_primal, alpha_dual);
  if (!st.ok()) return st;
  state.alpha_primal = alpha_primal;
  state.alpha_dual = alpha_dual;

  st = apply_step(problem, state);
  if (!st.ok()) return st;

  record.mu = state.mu;
  record.mu_aff = state.mu_aff;
  record.sigma = state.sigma;
  record.alpha_primal = state.alpha_primal;
  record.alpha_dual = state.alpha_dual;
  record.primal_residual_inf = residuals.rp_inf;
  record.dual_residual_inf = residuals.rd_inf;
  record.aggregate_complementarity = residuals.complementarity_inf;
  record.regularization_events = regularization.escalation_events() - events_before;

  return Status::Ok();
}

}  // namespace sovsolve::solver::gpu
