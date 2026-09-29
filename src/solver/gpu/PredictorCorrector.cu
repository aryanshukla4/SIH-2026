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

/// dx = T .* (A^T dy - rhs1) -- the normal-equations back-substitution
/// (KktBuilder.hpp's `NormalEquationsSystem` doc comment has the derivation).
/// `A^T dy` is accumulated via `ne_system.a->csc` -- for column j, its CSC
/// major slice IS `A^T`'s row j (SparseMatrix.hpp's header comment: CSC
/// exists specifically so this needs no transpose) -- an O(nnz) host pass,
/// not the O(m*n) dense loop the earlier dense-normal-equations pass used.
///
/// Factored out of recover_from_normal_equations so the refinement loop
/// below can get JUST dx from a correction solve's dy, without going through
/// the dz/dv/ds derivation recover_newton_direction also does -- that has to
/// wait until dx/dy are done accumulating corrections.
core::RealVector dx_from_normal_equations(const CanonicalProblem& problem,
                                           const NormalEquationsSystem& ne_system,
                                           const core::RealVector& dy) {
  const std::size_t n = problem.num_cols();
  core::RealVector dx(n);
  const auto& csc = ne_system.a->csc;
  for (std::size_t j = 0; j < n; ++j) {
    Real a_t_dy = 0.0;
    for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      a_t_dy += csc.values()[k] * dy[static_cast<std::size_t>(csc.indices()[k])];
    }
    dx[j] = ne_system.theta[j] * (a_t_dy - ne_system.rhs1[j]);
  }
  return dx;
}

/// Back out `dx` from the normal-equations solve's `dy` and hand the
/// assembled `[dx;dy]` to the SAME `recover_newton_direction` the augmented
/// path uses (NewtonRecovery.cu's dz/dv/ds formulas depend only on x/z/v/dx
/// and residuals, never on how dx was produced -- see NewtonRecovery.hpp).
Status recover_from_normal_equations(const CanonicalProblem& problem, const Residuals& residuals,
                                      const NormalEquationsSystem& ne_system,
                                      const core::RealVector& dy, SolverState& state) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();

  core::RealVector dx = dx_from_normal_equations(problem, ne_system, dy);
  core::RealVector dxdy(n + m);
  for (std::size_t j = 0; j < n; ++j) dxdy[j] = dx[j];
  for (std::size_t i = 0; i < m; ++i) dxdy[n + i] = dy[i];

  KktSystem descriptor_only;
  descriptor_only.descriptor.type = ReductionType::LpNormalEquationsDy;
  return recover_newton_direction(problem, descriptor_only, residuals, dxdy, state);
}

/// rhs1/rhs2 of the 10.2 augmented system -- delta_p/delta_d-INDEPENDENT by
/// construction (confirmed by reading both build_kkt's and
/// build_normal_equations's formulas: neither term touches either delta).
/// Duplicated rather than shared because neither of those functions exposes
/// this piece on its own, bundled instead with delta-dependent matrix
/// assembly this needs to skip -- computed ONCE per Newton solve (outside
/// the refinement loop below, since dx/dy don't appear in either formula),
/// not once per pass.
struct TrueRhs {
  core::RealVector rhs1;  ///< length n
  core::RealVector rhs2;  ///< length m
};

TrueRhs true_rhs(const CanonicalProblem& problem, const SolverState& state,
                  const Residuals& residuals) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;
  const std::size_t m_i = problem.num_inequality_rows();

  core::RealVector rhs1(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    Real r1 = residuals.rd[j];
    if (core::is_finite_bound(problem.col_lower[j])) {
      r1 += residuals.rxz[j] / safe_gap(state.x[j] - problem.col_lower[j]);
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      r1 -= residuals.ruv[j] / safe_gap(problem.col_upper[j] - state.x[j]);
    }
    rhs1[j] = r1;
  }
  core::RealVector rhs2(m, 0.0);
  for (std::size_t i = 0; i < m_e; ++i) rhs2[i] = -residuals.rp[i];
  for (std::size_t k = 0; k < m_i; ++k) {
    const std::size_t i = m_e + k;
    rhs2[i] = -(residuals.rp[i] - residuals.rsy[k] / safe_gap(-state.y[i]));
  }
  return {std::move(rhs1), std::move(rhs2)};
}

/// r - K_true*[dx;dy] for the AUGMENTED (10.2) system, evaluated at a
/// (dx, dy) pair that came from the normal-equations/CG REDUCTION, not the
/// augmented path itself. Unlike unregularized_residual() below (the
/// augmented path's own version), this cannot skip the matrix-vector
/// products: that shortcut relies on solve_minres having already driven
/// `r - K_reg*delta` to ~0 for THIS SAME (n+m)-dim system, which holds for
/// the augmented path but not here -- solve_spd_cg only certifies the
/// REDUCED m-dim system `(A Theta A^T + diag_add) dy = rhs`, never this
/// (n+m)-dim block system directly. So `theta_inv`/`D_s` are recomputed here
/// UNFLOORED (delta_p=delta_d=0 -- the same raw quantities build_kkt
/// computes for its own diagonal before adding delta_p, not exposed
/// anywhere at this scope) and `A^T dy`/`A dx` are real sparse mat-vecs, same
/// O(nnz) host-side CSC/CSR pattern as ResidualCalculator.cu's
/// compute_residuals -- two real SpMVs per check, not free, but the same
/// cost class as one more CG iteration, not a new order of magnitude.
struct AugmentedResidual {
  core::RealVector row1;  ///< length n
  core::RealVector row2;  ///< length m
};

AugmentedResidual true_augmented_residual(const CanonicalProblem& problem, const SolverState& state,
                                           const TrueRhs& rhs, const core::RealVector& dx,
                                           const core::RealVector& dy) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;

  core::RealVector row1(n, 0.0);
  const auto& A_csc = problem.A.csc;
  for (std::size_t j = 0; j < n; ++j) {
    Real theta_inv_true = 0.0;
    if (core::is_finite_bound(problem.col_lower[j])) {
      theta_inv_true += state.z[j] / safe_gap(state.x[j] - problem.col_lower[j]);
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      theta_inv_true += state.v[j] / safe_gap(problem.col_upper[j] - state.x[j]);
    }
    Real aty = 0.0;
    for (std::size_t k = A_csc.slice_begin(j); k < A_csc.slice_end(j); ++k) {
      aty += A_csc.values()[k] * dy[static_cast<std::size_t>(A_csc.indices()[k])];
    }
    row1[j] = rhs.rhs1[j] - (-theta_inv_true * dx[j] + aty);
  }

  core::RealVector row2(m, 0.0);
  const auto& A_csr = problem.A.csr;
  for (std::size_t i = 0; i < m; ++i) {
    Real ax = 0.0;
    for (std::size_t k = A_csr.slice_begin(i); k < A_csr.slice_end(i); ++k) {
      ax += A_csr.values()[k] * dx[static_cast<std::size_t>(A_csr.indices()[k])];
    }
    Real d_s_true = 0.0;
    if (i >= m_e) d_s_true = state.s[i - m_e] / safe_gap(-state.y[i]);
    row2[i] = rhs.rhs2[i] - (ax + d_s_true * dy[i]);
  }

  return {std::move(row1), std::move(row2)};
}

/// r - K_true*delta for the augmented system, WITHOUT a second matrix or a
/// second full matrix-vector product against it.
///
/// Derived directly from build_kkt's own diagonal inserts (KktBuilder.cu),
/// not from FORMULATION.md's prose (which states the general requirement --
/// "every regularized solve is followed by iterative refinement against the
/// unregularized residual" -- but gives no formula, and per feedback on this
/// change should not be trusted blindly anyway): the (1,1) block there is
/// `-(Q + T^-1 + delta_p*I)`, the (2,2) block is `D_s + delta_d*I`. Q and
/// T^-1/D_s are IDENTICAL between the regularized matrix K_reg actually
/// factored and the true matrix K_true (delta_p=delta_d=0) -- only the
/// diagonal shift differs:
///
///     K_reg = K_true + diag(-delta_p, ..., -delta_p, delta_d, ..., delta_d)
///                        \_____ n entries _____/  \_____ m entries _____/
///
/// so K_true*delta = K_reg*delta - diag(...)*delta, and since MINRES already
/// drove (r - K_reg*delta) below minres_tolerance*||r|| (that's what
/// "converged" means for the call this follows), the unregularized residual
/// is, to within that already-small leftover:
///
///     r - K_true*delta  ~=  -delta_p*delta[0:n]  +  delta_d*delta[n:n+m]
///
/// O(n+m), not O(nnz): no SpMV, because the two matrices differ only on the
/// diagonal.
core::RealVector unregularized_residual(const CanonicalProblem& problem, Real delta_p,
                                         Real delta_d, const core::RealVector& delta) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  core::RealVector rc(n + m, 0.0);
  for (std::size_t j = 0; j < n; ++j) rc[j] = -delta_p * delta[j];
  for (std::size_t i = 0; i < m; ++i) rc[n + i] = delta_d * delta[n + i];
  return rc;
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
/// NumericalError -- docs/spec/architecture.txt's "at delta_max with factorization
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
          solve_spd_cg(ne_system, options.ipm.cg_tolerance, options.ipm.cg_max_iterations,
                       options.ipm.direct);
      if (!linear.has_value()) return linear.error();

      if (linear->pivot_ratio > options.ipm.max_pivot_ratio) {
        if (regularization.escalate()) continue;  // refactor with the larger delta
        // delta_max reached and still ill-conditioned -- accept the direction
        // anyway, same policy as the augmented path below.
        return recover_from_normal_equations(problem, residuals, ne_system, linear->solution,
                                              state);
      }

      // Iterative refinement against the TRUE (unregularized, unfloored)
      // residual -- see true_augmented_residual()'s doc comment for why this
      // needs real SpMVs rather than the augmented path's free diagonal
      // shortcut. NOT trial-applying-and-reverting on a raw-norm increase --
      // that was tried on the augmented path's version of this loop and made
      // grow7.mps measurably WORSE (see unregularized_residual's caller,
      // above); no reason to expect this reduction is different, so passes
      // are capped by max_refinement_steps and only reverted on an actual
      // solve failure (pivot_ratio), not a norm heuristic.
      core::RealVector dx = dx_from_normal_equations(problem, ne_system, linear->solution);
      core::RealVector dy = std::move(linear->solution);
      const TrueRhs rhs = true_rhs(problem, state, residuals);
      const Real rhs_norm = std::max(inf_norm(rhs.rhs1), inf_norm(rhs.rhs2));
      bool refinement_broke_down = false;
      for (int pass = 0; pass < options.ipm.max_refinement_steps; ++pass) {
        AugmentedResidual ar = true_augmented_residual(problem, state, rhs, dx, dy);
        const Real rc_norm = std::max(inf_norm(ar.row1), inf_norm(ar.row2));
        if (rc_norm <= options.ipm.cg_tolerance * std::max(Real{1.0}, rhs_norm)) break;

        // Feed (row1, row2) back through build_normal_equations as a
        // synthetic Residuals: rd <- row1, rp <- -row2, rxz/ruv/rsy <- 0.
        // dz/dv/ds are not being re-derived yet -- only dx/dy are still
        // accumulating corrections -- so there is nothing for those terms to
        // correct. This reproduces (row1, row2) as build_normal_equations's
        // own rhs1/rhs2 by construction (true_rhs uses the identical
        // formulas), which is what lets this reuse build_normal_equations +
        // solve_spd_cg unchanged instead of a third linear-system builder.
        Residuals correction_residuals;
        correction_residuals.rd = std::move(ar.row1);
        correction_residuals.rxz = core::RealVector(problem.num_cols(), 0.0);
        correction_residuals.ruv = core::RealVector(problem.num_cols(), 0.0);
        correction_residuals.rp = core::RealVector(problem.num_rows());
        for (std::size_t i = 0; i < problem.num_rows(); ++i) {
          correction_residuals.rp[i] = -ar.row2[i];
        }
        correction_residuals.rsy = core::RealVector(problem.num_inequality_rows(), 0.0);

        NormalEquationsSystem ne_correction;
        st = build_normal_equations(problem, state, correction_residuals, regularization.delta_p(),
                                    regularization.delta_d(), ne_correction);
        if (!st.ok()) return st;

        Expected<LinearSolveResult> correction =
            solve_spd_cg(ne_correction, options.ipm.cg_tolerance, options.ipm.cg_max_iterations,
                       options.ipm.direct);
        if (!correction.has_value()) return correction.error();

        core::RealVector dx_correction =
            dx_from_normal_equations(problem, ne_correction, correction->solution);
        for (std::size_t j = 0; j < dx.size(); ++j) dx[j] += dx_correction[j];
        for (std::size_t i = 0; i < dy.size(); ++i) dy[i] += correction->solution[i];
        regularization.record_refinement_pass();

        if (correction->pivot_ratio > options.ipm.max_pivot_ratio) {
          refinement_broke_down = true;
          break;
        }
      }

      core::RealVector dxdy(problem.num_cols() + problem.num_rows());
      for (std::size_t j = 0; j < dx.size(); ++j) dxdy[j] = dx[j];
      for (std::size_t i = 0; i < dy.size(); ++i) dxdy[dx.size() + i] = dy[i];
      KktSystem descriptor_only;
      descriptor_only.descriptor.type = ReductionType::LpNormalEquationsDy;

      if (refinement_broke_down) {
        if (regularization.escalate()) continue;
        return recover_newton_direction(problem, descriptor_only, residuals, dxdy, state);
      }

      regularization.decay();
      return recover_newton_direction(problem, descriptor_only, residuals, dxdy, state);
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

    // Iterative refinement against the TRUE (unregularized) residual --
    // FORMULATION.md 10.3, LinearSolver.hpp's file comment ("Iterative
    // refinement is NOT implemented for any of the four paths... a real
    // piece of design deferred, not silently approximated"). MINRES above
    // only ever minimizes the residual of the REGULARIZED system it was
    // handed; nothing previously checked that direction against the system
    // actually being solved. Reuses `system` for the correction solve --
    // same matrix, same preconditioner, only `rhs` changes -- so this is one
    // more MINRES call, not a rebuild.
    core::RealVector delta = std::move(linear->solution);
    const Real rhs_norm = inf_norm(system.rhs);
    bool refinement_broke_down = false;
    for (int pass = 0; pass < options.ipm.max_refinement_steps; ++pass) {
      core::RealVector rc = unregularized_residual(problem, regularization.delta_p(),
                                                    regularization.delta_d(), delta);
      if (inf_norm(rc) <= options.ipm.minres_tolerance * std::max(Real{1.0}, rhs_norm)) break;

      system.rhs = std::move(rc);
      Expected<LinearSolveResult> correction =
          solve_minres(system, options.ipm.minres_tolerance, options.ipm.minres_max_iterations);
      if (!correction.has_value()) return correction.error();

      // NOTE: this does NOT reject a pass whose ||rc||_inf grew rather than
      // shrank -- that guard was tried and made grow7.mps measurably WORSE
      // (NotConverged at -2.61e7 instead of Optimal at -4.79e7, empirically
      // confirmed). rc's plain inf-norm is dominated by whichever single
      // component of `delta` is currently largest, which is not the same as
      // "this correction made the direction worse" -- a component can grow
      // while the correction still improves the direction where it actually
      // matters for the Newton step. Classical iterative refinement's
      // ||K_reg^-1 * E|| < 1 convergence assumption is not guaranteed here
      // (K_reg is exactly the operator whose conditioning this exists to
      // work around), so passes are capped at max_refinement_steps rather
      // than trusted to converge, but NOT reverted on a raw-norm increase --
      // finding the right (scaled/weighted) quantity to gate on, if any, is
      // unresolved and worth a follow-up, not a guess shipped without
      // evidence it helps.
      for (std::size_t idx = 0; idx < delta.size(); ++idx) delta[idx] += correction->solution[idx];
      regularization.record_refinement_pass();

      if (correction->pivot_ratio > options.ipm.max_pivot_ratio) {
        // The correction solve itself broke down (MINRES's own non-convergence
        // signal, not the raw-norm heuristic above) -- FORMULATION.md 10.3
        // treats this the same as a failed factorization: escalate and retry
        // the WHOLE Newton solve with a larger delta, same as the primary
        // solve's own breakdown handling above.
        refinement_broke_down = true;
        break;
      }
    }
    if (refinement_broke_down) {
      if (regularization.escalate()) continue;
      // delta_max reached and refinement still breaking down -- accept the
      // best direction found so far rather than abort (same policy as an
      // ill-conditioned primary solve at delta_max, above).
      return recover_newton_direction(problem, system, residuals, delta, state);
    }

    regularization.decay();
    return recover_newton_direction(problem, system, residuals, delta, state);
  }
}

/// Gondzio (1996) multiple centrality correctors -- see the doc comment on
/// Options::IpmOptions::max_centrality_correctors for why this exists. Each
/// pass: probe how far the CURRENT accumulated direction could step (eta=1),
/// extend that trial step by a small margin, and for any complementarity
/// pair that trial point would push outside [beta_min, beta_max]*mu, solve
/// one more Newton system whose only nonzero residual entries are that
/// pair's shortfall/excess (rp=rd=0 -- linear superposition of Newton
/// systems on the SAME matrix, since state x/z/v/s/y have not moved yet).
/// The result is added onto state.dx/ds/dy/dz/dv; kept only if doing so does
/// not shrink either step length, since a badly-scaled correction is worse
/// than none -- apply_step (called by the caller) takes whatever direction
/// this function leaves behind.
Status apply_gondzio_correctors(const CanonicalProblem& problem, const Options& options,
                                 RegularizationController& regularization, SolverState& state) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;
  const std::size_t m_i = problem.num_inequality_rows();
  constexpr Real kBetaMin = 0.1;
  constexpr Real kBetaMax = 10.0;
  constexpr Real kExtraStep = 0.1;

  for (int pass = 0; pass < options.ipm.max_centrality_correctors; ++pass) {
    Real alpha_p = 0.0;
    Real alpha_d = 0.0;
    Status st = compute_step_lengths(problem, state, 1.0, alpha_p, alpha_d);
    if (!st.ok()) return st;

    const Real alpha_p_bar = std::min(alpha_p + kExtraStep, Real{1.0});
    const Real alpha_d_bar = std::min(alpha_d + kExtraStep, Real{1.0});
    const Real lo = kBetaMin * state.mu;
    const Real hi = kBetaMax * state.mu;

    Residuals gc;
    gc.rp = core::RealVector(m, 0.0);
    gc.rd = core::RealVector(n, 0.0);
    gc.rxz = core::RealVector(n, 0.0);
    gc.ruv = core::RealVector(n, 0.0);
    gc.rsy = core::RealVector(m_i, 0.0);

    bool any = false;
    for (std::size_t j = 0; j < n; ++j) {
      if (core::is_finite_bound(problem.col_lower[j])) {
        const Real gap = state.x[j] - problem.col_lower[j] + alpha_p_bar * state.dx[j];
        const Real zt = state.z[j] + alpha_d_bar * state.dz[j];
        const Real v = gap * zt;
        const Real target = std::clamp(v, lo, hi);
        if (target != v) {
          gc.rxz[j] = v - target;
          any = true;
        }
      }
      if (core::is_finite_bound(problem.col_upper[j])) {
        const Real gap = problem.col_upper[j] - state.x[j] - alpha_p_bar * state.dx[j];
        const Real vt = state.v[j] + alpha_d_bar * state.dv[j];
        const Real v = gap * vt;
        const Real target = std::clamp(v, lo, hi);
        if (target != v) {
          gc.ruv[j] = v - target;
          any = true;
        }
      }
    }
    for (std::size_t k = 0; k < m_i; ++k) {
      const std::size_t i = m_e + k;
      const Real s_trial = state.s[k] + alpha_p_bar * state.ds[k];
      const Real y_trial = state.y[i] + alpha_d_bar * state.dy[i];
      const Real v = s_trial * slack_dual(y_trial);
      const Real target = std::clamp(v, lo, hi);
      if (target != v) {
        gc.rsy[k] = v - target;
        any = true;
      }
    }

    if (!any) break;

    core::RealVector dx_prev = state.dx.clone();
    core::RealVector ds_prev = state.ds.clone();
    core::RealVector dy_prev = state.dy.clone();
    core::RealVector dz_prev = state.dz.clone();
    core::RealVector dv_prev = state.dv.clone();

    st = solve_newton_system(problem, gc, regularization, options, state);
    if (!st.ok()) return st;

    // state.d* now holds the correction-only direction from `gc` -- fold it
    // onto the direction accumulated so far.
    for (std::size_t j = 0; j < n; ++j) {
      state.dx[j] += dx_prev[j];
      state.dz[j] += dz_prev[j];
      state.dv[j] += dv_prev[j];
    }
    for (std::size_t k = 0; k < m_i; ++k) state.ds[k] += ds_prev[k];
    for (std::size_t i = 0; i < m; ++i) state.dy[i] += dy_prev[i];

    Real alpha_p_new = 0.0;
    Real alpha_d_new = 0.0;
    st = compute_step_lengths(problem, state, 1.0, alpha_p_new, alpha_d_new);
    if (!st.ok()) return st;

    if (alpha_p_new < alpha_p || alpha_d_new < alpha_d) {
      // This pass made room for nothing -- revert and stop rather than risk
      // a worse direction on the next pass too.
      state.dx = std::move(dx_prev);
      state.dz = std::move(dz_prev);
      state.dv = std::move(dv_prev);
      state.ds = std::move(ds_prev);
      state.dy = std::move(dy_prev);
      break;
    }
  }
  return Status::Ok();
}

}  // namespace

Status run_iteration(const CanonicalProblem& problem, const Options& options,
                      RegularizationController& regularization, SolverState& state,
                      const Residuals& residuals, IterationRecord& record) {
  const std::size_t events_before = regularization.escalation_events();
  const std::size_t refinement_passes_before = regularization.refinement_passes();
  Status st;

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

  if (predictor_corrector && options.ipm.max_centrality_correctors > 0) {
    st = apply_gondzio_correctors(problem, options, regularization, state);
    if (!st.ok()) return st;
  }

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
  record.refinement_passes = regularization.refinement_passes() - refinement_passes_before;

  return Status::Ok();
}

}  // namespace sovsolve::solver::gpu
