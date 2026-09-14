#include "sovsolve/solver/HomogeneousStep.hpp"

#include <cmath>

namespace sovsolve::solver {

namespace {

using core::is_finite_bound;

/// One coordinate of the ratio test [AA] (1.21). `value` is the current
/// nonnegative quantity and `step` its direction; only a direction that
/// DECREASES it can bind.
void limit(Real value, Real step, Real& alpha) {
  if (step >= 0.0) return;
  const Real ratio = -value / step;
  if (ratio < alpha) alpha = ratio;
}

/// The four families of nonnegative quantities and their directions, visited in
/// one place so the ratio test, the centrality test and the trial point cannot
/// drift apart in which pairs they consider. `fn(primal, dual, d_primal, d_dual)`.
template <typename Fn>
void for_each_pair(const model::CanonicalProblem& problem, const SolverState& state,
                   bool affine, Fn&& fn) {
  const std::size_t n = problem.num_cols();
  const std::size_t m_i = problem.num_inequality_rows();
  const std::size_t m_e = problem.num_equality;
  const Real tau = state.tau;

  const core::RealVector& dx = affine ? state.dx_aff : state.dx;
  const core::RealVector& ds = affine ? state.ds_aff : state.ds;
  const core::RealVector& dy = affine ? state.dy_aff : state.dy;
  const core::RealVector& dz = affine ? state.dz_aff : state.dz;
  const core::RealVector& dv = affine ? state.dv_aff : state.dv;
  const Real dtau = affine ? state.dtau_aff : state.dtau;
  const Real dkappa = affine ? state.dkappa_aff : state.dkappa;

  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(problem.col_lower[j])) {
      const Real l = problem.col_lower[j];
      // `tau` scales the bound, so it appears in BOTH the quantity and its
      // direction. See the GENERALIZED note on homogeneous_alpha_max.
      fn(state.x[j] - l * tau, state.z[j], dx[j] - l * dtau, dz[j]);
    }
    if (is_finite_bound(problem.col_upper[j])) {
      const Real u = problem.col_upper[j];
      fn(u * tau - state.x[j], state.v[j], u * dtau - dx[j], dv[j]);
    }
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    // The inequality pair is `(s, -y_I)` -- FORMULATION.md section 6. The dual
    // is the NEGATED row multiplier, so its direction is negated too.
    fn(state.s[k], -state.y[m_e + k], ds[k], -dy[m_e + k]);
  }
  fn(state.tau, state.kappa, dtau, dkappa);
}

Real dual_objective(const model::CanonicalProblem& problem, const SolverState& state) {
  Real value = 0.0;
  for (std::size_t i = 0; i < problem.num_rows(); ++i) value += problem.b[i] * state.y[i];
  for (std::size_t j = 0; j < problem.num_cols(); ++j) {
    if (is_finite_bound(problem.col_lower[j])) value += problem.col_lower[j] * state.z[j];
    if (is_finite_bound(problem.col_upper[j])) value -= problem.col_upper[j] * state.v[j];
  }
  return value;
}

}  // namespace

Real homogeneous_gamma(Real alpha_max, const HomogeneousParameters& params) {
  // [AA] (1.12) takes alpha_max from argmax over [0, 1], so a ratio test that
  // found nothing binding contributes 1, not kUnboundedStep.
  const Real a = std::fmin(std::fmax(alpha_max, 0.0), 1.0);
  const Real slack = 1.0 - a;
  return slack * slack * std::fmin(slack, params.beta1);
}

std::size_t homogeneous_pair_count(const model::CanonicalProblem& problem) {
  std::size_t pairs = problem.num_inequality_rows();
  for (std::size_t j = 0; j < problem.num_cols(); ++j) {
    if (is_finite_bound(problem.col_lower[j])) ++pairs;
    if (is_finite_bound(problem.col_upper[j])) ++pairs;
  }
  return pairs + 1;  // (tau, kappa)
}

Real homogeneous_alpha_max(const model::CanonicalProblem& problem,
                           const SolverState& state, bool affine) {
  Real alpha = kUnboundedStep;
  for_each_pair(problem, state, affine,
                [&](Real primal, Real dual, Real d_primal, Real d_dual) {
                  limit(primal, d_primal, alpha);
                  limit(dual, d_dual, alpha);
                });
  return alpha;
}

Real homogeneous_step_size(const model::CanonicalProblem& problem,
                           const SolverState& state, bool affine,
                           const HomogeneousParameters& params) {
  const Real alpha_max = homogeneous_alpha_max(problem, state, affine);
  Real alpha = std::fmin(params.beta3 * alpha_max, 1.0);
  const auto pairs = static_cast<Real>(homogeneous_pair_count(problem));

  // [AA] (1.20). Rejecting and halving rather than solving for the largest
  // admissible alpha directly: the condition is quadratic in alpha in every
  // coordinate, so there is no single closed form, and [AA] section 1.4.3 says
  // only "the step size is reduced until the condition (1.20) is satisfied".
  //
  // The floor is not a tuning knob. Below it the returned step is 0 and the
  // caller must treat the iteration as stalled -- reporting a step of 1e-17 as
  // accepted would let a run grind through its whole iteration budget taking
  // steps that move nothing, which is exactly the failure mode README.md
  // records for gas11 on the direct path.
  constexpr Real kMinStep = 1e-12;
  for (; alpha > kMinStep; alpha *= 0.5) {
    Real gap = 0.0;
    for_each_pair(problem, state, affine,
                  [&](Real primal, Real dual, Real d_primal, Real d_dual) {
                    gap += (primal + alpha * d_primal) * (dual + alpha * d_dual);
                  });
    const Real floor_value = params.beta2 * gap / pairs;

    bool ok = true;
    for_each_pair(problem, state, affine,
                  [&](Real primal, Real dual, Real d_primal, Real d_dual) {
                    if (!ok) return;
                    const Real product =
                        (primal + alpha * d_primal) * (dual + alpha * d_dual);
                    if (product < floor_value) ok = false;
                  });
    if (ok) return alpha;
  }
  return 0.0;
}

core::Status homogeneous_starting_point(const model::CanonicalProblem& problem,
                                        SolverState& state) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_i = problem.num_inequality_rows();

  state.x = core::RealVector(n);
  state.s = core::RealVector(m_i, 1.0);
  state.y = core::RealVector(m, 0.0);
  state.z = core::RealVector(n, 0.0);
  state.v = core::RealVector(n, 0.0);

  // [AA] (1.22) sets `y := 0`, and that is NOT usable here unchanged.
  //
  // It is admissible in standard form only because standard form has no
  // inequality rows, so no `(s, -y_I)` pair exists for `y` to sit inside. Our
  // form has them, and at `y_I = 0` that pair's product is exactly 0 -- which
  // is not merely off-centre, it is ON the boundary. The centrality condition
  // (1.20) then fails at EVERY positive step, `homogeneous_step_size` returns
  // 0, and the solve stalls on iteration one with nothing to show for it.
  // Measured: that is precisely what happened before this line existed.
  //
  // Equality rows keep `y_i = 0` -- they are unrestricted and have no pair --
  // so a model with no inequality rows still gets `y = 0`, i.e. (1.22) verbatim.
  for (std::size_t k = 0; k < m_i; ++k) state.y[problem.num_equality + k] = -1.0;

  for (std::size_t j = 0; j < n; ++j) {
    const Real l = problem.col_lower[j];
    const Real u = problem.col_upper[j];
    const bool has_l = is_finite_bound(l);
    const bool has_u = is_finite_bound(u);

    if (has_l && has_u) {
      state.x[j] = 0.5 * (l + u);
    } else if (has_l) {
      state.x[j] = l + 1.0;  // standard form: l = 0, so x_j = 1 = e_j
    } else if (has_u) {
      state.x[j] = u - 1.0;
    } else {
      state.x[j] = 1.0;  // free column: [AA] (1.22)'s e, with no bound to respect
    }

    // [AA] (1.22) sets the dual slack to `e`. Ours splits across the two bound
    // duals, one per pair that exists, so that every pair starts at the same
    // product of 1 -- which is what makes the initial mu equal 1 below.
    if (has_l) state.z[j] = 1.0;
    if (has_u) state.v[j] = 1.0;
  }

  // Both bounds finite means `x - l tau` and `u tau - x` are each half the
  // interval rather than 1, so the pair products start at (u-l)/2 instead of 1.
  // That is accepted rather than corrected: scaling the duals to force a
  // product of 1 would make the starting dual depend on the bound widths, and
  // [AA] section 1.8 handles data range with scaling, not with the start point.

  state.tau = 1.0;
  state.kappa = 1.0;
  state.mu = homogeneous_mu(problem, state);
  return core::Status::Ok();
}

core::Status homogeneous_progress(const model::CanonicalProblem& problem,
                                  const SolverState& state,
                                  const HomogeneousResiduals& residuals,
                                  const HomogeneousReference& reference,
                                  HomogeneousProgress& out) {
  if (state.x.size() != problem.num_cols()) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "homogeneous_progress: SolverState does not match problem");
  }

  // `max(1, ||r^0||)` and not `||r^0||`: a run that STARTS feasible in one block
  // has a zero reference there, and dividing by it would make rho either NaN or
  // infinite on a residual that never needed to move. [AA] section 1.4.5 writes
  // the max for exactly this reason.
  out.rho_p = residuals.rp_inf / std::fmax(1.0, reference.rp_norm_0);
  out.rho_d = residuals.rd_inf / std::fmax(1.0, reference.rd_norm_0);
  out.rho_g = std::fabs(residuals.rg) / std::fmax(1.0, std::fabs(reference.rg_0));

  Real cx = 0.0;
  for (std::size_t j = 0; j < problem.num_cols(); ++j) cx += problem.c[j] * state.x[j];
  const Real dy = dual_objective(problem, state);
  out.rho_a = std::fabs(cx - dy) / (state.tau + std::fabs(dy));
  return core::Status::Ok();
}

HomogeneousTermination check_homogeneous_termination(
    const SolverState& state, const HomogeneousProgress& progress,
    const HomogeneousReference& reference, Real last_alpha,
    const HomogeneousParameters& params) {
  const Real tau = state.tau;
  const Real kappa = state.kappa;

  const bool feasible = progress.rho_p <= params.rho_p && progress.rho_d <= params.rho_d;

  // [AA] section 1.4.5, first test. Checked BEFORE the infeasibility tests:
  // both can hold at an iterate where tau is small but the problem is in fact
  // solvable, and the optimal verdict is the one backed by the objective
  // measure (1.24) rather than by tau alone.
  if (feasible && progress.rho_a <= params.rho_a) return HomogeneousTermination::Optimal;

  // Second test: a feasible point of the EMBEDDING with tau collapsed. The
  // gap-row residual is part of it -- without rho_G the iterate could satisfy
  // the two block residuals while the self-dual row, which is the equation
  // carrying kappa, was never solved at all, and it is kappa that makes the
  // certificate.
  if (feasible && progress.rho_g <= params.rho_g &&
      tau <= params.rho_i * std::fmax(1.0, kappa)) {
    return HomogeneousTermination::Infeasible;
  }

  // Third test. `min` rather than the `max` above: this branch has no residual
  // evidence behind it, so it demands the strictly harder condition on tau.
  if (reference.mu_0 > 0.0 && state.mu <= params.rho_mu * reference.mu_0 &&
      tau <= params.rho_i * std::fmin(1.0, kappa)) {
    return HomogeneousTermination::IllPosed;
  }

  // [AA] section 1.4.5's closing paragraph: the tight tolerances above exist to
  // serve basis identification, and when convergence is fast enough that the
  // iterate is trustworthy anyway, relaxing them by a factor of 100 is
  // accepted. The tau/kappa ratio is the second condition, whose stated purpose
  // is "to make sure that the problem is feasible".
  if (last_alpha > params.fast_convergence_alpha &&
      progress.rho_p <= params.late_relaxation * params.rho_p &&
      progress.rho_d <= params.late_relaxation * params.rho_d &&
      progress.rho_a <= params.late_relaxation * params.rho_a &&
      tau >= params.late_tau_kappa_ratio * kappa) {
    return HomogeneousTermination::Optimal;
  }

  return HomogeneousTermination::Continue;
}

}  // namespace sovsolve::solver
