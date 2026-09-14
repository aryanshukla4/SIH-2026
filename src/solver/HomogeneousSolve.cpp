#include "sovsolve/solver/HomogeneousSolve.hpp"

#include <cmath>

#include "sovsolve/solver/HomogeneousNewton.hpp"
#include "sovsolve/solver/HostKkt.hpp"
#include "sovsolve/solver/SolutionQuality.hpp"

namespace sovsolve::solver {

namespace {

using core::is_finite_bound;

/// Builds the Newton right-hand side for one direction.
///
/// SIGNS. `compute_homogeneous_residuals` is called with `mu = 0`, so its
/// complementarity blocks carry the raw products. The block rows as
/// `HomogeneousNewton` writes them are
///
///     R1: A dx + E_I ds - b dtau        equation value `A x + s - b tau`
///     R2: A'dy + dz - dv - c dtau       equation value `-(res.rd)`
///     R3: c'dx - b'dy - l'dz + u'dv + dkappa    equation value `res.rg`
///
/// so a Newton step scaled by `eta` takes `-eta * res.rp`, `+eta * res.rd` and
/// `-eta * res.rg`. The `rd` sign is the one worth stating: `res.rd` is stored
/// as `c tau - A'y - z + v`, already the negative of its equation, so it is the
/// ONE block that is not negated here. Getting it backwards produces a
/// direction that still passes the ratio test and simply never converges.
void build_rhs(const model::CanonicalProblem& problem, const SolverState& state,
               const HomogeneousResiduals& res, Real eta, Real target, bool corrector,
               HomogeneousNewtonRhs& rhs) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_i = problem.num_inequality_rows();
  const std::size_t m_e = problem.num_equality;

  if (rhs.rp.size() != m) rhs.rp = core::RealVector(m);
  if (rhs.rd.size() != n) rhs.rd = core::RealVector(n);
  if (rhs.rxz.size() != n) rhs.rxz = core::RealVector(n, 0.0);
  if (rhs.ruv.size() != n) rhs.ruv = core::RealVector(n, 0.0);
  if (rhs.rsy.size() != m_i) rhs.rsy = core::RealVector(m_i);

  for (std::size_t i = 0; i < m; ++i) rhs.rp[i] = -eta * res.rp[i];
  for (std::size_t j = 0; j < n; ++j) rhs.rd[j] = eta * res.rd[j];
  rhs.rg = -eta * res.rg;

  // [AA] (1.13): the corrector subtracts the affine direction's own
  // complementarity products, which is what makes it a second-order method
  // rather than a re-centred first-order one.
  for (std::size_t j = 0; j < n; ++j) {
    rhs.rxz[j] = 0.0;
    rhs.ruv[j] = 0.0;
    if (is_finite_bound(problem.col_lower[j])) {
      const Real l = problem.col_lower[j];
      rhs.rxz[j] = target - res.rxz[j];
      if (corrector) {
        rhs.rxz[j] -= (state.dx_aff[j] - l * state.dtau_aff) * state.dz_aff[j];
      }
    }
    if (is_finite_bound(problem.col_upper[j])) {
      const Real u = problem.col_upper[j];
      rhs.ruv[j] = target - res.ruv[j];
      if (corrector) {
        rhs.ruv[j] -= (u * state.dtau_aff - state.dx_aff[j]) * state.dv_aff[j];
      }
    }
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    rhs.rsy[k] = target - res.rsy[k];
    if (corrector) {
      rhs.rsy[k] -= state.ds_aff[k] * slack_dual(state.dy_aff[m_e + k]);
    }
  }
  rhs.rtk = target - res.rtk;
  if (corrector) rhs.rtk -= state.dtau_aff * state.dkappa_aff;
}

void take_step(SolverState& state, Real alpha) {
  for (std::size_t j = 0; j < state.x.size(); ++j) {
    state.x[j] += alpha * state.dx[j];
    state.z[j] += alpha * state.dz[j];
    state.v[j] += alpha * state.dv[j];
  }
  for (std::size_t i = 0; i < state.y.size(); ++i) state.y[i] += alpha * state.dy[i];
  for (std::size_t k = 0; k < state.s.size(); ++k) state.s[k] += alpha * state.ds[k];
  state.tau += alpha * state.dtau;
  state.kappa += alpha * state.dkappa;
}

core::SolverStatus status_for(HomogeneousVerdict verdict) {
  switch (verdict) {
    case HomogeneousVerdict::Optimal:
      return core::SolverStatus::Optimal;
    case HomogeneousVerdict::PrimalInfeasible:
      return core::SolverStatus::Infeasible;
    case HomogeneousVerdict::DualInfeasible:
      return core::SolverStatus::Unbounded;
    case HomogeneousVerdict::Indeterminate:
      break;
  }
  return core::SolverStatus::NotConverged;
}

void publish(const model::CanonicalProblem& problem, const SolverState& state,
             HsdResult& result) {
  const std::size_t n = problem.num_cols();
  result.x = core::RealVector(n);
  result.z = core::RealVector(n);
  result.v = core::RealVector(n);
  for (std::size_t j = 0; j < n; ++j) {
    result.x[j] = state.x[j];
    result.z[j] = state.z[j];
    result.v[j] = state.v[j];
  }
  result.y = core::RealVector(state.y.size());
  for (std::size_t i = 0; i < state.y.size(); ++i) result.y[i] = state.y[i];
  result.s = core::RealVector(state.s.size());
  for (std::size_t k = 0; k < state.s.size(); ++k) result.s[k] = state.s[k];
  result.tau = state.tau;
  result.kappa = state.kappa;
}

}  // namespace

core::Expected<HsdResult> solve_hsd(const model::CanonicalProblem& problem,
                                    const model::Options& options) {
  // model::HsdOptions carries plain numbers (model sits below solver and cannot
  // include from it); unpack them into the two structs the stages expect.
  HomogeneousParameters params;
  params.beta1 = options.hsd.beta1;
  params.beta2 = options.hsd.beta2;
  params.beta3 = options.hsd.beta3;
  params.rho_p = options.hsd.rho_p;
  params.rho_d = options.hsd.rho_d;
  params.rho_a = options.hsd.rho_a;
  params.rho_mu = options.hsd.rho_mu;
  params.rho_i = options.hsd.rho_i;
  params.rho_g = options.hsd.rho_g;

  HostKktOptions kkt_options;
  kkt_options.tolerance = options.hsd.cg_tolerance;
  kkt_options.max_iterations = options.hsd.cg_max_iterations;
  kkt_options.theta_inv_floor = options.hsd.theta_inv_floor;
  kkt_options.delta_d = options.hsd.delta_d;

  HsdResult result;

  SolverState state;
  core::Status status = homogeneous_starting_point(problem, state);
  if (!status.ok()) return status.error();

  HomogeneousResiduals residuals;
  status = compute_homogeneous_residuals(problem, state, 0.0, residuals);
  if (!status.ok()) return status.error();

  // [AA] section 1.4.5 measures every rho as a RELATIVE reduction from the
  // starting point, so the reference is captured once and never recomputed.
  HomogeneousReference reference;
  reference.rp_norm_0 = residuals.rp_inf;
  reference.rd_norm_0 = residuals.rd_inf;
  reference.rg_0 = residuals.rg;
  reference.mu_0 = homogeneous_mu(problem, state);
  state.mu = reference.mu_0;

  HomogeneousBorder border;
  HomogeneousNewtonWorkspace work;
  HomogeneousNewtonRhs rhs;
  Real last_alpha = 0.0;

  const std::size_t max_iterations = options.hsd.max_iterations;
  for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
    result.iterations = iteration;

    status = compute_homogeneous_residuals(problem, state, 0.0, residuals);
    if (!status.ok()) break;
    state.mu = homogeneous_mu(problem, state);

    status = homogeneous_progress(problem, state, residuals, reference, result.progress);
    if (!status.ok()) break;

    const HomogeneousTermination termination = check_homogeneous_termination(
        state, result.progress, reference, last_alpha, params);
    if (termination != HomogeneousTermination::Continue) {
      // `Optimal` here is the EMBEDDING's verdict, not the model's. Which one
      // the model gets is decided by `classify_homogeneous` below, from tau,
      // kappa and the two objective terms -- [AA] Theorems 2 and 3.
      break;
    }

    status = compute_homogeneous_border(problem, state, border);
    if (!status.ok()) break;

    HostKktSolver kkt(problem, border, kkt_options);
    status = refresh_border_solve(problem, border, kkt, work);
    if (!status.ok()) break;

    // Predictor: [AA] section 1.4.1's pure Newton direction, gamma = 0, eta = 1.
    build_rhs(problem, state, residuals, /*eta=*/1.0, /*target=*/0.0,
              /*corrector=*/false, rhs);
    status = solve_homogeneous_newton(problem, state, border, rhs, kkt, work,
                                      /*affine=*/true, state);
    if (!status.ok()) break;

    const Real alpha_affine = homogeneous_alpha_max(problem, state, /*affine=*/true);
    const Real gamma = homogeneous_gamma(alpha_affine, params);

    // Corrector: [AA] (1.13), with eta = 1 - gamma so that (1.10) and (1.11)
    // reduce the residuals and the complementary gap at the same rate.
    build_rhs(problem, state, residuals, /*eta=*/1.0 - gamma, /*target=*/gamma * state.mu,
              /*corrector=*/true, rhs);
    status = solve_homogeneous_newton(problem, state, border, rhs, kkt, work,
                                      /*affine=*/false, state);
    if (!status.ok()) break;

    result.kkt_solves += kkt.solves();
    result.cg_iterations += kkt.cg_iterations();
    if (kkt.hit_iteration_cap()) result.inexact_solves = true;

    const Real alpha = homogeneous_step_size(problem, state, /*affine=*/false, params);
    if (!(alpha > 0.0)) {
      // No positive step satisfies the centrality condition. Reported as a
      // stall rather than pushed through: [AA]'s Theorems 2 and 3 need strict
      // complementarity, which (1.20) is what delivers, so a step taken in
      // spite of it would produce a verdict the iterate does not support.
      publish(problem, state, result);
      result.verdict = HomogeneousVerdict::Indeterminate;
      result.status = core::SolverStatus::NotConverged;
      result.objective = problem.objective(result.x.span());
      return result;
    }
    take_step(state, alpha);
    last_alpha = alpha;
    result.iterations = iteration + 1;
  }

  if (!status.ok()) {
    publish(problem, state, result);
    result.status = core::SolverStatus::NumericalError;
    result.objective = problem.objective(result.x.span());
    return result;
  }

  result.verdict = classify_homogeneous(problem, state, params.rho_i);
  result.status = status_for(result.verdict);

  if (result.verdict == HomogeneousVerdict::Optimal) {
    status = recover_from_homogeneous(state);
    if (!status.ok()) return status.error();
    if (result.iterations >= max_iterations) result.status = core::SolverStatus::MaxIterations;
  } else if (result.iterations >= max_iterations &&
             result.verdict == HomogeneousVerdict::Indeterminate) {
    result.status = core::SolverStatus::MaxIterations;
  }

  publish(problem, state, result);
  result.objective = problem.objective(result.x.span());
  return result;
}

model::Solution to_canonical_solution(const model::CanonicalProblem& problem,
                                      const HsdResult& result) {
  const std::size_t m = problem.num_rows();
  const std::size_t n = problem.num_cols();

  model::Solution solution;
  solution.status = result.status;
  solution.iterations = result.iterations;

  solution.x.resize(n);
  for (std::size_t j = 0; j < n; ++j) solution.x[j] = result.x[j];
  solution.y.resize(m);
  for (std::size_t i = 0; i < m; ++i) solution.y[i] = result.y[i];
  solution.z.resize(n);
  solution.v.resize(n);
  for (std::size_t j = 0; j < n; ++j) {
    solution.z[j] = result.z[j];
    solution.v[j] = result.v[j];
  }

  // Slacks from row activity, the same choice PdlpSolution.hpp documents: the
  // iterate's own `s` carries the interior-point method's residual, so using it
  // would report a primal infeasibility that the recovered point does not
  // actually have.
  solution.s.resize(problem.num_inequality_rows());
  if (!solution.s.empty()) {
    const auto& csr = problem.A.csr;
    for (std::size_t k = 0; k < solution.s.size(); ++k) {
      const std::size_t i = problem.num_equality + k;
      Real activity = 0.0;
      for (std::size_t idx = csr.slice_begin(i); idx < csr.slice_end(i); ++idx) {
        activity +=
            csr.values()[idx] * solution.x[static_cast<std::size_t>(csr.indices()[idx])];
      }
      solution.s[k] = problem.b[i] - activity;
    }
  }

  solution.objective = problem.objective(solution.x.span());
  compute_solution_quality(problem, solution);
  return solution;
}

}  // namespace sovsolve::solver
