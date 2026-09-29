#include "sovsolve/solver/HomogeneousSolve.hpp"

#include <cmath>
#include <memory>

#include "sovsolve/solver/HomogeneousNewton.hpp"
#include "sovsolve/solver/HostKkt.hpp"
#include "sovsolve/solver/NormalFactor.hpp"
#include "sovsolve/solver/SolutionQuality.hpp"
#include "sovsolve/solver/simplex/SolveSimplex.hpp"

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

/// Every complementarity factor strictly positive, as compute_homogeneous_border
/// will recompute them: `x - l tau`, `u tau - x`, `z`, `v`, `s`, `-y_I`, `tau`,
/// `kappa`.
bool strictly_interior(const model::CanonicalProblem& problem, const SolverState& state) {
  const auto pos = [](Real v) { return v > 0.0 && std::isfinite(v); };
  if (!pos(state.tau) || !pos(state.kappa)) return false;
  for (std::size_t j = 0; j < state.x.size(); ++j) {
    if (core::is_finite_bound(problem.col_lower[j]) &&
        (!pos(state.x[j] - problem.col_lower[j] * state.tau) || !pos(state.z[j]))) {
      return false;
    }
    if (core::is_finite_bound(problem.col_upper[j]) &&
        (!pos(problem.col_upper[j] * state.tau - state.x[j]) || !pos(state.v[j]))) {
      return false;
    }
  }
  for (std::size_t k = 0; k < state.s.size(); ++k) {
    if (!pos(state.s[k]) || !pos(slack_dual(state.y[problem.num_equality + k]))) return false;
  }
  return true;
}

/// Ceiling for [AG99]'s x10 escalation. OURS: the paper states no cap; this
/// matches IpmOptions::delta_max, the GPU interior point's own ceiling.
constexpr Real kMaxDeltaD = 1e-2;

/// Is the primal feasible? Optimal = yes, Infeasible = no (proved), anything
/// else = undecided. The dual simplex solves the feasibility problem (zero
/// objective); it is the engine whose Infeasible is a proof, checked on every
/// infeasible Netlib model.
core::SolverStatus primal_feasibility(const model::CanonicalProblem& problem,
                                      const model::Options& options) {
  model::CanonicalProblem feasibility;
  feasibility.c = core::RealVector(problem.num_cols(), 0.0);
  feasibility.A = problem.A.clone();
  feasibility.Q = problem.Q.clone();
  feasibility.b = problem.b.clone();
  feasibility.col_lower = problem.col_lower.clone();
  feasibility.col_upper = problem.col_upper.clone();
  feasibility.num_range = problem.num_range;
  feasibility.num_equality = problem.num_equality;

  model::Options simplex_options = options;
  simplex_options.simplex.method = model::Method::DualSimplex;
  auto r = simplex::solve_simplex(feasibility, simplex_options);
  if (!r.has_value()) return core::SolverStatus::NotConverged;
  if (r->status == core::SolverStatus::Optimal || r->status == core::SolverStatus::Infeasible) {
    return r->status;
  }
  return core::SolverStatus::NotConverged;
}

/// A dual-infeasibility certificate plus a feasible primal is Unbounded.
core::SolverStatus settle_dual_certificate(const model::CanonicalProblem& problem,
                                           const model::Options& options) {
  const core::SolverStatus f = primal_feasibility(problem, options);
  return f == core::SolverStatus::Optimal ? core::SolverStatus::Unbounded : f;
}

/// Worst row violation of `x`, relative to 1 + |b_i| (inequality rows are
/// `a'x <= b`).
Real worst_row_violation(const model::CanonicalProblem& problem, const core::RealVector& x) {
  Real worst = 0.0;
  const auto& csr = problem.A.csr;
  for (std::size_t i = 0; i < problem.num_rows(); ++i) {
    Real act = 0.0;
    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      act += csr.values()[k] * x[static_cast<std::size_t>(csr.indices()[k])];
    }
    const Real slack = act - problem.b[i];
    const Real v = i < problem.num_equality ? std::fabs(slack) : std::max(slack, 0.0);
    worst = std::max(worst, v / (1.0 + std::fabs(problem.b[i])));
  }
  return worst;
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
  kkt_options.regularization_retries = options.hsd.regularization_retries;

  // The normal-equations pattern never changes, so its ordering and symbolic
  // factorization are computed once here and reused every iteration.
  std::unique_ptr<NormalFactor> factor;
  if (options.hsd.direct && problem.num_rows() > 0) {
    factor = std::make_unique<NormalFactor>(problem);
  }

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
  work.theta_inv_floor = kkt_options.theta_inv_floor;
  HomogeneousNewtonRhs rhs;
  Real last_alpha = 0.0;

  const std::size_t max_iterations = options.hsd.max_iterations;
  // Why the loop ended. Continue after the loop means it ran out of
  // iterations without any [AA] section 1.4.5 test firing.
  HomogeneousTermination stopped_by = HomogeneousTermination::Continue;

  Real b_norm = 0.0;
  for (std::size_t i = 0; i < problem.num_rows(); ++i) b_norm = std::fmax(b_norm, std::fabs(problem.b[i]));
  Real c_norm = 0.0;
  for (std::size_t j = 0; j < problem.num_cols(); ++j) c_norm = std::fmax(c_norm, std::fabs(problem.c[j]));
  bool primal_feasible_seen = false;
  for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
    // Another engine already won the race (core/Cancel.hpp). Every check here
    // is one interior-point iteration apart, each costing a KKT solve, so
    // there is no reason to batch it.
    //
    // `publish` FIRST, exactly as the numerical-failure path below does: it is
    // what sizes and fills `result`'s vectors, and a caller handed an
    // unpublished result reads uninitialized storage. Leaving it out is how
    // the first version of this crashed.
    if (core::is_cancelled(options.cancel)) {
      publish(problem, state, result);
      result.status = core::SolverStatus::NotConverged;
      result.objective = problem.objective(result.x.span());
      return result;
    }
    result.iterations = iteration;

    status = compute_homogeneous_residuals(problem, state, 0.0, residuals);
    if (!status.ok()) break;
    state.mu = homogeneous_mu(problem, state);
    // Evidence the PRIMAL is feasible: some recovered point x/tau met the
    // primal test (the same 10x bar as the Optimal stop below). Needed to
    // turn a dual-infeasibility certificate into "Unbounded" -- see the
    // verdict mapping after the loop.
    if (state.tau > 0.0 && residuals.rp_inf / state.tau <=
                               10.0 * options.tolerances.primal_feasibility * (1.0 + b_norm)) {
      primal_feasible_seen = true;
    }

    status = homogeneous_progress(problem, state, residuals, reference, result.progress);
    if (!status.ok()) break;

    HomogeneousTermination termination = check_homogeneous_termination(
        state, result.progress, reference, last_alpha, params);
    // [AA]'s rho_P/rho_D are reductions RELATIVE TO THE START. When the start
    // is far worse than the data (Netlib grow7/15/22: b = 0, ||r_p^0|| = 5.9e7)
    // "1e-8 of the start" still leaves an absolute residual near 0.6 -- an
    // Optimal that dual simplex would not recognise. So an Optimal stop must
    // also meet the project's own measure, relative to the data: the residuals
    // of the recovered point x/tau against 1 + ||b|| and 1 + ||c||.
    // The bar is 10x the project tolerance -- OURS. [AA] section 1.4.5 itself
    // accepts tolerances "relaxed by a factor 100" once convergence is fast;
    // 10 is the tighter of the two, chosen by measurement: the residual floors near
    // 3e-8 of (1 + ||b||) on grow15/grow22 however small the regularization
    // (1.5e-8, 1e-10, 1e-12 all give 2.6e-8 to 5.9e-8; dual simplex itself
    // reaches 3.7e-8 on grow7), while the infeasible CPLEX2 sits at 0.56.
    constexpr Real kAbsoluteFactor = 10.0;
    if (termination == HomogeneousTermination::Optimal && state.tau > 0.0) {
      const bool primal_ok = residuals.rp_inf / state.tau <=
                             kAbsoluteFactor * options.tolerances.primal_feasibility * (1.0 + b_norm);
      const bool dual_ok = residuals.rd_inf / state.tau <=
                           kAbsoluteFactor * options.tolerances.dual_feasibility * (1.0 + c_norm);
      if (!primal_ok || !dual_ok) termination = HomogeneousTermination::Continue;
    }
    stopped_by = termination;
    if (termination != HomogeneousTermination::Continue) {
      // `Optimal` here is the EMBEDDING's verdict, not the model's. Which one
      // the model gets is decided by `classify_homogeneous` below, from tau,
      // kappa and the two objective terms -- [AA] Theorems 2 and 3.
      break;
    }

    status = compute_homogeneous_border(problem, state, border);
    if (!status.ok()) break;

    HostKktSolver kkt(problem, border, kkt_options, factor.get());
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

    // [AG99] section 5: when a solve needs more than one step of iterative
    // refinement, "in the following interior point iteration the default
    // regularizations are multiplied by 10". The CG around the factor IS the
    // refinement: with an accurate factor it needs about one step, plus one
    // per dense column kept out of the factor. More than that means the factor
    // no longer describes the system -- measured on Netlib brandy, whose
    // dependent rows stall at delta_d = 1.5e-8 and converge at 1e-6. Raised
    // for one iteration at a time and reset once the solves are clean again.
    if (kkt.direct() && kkt.solves() > 0) {
      const std::size_t allowed =
          2 + (factor ? factor->dense_columns() : 0);  // OURS: 1 refinement step of slack
      const bool struggling = kkt.cg_iterations() > allowed * kkt.solves();
      // The DEFAULT times 10, not the current value times 10: AG99 raises it
      // for the following iteration only. Compounding (tried first) drove
      // fit1p/fit2p, whose dense columns make CG legitimately longer, to the
      // 1e-2 cap and lost both.
      kkt_options.delta_d = struggling ? std::min(options.hsd.delta_d * 10.0, kMaxDeltaD)
                                       : options.hsd.delta_d;
    }

    Real alpha = homogeneous_step_size(problem, state, /*affine=*/false, params);
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
    // The step test predicted every gap as `gap + alpha * dgap`; the next
    // border recomputes it as `u tau - x`, and once a gap is below rounding
    // relative to `u tau` the two can disagree in sign (Netlib `greenbea`,
    // iteration 71). Take the step, check the RECOMPUTED gaps, and halve on
    // a miss rather than let the border fail on it.
    take_step(state, alpha);
    for (int retry = 0; retry < 30 && !strictly_interior(problem, state); ++retry) {
      take_step(state, -alpha);
      alpha *= 0.5;
      take_step(state, alpha);
    }
    last_alpha = alpha;
    result.iterations = iteration + 1;
  }

  if (!status.ok()) {
    publish(problem, state, result);
    result.status = core::SolverStatus::NumericalError;
    result.objective = problem.objective(result.x.span());
    return result;
  }

  // The verdict must agree with the test that ended the run. classify_homogeneous
  // reads only tau against kappa, so on its own it will call a run that
  // stopped for ANY reason Optimal whenever tau has not collapsed -- measured
  // on the infeasible Netlib models PANG and GRAN, reported Optimal with
  // primal residuals of 19 and 4.4. So: an Optimal termination may only give
  // Optimal; an Infeasible or IllPosed one may only give an infeasibility
  // verdict (or none); a run that simply stopped gives none.
  result.verdict = classify_homogeneous(problem, state, params.rho_i);
  switch (stopped_by) {
    case HomogeneousTermination::Optimal:
      if (result.verdict != HomogeneousVerdict::Optimal) {
        result.verdict = HomogeneousVerdict::Indeterminate;
      }
      break;
    case HomogeneousTermination::Infeasible:
    case HomogeneousTermination::IllPosed:
      if (result.verdict == HomogeneousVerdict::Optimal) {
        result.verdict = HomogeneousVerdict::Indeterminate;
      }
      break;
    case HomogeneousTermination::Continue:
      result.verdict = HomogeneousVerdict::Indeterminate;
      break;
  }
  result.status = status_for(result.verdict);
  // A dual-infeasibility certificate proves the primal is unbounded OR
  // infeasible ([AA] section 1.4.5); only a feasible primal makes it
  // unbounded. Measured on Netlib CPLEX1 -- infeasible, and dual infeasible
  // as well -- this path reported Unbounded. So unless an iterate already
  // showed a feasible primal, settle it EXACTLY: the dual simplex on the
  // feasibility problem (same constraints, zero objective). Feasible ->
  // Unbounded; infeasible -> Infeasible, with the simplex's own proof; no
  // verdict -> none. Rare (a dual certificate at all is rare), so its cost is
  // paid only when the answer depends on it. OURS.
  if (result.verdict == HomogeneousVerdict::DualInfeasible && !primal_feasible_seen) {
    result.status = settle_dual_certificate(problem, options);
  }

  if (result.verdict == HomogeneousVerdict::Optimal) {
    status = recover_from_homogeneous(state);
    if (!status.ok()) return status.error();
    if (result.iterations >= max_iterations) result.status = core::SolverStatus::MaxIterations;
    // THE AMBIGUOUS ZONE. The stop accepts a residual up to 10x the tolerance
    // (above), and an infeasible model can have a point that close: Netlib
    // CPLEX2 stopped "optimal" with rows violated by 3.8e-8, and the dual
    // simplex proves it infeasible. When a row is still violated beyond the
    // tolerance itself, feasibility is settled exactly before Optimal is
    // reported. OURS.
    if (result.status == core::SolverStatus::Optimal &&
        worst_row_violation(problem, state.x) > options.tolerances.primal_feasibility) {
      const core::SolverStatus f = primal_feasibility(problem, options);
      if (f == core::SolverStatus::Infeasible) {
        result.status = core::SolverStatus::Infeasible;
        result.verdict = HomogeneousVerdict::PrimalInfeasible;
      } else if (f != core::SolverStatus::Optimal) {
        result.status = core::SolverStatus::NotConverged;
      }
    }
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
