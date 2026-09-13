#include "sovsolve/solver/gpu/Solve.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>

#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/ConvergenceChecker.hpp"
#include "sovsolve/solver/Initializer.hpp"
#include "sovsolve/solver/Logging.hpp"
#include "sovsolve/solver/Presolver.hpp"
#include "sovsolve/solver/Regularization.hpp"
#include "sovsolve/solver/Residuals.hpp"
#include "sovsolve/solver/Scaler.hpp"
#include "sovsolve/solver/SolutionReconstructor.hpp"
#include "sovsolve/solver/SolverState.hpp"
#include "sovsolve/solver/gpu/MuController.hpp"
#include "sovsolve/solver/gpu/PredictorCorrector.hpp"
#include "sovsolve/solver/gpu/ResidualCalculator.hpp"

namespace sovsolve::solver::gpu {

namespace {

/// b'y + l'z - u'v - 0.5*x'Qx (finite-bound terms only). NOT just b'y: with
/// stationarity (rd=0, Qx-A'y-z+v+c=0) and primal feasibility (rp=0)
/// substituted into c'x, the l'z/u'v/x'Qx terms are exactly what survives --
/// derived from the six-block Newton system the same way NewtonRecovery's
/// gap and the Mehrotra corrector cross terms were (PredictorCorrector.hpp).
/// Confirmed empirically: dropping l'z-u'v (the bug this replaces) left a
/// permanent, unclosable relative_gap on every Netlib instance with a finite
/// upper bound -- afiro-adjacent avgas/egout/rgn all reached primal/dual
/// residuals near 1e-12 yet reported NotConverged solely because of it.
///
/// The x'Qx term needs its 1/2: CanonicalProblem::objective() (Canonicalizer.cpp)
/// defines the primal objective as c'x + 0.5*x'Qx, so substituting stationarity
/// into the Lagrangian L = c'x + 0.5x'Qx - y'(Ax-b) - z'(x-l) + v'(x-u) leaves
/// -x'Qx + 0.5x'Qx = -0.5*x'Qx, not -x'Qx. Using the full x'Qx here (as this
/// comment used to say) double-counts the quadratic term and leaves a
/// permanent, unclosable relative_gap on every QP instance -- confirmed on
/// 2821.mps (and its -qmatrix/-quadobj/-summation/-duplicate MPS-encoding
/// variants): objective converges to exactly -6.0 with primal/dual residuals
/// at machine epsilon, yet dual_obj came out as -12.0 (x'Qx = 12.0 counted
/// once too often), a fixed gap of 6.0 that MaxIterations could never close.
Real dual_objective(const CanonicalProblem& problem, const SolverState& state) {
  Real dual_obj = 0.0;
  for (std::size_t i = 0; i < state.y.size(); ++i) dual_obj += problem.b[i] * state.y[i];

  for (std::size_t j = 0; j < state.x.size(); ++j) {
    if (core::is_finite_bound(problem.col_lower[j])) {
      dual_obj += problem.col_lower[j] * state.z[j];
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      dual_obj -= problem.col_upper[j] * state.v[j];
    }
  }

  if (!problem.Q.empty()) {
    const auto& csr = problem.Q.csr;
    Real xQx = 0.0;
    for (std::size_t i = 0; i < state.x.size(); ++i) {
      Real row_dot = 0.0;
      for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
        row_dot += csr.values()[k] * state.x[static_cast<std::size_t>(csr.indices()[k])];
      }
      xQx += state.x[i] * row_dot;
    }
    dual_obj -= 0.5 * xQx;
  }

  return dual_obj;
}

}  // namespace

Expected<Solution> solve_problem(const Problem& problem, const Options& options,
                                  const core::RealVector* warm_start_x) {
  auto canon = model::canonicalize(problem, options);
  if (!canon.has_value()) return canon.error();

  const std::size_t rows_before_presolve = canon->problem.num_rows();
  const std::size_t cols_before_presolve = canon->problem.num_cols();
  const std::size_t nnz_before_presolve = canon->problem.A.nnz();

  core::Status st = presolve(canon->problem, options, canon->transforms);
  if (!st.ok()) return st.error();

  log_presolve_summary(rows_before_presolve, cols_before_presolve, nnz_before_presolve,
                       canon->problem.num_rows(), canon->problem.num_cols(),
                       canon->problem.A.nnz(), options.log);

  st = scale(canon->problem, options, canon->transforms);
  if (!st.ok()) return st.error();

  // Module 22 warm start: forward-map the parent's ORIGINAL-space point
  // into THIS call's own freshly-built canonical space (transforms now
  // includes the ColumnScaling records scale() just pushed, which the
  // mapping needs). A null/absent hint is the ordinary path -- unaffected.
  core::RealVector warm_start_canonical;
  const core::RealVector* warm_start_canonical_ptr = nullptr;
  if (warm_start_x != nullptr) {
    warm_start_canonical =
        model::forward_map_to_canonical_hint(problem, canon->problem, canon->transforms,
                                             *warm_start_x);
    warm_start_canonical_ptr = &warm_start_canonical;
  }

  auto state = initialize(canon->problem, options, warm_start_canonical_ptr);
  if (!state.has_value()) return state.error();

  ConvergenceChecker checker(options, canon->problem);
  RegularizationController regularization(options);

  Solution best;
  bool have_best = false;
  Real best_metric = 0.0;
  SolverStatus status = SolverStatus::NotConverged;
  std::size_t iteration = 0;

  const auto start_time = std::chrono::steady_clock::now();

  for (;; ++iteration) {
    // update_mu + compute_residuals here is the ONLY place either runs now --
    // run_iteration used to redo both internally on this exact state, purely
    // to fill its own IterationRecord, which was a real, wasted SpMV-based
    // pass every single iteration (PredictorCorrector.hpp's doc comment on
    // `residuals`). Computed here because this loop needs it anyway, to
    // evaluate convergence and snapshot the best iterate BEFORE deciding
    // whether to take another step, rather than always checking one
    // iteration late -- then handed to run_iteration below instead of
    // letting it recompute the same thing.
    st = update_mu(canon->problem, *state);
    if (!st.ok()) return st.error();

    Residuals residuals;
    st = compute_residuals(canon->problem, *state, state->mu, residuals);
    if (!st.ok()) return st.error();

    const Real primal_obj = canon->problem.objective(state->x.span());
    const Real dual_obj = dual_objective(canon->problem, *state);

    status = checker.check(residuals, primal_obj, dual_obj, iteration);

    const Real metric =
        std::max({residuals.rp_inf, residuals.rd_inf, residuals.complementarity_inf});
    if (!have_best || metric < best_metric) {
      best.status = status;
      best.x = state->x.clone();
      best.s = state->s.clone();
      best.y = state->y.clone();
      best.z = state->z.clone();
      best.v = state->v.clone();
      best.objective = primal_obj;
      best.quality.primal_infeasibility = checker.primal_residual_relative(residuals.rp_inf);
      best.quality.dual_infeasibility = checker.dual_residual_relative(residuals.rd_inf);
      best.quality.relative_gap = ConvergenceChecker::gap_relative(primal_obj, dual_obj);
      best.quality.complementarity = state->mu;
      best.iterations = iteration;
      best_metric = metric;
      have_best = true;
    }

    if (status != SolverStatus::NotConverged) break;
    if (checker.is_stalled()) break;

    IterationRecord record;
    st = run_iteration(canon->problem, options, regularization, *state, residuals, record);
    if (!st.ok()) {
      // A mid-solve breakdown (e.g. the KKT system stayed ill-conditioned even
      // at delta_max -- which legitimately happens right near convergence,
      // when Theta^-1 naturally spikes for a tightly-bound variable, not just
      // on genuine divergence) does not invalidate iterates already found.
      // Same philosophy as a stall (Limits::stall_iterations): report the
      // best iterate seen, not nothing. `have_best` is always true here --
      // the residual/status snapshot above runs unconditionally before this.
      if (have_best) break;
      return st.error();
    }
    record.iteration = iteration;
    record.objective = primal_obj;
    log_iteration(record, options.log);
  }

  // NOT `best.status = status;` -- that was the bug. `status` here is
  // whatever the OUTER loop's break condition last observed (Optimal,
  // MaxIterations, a stall, or NumericalError from a mid-solve breakdown),
  // which is not necessarily the status of the snapshot actually being
  // returned: `best` can hold an EARLIER iteration's x/y/z/v (the one with
  // the smallest `metric`, line 146), and that earlier iteration's own
  // status was already recorded correctly at line 147, at the moment it was
  // captured. Overwriting it here with the break-time status discarded that
  // -- concretely, a stall on a LATER, WORSE iteration could relabel an
  // already-Optimal-quality best snapshot as NotConverged, exactly what
  // grow7.mps showed empirically: dual_infeasibility=5.7e-15 and
  // relative_gap=1.6e-16 (both many orders below tolerance) reported as
  // NotConverged, because the loop happened to stall a few iterations later
  // on a point that had since drifted worse. `best.status` (unmodified since
  // line 147) already carries the correct, snapshot-time answer.
  best.from_best_iterate = best.status != SolverStatus::Optimal;
  best.solve_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

  return reconstruct_solution(problem, canon->problem, canon->transforms, best);
}

}  // namespace sovsolve::solver::gpu
