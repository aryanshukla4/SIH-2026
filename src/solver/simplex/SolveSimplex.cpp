#include "sovsolve/solver/simplex/SolveSimplex.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "sovsolve/solver/simplex/DualSimplex.hpp"
#include "sovsolve/solver/simplex/PrimalSimplex.hpp"

namespace sovsolve::solver::simplex {
namespace {

using core::Real;
using core::is_finite_bound;

/// COST PERTURBATION, transcribed from Koberstein, "The dual simplex method,
/// techniques for a fast and stable implementation" (PhD thesis, Paderborn,
/// 2005), section 6.3.1. Returns false, leaving `out` untouched, when the
/// problem does not qualify.
///
/// THE PROBLEM IT SOLVES. In a dually degenerate iteration the entering
/// variable has zero reduced cost, the dual step is zero and the objective does
/// not move; long runs of them ("stalling", section 6.1.2) are what a 0/1 MILP
/// relaxation produces, because its perfectly conditioned data leaves no
/// rounding noise to break the ties. Measured here on MIPLIB pk1 (45 rows):
/// 2000-7500 pivots per branch-and-bound node, every LP Optimal at objective 0.
/// Perturbing the costs by small random amounts removes the ties.
///
/// WHEN (the thesis's test): the structural costs take fewer than n/4 distinct
/// values -- "we assume the problem to be significantly dual degenerate". Then
/// every structural column that is neither fixed nor free is perturbed. The
/// thesis's second trigger -- perturbing only degenerate positions after
/// `3 * refactor_interval` iterations without progress -- is NOT implemented; it
/// needs a stall detector inside the iteration, and the up-front test is the
/// path the thesis applies first.
///
/// HOW, the four steps, with `eps_D` the dual feasibility tolerance:
///   1. xi_j = 100 eps_D + psi |c_j|,  psi = 1e-5.
///   2. xi_j <- -0.5 xi_j (1 + mu) if u_j finite, +0.5 xi_j (1 + mu) otherwise,
///      mu uniform on [0, 1] -- (6.36). The sign keeps a column dual feasible
///      at the bound it can rest on.
///   3. xi_j <- w_{nu_j} xi_j, nu_j the nonzeros in column j, with
///      w = (1e-2, 1e-1, 1, 2, 5, 10, 20, 30, 40, 100) and w_10 for nu_j > 10
///      -- (6.37)-(6.38). Denser columns get LARGER perturbations, so ties
///      resolve toward sparse entering columns and the LU stays sparse.
///   4. Scale by 0.1 or 10 until |xi_j| lies in [xi_min, xi_max],
///      xi_min = min{1e-2 eps_D, psi}, xi_max = max{1e3 eps_D, 10 psi mean|c|}
///      -- (6.39)-(6.40).
///
/// Readings of the text, stated rather than hidden: "the size of the cost
/// coefficient" is taken as |c_j|, and (6.40)'s (1/n) sum c_j as the mean of
/// |c_j| -- a signed mean could cancel to zero on a model with costs of both
/// signs, which cannot be the intent of a magnitude bound. A column with no
/// nonzeros takes w_1. The random numbers come from a fixed-seed generator, so
/// a solve is reproducible.
bool make_cost_perturbation(const model::CanonicalProblem& problem,
                            const model::Options& options, std::vector<Real>& out) {
  const std::size_t n = problem.num_cols();
  if (n == 0) return false;

  std::vector<Real> sorted(problem.c.data(), problem.c.data() + n);
  std::sort(sorted.begin(), sorted.end());
  const auto distinct = static_cast<std::size_t>(
      std::unique(sorted.begin(), sorted.end()) - sorted.begin());
  if (4 * distinct >= n) return false;

  constexpr Real kPsi = 1e-5;
  static constexpr Real kWeights[10] = {1e-2, 1e-1, 1.0, 2.0,  5.0,
                                        10.0, 20.0, 30.0, 40.0, 100.0};
  const Real eps_d = options.simplex.dual_feasibility_tolerance;

  Real mean_abs = 0.0;
  for (std::size_t j = 0; j < n; ++j) mean_abs += std::fabs(problem.c[j]);
  mean_abs /= static_cast<Real>(n);
  const Real xi_min = std::min(1e-2 * eps_d, kPsi);
  const Real xi_max = std::max(1e3 * eps_d, kPsi * 10.0 * mean_abs);

  // A fixed-seed linear congruential generator: reproducible, and the quality
  // needed is only "not all the same", not statistical randomness.
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  auto uniform = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<Real>(state >> 11) * (1.0 / 9007199254740992.0);
  };

  const auto& csc = problem.A.csc;
  out.assign(problem.c.data(), problem.c.data() + n);
  for (std::size_t j = 0; j < n; ++j) {
    const bool has_l = is_finite_bound(problem.col_lower[j]);
    const bool has_u = is_finite_bound(problem.col_upper[j]);
    if (!has_l && !has_u) continue;                                   // free
    if (has_l && has_u && problem.col_lower[j] == problem.col_upper[j]) continue;  // fixed

    Real xi = 100.0 * eps_d + kPsi * std::fabs(problem.c[j]);         // step 1
    const Real mu = uniform();
    xi = (has_u ? -0.5 : 0.5) * xi * (1.0 + mu);                     // step 2
    const std::size_t nu = csc.slice_end(j) - csc.slice_begin(j);
    xi *= kWeights[std::min<std::size_t>(std::max<std::size_t>(nu, 1), 10) - 1];  // step 3
    for (int guard = 0; guard < 64 && std::fabs(xi) > xi_max; ++guard) xi *= 0.1;  // step 4
    for (int guard = 0; guard < 64 && std::fabs(xi) < xi_min; ++guard) xi *= 10.0;
    out[j] += xi;
  }
  return true;
}

/// The TRUE objective of a point the dual reported on perturbed costs.
Real true_objective(const model::CanonicalProblem& problem, const SimplexResult& r) {
  Real obj = 0.0;
  for (std::size_t j = 0; j < problem.num_cols(); ++j) obj += problem.c[j] * r.x[j];
  return obj;
}

void add_counts(SimplexResult& into, const SimplexResult& from) {
  into.iterations += from.iterations;
  into.refactorizations += from.refactorizations;
  into.basis_repairs += from.basis_repairs;
  into.bound_flips += from.bound_flips;
}

/// A status that settles the question about the model, as opposed to one that
/// only reports the solve ran out of something.
[[nodiscard]] bool is_verdict(core::SolverStatus status) {
  return status == core::SolverStatus::Optimal ||
         status == core::SolverStatus::Infeasible ||
         status == core::SolverStatus::Unbounded;
}

}  // namespace

core::Expected<SimplexResult> solve_simplex(const model::CanonicalProblem& problem,
                                            const model::Options& options,
                                            const Basis* warm_start) {
  if (options.simplex.method == model::Method::PrimalSimplex) {
    return solve_primal_simplex(problem, options, warm_start);
  }

  std::vector<Real> perturbed;
  const bool perturb = options.simplex.cost_perturbation &&
                       make_cost_perturbation(problem, options, perturbed);

  auto dual = solve_dual_simplex(problem, options, warm_start,
                                 perturb ? &perturbed : nullptr);
  if (!dual.has_value()) return dual;

  if (perturb) {
    // Removing the perturbation, as Koberstein section 6.3.1 does it.
    //
    // INFEASIBLE needs nothing: a Farkas certificate involves only A, b and the
    // bounds, never the costs, so the verdict stands for the true problem.
    if (dual->status == core::SolverStatus::Infeasible) {
      dual->objective = true_objective(problem, *dual);
      return dual;
    }
    // OPTIMAL for the perturbed costs means the basis is primal feasible; with
    // the true costs restored it may no longer be dual feasible, so "we switch
    // to the primal simplex method (to be more precise: primal phase II) to
    // restore dual feasibility ... Usually, only few iterations are necessary".
    // UNBOUNDED depends on the costs and must be confirmed on the true ones --
    // the thesis restarts to "confirm unboundedness"; the primal from the same
    // basis does that directly. A non-verdict goes the same way when cleanup is
    // allowed, exactly as it does unperturbed.
    const bool finish = dual->status == core::SolverStatus::Optimal ||
                        dual->status == core::SolverStatus::Unbounded ||
                        options.simplex.primal_cleanup;
    if (finish) {
      auto primal = solve_primal_simplex(problem, options, &dual->basis);
      if (primal.has_value()) {
        add_counts(*primal, *dual);
        return primal;
      }
    }
    // No cleanup: hand back the dual's point, but never its PERTURBED
    // objective, and never a verdict the true costs have not confirmed.
    dual->objective = true_objective(problem, *dual);
    if (dual->status == core::SolverStatus::Optimal ||
        dual->status == core::SolverStatus::Unbounded) {
      dual->status = core::SolverStatus::NotConverged;
    }
    return dual;
  }

  if (is_verdict(dual->status) || !options.simplex.primal_cleanup) return dual;

  // No verdict from the dual. Hand its basis to the primal, which is not
  // subject to the artificial bounds that trapped it. The dual's endpoint is
  // primal feasible for the model (it is feasible for the narrower boxed
  // problem, and the true bounds contain the box), so this normally costs no
  // phase-1 pivots at all.
  auto primal = solve_primal_simplex(problem, options, &dual->basis);
  if (!primal.has_value()) return dual;  // keep the better-understood outcome
  if (!is_verdict(primal->status) && is_verdict(dual->status)) return dual;

  // Iteration counts are reported cumulatively: the work really was done, and
  // attributing only the cleanup's pivots would understate the solve.
  primal->iterations += dual->iterations;
  primal->refactorizations += dual->refactorizations;
  primal->basis_repairs += dual->basis_repairs;
  primal->bound_flips += dual->bound_flips;
  return primal;
}

}  // namespace sovsolve::solver::simplex
