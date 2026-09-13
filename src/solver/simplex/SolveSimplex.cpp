#include "sovsolve/solver/simplex/SolveSimplex.hpp"

#include "sovsolve/solver/simplex/DualSimplex.hpp"
#include "sovsolve/solver/simplex/PrimalSimplex.hpp"

namespace sovsolve::solver::simplex {
namespace {

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

  auto dual = solve_dual_simplex(problem, options, warm_start);
  if (!dual.has_value()) return dual;
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
