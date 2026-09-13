#include "sovsolve/solver/simplex/SimplexSolution.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "sovsolve/core/Types.hpp"

namespace sovsolve::solver::simplex {
namespace {

using core::is_finite_bound;

Real relative(Real absolute, Real scale) { return absolute / (1.0 + scale); }

Real inf_norm(const core::RealVector& v) {
  Real worst = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) worst = std::fmax(worst, std::fabs(v[i]));
  return worst;
}

}  // namespace

model::Solution to_canonical_solution(const model::CanonicalProblem& problem,
                                      const SimplexResult& result) {
  const std::size_t m = problem.num_rows();
  const std::size_t n = problem.num_cols();
  const std::size_t equalities = problem.num_equality;
  const std::size_t inequalities = problem.num_inequality_rows();

  model::Solution solution;
  solution.status = result.status;
  solution.iterations = result.iterations;

  solution.x.resize(n);
  for (std::size_t j = 0; j < n; ++j) solution.x[j] = result.x[j];

  // `s` covers inequality rows only -- equality rows carry no slack, and their
  // logical is fixed at zero rather than represented here (FORMULATION.md
  // section 3). Its element `k` belongs to canonical row `num_equality + k`.
  solution.s.resize(inequalities);
  for (std::size_t k = 0; k < inequalities; ++k) {
    solution.s[k] = result.x[n + equalities + k];
  }

  solution.y.resize(m);
  for (std::size_t i = 0; i < m; ++i) solution.y[i] = result.y[i];

  // A reduced cost splits into the two bound duals by sign; both are
  // non-negative and at most one is nonzero, which is the complementarity the
  // IPM spends its whole run approaching and a vertex satisfies exactly.
  solution.z.resize(n);
  solution.v.resize(n);
  for (std::size_t j = 0; j < n; ++j) {
    const Real d = result.reduced_cost[j];
    solution.z[j] = d > 0.0 ? d : 0.0;
    solution.v[j] = d < 0.0 ? -d : 0.0;
  }

  solution.objective = problem.objective(solution.x.span());

  // ---- measured quality, not asserted ----------------------------------
  // Primal residual: A_E x - b_E on equality rows, A_I x + s - b_I on the
  // rest. The iteration maintains this identically by construction, so a
  // nonzero here is accumulated rounding in the basis factorization and is
  // exactly what a caller comparing engines wants to see.
  Real primal_residual = 0.0;
  const auto& csr = problem.A.csr;
  for (std::size_t i = 0; i < m; ++i) {
    Real activity = 0.0;
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      activity += csr.values()[k] * solution.x[static_cast<std::size_t>(csr.indices()[k])];
    }
    if (i >= equalities) activity += solution.s[i - equalities];
    primal_residual = std::fmax(primal_residual, std::fabs(activity - problem.b[i]));
  }

  // Dual residual: c - A'y - z + v. Zero by construction for every column,
  // basic and nonbasic alike, because `d` is defined as `c - Ahat' y`.
  Real dual_residual = 0.0;
  const auto& csc = problem.A.csc;
  for (std::size_t j = 0; j < n; ++j) {
    Real dot = 0.0;
    for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      dot += csc.values()[k] * solution.y[static_cast<std::size_t>(csc.indices()[k])];
    }
    const Real stationarity = problem.c[j] - dot - solution.z[j] + solution.v[j];
    dual_residual = std::fmax(dual_residual, std::fabs(stationarity));
  }

  // Bound violation against the model's own bounds, including the slacks'
  // implicit `s >= 0`.
  Real bound_violation = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(problem.col_lower[j])) {
      bound_violation =
          std::fmax(bound_violation, problem.col_lower[j] - solution.x[j]);
    }
    if (is_finite_bound(problem.col_upper[j])) {
      bound_violation =
          std::fmax(bound_violation, solution.x[j] - problem.col_upper[j]);
    }
  }
  for (std::size_t k = 0; k < inequalities; ++k) {
    bound_violation = std::fmax(bound_violation, -solution.s[k]);
  }

  // Dual objective, with the finite-bound terms the primal-dual gap needs.
  // `b'y` alone leaves a permanent unclosable gap on any model with an active
  // finite bound -- the same correction Solve.cu's `dual_objective()` carries,
  // restated here rather than shared because that one reads a SolverState.
  Real dual_objective = 0.0;
  for (std::size_t i = 0; i < m; ++i) dual_objective += problem.b[i] * solution.y[i];
  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(problem.col_lower[j])) {
      dual_objective += problem.col_lower[j] * solution.z[j];
    }
    if (is_finite_bound(problem.col_upper[j])) {
      dual_objective -= problem.col_upper[j] * solution.v[j];
    }
  }

  Real complementarity = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(problem.col_lower[j])) {
      complementarity = std::fmax(
          complementarity, std::fabs(solution.z[j] * (solution.x[j] - problem.col_lower[j])));
    }
    if (is_finite_bound(problem.col_upper[j])) {
      complementarity = std::fmax(
          complementarity, std::fabs(solution.v[j] * (problem.col_upper[j] - solution.x[j])));
    }
  }
  for (std::size_t k = 0; k < inequalities; ++k) {
    complementarity =
        std::fmax(complementarity, std::fabs(solution.y[equalities + k] * solution.s[k]));
  }

  solution.quality.primal_infeasibility = relative(primal_residual, inf_norm(problem.b));
  solution.quality.dual_infeasibility = relative(dual_residual, inf_norm(problem.c));
  solution.quality.relative_gap =
      relative(std::fabs(solution.objective - dual_objective), std::fabs(solution.objective));
  solution.quality.complementarity = complementarity;
  solution.quality.max_bound_violation = std::fmax(bound_violation, 0.0);
  return solution;
}

}  // namespace sovsolve::solver::simplex
