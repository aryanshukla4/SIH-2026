#include "sovsolve/solver/pdlp/PdlpSolution.hpp"

#include <cstddef>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/solver/SolutionQuality.hpp"

namespace sovsolve::solver::pdlp {

model::Solution to_canonical_solution(const model::CanonicalProblem& problem,
                                      const PdlpResult& result) {
  const std::size_t m = problem.num_rows();
  const std::size_t n = problem.num_cols();
  const std::size_t equalities = problem.num_equality;
  const std::size_t inequalities = problem.num_inequality_rows();

  model::Solution solution;
  solution.status = result.status;
  solution.iterations = result.iterations;

  solution.x.resize(n);
  for (std::size_t j = 0; j < n; ++j) solution.x[j] = result.x[j];

  solution.y.resize(m);
  for (std::size_t i = 0; i < m; ++i) solution.y[i] = result.y[i];

  // Slacks are recovered from row activity -- see PdlpSolution.hpp for why
  // this moves the visible error from `primal_infeasibility` to
  // `max_bound_violation`.
  solution.s.resize(inequalities);
  if (inequalities > 0) {
    const auto& csr = problem.A.csr;
    for (std::size_t k = 0; k < inequalities; ++k) {
      const std::size_t i = equalities + k;
      Real activity = 0.0;
      for (std::size_t idx = csr.slice_begin(i); idx < csr.slice_end(i); ++idx) {
        activity +=
            csr.values()[idx] * solution.x[static_cast<std::size_t>(csr.indices()[idx])];
      }
      solution.s[k] = problem.b[i] - activity;
    }
  }

  // Same split as the simplex: a reduced cost is non-negative on the lower
  // bound's dual and non-positive on the upper's, and at most one is nonzero.
  solution.z.resize(n);
  solution.v.resize(n);
  for (std::size_t j = 0; j < n; ++j) {
    const Real d = result.reduced_cost[j];
    solution.z[j] = d > 0.0 ? d : 0.0;
    solution.v[j] = d < 0.0 ? -d : 0.0;
  }

  solution.objective = problem.objective(solution.x.span());
  compute_solution_quality(problem, solution);
  return solution;
}

}  // namespace sovsolve::solver::pdlp
