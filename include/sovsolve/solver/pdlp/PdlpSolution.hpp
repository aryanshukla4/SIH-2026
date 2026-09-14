// Module 24: pack a PDLP result into the canonical-space `model::Solution`
// the existing postsolve already consumes.
//
// Like Module 23, there is no new postsolve code: `recover_solution()` inverts
// canonicalization, presolve and scaling from `x`, `s`, `y`, `z`, `v` alone.
//
// One thing PDLP has to do that the simplex does not. The simplex carries a
// logical variable per row, so the slack `s` is literally a component of its
// working vector. PDLP's iterate is only `(x, y)` -- there are no logicals at
// all -- so the slack has to be RECOVERED from the row activity:
//
//     s_k = b_i - (A x)_i      for inequality row  i = num_equality + k
//
// which is the unique `s` satisfying that row exactly. The consequence is that
// the primal residual on inequality rows is then identically zero by
// construction, and the real infeasibility shows up instead as `s_k < 0`, i.e.
// as a bound violation. That is not a trick to flatter the metrics -- it is
// where a first-order method's error genuinely lives, since PDHG projects `x`
// onto its box every iteration but nothing forces `A_I x <= b_I` until
// convergence. `solution.quality.max_bound_violation` is therefore the number
// to read for a PDLP run, not `primal_infeasibility`.

#ifndef SOVSOLVE_SOLVER_PDLP_PDLP_SOLUTION_HPP
#define SOVSOLVE_SOLVER_PDLP_PDLP_SOLUTION_HPP

#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/solver/pdlp/Pdlp.hpp"

namespace sovsolve::solver::pdlp {

/// Build the canonical-space `Solution` for `result`, with quality metrics
/// measured by the same shared routine the simplex uses
/// (solver/SolutionQuality.hpp), so the engines' reported numbers are
/// comparable rather than merely similarly named.
[[nodiscard]] model::Solution to_canonical_solution(
    const model::CanonicalProblem& problem, const PdlpResult& result);

}  // namespace sovsolve::solver::pdlp

#endif  // SOVSOLVE_SOLVER_PDLP_PDLP_SOLUTION_HPP
