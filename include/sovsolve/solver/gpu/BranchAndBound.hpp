// Module 22 (added post-v3, see docs/spec/module.txt): branch-and-bound over the
// existing continuous LP/QP relaxation solver.
//
// Not in the original locked docs/spec/module.txt spec, which predates any MILP
// requirement. Recorded there as its own module rather than folded silently
// into an existing one, matching the file's own "no module silently
// implements another module's job" rule.
//
// Design: `solve_problem` (Solve.hpp) is reused COMPLETELY UNCHANGED as the
// per-node relaxation engine -- every fix this session (initialization,
// step-length degeneracy, the convergence/stall detector, the apply_step
// clamp) benefits every node for free, and this file adds zero new IPM
// math. A node is just the original Problem with tightened column bounds;
// solving it is exactly a call to solve_problem. The only genuinely new
// logic here is the search: best-first over relaxation bounds, most-
// fractional branching, and incumbent/pruning bookkeeping.
//
// Known, explicit limitation (not attempted this pass): each node calls
// solve_problem cold -- there is no warm start from the parent's solution.
// Warm-starting an interior-point method across a bound change is itself a
// hard, still-researched problem (it is the reason production MILP solvers
// use simplex, which warm-starts trivially, for node relaxations instead of
// IPM). Cold-starting is correct, just not fast -- see the docs/spec/module.txt Module
// 22 entry for the honest scope statement.

#ifndef SOVSOLVE_SOLVER_GPU_BRANCH_AND_BOUND_HPP
#define SOVSOLVE_SOLVER_GPU_BRANCH_AND_BOUND_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"

namespace sovsolve::solver::gpu {

using core::Expected;
using model::Options;
using model::Problem;
using model::Solution;

/// The entry point callers (the CLI, tests) should use instead of calling
/// solve_problem directly: dispatches to solve_problem unchanged when
/// `problem` has no discrete columns, or runs branch-and-bound over it when
/// `problem.has_discrete()` is true.
///
/// Returns `UnsupportedFeature` for a model containing a SemiContinuous
/// column -- true semi-continuous branching ("0, or within [l,u]") is a
/// different search rule than the integer floor/ceil branching implemented
/// here, and silently treating it as an ordinary bounded column would
/// accept points the model does not actually allow. Same "report, never
/// silently drop" policy this codebase already applies to SOS sets
/// (Problem.hpp's `SosSet` doc comment).
[[nodiscard]] Expected<Solution> solve(const Problem& problem, const Options& options = {});

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_BRANCH_AND_BOUND_HPP
