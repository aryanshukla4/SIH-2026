// The whole pipeline, start to finish: canonicalize -> scale -> initialize
// -> [predictor-corrector iterate + check convergence] -> reconstruct.
//
// Lives under gpu/ for the same reason PredictorCorrector.hpp does -- it
// calls gpu::run_iteration every loop pass. Unlike PredictorCorrector, it
// also calls host-only functions (model::canonicalize, solver::initialize,
// solver::ConvergenceChecker, solver::reconstruct_solution), so
// sovsolve_solver_gpu links sovsolve_solver for this one file. That is a
// ONE-WAY edge -- sovsolve_solver does not, and must not, link back
// (Regularization.hpp's header comment explains why a cycle would be a
// problem) -- and sovsolve_solver_gpu already requires the CUDA toolkit to
// exist at all, so depending on the always-available host library costs it
// nothing it doesn't already have.
//
// Solution mapping itself -- primal/dual recovery, sign flips, reduced-cost
// reconstruction for substituted columns, bound-violation checking against
// the ORIGINAL problem -- is NOT reimplemented here. It already exists and
// is tested, as part of the ingestion layer: model::recover_solution()
// (Canonicalizer.cpp) does all of it, and solver::reconstruct_solution()
// (SolutionReconstructor.hpp) is a thin wrapper over it. This file's only
// new responsibility is the missing piece between them: packaging a
// SolverState (the solve loop's own iterate) into a canonical-space
// model::Solution, and running the loop that produces one.

#ifndef SOVSOLVE_SOLVER_GPU_SOLVE_HPP
#define SOVSOLVE_SOLVER_GPU_SOLVE_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"

namespace sovsolve::solver::gpu {

using core::Expected;
using model::Options;
using model::Problem;
using model::Solution;

/// `warm_start_x`, when non-null, is a point in the ORIGINAL problem's
/// variable space (typically a previously reconstructed Solution's `x`) --
/// forward-mapped internally (model::forward_map_to_canonical_hint) into
/// this call's OWN freshly-canonicalized space before Initializer uses it.
/// Module 22's branch-and-bound is the intended caller: it seeds a child
/// node from its parent's solution instead of bound-midpoint everywhere.
/// Ignored (nullptr) for an ordinary top-level solve.
///
/// Solves `problem` end to end. On stall or the iteration limit, the
/// returned Solution is the best iterate seen (by the worst of the three
/// FORMULATION.md 9 relative criteria), not the last one, with
/// `from_best_iterate` set accordingly -- never a status of Infeasible from
/// here (that verdict, when it happens, comes from model::canonicalize()
/// failing before this function's loop ever starts).
[[nodiscard]] Expected<Solution> solve_problem(const Problem& problem,
                                                const Options& options = {},
                                                const core::RealVector* warm_start_x = nullptr);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_SOLVE_HPP
