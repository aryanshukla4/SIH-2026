// Root cover and GCD cuts for branch-and-bound, shared by Module 22
// (gpu/BranchAndBound.cu) and Module 28 (MilpSolve.cpp).
//
// MOVED, not rewritten. This code lived in BranchAndBound.cu's anonymous
// namespace, wired to the GPU interior-point solver for its root relaxations --
// although nothing in the cut logic itself touches the GPU. Module 28 needed
// the same cuts, and a second copy would have been two implementations of one
// separation routine free to drift apart. So the routine is here, and each
// caller injects its own LP solver for the root relaxation: Module 22 passes
// solve_problem, Module 28 passes the dual simplex. The doc comments below are
// the originals, including the measurements on markshare_4_0 that motivated
// the cross-row aggregation.
//
// "Cut-and-branch": cuts are separated against the ROOT relaxation only, in a
// bounded number of rounds, and then held fixed for the whole search. Every cut
// is globally valid, so adding them once benefits every node.

#ifndef SOVSOLVE_SOLVER_MILP_CUTS_HPP
#define SOVSOLVE_SOLVER_MILP_CUTS_HPP

#include <cstddef>
#include <functional>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"

namespace sovsolve::solver {

/// What the cutting loop added.
struct CutStatistics {
  std::size_t rounds = 0;
  std::size_t cuts = 0;
};

/// Solves the LP relaxation of the problem it is handed and returns the
/// solution in the ORIGINAL space -- separation reads `x` directly.
using RelaxationSolver =
    std::function<core::Expected<model::Solution>(const model::Problem&)>;

/// Appends violated knapsack cover and aggregated GCD cuts to `problem` as new
/// rows, re-solving the relaxation with `solve` between rounds. Stops when a
/// round finds nothing, or at the round cap. A root relaxation that is not
/// Optimal stops the loop quietly: the search that follows handles that case
/// itself, and cuts cannot safely be separated from an unreliable point.
[[nodiscard]] core::Status add_root_cuts(model::Problem& problem, const RelaxationSolver& solve,
                                         CutStatistics* stats = nullptr);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_MILP_CUTS_HPP
