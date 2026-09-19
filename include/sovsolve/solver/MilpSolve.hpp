// Module 28: branch-and-bound on the warm-started dual simplex, with
// reliability branching.
//
// WHY A SECOND BRANCH-AND-BOUND. Module 22 (gpu/BranchAndBound.cu) solves
// every node with the GPU interior-point method, starting cold, and its own
// header names the problem: "production MILP solvers use simplex, which
// warm-starts trivially, for node relaxations instead of IPM". A child differs
// from its parent by one bound, so the parent's optimal basis stays DUAL
// feasible for the child -- the costs have not changed -- and the dual simplex
// resumes from it, usually in a handful of pivots. An interior-point method
// has no basis to resume from.
//
// That is also what makes the branching rule possible. Both sources below
// define strong branching as "perform at most gamma DUAL SIMPLEX iterations"
// on each child, warm-started from the parent's basis -- a probe that means
// nothing on an interior-point method.
//
// SOURCES, transcribed, nothing derived:
//
//   [AKM]  T. Achterberg, T. Koch, A. Martin, "Branching rules revisited",
//          Operations Research Letters 33 (2005) -- ZIB report ZR-04-13.
//   [CIP]  T. Achterberg, "Constraint Integer Programming", PhD thesis,
//          TU Berlin (2007), chapter 5.
//
//   score     [CIP] (5.2), the product  max{q-, eps} * max{q+, eps},
//             which [CIP] section 5.11 measures as better than [AKM] (3)'s
//             weighted sum.
//   pseudocost [AKM] (4)-(5) / [CIP] (5.3)-(5.4): mean objective gain per unit
//             of rounding, over past feasible branchings; an uninitialized
//             direction takes the average of the initialized ones, or 1.
//   reliability [AKM] Algorithm 3 / [CIP] Algorithm 5.2: strong-branch the
//             candidates, best pseudocost score first, whose pseudocosts are
//             unreliable (min(eta-, eta+) < eta_rel), updating the
//             pseudocosts with what the probes find; stop after `lambda`
//             score updates without a new best, or `kappa` candidates.
//   parameters [CIP] sections 5.4 and 5.7 -- see MilpOptions.
//   node selection [CIP] chapter 6, SCIP's default (section 6.6): best
//             estimate search with plunging, and every 10th plunge a
//             best-bound leaf instead. Estimate (section 6.4, Forrest et al.)
//             `e_Q = c_Q + sum min{Psi- f-, Psi+ f+}`; plunge limits and the
//             0.25 gap abort from section 6.3; child order by Martin's rule
//             (section 6.1). Pure best first (section 6.2) remains selectable.
//   heuristics [CIP] chapter 9: simple rounding (9.1.2) on every node LP,
//             and the generic dive of Algorithm 9.1 with one-level
//             backtracking, rotating fractionality, coefficient, line search
//             and pseudocost diving (9.2.1, 9.2.2, 9.2.4, 9.2.5) under the 5%
//             iteration quota. No domain propagation (step 6): this engine
//             has none. At the root, the objective feasibility pump (9.3.3,
//             Berthold 2006 Algorithm 3 stages 1-2, SCIP's +-1/0 costs for
//             general integers) and RENS (9.1.1) on this same search.
//   propagation [CIP] chapter 7: Algorithm 7.1 on every row at every node
//             and in dives, the objective cutoff as one more row (7.6), and
//             root reduced cost strengthening of the global bounds (7.7).
//
// WHAT THE SEARCH WORKS ON. The model is canonicalized and scaled ONCE, at the
// root, and a node is just a set of canonical column bounds. Canonicalization
// never shifts, reflects or splits a column (Transform.hpp: every column is
// kept or substituted out), and scaling multiplies each by one factor, so an
// original integer column maps to exactly one canonical column with
// `x_original = s * x_canonical`. General LP presolve is deliberately NOT run:
// it merges parallel columns and substitutes, and neither respects
// integrality. The MILP-specific row tightening Module 22 uses
// (MilpPresolve.hpp) runs on the original model first, as there.
//
// This also removes a cost Module 22 pays on every node -- `solve_problem`
// canonicalizes, presolves and scales afresh per call.
//
// A SOUNDNESS FIX over Module 22, made while porting. There, a node whose
// relaxation did not converge is dropped ("pruning here is the SAFE choice"),
// and the search can still report Optimal. It is not safe: an unexplored
// subtree may hold the optimum. Here such nodes are counted, and any count
// above zero forbids an Optimal verdict -- the run reports NotConverged with
// its incumbent instead.

#ifndef SOVSOLVE_SOLVER_MILP_SOLVE_HPP
#define SOVSOLVE_SOLVER_MILP_SOLVE_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"

namespace sovsolve::solver {

/// Where the effort went. The branching rules trade node count against work
/// per node, so comparing them on nodes alone would flatter strong branching;
/// these make the trade visible.
struct MilpStatistics {
  std::size_t nodes = 0;
  std::size_t node_lp_iterations = 0;
  std::size_t strong_branching_probes = 0;
  std::size_t strong_branching_iterations = 0;
  /// Probes that proved one side infeasible, letting the node be branched on
  /// that variable with the dead side discarded ([CIP] section 5.4).
  std::size_t strong_branching_fixings = 0;
  /// Nodes whose relaxation did not converge. Any nonzero value forbids an
  /// Optimal verdict -- see the header.
  std::size_t unreliable_nodes = 0;
  /// Root cover/GCD cuts appended before the search.
  std::size_t root_cuts = 0;
  /// Points that improved the incumbent (integral node LPs and heuristics
  /// alike), and the node count
  /// at which the first one was found. Node selection ([CIP] chapter 6) is
  /// judged on these as much as on the bound.
  std::size_t incumbents = 0;
  std::size_t first_incumbent_node = 0;
  /// Nodes chosen by continuing a plunge (a child or sibling of the node just
  /// processed) rather than from the leaf queue.
  std::size_t plunge_steps = 0;
  /// [CIP] chapter 9: dives started, the LP iterations they spent (NOT part
  /// of node_lp_iterations), and incumbents found by each heuristic.
  std::size_t dives = 0;
  std::size_t dive_lp_iterations = 0;
  std::size_t dive_solutions = 0;
  std::size_t rounding_solutions = 0;
  /// Feasibility pump rounds (LP solves) and whether it found a point; RENS
  /// sub-MIP nodes and whether it improved the incumbent. Their LP work is
  /// NOT in node_lp_iterations.
  std::size_t pump_rounds = 0;
  std::size_t pump_lp_iterations = 0;
  std::size_t pump_solutions = 0;
  std::size_t rens_nodes = 0;
  std::size_t rens_solutions = 0;
  /// [CIP] chapter 7: bounds tightened by propagation (local, all nodes and
  /// dives), nodes it proved empty before their LP was solved, and global
  /// bounds tightened by root reduced cost strengthening.
  std::size_t propagation_tightenings = 0;
  std::size_t propagation_cutoffs = 0;
  std::size_t redcost_tightenings = 0;
  /// [CIP] 8.8: LOCAL bounds tightened at a node by its own LP's reduced
  /// costs, inherited by its subtree.
  std::size_t local_redcost_tightenings = 0;
};

/// Solves a mixed-integer LINEAR program. A model with no discrete columns is
/// handed to `solve_lp` unchanged. Quadratic objectives and semi-continuous
/// columns are refused with `UnsupportedFeature`.
[[nodiscard]] core::Expected<model::Solution> solve_milp(const model::Problem& problem,
                                                         const model::Options& options,
                                                         MilpStatistics* statistics = nullptr);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_MILP_SOLVE_HPP
