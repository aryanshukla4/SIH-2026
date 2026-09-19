#include "sovsolve/solver/gpu/BranchAndBound.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <queue>
#include <vector>

#include <algorithm>
#include <utility>

#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/solver/MilpCuts.hpp"
#include "sovsolve/solver/MilpPresolve.hpp"
#include "sovsolve/solver/gpu/Solve.hpp"

namespace sovsolve::solver::gpu {

namespace {

using core::Real;
using core::RealVector;
using core::SolverStatus;
using core::VarType;

/// One pending subproblem: the original Problem with these column bounds
/// instead of its own. `bound` is the relaxation objective of the PARENT
/// this node was split from -- a valid bound on every descendant's
/// objective, since tightening a column's bounds can only make a
/// minimization's (or maximization's) optimum worse or equal, never better.
/// Meaningless for the root node (there is no parent), which is never
/// compared against anything before it is itself solved.
///
/// Move-only, like Problem/RealVector -- col_lower/col_upper are RealVector.
struct Node {
  RealVector col_lower;
  RealVector col_upper;
  Real bound;
  /// `minimize ? bound : -bound` -- orders the queue so the most PROMISING
  /// node (smallest bound for a minimization, largest for a maximization)
  /// is always popped first, without templating the comparator on sense.
  Real priority;
  /// The PARENT node's own converged, ORIGINAL-space solution -- passed to
  /// solve_problem as this node's warm-start hint (Solve.hpp,
  /// model::forward_map_to_canonical_hint). Empty for the root (no parent
  /// to warm-start from), which solve_problem reads as "no hint."
  RealVector warm_x;
};

struct NodeOrder {
  // std::priority_queue is a max-heap by default; negate the sense so the
  // node with the smallest `priority` (the most promising one) comes out
  // first -- best-first search.
  bool operator()(const Node& a, const Node& b) const noexcept { return a.priority > b.priority; }
};

[[nodiscard]] bool better(Real candidate, Real current, bool minimize) noexcept {
  return minimize ? candidate < current : candidate > current;
}

/// The most-fractional discrete column of `x` (the one closest to a
/// half-integer), or `n` if every discrete column is already within `tol`
/// of an integer. Simple, standard "most-fractional" branching -- not the
/// pseudocost/strong-branching schemes production solvers layer on top,
/// which pick the branching variable using historical impact rather than
/// just how fractional it looks right now.
[[nodiscard]] std::size_t most_fractional_column(const Problem& problem, const RealVector& x,
                                                  Real tol) {
  const std::size_t n = problem.col_type.size();
  std::size_t best_col = n;
  Real worst_distance = tol;
  for (std::size_t j = 0; j < n; ++j) {
    if (problem.col_type[j] == VarType::Continuous) continue;
    const Real frac = x[j] - std::floor(x[j]);
    const Real distance_from_integer = std::min(frac, 1.0 - frac);
    if (distance_from_integer > worst_distance) {
      worst_distance = distance_from_integer;
      best_col = j;
    }
  }
  return best_col;
}

}  // namespace

Expected<Solution> solve(const Problem& problem, const Options& options) {
  if (!problem.has_discrete()) return solve_problem(problem, options);

  for (const auto t : problem.col_type) {
    if (t == VarType::SemiContinuous) {
      return core::make_error(
          core::ErrorCode::UnsupportedFeature,
          "branch-and-bound: semi-continuous columns are not supported -- "
          "their branching rule (0, or within [l,u]) differs from the "
          "integer floor/ceil branching implemented here");
    }
  }

  const auto& milp = options.milp;
  const bool minimize = problem.sense == core::ObjSense::Minimize;
  const auto start_time = std::chrono::steady_clock::now();

  // One deep copy of the whole Problem for the entire search (A/Q/c never
  // change across nodes, only col_lower/col_upper do) -- see Problem::clone()
  // and BranchAndBound.hpp's doc comment on why this, not a clone per node.
  Problem working = problem.clone();

  // GCD/knapsack row tightening (MilpPresolve.hpp), run once on the root --
  // it only touches row_lower/row_upper, which `working` never changes
  // again after this (only col_lower/col_upper do, per node), so this one
  // pass benefits every node in the search for free. A provable
  // infeasibility here is caught before a single relaxation is solved.
  // Run BEFORE GCD tightening: stripping an absorbing continuous singleton
  // out of an equality row (markshare_4_0.mps's exact shape -- every row is
  // 30 binary columns plus one non-negative continuous "deviation" column)
  // is what lets that row newly qualify as all-discrete, which is what
  // tighten_integer_rows (and cover-cut separation below) require.
  std::vector<AbsorbingColumnElimination> absorbing_eliminations;
  eliminate_equality_row_absorbing_singletons(working, absorbing_eliminations);
  if (const auto st = tighten_integer_rows(working); !st.ok()) return st.error();

  // Root cover and GCD cuts -- moved to MilpCuts.{hpp,cpp} so Module 28 shares
  // the one implementation. This caller supplies the GPU interior-point solver
  // for the root relaxations, as before.
  if (const auto st = add_root_cuts(
          working, [&options](const Problem& p) { return solve_problem(p, options); });
      !st.ok()) {
    return st.error();
  }

  std::priority_queue<Node, std::vector<Node>, NodeOrder> pending;
  // The root's bound/priority are placeholders: nothing is compared against
  // them before the root itself is popped and solved below.
  pending.push(Node{problem.col_lower.clone(), problem.col_upper.clone(), 0.0, 0.0, RealVector()});

  bool have_incumbent = false;
  Solution incumbent;
  std::size_t nodes_explored = 0;
  bool proved_optimal_or_gap_closed = false;
  Real final_best_bound = 0.0;

  while (!pending.empty()) {
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    if (nodes_explored >= milp.node_limit || elapsed >= milp.time_limit_seconds) break;

    // std::priority_queue has no "extract top" -- top() only returns a const
    // reference. Moving out of it just before pop() is the standard,
    // well-known workaround: that heap slot is destroyed by pop() on the
    // next line regardless, so nothing else ever observes the moved-from
    // state.
    Node node = std::move(const_cast<Node&>(pending.top()));
    pending.pop();
    ++nodes_explored;

    // Best-first pruning/termination: every node still in the queue has a
    // priority no better than this one (heap property), so once THIS one
    // can no longer beat the incumbent -- either strictly, or within the
    // accepted mip gap -- nothing left in the queue can either. That is
    // what turns "stop searching" into a proof of optimality (up to the
    // gap), not just a heuristic cutoff.
    if (have_incumbent && nodes_explored > 1) {
      const Real gap =
          std::fabs(incumbent.objective - node.bound) / (1.0 + std::fabs(incumbent.objective));
      if (!better(node.bound, incumbent.objective, minimize) || gap < milp.gap_tolerance) {
        proved_optimal_or_gap_closed = true;
        final_best_bound = node.bound;
        break;
      }
    }

    working.col_lower = node.col_lower.clone();
    working.col_upper = node.col_upper.clone();

    const RealVector* hint = node.warm_x.empty() ? nullptr : &node.warm_x;
    auto relaxation = solve_problem(working, options, hint);
    if (!relaxation.has_value()) {
      const auto code = relaxation.error().code;
      if (code == core::ErrorCode::PrimalInfeasible) continue;  // pruned: subtree is infeasible
      // An unbounded relaxation on a non-root node is a genuine edge case
      // this pass does not attempt to reason through fully -- a bounded
      // MILP CAN have an unbounded LP relaxation if the unbounded direction
      // requires non-integer values the whole way out. Treated
      // conservatively here as "the search cannot proceed," matching how
      // the codebase already refuses to convert an unresolved case into a
      // silently wrong answer elsewhere (Presolver.cpp's empty-row check).
      return relaxation.error();
    }

    if (relaxation->status != SolverStatus::Optimal) {
      // NotConverged/MaxIterations/etc: this node's own bound cannot be
      // trusted (see ConvergenceChecker.hpp -- these statuses cover "still
      // running" and "gave up" alike). Pruning here is the SAFE choice, not
      // a shortcut: using an unreliable bound to cut off a subtree could
      // discard the true optimum. It is a real, honest limitation -- a
      // hard-to-solve relaxation makes its whole subtree unexplored rather
      // than retried -- recorded here rather than silently accepted.
      continue;
    }

    if (have_incumbent && !better(relaxation->objective, incumbent.objective, minimize)) continue;

    const std::size_t frac_col =
        most_fractional_column(problem, relaxation->x, milp.integer_tolerance);
    if (frac_col == problem.col_type.size()) {
      // Every discrete column is already integral: a genuine MILP-feasible
      // point, and by construction the best one this subtree can produce.
      have_incumbent = true;
      incumbent = std::move(*relaxation);
      continue;
    }

    const Real v = relaxation->x[frac_col];
    const Real floor_v = std::floor(v);
    const Real ceil_v = std::ceil(v);
    const Real bound = relaxation->objective;

    Node left{node.col_lower.clone(), node.col_upper.clone(), bound, 0.0,
              relaxation->x.clone()};
    left.col_upper[frac_col] = std::min(left.col_upper[frac_col], floor_v);
    left.priority = minimize ? left.bound : -left.bound;

    Node right{node.col_lower.clone(), node.col_upper.clone(), bound, 0.0,
               relaxation->x.clone()};
    right.col_lower[frac_col] = std::max(right.col_lower[frac_col], ceil_v);
    right.priority = minimize ? right.bound : -right.bound;

    if (left.col_lower[frac_col] <= left.col_upper[frac_col]) pending.push(std::move(left));
    if (right.col_lower[frac_col] <= right.col_upper[frac_col]) pending.push(std::move(right));
  }

  Solution result;
  if (have_incumbent) {
    result = std::move(incumbent);

    // Recover each absorbing column's TRUE value from its row equation,
    // using the FINAL solution's other columns -- the empty-column
    // presolve rule that fixed it only guaranteed its OWN bound, not that
    // it satisfies the row (see AbsorbingColumnElimination's doc comment).
    // `working`'s row structure for these ORIGINAL rows is unchanged by
    // per-node bound tightening or by cover cuts (which only ever APPEND
    // new rows), so indexing it here after the whole search is still
    // exactly the row this elimination came from.
    const auto& csr = working.A.csr;
    for (const auto& elim : absorbing_eliminations) {
      Real other_terms = 0.0;
      for (auto k = csr.slice_begin(elim.row); k < csr.slice_end(elim.row); ++k) {
        const auto col2 = static_cast<std::size_t>(csr.indices()[k]);
        other_terms += csr.values()[k] * result.x[col2];
      }
      result.x[elim.col] = (elim.b_i - other_terms) / elim.a_ij;
    }
  }

  if (proved_optimal_or_gap_closed) {
    result.status = SolverStatus::Optimal;
    result.best_bound = final_best_bound;
  } else if (pending.empty()) {
    // Search space fully exhausted with no gap-closing break above: the
    // incumbent (if any) IS the proven bound.
    result.status = have_incumbent ? SolverStatus::Optimal : SolverStatus::Infeasible;
    result.best_bound = have_incumbent ? result.objective : 0.0;
  } else {
    // Node/time limit hit, search space not exhausted: report the best
    // incumbent found so far (or none) as NotConverged -- NEVER Infeasible.
    // Same "stagnation is not a verdict about the model" principle
    // module.txt's Module 18 applies to the continuous solver's own stall
    // detection. The best remaining bound anywhere in the tree is still
    // informative -- it shows the true, still-open gap -- so report it
    // rather than 0.
    result.status = SolverStatus::NotConverged;
    result.best_bound = pending.top().bound;
  }
  result.nodes_explored = nodes_explored;
  result.from_best_iterate = result.status != SolverStatus::Optimal;
  result.solve_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  return Expected<Solution>(std::move(result));
}

}  // namespace sovsolve::solver::gpu
