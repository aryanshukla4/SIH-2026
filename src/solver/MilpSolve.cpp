#include "sovsolve/solver/MilpSolve.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#include "sovsolve/solver/MilpCuts.hpp"
#include "sovsolve/solver/MilpPresolve.hpp"
#include "sovsolve/solver/Scaler.hpp"
#include "sovsolve/solver/SolutionReconstructor.hpp"
#include "sovsolve/solver/simplex/SimplexSolution.hpp"
#include "sovsolve/solver/simplex/SolveSimplex.hpp"

namespace sovsolve::solver {

namespace {

using core::Real;
using core::SolverStatus;
using model::BranchingRule;
using model::Options;
using model::Problem;
using model::Solution;
using simplex::Basis;
using simplex::SimplexResult;

constexpr Real kInf = std::numeric_limits<Real>::infinity();

// --------------------------------------------------------------------------
// The integer columns, as the search sees them
// --------------------------------------------------------------------------

/// An original integer column, located in canonical space. `x_original =
/// scale * x_canonical`, because canonicalization only ever keeps a column
/// and scaling divides its bounds by one factor (Scaler.cpp).
struct IntegerColumn {
  std::size_t canonical = 0;
  std::size_t original = 0;
  Real scale = 1.0;
};

/// [AKM] (4)-(5): the running sums and counts behind one column's
/// pseudocosts, per direction.
struct Pseudocost {
  Real sum_down = 0.0;
  Real sum_up = 0.0;
  Real count_down = 0.0;
  Real count_up = 0.0;
};

class PseudocostTable {
 public:
  explicit PseudocostTable(std::size_t columns) : entries_(columns) {}

  void record(std::size_t k, bool up, Real gain_per_unit) {
    if (!(gain_per_unit >= 0.0) || !std::isfinite(gain_per_unit)) return;
    Pseudocost& e = entries_[k];
    if (up) {
      e.sum_up += gain_per_unit;
      e.count_up += 1.0;
    } else {
      e.sum_down += gain_per_unit;
      e.count_down += 1.0;
    }
  }

  /// `psi`: the mean for this column, or -- [AKM] section 2.2 -- "the average
  /// of the initialized ... pseudocosts over all variables" if it has none,
  /// "set to 1 in the case that all ... are uninitialized".
  [[nodiscard]] Real value(std::size_t k, bool up) const {
    const Pseudocost& e = entries_[k];
    const Real count = up ? e.count_up : e.count_down;
    if (count > 0.0) return (up ? e.sum_up : e.sum_down) / count;
    return average(up);
  }

  /// `min{eta-, eta+}`, compared against `eta_rel`.
  [[nodiscard]] Real reliability(std::size_t k) const {
    return std::min(entries_[k].count_down, entries_[k].count_up);
  }

 private:
  /// The average over initialized columns of each column's OWN mean -- the
  /// quantity [AKM] names. Recomputed rather than cached, because a column's
  /// mean moves every time it is updated; O(columns) per call, against an LP
  /// solve per node.
  [[nodiscard]] Real average(bool up) const {
    Real sum = 0.0;
    Real columns = 0.0;
    for (const Pseudocost& e : entries_) {
      const Real count = up ? e.count_up : e.count_down;
      if (count > 0.0) {
        sum += (up ? e.sum_up : e.sum_down) / count;
        columns += 1.0;
      }
    }
    return columns > 0.0 ? sum / columns : 1.0;
  }

  std::vector<Pseudocost> entries_;
};

/// [CIP] (5.2): `max{q-, eps} * max{q+, eps}`.
[[nodiscard]] Real product_score(Real down, Real up, Real eps) {
  return std::max(down, eps) * std::max(up, eps);
}

// --------------------------------------------------------------------------
// Nodes
// --------------------------------------------------------------------------

/// Which branching created a node, so its own LP can update a pseudocost once
/// it is solved ([AKM] section 2.2: the gain is measured on the CHILD).
struct Origin {
  std::size_t column = 0;  ///< index into the IntegerColumn list
  bool up = false;
  Real fraction = 0.0;     ///< `f-` or `f+` of the variable at the parent
  Real parent_objective = 0.0;
  bool present = false;
};

struct Node {
  std::vector<Real> lower;
  std::vector<Real> upper;
  /// The parent's LP objective: a valid lower bound for this subtree, since a
  /// child's feasible set is a subset of its parent's.
  Real bound = 0.0;
  std::size_t depth = 0;
  /// The parent's optimal basis, shared by both children. Still DUAL feasible
  /// for either child -- only a bound moved -- which is why the dual simplex
  /// resumes from it.
  std::shared_ptr<const Basis> warm;
  Origin origin;
};

struct NodeOrder {
  // std::priority_queue is a max-heap; this makes the SMALLEST bound come out
  // first -- best-first search, the same order Module 22 uses.
  bool operator()(const Node& a, const Node& b) const noexcept { return a.bound > b.bound; }
};

// --------------------------------------------------------------------------
// The search
// --------------------------------------------------------------------------

/// A fractional integer column at the current node.
struct Candidate {
  std::size_t column = 0;  ///< index into the IntegerColumn list
  Real value = 0.0;        ///< ORIGINAL-space value
  Real frac_down = 0.0;    ///< `f- = v - floor(v)`
  Real frac_up = 0.0;      ///< `f+ = ceil(v) - v`
  Real score = 0.0;
  bool down_infeasible = false;
  bool up_infeasible = false;
};

class BranchAndBound {
 public:
  BranchAndBound(Problem& working, model::CanonicalResult& canon,
                 std::vector<IntegerColumn> integers, const Options& options,
                 MilpStatistics& stats)
      : working_(working),
        canon_(canon),
        integers_(std::move(integers)),
        options_(options),
        stats_(stats),
        pseudocosts_(integers_.size()) {
    node_lp_ = options;
    node_lp_.simplex.method = model::Method::DualSimplex;
    probe_lp_ = node_lp_;
    // A probe that runs out of iterations is an ESTIMATE, not a failure; the
    // primal cleanup exists to finish a solve that must produce a verdict,
    // and would turn a cheap probe into a full solve.
    probe_lp_.simplex.primal_cleanup = false;
    // Probes also run UNPERTURBED. Removing a perturbation means a primal
    // phase II after every optimal solve (SolveSimplex.cpp), and a probe is a
    // short, iteration-capped estimate -- that cleanup would be most of its
    // cost. Node LPs, which must be exact, keep it.
    probe_lp_.simplex.cost_perturbation = false;
  }

  core::Expected<Solution> run();

 private:
  core::Expected<SimplexResult> solve_with(const std::vector<Real>& lower,
                                           const std::vector<Real>& upper,
                                           const model::Options& lp, const Basis* warm);
  [[nodiscard]] std::vector<Candidate> fractional(const SimplexResult& lp) const;
  [[nodiscard]] std::size_t select(std::vector<Candidate>& candidates, const Node& node,
                                   const SimplexResult& lp);
  void strong_branch(Candidate& c, const Node& node, const SimplexResult& lp);
  [[nodiscard]] Real elapsed() const;
  [[nodiscard]] core::Expected<Solution> incumbent_solution(const SimplexResult& lp);

  Problem& working_;
  model::CanonicalResult& canon_;
  std::vector<IntegerColumn> integers_;
  const Options& options_;
  MilpStatistics& stats_;
  model::Options node_lp_;
  model::Options probe_lp_;
  PseudocostTable pseudocosts_;
  std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
  std::size_t solved_node_lps_ = 0;
};

Real BranchAndBound::elapsed() const {
  return std::chrono::duration<Real>(std::chrono::steady_clock::now() - start_).count();
}

core::Expected<SimplexResult> BranchAndBound::solve_with(const std::vector<Real>& lower,
                                                         const std::vector<Real>& upper,
                                                         const model::Options& lp,
                                                         const Basis* warm) {
  model::CanonicalProblem& p = canon_.problem;
  for (std::size_t j = 0; j < lower.size(); ++j) {
    p.col_lower[j] = lower[j];
    p.col_upper[j] = upper[j];
  }
  return simplex::solve_simplex(p, lp, warm);
}

std::vector<Candidate> BranchAndBound::fractional(const SimplexResult& lp) const {
  std::vector<Candidate> out;
  const Real tol = options_.milp.integer_tolerance;
  for (std::size_t k = 0; k < integers_.size(); ++k) {
    const IntegerColumn& ic = integers_[k];
    const Real v = ic.scale * lp.x[ic.canonical];
    const Real down = v - std::floor(v);
    const Real up = std::ceil(v) - v;
    if (std::min(down, up) <= tol) continue;
    Candidate c;
    c.column = k;
    c.value = v;
    c.frac_down = down;
    c.frac_up = up;
    out.push_back(c);
  }
  return out;
}

/// One strong-branching evaluation: both children, from the NODE's own
/// optimal basis, each capped at `gamma` dual simplex iterations ([CIP]
/// section 5.4). The gains update the pseudocosts ([AKM] Algorithm 3 step 2b)
/// and become the candidate's score (step 2c).
void BranchAndBound::strong_branch(Candidate& c, const Node& node, const SimplexResult& lp) {
  const auto& m = options_.milp;
  const IntegerColumn& ic = integers_[c.column];

  // gamma = 2 * (average node LP iterations so far), clamped to [10, 500].
  const Real average =
      solved_node_lps_ > 0
          ? static_cast<Real>(stats_.node_lp_iterations) / static_cast<Real>(solved_node_lps_)
          : 0.0;
  const auto gamma = static_cast<std::size_t>(std::clamp<Real>(
      2.0 * average, static_cast<Real>(m.strong_iterations_min),
      static_cast<Real>(m.strong_iterations_max)));
  probe_lp_.simplex.max_iterations = gamma;

  std::vector<Real> lower = node.lower;
  std::vector<Real> upper = node.upper;
  const Real floor_v = std::floor(c.value);
  const Real ceil_v = std::ceil(c.value);

  Real gains[2] = {0.0, 0.0};
  bool known[2] = {false, false};
  for (int side = 0; side < 2; ++side) {
    const bool up = side == 1;
    std::vector<Real> lo = lower;
    std::vector<Real> hi = upper;
    if (up) {
      lo[ic.canonical] = std::max(lo[ic.canonical], ceil_v / ic.scale);
    } else {
      hi[ic.canonical] = std::min(hi[ic.canonical], floor_v / ic.scale);
    }
    ++stats_.strong_branching_probes;
    if (lo[ic.canonical] > hi[ic.canonical]) {
      (up ? c.up_infeasible : c.down_infeasible) = true;
      continue;
    }
    auto probe = solve_with(lo, hi, probe_lp_, &lp.basis);
    if (!probe.has_value()) continue;
    stats_.strong_branching_iterations += probe->iterations;
    if (probe->status == SolverStatus::Infeasible) {
      // A verdict, not an estimate: the dual simplex only reports Infeasible
      // with no artificial bound in play (DualSimplex.cpp). So this side is
      // empty -- [CIP] section 5.4's "variable fixings due to infeasible
      // strong branching LPs".
      (up ? c.up_infeasible : c.down_infeasible) = true;
      continue;
    }
    if (probe->status == SolverStatus::Optimal ||
        probe->status == SolverStatus::MaxIterations) {
      // An iteration-limited dual simplex sits at a dual-feasible basis, so
      // its objective is how far the bound has risen SO FAR -- the estimate
      // strong branching is defined to use.
      gains[side] = std::max(0.0, probe->objective - lp.objective);
      known[side] = true;
      const Real f = up ? c.frac_up : c.frac_down;
      pseudocosts_.record(c.column, up, gains[side] / f);
    }
  }

  const Real down_estimate =
      known[0] ? gains[0] : c.frac_down * pseudocosts_.value(c.column, false);
  const Real up_estimate = known[1] ? gains[1] : c.frac_up * pseudocosts_.value(c.column, true);
  c.score = product_score(down_estimate, up_estimate, m.score_epsilon);
}

std::size_t BranchAndBound::select(std::vector<Candidate>& candidates, const Node& node,
                                   const SimplexResult& lp) {
  const auto& m = options_.milp;

  if (m.branching == BranchingRule::MostFractional) {
    std::size_t best = 0;
    Real best_distance = -1.0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const Real d = std::min(candidates[i].frac_down, candidates[i].frac_up);
      if (d > best_distance) {
        best_distance = d;
        best = i;
      }
    }
    return best;
  }

  // [AKM] Algorithm 3 step 2: pseudocost scores, then sort non-increasing.
  for (Candidate& c : candidates) {
    c.score = product_score(c.frac_down * pseudocosts_.value(c.column, false),
                            c.frac_up * pseudocosts_.value(c.column, true), m.score_epsilon);
  }
  std::vector<std::size_t> order(candidates.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return candidates[a].score > candidates[b].score;
  });

  if (m.branching == BranchingRule::Reliability) {
    // [CIP] section 5.7: strong-branching iterations are budgeted at half the
    // node-LP iterations plus 100000. Past the budget the reliability
    // threshold drops to 0 (pure pseudocost); over its upper half it falls
    // linearly from eta_rel to 1. [CIP] also RAISES eta_rel when strong
    // branching is very cheap; that half is not implemented, because the
    // thesis states the increase only as "proportional" without a constant.
    const Real budget = 0.5 * static_cast<Real>(stats_.node_lp_iterations) + 100000.0;
    const auto used = static_cast<Real>(stats_.strong_branching_iterations);
    Real eta_rel = m.reliability;
    if (used > budget) {
      eta_rel = 0.0;
    } else if (used > 0.5 * budget) {
      const Real t = (used - 0.5 * budget) / (0.5 * budget);
      eta_rel = m.reliability + t * (1.0 - m.reliability);
    }

    // Step 2d watches `s = max_{k in F} s_k` over ALL candidates -- reliable
    // ones keep their pseudocost scores and still compete -- and stops once
    // it has not changed for `lambda` consecutive score updates. Sorted
    // descending, so it starts as the first candidate's score.
    auto current_max = [&candidates]() {
      Real best = -kInf;
      for (const Candidate& k : candidates) best = std::max(best, k.score);
      return best;
    };
    Real max_score = candidates[order.front()].score;
    std::size_t unchanged = 0;
    std::size_t evaluated = 0;
    for (std::size_t idx : order) {
      if (evaluated >= m.max_strong_candidates) break;
      if (elapsed() >= m.time_limit_seconds) break;
      Candidate& c = candidates[idx];
      if (pseudocosts_.reliability(c.column) >= eta_rel) continue;  // reliable
      strong_branch(c, node, lp);
      ++evaluated;
      // One dead side makes this the branch to take: the other child is the
      // only one that exists. Both dead means the node itself is empty.
      if (c.down_infeasible || c.up_infeasible) {
        ++stats_.strong_branching_fixings;
        return idx;
      }
      const Real now = current_max();
      if (now != max_score) {
        max_score = now;
        unchanged = 0;
      } else if (++unchanged >= m.lookahead) {
        break;
      }
    }
  }

  std::size_t best = order.front();
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].score > candidates[best].score) best = i;
  }
  return best;
}

core::Expected<Solution> BranchAndBound::incumbent_solution(const SimplexResult& lp) {
  const Solution canonical = simplex::to_canonical_solution(canon_.problem, lp);
  return reconstruct_solution(working_, canon_.problem, canon_.transforms, canonical);
}

core::Expected<Solution> BranchAndBound::run() {
  const auto& m = options_.milp;
  const std::size_t n = canon_.problem.num_cols();

  std::priority_queue<Node, std::vector<Node>, NodeOrder> pending;
  {
    Node root;
    root.lower.resize(n);
    root.upper.resize(n);
    for (std::size_t j = 0; j < n; ++j) {
      root.lower[j] = canon_.problem.col_lower[j];
      root.upper[j] = canon_.problem.col_upper[j];
    }
    root.bound = -kInf;
    pending.push(std::move(root));
  }

  bool have_incumbent = false;
  Real incumbent_objective = kInf;  // canonical (minimization) objective
  Solution incumbent;
  bool closed = false;
  Real closing_bound = 0.0;

  while (!pending.empty()) {
    if (stats_.nodes >= m.node_limit || elapsed() >= m.time_limit_seconds) break;

    Node node = std::move(const_cast<Node&>(pending.top()));
    pending.pop();

    // Best-first: everything still queued has a bound no better than this
    // one, so once this cannot beat the incumbent (or is within the gap),
    // nothing can -- which is what makes stopping a proof.
    if (have_incumbent) {
      const Real gap = std::fabs(incumbent_objective - node.bound) /
                       (1.0 + std::fabs(incumbent_objective));
      if (node.bound >= incumbent_objective || gap < m.gap_tolerance) {
        closed = true;
        // The proven bound is the better of the two: the incumbent itself
        // when the queue can no longer beat it, the queue's best otherwise.
        closing_bound = std::min(node.bound, incumbent_objective);
        break;
      }
    }

    ++stats_.nodes;
    auto lp = solve_with(node.lower, node.upper, node_lp_, node.warm.get());
    if (!lp.has_value()) return lp.error();
    stats_.node_lp_iterations += lp->iterations;
    ++solved_node_lps_;

    if (lp->status == SolverStatus::Infeasible) continue;
    if (lp->status == SolverStatus::Unbounded) {
      if (stats_.nodes == 1) {
        return core::make_error(core::ErrorCode::Unbounded,
                                "solve_milp: the root LP relaxation is unbounded, so the "
                                "MILP is unbounded or infeasible; neither is proven here");
      }
      // A bounded root cannot have an unbounded child: a child's feasible set
      // is a subset of the root's. Treat it as an unreliable node.
      ++stats_.unreliable_nodes;
      continue;
    }
    if (lp->status != SolverStatus::Optimal) {
      // No trustworthy bound for this subtree. Dropping it is NOT safe -- the
      // optimum may be in it -- so it is counted, and a nonzero count forbids
      // claiming optimality at the end.
      ++stats_.unreliable_nodes;
      continue;
    }

    // [AKM] section 2.2: the gain per unit, measured on the child actually
    // solved, updates the pseudocost of the variable that created it.
    if (node.origin.present && node.origin.fraction > 0.0) {
      const Real gain = std::max(0.0, lp->objective - node.origin.parent_objective);
      pseudocosts_.record(node.origin.column, node.origin.up, gain / node.origin.fraction);
    }

    if (have_incumbent && lp->objective >= incumbent_objective) continue;

    std::vector<Candidate> candidates = fractional(*lp);
    if (candidates.empty()) {
      auto solution = incumbent_solution(*lp);
      if (!solution.has_value()) return solution.error();
      have_incumbent = true;
      incumbent_objective = lp->objective;
      incumbent = std::move(*solution);
      continue;
    }

    const std::size_t pick = select(candidates, node, *lp);
    const Candidate& c = candidates[pick];
    if (c.down_infeasible && c.up_infeasible) continue;  // the node is empty

    const IntegerColumn& ic = integers_[c.column];
    const auto basis = std::make_shared<const Basis>(lp->basis);
    for (int side = 0; side < 2; ++side) {
      const bool up = side == 1;
      if (up ? c.up_infeasible : c.down_infeasible) continue;
      Node child;
      child.lower = node.lower;
      child.upper = node.upper;
      if (up) {
        child.lower[ic.canonical] =
            std::max(child.lower[ic.canonical], std::ceil(c.value) / ic.scale);
      } else {
        child.upper[ic.canonical] =
            std::min(child.upper[ic.canonical], std::floor(c.value) / ic.scale);
      }
      if (child.lower[ic.canonical] > child.upper[ic.canonical]) continue;
      child.bound = lp->objective;
      child.depth = node.depth + 1;
      child.warm = basis;
      child.origin = Origin{c.column, up, up ? c.frac_up : c.frac_down, lp->objective, true};
      pending.push(std::move(child));
    }
  }

  // Map a canonical bound to the original objective. The canonical objective
  // is the original's, negated for a maximization, and may differ from it by
  // a constant (a substituted-out column's contribution). The constant is
  // taken from the incumbent, whose value is known in both spaces, rather
  // than assumed to be zero.
  const bool maximize = working_.sense == core::ObjSense::Maximize;
  const Real sign = maximize ? -1.0 : 1.0;
  const Real offset = have_incumbent ? incumbent.objective - sign * incumbent_objective : 0.0;
  auto to_original = [sign, offset](Real canonical) { return sign * canonical + offset; };

  Solution result;
  if (have_incumbent) result = std::move(incumbent);

  const bool sound = stats_.unreliable_nodes == 0;
  if (closed && sound) {
    result.status = SolverStatus::Optimal;
    result.best_bound = to_original(closing_bound);
  } else if (pending.empty() && sound) {
    // Every node was solved or pruned on a trustworthy bound.
    result.status = have_incumbent ? SolverStatus::Optimal : SolverStatus::Infeasible;
    result.best_bound = have_incumbent ? result.objective : 0.0;
  } else {
    // A limit, or an unexplored subtree: report what is known, never a
    // verdict the search did not earn.
    result.status = SolverStatus::NotConverged;
    result.best_bound = pending.empty() ? (have_incumbent ? result.objective : 0.0)
                                        : to_original(pending.top().bound);
  }
  result.nodes_explored = stats_.nodes;
  result.from_best_iterate = result.status != SolverStatus::Optimal;
  return result;
}

core::SolverStatus verdict_status(core::ErrorCode code) {
  return code == core::ErrorCode::PrimalInfeasible ? SolverStatus::Infeasible
                                                   : SolverStatus::Unbounded;
}

bool is_verdict(core::ErrorCode code) {
  return code == core::ErrorCode::PrimalInfeasible || code == core::ErrorCode::Unbounded;
}

Solution verdict_solution(const Problem& problem, SolverStatus status) {
  Solution s;
  s.status = status;
  s.x.resize(problem.num_cols());
  s.x.assign(0.0);
  s.y.resize(problem.num_rows());
  s.y.assign(0.0);
  s.s.resize(problem.num_rows());
  s.s.assign(0.0);
  s.z.resize(problem.num_cols());
  s.z.assign(0.0);
  s.v.resize(problem.num_cols());
  s.v.assign(0.0);
  return s;
}

}  // namespace

core::Expected<Solution> solve_milp(const Problem& problem, const Options& options,
                                    MilpStatistics* statistics) {
  const auto start = std::chrono::steady_clock::now();
  MilpStatistics local;
  MilpStatistics& stats = statistics != nullptr ? *statistics : local;
  stats = MilpStatistics{};

  if (!problem.has_discrete()) return solve_lp(problem, options);
  if (problem.has_quadratic()) {
    return core::make_error(core::ErrorCode::UnsupportedFeature,
                            "solve_milp: quadratic objectives are not supported by the "
                            "simplex-based branch-and-bound");
  }
  for (const auto t : problem.col_type) {
    if (t == core::VarType::SemiContinuous) {
      return core::make_error(core::ErrorCode::UnsupportedFeature,
                              "solve_milp: semi-continuous columns are not supported");
    }
  }

  // The original-space MILP reductions Module 22 uses, in the same order and
  // for the same reasons -- see BranchAndBound.cu.
  Problem working = problem.clone();
  std::vector<AbsorbingColumnElimination> absorbing;
  eliminate_equality_row_absorbing_singletons(working, absorbing);
  if (const auto st = tighten_integer_rows(working); !st.ok()) {
    if (is_verdict(st.error().code)) {
      return verdict_solution(problem, verdict_status(st.error().code));
    }
    return st.error();
  }

  // Root cover and GCD cuts, shared with Module 22 (MilpCuts.hpp). Separated
  // against the LP relaxation, solved here by the same dual simplex the search
  // uses. They are appended as ORIGINAL rows before canonicalization, so the
  // search below sees them as ordinary constraints.
  CutStatistics cut_stats;
  if (options.milp.root_cuts) {
    model::Options relaxation = options;
    relaxation.simplex.method = model::Method::DualSimplex;
    const auto st = add_root_cuts(working, [&relaxation](const Problem& p) {
      return solve_lp(p, relaxation);
    }, &cut_stats);
    if (!st.ok()) return st.error();
  }
  stats.root_cuts = cut_stats.cuts;

  auto canon = model::canonicalize(working, options);
  if (!canon.has_value()) {
    if (is_verdict(canon.error().code)) {
      return verdict_solution(problem, verdict_status(canon.error().code));
    }
    return canon.error();
  }
  if (const auto st = scale(canon->problem, options, canon->transforms); !st.ok()) {
    return st.error();
  }

  // Locate every original integer column in canonical space.
  std::vector<Real> column_scale(canon->problem.num_cols(), 1.0);
  std::vector<std::size_t> original_of(canon->problem.num_cols(),
                                       std::numeric_limits<std::size_t>::max());
  for (const auto& rec : canon->transforms.records()) {
    if (rec.kind == model::TransformKind::KeepColumn) {
      original_of[static_cast<std::size_t>(rec.secondary)] =
          static_cast<std::size_t>(rec.primary);
    } else if (rec.kind == model::TransformKind::ColumnScaling) {
      column_scale[static_cast<std::size_t>(rec.primary)] = rec.value;
    } else if (rec.kind == model::TransformKind::RemoveFixedVariable) {
      // An integer column fixed at a fractional value has no integral point.
      const auto j = static_cast<std::size_t>(rec.primary);
      if (working.col_type[j] != core::VarType::Continuous) {
        const Real v = rec.value;
        if (std::fabs(v - std::round(v)) > options.milp.integer_tolerance) {
          return verdict_solution(problem, SolverStatus::Infeasible);
        }
      }
    }
  }
  std::vector<IntegerColumn> integers;
  for (std::size_t j = 0; j < original_of.size(); ++j) {
    const std::size_t orig = original_of[j];
    if (orig == std::numeric_limits<std::size_t>::max()) continue;
    if (working.col_type[orig] == core::VarType::Continuous) continue;
    integers.push_back(IntegerColumn{j, orig, column_scale[j]});
  }

  BranchAndBound search(working, *canon, std::move(integers), options, stats);
  auto result = search.run();
  if (!result.has_value()) return result.error();

  // Absorbing columns come back from their row equation, exactly as in
  // Module 22: the elimination guaranteed the column's own bound, not the row.
  if (result->x.size() == working.num_cols()) {
    const auto& csr = working.A.csr;
    for (const auto& elim : absorbing) {
      Real other = 0.0;
      for (auto k = csr.slice_begin(elim.row); k < csr.slice_end(elim.row); ++k) {
        other += csr.values()[k] * result->x[static_cast<std::size_t>(csr.indices()[k])];
      }
      result->x[elim.col] = (elim.b_i - other) / elim.a_ij;
    }
  }

  result->solve_time_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace sovsolve::solver
