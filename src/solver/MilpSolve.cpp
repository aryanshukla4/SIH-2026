#include "sovsolve/solver/MilpSolve.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/MilpCanonicalPresolve.hpp"
#include "sovsolve/solver/LpSolve.hpp"
#include "sovsolve/solver/MilpCuts.hpp"
#include "sovsolve/solver/MilpPresolve.hpp"
#include "sovsolve/solver/MilpSeparators.hpp"
#include "sovsolve/solver/Scaler.hpp"
#include "sovsolve/solver/SolutionReconstructor.hpp"
#include "sovsolve/solver/simplex/PrimalSimplex.hpp"
#include "sovsolve/solver/simplex/SimplexSolution.hpp"
#include "sovsolve/solver/simplex/SolveSimplex.hpp"

namespace sovsolve::solver {

namespace {

using core::Real;
using core::SolverStatus;
using core::is_finite_bound;
using model::BranchingRule;
using model::NodeSelection;
using model::Options;
using model::Problem;
using model::Solution;
using simplex::Basis;
using simplex::SimplexResult;

constexpr Real kInf = std::numeric_limits<Real>::infinity();
constexpr std::size_t kNoColumn = std::numeric_limits<std::size_t>::max();

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
  /// How often column `k` has been branched on in direction `up`, with the
  /// child's LP solved -- [CIP] Algorithm 7.7's "number of evaluated
  /// downwards branchings", read per direction.
  [[nodiscard]] Real count(std::size_t k, bool up) const {
    return up ? entries_[k].count_up : entries_[k].count_down;
  }

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

// --------------------------------------------------------------------------
// Conflict analysis data -- [CIP] chapter 11
// --------------------------------------------------------------------------

/// Why a local bound change happened: the vertices of [CIP] 11.2.1's
/// generalized conflict graph are bound changes, and the arcs into a vertex
/// are its reason.
enum class Reason : std::uint8_t {
  Branch,    ///< a branching decision: no reason, the first vertex of its depth
  Row,       ///< Algorithm 7.1 on row `source` (source == m: the objective row)
  Conflict,  ///< propagation of conflict constraint `source`
  Global,    ///< the node met a (tightened) global bound: valid everywhere
  Leaf,      ///< local reduced cost strengthening: kept, never resolved
};

struct BoundChange {
  std::size_t col = 0;
  bool upper = false;       ///< changes the upper bound (else the lower)
  Real value = 0.0;
  Real old_value = 0.0;
  std::size_t depth = 0;
  Reason reason = Reason::Branch;
  std::size_t source = 0;
  /// Trail length when the reason's bounds were read: the deduction used the
  /// bounds in force at that point, so its reasons are the latest changes
  /// BEFORE it.
  std::size_t snapshot = 0;
  /// Row deduction from the MAX activity (the lambda side) rather than the min.
  bool from_lambda = false;
};

/// The trail from the root to a node, shared between siblings: a node owns
/// only its own segment and points at its parent's.
struct TrailSegment {
  std::shared_ptr<const TrailSegment> parent;
  std::vector<BoundChange> changes;
  std::size_t base = 0;
  [[nodiscard]] std::size_t end() const noexcept { return base + changes.size(); }
};

/// One literal of a bound disjunction (11.14): `x_col <= bound` (upper) or
/// `x_col >= bound`, canonical units.
struct Literal {
  std::size_t col = 0;
  bool upper = false;
  Real bound = 0.0;
};

struct ConflictConstraint {
  std::vector<Literal> literals;
  std::size_t age = 0;
  bool alive = true;
  /// The proof used the objective cutoff (the objective row, a conflict that
  /// did, or a reduced-cost bound): valid only for points better than
  /// `cutoff_value`, the canonical incumbent objective when it was derived.
  bool uses_cutoff = false;
  Real cutoff_value = 0.0;
  /// [CIP] 7.4: the two watched literals (indices into `literals`; equal for
  /// a one-literal conflict).
  std::size_t w1 = 0;
  std::size_t w2 = 0;
};

/// Why propagation failed, for the analysis.
struct Cause {
  enum class Kind : std::uint8_t { None, RowMin, RowMax, Var, Conflict } kind = Kind::None;
  std::size_t index = 0;
  std::size_t snapshot = 0;
};

/// Records bound changes as propagation makes them.
struct TrailWriter {
  std::vector<BoundChange>* changes = nullptr;
  std::size_t base = 0;
  std::size_t depth = 0;
  [[nodiscard]] std::size_t size() const noexcept { return base + changes->size(); }
};

struct Node {
  std::vector<Real> lower;
  std::vector<Real> upper;
  /// The parent's LP objective: a valid lower bound for this subtree, since a
  /// child's feasible set is a subset of its parent's.
  Real bound = 0.0;
  std::size_t depth = 0;
  /// [CIP] section 6.4's best estimate `e_Q`, the estimated objective of the
  /// best integer point in this subtree. See `estimate()` for how an unsolved
  /// child gets one.
  Real estimate = 0.0;
  /// The parent's optimal basis, shared by both children. Still DUAL feasible
  /// for either child -- only a bound moved -- which is why the dual simplex
  /// resumes from it.
  std::shared_ptr<const Basis> warm;
  Origin origin;
  /// The bound changes from the root to this node, its branching decision
  /// last ([CIP] chapter 11).
  std::shared_ptr<const TrailSegment> trail;
};

/// The open nodes -- the leaves of the tree -- held so that BOTH orders
/// [CIP] chapter 6 needs are available at O(log n): the smallest dual bound
/// (best first, and the global lower bound) and the smallest estimate (best
/// estimate). [CIP] section 3.3.6 keeps a priority queue for the leaves; two
/// ordered sets over one pool are that, for two keys at once.
///
/// Ties, [CIP] section 6.2: equal bounds are broken by the better estimate,
/// equal estimates by the better bound, and then by DEPTH, deeper first -- the
/// thesis's "stay close to the previous subproblem". The id makes the order
/// total and therefore deterministic.
class OpenNodes {
 public:
  using Id = std::size_t;

  Id insert(Node node) {
    const Id id = next_++;
    by_bound_.insert(bound_key(node, id));
    by_estimate_.insert(estimate_key(node, id));
    nodes_.emplace(id, std::move(node));
    return id;
  }

  [[nodiscard]] bool contains(Id id) const { return nodes_.count(id) != 0; }
  [[nodiscard]] const Node& at(Id id) const { return nodes_.at(id); }
  [[nodiscard]] bool empty() const noexcept { return nodes_.empty(); }

  Node take(Id id) {
    auto it = nodes_.find(id);
    Node node = std::move(it->second);
    by_bound_.erase(bound_key(node, id));
    by_estimate_.erase(estimate_key(node, id));
    nodes_.erase(it);
    return node;
  }

  [[nodiscard]] Id best_bound() const { return std::get<3>(*by_bound_.begin()); }
  [[nodiscard]] Id best_estimate() const { return std::get<3>(*by_estimate_.begin()); }
  /// The global lower bound: no open node's subtree can do better.
  [[nodiscard]] Real lower_bound() const { return std::get<0>(*by_bound_.begin()); }

 private:
  using Key = std::tuple<Real, Real, long long, Id>;
  static Key bound_key(const Node& n, Id id) {
    return {n.bound, n.estimate, -static_cast<long long>(n.depth), id};
  }
  static Key estimate_key(const Node& n, Id id) {
    return {n.estimate, n.bound, -static_cast<long long>(n.depth), id};
  }

  std::map<Id, Node> nodes_;
  std::set<Key> by_bound_;
  std::set<Key> by_estimate_;
  Id next_ = 0;
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
                 MilpStatistics& stats, const CanonicalPostsolve* postsolve)
      : working_(working),
        canon_(canon),
        integers_(std::move(integers)),
        options_(options),
        stats_(stats),
        postsolve_(postsolve != nullptr && postsolve->removes_columns() ? postsolve : nullptr),
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

    const model::CanonicalProblem& p = canon_.problem;
    root_lower_.assign(p.col_lower.data(), p.col_lower.data() + p.num_cols());
    root_upper_.assign(p.col_upper.data(), p.col_upper.data() + p.num_cols());
    down_locks_.assign(integers_.size(), 0);
    up_locks_.assign(integers_.size(), 0);
    binary_.assign(integers_.size(), 0);
    global_lower_ = root_lower_;
    global_upper_ = root_upper_;
    integer_scale_.assign(p.num_cols(), 0.0);
    for (const IntegerColumn& ic : integers_) integer_scale_[ic.canonical] = ic.scale;
    integer_index_.assign(p.num_cols(), kNoColumn);
    for (std::size_t k = 0; k < integers_.size(); ++k) integer_index_[integers_[k].canonical] = k;
    watchers_.assign(p.num_cols(), {});
    // [CIP] 7.6: "all objective coefficients are integral ... and all objective
    // coefficients for continuous variables are zero". Canonical c_j is
    // s_j times the original coefficient, since x_original = s_j x_j.
    integral_objective_ = true;
    for (std::size_t j = 0; j < p.num_cols(); ++j) {
      if (p.c[j] == 0.0) continue;
      if (integer_scale_[j] == 0.0) {
        integral_objective_ = false;
        break;
      }
      const Real original = p.c[j] / integer_scale_[j];
      if (std::fabs(original - std::round(original)) > 1e-9) {
        integral_objective_ = false;
        break;
      }
    }
    // The transform stack is keyed by PLAIN canonical column -- the space the
    // model had when it was canonicalized and scaled. Presolve stage B removes
    // columns after that, so these two maps are built in the plain space the
    // records speak and then compacted into the reduced one everything
    // downstream (`lit.col` at the conflict pool, `j` in the RENS sub-MIP)
    // indexes them by. With stage B off the two spaces coincide and the
    // compaction is the identity.
    const std::size_t plain_cols =
        postsolve_ != nullptr ? postsolve_->plain_columns : p.num_cols();
    std::vector<std::size_t> original_of_plain(plain_cols, kNoColumn);
    std::vector<Real> scale_of_plain(plain_cols, 1.0);
    for (const auto& rec : canon_.transforms.records()) {
      if (rec.kind == model::TransformKind::KeepColumn) {
        const auto at = static_cast<std::size_t>(rec.secondary);
        if (at < plain_cols) original_of_plain[at] = static_cast<std::size_t>(rec.primary);
      } else if (rec.kind == model::TransformKind::ColumnScaling) {
        const auto at = static_cast<std::size_t>(rec.primary);
        if (at < plain_cols) scale_of_plain[at] = rec.value;
      }
    }
    original_of_.assign(p.num_cols(), kNoColumn);
    scale_of_.assign(p.num_cols(), 1.0);
    for (std::size_t j = 0; j < plain_cols; ++j) {
      const std::size_t at =
          postsolve_ != nullptr ? postsolve_->new_of_old[j] : j;
      if (at == CanonicalPostsolve::kRemoved || at >= p.num_cols()) continue;
      original_of_[at] = original_of_plain[j];
      scale_of_[at] = scale_of_plain[j];
    }
    original_of_plain_ = std::move(original_of_plain);
    scale_of_plain_ = std::move(scale_of_plain);

    // What each REDUCED column is, written in plain columns. A stage B merge
    // makes the survivor mean `x_j + lambda x_k`, so a point arriving from
    // ORIGINAL space (the RENS sub-MIP's solution) cannot simply be read off
    // the survivor's own original column -- it has to be re-folded the same
    // way presolve folded it. Replayed FORWARD, which is the direction that
    // composes: a later merge folds whatever the earlier ones already built.
    std::vector<std::vector<std::pair<std::size_t, Real>>> fold(plain_cols);
    for (std::size_t j = 0; j < plain_cols; ++j) fold[j].emplace_back(j, 1.0);
    if (postsolve_ != nullptr) {
      for (const auto& rec : postsolve_->undo) {
        if (rec.kind == PostsolveColumn::Kind::MergeParallel) {
          for (const auto& [pc, coef] : fold[rec.column]) {
            fold[rec.partner].emplace_back(pc, rec.pivot * coef);
          }
        }
        fold[rec.column].clear();  // removed either way
      }
    }
    fold_.assign(p.num_cols(), {});
    for (std::size_t j = 0; j < plain_cols; ++j) {
      const std::size_t at =
          postsolve_ != nullptr ? postsolve_->new_of_old[j] : j;
      if (at == CanonicalPostsolve::kRemoved || at >= p.num_cols()) continue;
      fold_[at] = std::move(fold[j]);
    }
    const auto& csc = p.A.csc;
    for (std::size_t k = 0; k < integers_.size(); ++k) {
      const std::size_t j = integers_[k].canonical;
      for (std::size_t q = csc.slice_begin(j); q < csc.slice_end(j); ++q) {
        const auto row = static_cast<std::size_t>(csc.indices()[q]);
        const Real a = csc.values()[q];
        if (row < p.num_equality) {
          ++down_locks_[k];
          ++up_locks_[k];
        } else if (a > 0.0) {
          ++up_locks_[k];
        } else if (a < 0.0) {
          ++down_locks_[k];
        }
      }
      const Real s = integers_[k].scale;
      binary_[k] = static_cast<char>(s * root_lower_[j] == 0.0 && s * root_upper_[j] == 1.0);
    }
  }

  core::Expected<Solution> run();

  /// Root cuts were appended to the canonical problem: solutions are rebuilt
  /// against `plain`, the same model without them (MilpSolve.cpp, stage 8b).
  void set_reconstruction(const model::CanonicalResult* plain) { plain_ = plain; }
  /// The cut loop's last basis, so the root LP resumes instead of restarting.
  void set_root_basis(simplex::Basis basis) {
    root_basis_ = std::make_shared<const Basis>(std::move(basis));
  }

 private:
  core::Expected<SimplexResult> solve_with(const std::vector<Real>& lower,
                                           const std::vector<Real>& upper,
                                           const model::Options& lp, const Basis* warm);
  [[nodiscard]] std::vector<Candidate> fractional(const SimplexResult& lp) const;
  [[nodiscard]] std::size_t select(std::vector<Candidate>& candidates, const Node& node,
                                   const SimplexResult& lp);
  void strong_branch(Candidate& c, const Node& node, const SimplexResult& lp);
  [[nodiscard]] Real elapsed() const;
  [[nodiscard]] Real estimate(const std::vector<Candidate>& candidates, Real bound) const;
  [[nodiscard]] core::Expected<Solution> incumbent_solution(const SimplexResult& lp);

  // --- incumbent ------------------------------------------------------------
  [[nodiscard]] bool dominated(Real bound) const;
  /// Adopt `lp`'s point if it beats the incumbent. `objective` is canonical.
  [[nodiscard]] core::Status offer_incumbent(const SimplexResult& lp, Real objective);

  // --- primal heuristics, [CIP] chapter 9 ------------------------------------
  [[nodiscard]] core::Status simple_rounding(const SimplexResult& lp, std::size_t* counter);
  [[nodiscard]] core::Status dive(const Node& node, const SimplexResult& lp);
  [[nodiscard]] bool dive_budget_left() const;
  [[nodiscard]] core::Status feasibility_pump(const SimplexResult& root);
  /// [CIP] Algorithm 7.1 over every row (and the objective cutoff row once an
  /// incumbent exists), to a fixed point. Tightens `lower`/`upper` in place;
  /// returns false when the bounds admit no feasible point.
  [[nodiscard]] bool propagate(std::vector<Real>& lower, std::vector<Real>& upper,
                               TrailWriter* trail = nullptr, Cause* cause = nullptr);

  // --- conflict analysis, [CIP] chapter 11 ----------------------------------
  /// The node's whole trail, flattened: the segments from the root plus the
  /// node's own changes.
  [[nodiscard]] static std::vector<BoundChange> flatten(
      const std::shared_ptr<const TrailSegment>& segment, const std::vector<BoundChange>& local);
  /// Resolve an initial conflict set (trail positions) and store the
  /// resulting FUIP conflict constraints.
  void analyze_conflict(const std::vector<BoundChange>& trail, std::vector<std::size_t> initial,
                        bool uses_cutoff);
  /// Whether a propagation failure's own proof involved the cutoff.
  [[nodiscard]] bool cause_uses_cutoff(const Cause& cause) const;
  /// Make room in the pool (Witzig et al.): the oldest live conflicts go.
  void trim_conflict_pool();
  /// The initial conflict set of a propagation failure.
  [[nodiscard]] std::vector<std::size_t> propagation_conflict(const std::vector<BoundChange>& trail,
                                                              const Cause& cause);
  /// [CIP] Algorithm 11.1 on the Farkas ray of an infeasible node LP.
  [[nodiscard]] std::vector<std::size_t> lp_conflict(const std::vector<BoundChange>& trail,
                                                     const std::vector<Real>& ray,
                                                     const std::vector<Real>& lower,
                                                     const std::vector<Real>& upper);
  /// [CIP] Algorithm 7.11, on every incumbent improvement.
  void root_reduced_cost_strengthening();
  [[nodiscard]] core::Status rens(const SimplexResult& root);
  /// Canonical structural point of an ORIGINAL-space solution of `working_`,
  /// or false if some canonical column cannot be recovered.
  [[nodiscard]] bool to_canonical(const Solution& original, std::vector<Real>& x) const;
  /// Offer an integer point for the canonical STRUCTURAL values `x`, after
  /// checking it against every row and bound -- a heuristic's point is a
  /// claim, and only a checked claim may prune.
  [[nodiscard]] core::Status offer_point(const SimplexResult& base, const std::vector<Real>& x,
                                         bool* accepted);

  Problem& working_;
  model::CanonicalResult& canon_;
  std::vector<IntegerColumn> integers_;
  const Options& options_;
  MilpStatistics& stats_;
  /// Stage B's column map, or null when the presolved and plain column spaces
  /// coincide. Set at construction because the constructor already needs it.
  const CanonicalPostsolve* postsolve_ = nullptr;
  model::Options node_lp_;
  model::Options probe_lp_;
  PseudocostTable pseudocosts_;
  std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
  std::size_t solved_node_lps_ = 0;
  /// Every integer column's ORIGINAL-space value in the root LP, for Martin's
  /// child-selection rule ([CIP] section 6.1).
  std::vector<Real> root_values_;

  bool have_incumbent_ = false;
  Real incumbent_objective_ = kInf;  ///< canonical (minimization) objective
  Solution incumbent_;

  /// [CIP] Definition 3.3 / Example 3.4, per integer column: the rows that
  /// block moving it down and up. Canonical rows are `A_E x = b_E` (lock
  /// both ways) and `A_I x <= b_I` (a positive coefficient locks up, a
  /// negative one down). Scaling is positive, so directions carry over.
  std::vector<std::size_t> down_locks_;
  std::vector<std::size_t> up_locks_;
  std::vector<char> binary_;
  /// The root's canonical column bounds, against which a heuristic point is
  /// checked. `canon_.problem`'s own bounds are overwritten by every solve.
  std::vector<Real> root_lower_;
  std::vector<Real> root_upper_;
  std::size_t next_dive_rule_ = 0;
  /// Canonical column -> original column and scale (`x_original = s x`);
  /// `kNoColumn` for a column with no original (a range column).
  std::vector<std::size_t> original_of_;
  std::vector<Real> scale_of_;
  /// The same two maps in PLAIN canonical space, plus what each reduced
  /// column is as a combination of plain ones. Only `to_canonical` needs
  /// them, and only because stage B can merge two columns into one.
  std::vector<std::size_t> original_of_plain_;
  std::vector<Real> scale_of_plain_;
  std::vector<std::vector<std::pair<std::size_t, Real>>> fold_;
  std::size_t last_improvement_node_ = 0;
  const model::CanonicalResult* plain_ = nullptr;
  std::shared_ptr<const Basis> root_basis_;

  // --- domain propagation, [CIP] chapter 7 ----------------------------------
  /// Global bounds: the root's, tightened by root reduced cost strengthening.
  /// Every node is intersected with them before it is propagated.
  std::vector<Real> global_lower_;
  std::vector<Real> global_upper_;
  /// Per canonical column: the scale of an integer column (`x_original =
  /// s x`), 0 for a continuous one -- integrality is rounded in ORIGINAL space.
  std::vector<Real> integer_scale_;
  /// The objective is integral on every integer point ([CIP] 7.6's test).
  bool integral_objective_ = false;
  /// Root LP: objective, point and reduced costs, for Algorithm 7.11.
  bool have_root_ = false;
  Real root_objective_ = 0.0;
  std::vector<Real> root_x_;
  std::vector<Real> root_reduced_cost_;
  /// [CIP] 8.8's "active region" of each column: the smallest interval
  /// containing every node-LP value it has taken during the search.
  std::vector<Real> active_min_;
  std::vector<Real> active_max_;

  std::vector<ConflictConstraint> conflicts_;
  /// Per canonical column, the conflicts watching a literal on it. Entries go
  /// stale when a watch moves and are skipped (and dropped) when met.
  std::vector<std::vector<std::size_t>> watchers_;
  /// Incremental watching across nodes: the bounds the last node's
  /// propagation ended with, and the conflicts it left with a false watch
  /// (they forced a literal, proved the node empty, or are new) -- those are
  /// re-examined at the next node whatever changed.
  std::vector<Real> prev_lower_;
  std::vector<Real> prev_upper_;
  std::vector<std::size_t> recheck_;
  /// Canonical column -> index into `integers_`, or kNoColumn.
  std::vector<std::size_t> integer_index_;
  std::size_t alive_conflicts_ = 0;
  std::size_t oldest_alive_ = 0;
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

/// [CIP] section 6.4, the best estimate rule of Forrest et al.:
///
///     e_Q = c_Q + sum_{j fractional} min{ Psi-_j f-_j , Psi+_j f+_j }
///
/// with `Psi` the pseudocosts -- "the estimated minimum value of a rounded
/// solution". It needs the node's LP solution, so it is computed when a node
/// is SOLVED and handed to both children, the same way they inherit its dual
/// bound (section 6.3: "the child nodes inherit the dual bound of their parent
/// node"). The thesis does not say how an unsolved child's estimate is formed;
/// inheriting it is the reading that adds nothing to the text.
Real BranchAndBound::estimate(const std::vector<Candidate>& candidates, Real bound) const {
  Real e = bound;
  for (const Candidate& c : candidates) {
    e += std::min(c.frac_down * pseudocosts_.value(c.column, false),
                  c.frac_up * pseudocosts_.value(c.column, true));
  }
  return e;
}

core::Expected<Solution> BranchAndBound::incumbent_solution(const SimplexResult& lp) {
  if (plain_ != nullptr) {
    // Stage A presolve (rows deleted, coefficients changed) and root cuts
    // (rows appended) leave the COLUMNS alone, so the structural part of `x`
    // is already a point of the plain canonical model. Stage B does not: it
    // removes columns, and `expand_canonical_point` replays its undo stack to
    // put the missing ones back. Either way the logicals are then recomputed
    // on the plain model as b - A x, and the duals -- meaningless for a MILP
    // solution -- are zeroed.
    const model::CanonicalProblem& pp = plain_->problem;
    const std::size_t n = pp.num_cols();
    const std::size_t m0 = pp.num_rows();
    SimplexResult r = lp;
    if (postsolve_ != nullptr) {
      const std::size_t reduced = canon_.problem.num_cols();
      if (r.x.size() < reduced) {
        return core::make_error(core::ErrorCode::DimensionMismatch,
                                "MIP postsolve: LP point shorter than the reduced model");
      }
      std::vector<Real> plain_x;
      expand_canonical_point(*postsolve_, core::HostSpan<const Real>(r.x.data(), reduced),
                             plain_x);
      r.x.assign(plain_x.begin(), plain_x.end());
    }
    r.x.resize(n + m0);
    const auto& csr = pp.A.csr;
    for (std::size_t i = 0; i < m0; ++i) {
      Real act = 0.0;
      for (std::size_t q = csr.slice_begin(i); q < csr.slice_end(i); ++q) {
        act += csr.values()[q] * r.x[static_cast<std::size_t>(csr.indices()[q])];
      }
      r.x[n + i] = i < pp.num_equality ? 0.0 : std::max(pp.b[i] - act, 0.0);
    }
    r.y.assign(m0, 0.0);
    r.reduced_cost.assign(n + m0, 0.0);
    const Solution canonical = simplex::to_canonical_solution(plain_->problem, r);
    return reconstruct_solution(working_, plain_->problem, plain_->transforms, canonical);
  }
  const Solution canonical = simplex::to_canonical_solution(canon_.problem, lp);
  return reconstruct_solution(working_, canon_.problem, canon_.transforms, canonical);
}

bool BranchAndBound::dominated(Real bound) const {
  // A node cannot improve on the incumbent: its bound is no better, or within
  // the gap tolerance of it.
  if (!have_incumbent_) return false;
  const Real gap =
      std::fabs(incumbent_objective_ - bound) / (1.0 + std::fabs(incumbent_objective_));
  return bound >= incumbent_objective_ || gap < options_.milp.gap_tolerance;
}

core::Status BranchAndBound::offer_incumbent(const SimplexResult& lp, Real objective) {
  if (have_incumbent_ && objective >= incumbent_objective_) return core::Status::Ok();
  auto solution = incumbent_solution(lp);
  if (!solution.has_value()) return solution.error();
  if (!have_incumbent_) stats_.first_incumbent_node = stats_.nodes;
  ++stats_.incumbents;
  last_improvement_node_ = stats_.nodes;
  have_incumbent_ = true;
  incumbent_objective_ = objective;
  incumbent_ = std::move(*solution);
  if (options_.milp.propagation) root_reduced_cost_strengthening();
  // Witzig et al. section 3: a conflict derived under the objective cutoff is
  // deleted once the incumbent it was derived with is "sufficiently worse"
  // than the new one -- 5%.
  for (ConflictConstraint& cc : conflicts_) {
    if (!cc.alive || !cc.uses_cutoff) continue;
    const Real worse = cc.cutoff_value - objective;
    if (worse > options_.milp.conflict_cutoff_drop * std::max(1.0, std::fabs(objective))) {
      cc.alive = false;
      --alive_conflicts_;
    }
  }
  return core::Status::Ok();
}

// --------------------------------------------------------------------------
// Primal heuristics, [CIP] chapter 9
// --------------------------------------------------------------------------

core::Status BranchAndBound::offer_point(const SimplexResult& base, const std::vector<Real>& x,
                                         bool* accepted) {
  *accepted = false;
  const model::CanonicalProblem& p = canon_.problem;
  const std::size_t n = p.num_cols();
  const std::size_t rows = p.num_rows();
  const Real tol = options_.milp.integer_tolerance;

  for (std::size_t j = 0; j < n; ++j) {
    if (x[j] < root_lower_[j] - tol * (1.0 + std::fabs(root_lower_[j]))) return core::Status::Ok();
    if (x[j] > root_upper_[j] + tol * (1.0 + std::fabs(root_upper_[j]))) return core::Status::Ok();
  }
  for (const IntegerColumn& ic : integers_) {
    const Real v = ic.scale * x[ic.canonical];
    if (std::fabs(v - std::round(v)) > tol) return core::Status::Ok();
  }

  SimplexResult candidate = base;
  const auto& csr = p.A.csr;
  for (std::size_t i = 0; i < rows; ++i) {
    Real activity = 0.0;
    for (std::size_t q = csr.slice_begin(i); q < csr.slice_end(i); ++q) {
      activity += csr.values()[q] * x[static_cast<std::size_t>(csr.indices()[q])];
    }
    const Real b = p.b[i];
    const Real slack = b - activity;
    const Real allowed = tol * (1.0 + std::fabs(b));
    if (i < p.num_equality ? std::fabs(slack) > allowed : slack < -allowed) {
      return core::Status::Ok();
    }
    // The logical of row i: fixed at zero on an equality, the slack `s` on an
    // inequality (Basis.hpp's `[A | I]` form).
    candidate.x[n + i] = i < p.num_equality ? 0.0 : std::max(slack, 0.0);
  }
  Real objective = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    candidate.x[j] = x[j];
    objective += p.c[j] * x[j];
  }
  if (have_incumbent_ && objective >= incumbent_objective_) return core::Status::Ok();
  candidate.objective = objective;
  *accepted = true;
  return offer_incumbent(candidate, objective);
}

/// [CIP] section 9.1.2: "If zeta-_j = 0, we can safely set x_j := floor(x_j)
/// without violating any linear constraint. On the other hand, if
/// zeta+_j = 0, we can set x_j := ceil(x_j). The heuristic will succeed if all
/// fractional variables have either" lock at zero.
core::Status BranchAndBound::simple_rounding(const SimplexResult& lp, std::size_t* counter) {
  const std::vector<Candidate> fractional_columns = fractional(lp);
  if (fractional_columns.empty()) return core::Status::Ok();
  std::vector<Real> x(lp.x.begin(), lp.x.begin() + static_cast<std::ptrdiff_t>(
                                                         canon_.problem.num_cols()));
  for (const Candidate& c : fractional_columns) {
    const IntegerColumn& ic = integers_[c.column];
    Real target = 0.0;
    if (down_locks_[c.column] == 0) {
      target = std::floor(c.value);
    } else if (up_locks_[c.column] == 0) {
      target = std::ceil(c.value);
    } else {
      return core::Status::Ok();
    }
    x[ic.canonical] = target / ic.scale;
  }
  bool accepted = false;
  if (const auto st = offer_point(lp, x, &accepted); !st.ok()) return st;
  if (accepted && counter != nullptr) ++*counter;
  return core::Status::Ok();
}

bool BranchAndBound::dive_budget_left() const {
  const auto& m = options_.milp;
  const Real quota = m.dive_quota * static_cast<Real>(stats_.node_lp_iterations) +
                     static_cast<Real>(m.dive_allowance);
  return static_cast<Real>(stats_.dive_lp_iterations) < quota;
}

/// [CIP] Algorithm 9.1, the generic dive, from the node's optimal LP:
///
///   2. no fractional integer column: the LP point is integral -- offer it;
///   3. simple rounding, for an intermediate solution;
///   4. choose a fractional column and a direction (the four rules below);
///   5. tighten its bound;  6. [domain propagation: none in this engine];
///   7. resolve with the dual simplex from the current basis -- it stays dual
///      feasible, only a bound moved;
///   8. if infeasible, undo, take the OPPOSITE rounding and resolve (the
///      "one level of backtracking");
///   9. still infeasible: stop.
///
/// Also stopped by an LP without a trustworthy optimum and by an LP bound that
/// can no longer beat the incumbent -- "the diving process cannot produce a
/// better solution" (Berthold, "Primal Heuristics for Mixed Integer
/// Programs", diploma thesis, ZIB 2006, section 3.1 -- [B] below) -- and by
/// the 5% iteration quota, EXCEPT in a promising dive: [B] Algorithm 1 keeps
/// going past the limits while the fractional count, n_fr at the start,
/// is at most n_fr - i/2 after i steps. That cannot run away: it forces the
/// count down by one per two steps. ([B]'s other exception, i < i_min, is
/// not implemented: [B] gives no value for i_min.)
///
/// Step 4, the rules, rotated one per dive:
///   fractionality (9.2.2)  smallest min{f-, f+}, to the nearest integer;
///   coefficient   (9.2.1)  smallest min{zeta-, zeta+}, toward the smaller
///                          lock; ties by the smaller rounding distance;
///   line search   (9.2.4)  x below its root value: round down, ratio
///                          f- / (x_root - x); above: up, f+ / (x - x_root);
///                          smallest ratio -- "the first variable" the line
///                          from the root LP point through x hits an integer;
///   pseudocost    (9.2.5)  direction from the root (+-0.4), then the
///                          fraction (< 0.3 down, > 0.7 up), then the smaller
///                          pseudocost; select the maximum of
///                          f+ (1 + Psi+)/(1 + Psi-) going down, or
///                          f- (1 + Psi-)/(1 + Psi+) going up;
///   vectorlength  (9.2.6)  round toward the WORSE objective (up if c_j >= 0),
///                          smallest (objective loss) / (|A_j| + 1): cost per
///                          row the fixing "repairs". [B] 3.1.1 found it the
///                          most successful diver at the root, 93 of 129
///                          instances against 71 and 68.
/// Common to all ([CIP] 9.2): columns with a zero lock are left to simple
/// rounding, and binaries are preferred over general integers. A rule with no
/// usable column (line search when every column sits at its root value)
/// falls back to fractionality. A column with a zero lock that is chosen
/// anyway (only such columns left) is bounded in its NON-trivial direction:
/// "otherwise, the developing of x would be similar to the one of a rounding
/// heuristic but with much more computational effort" ([B] 3.1.1).
///
/// READING, pseudocost diving: the extracted text lost its floor/ceiling
/// marks; the factor that goes with rounding DOWN is taken as f+ = ceil - x,
/// because the thesis says the measure "prefers variables that are close to
/// their rounded value" -- rounding down, that is a LARGE f+. [B] 3.1.1
/// describes the score as "the fractionality f(x) times the quotient" of the
/// pseudocosts, f(x) = distance to the NEAREST integer (his Definition 1.4);
/// the two texts differ for a column rounded away from its nearest integer.
/// [CIP] is followed here as the later description of the same heuristic.
core::Status BranchAndBound::dive(const Node& node, const SimplexResult& lp) {
  if (!dive_budget_left()) return core::Status::Ok();
  ++stats_.dives;
  const std::size_t rule = next_dive_rule_++ % 5;
  std::size_t steps = 0;
  std::size_t start_fractional = 0;

  std::vector<Real> lower = node.lower;
  std::vector<Real> upper = node.upper;
  SimplexResult current = lp;

  for (;;) {
    const std::vector<Candidate> cands = fractional(current);
    if (cands.empty()) {
      bool accepted = false;
      std::vector<Real> x(current.x.begin(),
                          current.x.begin() + static_cast<std::ptrdiff_t>(
                                                  canon_.problem.num_cols()));
      if (const auto st = offer_point(current, x, &accepted); !st.ok()) return st;
      if (accepted) ++stats_.dive_solutions;
      return core::Status::Ok();
    }
    if (const auto st = simple_rounding(current, &stats_.dive_solutions); !st.ok()) return st;

    if (steps == 0) start_fractional = cands.size();
    const bool promising = 2 * cands.size() + steps <= 2 * start_fractional;
    if (steps > 0 && !dive_budget_left() && !promising) return core::Status::Ok();

    // Columns a dive should choose among.
    std::vector<const Candidate*> pool;
    for (const Candidate& c : cands) {
      if (down_locks_[c.column] > 0 && up_locks_[c.column] > 0) pool.push_back(&c);
    }
    if (pool.empty()) {
      for (const Candidate& c : cands) pool.push_back(&c);
    }
    bool any_binary = false;
    for (const Candidate* c : pool) any_binary = any_binary || binary_[c->column] != 0;
    if (any_binary) {
      std::vector<const Candidate*> binaries;
      for (const Candidate* c : pool) {
        if (binary_[c->column] != 0) binaries.push_back(c);
      }
      pool.swap(binaries);
    }

    const Candidate* pick = nullptr;
    bool up = false;
    auto nearest = [](const Candidate& c) { return c.frac_up < c.frac_down; };
    auto by_fractionality = [&]() {
      Real best = kInf;
      for (const Candidate* c : pool) {
        const Real f = std::min(c->frac_down, c->frac_up);
        if (f < best) {
          best = f;
          pick = c;
          up = nearest(*c);
        }
      }
    };
    if (rule == 0) {
      by_fractionality();
    } else if (rule == 1) {
      std::size_t best_locks = std::numeric_limits<std::size_t>::max();
      Real best_distance = kInf;
      for (const Candidate* c : pool) {
        const std::size_t dl = down_locks_[c->column];
        const std::size_t ul = up_locks_[c->column];
        const bool go_up = ul < dl || (ul == dl && nearest(*c));
        const std::size_t locks = std::min(dl, ul);
        const Real distance = go_up ? c->frac_up : c->frac_down;
        if (locks < best_locks || (locks == best_locks && distance < best_distance)) {
          best_locks = locks;
          best_distance = distance;
          pick = c;
          up = go_up;
        }
      }
    } else if (rule == 2) {
      Real best = kInf;
      for (const Candidate* c : pool) {
        const Real root = root_values_[c->column];
        Real ratio = kInf;
        bool go_up = false;
        if (c->value < root) {
          ratio = c->frac_down / (root - c->value);
        } else if (c->value > root) {
          ratio = c->frac_up / (c->value - root);
          go_up = true;
        } else {
          continue;
        }
        if (ratio < best) {
          best = ratio;
          pick = c;
          up = go_up;
        }
      }
      if (pick == nullptr) by_fractionality();
    } else if (rule == 4) {
      const auto& p = canon_.problem;
      Real best = kInf;
      for (const Candidate* c : pool) {
        const IntegerColumn& ic = integers_[c->column];
        // The canonical cost per ORIGINAL unit: x_original = s x_canonical.
        const Real cost = p.c[ic.canonical] / ic.scale;
        const bool go_up = cost >= 0.0;
        const Real loss = (go_up ? c->frac_up : c->frac_down) * std::fabs(cost);
        const auto length = static_cast<Real>(p.A.csc.slice_end(ic.canonical) -
                                              p.A.csc.slice_begin(ic.canonical));
        const Real score = loss / (length + 1.0);
        if (score < best) {
          best = score;
          pick = c;
          up = go_up;
        }
      }
    } else {
      Real best = -kInf;
      for (const Candidate* c : pool) {
        const Real root = root_values_[c->column];
        const Real psi_down = pseudocosts_.value(c->column, false);
        const Real psi_up = pseudocosts_.value(c->column, true);
        bool go_up = false;
        if (c->value < root - 0.4) {
          go_up = false;
        } else if (c->value > root + 0.4) {
          go_up = true;
        } else if (c->frac_down < 0.3) {
          go_up = false;
        } else if (c->frac_down > 0.7) {
          go_up = true;
        } else {
          go_up = !(psi_down < psi_up);
        }
        const Real score = go_up ? c->frac_down * (1.0 + psi_down) / (1.0 + psi_up)
                                 : c->frac_up * (1.0 + psi_up) / (1.0 + psi_down);
        if (score > best) {
          best = score;
          pick = c;
          up = go_up;
        }
      }
    }
    if (pick == nullptr) return core::Status::Ok();
    if (down_locks_[pick->column] == 0 && up_locks_[pick->column] > 0) up = true;
    if (up_locks_[pick->column] == 0 && down_locks_[pick->column] > 0) up = false;

    const IntegerColumn& ic = integers_[pick->column];
    const std::size_t j = ic.canonical;
    const Real saved_lower = lower[j];
    const Real saved_upper = upper[j];
    core::Expected<SimplexResult> next = core::make_error(core::ErrorCode::NumericalError, "");
    std::vector<Real> try_lower;
    std::vector<Real> try_upper;
    for (int attempt = 0; attempt < 2; ++attempt) {
      const bool go_up = attempt == 0 ? up : !up;  // attempt 1 is step 8
      try_lower = lower;
      try_upper = upper;
      try_lower[j] = saved_lower;
      try_upper[j] = saved_upper;
      if (go_up) {
        try_lower[j] = std::max(try_lower[j], std::ceil(pick->value) / ic.scale);
      } else {
        try_upper[j] = std::min(try_upper[j], std::floor(pick->value) / ic.scale);
      }
      if (try_lower[j] > try_upper[j]) continue;
      // Step 6: propagate the bound change. An empty domain is an infeasible
      // LP without solving one, and takes the same backtrack.
      if (options_.milp.propagation && !propagate(try_lower, try_upper)) continue;
      next = solve_with(try_lower, try_upper, node_lp_, &current.basis);
      if (!next.has_value()) return next.error();
      stats_.dive_lp_iterations += next->iterations;
      if (next->status != SolverStatus::Infeasible) {
        lower.swap(try_lower);
        upper.swap(try_upper);
        break;
      }
    }
    if (!next.has_value() || next->status != SolverStatus::Optimal) return core::Status::Ok();
    if (dominated(next->objective)) return core::Status::Ok();
    current = std::move(*next);
    ++steps;
  }
}

// --------------------------------------------------------------------------
// Domain propagation -- [CIP] chapter 7
// --------------------------------------------------------------------------

namespace {

/// [CIP] (7.2): the feasibility tolerance epsilon-hat.
constexpr Real kPropTolerance = 1e-6;

/// [CIP] (7.3): a bound change is accepted only if it cuts off at least 5% of
/// the domain's width or of the bound's magnitude (at least 1), or makes an
/// infinite bound finite -- otherwise a chain like 0.2 <= x - y <= 0.8 on
/// {0..1000} walks the bounds down one unit per round, 1000 rounds long.
Real min_change(Real lo, Real hi, Real bound) {
  return 0.05 * std::max(std::min(hi - lo, std::fabs(bound)), 1.0);
}

}  // namespace

/// [CIP] Algorithm 7.1, applied to the canonical rows -- `A_E x = b_E`
/// (lambda = rho = b) and `A_I x <= b_I` (lambda = -inf) -- and, once there is
/// an incumbent, to the objective cutoff row of 7.6. Activity bounds
/// (Definition 7.1) are recomputed from the current bounds each time a row is
/// visited rather than kept up to date by the thesis's event handler
/// (Algorithm 7.2): simpler, and a deduction made from activities that a
/// tightening earlier in the same visit has made stale is still VALID -- the
/// stale ones are looser -- only possibly weaker; the row is re-queued anyway.
///
/// Per variable ([CIP] 7.1, Reduction 5), with residual activities
/// alpha_j = min(a'x) - a_j x_j and beta_j likewise:
///     a_j > 0:  (lambda - beta_j)/a_j <= x_j <= (rho - alpha_j)/a_j
///     a_j < 0:  (rho - alpha_j)/a_j <= x_j <= (lambda - beta_j)/a_j
/// then (7.2): relax to five digits, u <- 10^-5 ceil(10^5 u - eps),
/// l <- 10^-5 floor(10^5 l + eps) -- the extracted text lost the rounding
/// marks, but "we slightly RELAX the newly calculated bounds" fixes their
/// direction -- and for an integer column round inward, u <- floor(u + eps),
/// l <- ceil(l - eps), in original space. Accepted per (7.3). Reduction 4
/// (min activity above rho, max below lambda) proves the node empty.
///
/// NOT from the thesis: a cap on visits per row per call
/// (MilpOptions::propagation_row_visits, default 20). [CIP] has no explicit
/// limit -- (7.3) is what bounds the work there -- and the cap only stops
/// propagation early, which is always safe.
bool BranchAndBound::propagate(std::vector<Real>& lower, std::vector<Real>& upper,
                               TrailWriter* trail, Cause* cause) {
  const model::CanonicalProblem& p = canon_.problem;
  const std::size_t m = p.num_rows();
  const std::size_t n = p.num_cols();
  const auto& csr = p.A.csr;
  const auto& csc = p.A.csc;

  // The objective row: c'x <= incumbent - delta, or with an integral
  // objective, <= incumbent - (1 - delta) ([CIP] 7.6).
  const bool objective_row = have_incumbent_;
  const Real cutoff = integral_objective_ ? incumbent_objective_ - (1.0 - kPropTolerance)
                                          : incumbent_objective_ - kPropTolerance;
  const std::size_t rows = m + (objective_row ? 1 : 0);

  std::deque<std::size_t> queue;
  std::vector<char> queued(rows, 1);
  for (std::size_t r = 0; r < rows; ++r) queue.push_back(r);
  std::size_t visits = 0;
  const std::size_t max_visits = options_.milp.propagation_row_visits * rows;

  auto fail = [&](Cause::Kind kind, std::size_t index, std::size_t snapshot) {
    if (cause != nullptr) *cause = Cause{kind, index, snapshot};
    return false;
  };
  auto record = [&](std::size_t j, bool up, Real value, Real old, Reason reason,
                    std::size_t source, std::size_t snapshot, bool from_lambda) {
    if (trail == nullptr) return;
    trail->changes->push_back(BoundChange{j, up, value, old, trail->depth, reason, source,
                                          snapshot, from_lambda});
  };
  // [CIP] 7.4: conflicts are looked at only through their watched columns.
  // The columns to look through are those whose bounds are tighter than the
  // root's -- a watched literal can only have become false there -- and those
  // tightened during this call.
  std::vector<std::size_t> changed_cols;
  std::vector<char> col_marked;
  const bool have_conflicts = alive_conflicts_ > 0;
  // A node's propagation (trail given) compares with the bounds the previous
  // node's propagation ended with: after that call every conflict it did not
  // put on `recheck_` had two non-false watches, and a watch can turn false
  // only where a bound is now TIGHTER. Comparing with the root instead (the
  // first version) re-examined every conflict watching any branched column at
  // every node -- 1240 examinations a node on gen-ip054. A dive (no trail)
  // starts from a node and only tightens it, so it compares with the root.
  // Dives too: a dive starts from the node just propagated -- whose end state
  // IS prev_ -- and only tightens it, so the same reference holds; a dive
  // reads the recheck list but leaves it for the next node.
  const bool incremental = prev_lower_.size() == n;
  if (have_conflicts) {
    col_marked.assign(n, 0);
    const auto& ref_lower = incremental ? prev_lower_ : root_lower_;
    const auto& ref_upper = incremental ? prev_upper_ : root_upper_;
    for (std::size_t j = 0; j < n; ++j) {
      if (lower[j] > ref_lower[j] || upper[j] < ref_upper[j]) {
        col_marked[j] = 1;
        changed_cols.push_back(j);
      }
    }
  }
  std::vector<std::size_t> recheck_next;
  // Records where the node propagation ended, whatever the outcome.
  struct EndState {
    BranchAndBound* self;
    bool active;
    std::vector<Real>& lower;
    std::vector<Real>& upper;
    std::vector<std::size_t>& next;
    ~EndState() {
      if (!active) return;
      self->prev_lower_ = lower;
      self->prev_upper_ = upper;
      self->recheck_ = std::move(next);
    }
  } end_state{this, trail != nullptr, lower, upper, recheck_next};
  auto requeue = [&](std::size_t j) {
    if (have_conflicts && col_marked[j] == 0) {
      col_marked[j] = 1;
      changed_cols.push_back(j);
    }
    for (std::size_t q = csc.slice_begin(j); q < csc.slice_end(j); ++q) {
      const auto row = static_cast<std::size_t>(csc.indices()[q]);
      if (queued[row] == 0) {
        queued[row] = 1;
        queue.push_back(row);
      }
    }
    if (objective_row && p.c[j] != 0.0 && queued[m] == 0) {
      queued[m] = 1;
      queue.push_back(m);
    }
  };

  std::vector<std::pair<std::size_t, Real>> entries;
  for (;;) {
  while (!queue.empty() && visits++ < max_visits) {
    const std::size_t r = queue.front();
    queue.pop_front();
    queued[r] = 0;
    const std::size_t snap = trail != nullptr ? trail->size() : 0;

    entries.clear();
    Real lambda = -kInf;
    Real rho = kInf;
    if (r < m) {
      for (std::size_t q = csr.slice_begin(r); q < csr.slice_end(r); ++q) {
        entries.emplace_back(static_cast<std::size_t>(csr.indices()[q]), csr.values()[q]);
      }
      rho = p.b[r];
      if (r < p.num_equality) lambda = p.b[r];
    } else {
      for (std::size_t j = 0; j < n; ++j) {
        if (p.c[j] != 0.0) entries.emplace_back(j, p.c[j]);
      }
      rho = cutoff;
    }

    // Activity bounds, infinite contributions counted separately ([CIP] 7.1).
    // "Infinite" is this codebase's convention -- a bound of magnitude 1e20 or
    // more (Types.hpp) -- NOT IEEE infinity: summing a 1e20 "bound" as a
    // number would swamp the finite part of the activity and could make a
    // residual too LARGE, i.e. an invalid, too-tight deduction.
    Real min_act = 0.0, max_act = 0.0;
    std::size_t min_inf = 0, max_inf = 0;
    auto lo_bound = [&](std::size_t j, Real a) { return a > 0.0 ? lower[j] : upper[j]; };
    auto hi_bound = [&](std::size_t j, Real a) { return a > 0.0 ? upper[j] : lower[j]; };
    for (const auto& [j, a] : entries) {
      if (is_finite_bound(lo_bound(j, a))) min_act += a * lo_bound(j, a); else ++min_inf;
      if (is_finite_bound(hi_bound(j, a))) max_act += a * hi_bound(j, a); else ++max_inf;
    }
    // Reduction 4.
    if (min_inf == 0 && min_act > rho + kPropTolerance * (1.0 + std::fabs(rho))) {
      return fail(Cause::Kind::RowMin, r, snap);
    }
    if (max_inf == 0 && std::isfinite(lambda) &&
        max_act < lambda - kPropTolerance * (1.0 + std::fabs(lambda))) {
      return fail(Cause::Kind::RowMax, r, snap);
    }

    for (const auto& [j, a] : entries) {
      // Residuals alpha_j (min) and beta_j (max) over every column but j.
      Real alpha = -kInf;
      if (is_finite_bound(lo_bound(j, a))) {
        if (min_inf == 0) alpha = min_act - a * lo_bound(j, a);
      } else if (min_inf == 1) {
        alpha = min_act;
      }
      Real beta = kInf;
      if (is_finite_bound(hi_bound(j, a))) {
        if (max_inf == 0) beta = max_act - a * hi_bound(j, a);
      } else if (max_inf == 1) {
        beta = max_act;
      }

      Real new_lo = -kInf;
      Real new_hi = kInf;
      if (std::isfinite(rho) && std::isfinite(alpha)) {
        (a > 0.0 ? new_hi : new_lo) = (rho - alpha) / a;
      }
      if (std::isfinite(lambda) && std::isfinite(beta)) {
        (a > 0.0 ? new_lo : new_hi) = (lambda - beta) / a;
      }

      const Real s = integer_scale_[j];
      bool changed = false;
      if (is_finite_bound(new_hi)) {
        new_hi = 1e-5 * std::ceil(1e5 * new_hi - kPropTolerance);
        if (s != 0.0) new_hi = std::floor(s * new_hi + kPropTolerance) / s;
        if (new_hi < upper[j] &&
            (!is_finite_bound(upper[j]) ||
             new_hi < upper[j] - min_change(lower[j], upper[j], upper[j]))) {
          // An upper bound comes from rho when a > 0, from lambda when a < 0.
          record(j, true, new_hi, upper[j], Reason::Row, r, snap, a < 0.0);
          upper[j] = new_hi;
          changed = true;
        }
      }
      if (is_finite_bound(new_lo)) {
        new_lo = 1e-5 * std::floor(1e5 * new_lo + kPropTolerance);
        if (s != 0.0) new_lo = std::ceil(s * new_lo - kPropTolerance) / s;
        if (new_lo > lower[j] &&
            (!is_finite_bound(lower[j]) ||
             new_lo > lower[j] + min_change(lower[j], upper[j], lower[j]))) {
          record(j, false, new_lo, lower[j], Reason::Row, r, snap, a > 0.0);
          lower[j] = new_lo;
          changed = true;
        }
      }
      if (!changed) continue;
      ++stats_.propagation_tightenings;
      if (lower[j] > upper[j] + kPropTolerance * (1.0 + std::fabs(upper[j]))) {
        return fail(Cause::Kind::Var, j, trail != nullptr ? trail->size() : 0);
      }
      if (lower[j] > upper[j]) lower[j] = upper[j];  // equal within tolerance
      requeue(j);
    }
  }

  // The conflict constraints ([CIP] 11.3: "solely used for domain
  // propagation"), by [CIP] Algorithm 7.7's two watched literals, generalized
  // from 0/1 fixings to bound disjunctions: a literal is TRUE (implied by the
  // bounds), FALSE (excluded by them) or open. A conflict is examined only
  // when a column it watches changed; if a watched literal is false it looks
  // for replacements, and with none left it either forces its last open
  // literal or proves the node empty. A conflict examined without a deduction
  // ages, and is dropped at the limit.
  bool deduced = false;
  if (have_conflicts) {
    enum class State : std::uint8_t { True, False, Open };
    auto state = [&](const Literal& lit) {
      const Real tol = kPropTolerance * (1.0 + std::fabs(lit.bound));
      if (lit.upper) {
        if (upper[lit.col] <= lit.bound + tol) return State::True;
        return lower[lit.col] <= lit.bound + tol ? State::Open : State::False;
      }
      if (lower[lit.col] >= lit.bound - tol) return State::True;
      return upper[lit.col] >= lit.bound - tol ? State::Open : State::False;
    };
    // Algorithm 7.7 step 5: among open literals, prefer those whose
    // falsifying branch direction has been explored least -- a literal
    // "x <= w" is falsified by branching UP, "x >= w" by branching down.
    auto branchings = [&](const Literal& lit) {
      const std::size_t k = integer_index_[lit.col];
      return k == kNoColumn ? 0.0 : pseudocosts_.count(k, lit.upper);
    };
    enum class Outcome : std::uint8_t { Done, Empty };
    std::vector<char> recheck_marked(conflicts_.size(), 0);
    auto mark_recheck = [&](std::size_t k) {
      if (trail == nullptr || k >= recheck_marked.size() || recheck_marked[k] != 0) return;
      recheck_marked[k] = 1;
      recheck_next.push_back(k);
    };
    // Examine conflict k; afterwards, if a watch is still false, it goes on
    // the recheck list for the next node.
    auto examine = [&](std::size_t k) -> Outcome {
      ConflictConstraint& cc = conflicts_[k];
      ++stats_.conflict_checks;
      const State s1 = state(cc.literals[cc.w1]);
      const State s2 = state(cc.literals[cc.w2]);
      if (s1 == State::Open && s2 == State::Open) return Outcome::Done;
      if (s1 == State::True || s2 == State::True) {
        if (s1 == State::False || s2 == State::False) mark_recheck(k);
        return Outcome::Done;  // satisfied here
      }
      // Steps 4-5: keep an open watch; otherwise take the least-branched
      // open literals as watches.
      std::size_t best1 = kNoColumn, best2 = kNoColumn;
      Real key1 = kInf, key2 = kInf;
      for (std::size_t q = 0; q < cc.literals.size(); ++q) {
        const State st = state(cc.literals[q]);
        if (st == State::True) {
          mark_recheck(k);
          return Outcome::Done;
        }
        if (st == State::False) continue;
        const bool was_watched = q == cc.w1 || q == cc.w2;
        const Real key = was_watched ? -1.0 : branchings(cc.literals[q]);
        if (key < key1) {
          best2 = best1;
          key2 = key1;
          best1 = q;
          key1 = key;
        } else if (key < key2) {
          best2 = q;
          key2 = key;
        }
      }
      const std::size_t snap = trail != nullptr ? trail->size() : 0;
      if (best1 == kNoColumn) {  // step 6: every literal false
        cc.age = 0;
        ++stats_.conflict_cutoffs;
        mark_recheck(k);
        fail(Cause::Kind::Conflict, k, snap);
        return Outcome::Empty;
      }
      if (best2 == kNoColumn) {  // step 7: the last open literal must hold
        const Literal lit = cc.literals[best1];
        if (lit.upper) {
          record(lit.col, true, lit.bound, upper[lit.col], Reason::Conflict, k, snap, false);
          upper[lit.col] = lit.bound;
        } else {
          record(lit.col, false, lit.bound, lower[lit.col], Reason::Conflict, k, snap, false);
          lower[lit.col] = lit.bound;
        }
        cc.age = 0;
        ++stats_.conflict_deductions;
        deduced = true;
        requeue(lit.col);
        mark_recheck(k);
        return Outcome::Done;
      }
      // Step 8: move the watches to two open literals.
      const std::size_t old1 = cc.literals[cc.w1].col;
      const std::size_t old2 = cc.literals[cc.w2].col;
      cc.w1 = best1;
      cc.w2 = best2;
      for (std::size_t q : {best1, best2}) {
        const std::size_t col = cc.literals[q].col;
        if (col != old1 && col != old2) watchers_[col].push_back(k);
      }
      if (++cc.age >= options_.milp.conflict_max_age) {
        cc.alive = false;
        --alive_conflicts_;
      }
      return Outcome::Done;
    };

    if (incremental && !recheck_.empty()) {
      std::vector<std::size_t> pending;
      if (trail != nullptr) {
        pending = std::move(recheck_);
        recheck_.clear();
      } else {
        pending = recheck_;  // a dive: read, do not consume
      }
      for (std::size_t k : pending) {
        if (k >= conflicts_.size() || !conflicts_[k].alive) continue;
        if (examine(k) == Outcome::Empty) return false;
      }
    }
    for (std::size_t c = 0; c < changed_cols.size(); ++c) {
      const std::size_t j = changed_cols[c];
      auto& list = watchers_[j];
      for (std::size_t idx = 0; idx < list.size(); ++idx) {
        const std::size_t k = list[idx];
        ConflictConstraint& cc = conflicts_[k];
        const bool watches_j = cc.literals[cc.w1].col == j || cc.literals[cc.w2].col == j;
        if (!cc.alive || !watches_j) {
          list[idx] = list.back();  // stale: the watch moved, or the conflict died
          list.pop_back();
          --idx;
          continue;
        }
        if (examine(k) == Outcome::Empty) return false;
      }
    }
    changed_cols.clear();
    // Columns examined in this pass may be tightened again by the next row
    // pass; they are re-marked then.
    std::fill(col_marked.begin(), col_marked.end(), 0);
  }
  if (!deduced || queue.empty() || visits >= max_visits) break;
  }
  return true;
}

// --------------------------------------------------------------------------
// Conflict analysis -- [CIP] chapter 11
// --------------------------------------------------------------------------

std::vector<BoundChange> BranchAndBound::flatten(
    const std::shared_ptr<const TrailSegment>& segment, const std::vector<BoundChange>& local) {
  std::vector<const TrailSegment*> chain;
  for (const TrailSegment* s = segment.get(); s != nullptr; s = s->parent.get()) chain.push_back(s);
  std::vector<BoundChange> out;
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    out.insert(out.end(), (*it)->changes.begin(), (*it)->changes.end());
  }
  out.insert(out.end(), local.begin(), local.end());
  return out;
}

namespace {

/// Per (column, side), the trail positions of its changes, ascending -- the
/// lookup "which change set this bound, as of trail length t".
class TrailIndex {
 public:
  explicit TrailIndex(const std::vector<BoundChange>& trail) {
    entries_.reserve(trail.size());
    for (std::size_t pos = 0; pos < trail.size(); ++pos) {
      entries_.emplace_back(key(trail[pos].col, trail[pos].upper), pos);
    }
    std::sort(entries_.begin(), entries_.end());
  }
  /// The latest change of (col, side) before `snapshot`, or npos.
  [[nodiscard]] std::size_t before(std::size_t col, bool upper, std::size_t snapshot) const {
    const std::size_t k = key(col, upper);
    auto it = std::lower_bound(entries_.begin(), entries_.end(), std::make_pair(k, snapshot));
    if (it == entries_.begin()) return npos;
    --it;
    return it->first == k ? it->second : npos;
  }
  static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

 private:
  static std::size_t key(std::size_t col, bool upper) { return 2 * col + (upper ? 1 : 0); }
  /// (column-side key, trail position), sorted: one allocation and one sort
  /// per analysis. A map of per-key vectors cost ~3 ms an analysis on
  /// gen-ip054 -- 6 s of a 21 s run (measured, pinned to one core).
  std::vector<std::pair<std::size_t, std::size_t>> entries_;
};

}  // namespace

std::vector<std::size_t> BranchAndBound::propagation_conflict(
    const std::vector<BoundChange>& trail, const Cause& cause) {
  const model::CanonicalProblem& p = canon_.problem;
  const std::size_t m = p.num_rows();
  const TrailIndex index(trail);
  std::vector<std::size_t> out;
  auto add = [&](std::size_t col, bool upper, std::size_t snapshot) {
    const std::size_t pos = index.before(col, upper, snapshot);
    if (pos != TrailIndex::npos) out.push_back(pos);
  };
  switch (cause.kind) {
    case Cause::Kind::RowMin:
    case Cause::Kind::RowMax: {
      // The bounds the violated activity bound was built from.
      const bool max_side = cause.kind == Cause::Kind::RowMax;
      auto visit = [&](std::size_t j, Real a) {
        const bool upper_bound = max_side ? a > 0.0 : a < 0.0;
        add(j, upper_bound, cause.snapshot);
      };
      if (cause.index < m) {
        const auto& csr = p.A.csr;
        for (std::size_t q = csr.slice_begin(cause.index); q < csr.slice_end(cause.index); ++q) {
          visit(static_cast<std::size_t>(csr.indices()[q]), csr.values()[q]);
        }
      } else {
        for (std::size_t j = 0; j < p.num_cols(); ++j) {
          if (p.c[j] != 0.0) visit(j, p.c[j]);
        }
      }
      break;
    }
    case Cause::Kind::Var:
      add(cause.index, false, cause.snapshot);
      add(cause.index, true, cause.snapshot);
      break;
    case Cause::Kind::Conflict:
      for (const Literal& lit : conflicts_[cause.index].literals) {
        // "x <= w" is false because of x's LOWER bound, "x >= w" of its upper.
        add(lit.col, !lit.upper, cause.snapshot);
      }
      break;
    case Cause::Kind::None:
      break;
  }
  return out;
}

/// [CIP] Algorithm 11.1, in activity form. The Farkas ray y of the dual
/// simplex aggregates the rows into sum_j alpha_j x_j + sum_{i ineq} y_i s_i
/// = y'b (s_i >= 0 the slacks); over the node's bounds its activity cannot
/// reach y'b -- that gap d > 0 is [CIP]'s infeasibility measure. Local bound
/// changes are relaxed, latest first and one change at a time ("always
/// relaxing to the previously active bound"), while d stays positive; the
/// changes that could not be relaxed are the initial conflict set. The side
/// of the proof is read off the numbers (min activity above y'b, or max below
/// it) rather than assumed from the certificate's sign convention.
std::vector<std::size_t> BranchAndBound::lp_conflict(const std::vector<BoundChange>& trail,
                                                     const std::vector<Real>& ray,
                                                     const std::vector<Real>& lower,
                                                     const std::vector<Real>& upper) {
  std::vector<std::size_t> out;
  const model::CanonicalProblem& p = canon_.problem;
  const std::size_t m = p.num_rows();
  const std::size_t n = p.num_cols();
  if (ray.size() != m) return out;
  std::vector<Real> alpha(n, 0.0);
  Real beta = 0.0;
  const auto& csr = p.A.csr;
  for (std::size_t i = 0; i < m; ++i) {
    if (ray[i] == 0.0) continue;
    beta += ray[i] * p.b[i];
    for (std::size_t q = csr.slice_begin(i); q < csr.slice_end(i); ++q) {
      alpha[static_cast<std::size_t>(csr.indices()[q])] += ray[i] * csr.values()[q];
    }
  }
  // Slacks s_i in [0, inf): a positive y_i adds [0, inf) to the activity.
  bool slack_pos = false, slack_neg = false;
  for (std::size_t i = p.num_equality; i < m; ++i) {
    if (ray[i] > 0.0) slack_pos = true;
    if (ray[i] < 0.0) slack_neg = true;
  }
  Real min_act = 0.0, max_act = 0.0;
  bool min_ok = !slack_neg, max_ok = !slack_pos;
  for (std::size_t j = 0; j < n; ++j) {
    const Real a = alpha[j];
    if (std::fabs(a) <= 1e-12) continue;
    const Real lo = a > 0.0 ? lower[j] : upper[j];
    const Real hi = a > 0.0 ? upper[j] : lower[j];
    if (is_finite_bound(lo)) min_act += a * lo; else min_ok = false;
    if (is_finite_bound(hi)) max_act += a * hi; else max_ok = false;
  }
  const Real tol = 1e-9 * (1.0 + std::fabs(beta));
  bool max_side = false;
  Real d = 0.0;
  if (min_ok && min_act > beta + tol) {
    d = min_act - beta;
  } else if (max_ok && max_act < beta - tol) {
    d = beta - max_act;
    max_side = true;
  } else {
    return out;  // not a usable proof numerically; skip the analysis
  }

  const TrailIndex index(trail);
  // Columns whose relevant bound is local, deepest latest change first.
  std::vector<std::pair<std::size_t, std::size_t>> order;  // (latest position, column)
  for (std::size_t j = 0; j < n; ++j) {
    const Real a = alpha[j];
    if (std::fabs(a) <= 1e-12) continue;
    const bool upper_side = max_side ? a > 0.0 : a < 0.0;
    const std::size_t pos = index.before(j, upper_side, trail.size());
    if (pos != TrailIndex::npos) order.emplace_back(pos, j);
  }
  std::sort(order.rbegin(), order.rend());
  for (const auto& [latest, j] : order) {
    const Real a = std::fabs(alpha[j]);
    const bool upper_side = max_side ? alpha[j] > 0.0 : alpha[j] < 0.0;
    std::size_t pos = latest;
    while (pos != TrailIndex::npos) {
      const BoundChange& e = trail[pos];
      if (e.reason == Reason::Global || e.depth == 0) {
        // Not a local change, but it may rest on the objective cutoff: the
        // analysis decides (and drops it).
        out.push_back(pos);
        break;
      }
      if (!is_finite_bound(e.old_value)) {
        out.push_back(pos);
        break;
      }
      const Real relaxed = d - a * std::fabs(e.value - e.old_value);
      if (relaxed <= tol) {
        out.push_back(pos);  // needed: relaxing it would lose the proof
        break;
      }
      d = relaxed;
      pos = index.before(j, upper_side, pos);
    }
  }
  return out;
}

/// [CIP] 11.1.1 and 11.3: resolve the conflict set through the conflict
/// graph, deepest level first. At each depth level, the latest resolvable
/// change is replaced by its reasons until one change remains at that level
/// -- its first unique implication point -- and the set is then emitted as a
/// conflict constraint: "one FUIP conflict constraint for every depth level",
/// at most 10 per conflict. Changes at depth 0 and changes that met a global
/// bound hold everywhere and are dropped. Reconvergence constraints and
/// non-chronological backtracking are not implemented: every open node is
/// propagated against all conflicts when it is selected, which prunes the
/// same subtrees, only when they are reached.
bool BranchAndBound::cause_uses_cutoff(const Cause& cause) const {
  const std::size_t m = canon_.problem.num_rows();
  if (cause.kind == Cause::Kind::RowMin || cause.kind == Cause::Kind::RowMax) {
    return cause.index >= m;
  }
  if (cause.kind == Cause::Kind::Conflict) return conflicts_[cause.index].uses_cutoff;
  return false;
}

void BranchAndBound::trim_conflict_pool() {
  const auto& opt = options_.milp;
  const std::size_t size = std::clamp<std::size_t>(
      canon_.problem.num_cols() + canon_.problem.num_rows(), opt.conflict_pool_min,
      opt.conflict_pool_max);
  while (alive_conflicts_ >= size && oldest_alive_ < conflicts_.size()) {
    if (conflicts_[oldest_alive_].alive) {
      conflicts_[oldest_alive_].alive = false;
      --alive_conflicts_;
    }
    ++oldest_alive_;
  }
}

void BranchAndBound::analyze_conflict(const std::vector<BoundChange>& trail,
                                      std::vector<std::size_t> initial, bool uses_cutoff) {
  ++stats_.conflicts_analyzed;
  const model::CanonicalProblem& p = canon_.problem;
  const std::size_t m = p.num_rows();
  const TrailIndex index(trail);

  // Whether a change that holds without branching -- a global bound, or a
  // deduction at the root -- still rests on the objective cutoff. Global
  // bounds are tightened only by root reduced cost strengthening (Algorithm
  // 7.11), which is cutoff reasoning; a reduced-cost leaf is too; a row
  // deduction is if it used the objective row or any reason that did.
  // Dropping such a change from a conflict is right -- it holds everywhere
  // the cutoff does -- but the conflict then holds only under the cutoff.
  // Missing this made valid-looking conflicts that excluded worse-than-
  // incumbent points while claiming to exclude nothing; the per-conflict
  // enumeration check in milp_test found it.
  std::map<std::size_t, bool> cutoff_memo;
  std::function<bool(std::size_t)> rests_on_cutoff = [&](std::size_t pos) -> bool {
    auto it = cutoff_memo.find(pos);
    if (it != cutoff_memo.end()) return it->second;
    cutoff_memo[pos] = false;  // guards against cycles; set properly below
    const BoundChange& e = trail[pos];
    bool result = false;
    if (e.reason == Reason::Global || e.reason == Reason::Leaf) {
      result = true;
    } else if (e.reason == Reason::Row) {
      if (e.source >= m) {
        result = true;
      } else {
        const auto& csr = p.A.csr;
        for (std::size_t q = csr.slice_begin(e.source); q < csr.slice_end(e.source) && !result;
             ++q) {
          const auto k = static_cast<std::size_t>(csr.indices()[q]);
          if (k == e.col) continue;
          const bool upper_bound = e.from_lambda ? csr.values()[q] > 0.0 : csr.values()[q] < 0.0;
          const std::size_t r = index.before(k, upper_bound, e.snapshot);
          if (r != TrailIndex::npos) result = rests_on_cutoff(r);
        }
      }
    } else if (e.reason == Reason::Conflict) {
      result = conflicts_[e.source].uses_cutoff;
      for (const Literal& lit : conflicts_[e.source].literals) {
        if (result) break;
        if (lit.col == e.col && lit.upper == e.upper) continue;
        const std::size_t r = index.before(lit.col, !lit.upper, e.snapshot);
        if (r != TrailIndex::npos) result = rests_on_cutoff(r);
      }
    }
    cutoff_memo[pos] = result;
    return result;
  };

  std::set<std::size_t> set;
  auto insert = [&](std::size_t pos) {
    if (pos == TrailIndex::npos) return;
    const BoundChange& e = trail[pos];
    if (e.reason == Reason::Global || e.depth == 0) {
      if (rests_on_cutoff(pos)) uses_cutoff = true;
      return;
    }
    set.insert(pos);
  };
  for (std::size_t pos : initial) insert(pos);
  if (set.empty()) return;

  // Reasons of a propagated change: the bounds its row (or conflict) read.
  auto resolve = [&](std::size_t pos) {
    const BoundChange& e = trail[pos];
    set.erase(pos);
    if (e.reason == Reason::Row && e.source >= m) uses_cutoff = true;
    if (e.reason == Reason::Conflict && conflicts_[e.source].uses_cutoff) uses_cutoff = true;
    if (e.reason == Reason::Row) {
      auto visit = [&](std::size_t k, Real a) {
        if (k == e.col) return;
        const bool upper_bound = e.from_lambda ? a > 0.0 : a < 0.0;
        insert(index.before(k, upper_bound, e.snapshot));
      };
      if (e.source < m) {
        const auto& csr = p.A.csr;
        for (std::size_t q = csr.slice_begin(e.source); q < csr.slice_end(e.source); ++q) {
          visit(static_cast<std::size_t>(csr.indices()[q]), csr.values()[q]);
        }
      } else {
        for (std::size_t j = 0; j < p.num_cols(); ++j) {
          if (p.c[j] != 0.0) visit(j, p.c[j]);
        }
      }
    } else if (e.reason == Reason::Conflict) {
      for (const Literal& lit : conflicts_[e.source].literals) {
        if (lit.col == e.col && lit.upper == e.upper) continue;  // the literal it enforced
        insert(index.before(lit.col, !lit.upper, e.snapshot));
      }
    }
  };

  auto emit = [&]() {
    // Latest change per (column, side): it implies the earlier ones.
    std::map<std::pair<std::size_t, bool>, std::size_t> latest;
    for (std::size_t pos : set) {
      auto key = std::make_pair(trail[pos].col, trail[pos].upper);
      auto it = latest.find(key);
      if (it == latest.end() || it->second < pos) latest[key] = pos;
    }
    ConflictConstraint cc;
    cc.uses_cutoff = uses_cutoff;
    for (const auto& [key, pos] : latest) {
      if (trail[pos].reason == Reason::Leaf) cc.uses_cutoff = true;  // a reduced-cost bound
    }
    cc.cutoff_value = incumbent_objective_;
    for (const auto& [key, pos] : latest) {
      const BoundChange& e = trail[pos];
      const Real s = integer_scale_[e.col];
      // (11.14): the negation of each bound change, strict for integers
      // (x < v  ->  x <= v - 1 in original units), relaxed to equality for
      // continuous columns.
      Real bound = e.value;
      if (s != 0.0) {
        const Real v = std::round(s * e.value);
        bound = (e.upper ? v + 1.0 : v - 1.0) / s;
      }
      cc.literals.push_back(Literal{e.col, !e.upper, bound});
    }
    if (cc.literals.empty()) return false;
    trim_conflict_pool();
    if (stats_.conflict_log != nullptr) {
      ConflictRecord rec;
      rec.uses_cutoff = cc.uses_cutoff;
      rec.incumbent = have_incumbent_ ? incumbent_.objective : 0.0;
      for (const Literal& lit : cc.literals) {
        rec.literals.push_back(ConflictRecord::Literal{original_of_[lit.col], lit.upper,
                                                       lit.bound * scale_of_[lit.col]});
      }
      stats_.conflict_log->push_back(std::move(rec));
    }
    // A ONE-literal conflict is a bound that holds everywhere the cutoff does:
    // it goes straight into the global bounds (which the analysis already
    // treats as cutoff-dependent) instead of being watched -- two watches on
    // one literal would never fire, since they act only when a watch turns
    // false. Kept in the pool as dead, so every conflict keeps its id.
    if (cc.literals.size() == 1) {
      const Literal& lit = cc.literals.front();
      if (lit.upper) {
        global_upper_[lit.col] = std::min(global_upper_[lit.col], lit.bound);
      } else {
        global_lower_[lit.col] = std::max(global_lower_[lit.col], lit.bound);
      }
      ++stats_.conflict_deductions;
      cc.alive = false;
      conflicts_.push_back(std::move(cc));
      ++stats_.conflict_constraints;
      return true;
    }
    // Watches: the two literals whose bound changes came latest on the trail
    // -- the usual SAT choice: going back up the tree, they are the first to
    // become open again.
    {
      std::size_t pos1 = 0, pos2 = 0, lit1 = 0, lit2 = 0, q = 0;
      bool have1 = false, have2 = false;
      for (const auto& [key, pos] : latest) {
        if (!have1 || pos > pos1) {
          pos2 = pos1; lit2 = lit1; have2 = have1;
          pos1 = pos; lit1 = q; have1 = true;
        } else if (!have2 || pos > pos2) {
          pos2 = pos; lit2 = q; have2 = true;
        }
        ++q;
      }
      cc.w1 = lit1;
      cc.w2 = have2 ? lit2 : lit1;
    }
    const std::size_t id = conflicts_.size();
    recheck_.push_back(id);  // all its literals are false here: look again next node
    watchers_[cc.literals[cc.w1].col].push_back(id);
    if (cc.literals[cc.w2].col != cc.literals[cc.w1].col) {
      watchers_[cc.literals[cc.w2].col].push_back(id);
    }
    conflicts_.push_back(std::move(cc));
    ++alive_conflicts_;
    ++stats_.conflict_constraints;
    return true;
  };

  std::size_t max_depth = 0;
  for (std::size_t pos : set) max_depth = std::max(max_depth, trail[pos].depth);
  std::size_t emitted = 0;
  std::set<std::size_t> last;
  for (std::size_t d = max_depth; d >= 1 && emitted < 10; --d) {
    for (;;) {
      std::size_t at_level = 0;
      std::size_t candidate = TrailIndex::npos;
      for (auto it = set.rbegin(); it != set.rend(); ++it) {
        const BoundChange& e = trail[*it];
        if (e.depth != d) continue;
        ++at_level;
        if (candidate == TrailIndex::npos &&
            (e.reason == Reason::Row || e.reason == Reason::Conflict)) {
          candidate = *it;
        }
      }
      if (at_level <= 1 || candidate == TrailIndex::npos) break;
      resolve(candidate);
    }
    if (set.empty()) break;
    if (set != last) {
      if (emit()) ++emitted;
      last = set;
    }
  }
}

/// [CIP] Algorithm 7.11. With c_R, x_R, r_R the root LP's objective, point and
/// reduced costs and c^ the incumbent: a column with r_j > 0 cannot exceed
/// x_R,j + (c^ - c_R)/r_j in any solution better than c^ (the LP bound would
/// already be c^), and r_j < 0 bounds it from below the same way. GLOBAL
/// bounds, re-derived on every incumbent improvement; the new bound does not
/// depend on the current one, so even small tightenings are safe to accept.
void BranchAndBound::root_reduced_cost_strengthening() {
  if (!have_root_) return;
  const Real gap = incumbent_objective_ - root_objective_;
  if (!(gap >= 0.0)) return;
  const Real tol = options_.simplex.dual_feasibility_tolerance;
  for (std::size_t j = 0; j < root_x_.size(); ++j) {
    const Real r = root_reduced_cost_[j];
    if (std::fabs(r) <= tol) continue;
    const Real s = integer_scale_[j];
    if (r > 0.0) {
      Real u = root_x_[j] + gap / r;
      if (s != 0.0) u = std::floor(s * u + kPropTolerance) / s;
      if (u < global_upper_[j]) {
        global_upper_[j] = std::max(u, global_lower_[j]);
        ++stats_.redcost_tightenings;
      }
    } else {
      Real l = root_x_[j] + gap / r;
      if (s != 0.0) l = std::ceil(s * l - kPropTolerance) / s;
      if (l > global_lower_[j]) {
        global_lower_[j] = std::min(l, global_upper_[j]);
        ++stats_.redcost_tightenings;
      }
    }
  }
}

// --------------------------------------------------------------------------
// The objective feasibility pump -- Berthold 2006, Algorithm 3; [CIP] 9.3.3
// --------------------------------------------------------------------------

namespace {

/// A fixed-seed generator: the pump's random choices must not make two runs
/// of the same model differ.
struct Lcg {
  std::uint64_t state = 0x2545F4914F6CDD1DULL;
  Real uniform() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<Real>(state >> 11) * (1.0 / 9007199254740992.0);
  }
};

/// [B] Definition 3.2 / [CIP] (9.1): round to the nearest integer.
Real round_half_up(Real v) { return std::floor(v + 0.5); }

}  // namespace

/// The pump alternates two sequences ([B] 3.1.2): x_tilde, integral on S, the
/// rounding of the last LP point; and x_bar, feasible for the LP, the point of
/// the LP polyhedron closest to x_tilde in the L1 distance over S --
///
///     Delta_S(x, x~) = sum_{j in S, x~_j = l_j} (x_j - l_j)
///                    + sum_{j in S, x~_j = u_j} (u_j - x_j) + ...
///
/// -- until they meet. Stage 1 takes S = the binaries, stage 2 S = all
/// integer columns, starting from the stage-1 rounding whose LP point was
/// closest ([B] Algorithm 3). Stage 3, a sub-MIP enumeration, is skipped, as
/// SCIP skips it ([CIP] 9.3.3) and as Berthold's own measurements did.
///
/// GENERAL INTEGERS: SCIP's form, not (3.2)'s. "It does not add auxiliary
/// variables d_j ... Instead, we set the objective coefficient to +1, -1, or
/// 0, depending on whether we want to round the variable down or up, or leave
/// it on its integral value" ([CIP] 9.3.3). So every round changes COSTS only,
/// the previous basis stays primal feasible, and the primal simplex resumes
/// from it -- "a change in the objective function does not destroy primal
/// feasibility" ([CIP] 9.3).
///
/// OBJECTIVE PUMP ([B] Definition 3.3): the LP minimizes
///     (1 - alpha) Delta + alpha (||Delta|| / ||c||) c'x,
/// alpha_0 = 1, alpha <- 0.95 alpha per round.
///
/// CYCLES ([B] Algorithm 3 steps 4-7): a 1-cycle (x~ unchanged) flips the T
/// most fractional columns (f > 0.02) of S, T uniform in [10, 30] -- as in
/// Fischetti, Glover and Lodi, "The feasibility pump", Math. Programming 104
/// (2005) ([FGL]), Figure 1 step 9: "flip the TT = rand(T/2, 3T/2) entries
/// with highest |x*_j - x~_j|", T = 20. A longer cycle -- x~ already seen
/// with alpha within 0.005 -- gets [FGL] section 3's random perturbation:
/// "for each j in I we generate a uniformly random value rho_j in [-0.3, 0.7]
/// and flip x~_j in case |x*_j - x~_j| + max{rho_j, 0} > 0.5".
///
/// A FLIP on a general integer ([FGL] states the rule for 0-1 MIPs) moves
/// x~_j to the other integer neighbour of x*_j, or one step off x*_j when x*_j
/// is itself integral, kept within the column's bounds.
///
/// Every candidate goes through offer_point: a rounding that violates a row
/// is rejected there, never trusted.
core::Status BranchAndBound::feasibility_pump(const SimplexResult& root) {
  const auto& m = options_.milp;
  model::CanonicalProblem& p = canon_.problem;
  const std::size_t n = p.num_cols();
  if (integers_.empty()) return core::Status::Ok();

  Real c_norm = 0.0;
  for (std::size_t j = 0; j < n; ++j) c_norm += p.c[j] * p.c[j];
  c_norm = std::sqrt(c_norm);

  // The pump's costs replace the true ones ONLY for the duration of each LP
  // solve: offer_point and the solution reconstruction read `p.c`, and a
  // candidate scored against the pump objective would be a wrong incumbent.
  const std::vector<Real> true_costs(p.c.data(), p.c.data() + n);

  Lcg rng;
  model::Options lp_options = node_lp_;
  lp_options.simplex.cost_perturbation = false;

  bool any_general = false;
  for (std::size_t k = 0; k < integers_.size(); ++k) any_general |= binary_[k] == 0;

  auto value = [&](const SimplexResult& r, std::size_t k) {
    return integers_[k].scale * r.x[integers_[k].canonical];
  };

  SimplexResult x_bar = root;
  // Stage-2 start: the stage-1 LP point closest to its rounding.
  SimplexResult best_stage1 = root;
  Real best_stage1_distance = kInf;

  for (int stage = 1; stage <= 2; ++stage) {
    if (stage == 2 && !any_general) break;  // "the algorithm will skip Stage 2 for MBPs"
    std::vector<std::size_t> S;
    for (std::size_t k = 0; k < integers_.size(); ++k) {
      if (stage == 2 || binary_[k] != 0) S.push_back(k);
    }
    if (S.empty()) continue;
    if (stage == 2) x_bar = best_stage1;

    const std::size_t max_rounds = stage == 1 ? m.fp_max_rounds_stage1 : m.fp_max_rounds_stage2;
    const std::size_t max_stalls = stage == 1 ? m.fp_max_stalls_stage1 : m.fp_max_stalls_stage2;
    Real alpha = 1.0;
    std::size_t perturbations = 0;
    std::map<std::vector<Real>, std::vector<Real>> visited;
    std::vector<Real> fractionality_history;
    std::vector<Real> previous;  // x~_{t-1}, over S
    std::vector<Real> tilde(S.size());
    for (std::size_t i = 0; i < S.size(); ++i) tilde[i] = round_half_up(value(x_bar, S[i]));

    for (std::size_t t = 0;; ++t) {
      if (elapsed() >= m.time_limit_seconds) return core::Status::Ok();

      // Step 4-5: a 1-cycle.
      if (!previous.empty() && tilde == previous) {
        std::vector<std::pair<Real, std::size_t>> by_fraction;
        for (std::size_t i = 0; i < S.size(); ++i) {
          const Real v = value(x_bar, S[i]);
          const Real f = std::fabs(v - round_half_up(v));
          if (f > 0.02) by_fraction.emplace_back(-f, i);
        }
        std::sort(by_fraction.begin(), by_fraction.end());
        const auto T = static_cast<std::size_t>(10 + std::floor(rng.uniform() * 21.0));
        for (std::size_t q = 0; q < std::min(T, by_fraction.size()); ++q) {
          const std::size_t i = by_fraction[q].second;
          const Real v = value(x_bar, S[i]);
          tilde[i] = tilde[i] > v ? std::floor(v) : std::ceil(v);
        }
      }
      // Steps 6-7: a longer cycle.
      auto seen = [&]() {
        auto it = visited.find(tilde);
        if (it == visited.end()) return false;
        for (Real a : it->second) {
          if (std::fabs(a - alpha) <= 0.005) return true;
        }
        return false;
      };
      while (seen()) {
        if (stage == 2 && ++perturbations > 100) return core::Status::Ok();  // "goto Stage 3"
        for (std::size_t i = 0; i < S.size(); ++i) {
          const Real v = value(x_bar, S[i]);
          const Real rho = -0.3 + rng.uniform();  // uniform on [-0.3, 0.7]
          if (std::fabs(v - tilde[i]) + std::max(rho, 0.0) <= 0.5) continue;
          Real flipped = tilde[i] > v ? std::floor(v) : std::ceil(v);
          if (flipped == tilde[i]) flipped += tilde[i] >= v ? -1.0 : 1.0;
          const IntegerColumn& ic = integers_[S[i]];
          const Real lo = std::ceil(ic.scale * root_lower_[ic.canonical] - 1e-9);
          const Real hi = std::floor(ic.scale * root_upper_[ic.canonical] + 1e-9);
          tilde[i] = std::clamp(flipped, lo, hi);
        }
        if (stage == 1 && perturbations++ > 1000) return core::Status::Ok();
      }

      // Step 8: is x~ (the LP point with S rounded) feasible for the MIP?
      {
        std::vector<Real> x(x_bar.x.begin(), x_bar.x.begin() + static_cast<std::ptrdiff_t>(n));
        for (std::size_t i = 0; i < S.size(); ++i) {
          x[integers_[S[i]].canonical] = tilde[i] / integers_[S[i]].scale;
        }
        bool accepted = false;
        if (const auto st = offer_point(x_bar, x, &accepted); !st.ok()) return st;
        if (accepted) {
          ++stats_.pump_solutions;
          return core::Status::Ok();
        }
      }
      visited[tilde].push_back(alpha);                      // step 9
      if (t > 0) alpha *= m.fp_alpha_factor;                // step 10
      if (t >= max_rounds) break;                           // step 11

      // Step 12: the LP closest to x~ under the combined objective.
      std::vector<Real> delta(n, 0.0);
      Real delta_norm = 0.0;
      for (std::size_t i = 0; i < S.size(); ++i) {
        const IntegerColumn& ic = integers_[S[i]];
        const Real v = value(x_bar, S[i]);
        const Real lo = ic.scale * root_lower_[ic.canonical];
        const Real hi = ic.scale * root_upper_[ic.canonical];
        Real direction = 0.0;  // +1: minimize x (push down), -1: push up
        if (tilde[i] <= lo) {
          direction = 1.0;
        } else if (tilde[i] >= hi) {
          direction = -1.0;
        } else if (v > tilde[i]) {
          direction = 1.0;
        } else if (v < tilde[i]) {
          direction = -1.0;
        }
        // Per ORIGINAL unit: x_original = s x_canonical.
        delta[ic.canonical] = direction * ic.scale;
        delta_norm += delta[ic.canonical] * delta[ic.canonical];
      }
      delta_norm = std::sqrt(delta_norm);
      const Real weight = c_norm > 0.0 ? alpha * delta_norm / c_norm : 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        p.c[j] = (1.0 - alpha) * delta[j] + weight * true_costs[j];
        p.col_lower[j] = root_lower_[j];
        p.col_upper[j] = root_upper_[j];
      }
      auto next = simplex::solve_primal_simplex(p, lp_options, &x_bar.basis);
      for (std::size_t j = 0; j < n; ++j) p.c[j] = true_costs[j];
      if (!next.has_value()) return next.error();
      ++stats_.pump_rounds;
      stats_.pump_lp_iterations += next->iterations;
      if (next->status != SolverStatus::Optimal) return core::Status::Ok();
      previous = tilde;
      x_bar = std::move(*next);

      Real distance = 0.0;
      Real fractionality = 0.0;
      for (std::size_t i = 0; i < S.size(); ++i) {
        const Real v = value(x_bar, S[i]);
        distance += std::fabs(v - previous[i]);
        fractionality += std::fabs(v - round_half_up(v));
      }
      if (stage == 1 && distance < best_stage1_distance) {
        best_stage1_distance = distance;
        best_stage1 = x_bar;
      }
      // Step 13: the LP point already IS the rounding.
      if (distance <= m.integer_tolerance) {
        if (stage == 2) {
          bool accepted = false;
          std::vector<Real> x(x_bar.x.begin(), x_bar.x.begin() + static_cast<std::ptrdiff_t>(n));
          if (const auto st = offer_point(x_bar, x, &accepted); !st.ok()) return st;
          if (accepted) ++stats_.pump_solutions;
          return core::Status::Ok();
        }
        break;
      }
      // Step 14: stalling -- not 10% less fractional than maxStalls rounds ago.
      fractionality_history.push_back(fractionality);
      if (fractionality_history.size() > max_stalls &&
          fractionality >
              0.9 * fractionality_history[fractionality_history.size() - 1 - max_stalls]) {
        break;
      }
      for (std::size_t i = 0; i < S.size(); ++i) tilde[i] = round_half_up(value(x_bar, S[i]));
    }
  }
  return core::Status::Ok();
}

// --------------------------------------------------------------------------
// RENS -- [CIP] 9.1.1, Berthold 2006 section 3.2.1 and Algorithm 4
// --------------------------------------------------------------------------

bool BranchAndBound::to_canonical(const Solution& original, std::vector<Real>& x) const {
  const model::CanonicalProblem& p = canon_.problem;
  const std::size_t n = p.num_cols();
  if (original.x.size() != working_.num_cols()) return false;
  x.assign(n, 0.0);
  std::vector<std::size_t> unmapped;
  for (std::size_t j = 0; j < n; ++j) {
    // `fold_[j]` is one term per plain column folded into this one -- exactly
    // one for a column stage B left alone, two or more for a merge survivor,
    // whose value is the same combination presolve formed.
    Real v = 0.0;
    bool mapped = !fold_[j].empty();
    for (const auto& [pc, coef] : fold_[j]) {
      if (original_of_plain_[pc] == kNoColumn) {
        mapped = false;
        break;
      }
      v += coef * original.x[original_of_plain_[pc]] / scale_of_plain_[pc];
    }
    if (!mapped) {
      unmapped.push_back(j);
      continue;
    }
    x[j] = v;
  }
  // A range column t of `a'x + t = b` (Canonical.hpp) has no original; it is
  // whatever makes its own equality row hold.
  const auto& csc = p.A.csc;
  const auto& csr = p.A.csr;
  for (const std::size_t j : unmapped) {
    if (csc.slice_end(j) - csc.slice_begin(j) != 1) return false;
    const auto row = static_cast<std::size_t>(csc.indices()[csc.slice_begin(j)]);
    const Real coef = csc.values()[csc.slice_begin(j)];
    if (row >= p.num_equality || coef == 0.0) return false;
    Real rest = 0.0;
    for (std::size_t q = csr.slice_begin(row); q < csr.slice_end(row); ++q) {
      const auto k = static_cast<std::size_t>(csr.indices()[q]);
      if (k != j) rest += csr.values()[q] * x[k];
    }
    x[j] = (p.b[row] - rest) / coef;
  }
  return true;
}

/// "Create a sub-MIP of the original MIP by changing the bounds of all
/// integer variables to l_j = floor(x_j) and u_j = ceil(x_j)" ([B] 3.2.1),
/// x the root LP optimum -- so an integer column integral in x is FIXED -- and
/// solve it. Every feasible point of the sub-MIP is a rounding of x, and its
/// optimum is the best rounding any pure rounding heuristic could produce.
///
/// The sub-MIP is solved by THIS branch-and-bound (solve_milp), on a copy of
/// the model with the integer bounds changed, and with RENS and the pump off
/// inside it; root cuts are also off, as [B] ran the sub-problem with
/// "expensive presolving strategies and heuristics ... deactivated". The
/// "after presolving" size test uses this project's canonicalization, which
/// substitutes out every fixed column.
core::Status BranchAndBound::rens(const SimplexResult& root) {
  const auto& m = options_.milp;
  model::Problem sub = working_.clone();
  std::size_t fractional_count = 0;
  for (std::size_t k = 0; k < integers_.size(); ++k) {
    const IntegerColumn& ic = integers_[k];
    const Real v = ic.scale * root.x[ic.canonical];
    Real lo = std::floor(v);
    Real hi = std::ceil(v);
    if (std::fabs(v - std::round(v)) <= m.integer_tolerance) {
      lo = hi = std::round(v);
    } else {
      ++fractional_count;
    }
    sub.col_lower[ic.original] = std::max(sub.col_lower[ic.original], lo);
    sub.col_upper[ic.original] = std::min(sub.col_upper[ic.original], hi);
  }
  if (static_cast<Real>(fractional_count) >
      m.rens_max_fractional_ratio * static_cast<Real>(integers_.size())) {
    return core::Status::Ok();
  }
  auto sub_canon = model::canonicalize(sub, options_);
  if (!sub_canon.has_value()) return core::Status::Ok();  // e.g. proven empty
  if (static_cast<Real>(sub_canon->problem.num_cols()) >
      (1.0 - m.rens_min_reduction) * static_cast<Real>(canon_.problem.num_cols())) {
    return core::Status::Ok();
  }

  model::Options sub_options = options_;
  sub_options.milp.rens = false;
  sub_options.milp.feasibility_pump = false;
  sub_options.milp.root_cuts = false;
  sub_options.milp.node_limit = m.rens_node_limit;
  sub_options.milp.stall_node_limit = m.rens_stall_nodes;
  sub_options.milp.time_limit_seconds = std::max(0.0, m.time_limit_seconds - elapsed());
  MilpStatistics sub_stats;
  auto result = solve_milp(sub, sub_options, &sub_stats);
  stats_.rens_nodes += sub_stats.nodes;
  if (!result.has_value()) return core::Status::Ok();  // a failed heuristic is not an error
  if (sub_stats.incumbents == 0) return core::Status::Ok();

  std::vector<Real> x;
  if (!to_canonical(*result, x)) return core::Status::Ok();
  bool accepted = false;
  if (const auto st = offer_point(root, x, &accepted); !st.ok()) return st;
  if (accepted) ++stats_.rens_solutions;
  return core::Status::Ok();
}

core::Expected<Solution> BranchAndBound::run() {
  const auto& m = options_.milp;
  const std::size_t n = canon_.problem.num_cols();

  OpenNodes open;
  {
    Node root;
    root.lower.resize(n);
    root.upper.resize(n);
    for (std::size_t j = 0; j < n; ++j) {
      root.lower[j] = canon_.problem.col_lower[j];
      root.upper[j] = canon_.problem.col_upper[j];
    }
    root.bound = -kInf;
    root.estimate = -kInf;
    root.warm = root_basis_;
    open.insert(std::move(root));
  }

  bool closed = false;
  Real closing_bound = 0.0;

  auto dominated = [this](Real bound) { return this->dominated(bound); };

  // PLUNGING, [CIP] section 6.3. `children` are the open children of the node
  // just processed, `siblings` the open siblings of it. A plunge continues
  // with a child, else a sibling, and ends -- returning to the leaf queue --
  // when neither is left or the abort test below fires.
  std::vector<OpenNodes::Id> children;
  std::vector<OpenNodes::Id> siblings;
  std::size_t plunge_steps = 0;  // steps in the CURRENT plunge
  std::size_t plunges = 0;       // plunges started, for `best_frequency`
  std::size_t max_depth = 0;     // `dmax`, over processed nodes
  const bool plunging = m.node_selection == NodeSelection::Interleaved;

  auto next_node = [&]() -> OpenNodes::Id {
    if (plunging) {
      // "During each plunge, we perform a certain minimal number of plunging
      // steps, but we abort the plunging after a certain total number of steps
      // or if the local relative gap gamma(Q) = (c_Q - c_lower)/(c_upper -
      // c_lower) of the current subproblem Q exceeds ... 0.25", with the
      // minimum and maximum "0.1 dmax and 0.5 dmax". With no incumbent
      // `c_upper` is infinite and gamma is 0, so only the maximum applies.
      const auto dmax = static_cast<Real>(max_depth);
      const auto min_steps = static_cast<std::size_t>(m.plunge_min_depth_fraction * dmax);
      const auto max_steps = static_cast<std::size_t>(m.plunge_max_depth_fraction * dmax);
      auto admissible = [&](OpenNodes::Id id) {
        if (!open.contains(id)) return false;
        if (plunge_steps >= max_steps) return false;
        if (plunge_steps < min_steps || !have_incumbent_) return true;
        const Real lower = open.lower_bound();
        const Real width = incumbent_objective_ - lower;
        if (!(width > 0.0)) return false;
        return (open.at(id).bound - lower) / width <= m.plunge_max_gap;
      };
      for (const auto* pool : {&children, &siblings}) {
        for (OpenNodes::Id id : *pool) {
          if (admissible(id)) {
            ++plunge_steps;
            ++stats_.plunge_steps;
            return id;
          }
        }
      }
      // The plunge is over: a leaf from the queue starts the next one.
      // [CIP] section 6.6: every `bestfreq`-th a best-BOUND leaf, so the
      // global dual bound keeps moving; otherwise the best estimate.
      plunge_steps = 0;
      ++plunges;
      if (m.best_frequency == 0 || plunges % m.best_frequency != 0) {
        return open.best_estimate();
      }
    }
    return open.best_bound();
  };

  while (!open.empty()) {
    if (stats_.nodes >= m.node_limit || elapsed() >= m.time_limit_seconds) break;
    if (m.stall_node_limit > 0 && have_incumbent_ &&
        stats_.nodes - last_improvement_node_ >= m.stall_node_limit) {
      break;
    }

    // The global lower bound is the smallest bound of any open node, whatever
    // order they are PROCESSED in. Once it cannot beat the incumbent (or is
    // within the gap) nothing open can -- which is what makes stopping a proof.
    if (dominated(open.lower_bound())) {
      closed = true;
      closing_bound = std::min(open.lower_bound(), incumbent_objective_);
      break;
    }

    const OpenNodes::Id id = next_node();
    Node node = open.take(id);
    siblings.clear();
    for (OpenNodes::Id c : children) {
      if (c != id && open.contains(c)) siblings.push_back(c);
    }
    children.clear();

    // Selected out of bound order, so it can be dominated while the global
    // bound is not. Discarding it is pruning by bound, and safe.
    if (dominated(node.bound)) continue;

    // [CIP] chapter 7: the node's bounds, intersected with the global ones,
    // propagated to a fixed point before the LP sees them. A node proven empty
    // here is pruned exactly as an infeasible LP would prune it.
    // [CIP] chapter 11: this node's own bound changes, on top of the trail
    // it inherited. Recorded only when conflicts are analyzed.
    const bool conflicts_on = m.propagation && m.conflict_analysis;
    std::vector<BoundChange> local;
    TrailWriter writer{&local, node.trail ? node.trail->end() : 0, node.depth};
    if (m.propagation) {
      std::size_t empty_col = n;
      for (std::size_t j = 0; j < n; ++j) {
        if (global_lower_[j] > node.lower[j]) {
          if (conflicts_on) {
            local.push_back(BoundChange{j, false, global_lower_[j], node.lower[j], node.depth,
                                        Reason::Global, 0, writer.size(), false});
          }
          node.lower[j] = global_lower_[j];
        }
        if (global_upper_[j] < node.upper[j]) {
          if (conflicts_on) {
            local.push_back(BoundChange{j, true, global_upper_[j], node.upper[j], node.depth,
                                        Reason::Global, 0, writer.size(), false});
          }
          node.upper[j] = global_upper_[j];
        }
        if (empty_col == n && node.lower[j] > node.upper[j]) empty_col = j;
      }
      Cause cause;
      bool feasible = empty_col == n;
      if (!feasible) {
        cause = Cause{Cause::Kind::Var, empty_col, writer.size()};
      } else {
        feasible = propagate(node.lower, node.upper, conflicts_on ? &writer : nullptr, &cause);
      }
      if (!feasible) {
        ++stats_.propagation_cutoffs;
        if (conflicts_on && cause.kind != Cause::Kind::None) {
          const auto trail = flatten(node.trail, local);
          analyze_conflict(trail, propagation_conflict(trail, cause), cause_uses_cutoff(cause));
        }
        continue;
      }
    }

    ++stats_.nodes;
    max_depth = std::max(max_depth, node.depth);
    auto lp = solve_with(node.lower, node.upper, node_lp_, node.warm.get());
    if (!lp.has_value()) return lp.error();
    stats_.node_lp_iterations += lp->iterations;
    ++solved_node_lps_;

    if (lp->status == SolverStatus::Infeasible) {
      if (conflicts_on && !lp->infeasibility_certificate.empty()) {
        const auto trail = flatten(node.trail, local);
        // The node LP carries no objective row: its infeasibility proof is
        // cutoff-free (reduced-cost bounds in the set still mark it).
        analyze_conflict(trail,
                         lp_conflict(trail, lp->infeasibility_certificate, node.lower, node.upper),
                         false);
      }
      continue;
    }
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

    if (!have_root_) {
      have_root_ = true;
      root_objective_ = lp->objective;
      root_x_.assign(lp->x.begin(), lp->x.begin() + static_cast<std::ptrdiff_t>(n));
      root_reduced_cost_.assign(lp->reduced_cost.begin(),
                                lp->reduced_cost.begin() + static_cast<std::ptrdiff_t>(n));
    }
    if (root_values_.empty()) {
      root_values_.resize(integers_.size());
      for (std::size_t k = 0; k < integers_.size(); ++k) {
        root_values_[k] = integers_[k].scale * lp->x[integers_[k].canonical];
      }
    }

    if (active_min_.empty()) {
      active_min_.assign(n, kInf);
      active_max_.assign(n, -kInf);
    }
    for (std::size_t j = 0; j < n; ++j) {
      active_min_[j] = std::min(active_min_[j], lp->x[j]);
      active_max_[j] = std::max(active_max_[j], lp->x[j]);
    }

    // [AKM] section 2.2: the gain per unit, measured on the child actually
    // solved, updates the pseudocost of the variable that created it.
    if (node.origin.present && node.origin.fraction > 0.0) {
      const Real gain = std::max(0.0, lp->objective - node.origin.parent_objective);
      pseudocosts_.record(node.origin.column, node.origin.up, gain / node.origin.fraction);
    }

    if (dominated(lp->objective)) continue;

    std::vector<Candidate> candidates = fractional(*lp);
    if (candidates.empty()) {
      if (const auto st = offer_incumbent(*lp, lp->objective); !st.ok()) return st.error();
      continue;
    }

    // [CIP] chapter 9. Simple rounding "is applied after the solving of every
    // LP"; a dive follows while the iteration quota allows. Both only ever
    // ADD an incumbent -- the node itself is branched on as before.
    if (m.heuristics) {
      if (const auto st = simple_rounding(*lp, &stats_.rounding_solutions); !st.ok()) {
        return st.error();
      }
      if (stats_.nodes == 1) {
        // Root only, both ([CIP] 9.1.1 for RENS; the pump as a start
        // heuristic, so only while nothing has been found).
        if (m.feasibility_pump && !have_incumbent_) {
          if (const auto st = feasibility_pump(*lp); !st.ok()) return st.error();
        }
        if (m.rens) {
          if (const auto st = rens(*lp); !st.ok()) return st.error();
        }
        if (dominated(lp->objective)) continue;
      }
      if (const auto st = dive(node, *lp); !st.ok()) return st.error();
      if (dominated(lp->objective)) continue;
    }

    const std::size_t pick = select(candidates, node, *lp);
    const Candidate& c = candidates[pick];
    if (c.down_infeasible && c.up_infeasible) continue;  // the node is empty

    // [CIP] 8.8, reduced cost strengthening, on this node's LP: a nonbasic
    // column at its lower bound with r_j > 0 cannot rise above
    // l_j + (c^ - c)/r_j in any solution better than the incumbent c^ (the LP
    // bound of such a point would already reach c^); r_j < 0 at the upper
    // bound likewise. Valid for this node's subtree -- installed on the node's
    // bounds, which both children copy. Installed "only if the variable is of
    // integer type or if at least 20% of the local domain of a continuous
    // variable is cut off", and for a continuous one only if it cuts into the
    // column's active region.
    if (m.propagation && have_incumbent_) {
      const Real gap = incumbent_objective_ - lp->objective;
      const Real tol = options_.simplex.dual_feasibility_tolerance;
      for (std::size_t j = 0; j < n && gap >= 0.0; ++j) {
        const auto st = lp->basis.status[j];
        const Real r = lp->reduced_cost[j];
        const Real s = integer_scale_[j];
        const Real width = node.upper[j] - node.lower[j];
        if (st == simplex::VarStatus::AtLower && r > tol) {
          Real u = node.lower[j] + gap / r;
          if (s != 0.0) {
            u = std::floor(s * u + kPropTolerance) / s;
          } else if (!(is_finite_bound(width) && node.upper[j] - u >= 0.2 * width &&
                       u < active_max_[j])) {
            continue;
          }
          if (u < node.upper[j]) {
            if (conflicts_on) {
              local.push_back(BoundChange{j, true, u, node.upper[j], node.depth, Reason::Leaf, 0,
                                          writer.size(), false});
            }
            node.upper[j] = u;
            ++stats_.local_redcost_tightenings;
          }
        } else if (st == simplex::VarStatus::AtUpper && r < -tol) {
          Real l = node.upper[j] + gap / r;
          if (s != 0.0) {
            l = std::ceil(s * l - kPropTolerance) / s;
          } else if (!(is_finite_bound(width) && l - node.lower[j] >= 0.2 * width &&
                       l > active_min_[j])) {
            continue;
          }
          if (l > node.lower[j]) {
            if (conflicts_on) {
              local.push_back(BoundChange{j, false, l, node.lower[j], node.depth, Reason::Leaf, 0,
                                          writer.size(), false});
            }
            node.lower[j] = l;
            ++stats_.local_redcost_tightenings;
          }
        }
      }
    }

    const Real node_estimate = estimate(candidates, lp->objective);
    const IntegerColumn& ic = integers_[c.column];
    const auto basis = std::make_shared<const Basis>(lp->basis);

    // Martin's rule, [CIP] section 6.1: plunge first into the child that
    // pushes the variable FURTHER from its root LP value -- "on the path to
    // the current node the value of the variable has the tendency to be
    // pushed" that way. Section 6.3 applies it during plunging. The thesis
    // gives no rule for a value equal to the root's; up is taken then.
    const bool up_first = c.value >= root_values_[c.column];
    std::shared_ptr<const TrailSegment> own_segment;
    if (conflicts_on) {
      auto seg = std::make_shared<TrailSegment>();
      seg->parent = node.trail;
      seg->base = node.trail ? node.trail->end() : 0;
      seg->changes = std::move(local);
      own_segment = std::move(seg);
    }
    for (int side = 0; side < 2; ++side) {
      const bool up = side == 0 ? up_first : !up_first;
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
      child.estimate = node_estimate;
      child.depth = node.depth + 1;
      child.warm = basis;
      child.origin = Origin{c.column, up, up ? c.frac_up : c.frac_down, lp->objective, true};
      if (conflicts_on) {
        // The branching decision: the first vertex of the child's depth level.
        auto seg = std::make_shared<TrailSegment>();
        seg->parent = own_segment;
        seg->base = own_segment->end();
        const std::size_t j = ic.canonical;
        seg->changes.push_back(up ? BoundChange{j, false, child.lower[j], node.lower[j],
                                                child.depth, Reason::Branch, 0, seg->base, false}
                                  : BoundChange{j, true, child.upper[j], node.upper[j],
                                                child.depth, Reason::Branch, 0, seg->base, false});
        child.trail = std::move(seg);
      }
      children.push_back(open.insert(std::move(child)));
    }
  }

  // Map a canonical bound to the original objective. The canonical objective
  // is the original's, negated for a maximization, and differs from it by a
  // constant: every column substituted out -- by the canonicalizer, by the LP
  // presolver, or by stage B of the MIP presolve -- folded its contribution
  // into `obj_offset`, and the reader's objective row contributes
  // `obj_constant`. `incumbent_objective_` is c'x on the REDUCED model with
  // neither term, so
  //
  //     original = sign * (c'x + obj_offset) + obj_constant.
  //
  // Taking the constant from the incumbent instead would give the same number
  // whenever there is one, and zero -- which is wrong by exactly this
  // constant -- when a limit is hit before any solution is found.
  const bool maximize = working_.sense == core::ObjSense::Maximize;
  const Real sign = maximize ? -1.0 : 1.0;
  const Real offset = sign * canon_.problem.obj_offset + working_.obj_constant;
  auto to_original = [sign, offset](Real canonical) { return sign * canonical + offset; };

  Solution result;
  if (have_incumbent_) result = std::move(incumbent_);

  const bool sound = stats_.unreliable_nodes == 0;
  if (closed && sound) {
    result.status = SolverStatus::Optimal;
    result.best_bound = to_original(closing_bound);
  } else if (open.empty() && sound) {
    // Every node was solved or pruned on a trustworthy bound.
    result.status = have_incumbent_ ? SolverStatus::Optimal : SolverStatus::Infeasible;
    result.best_bound = have_incumbent_ ? result.objective : 0.0;
  } else {
    // A limit, or an unexplored subtree: report what is known, never a
    // verdict the search did not earn.
    result.status = SolverStatus::NotConverged;
    result.best_bound = open.empty() ? (have_incumbent_ ? result.objective : 0.0)
                                     : to_original(open.lower_bound());
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
  auto* const conflict_log = stats.conflict_log;  // a hook the caller set, not a statistic
  stats = MilpStatistics{};
  stats.conflict_log = conflict_log;

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

  // ---- Root cutting planes (stage 8b): cut-and-branch, [CIP] 8.10. ----
  // Rounds of: solve the LP, separate GMI and c-MIR cuts against it, select
  // by [CIP] Algorithm 3.2, append the selected cuts as rows. Stops at
  // `cut_rounds` (Wolter's MAXROUNDS = 15), an integral or non-optimal LP, or
  // a round that selects nothing.
  std::optional<model::CanonicalResult> plain;
  auto make_plain = [&]() -> core::Status {
    if (plain.has_value()) return core::Status::Ok();
    // The unmodified canonical model, for rebuilding solutions. Canonicalization
    // and scaling are deterministic, so this is the model before presolve and
    // cuts.
    auto again = model::canonicalize(working, options);
    if (!again.has_value()) return again.error();
    if (const auto st = scale(again->problem, options, again->transforms); !st.ok()) return st;
    plain = std::move(*again);
    return core::Status::Ok();
  };

  // ---- MIP presolve (Module 29): [CIP] chapter 10, [AGH]. ----
  // Stage B removes columns, so the presolved model no longer shares a column
  // space with `plain`. `postsolve` is the map back, and it must outlive the
  // search that produces points in the reduced space.
  CanonicalPostsolve postsolve;
  if (options.milp.presolve && !integers.empty()) {
    if (const auto st = make_plain(); !st.ok()) return st.error();
    std::vector<Real> integer_scale(canon->problem.num_cols(), 0.0);
    for (const IntegerColumn& ic : integers) integer_scale[ic.canonical] = ic.scale;
    CanonicalPresolveStats ps;
    const auto presolve_started = std::chrono::steady_clock::now();
    const auto st = presolve_canonical(canon->problem, integer_scale,
                                       options.milp.presolve_rounds,
                                       options.milp.presolve_columns, postsolve, ps);
    stats.presolve_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - presolve_started)
            .count();
    stats.presolve_rounds = ps.rounds;
    stats.presolve_bounds = ps.bounds_tightened;
    stats.presolve_coefficients = ps.coefficients_tightened;
    stats.presolve_rows_removed = ps.rows_removed;
    stats.presolve_fixed = ps.columns_fixed;
    stats.presolve_substituted = ps.columns_substituted;
    stats.presolve_merged = ps.columns_merged;
    if (!st.ok()) {
      if (st.error().code == core::ErrorCode::PrimalInfeasible) {
        return verdict_solution(problem, SolverStatus::Infeasible);
      }
      return st.error();
    }
    // Stage B renumbers the columns. It never removes an INTEGER column
    // (MilpCanonicalPresolve.hpp says why), so every one of these survives and
    // this is a pure remap -- but assert that rather than trusting it, because
    // a stale index here would branch on the wrong variable in silence.
    for (IntegerColumn& ic : integers) {
      const std::size_t moved = postsolve.new_of_old[ic.canonical];
      if (moved == CanonicalPostsolve::kRemoved) {
        return core::make_error(core::ErrorCode::DimensionMismatch,
                                "MIP presolve removed an integer column");
      }
      ic.canonical = moved;
    }
  }

  std::optional<simplex::Basis> cut_basis;
  if ((options.milp.gomory_cuts || options.milp.cmir_cuts) && !integers.empty()) {
    const std::size_t n = canon->problem.num_cols();
    std::vector<Real> integer_scale(n, 0.0);
    for (const IntegerColumn& ic : integers) integer_scale[ic.canonical] = ic.scale;
    std::vector<Real> lo(canon->problem.col_lower.data(), canon->problem.col_lower.data() + n);
    std::vector<Real> hi(canon->problem.col_upper.data(), canon->problem.col_upper.data() + n);
    SeparationInput in{&canon->problem, &integer_scale, &lo, &hi,
                       options.milp.cut_violation_margin};
    CmirState cmir_state;
    model::Options lp_options = options;
    lp_options.simplex.method = model::Method::DualSimplex;
    for (std::size_t round = 0; round < options.milp.cut_rounds; ++round) {
      auto lp = simplex::solve_simplex(canon->problem, lp_options,
                                       cut_basis.has_value() ? &*cut_basis : nullptr);
      if (!lp.has_value()) return lp.error();
      if (lp->status != SolverStatus::Optimal) break;
      // The CANONICAL (minimization) objective, INCLUDING obj_offset -- not
      // the original one, so that "cuts never loosen the root bound" stays a
      // >= for a maximization too. The offset has to be in it: `lp->objective`
      // is c'x on the PRESOLVED model, and presolve stage B folds a
      // substituted column's contribution into obj_offset, so without it the
      // two arms of a stage B A/B differ by that constant and the reduced one
      // reads as a weaker relaxation when it is nothing of the kind.
      const Real root = lp->objective + canon->problem.obj_offset;
      if (round == 0) stats.root_bound_before_cuts = root;
      stats.root_bound_after_cuts = root;
      bool integral = true;
      for (const IntegerColumn& ic : integers) {
        const Real v = ic.scale * lp->x[ic.canonical];
        if (std::fabs(v - std::round(v)) > options.milp.integer_tolerance) {
          integral = false;
          break;
        }
      }
      if (integral) break;

      std::vector<Cut> cuts;
      if (options.milp.gomory_cuts) {
        auto g = separate_gomory(in, *lp);
        stats.gomory_cuts += g.size();
        for (auto& c : g) cuts.push_back(std::move(c));
      }
      if (options.milp.cmir_cuts) {
        auto c = separate_cmir(in, *lp, round, cmir_state);
        stats.cmir_cuts += c.size();
        for (auto& k : c) cuts.push_back(std::move(k));
      }
      const std::vector<Real> x(lp->x.begin(), lp->x.begin() + static_cast<std::ptrdiff_t>(n));
      // [CIP] 3.3.8: at most 2000 cuts enter per root round.
      std::vector<Cut> chosen = select_cuts(std::move(cuts), canon->problem, x, 2000);
      if (chosen.empty()) break;

      if (const auto st = make_plain(); !st.ok()) return st.error();
      simplex::Basis basis = lp->basis;
      if (const auto st = append_cuts(canon->problem, chosen, &basis); !st.ok()) {
        return st.error();
      }
      cut_basis = std::move(basis);
      stats.cuts_added += chosen.size();
      ++stats.cut_rounds;
    }
  }

  BranchAndBound search(working, *canon, std::move(integers), options, stats, &postsolve);
  if (plain.has_value()) search.set_reconstruction(&*plain);
  if (cut_basis.has_value()) search.set_root_basis(std::move(*cut_basis));
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
