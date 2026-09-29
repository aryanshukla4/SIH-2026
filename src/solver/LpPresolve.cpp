#include "sovsolve/solver/LpPresolve.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sovsolve/core/SparseBuilder.hpp"

namespace sovsolve::solver {

namespace {

using core::ErrorCode;
using core::Index;
using core::INF;
using core::is_finite_bound;
using core::Real;
using core::Status;
using Rec = LpPostsolveRecord;

/// Reductions act on a violation at most this large (relative to the row) --
/// the same scale as `Tolerances::bound_violation`.
constexpr Real kActTol = 1e-9;

/// A PrimalInfeasible verdict needs a violation this large. Everything between
/// the two is left for the engine: a missed verdict costs a solve, a false one
/// is a wrong answer.
constexpr Real kVerdictTol = 1e-6;

/// Strict dual dominance needs this margin ([AA95] (26)), relative to |c_j|.
constexpr Real kDominanceTol = 1e-7;

/// [AGH] 4.5's Markowitz guard: a pivot this small against the largest entry
/// of its row amplifies rounding into the recovered value.
constexpr Real kPivotRatio = 0.01;

/// Relative agreement for two columns to count as parallel.
constexpr Real kParallelTol = 1e-12;

constexpr std::uint8_t kFlagEquality = 1;   ///< Substitute: row was an equality
constexpr std::uint8_t kFlagUpper = 1;      ///< SingletonIneq: the new bound is an upper one
constexpr std::uint8_t kFlagAtMax = 1;      ///< Forcing: columns sit at the max-activity bounds
constexpr std::uint8_t kFlagLowerFrom = 1;  ///< Doubleton: col2's lower bound came from col
constexpr std::uint8_t kFlagUpperFrom = 2;  ///< Doubleton: col2's upper bound came from col

/// Activity range of a sum `sum a_k x_k` over bounded x, with the infinite
/// contributions counted separately so one can be excluded later.
struct Activity {
  Real min = 0.0, max = 0.0;
  std::size_t min_inf = 0, max_inf = 0;
};

class Presolve {
 public:
  Presolve(const model::CanonicalProblem& p, LpPresolveStats& stats,
           std::vector<Rec>& records, std::vector<std::pair<std::size_t, Real>>& entries)
      : p_(p),
        csr_(p.A.csr),
        csc_(p.A.csc),
        m_(p.num_rows()),
        n_(p.num_cols()),
        stats_(stats),
        records_(records),
        entries_(entries) {
    b_.assign(p.b.data(), p.b.data() + m_);
    c_.assign(p.c.data(), p.c.data() + n_);
    lo_.assign(p.col_lower.data(), p.col_lower.data() + n_);
    hi_.assign(p.col_upper.data(), p.col_upper.data() + n_);
    row_on_.assign(m_, 1);
    col_on_.assign(n_, 1);
    row_cnt_.assign(m_, 0);
    col_cnt_.assign(n_, 0);
    for (std::size_t i = 0; i < m_; ++i) row_cnt_[i] = csr_.slice_nnz(i);
    for (std::size_t j = 0; j < n_; ++j) col_cnt_[j] = csc_.slice_nnz(j);

    // Range columns (Canonical.hpp) and the rows they sit in are left alone:
    // removing either would change what the range means.
    col_protect_.assign(n_, 0);
    row_protect_.assign(m_, 0);
    for (std::size_t j = n_ - p.num_range; j < n_; ++j) {
      col_protect_[j] = 1;
      for (auto k = csc_.slice_begin(j); k < csc_.slice_end(j); ++k) {
        row_protect_[static_cast<std::size_t>(csc_.indices()[k])] = 1;
      }
    }
  }

  Status run() {
    for (std::size_t round = 0; round < kMaxRounds; ++round) {
      changed_ = false;
      if (Status st = row_pass(); !st.ok()) return st;
      if (Status st = column_pass(); !st.ok()) return st;
      dominated_column_pass();
      parallel_column_pass();
      ++stats_.rounds;
      if (!changed_) return Status::Ok();
    }
    // Out of rounds: finish with the reductions that cannot chain, so the
    // startability contract (no empty row, no fixed column) still holds.
    for (;;) {
      changed_ = false;
      if (Status st = cleanup_pass(); !st.ok()) return st;
      if (!changed_) return Status::Ok();
    }
  }

  // -- results ------------------------------------------------------------
  Real obj_delta() const { return obj_delta_; }
  const std::vector<Real>& cost() const { return c_; }
  const std::vector<Real>& rhs() const { return b_; }
  const std::vector<Real>& lower() const { return lo_; }
  const std::vector<Real>& upper() const { return hi_; }
  bool row_on(std::size_t i) const { return row_on_[i] != 0; }
  bool col_on(std::size_t j) const { return col_on_[j] != 0; }

 private:
  static constexpr std::size_t kMaxRounds = 200;

  bool is_eq(std::size_t i) const { return i < p_.num_equality; }

  Real row_scale(std::size_t i) const { return 1.0 + std::fabs(b_[i]); }

  template <typename F>
  void for_row(std::size_t i, F&& f) const {
    for (auto k = csr_.slice_begin(i); k < csr_.slice_end(i); ++k) {
      const auto j = static_cast<std::size_t>(csr_.indices()[k]);
      if (col_on_[j]) f(j, csr_.values()[k]);
    }
  }

  template <typename F>
  void for_col(std::size_t j, F&& f) const {
    for (auto k = csc_.slice_begin(j); k < csc_.slice_end(j); ++k) {
      const auto i = static_cast<std::size_t>(csc_.indices()[k]);
      if (row_on_[i]) f(i, csc_.values()[k]);
    }
  }

  /// Activity of row `i` over its active columns, skipping column `skip`.
  Activity activity(std::size_t i, std::size_t skip = static_cast<std::size_t>(-1)) const {
    Activity a;
    for_row(i, [&](std::size_t j, Real v) {
      if (j == skip) return;
      const Real at_min = v > 0.0 ? lo_[j] : hi_[j];
      const Real at_max = v > 0.0 ? hi_[j] : lo_[j];
      if (is_finite_bound(at_min)) a.min += v * at_min; else ++a.min_inf;
      if (is_finite_bound(at_max)) a.max += v * at_max; else ++a.max_inf;
    });
    return a;
  }

  // -- primitive edits ---------------------------------------------------

  void remove_row(std::size_t i) {
    row_on_[i] = 0;
    for_row(i, [&](std::size_t j, Real) { --col_cnt_[j]; });
    changed_ = true;
  }

  /// Remove column `j` at `value`, folding it into every active row and the
  /// objective ([AA95] (iv)).
  void fix_column(std::size_t j, Real value) {
    for_col(j, [&](std::size_t i, Real v) {
      b_[i] -= v * value;
      --row_cnt_[i];
    });
    obj_delta_ += c_[j] * value;
    col_on_[j] = 0;
    Rec r;
    r.kind = Rec::Kind::FixColumn;
    r.col = j;
    r.value = value;
    records_.push_back(r);
    ++stats_.fixed_columns;
    changed_ = true;
  }

  void drop_row(std::size_t i) {
    Rec r;
    r.kind = Rec::Kind::DropRow;
    r.row = i;
    records_.push_back(r);
    remove_row(i);
  }

  Status infeasible(const std::string& what, std::size_t index) const {
    return core::make_error(ErrorCode::PrimalInfeasible,
                            "lp presolve: " + what + " " + std::to_string(index));
  }

  /// After a bound moved: a column whose box collapsed is fixed, one whose
  /// box crossed by more than the verdict tolerance proves infeasibility.
  Status settle_bounds(std::size_t j) {
    if (!is_finite_bound(lo_[j]) || !is_finite_bound(hi_[j])) return Status::Ok();
    const Real width = hi_[j] - lo_[j];
    const Real scale = 1.0 + std::max(std::fabs(lo_[j]), std::fabs(hi_[j]));
    if (width < -kVerdictTol * scale) return infeasible("crossed bounds on column", j);
    if (width <= kActTol * scale) fix_column(j, width < 0.0 ? lo_[j] : 0.5 * (lo_[j] + hi_[j]));
    return Status::Ok();
  }

  // -- rows ----------------------------------------------------------------

  Status row_pass() {
    for (std::size_t i = 0; i < m_; ++i) {
      if (!row_on_[i] || row_protect_[i]) continue;
      if (row_cnt_[i] == 0) {
        if (Status st = empty_row(i); !st.ok()) return st;
      } else if (row_cnt_[i] == 1) {
        if (Status st = singleton_row(i); !st.ok()) return st;
      } else {
        if (Status st = activity_row(i); !st.ok()) return st;
      }
    }
    return Status::Ok();
  }

  /// [AA95] (i). An equality needs `0 = b`; an inequality `0 <= b`.
  Status empty_row(std::size_t i) {
    const Real tol = kVerdictTol * row_scale(i);
    if (is_eq(i) ? std::fabs(b_[i]) > tol : b_[i] < -tol) return infeasible("empty row", i);
    drop_row(i);
    ++stats_.empty_rows;
    return Status::Ok();
  }

  /// [AA95] (v).
  Status singleton_row(std::size_t i) {
    std::size_t j = 0;
    Real a = 0.0;
    for_row(i, [&](std::size_t col, Real v) { j = col; a = v; });
    if (col_protect_[j]) return Status::Ok();

    if (is_eq(i)) {
      const Real value = b_[i] / a;
      const Real scale = 1.0 + std::fabs(value);
      if ((is_finite_bound(lo_[j]) && value < lo_[j] - kVerdictTol * scale) ||
          (is_finite_bound(hi_[j]) && value > hi_[j] + kVerdictTol * scale)) {
        return infeasible("singleton equality outside the bounds of its column, row", i);
      }
      Rec r;
      r.kind = Rec::Kind::SingletonEq;
      r.row = i;
      r.col = j;
      r.a = a;
      records_.push_back(r);
      remove_row(i);
      fix_column(j, value);
      ++stats_.singleton_rows;
      return Status::Ok();
    }

    // a x_j <= b: an upper bound on x_j when a > 0, a lower one when a < 0.
    const Real bound = b_[i] / a;
    const bool upper = a > 0.0;
    const Real current = upper ? hi_[j] : lo_[j];
    const Real scale = 1.0 + std::fabs(bound);
    const bool tighter = upper ? (!is_finite_bound(current) || bound < current - kActTol * scale)
                               : (!is_finite_bound(current) || bound > current + kActTol * scale);
    if (!tighter) {
      drop_row(i);  // implied by the column's own bound
      ++stats_.redundant_rows;
      return Status::Ok();
    }
    Rec r;
    r.kind = Rec::Kind::SingletonIneq;
    r.row = i;
    r.col = j;
    r.a = a;
    r.flags = upper ? kFlagUpper : 0;
    records_.push_back(r);
    remove_row(i);
    if (upper) hi_[j] = bound; else lo_[j] = bound;
    ++stats_.singleton_rows;
    return settle_bounds(j);
  }

  /// [AA95] (ix)-(x): infeasible, forcing and redundant rows.
  Status activity_row(std::size_t i) {
    const Activity act = activity(i);
    const Real b = b_[i];
    const Real vt = kVerdictTol * (1.0 + std::max({std::fabs(b), std::fabs(act.min),
                                                   std::fabs(act.max)}));
    const Real at = kActTol * row_scale(i);

    if (act.min_inf == 0 && act.min > b + vt) return infeasible("row cannot reach its bound", i);
    if (is_eq(i) && act.max_inf == 0 && act.max < b - vt) {
      return infeasible("equality row cannot reach its right-hand side", i);
    }

    if (!is_eq(i) && act.max_inf == 0 && act.max <= b + at) {
      drop_row(i);
      ++stats_.redundant_rows;
      return Status::Ok();
    }

    const bool force_min = act.min_inf == 0 && act.min >= b - at;
    const bool force_max = is_eq(i) && act.max_inf == 0 && act.max <= b + at;
    if (!force_min && !force_max) return Status::Ok();

    // Every column in the row is pinned to the bound that attains the forcing
    // activity. Record the row first: postsolve restores the columns, then
    // picks this row's dual.
    Rec r;
    r.kind = Rec::Kind::Forcing;
    r.row = i;
    r.flags = force_min ? 0 : kFlagAtMax;
    r.entries_begin = entries_.size();
    std::vector<std::pair<std::size_t, Real>> cols;
    for_row(i, [&](std::size_t j, Real v) { cols.emplace_back(j, v); });
    for (const auto& [j, v] : cols) {
      if (col_protect_[j]) return Status::Ok();
    }
    entries_.insert(entries_.end(), cols.begin(), cols.end());
    r.entries_end = entries_.size();
    records_.push_back(r);
    remove_row(i);
    for (const auto& [j, v] : cols) {
      const bool at_lower = force_min ? v > 0.0 : v < 0.0;
      fix_column(j, at_lower ? lo_[j] : hi_[j]);
    }
    ++stats_.forcing_rows;
    return Status::Ok();
  }

  // -- columns ---------------------------------------------------------------

  Status column_pass() {
    for (std::size_t j = 0; j < n_; ++j) {
      if (!col_on_[j] || col_protect_[j]) continue;
      if (Status st = settle_bounds(j); !st.ok()) return st;
      if (!col_on_[j]) continue;
      if (col_cnt_[j] == 0) {
        empty_column(j);
      } else if (col_cnt_[j] == 1) {
        if (Status st = singleton_column(j); !st.ok()) return st;
      }
    }
    return Status::Ok();
  }

  /// [AA95] (ii). A cost pointing at an infinite bound proves only dual
  /// infeasibility -- unbounded OR infeasible -- so it is left to the engine.
  void empty_column(std::size_t j) {
    Real value = 0.0;
    if (c_[j] > 0.0) {
      if (!is_finite_bound(lo_[j])) return;
      value = lo_[j];
    } else if (c_[j] < 0.0) {
      if (!is_finite_bound(hi_[j])) return;
      value = hi_[j];
    } else {
      value = is_finite_bound(lo_[j]) ? lo_[j] : is_finite_bound(hi_[j]) ? hi_[j] : 0.0;
      if (is_finite_bound(lo_[j]) && is_finite_bound(hi_[j]) && lo_[j] <= 0.0 && 0.0 <= hi_[j]) {
        value = 0.0;
      }
    }
    fix_column(j, value);
    ++stats_.empty_columns;
  }

  Status singleton_column(std::size_t j) {
    std::size_t i = 0;
    Real a = 0.0;
    for_col(j, [&](std::size_t row, Real v) { i = row; a = v; });
    if (row_protect_[i]) return Status::Ok();

    const bool free = !is_finite_bound(lo_[j]) && !is_finite_bound(hi_[j]);

    if (!is_eq(i)) {
      // [AA95] (vi) on an inequality. x_j free absorbs the row entirely; the
      // substitution sets the row tight, which is optimal exactly when the
      // row dual c_j / a is <= 0, the sign an inequality's dual must have.
      if (free && c_[j] / a <= 0.0) {
        substitute(i, j, a, false);
        ++stats_.free_singletons;
      }
      return Status::Ok();
    }

    Real row_max = 0.0;
    for_row(i, [&](std::size_t, Real v) { row_max = std::max(row_max, std::fabs(v)); });
    const bool stable_pivot = std::fabs(a) >= kPivotRatio * row_max;

    if (free) {
      substitute(i, j, a, true);
      ++stats_.free_singletons;
      return Status::Ok();
    }

    if (stable_pivot && implied_free(i, j, a)) {
      substitute(i, j, a, true);
      ++stats_.implied_free_singletons;
      return Status::Ok();
    }

    if (row_cnt_[i] == 2) return doubleton(i, j, a);
    return Status::Ok();
  }

  /// [AA95] (13)-(15): the bounds row `i` alone implies for x_j contain x_j's
  /// own bounds, so those can never bind.
  bool implied_free(std::size_t i, std::size_t j, Real a) const {
    const Activity rest = activity(i, j);
    // x_j = (b - rest) / a
    Real imp_lo = -INF, imp_hi = INF;
    if (a > 0.0) {
      if (rest.max_inf == 0) imp_lo = (b_[i] - rest.max) / a;
      if (rest.min_inf == 0) imp_hi = (b_[i] - rest.min) / a;
    } else {
      if (rest.min_inf == 0) imp_lo = (b_[i] - rest.min) / a;
      if (rest.max_inf == 0) imp_hi = (b_[i] - rest.max) / a;
    }
    const bool lo_ok = !is_finite_bound(lo_[j]) ||
                       (is_finite_bound(imp_lo) &&
                        imp_lo >= lo_[j] - kActTol * (1.0 + std::fabs(lo_[j])));
    const bool hi_ok = !is_finite_bound(hi_[j]) ||
                       (is_finite_bound(imp_hi) &&
                        imp_hi <= hi_[j] + kActTol * (1.0 + std::fabs(hi_[j])));
    return lo_ok && hi_ok;
  }

  /// [AA95] (vi)/(viii): eliminate x_j = (b_i - sum_{k != j} a_ik x_k) / a with
  /// its row. The cost c_j x_j becomes t (b_i - sum a_ik x_k), t = c_j / a.
  void substitute(std::size_t i, std::size_t j, Real a, bool equality) {
    const Real t = c_[j] / a;
    Rec r;
    r.kind = Rec::Kind::Substitute;
    r.row = i;
    r.col = j;
    r.a = a;
    r.value = b_[i];
    r.value2 = t;
    r.flags = equality ? kFlagEquality : 0;
    r.entries_begin = entries_.size();
    for_row(i, [&](std::size_t k, Real v) {
      if (k == j) return;
      entries_.emplace_back(k, v);
      c_[k] -= t * v;
    });
    r.entries_end = entries_.size();
    records_.push_back(r);
    obj_delta_ += t * b_[i];
    remove_row(i);
    col_on_[j] = 0;
    changed_ = true;
  }

  /// [AA95] (vii): a_j x_j + a_k x_k = b with x_j a column singleton. x_j's
  /// bounds move onto x_k, then x_j is substituted out.
  Status doubleton(std::size_t i, std::size_t j, Real a_j) {
    std::size_t k = 0;
    Real a_k = 0.0;
    for_row(i, [&](std::size_t col, Real v) {
      if (col != j) { k = col; a_k = v; }
    });
    if (col_protect_[k]) return Status::Ok();
    if (std::fabs(a_j) < kPivotRatio * std::fabs(a_k)) return Status::Ok();

    // x_k = b/a_k + r x_j, r = -a_j/a_k, over x_j in [lo_j, hi_j].
    const Real base = b_[i] / a_k;
    const Real r = -a_j / a_k;
    const Real from_lo = is_finite_bound(lo_[j]) ? base + r * lo_[j] : (r > 0.0 ? -INF : INF);
    const Real from_hi = is_finite_bound(hi_[j]) ? base + r * hi_[j] : (r > 0.0 ? INF : -INF);
    const Real k_lo = std::min(from_lo, from_hi);
    const Real k_hi = std::max(from_lo, from_hi);

    std::uint8_t flags = 0;
    if (is_finite_bound(k_lo) &&
        (!is_finite_bound(lo_[k]) || k_lo > lo_[k] + kActTol * (1.0 + std::fabs(k_lo)))) {
      lo_[k] = k_lo;
      flags |= kFlagLowerFrom;
    }
    if (is_finite_bound(k_hi) &&
        (!is_finite_bound(hi_[k]) || k_hi < hi_[k] - kActTol * (1.0 + std::fabs(k_hi)))) {
      hi_[k] = k_hi;
      flags |= kFlagUpperFrom;
    }

    const Real t = c_[j] / a_j;
    Rec rec;
    rec.kind = Rec::Kind::Doubleton;
    rec.row = i;
    rec.col = j;
    rec.col2 = k;
    rec.a = a_j;
    rec.a2 = a_k;
    rec.value = b_[i];
    rec.value2 = t;
    rec.flags = flags;
    records_.push_back(rec);
    c_[k] -= t * a_k;
    obj_delta_ += t * b_[i];
    remove_row(i);
    col_on_[j] = 0;
    changed_ = true;
    ++stats_.doubletons;
    return settle_bounds(k);
  }

  // -- dual reductions -------------------------------------------------------

  /// [AA95] section 3.4 (xi). Bounds on the row duals -- an inequality's dual
  /// is <= 0 (d = c - A'y, rows a'x <= b), and each column singleton bounds its
  /// row's dual through its own reduced-cost sign (Table 2) -- bound
  /// sum_i a_ij y_i, and a column whose reduced cost is then provably of one
  /// sign sits at the matching bound in every optimal solution.
  void dominated_column_pass() {
    std::vector<Real> ylo(m_, -INF), yhi(m_, INF);
    for (std::size_t i = 0; i < m_; ++i) {
      if (row_on_[i] && !is_eq(i)) yhi[i] = 0.0;
    }
    for (std::size_t j = 0; j < n_; ++j) {
      if (!col_on_[j] || col_cnt_[j] != 1) continue;
      std::size_t i = 0;
      Real a = 0.0;
      for_col(j, [&](std::size_t row, Real v) { i = row; a = v; });
      const Real ratio = c_[j] / a;
      const bool lo_fin = is_finite_bound(lo_[j]);
      const bool hi_fin = is_finite_bound(hi_[j]);
      // d_j = c_j - a y_i >= 0 when only the lower bound is finite, <= 0 when
      // only the upper one is, = 0 when neither is.
      const bool d_nonneg = !hi_fin;
      const bool d_nonpos = !lo_fin;
      if (d_nonneg) {  // a y_i <= c_j
        if (a > 0.0) yhi[i] = std::min(yhi[i], ratio); else ylo[i] = std::max(ylo[i], ratio);
      }
      if (d_nonpos) {  // a y_i >= c_j
        if (a > 0.0) ylo[i] = std::max(ylo[i], ratio); else yhi[i] = std::min(yhi[i], ratio);
      }
    }
    for (std::size_t i = 0; i < m_; ++i) {
      // Crossed dual bounds mean the dual is infeasible: the model is
      // unbounded or infeasible, which presolve does not decide.
      if (row_on_[i] && ylo[i] > yhi[i]) return;
    }

    for (std::size_t j = 0; j < n_; ++j) {
      if (!col_on_[j] || col_protect_[j] || col_cnt_[j] == 0) continue;
      // Range of sum_i a_ij y_i.
      Real emin = 0.0, emax = 0.0;
      bool min_fin = true, max_fin = true;
      for_col(j, [&](std::size_t i, Real a) {
        const Real at_min = a > 0.0 ? ylo[i] : yhi[i];
        const Real at_max = a > 0.0 ? yhi[i] : ylo[i];
        if (is_finite_bound(at_min)) emin += a * at_min; else min_fin = false;
        if (is_finite_bound(at_max)) emax += a * at_max; else max_fin = false;
      });
      const Real margin = kDominanceTol * (1.0 + std::fabs(c_[j]));
      // d_j = c_j - sum a y lies in [c_j - emax, c_j - emin].
      if (max_fin && c_[j] - emax > margin && is_finite_bound(lo_[j])) {
        fix_column(j, lo_[j]);
        ++stats_.dominated_columns;
      } else if (min_fin && c_[j] - emin < -margin && is_finite_bound(hi_[j])) {
        fix_column(j, hi_[j]);
        ++stats_.dominated_columns;
      }
    }
  }

  // -- parallel columns --------------------------------------------------------

  /// [AA95] section 3.6 (xvii): A_k = s A_j and c_k = s c_j, so the pair only
  /// ever appears as x_j + s x_k. Also catches split free variables.
  void parallel_column_pass() {
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> buckets;
    std::vector<std::pair<std::size_t, Real>> col_a, col_b;
    for (std::size_t j = 0; j < n_; ++j) {
      if (!col_on_[j] || col_protect_[j] || col_cnt_[j] == 0) continue;
      std::uint64_t h = 1469598103934665603ULL;
      Real first = 0.0;
      for_col(j, [&](std::size_t i, Real a) {
        if (first == 0.0) first = a;
        const Real q = a / first;
        // Quantize the normalized value so equal ratios hash alike.
        const auto qv = static_cast<std::int64_t>(std::llround(q * 1e9));
        h = (h ^ static_cast<std::uint64_t>(i)) * 1099511628211ULL;
        h = (h ^ static_cast<std::uint64_t>(qv)) * 1099511628211ULL;
      });
      buckets[h].push_back(j);
    }

    for (auto& [hash, cols] : buckets) {
      if (cols.size() < 2) continue;
      for (std::size_t p = 0; p < cols.size(); ++p) {
        const std::size_t j = cols[p];
        if (!col_on_[j]) continue;
        for (std::size_t q = p + 1; q < cols.size(); ++q) {
          const std::size_t k = cols[q];
          if (!col_on_[k] || col_cnt_[k] != col_cnt_[j]) continue;
          Real s = 0.0;
          if (!parallel(j, k, col_a, col_b, s)) continue;
          if (std::fabs(c_[k] - s * c_[j]) > kParallelTol * (1.0 + std::fabs(c_[k]))) continue;
          merge(j, k, s);
        }
      }
    }
  }

  bool parallel(std::size_t j, std::size_t k, std::vector<std::pair<std::size_t, Real>>& a,
                std::vector<std::pair<std::size_t, Real>>& b, Real& s) const {
    a.clear();
    b.clear();
    for_col(j, [&](std::size_t i, Real v) { a.emplace_back(i, v); });
    for_col(k, [&](std::size_t i, Real v) { b.emplace_back(i, v); });
    if (a.size() != b.size() || a.empty()) return false;
    s = b[0].second / a[0].second;
    for (std::size_t t = 0; t < a.size(); ++t) {
      if (a[t].first != b[t].first) return false;
      if (std::fabs(b[t].second - s * a[t].second) > kParallelTol * std::fabs(b[t].second)) {
        return false;
      }
    }
    return true;
  }

  /// x_j' = x_j + s x_k over the Minkowski-sum box ([AA95] Table 4).
  void merge(std::size_t j, std::size_t k, Real s) {
    Rec r;
    r.kind = Rec::Kind::Parallel;
    r.col = k;
    r.col2 = j;
    r.a = s;
    r.lo = lo_[j];
    r.hi = hi_[j];
    r.lo2 = lo_[k];
    r.hi2 = hi_[k];
    records_.push_back(r);

    const Real k_min = s > 0.0 ? lo_[k] : hi_[k];
    const Real k_max = s > 0.0 ? hi_[k] : lo_[k];
    lo_[j] = is_finite_bound(lo_[j]) && is_finite_bound(k_min) ? lo_[j] + s * k_min : -INF;
    hi_[j] = is_finite_bound(hi_[j]) && is_finite_bound(k_max) ? hi_[j] + s * k_max : INF;

    for_col(k, [&](std::size_t i, Real) { --row_cnt_[i]; });
    col_on_[k] = 0;
    changed_ = true;
    ++stats_.parallel_columns;
  }

  // -- the fallback when rounds run out ------------------------------------------

  Status cleanup_pass() {
    for (std::size_t j = 0; j < n_; ++j) {
      if (!col_on_[j] || col_protect_[j]) continue;
      if (Status st = settle_bounds(j); !st.ok()) return st;
    }
    for (std::size_t i = 0; i < m_; ++i) {
      if (row_on_[i] && !row_protect_[i] && row_cnt_[i] == 0) {
        if (Status st = empty_row(i); !st.ok()) return st;
      }
    }
    return Status::Ok();
  }

  const model::CanonicalProblem& p_;
  const core::CsrMatrix<>& csr_;
  const core::CscMatrix<>& csc_;
  std::size_t m_, n_;
  LpPresolveStats& stats_;
  std::vector<Rec>& records_;
  std::vector<std::pair<std::size_t, Real>>& entries_;

  std::vector<Real> b_, c_, lo_, hi_;
  std::vector<char> row_on_, col_on_, row_protect_, col_protect_;
  std::vector<std::size_t> row_cnt_, col_cnt_;
  Real obj_delta_ = 0.0;
  bool changed_ = false;
};

model::CanonicalProblem clone_problem(const model::CanonicalProblem& p) {
  model::CanonicalProblem out;
  out.c = p.c.clone();
  out.A = p.A.clone();
  out.Q = p.Q.clone();
  out.b = p.b.clone();
  out.col_lower = p.col_lower.clone();
  out.col_upper = p.col_upper.clone();
  out.num_range = p.num_range;
  out.num_equality = p.num_equality;
  out.obj_offset = p.obj_offset;
  out.objective_negated = p.objective_negated;
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------

Status lp_presolve(model::CanonicalProblem& problem, const model::Options& options,
                   LpPostsolve& post, LpPresolveStats* stats_out) {
  post = LpPostsolve{};
  LpPresolveStats stats;
  stats.rows_before = problem.num_rows();
  stats.cols_before = problem.num_cols();
  stats.nnz_before = problem.A.nnz();
  const auto finish_stats = [&]() {
    stats.rows_after = problem.num_rows();
    stats.cols_after = problem.num_cols();
    stats.nnz_after = problem.A.nnz();
    if (stats_out != nullptr) *stats_out = stats;
  };
  if (!options.presolve.enabled || !problem.Q.empty()) {
    finish_stats();
    return Status::Ok();
  }

  post.original_ = clone_problem(problem);
  const model::CanonicalProblem& orig = post.original_;
  Presolve ps(orig, stats, post.records_, post.entries_);
  if (Status st = ps.run(); !st.ok()) return st;

  if (post.records_.empty()) {
    post = LpPostsolve{};
    finish_stats();
    return Status::Ok();
  }

  // Rebuild the reduced problem: surviving equalities first (the original
  // order already has them first), then surviving inequalities; columns in
  // order, which keeps the protected range columns trailing.
  const std::size_t m0 = orig.num_rows();
  const std::size_t n0 = orig.num_cols();
  std::vector<Index> new_row(m0, -1), new_col(n0, -1);
  std::size_t m1 = 0, n1 = 0, eq1 = 0;
  for (std::size_t i = 0; i < m0; ++i) {
    if (!ps.row_on(i)) continue;
    new_row[i] = static_cast<Index>(m1++);
    post.kept_row_.push_back(i);
    if (i < orig.num_equality) ++eq1;
  }
  for (std::size_t j = 0; j < n0; ++j) {
    if (!ps.col_on(j)) continue;
    new_col[j] = static_cast<Index>(n1++);
    post.kept_col_.push_back(j);
  }

  const auto& csr = orig.A.csr;
  core::SparseBuilder builder(m1, n1);
  const auto enumerate = [&](auto&& emit) {
    for (std::size_t i = 0; i < m0; ++i) {
      if (new_row[i] < 0) continue;
      for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
        const auto j = static_cast<std::size_t>(csr.indices()[k]);
        if (new_col[j] < 0) continue;
        emit(new_row[i], new_col[j], csr.values()[k]);
      }
    }
  };
  enumerate([&](Index r, Index c, Real) { builder.count(r, c); });
  if (Status st = builder.allocate(); !st.ok()) return st;
  enumerate([&](Index r, Index c, Real v) { builder.insert(r, c, v); });

  model::CanonicalProblem reduced;
  reduced.A = builder.finish();
  reduced.b = core::RealVector(m1, 0.0);
  reduced.c = core::RealVector(n1, 0.0);
  reduced.col_lower = core::RealVector(n1, 0.0);
  reduced.col_upper = core::RealVector(n1, 0.0);
  for (std::size_t r = 0; r < m1; ++r) reduced.b[r] = ps.rhs()[post.kept_row_[r]];
  for (std::size_t r = 0; r < n1; ++r) {
    const std::size_t j = post.kept_col_[r];
    reduced.c[r] = ps.cost()[j];
    reduced.col_lower[r] = ps.lower()[j];
    reduced.col_upper[r] = ps.upper()[j];
  }
  reduced.num_equality = eq1;
  reduced.num_range = orig.num_range;
  reduced.obj_offset = orig.obj_offset + ps.obj_delta();
  reduced.objective_negated = orig.objective_negated;

  post.final_cost_ = ps.cost();
  post.active_ = true;
  problem = std::move(reduced);
  finish_stats();
  return Status::Ok();
}

// ---------------------------------------------------------------------------

model::Solution LpPostsolve::expand(const model::Solution& reduced) const {
  if (!active_) {
    model::Solution copy;
    copy.status = reduced.status;
    copy.objective = reduced.objective;
    copy.x = reduced.x.clone();
    copy.s = reduced.s.clone();
    copy.y = reduced.y.clone();
    copy.z = reduced.z.clone();
    copy.v = reduced.v.clone();
    copy.quality = reduced.quality;
    copy.iterations = reduced.iterations;
    copy.matrix_products = reduced.matrix_products;
    copy.from_best_iterate = reduced.from_best_iterate;
    return copy;
  }

  const model::CanonicalProblem& p = original_;
  const std::size_t m0 = p.num_rows();
  const std::size_t n0 = p.num_cols();
  const auto& csr = p.A.csr;
  const auto& csc = p.A.csc;

  std::vector<Real> x(n0, 0.0), y(m0, 0.0), d(n0, 0.0);
  std::vector<Real> cc = final_cost_;
  const auto get = [](const core::RealVector& v, std::size_t i) {
    return i < v.size() ? v[i] : 0.0;
  };
  for (std::size_t r = 0; r < kept_col_.size(); ++r) x[kept_col_[r]] = get(reduced.x, r);
  for (std::size_t r = 0; r < kept_row_.size(); ++r) y[kept_row_[r]] = get(reduced.y, r);

  // d = cc - A'y. A kept column takes the engine's own z - v instead, so an
  // interior-point answer keeps the dual residual it was reported with rather
  // than having it silently redistributed.
  std::vector<char> kept(n0, 0);
  for (std::size_t r = 0; r < kept_col_.size(); ++r) {
    kept[kept_col_[r]] = 1;
    d[kept_col_[r]] = get(reduced.z, r) - get(reduced.v, r);
  }
  for (std::size_t j = 0; j < n0; ++j) {
    if (kept[j]) continue;
    Real s = cc[j];
    for (auto k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      s -= csc.values()[k] * y[static_cast<std::size_t>(csc.indices()[k])];
    }
    d[j] = s;
  }

  const auto add_row_dual = [&](std::size_t i, Real delta) {
    if (delta == 0.0) return;
    y[i] += delta;
    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      d[static_cast<std::size_t>(csr.indices()[k])] -= csr.values()[k] * delta;
    }
  };
  const auto restore_cost = [&](std::size_t k, Real delta) {
    cc[k] += delta;
    d[k] += delta;
  };

  for (auto it = records_.rbegin(); it != records_.rend(); ++it) {
    const Rec& r = *it;
    switch (r.kind) {
      case Rec::Kind::FixColumn:
        x[r.col] = r.value;
        break;
      case Rec::Kind::DropRow:
        break;
      case Rec::Kind::SingletonEq:
        // (50): the row takes the column's whole reduced cost.
        add_row_dual(r.row, d[r.col] / r.a);
        break;
      case Rec::Kind::SingletonIneq: {
        // The row became the column's bound. If the column's reduced cost
        // presses on that bound, the ROW is what is active: move it there.
        // The sign comes out right by construction (y <= 0 for a'x <= b).
        const bool upper = (r.flags & kFlagUpper) != 0;
        if ((upper && d[r.col] < 0.0) || (!upper && d[r.col] > 0.0)) {
          add_row_dual(r.row, d[r.col] / r.a);
        }
        break;
      }
      case Rec::Kind::Forcing: {
        // (56)-(58), generalized: every column sits at the bound the forcing
        // activity needs, so each bounds y_i from one side; the dual closest
        // to zero that satisfies all of them is taken.
        const bool at_max = (r.flags & kFlagAtMax) != 0;
        Real yi = 0.0;
        for (std::size_t e = r.entries_begin; e < r.entries_end; ++e) {
          const auto [j, a] = entries_[e];
          const Real ratio = d[j] / a;
          yi = at_max ? std::max(yi, ratio) : std::min(yi, ratio);
        }
        add_row_dual(r.row, yi);
        break;
      }
      case Rec::Kind::Substitute: {
        Real rest = 0.0;
        for (std::size_t e = r.entries_begin; e < r.entries_end; ++e) {
          rest += entries_[e].second * x[entries_[e].first];
        }
        x[r.col] = (r.value - rest) / r.a;
        add_row_dual(r.row, d[r.col] / r.a);
        for (std::size_t e = r.entries_begin; e < r.entries_end; ++e) {
          restore_cost(entries_[e].first, r.value2 * entries_[e].second);
        }
        break;
      }
      case Rec::Kind::Doubleton: {
        x[r.col] = (r.value - r.a2 * x[r.col2]) / r.a;
        add_row_dual(r.row, d[r.col] / r.a);
        restore_cost(r.col2, r.value2 * r.a2);
        // (55): a bound of col2 that came from col is really col's bound, so
        // a reduced cost pressing on it belongs to the row.
        const bool lower_from = (r.flags & kFlagLowerFrom) != 0;
        const bool upper_from = (r.flags & kFlagUpperFrom) != 0;
        const Real dk = d[r.col2];
        if ((dk > 0.0 && lower_from) || (dk < 0.0 && upper_from)) {
          add_row_dual(r.row, dk / r.a2);
        }
        break;
      }
      case Rec::Kind::Parallel: {
        // Split merged x' = x_j + s x_k back inside both boxes, putting each
        // part on the bound its reduced cost (d_k = s d_j) asks for.
        const std::size_t k = r.col, j = r.col2;
        const Real s = r.a;
        const Real merged = x[j];
        const Real dj = d[j];
        const auto clamp = [](Real v, Real lo, Real hi) {
          if (is_finite_bound(lo) && v < lo) v = lo;
          if (is_finite_bound(hi) && v > hi) v = hi;
          return v;
        };
        const Real xj_pref = dj > 0.0 ? r.lo : dj < 0.0 ? r.hi : clamp(merged, r.lo, r.hi);
        Real xk = 0.0;
        if (is_finite_bound(xj_pref)) {
          xk = clamp((merged - xj_pref) / s, r.lo2, r.hi2);
        } else {
          const Real dk = s * dj;
          const Real xk_pref = dk > 0.0 ? r.lo2 : r.hi2;
          xk = is_finite_bound(xk_pref) ? xk_pref : clamp(0.0, r.lo2, r.hi2);
        }
        x[k] = xk;
        x[j] = merged - s * xk;
        break;
      }
    }
  }

  model::Solution out;
  out.status = reduced.status;
  out.objective = reduced.objective;
  out.quality = reduced.quality;
  out.iterations = reduced.iterations;
  out.matrix_products = reduced.matrix_products;
  out.from_best_iterate = reduced.from_best_iterate;
  out.solve_time_seconds = reduced.solve_time_seconds;

  out.x = core::RealVector(n0, 0.0);
  out.z = core::RealVector(n0, 0.0);
  out.v = core::RealVector(n0, 0.0);
  for (std::size_t j = 0; j < n0; ++j) {
    out.x[j] = x[j];
    if (d[j] > 0.0) out.z[j] = d[j]; else out.v[j] = -d[j];
  }
  out.y = core::RealVector(m0, 0.0);
  for (std::size_t i = 0; i < m0; ++i) out.y[i] = y[i];
  const std::size_t ineq = p.num_inequality_rows();
  out.s = core::RealVector(ineq, 0.0);
  for (std::size_t t = 0; t < ineq; ++t) {
    const std::size_t i = p.num_equality + t;
    Real act = 0.0;
    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      act += csr.values()[k] * x[static_cast<std::size_t>(csr.indices()[k])];
    }
    out.s[t] = p.b[i] - act;
  }
  return out;
}

// ---------------------------------------------------------------------------

void unscale_canonical_solution(const model::TransformStack& scaling, model::Solution& s) {
  std::vector<Real> row_scale(s.y.size(), 1.0);
  std::vector<Real> col_scale(s.x.size(), 1.0);
  for (const auto& rec : scaling.records()) {
    if (rec.primary < 0) continue;
    const auto k = static_cast<std::size_t>(rec.primary);
    if (rec.kind == model::TransformKind::RowScaling && k < row_scale.size()) {
      row_scale[k] *= rec.value;
    } else if (rec.kind == model::TransformKind::ColumnScaling && k < col_scale.size()) {
      col_scale[k] *= rec.value;
    }
  }
  for (std::size_t j = 0; j < s.x.size(); ++j) s.x[j] *= col_scale[j];
  for (std::size_t j = 0; j < s.z.size() && j < col_scale.size(); ++j) {
    if (col_scale[j] != 0.0) s.z[j] /= col_scale[j];
  }
  for (std::size_t j = 0; j < s.v.size() && j < col_scale.size(); ++j) {
    if (col_scale[j] != 0.0) s.v[j] /= col_scale[j];
  }
  for (std::size_t i = 0; i < s.y.size(); ++i) s.y[i] *= row_scale[i];
}

}  // namespace sovsolve::solver
