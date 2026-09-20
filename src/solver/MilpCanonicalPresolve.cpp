#include "sovsolve/solver/MilpCanonicalPresolve.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>

#include "sovsolve/core/SparseBuilder.hpp"

namespace sovsolve::solver {

namespace {

using core::is_finite_bound;
using core::Real;

constexpr Real kTol = 1e-6;  // [CIP] (7.2)'s feasibility tolerance epsilon-hat

/// The bound a substitution or a merge may silently drop. Tighter than the
/// feasibility tolerance on purpose: dropping a bound that really binds ADDS
/// points to the model, and a wrong answer that is feasible-looking is worse
/// than a missed reduction. OURS -- [AGH] 4.5 states the condition exactly.
constexpr Real kImpliedFreeTol = 1e-9;

/// [AGH] 4.5's Markowitz threshold on the pivot: |a_ij| >= 0.01 max|a_.j| or
/// |a_ij| >= 0.01 max|a_i.|. The paper's own value, and its own warning that
/// this reduction is "one of the main sources for numerical issues".
constexpr Real kMarkowitz = 0.01;
/// An absolute floor under the pivot as well, since the Markowitz test passes
/// trivially for a singleton column however small its one entry is. OURS.
constexpr Real kMinPivot = 1e-7;

/// [AGH] 4.5 caps the "estimated fill-in" without saying where. A substitution
/// writes row i into every OTHER row holding column j, so it creates at most
/// (|A_.j| - 1) (|A_i.| - 1) new non-zeros. A free column singleton scores 0.
/// The value is OURS.
constexpr std::size_t kMaxSubstitutionFill = 32;

/// Each substitution rescans the matrix, so the pass is O(nnz) per reduction.
/// Root presolve runs once, but a cap keeps a pathological model bounded. OURS.
constexpr std::size_t kMaxSubstitutions = 4096;

/// Coefficients this small, RELATIVE to the largest in the row they land in,
/// are cancellation residue and are dropped. Keeping them would densify A with
/// noise; [AGH] 5.3 treats exactly this cancellation as a reduction in its own
/// right. OURS.
constexpr Real kCancelTol = 1e-12;

/// Parallel-column detection buckets columns by support and then compares
/// within a bucket pairwise. A degenerate bucket would make that quadratic.
/// [AGH] 6.3 reports the same hashing needs no work limit in practice; this
/// one is cheap insurance. OURS.
constexpr std::size_t kMaxParallelComparisons = 1u << 20;

/// [CIP] (7.3): accept a bound change only if it removes at least 5% of the
/// domain (or of the bound's magnitude, at least 1), or makes it finite.
Real min_change(Real lo, Real hi, Real bound) {
  return 0.05 * std::max(std::min(hi - lo, std::fabs(bound)), 1.0);
}

struct Row {
  std::vector<std::pair<std::size_t, Real>> entries;
  Real b = 0.0;
  bool equality = false;
  bool alive = true;
};

struct Activity {
  Real min = 0.0, max = 0.0;
  std::size_t min_inf = 0, max_inf = 0;
};

/// Working state for one presolve run. A class rather than one long function
/// because stage B needs the column view of the matrix as well as the row
/// view, and both stages share the bounds, the costs and the undo stack.
class Presolve {
 public:
  Presolve(const model::CanonicalProblem& p, const std::vector<Real>& integer_scale,
           bool allow_column_removal, CanonicalPostsolve& post, CanonicalPresolveStats& stats)
      : n_(p.num_cols()),
        scale_(integer_scale),
        // Range columns are identified by POSITION (Canonical.hpp), so they
        // must keep theirs: stage B never removes one.
        first_range_(p.num_cols() - p.num_range),
        // A quadratic objective would need its own substitution rule; the MILP
        // path never has one, and silently mangling Q would be worse than
        // declining the reduction.
        allow_removal_(allow_column_removal && p.Q.empty()),
        post_(post),
        stats_(stats),
        obj_offset_(p.obj_offset),
        work_(p.num_cols(), 0.0),
        marked_(p.num_cols(), 0) {
    lo_.assign(p.col_lower.data(), p.col_lower.data() + n_);
    hi_.assign(p.col_upper.data(), p.col_upper.data() + n_);
    cost_.assign(p.c.data(), p.c.data() + n_);
    alive_.assign(n_, 1);

    const std::size_t m = p.num_rows();
    rows_.resize(m);
    const auto& csr = p.A.csr;
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t q = csr.slice_begin(i); q < csr.slice_end(i); ++q) {
        rows_[i].entries.emplace_back(static_cast<std::size_t>(csr.indices()[q]),
                                      csr.values()[q]);
      }
      rows_[i].b = p.b[i];
      rows_[i].equality = i < p.num_equality;
    }
  }

  [[nodiscard]] core::Status run(std::size_t max_rounds);
  [[nodiscard]] core::Status write_back(model::CanonicalProblem& p) const;

 private:
  [[nodiscard]] static core::Status infeasible(const char* why) {
    return core::make_error(core::ErrorCode::PrimalInfeasible, why);
  }

  [[nodiscard]] Activity activity(const Row& r) const {
    Activity a;
    for (const auto& [j, v] : r.entries) {
      const Real l = v > 0.0 ? lo_[j] : hi_[j];
      const Real h = v > 0.0 ? hi_[j] : lo_[j];
      if (is_finite_bound(l)) a.min += v * l; else ++a.min_inf;
      if (is_finite_bound(h)) a.max += v * h; else ++a.max_inf;
    }
    return a;
  }

  /// True when `j` may be taken out of the model altogether.
  [[nodiscard]] bool removable(std::size_t j) const {
    return alive_[j] != 0 && j < first_range_ && scale_[j] == 0.0;
  }

  [[nodiscard]] core::Status propagate(Row& row, bool& changed);
  void dual_fixing(bool& changed);
  [[nodiscard]] core::Status substitute_implied_free(bool& changed);
  void merge_parallel_columns(bool& changed);

  /// Row `dst` -= factor * row `src`, skipping column `drop` in both. The
  /// scatter/gather through `work_` keeps this linear in the two supports.
  void axpy_row(Row& dst, const Row& src, Real factor, std::size_t drop);

  // Declaration order is the initialization order; keep it matching the
  // constructor's member-initializer list.
  std::size_t n_;
  const std::vector<Real>& scale_;
  std::size_t first_range_;
  bool allow_removal_;
  CanonicalPostsolve& post_;
  CanonicalPresolveStats& stats_;
  Real obj_offset_;
  std::vector<Real> work_;
  std::vector<char> marked_;

  std::vector<Row> rows_;
  std::vector<Real> lo_, hi_, cost_;
  std::vector<char> alive_;
  std::size_t substitutions_ = 0;
};

// ---------------------------------------------------------------------------
// Stage A: one row's propagation ([CIP] Algorithm 10.1 steps 1c, 1d, 1f)
// ---------------------------------------------------------------------------

core::Status Presolve::propagate(Row& row, bool& changed) {
  // [CIP] 10.1 step 1g: at most 10 passes over one row while it changes.
  for (int pass = 0; pass < 10; ++pass) {
    bool row_changed = false;
    const Real rho = row.b;
    const Real lambda = row.equality ? row.b : -1e300;
    const bool has_lambda = row.equality;

    // A row emptied by a substitution says only `0 (=|<=) b`. The canonical
    // form forbids an all-zero row (Canonical.hpp), so it leaves here one way
    // or the other rather than reaching the solver.
    if (row.entries.empty()) {
      if (rho < -kTol * (1.0 + std::fabs(rho))) {
        return infeasible("presolve: an empty row's right-hand side is negative");
      }
      if (has_lambda && rho > kTol * (1.0 + std::fabs(rho))) {
        return infeasible("presolve: an empty equality has a non-zero right-hand side");
      }
      row.alive = false;
      ++stats_.rows_removed;
      changed = true;
      return core::Status::Ok();
    }

    Activity a = activity(row);

    // Step 1d: infeasibility, then redundancy.
    if (a.min_inf == 0 && a.min > rho + kTol * (1.0 + std::fabs(rho))) {
      return infeasible("presolve: a row's minimum activity exceeds its right-hand side");
    }
    if (has_lambda && a.max_inf == 0 && a.max < lambda - kTol * (1.0 + std::fabs(lambda))) {
      return infeasible("presolve: an equality's maximum activity is below its right-hand side");
    }
    if (!row.equality && a.max_inf == 0 && a.max <= rho + kTol * (1.0 + std::fabs(rho))) {
      row.alive = false;  // can never be violated
      ++stats_.rows_removed;
      changed = true;
      return core::Status::Ok();
    }

    // Step 1c: bound tightening, Algorithm 7.1.
    for (const auto& [j, v] : row.entries) {
      const Real lpart = v > 0.0 ? lo_[j] : hi_[j];
      const Real hpart = v > 0.0 ? hi_[j] : lo_[j];
      Real alpha = -1e300, beta = 1e300;
      bool alpha_ok = false, beta_ok = false;
      if (is_finite_bound(lpart)) {
        if (a.min_inf == 0) { alpha = a.min - v * lpart; alpha_ok = true; }
      } else if (a.min_inf == 1) {
        alpha = a.min;
        alpha_ok = true;
      }
      if (is_finite_bound(hpart)) {
        if (a.max_inf == 0) { beta = a.max - v * hpart; beta_ok = true; }
      } else if (a.max_inf == 1) {
        beta = a.max;
        beta_ok = true;
      }
      Real new_lo = -1e300, new_hi = 1e300;
      bool lo_set = false, hi_set = false;
      if (alpha_ok) {
        const Real bnd = (rho - alpha) / v;
        if (v > 0.0) { new_hi = bnd; hi_set = true; } else { new_lo = bnd; lo_set = true; }
      }
      if (has_lambda && beta_ok) {
        const Real bnd = (lambda - beta) / v;
        if (v > 0.0) { new_lo = bnd; lo_set = true; } else { new_hi = bnd; hi_set = true; }
      }
      const Real s = scale_[j];
      if (hi_set && is_finite_bound(new_hi)) {
        new_hi = 1e-5 * std::ceil(1e5 * new_hi - kTol);
        if (s != 0.0) new_hi = std::floor(s * new_hi + kTol) / s;
        if (new_hi < hi_[j] &&
            (!is_finite_bound(hi_[j]) || new_hi < hi_[j] - min_change(lo_[j], hi_[j], hi_[j]))) {
          hi_[j] = new_hi;
          ++stats_.bounds_tightened;
          row_changed = true;
        }
      }
      if (lo_set && is_finite_bound(new_lo)) {
        new_lo = 1e-5 * std::floor(1e5 * new_lo + kTol);
        if (s != 0.0) new_lo = std::ceil(s * new_lo - kTol) / s;
        if (new_lo > lo_[j] &&
            (!is_finite_bound(lo_[j]) || new_lo > lo_[j] + min_change(lo_[j], hi_[j], lo_[j]))) {
          lo_[j] = new_lo;
          ++stats_.bounds_tightened;
          row_changed = true;
        }
      }
      if (lo_[j] > hi_[j] + kTol * (1.0 + std::fabs(hi_[j]))) {
        return infeasible("presolve: a column's bounds crossed");
      }
      if (lo_[j] > hi_[j]) lo_[j] = hi_[j];
      // No early exit: activities made stale by a tightening earlier in this
      // scan are looser, so later deductions from them stay valid.
    }
    if (row_changed) {
      changed = true;
      continue;
    }

    // Step 1f: coefficient tightening, inequality rows (lambda = -inf) and
    // integer columns with finite integral bounds. In original units v = s x,
    // a column steps by 1 and its coefficient is a/s; beta is the maximum
    // activity. Condition "the row is redundant whenever x_j is off its
    // bound": beta - a_v <= rho for a_v > 0 (off the upper bound),
    // beta + a_v <= rho for a_v < 0 (off the lower bound). Then
    //   a_v > 0:  a'_v = beta - rho,  rho -= (a_v - a'_v) U
    //   a_v < 0:  a'_v = rho - beta,  rho -= (a_v - a'_v) L
    // -- [CIP] step 1f with lambda = -inf, whose other side then drops.
    if (!row.equality && a.max_inf == 0) {
      for (auto& [j, v] : row.entries) {
        const Real s = scale_[j];
        if (s == 0.0 || !is_finite_bound(lo_[j]) || !is_finite_bound(hi_[j])) continue;
        const Real av = v / s;
        const Real L = s * lo_[j];
        const Real U = s * hi_[j];
        const Real tol = kTol * (1.0 + std::fabs(rho));
        if (av > 0.0 && a.max - av <= rho + tol) {
          const Real anew = a.max - rho;
          if (anew > kTol && anew < av - kTol) {
            row.b -= (av - anew) * U;
            v = anew * s;
            ++stats_.coefficients_tightened;
            row_changed = true;
            break;
          }
        } else if (av < 0.0 && a.max + av <= rho + tol) {
          const Real anew = rho - a.max;
          if (anew < -kTol && anew > av + kTol) {
            row.b -= (av - anew) * L;
            v = anew * s;
            ++stats_.coefficients_tightened;
            row_changed = true;
            break;
          }
        }
      }
    }
    if (!row_changed) break;
    changed = true;
  }
  return core::Status::Ok();
}

// ---------------------------------------------------------------------------
// Stage A: dual fixing ([CIP] Algorithm 10.14)
// ---------------------------------------------------------------------------

void Presolve::dual_fixing(bool& changed) {
  // With no row resisting a move in the direction the objective favours, the
  // column goes to that bound. Locks as in [CIP] Example 3.4: an equality
  // locks both ways, `a'x <= b` locks up for a > 0 and down for a < 0.
  std::vector<std::size_t> down(n_, 0), up(n_, 0);
  for (const Row& row : rows_) {
    if (!row.alive) continue;
    for (const auto& [j, v] : row.entries) {
      if (row.equality) {
        ++down[j];
        ++up[j];
      } else if (v > 0.0) {
        ++up[j];
      } else if (v < 0.0) {
        ++down[j];
      }
    }
  }
  for (std::size_t j = 0; j < n_; ++j) {
    if (alive_[j] == 0 || lo_[j] == hi_[j]) continue;
    const Real c = cost_[j];
    if (c >= 0.0 && down[j] == 0 && is_finite_bound(lo_[j])) {
      hi_[j] = lo_[j];
    } else if (c <= 0.0 && up[j] == 0 && is_finite_bound(hi_[j])) {
      lo_[j] = hi_[j];
    } else {
      continue;  // an infinite target bound: left for the LP to report
    }
    ++stats_.columns_fixed;
    changed = true;
  }
}

// ---------------------------------------------------------------------------
// Stage B helpers
// ---------------------------------------------------------------------------

void Presolve::axpy_row(Row& dst, const Row& src, Real factor, std::size_t drop) {
  std::vector<std::size_t> touched;
  touched.reserve(dst.entries.size() + src.entries.size());
  const auto scatter = [&](std::size_t k, Real v) {
    if (marked_[k] == 0) {
      marked_[k] = 1;
      touched.push_back(k);
    }
    work_[k] += v;
  };
  for (const auto& [k, v] : dst.entries) {
    if (k != drop) scatter(k, v);
  }
  for (const auto& [k, v] : src.entries) {
    if (k != drop) scatter(k, -factor * v);
  }

  Real biggest = 0.0;
  for (const std::size_t k : touched) biggest = std::max(biggest, std::fabs(work_[k]));
  const Real drop_below = kCancelTol * std::max(1.0, biggest);

  dst.entries.clear();
  for (const std::size_t k : touched) {
    const Real v = work_[k];
    work_[k] = 0.0;
    marked_[k] = 0;
    if (std::fabs(v) > drop_below) dst.entries.emplace_back(k, v);
  }
  // Row entries stay sorted by column, as they were when read out of the CSR.
  std::sort(dst.entries.begin(), dst.entries.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  dst.b -= factor * src.b;
}

// ---------------------------------------------------------------------------
// Stage B.1: substitute implied free variables ([AGH] 4.5)
// ---------------------------------------------------------------------------

core::Status Presolve::substitute_implied_free(bool& changed) {
  if (!allow_removal_) return core::Status::Ok();

  for (;;) {
    if (substitutions_ >= kMaxSubstitutions) return core::Status::Ok();

    // The column view, rebuilt per reduction. A substitution rewrites an
    // unbounded number of rows, so an incrementally maintained view would have
    // to track every created and cancelled non-zero; at O(nnz) per reduction
    // in a routine that runs once at the root, rebuilding is the honest cost.
    std::vector<std::vector<std::size_t>> col_rows(n_);
    for (std::size_t i = 0; i < rows_.size(); ++i) {
      if (!rows_[i].alive) continue;
      for (const auto& [j, v] : rows_[i].entries) {
        (void)v;
        col_rows[j].push_back(i);
      }
    }

    std::size_t pick_row = 0, pick_col = 0;
    Real pick_pivot = 0.0;
    bool found = false;

    for (std::size_t i = 0; i < rows_.size() && !found; ++i) {
      const Row& row = rows_[i];
      if (!row.alive || !row.equality || row.entries.empty()) continue;

      Real row_max = 0.0;
      for (const auto& [k, v] : row.entries) {
        (void)k;
        row_max = std::max(row_max, std::fabs(v));
      }
      const Activity a = activity(row);

      for (const auto& [j, aij] : row.entries) {
        if (!removable(j)) continue;
        if (lo_[j] == hi_[j]) continue;  // already fixed; nothing to gain
        if (std::fabs(aij) < kMinPivot) continue;

        // [AGH] 4.5's Markowitz safeguard.
        Real col_max = 0.0;
        for (const std::size_t r : col_rows[j]) {
          for (const auto& [k, v] : rows_[r].entries) {
            if (k == j) col_max = std::max(col_max, std::fabs(v));
          }
        }
        if (std::fabs(aij) < kMarkowitz * col_max && std::fabs(aij) < kMarkowitz * row_max) {
          continue;
        }

        // Fill-in: row i is written into every OTHER row holding column j.
        const std::size_t others = col_rows[j].size() - 1;
        if (others * (row.entries.size() - 1) > kMaxSubstitutionFill) continue;

        // Implied free ([AGH] 4.5): the range row i alone forces on x_j, from
        // the OTHER columns' bounds, must sit inside x_j's own bounds.
        Real smin = a.min, smax = a.max;
        std::size_t smin_inf = a.min_inf, smax_inf = a.max_inf;
        const Real l = aij > 0.0 ? lo_[j] : hi_[j];
        const Real h = aij > 0.0 ? hi_[j] : lo_[j];
        if (is_finite_bound(l)) smin -= aij * l; else --smin_inf;
        if (is_finite_bound(h)) smax -= aij * h; else --smax_inf;

        // x_j = (b_i - A_iS x_S) / a_ij, so the largest A_iS x_S gives the
        // smallest x_j when a_ij > 0 and the largest when a_ij < 0.
        Real implied_lo = 0.0, implied_hi = 0.0;
        bool implied_lo_finite = false, implied_hi_finite = false;
        if (aij > 0.0) {
          implied_lo_finite = smax_inf == 0;
          implied_lo = (row.b - smax) / aij;
          implied_hi_finite = smin_inf == 0;
          implied_hi = (row.b - smin) / aij;
        } else {
          implied_lo_finite = smin_inf == 0;
          implied_lo = (row.b - smin) / aij;
          implied_hi_finite = smax_inf == 0;
          implied_hi = (row.b - smax) / aij;
        }
        const bool lower_free =
            !is_finite_bound(lo_[j]) ||
            (implied_lo_finite && implied_lo >= lo_[j] - kImpliedFreeTol * (1.0 + std::fabs(lo_[j])));
        const bool upper_free =
            !is_finite_bound(hi_[j]) ||
            (implied_hi_finite && implied_hi <= hi_[j] + kImpliedFreeTol * (1.0 + std::fabs(hi_[j])));
        if (!lower_free || !upper_free) continue;

        pick_row = i;
        pick_col = j;
        pick_pivot = aij;
        found = true;
        break;
      }
    }
    if (!found) return core::Status::Ok();

    Row& src = rows_[pick_row];
    const std::size_t j = pick_col;
    const Real pivot = pick_pivot;

    PostsolveColumn rec;
    rec.kind = PostsolveColumn::Kind::SubstituteEquality;
    rec.column = j;
    rec.pivot = pivot;
    rec.rhs = src.b;
    rec.removed_lower = lo_[j];
    rec.removed_upper = hi_[j];
    for (const auto& [k, v] : src.entries) {
      if (k != j) rec.row.emplace_back(k, v);
    }
    post_.undo.push_back(std::move(rec));

    // The objective: c_j x_j = c_j b_i / a_ij - sum_S (c_j a_ik / a_ij) x_k.
    if (cost_[j] != 0.0) {
      const Real f = cost_[j] / pivot;
      obj_offset_ += f * src.b;
      for (const auto& [k, v] : src.entries) {
        if (k != j) cost_[k] -= f * v;
      }
      cost_[j] = 0.0;
    }

    // Every other row holding the column takes row i in its place.
    for (std::size_t r = 0; r < rows_.size(); ++r) {
      if (r == pick_row || !rows_[r].alive) continue;
      Real arj = 0.0;
      for (const auto& [k, v] : rows_[r].entries) {
        if (k == j) { arj = v; break; }
      }
      if (arj == 0.0) continue;
      axpy_row(rows_[r], src, arj / pivot, j);
    }

    src.alive = false;  // the equality has done its work
    alive_[j] = 0;
    ++stats_.columns_substituted;
    ++substitutions_;
    changed = true;
  }
}

// ---------------------------------------------------------------------------
// Stage B.2: parallel columns ([AGH] 6.3)
// ---------------------------------------------------------------------------

void Presolve::merge_parallel_columns(bool& changed) {
  if (!allow_removal_) return;

  // Column view over the live rows. Rows are visited in increasing order, so
  // each column's list comes out sorted by row -- which is what makes the
  // pattern comparison below a single linear scan.
  std::vector<std::vector<std::pair<std::size_t, Real>>> cols(n_);
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    if (!rows_[i].alive) continue;
    for (const auto& [j, v] : rows_[i].entries) cols[j].emplace_back(i, v);
  }

  // [AGH] 6.3 hashes columns and scans each bucket linearly. The hash covers
  // the SUPPORT only: two parallel columns share their rows exactly, and
  // hashing floating-point ratios would make the bucketing depend on rounding.
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> buckets;
  for (std::size_t j = 0; j < n_; ++j) {
    if (!removable(j) || cols[j].empty()) continue;
    std::uint64_t h = 1469598103934665603ull;
    for (const auto& [i, v] : cols[j]) {
      (void)v;
      h = (h ^ static_cast<std::uint64_t>(i)) * 1099511628211ull;
    }
    buckets[h].push_back(j);
  }

  std::vector<char> touched(n_, 0);
  std::vector<std::size_t> dropped;
  std::size_t comparisons = 0;
  bool out_of_budget = false;

  for (const auto& [h, bucket] : buckets) {
    (void)h;
    if (out_of_budget) break;
    for (std::size_t x = 0; x < bucket.size() && !out_of_budget; ++x) {
      const std::size_t j = bucket[x];
      if (touched[j] != 0) continue;
      for (std::size_t y = x + 1; y < bucket.size(); ++y) {
        const std::size_t k = bucket[y];
        if (touched[k] != 0) continue;
        if (++comparisons > kMaxParallelComparisons) {
          // Stop looking, but still fall through to the sweep below: the
          // merges already recorded must be applied to the matrix.
          out_of_budget = true;
          break;
        }
        if (cols[j].size() != cols[k].size()) continue;

        // A_.k = lambda A_.j, with lambda read off the first shared row.
        bool same = true;
        for (std::size_t t = 0; t < cols[j].size(); ++t) {
          if (cols[j][t].first != cols[k][t].first) { same = false; break; }
        }
        if (!same) continue;
        const Real lambda = cols[k][0].second / cols[j][0].second;
        if (!std::isfinite(lambda) || std::fabs(lambda) < kMinPivot) continue;
        for (std::size_t t = 0; t < cols[j].size() && same; ++t) {
          const Real want = lambda * cols[j][t].second;
          if (std::fabs(cols[k][t].second - want) >
              kImpliedFreeTol * std::max(1.0, std::fabs(cols[k][t].second))) {
            same = false;
          }
        }
        if (!same) continue;
        // ... and c_k = lambda c_j, or the merged column cannot carry c_j.
        if (std::fabs(cost_[k] - lambda * cost_[j]) >
            kImpliedFreeTol * std::max(1.0, std::fabs(cost_[k]))) {
          continue;
        }

        // Merge into y = x_j + lambda x_k, keeping column j's entries and
        // cost. [AGH] (6.1)'s Minkowski-sum bounds; an infinite endpoint on
        // either side makes the corresponding bound of y infinite.
        const Real klo = lambda > 0.0 ? lo_[k] : hi_[k];
        const Real khi = lambda > 0.0 ? hi_[k] : lo_[k];
        const Real ylo = (is_finite_bound(lo_[j]) && is_finite_bound(klo))
                             ? lo_[j] + lambda * klo
                             : -core::INF;
        const Real yhi = (is_finite_bound(hi_[j]) && is_finite_bound(khi))
                             ? hi_[j] + lambda * khi
                             : core::INF;

        PostsolveColumn rec;
        rec.kind = PostsolveColumn::Kind::MergeParallel;
        rec.column = k;
        rec.partner = j;
        rec.pivot = lambda;
        rec.removed_lower = lo_[k];
        rec.removed_upper = hi_[k];
        rec.partner_lower = lo_[j];
        rec.partner_upper = hi_[j];
        post_.undo.push_back(std::move(rec));

        lo_[j] = ylo;
        hi_[j] = yhi;
        alive_[k] = 0;
        cost_[k] = 0.0;
        dropped.push_back(k);
        // Both columns' bounds and cost now mean something new, so neither
        // takes part in another merge until the view is rebuilt next round.
        touched[j] = 1;
        touched[k] = 1;
        ++stats_.columns_merged;
        changed = true;
        break;
      }
    }
  }

  if (dropped.empty()) return;
  std::vector<char> gone(n_, 0);
  for (const std::size_t k : dropped) gone[k] = 1;
  const auto merged_away = [&](const std::pair<std::size_t, Real>& e) {
    return gone[e.first] != 0;
  };
  for (Row& row : rows_) {
    if (!row.alive) continue;
    row.entries.erase(std::remove_if(row.entries.begin(), row.entries.end(), merged_away),
                      row.entries.end());
  }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

core::Status Presolve::run(std::size_t max_rounds) {
  // Integer bounds are integral in original units.
  for (std::size_t j = 0; j < n_; ++j) {
    const Real s = scale_[j];
    if (s == 0.0) continue;
    if (is_finite_bound(lo_[j])) lo_[j] = std::ceil(s * lo_[j] - kTol) / s;
    if (is_finite_bound(hi_[j])) hi_[j] = std::floor(s * hi_[j] + kTol) / s;
    if (lo_[j] > hi_[j] + kTol) {
      return infeasible("presolve: an integer column has no integer value");
    }
  }

  bool any_change = true;
  for (std::size_t round = 0; round < max_rounds && any_change; ++round) {
    any_change = false;
    ++stats_.rounds;

    for (Row& row : rows_) {
      if (!row.alive) continue;
      if (const auto st = propagate(row, any_change); !st.ok()) return st;
    }
    dual_fixing(any_change);
    if (const auto st = substitute_implied_free(any_change); !st.ok()) return st;
    merge_parallel_columns(any_change);
  }
  return core::Status::Ok();
}

core::Status Presolve::write_back(model::CanonicalProblem& p) const {
  // Surviving columns keep their relative order, so the trailing range block
  // stays trailing and `num_range` still names it.
  post_.plain_columns = n_;
  post_.new_of_old.assign(n_, CanonicalPostsolve::kRemoved);
  std::size_t kept_cols = 0;
  for (std::size_t j = 0; j < n_; ++j) {
    if (alive_[j] != 0) post_.new_of_old[j] = kept_cols++;
  }

  core::RealVector c(kept_cols), col_lower(kept_cols), col_upper(kept_cols);
  for (std::size_t j = 0; j < n_; ++j) {
    const std::size_t nj = post_.new_of_old[j];
    if (nj == CanonicalPostsolve::kRemoved) continue;
    c[nj] = cost_[j];
    col_lower[nj] = lo_[j];
    col_upper[nj] = hi_[j];
  }

  // Equalities first, as the canonical form requires (Canonical.hpp).
  std::vector<std::size_t> keep;
  std::size_t equalities = 0;
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    if (rows_[i].alive && rows_[i].equality) {
      keep.push_back(i);
      ++equalities;
    }
  }
  for (std::size_t i = 0; i < rows_.size(); ++i) {
    if (rows_[i].alive && !rows_[i].equality) keep.push_back(i);
  }

  const auto enumerate = [&](auto&& emit) {
    for (std::size_t r = 0; r < keep.size(); ++r) {
      for (const auto& [j, v] : rows_[keep[r]].entries) {
        const std::size_t nj = post_.new_of_old[j];
        if (v != 0.0 && nj != CanonicalPostsolve::kRemoved) {
          emit(static_cast<core::Index>(r), static_cast<core::Index>(nj), v);
        }
      }
    }
  };
  core::SparseBuilder builder(keep.size(), kept_cols);
  enumerate([&](core::Index r, core::Index col, Real) { builder.count(r, col); });
  if (const auto st = builder.allocate(); !st.ok()) return st;
  enumerate([&](core::Index r, core::Index col, Real v) { builder.insert(r, col, v); });

  p.A = builder.finish(true, 0.0);
  core::RealVector b(keep.size());
  for (std::size_t r = 0; r < keep.size(); ++r) b[r] = rows_[keep[r]].b;
  p.b = std::move(b);
  p.c = std::move(c);
  p.col_lower = std::move(col_lower);
  p.col_upper = std::move(col_upper);
  p.num_equality = equalities;
  p.obj_offset = obj_offset_;
  return core::Status::Ok();
}

}  // namespace

void expand_canonical_point(const CanonicalPostsolve& post,
                            core::HostSpan<const Real> presolved_x,
                            std::vector<Real>& plain_x) {
  plain_x.assign(post.plain_columns, 0.0);
  for (std::size_t j = 0; j < post.plain_columns; ++j) {
    const std::size_t nj = post.new_of_old[j];
    if (nj != CanonicalPostsolve::kRemoved && nj < presolved_x.size()) {
      plain_x[j] = presolved_x[nj];
    }
  }

  // Reverse order. A record may name a column that a LATER record removed --
  // that one has already been restored by the time this one is replayed -- and
  // it may name the survivor of a later merge, which still holds the merged
  // value `y` that the record was built against. Both need this direction.
  for (auto it = post.undo.rbegin(); it != post.undo.rend(); ++it) {
    const PostsolveColumn& rec = *it;
    const auto clamp_to = [](Real v, Real lo, Real hi) {
      if (is_finite_bound(lo) && v < lo) v = lo;
      if (is_finite_bound(hi) && v > hi) v = hi;
      return v;
    };

    if (rec.kind == PostsolveColumn::Kind::SubstituteEquality) {
      Real acc = rec.rhs;
      for (const auto& [k, v] : rec.row) {
        if (k < plain_x.size()) acc -= v * plain_x[k];
      }
      // The implied-free test guaranteed this lands inside the bounds; the
      // clamp only absorbs rounding.
      plain_x[rec.column] = clamp_to(acc / rec.pivot, rec.removed_lower, rec.removed_upper);
      continue;
    }

    // Split y = x_partner + lambda x_column back into the pair. Any x_column
    // with x_partner = y - lambda x_column inside the partner's bounds will
    // do, so take the admissible value nearest zero.
    const Real y = plain_x[rec.partner];
    const Real lambda = rec.pivot;
    Real klo = -1e300, khi = 1e300;
    if (is_finite_bound(rec.partner_upper)) {
      const Real t = (y - rec.partner_upper) / lambda;
      if (lambda > 0.0) klo = std::max(klo, t); else khi = std::min(khi, t);
    }
    if (is_finite_bound(rec.partner_lower)) {
      const Real t = (y - rec.partner_lower) / lambda;
      if (lambda > 0.0) khi = std::min(khi, t); else klo = std::max(klo, t);
    }
    if (is_finite_bound(rec.removed_lower)) klo = std::max(klo, rec.removed_lower);
    if (is_finite_bound(rec.removed_upper)) khi = std::min(khi, rec.removed_upper);

    const Real xk = klo > khi ? klo : std::min(std::max(0.0, klo), khi);
    plain_x[rec.column] = xk;
    plain_x[rec.partner] = clamp_to(y - lambda * xk, rec.partner_lower, rec.partner_upper);
  }
}

core::Status presolve_canonical(model::CanonicalProblem& p,
                                const std::vector<Real>& integer_scale, std::size_t max_rounds,
                                bool allow_column_removal, CanonicalPostsolve& postsolve,
                                CanonicalPresolveStats& stats) {
  postsolve.undo.clear();
  Presolve presolve(p, integer_scale, allow_column_removal, postsolve, stats);
  if (const auto st = presolve.run(max_rounds); !st.ok()) {
    // The identity map, so a caller that reports infeasibility can still make
    // sense of `postsolve` without a special case.
    postsolve.plain_columns = p.num_cols();
    postsolve.new_of_old.resize(p.num_cols());
    for (std::size_t j = 0; j < postsolve.new_of_old.size(); ++j) postsolve.new_of_old[j] = j;
    postsolve.undo.clear();
    return st;
  }
  return presolve.write_back(p);
}

}  // namespace sovsolve::solver
