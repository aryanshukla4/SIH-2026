#include "sovsolve/solver/simplex/LuFactor.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace sovsolve::solver::simplex {
namespace {

using core::ErrorCode;

[[nodiscard]] Status singular(std::size_t step) {
  return core::make_error(ErrorCode::NumericalError,
                          "basis is singular: no acceptable pivot at elimination step " +
                              std::to_string(step));
}

/// Elimination workspace.
///
/// The active submatrix is held column-wise with exact values, and row-wise as
/// a PATTERN ONLY. Storing values twice would mean keeping two copies in step
/// through every fill-in update, which is the classic place for a sparse LU to
/// go subtly wrong; storing the pattern once and re-reading values from the
/// columns cannot drift.
///
/// `row_pattern` is a superset: an entry is appended when fill creates it and
/// is never eagerly removed, because removal would need a position map per
/// entry. Stale entries are pruned the one time the row is scanned -- when it
/// is chosen as a pivot row -- which is also the only time the pattern is
/// read. Every entry is therefore validated against the column before use.
struct Workspace {
  std::size_t dim = 0;

  std::vector<std::vector<std::pair<Index, Real>>> col_entries;
  std::vector<std::vector<Index>> row_pattern;

  /// Counts over the ACTIVE submatrix only -- entries in pivotal rows or
  /// columns are excluded. These drive the Markowitz cost, so counting a
  /// already-eliminated entry would bias the pivot choice toward fill.
  std::vector<Index> col_count;
  std::vector<Index> row_count;

  std::vector<char> col_active;
  std::vector<char> row_active;

  /// Columns bucketed by active nonzero count, for the increasing-count pivot
  /// scan. Lazily maintained: a column is pushed whenever its count changes
  /// and stale copies are discarded when the bucket is next scanned, which is
  /// far cheaper than locating and erasing the old copy.
  std::vector<std::vector<Index>> bucket;

  // Dense scratch for the rank-1 column updates.
  std::vector<Real> dense;
  std::vector<char> in_dense;
  std::vector<Index> touched;

  // Dense scratch for the gathered pivot row.
  std::vector<Real> pivot_row_value;
  std::vector<char> pivot_row_mark;
  std::vector<Index> pivot_row_cols;

  std::vector<std::pair<Index, Real>> multipliers;

  void resize(std::size_t n) {
    dim = n;
    col_entries.assign(n, {});
    row_pattern.assign(n, {});
    col_count.assign(n, 0);
    row_count.assign(n, 0);
    col_active.assign(n, 1);
    row_active.assign(n, 1);
    bucket.assign(n + 1, {});
    dense.assign(n, 0.0);
    in_dense.assign(n, 0);
    pivot_row_value.assign(n, 0.0);
    pivot_row_mark.assign(n, 0);
  }
};

}  // namespace

void LuFactorization::clear() {
  dim_ = 0;
  factorized_ = false;
  pivot_row_.clear();
  pivot_col_.clear();
  l_start_.clear();
  l_row_.clear();
  l_value_.clear();
  u_start_.clear();
  u_index_.clear();
  u_value_.clear();
  u_diag_.clear();
  eta_slot_.clear();
  eta_start_.assign(1, 0);
  eta_index_.clear();
  eta_value_.clear();
  eta_pivot_.clear();
  work_.clear();
}

Status LuFactorization::factorize(const AugmentedMatrix& matrix, const Basis& basis,
                                  Real pivot_tolerance) {
  return factorize_impl(matrix, basis, pivot_tolerance, nullptr, nullptr);
}

Status LuFactorization::factorize_repairing(const AugmentedMatrix& matrix, Basis& basis,
                                            Real pivot_tolerance, std::size_t* repairs) {
  const std::size_t rows = matrix.num_rows();
  std::size_t replaced = 0;

  // Every repair puts at least one logical into the basis that was not there
  // before, and there are only `rows` logicals, so this loop is bounded. A
  // stall is repaired as a whole (see factorize_impl): all its active slots at
  // once, paired with the logicals of all its active rows. One at a time is
  // also correct but refactorizes once per column -- measured at 1418
  // refactorizations for one crossover crash basis on Netlib dfl001.
  std::vector<std::size_t> stall_rows;
  std::vector<std::size_t> stall_slots;
  for (std::size_t attempt = 0; attempt <= rows; ++attempt) {
    const Status status =
        factorize_impl(matrix, basis, pivot_tolerance, &stall_rows, &stall_slots);
    if (status.ok()) {
      if (repairs != nullptr) *repairs = replaced;
      return status;
    }
    // A dimension mismatch is the caller's bug, not a singular basis.
    if (status.error().code != ErrorCode::NumericalError) return status;
    if (stall_rows.empty() || stall_rows.size() != stall_slots.size()) return status;

    for (std::size_t k = 0; k < stall_rows.size(); ++k) {
      const std::size_t logical = matrix.logical_of_row(stall_rows[k]);
      if (basis.status[logical] == VarStatus::Basic) {
        // The header argues this is unreachable: a basic logical's column is
        // the unit vector of its row, so an active row with its logical in an
        // active slot would have offered a pivot. If it is ever reached, the
        // argument is wrong somewhere and silently carrying on would put a
        // column in the basis twice. Report instead.
        return status;
      }
      const std::size_t slot = stall_slots[k];
      const auto displaced = static_cast<std::size_t>(basis.basic[slot]);
      // Any non-Basic status makes the basis well formed again. Which bound
      // the displaced column should actually rest on depends on the caller's
      // WORKING bounds (phase 1 installs artificial ones this class cannot
      // see), so this picks from the model's own bounds and leaves the caller
      // to re-place it.
      if (core::is_finite_bound(matrix.lower(displaced))) {
        basis.status[displaced] = core::is_finite_bound(matrix.upper(displaced)) &&
                                          matrix.lower(displaced) == matrix.upper(displaced)
                                      ? VarStatus::Fixed
                                      : VarStatus::AtLower;
      } else if (core::is_finite_bound(matrix.upper(displaced))) {
        basis.status[displaced] = VarStatus::AtUpper;
      } else {
        basis.status[displaced] = VarStatus::Free;
      }
      basis.basic[slot] = static_cast<Index>(logical);
      basis.status[logical] = VarStatus::Basic;
      ++replaced;
    }
  }

  if (repairs != nullptr) *repairs = replaced;
  return core::make_error(ErrorCode::NumericalError,
                          "basis remained singular after " + std::to_string(replaced) +
                              " column substitutions");
}

Status LuFactorization::factorize_impl(const AugmentedMatrix& matrix, const Basis& basis,
                                       Real pivot_tolerance,
                                       std::vector<std::size_t>* stall_rows,
                                       std::vector<std::size_t>* stall_slots) {
  clear();

  if (!basis.validate()) {
    return core::make_error(ErrorCode::DimensionMismatch,
                            "basis does not satisfy its own invariants");
  }
  if (basis.num_rows() != matrix.num_rows()) {
    return core::make_error(ErrorCode::DimensionMismatch,
                            "basis has " + std::to_string(basis.num_rows()) +
                                " slots for a matrix with " +
                                std::to_string(matrix.num_rows()) + " rows");
  }

  dim_ = matrix.num_rows();
  work_.assign(dim_, 0.0);
  if (dim_ == 0) {
    factorized_ = true;
    return Status::Ok();
  }

  Workspace ws;
  ws.resize(dim_);

  // Load the basis columns. Slot r holds `Ahat` column `basis.basic[r]`.
  for (std::size_t r = 0; r < dim_; ++r) {
    const auto w = static_cast<std::size_t>(basis.basic[r]);
    auto& col = ws.col_entries[r];
    matrix.for_each_in_column(w, [&](std::size_t i, Real value) {
      if (std::fabs(value) <= kLuDropTolerance) return;
      col.emplace_back(static_cast<Index>(i), value);
      ws.row_pattern[i].push_back(static_cast<Index>(r));
      ++ws.row_count[i];
    });
    ws.col_count[r] = static_cast<Index>(col.size());
    ws.bucket[col.size()].push_back(static_cast<Index>(r));
  }

  pivot_row_.resize(dim_);
  pivot_col_.resize(dim_);
  u_diag_.resize(dim_);
  l_start_.assign(dim_ + 1, 0);
  u_start_.assign(dim_ + 1, 0);

  for (std::size_t step = 0; step < dim_; ++step) {
    // ---- pivot selection: Markowitz cost under a threshold constraint ----
    Index best_col = -1;
    Index best_row = -1;
    Real best_value = 0.0;
    double best_cost = std::numeric_limits<double>::max();
    std::size_t examined = 0;

    for (std::size_t count = 1; count <= dim_ && examined < kMarkowitzCandidates; ++count) {
      auto& bk = ws.bucket[count];
      std::size_t idx = 0;
      while (idx < bk.size() && examined < kMarkowitzCandidates) {
        const Index q = bk[idx];
        const auto qi = static_cast<std::size_t>(q);
        if (ws.col_active[qi] == 0 || static_cast<std::size_t>(ws.col_count[qi]) != count) {
          bk[idx] = bk.back();
          bk.pop_back();
          continue;
        }

        Real max_abs = 0.0;
        for (const auto& [row, value] : ws.col_entries[qi]) {
          if (ws.row_active[static_cast<std::size_t>(row)] == 0) continue;
          max_abs = std::fmax(max_abs, std::fabs(value));
        }
        if (max_abs > kLuDropTolerance) {
          const Real threshold = pivot_tolerance * max_abs;
          for (const auto& [row, value] : ws.col_entries[qi]) {
            const auto ri = static_cast<std::size_t>(row);
            if (ws.row_active[ri] == 0) continue;
            if (std::fabs(value) < threshold) continue;
            const double cost = static_cast<double>(ws.row_count[ri] - 1) *
                                static_cast<double>(ws.col_count[qi] - 1);
            const bool better =
                best_col < 0 || cost < best_cost ||
                (cost == best_cost && std::fabs(value) > std::fabs(best_value));
            if (better) {
              best_cost = cost;
              best_col = q;
              best_row = row;
              best_value = value;
            }
          }
          ++examined;
        }
        ++idx;
      }

      // A count-`count` column cannot produce a cost below `(count - 1)^2`
      // times the smallest possible row count, so once the best candidate is
      // at or under that bound no unexamined column can beat it.
      const auto bound = static_cast<double>(count - 1) * static_cast<double>(count - 1);
      if (best_col >= 0 && best_cost <= bound) break;
    }

    if (best_col < 0) {
      // Every active column is below the drop tolerance on every active row:
      // the pivot search above rejects a column only when its largest active
      // entry is. So the whole active submatrix is numerically zero, and ALL
      // the active slots are dependent on the columns already pivoted -- not
      // just one of them. Report them all, so the caller can repair them in
      // one pass instead of refactorizing once per column.
      if (stall_rows != nullptr && stall_slots != nullptr) {
        stall_rows->clear();
        stall_slots->clear();
        for (std::size_t i = 0; i < dim_; ++i) {
          if (ws.row_active[i] != 0) stall_rows->push_back(i);
        }
        // Structural slots first: displacing those strictly increases the
        // number of basic logicals, which is bounded by m, so the repair loop
        // cannot cycle. An active logical slot is still progress, since the
        // logical installed there is a different one.
        for (std::size_t q = 0; q < dim_; ++q) {
          if (ws.col_active[q] != 0 && !matrix.is_logical(static_cast<std::size_t>(basis.basic[q]))) {
            stall_slots->push_back(q);
          }
        }
        for (std::size_t q = 0; q < dim_; ++q) {
          if (ws.col_active[q] != 0 && matrix.is_logical(static_cast<std::size_t>(basis.basic[q]))) {
            stall_slots->push_back(q);
          }
        }
      }
      return singular(step);
    }

    const auto pivot_q = static_cast<std::size_t>(best_col);
    const auto pivot_p = static_cast<std::size_t>(best_row);

    // ---- gather the pivot row, pruning stale pattern entries ----
    ws.pivot_row_cols.clear();
    {
      auto& pattern = ws.row_pattern[pivot_p];
      std::size_t write = 0;
      for (std::size_t t = 0; t < pattern.size(); ++t) {
        const Index q = pattern[t];
        const auto qi = static_cast<std::size_t>(q);
        if (ws.col_active[qi] == 0) continue;
        if (ws.pivot_row_mark[qi] != 0) continue;

        bool found = false;
        Real value = 0.0;
        for (const auto& entry : ws.col_entries[qi]) {
          if (static_cast<std::size_t>(entry.first) == pivot_p) {
            value = entry.second;
            found = true;
            break;
          }
        }
        if (!found) continue;

        ws.pivot_row_mark[qi] = 1;
        ws.pivot_row_value[qi] = value;
        pattern[write++] = q;
        if (qi != pivot_q) ws.pivot_row_cols.push_back(q);
      }
      pattern.resize(write);
    }

    // ---- multipliers from the pivot column ----
    ws.multipliers.clear();
    for (const auto& [row, value] : ws.col_entries[pivot_q]) {
      const auto ri = static_cast<std::size_t>(row);
      if (ws.row_active[ri] == 0 || ri == pivot_p) continue;
      ws.multipliers.emplace_back(row, value / best_value);
    }

    // ---- record the factor ----
    pivot_row_[step] = best_row;
    pivot_col_[step] = best_col;
    u_diag_[step] = best_value;
    for (const auto& [row, mult] : ws.multipliers) {
      l_row_.push_back(row);
      l_value_.push_back(mult);
    }
    l_start_[step + 1] = static_cast<Index>(l_row_.size());
    for (const Index q : ws.pivot_row_cols) {
      u_index_.push_back(q);
      u_value_.push_back(ws.pivot_row_value[static_cast<std::size_t>(q)]);
    }
    u_start_[step + 1] = static_cast<Index>(u_index_.size());

    ws.row_active[pivot_p] = 0;
    ws.col_active[pivot_q] = 0;

    // Every active row of the pivot column loses that entry with the column.
    for (const auto& entry : ws.col_entries[pivot_q]) {
      const auto ri = static_cast<std::size_t>(entry.first);
      if (ws.row_active[ri] == 0) continue;
      --ws.row_count[ri];
    }

    // ---- rank-1 update of the remaining pivot-row columns ----
    for (const Index q : ws.pivot_row_cols) {
      const auto qi = static_cast<std::size_t>(q);
      const Real row_value = ws.pivot_row_value[qi];
      auto& col = ws.col_entries[qi];

      ws.touched.clear();
      for (const auto& [row, value] : col) {
        const auto ri = static_cast<std::size_t>(row);
        if (ws.row_active[ri] == 0) continue;  // drops the pivot row with it
        ws.dense[ri] = value;
        ws.in_dense[ri] = 1;
        ws.touched.push_back(row);
      }

      for (const auto& [row, mult] : ws.multipliers) {
        const auto ri = static_cast<std::size_t>(row);
        const Real delta = -mult * row_value;
        if (ws.in_dense[ri] != 0) {
          ws.dense[ri] += delta;
        } else {
          ws.dense[ri] = delta;
          ws.in_dense[ri] = 1;
          ws.touched.push_back(row);
          ws.row_pattern[ri].push_back(q);
          ++ws.row_count[ri];
        }
      }

      col.clear();
      for (const Index row : ws.touched) {
        const auto ri = static_cast<std::size_t>(row);
        const Real value = ws.dense[ri];
        ws.dense[ri] = 0.0;
        ws.in_dense[ri] = 0;
        if (std::fabs(value) <= kLuDropTolerance) {
          // Exact or near-exact cancellation. The entry is gone, so the row
          // loses it whether it was original or fill created a moment ago.
          --ws.row_count[ri];
          continue;
        }
        col.emplace_back(row, value);
      }
      ws.col_count[qi] = static_cast<Index>(col.size());
      ws.bucket[col.size()].push_back(q);
    }

    for (const Index q : ws.row_pattern[pivot_p]) {
      ws.pivot_row_mark[static_cast<std::size_t>(q)] = 0;
    }
    ws.col_entries[pivot_q].clear();
  }

  factorized_ = true;
  return Status::Ok();
}

void LuFactorization::ftran(core::HostSpan<Real> v) const {
  if (dim_ == 0) return;

  // L: forward, in row space, in place on the caller's buffer.
  for (std::size_t k = 0; k < dim_; ++k) {
    const Real pivot = v[static_cast<std::size_t>(pivot_row_[k])];
    if (pivot == 0.0) continue;
    const auto begin = static_cast<std::size_t>(l_start_[k]);
    const auto end = static_cast<std::size_t>(l_start_[k + 1]);
    for (std::size_t t = begin; t < end; ++t) {
      v[static_cast<std::size_t>(l_row_[t])] -= l_value_[t] * pivot;
    }
  }

  // U: back substitution, writing slot space into the scratch buffer. This
  // cannot be done in place -- step k writes slot `pivot_col_[k]`, and some
  // earlier step still has to read row `pivot_row_[j]` with the same numeric
  // index.
  work_.assign(dim_, 0.0);
  for (std::size_t k = dim_; k-- > 0;) {
    Real sum = v[static_cast<std::size_t>(pivot_row_[k])];
    const auto begin = static_cast<std::size_t>(u_start_[k]);
    const auto end = static_cast<std::size_t>(u_start_[k + 1]);
    for (std::size_t t = begin; t < end; ++t) {
      sum -= u_value_[t] * work_[static_cast<std::size_t>(u_index_[t])];
    }
    work_[static_cast<std::size_t>(pivot_col_[k])] = sum / u_diag_[k];
  }
  for (std::size_t i = 0; i < dim_; ++i) v[i] = work_[i];

  apply_etas_forward(v);
}

void LuFactorization::btran(core::HostSpan<Real> v) const {
  if (dim_ == 0) return;

  // rho = B^-T v = (E^-1 B_base^-1)^T v = B_base^-T (E^-T v), so the eta file
  // is applied FIRST here and last in ftran.
  apply_etas_reverse(v);

  // U^-T: forward, consuming `v` (slot space) and producing row space.
  work_.assign(dim_, 0.0);
  for (std::size_t k = 0; k < dim_; ++k) {
    const Real value = v[static_cast<std::size_t>(pivot_col_[k])] / u_diag_[k];
    work_[static_cast<std::size_t>(pivot_row_[k])] = value;
    if (value == 0.0) continue;
    const auto begin = static_cast<std::size_t>(u_start_[k]);
    const auto end = static_cast<std::size_t>(u_start_[k + 1]);
    for (std::size_t t = begin; t < end; ++t) {
      v[static_cast<std::size_t>(u_index_[t])] -= u_value_[t] * value;
    }
  }

  // L^-T: reverse, in row space.
  for (std::size_t k = dim_; k-- > 0;) {
    const auto p = static_cast<std::size_t>(pivot_row_[k]);
    Real sum = work_[p];
    const auto begin = static_cast<std::size_t>(l_start_[k]);
    const auto end = static_cast<std::size_t>(l_start_[k + 1]);
    for (std::size_t t = begin; t < end; ++t) {
      sum -= l_value_[t] * work_[static_cast<std::size_t>(l_row_[t])];
    }
    work_[p] = sum;
  }
  for (std::size_t i = 0; i < dim_; ++i) v[i] = work_[i];
}

Status LuFactorization::update(std::size_t leaving_slot,
                               core::HostSpan<const Real> entering_ftran) {
  if (leaving_slot >= dim_ || entering_ftran.size() != dim_) {
    return core::make_error(ErrorCode::DimensionMismatch,
                            "product-form update called with a slot or column "
                            "that does not match the factorized dimension");
  }

  const Real pivot = entering_ftran[leaving_slot];
  if (std::fabs(pivot) < kMinEtaPivot) {
    return core::make_error(ErrorCode::NumericalError,
                            "product-form pivot " + std::to_string(pivot) +
                                " is below the eta floor; refactorize instead");
  }

  eta_slot_.push_back(static_cast<Index>(leaving_slot));
  eta_pivot_.push_back(pivot);
  for (std::size_t i = 0; i < dim_; ++i) {
    if (i == leaving_slot) continue;
    const Real value = entering_ftran[i];
    if (std::fabs(value) <= kLuDropTolerance) continue;
    eta_index_.push_back(static_cast<Index>(i));
    eta_value_.push_back(value);
  }
  eta_start_.push_back(static_cast<Index>(eta_index_.size()));
  return Status::Ok();
}

void LuFactorization::apply_etas_forward(core::HostSpan<Real> v) const {
  const std::size_t count = eta_slot_.size();
  for (std::size_t t = 0; t < count; ++t) {
    const auto slot = static_cast<std::size_t>(eta_slot_[t]);
    const Real scaled = v[slot] / eta_pivot_[t];
    if (scaled != 0.0) {
      const auto begin = static_cast<std::size_t>(eta_start_[t]);
      const auto end = static_cast<std::size_t>(eta_start_[t + 1]);
      for (std::size_t k = begin; k < end; ++k) {
        v[static_cast<std::size_t>(eta_index_[k])] -= scaled * eta_value_[k];
      }
    }
    v[slot] = scaled;
  }
}

void LuFactorization::apply_etas_reverse(core::HostSpan<Real> v) const {
  for (std::size_t t = eta_slot_.size(); t-- > 0;) {
    const auto slot = static_cast<std::size_t>(eta_slot_[t]);
    const auto begin = static_cast<std::size_t>(eta_start_[t]);
    const auto end = static_cast<std::size_t>(eta_start_[t + 1]);
    Real dot = 0.0;
    for (std::size_t k = begin; k < end; ++k) {
      dot += eta_value_[k] * v[static_cast<std::size_t>(eta_index_[k])];
    }
    v[slot] = (v[slot] - dot) / eta_pivot_[t];
  }
}

}  // namespace sovsolve::solver::simplex
