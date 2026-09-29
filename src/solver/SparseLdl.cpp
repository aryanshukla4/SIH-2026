#include "sovsolve/solver/SparseLdl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <set>
#include <utility>
#include <vector>

namespace sovsolve::solver {

namespace {

constexpr std::size_t kNone = static_cast<std::size_t>(-1);

/// A pivot replaced by [W99]'s rule. Large enough that 1/d is negligible
/// against any real entry, small enough that d * y stays finite.
constexpr Real kHugePivot = 1e128;

/// Minimum degree on the explicit elimination graph. Deterministic: the set
/// orders by (degree, index).
std::vector<std::size_t> minimum_degree(std::size_t n, const std::vector<std::size_t>& col_ptr,
                                        const std::vector<std::size_t>& row_idx) {
  std::vector<std::vector<std::size_t>> adj(n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
      if (row_idx[p] != j) adj[j].push_back(row_idx[p]);
    }
    std::sort(adj[j].begin(), adj[j].end());
    adj[j].erase(std::unique(adj[j].begin(), adj[j].end()), adj[j].end());
  }

  std::set<std::pair<std::size_t, std::size_t>> queue;
  for (std::size_t j = 0; j < n; ++j) queue.emplace(adj[j].size(), j);

  std::vector<std::size_t> order;
  order.reserve(n);
  std::vector<std::size_t> merged;
  while (!queue.empty()) {
    const std::size_t v = queue.begin()->second;
    queue.erase(queue.begin());
    order.push_back(v);
    const std::vector<std::size_t> nb = std::move(adj[v]);
    adj[v].clear();
    // Eliminating v joins its neighbours into a clique.
    for (const std::size_t u : nb) {
      queue.erase({adj[u].size(), u});
      merged.clear();
      std::set_union(adj[u].begin(), adj[u].end(), nb.begin(), nb.end(),
                     std::back_inserter(merged));
      merged.erase(std::remove_if(merged.begin(), merged.end(),
                                  [&](std::size_t w) { return w == u || w == v; }),
                   merged.end());
      adj[u].swap(merged);
      queue.emplace(adj[u].size(), u);
    }
  }
  return order;
}

}  // namespace

core::Status SparseLdl::analyze(std::size_t n, const std::vector<std::size_t>& col_ptr,
                                const std::vector<std::size_t>& row_idx) {
  n_ = n;
  col_ptr_ = col_ptr;
  row_idx_ = row_idx;
  perm_ = minimum_degree(n, col_ptr, row_idx);
  pinv_.assign(n, 0);
  for (std::size_t k = 0; k < n; ++k) pinv_[perm_[k]] = k;

  // Elimination tree and column counts, Davis ch. 4 (ldl_symbolic).
  parent_.assign(n, -1);
  flag_.assign(n, kNone);
  lnz_.assign(n, 0);
  for (std::size_t k = 0; k < n; ++k) {
    flag_[k] = k;
    const std::size_t kk = perm_[k];
    for (std::size_t p = col_ptr_[kk]; p < col_ptr_[kk + 1]; ++p) {
      std::size_t i = pinv_[row_idx_[p]];
      if (i >= k) continue;
      for (; flag_[i] != k; i = static_cast<std::size_t>(parent_[i])) {
        if (parent_[i] == -1) parent_[i] = static_cast<std::ptrdiff_t>(k);
        ++lnz_[i];
        flag_[i] = k;
      }
    }
  }
  lp_.assign(n + 1, 0);
  for (std::size_t k = 0; k < n; ++k) lp_[k + 1] = lp_[k] + lnz_[k];
  li_.assign(lp_[n], 0);
  lx_.assign(lp_[n], 0.0);
  d_.assign(n, 0.0);
  y_.assign(n, 0.0);
  work_.assign(n, 0.0);
  pattern_.assign(n, 0);
  return core::Status::Ok();
}

core::Status SparseLdl::factorize(const std::vector<Real>& values, Real pivot_tolerance) {
  if (values.size() != row_idx_.size()) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "SparseLdl::factorize: values do not match the analysed pattern");
  }
  const std::size_t n = n_;
  modified_ = 0;
  std::fill(lnz_.begin(), lnz_.end(), 0);
  std::fill(flag_.begin(), flag_.end(), kNone);

  // Up-looking LDL', Davis ch. 4 (ldl_numeric): row k of L from a sparse
  // triangular solve whose pattern is the reach of column k in the etree.
  for (std::size_t k = 0; k < n; ++k) {
    y_[k] = 0.0;
    std::size_t top = n;
    flag_[k] = k;
    const std::size_t kk = perm_[k];
    Real diag = 0.0;
    for (std::size_t p = col_ptr_[kk]; p < col_ptr_[kk + 1]; ++p) {
      std::size_t i = pinv_[row_idx_[p]];
      if (i > k) continue;
      y_[i] += values[p];
      if (i == k) diag += values[p];
      std::size_t len = 0;
      for (; flag_[i] != k; i = static_cast<std::size_t>(parent_[i])) {
        pattern_[len++] = i;
        flag_[i] = k;
      }
      while (len > 0) pattern_[--top] = pattern_[--len];
    }
    Real d = y_[k];
    y_[k] = 0.0;
    for (; top < n; ++top) {
      const std::size_t i = pattern_[top];
      const Real yi = y_[i];
      y_[i] = 0.0;
      const std::size_t end = lp_[i] + lnz_[i];
      for (std::size_t p = lp_[i]; p < end; ++p) y_[li_[p]] -= lx_[p] * yi;
      const Real lki = yi / d_[i];
      d -= lki * yi;
      li_[end] = k;
      lx_[end] = lki;
      ++lnz_[i];
    }
    // [AG99] section 5 / [W99]: a pivot that lost (almost) everything to
    // cancellation belongs to a dependent row; drop that component.
    if (!std::isfinite(d) || d <= pivot_tolerance * std::fabs(diag)) {
      d = kHugePivot;
      ++modified_;
    }
    d_[k] = d;
  }
  return core::Status::Ok();
}

void SparseLdl::solve(std::vector<Real>& x) const {
  const std::size_t n = n_;
  for (std::size_t k = 0; k < n; ++k) work_[k] = x[perm_[k]];
  for (std::size_t j = 0; j < n; ++j) {
    const Real wj = work_[j];
    for (std::size_t p = lp_[j]; p < lp_[j + 1]; ++p) work_[li_[p]] -= lx_[p] * wj;
  }
  for (std::size_t j = 0; j < n; ++j) work_[j] /= d_[j];
  for (std::size_t j = n; j-- > 0;) {
    Real wj = work_[j];
    for (std::size_t p = lp_[j]; p < lp_[j + 1]; ++p) wj -= lx_[p] * work_[li_[p]];
    work_[j] = wj;
  }
  for (std::size_t k = 0; k < n; ++k) x[perm_[k]] = work_[k];
}

}  // namespace sovsolve::solver
