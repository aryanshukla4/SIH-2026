#include "sovsolve/solver/SparseLdl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace sovsolve::solver {

namespace {

constexpr std::size_t kNone = static_cast<std::size_t>(-1);

/// A pivot replaced by [W99]'s rule. Large enough that 1/d is negligible
/// against any real entry, small enough that d * y stays finite.
constexpr Real kHugePivot = 1e128;

/// Approximate minimum degree on the QUOTIENT graph.
///
/// NOT TRANSCRIBED from a paper: written from general knowledge of the method
/// of Amestoy, Davis & Duff (SIAM J. Matrix Anal. Appl. 17, 1996), which the
/// project does not have. Judged by measurement instead -- fill against
/// CHOLMOD's own AMD on the same matrices, and ordering time against the
/// explicit-graph minimum degree it replaced (in git history).
///
/// An explicit elimination graph stores every fill edge and so costs
/// time and memory proportional to the factor. The quotient graph instead
/// represents an eliminated pivot by one ELEMENT, the clique of its remaining
/// neighbours, so a variable's neighbourhood is its variable list A_i plus
/// the union of its elements' lists; storage never exceeds the original
/// graph's.
///
/// Eliminating pivot p:
///   L_p  = A_p u (union of L_e over e in E_p), minus p; those elements are
///          absorbed into p, which becomes the new element.
///   For each i in L_p: E_i <- (E_i \ absorbed) u {p}; A_i <- A_i \ L_p \ {p}.
///   Degree: with w(e) = |L_e \ L_p| for each other element of i,
///          d_i = min( n_left - 1,  d_i_old + |L_p| - 1,
///                     |A_i| + |L_p| - 1 + sum_e w(e) )
///   -- an upper bound on the true external degree, which is what makes the
///   ordering "approximate". An element with w(e) = 0 lies inside L_p and is
///   absorbed as well.
///
/// Ties go to the most recently updated variable of the lowest degree
/// (buckets are LIFO), which is deterministic run to run.
std::vector<std::size_t> approximate_minimum_degree(std::size_t n,
                                                    const std::vector<std::size_t>& col_ptr,
                                                    const std::vector<std::size_t>& row_idx) {
  constexpr std::size_t kNil = static_cast<std::size_t>(-1);
  std::vector<std::vector<std::size_t>> A(n), E(n), L(n);
  std::vector<std::size_t> lsize(n, 0), degree(n, 0);
  std::vector<char> variable(n, 1), element_alive(n, 0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
      if (row_idx[p] != j) A[j].push_back(row_idx[p]);
    }
    std::sort(A[j].begin(), A[j].end());
    A[j].erase(std::unique(A[j].begin(), A[j].end()), A[j].end());
    degree[j] = A[j].size();
  }

  // Degree buckets: doubly linked, LIFO.
  std::vector<std::size_t> head(n + 1, kNil), next(n, kNil), prev(n, kNil);
  const auto insert = [&](std::size_t i) {
    const std::size_t d = std::min(degree[i], n);
    next[i] = head[d];
    prev[i] = kNil;
    if (head[d] != kNil) prev[head[d]] = i;
    head[d] = i;
  };
  const auto remove = [&](std::size_t i) {
    const std::size_t d = std::min(degree[i], n);
    if (prev[i] != kNil) next[prev[i]] = next[i]; else head[d] = next[i];
    if (next[i] != kNil) prev[next[i]] = prev[i];
  };
  for (std::size_t j = n; j-- > 0;) insert(j);

  std::vector<std::size_t> mark(n, kNil), wstamp(n, kNil), w(n, 0);
  std::vector<std::size_t> order;
  order.reserve(n);
  std::size_t min_degree = 0;

  for (std::size_t k = 0; k < n; ++k) {
    while (head[min_degree] == kNil) ++min_degree;
    const std::size_t p = head[min_degree];
    remove(p);
    order.push_back(p);
    variable[p] = 0;

    // L_p, absorbing p's elements.
    std::vector<std::size_t> lp;
    mark[p] = k;
    for (const std::size_t v : A[p]) {
      if (variable[v] && mark[v] != k) { mark[v] = k; lp.push_back(v); }
    }
    for (const std::size_t e : E[p]) {
      if (!element_alive[e]) continue;
      for (const std::size_t v : L[e]) {
        if (variable[v] && mark[v] != k) { mark[v] = k; lp.push_back(v); }
      }
      element_alive[e] = 0;
      L[e].clear();
      L[e].shrink_to_fit();
    }
    A[p].clear();
    A[p].shrink_to_fit();
    E[p].clear();
    E[p].shrink_to_fit();
    std::sort(lp.begin(), lp.end());  // deterministic order for the updates below
    element_alive[p] = 1;
    lsize[p] = lp.size();

    // w(e) = |L_e \ L_p| for every live element meeting L_p.
    for (const std::size_t i : lp) {
      for (const std::size_t e : E[i]) {
        if (!element_alive[e] || e == p) continue;
        if (wstamp[e] != k) { wstamp[e] = k; w[e] = lsize[e]; }
        --w[e];
      }
    }

    const std::size_t left = n - k - 1;
    for (const std::size_t i : lp) {
      remove(i);
      // Elements: drop the absorbed ones and those now inside L_p, add p.
      std::size_t ext = 0;
      std::size_t out = 0;
      for (const std::size_t e : E[i]) {
        if (!element_alive[e] || e == p) continue;
        if (wstamp[e] == k && w[e] == 0) {  // L_e inside L_p: absorb into p
          element_alive[e] = 0;
          continue;
        }
        E[i][out++] = e;
        ext += wstamp[e] == k ? w[e] : lsize[e];
      }
      E[i].resize(out);
      E[i].push_back(p);
      // Variables: those in L_p are now reached through element p.
      out = 0;
      for (const std::size_t v : A[i]) {
        if (variable[v] && mark[v] != k) A[i][out++] = v;
      }
      A[i].resize(out);

      const std::size_t bound = A[i].size() + (lp.size() - 1) + ext;
      degree[i] = std::min({left > 0 ? left - 1 : 0, degree[i] + lp.size() - 1, bound});
      insert(i);
      if (degree[i] < min_degree) min_degree = degree[i];
    }
    L[p] = std::move(lp);
    // Elements absorbed during the w pass leave stale ids behind in other
    // variables' E lists; they are skipped (element_alive) when met.
  }
  return order;
}

}  // namespace

core::Status SparseLdl::analyze(std::size_t n, const std::vector<std::size_t>& col_ptr,
                                const std::vector<std::size_t>& row_idx) {
  n_ = n;
  col_ptr_ = col_ptr;
  row_idx_ = row_idx;
  perm_ = approximate_minimum_degree(n, col_ptr, row_idx);
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
    for (std::size_t p = col_ptr_[kk]; p < col_ptr_[kk + 1]; ++p) {
      std::size_t i = pinv_[row_idx_[p]];
      if (i > k) continue;
      y_[i] += values[p];
      std::size_t len = 0;
      for (; flag_[i] != k; i = static_cast<std::size_t>(parent_[i])) {
        pattern_[len++] = i;
        flag_[i] = k;
      }
      while (len > 0) pattern_[--top] = pattern_[--len];
    }
    Real d = y_[k];
    const Real diag = std::fabs(d);  // this pivot's entry before elimination
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
    if (!std::isfinite(d) || d <= pivot_tolerance * diag) {
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
