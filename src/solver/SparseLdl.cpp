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

/// Widest supernode, in columns. OURS: keeps a dense block cache-sized; the
/// ordering and fill do not depend on it.
constexpr std::size_t kMaxSupernode = 64;

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

  // The full pattern of L, column by column in increasing row order: row k
  // of L is the etree reach of column k, the same walk as above.
  li_.assign(lp_[n], 0);
  std::vector<std::size_t> fill(n, 0);
  std::fill(flag_.begin(), flag_.end(), kNone);
  for (std::size_t k = 0; k < n; ++k) {
    flag_[k] = k;
    const std::size_t kk = perm_[k];
    for (std::size_t p = col_ptr_[kk]; p < col_ptr_[kk + 1]; ++p) {
      std::size_t i = pinv_[row_idx_[p]];
      if (i >= k) continue;
      for (; flag_[i] != k; i = static_cast<std::size_t>(parent_[i])) {
        li_[lp_[i] + fill[i]++] = k;
        flag_[i] = k;
      }
    }
  }
  lx_.assign(lp_[n], 0.0);
  d_.assign(n, 0.0);
  diag_.assign(n, 0.0);
  work_.assign(n, 0.0);
  relmap_.assign(n, 0);

  // Supernodes: column c joins c-1's when c is c-1's etree parent and c-1's
  // pattern is exactly {c} u c's pattern. OURS: capped at kMaxSupernode
  // columns so a block stays cache-sized.
  super_.clear();
  super_of_.assign(n, 0);
  for (std::size_t c = 0; c < n; ++c) {
    const bool extend = c > 0 && parent_[c - 1] == static_cast<std::ptrdiff_t>(c) &&
                        lnz_[c - 1] == lnz_[c] + 1 && c - super_.back() < kMaxSupernode;
    if (!extend) super_.push_back(c);
    super_of_[c] = super_.size() - 1;
  }
  super_.push_back(n);
  const std::size_t ns = super_.size() - 1;
  blk_off_.assign(ns + 1, 0);
  for (std::size_t s = 0; s < ns; ++s) {
    const std::size_t f = super_[s];
    const std::size_t ncol = super_[s + 1] - f;
    blk_off_[s + 1] = blk_off_[s] + (lnz_[f] + 1) * ncol;
  }
  blk_.assign(blk_off_[ns], 0.0);
  next_row_.assign(ns, 0);
  pending_.assign(ns, {});
  return core::Status::Ok();
}

core::Status SparseLdl::factorize(const std::vector<Real>& values, Real pivot_tolerance) {
  if (values.size() != row_idx_.size()) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "SparseLdl::factorize: values do not match the analysed pattern");
  }
  modified_ = 0;
  std::fill(blk_.begin(), blk_.end(), 0.0);
  for (auto& list : pending_) list.clear();
  const std::size_t ns = super_.size() - 1;

  for (std::size_t s = 0; s < ns; ++s) {
    const std::size_t f = super_[s];
    const std::size_t ncol = super_[s + 1] - f;
    const std::size_t nrow = lnz_[f] + 1;
    const std::size_t* rows_below = li_.data() + lp_[f];  // rows[1..nrow)
    Real* B = blk_.data() + blk_off_[s];
    const auto row_at = [&](std::size_t i) { return i == 0 ? f : rows_below[i - 1]; };
    for (std::size_t i = 0; i < nrow; ++i) relmap_[row_at(i)] = i;

    // Assemble this supernode's columns of A (lower triangle).
    for (std::size_t c = f; c < f + ncol; ++c) {
      const std::size_t kk = perm_[c];
      Real* col = B + nrow * (c - f);
      diag_[c] = 0.0;
      for (std::size_t p = col_ptr_[kk]; p < col_ptr_[kk + 1]; ++p) {
        const std::size_t r = pinv_[row_idx_[p]];
        if (r < c) continue;
        col[relmap_[r]] += values[p];
        if (r == c) diag_[c] += values[p];
      }
    }

    // Updates from every earlier supernode D whose rows reach this one:
    // B(i, j) -= sum_k L_D(i, k) d_k L_D(j, k), for D's rows i >= j in S.
    for (const std::size_t D : pending_[s]) {
      const std::size_t fd = super_[D];
      const std::size_t ncd = super_[D + 1] - fd;
      const std::size_t nrd = lnz_[fd] + 1;
      const std::size_t* rd = li_.data() + lp_[fd];  // D's rows 1..nrd
      const Real* BD = blk_.data() + blk_off_[D];
      const std::size_t q0 = next_row_[D];
      std::size_t q1 = q0;
      while (q1 < nrd && rd[q1 - 1] < f + ncol) ++q1;
      // The update block is formed DENSE first -- contiguous loops the
      // compiler can vectorise -- then scattered into B once, instead of
      // scattering every multiply-add through relmap_.
      const std::size_t h = nrd - q0;  // rows of the update
      const std::size_t w = q1 - q0;   // its columns
      update_.assign(h * w, 0.0);
      for (std::size_t k = 0; k < ncd; ++k) {
        const Real* lk = BD + nrd * k + q0;
        const Real dk = d_[fd + k];
        for (std::size_t a = 0; a < w; ++a) {
          const Real t = lk[a] * dk;
          if (t == 0.0) continue;
          Real* u = update_.data() + h * a;
          for (std::size_t b = a; b < h; ++b) u[b] += lk[b] * t;
        }
      }
      for (std::size_t a = 0; a < w; ++a) {
        Real* col = B + nrow * (rd[q0 + a - 1] - f);
        const Real* u = update_.data() + h * a;
        for (std::size_t b = a; b < h; ++b) col[relmap_[rd[q0 + b - 1]]] -= u[b];
      }
      next_row_[D] = q1;
      if (q1 < nrd) pending_[super_of_[rd[q1 - 1]]].push_back(D);
    }

    // Dense LDL' of the block, with [W99]'s pivot skipping.
    for (std::size_t jj = 0; jj < ncol; ++jj) {
      Real* cj = B + nrow * jj;
      for (std::size_t kk = 0; kk < jj; ++kk) {
        const Real* ck = B + nrow * kk;
        const Real t = ck[jj] * d_[f + kk];
        if (t == 0.0) continue;
        for (std::size_t i = jj; i < nrow; ++i) cj[i] -= ck[i] * t;
      }
      Real d = cj[jj];
      if (!std::isfinite(d) || d <= pivot_tolerance * std::fabs(diag_[f + jj])) {
        d = kHugePivot;
        ++modified_;
      }
      d_[f + jj] = d;
      const Real inv = 1.0 / d;
      for (std::size_t i = jj + 1; i < nrow; ++i) cj[i] *= inv;
    }

    // Hand this supernode to the first later supernode its rows reach.
    if (ncol < nrow) {
      next_row_[s] = ncol;
      pending_[super_of_[rows_below[ncol - 1]]].push_back(s);
    }

    // Column form for the solve: column c's below-diagonal entries are the
    // block rows after its own.
    for (std::size_t jj = 0; jj < ncol; ++jj) {
      const std::size_t c = f + jj;
      const Real* cj = B + nrow * jj;
      for (std::size_t i = jj + 1; i < nrow; ++i) lx_[lp_[c] + (i - jj - 1)] = cj[i];
    }
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
