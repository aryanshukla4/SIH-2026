#include "sovsolve/solver/NormalFactor.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "sovsolve/solver/SparseLdl.hpp"

#ifdef SOVSOLVE_HAVE_CHOLMOD
#include <cholmod.h>
#endif

namespace sovsolve::solver {

namespace {

/// A column counts as dense past this many nonzeros... ([G97] leaves the
/// threshold to the implementation; this is OURS, tuned so Netlib `israel`'s
/// 19 dense columns are caught and a 2-nonzero network column never is.)
std::size_t dense_threshold(std::size_t m) { return std::max<std::size_t>(20, m / 10); }

/// ...and at most this many are kept out, the densest first: each one costs
/// the CG around the factor roughly one extra iteration.
constexpr std::size_t kMaxDense = 50;

/// SparseLdl's pivot rule: a pivot keeping less than this fraction of its
/// original diagonal is a dependent row ([W99], see SparseLdl.hpp).
constexpr Real kPivotTolerance = 1e-14;  // OURS, see SparseLdl.hpp's pivot rule

}  // namespace

// ---------------------------------------------------------------------------

struct NormalFactor::Backend {
#ifdef SOVSOLVE_HAVE_CHOLMOD
  cholmod_common common{};
  cholmod_sparse* a = nullptr;
  cholmod_factor* l = nullptr;
  std::vector<std::size_t> lower_pos;  ///< full-pattern position of each lower entry

  Backend() { cholmod_start(&common); }
  ~Backend() {
    if (l != nullptr) cholmod_free_factor(&l, &common);
    if (a != nullptr) cholmod_free_sparse(&a, &common);
    cholmod_finish(&common);
  }
#else
  SparseLdl ldl;
#endif
};

NormalFactor::NormalFactor(const model::CanonicalProblem& problem)
    : problem_(&problem), m_(problem.num_rows()), backend_(std::make_unique<Backend>()) {
  const std::size_t n = problem.num_cols();
  const auto& csc = problem.A.csc;
  const auto& csr = problem.A.csr;

  // Dense columns: the densest above the threshold, at most kMaxDense.
  is_dense_.assign(n, 0);
  std::vector<std::size_t> candidates;
  for (std::size_t j = 0; j < n; ++j) {
    if (csc.slice_nnz(j) > dense_threshold(m_)) candidates.push_back(j);
  }
  std::stable_sort(candidates.begin(), candidates.end(), [&](std::size_t a, std::size_t b) {
    return csc.slice_nnz(a) > csc.slice_nnz(b);
  });
  if (candidates.size() > kMaxDense) candidates.resize(kMaxDense);
  for (const std::size_t j : candidates) is_dense_[j] = 1;
  num_dense_ = candidates.size();

  // Pattern of A_S A_S', row by row (it is symmetric, so rows are columns).
  std::vector<std::size_t> mark(m_, static_cast<std::size_t>(-1));
  std::vector<std::size_t> row;
  col_ptr_.assign(m_ + 1, 0);
  for (std::size_t i = 0; i < m_; ++i) {
    row.clear();
    row.push_back(i);
    mark[i] = i;
    for (auto p = csr.slice_begin(i); p < csr.slice_end(i); ++p) {
      const auto j = static_cast<std::size_t>(csr.indices()[p]);
      if (is_dense_[j]) continue;
      for (auto q = csc.slice_begin(j); q < csc.slice_end(j); ++q) {
        const auto r = static_cast<std::size_t>(csc.indices()[q]);
        if (mark[r] != i) {
          mark[r] = i;
          row.push_back(r);
        }
      }
    }
    std::sort(row.begin(), row.end());
    row_idx_.insert(row_idx_.end(), row.begin(), row.end());
    col_ptr_[i + 1] = row_idx_.size();
  }
  values_.assign(row_idx_.size(), 0.0);
  work_.assign(m_, 0.0);

#ifdef SOVSOLVE_HAVE_CHOLMOD
  Backend& b = *backend_;
  for (std::size_t i = 0; i < m_; ++i) {
    for (std::size_t p = col_ptr_[i]; p < col_ptr_[i + 1]; ++p) {
      if (row_idx_[p] >= i) b.lower_pos.push_back(p);
    }
  }
  b.a = cholmod_allocate_sparse(m_, m_, b.lower_pos.size(), /*sorted=*/1, /*packed=*/1,
                                /*stype=*/-1, CHOLMOD_REAL, &b.common);
  auto* ap = static_cast<int*>(b.a->p);
  auto* ai = static_cast<int*>(b.a->i);
  std::size_t t = 0;
  for (std::size_t i = 0; i < m_; ++i) {
    ap[i] = static_cast<int>(t);
    for (std::size_t p = col_ptr_[i]; p < col_ptr_[i + 1]; ++p) {
      if (row_idx_[p] >= i) ai[t++] = static_cast<int>(row_idx_[p]);
    }
  }
  ap[m_] = static_cast<int>(t);
  b.l = cholmod_analyze(b.a, &b.common);
#else
  (void)backend_->ldl.analyze(m_, col_ptr_, row_idx_);
#endif
}

NormalFactor::~NormalFactor() = default;

core::Status NormalFactor::factorize(const core::RealVector& theta,
                                     const core::RealVector& diag_add) {
  ready_ = false;
  const auto& csc = problem_->A.csc;
  const auto& csr = problem_->A.csr;

  // M = A_S Theta A_S' + D, one row at a time through a dense accumulator.
  for (std::size_t i = 0; i < m_; ++i) {
    for (auto p = csr.slice_begin(i); p < csr.slice_end(i); ++p) {
      const auto j = static_cast<std::size_t>(csr.indices()[p]);
      if (is_dense_[j]) continue;
      const Real t = csr.values()[p] * theta[j];
      for (auto q = csc.slice_begin(j); q < csc.slice_end(j); ++q) {
        work_[static_cast<std::size_t>(csc.indices()[q])] += t * csc.values()[q];
      }
    }
    work_[i] += diag_add[i];
    for (std::size_t p = col_ptr_[i]; p < col_ptr_[i + 1]; ++p) {
      values_[p] = work_[row_idx_[p]];
      work_[row_idx_[p]] = 0.0;
    }
  }

#ifdef SOVSOLVE_HAVE_CHOLMOD
  Backend& b = *backend_;
  auto* ax = static_cast<double*>(b.a->x);
  for (std::size_t t = 0; t < b.lower_pos.size(); ++t) ax[t] = values_[b.lower_pos[t]];
  cholmod_factorize(b.a, b.l, &b.common);
  if (b.common.status != CHOLMOD_OK || b.l->minor < b.l->n) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "NormalFactor: normal equations not positive definite");
  }
#else
  if (core::Status st = backend_->ldl.factorize(values_, kPivotTolerance); !st.ok()) return st;
#endif
  ready_ = true;
  return core::Status::Ok();
}

void NormalFactor::solve(std::vector<Real>& x) const {
#ifdef SOVSOLVE_HAVE_CHOLMOD
  Backend& b = *backend_;
  cholmod_dense* rhs = cholmod_allocate_dense(m_, 1, m_, CHOLMOD_REAL, &b.common);
  auto* r = static_cast<double*>(rhs->x);
  std::copy(x.begin(), x.end(), r);
  cholmod_dense* sol = cholmod_solve(CHOLMOD_A, b.l, rhs, &b.common);
  const auto* s = static_cast<const double*>(sol->x);
  std::copy(s, s + m_, x.begin());
  cholmod_free_dense(&sol, &b.common);
  cholmod_free_dense(&rhs, &b.common);
#else
  backend_->ldl.solve(x);
#endif
}

std::size_t NormalFactor::nnz_l() const noexcept {
#ifdef SOVSOLVE_HAVE_CHOLMOD
  return static_cast<std::size_t>(backend_->common.lnz);
#else
  return backend_->ldl.nnz_l();
#endif
}

std::size_t NormalFactor::modified_pivots() const noexcept {
#ifdef SOVSOLVE_HAVE_CHOLMOD
  return 0;
#else
  return backend_->ldl.modified_pivots();
#endif
}

const char* NormalFactor::backend() const noexcept {
#ifdef SOVSOLVE_HAVE_CHOLMOD
  return "cholmod";
#else
  return "sparse-ldl";
#endif
}

}  // namespace sovsolve::solver
