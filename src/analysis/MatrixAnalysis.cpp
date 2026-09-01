#include "sovsolve/analysis/MatrixAnalysis.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace sovsolve::analysis {
namespace {

using core::Index;
using core::Real;

std::size_t log2_bucket(std::size_t n) noexcept {
  std::size_t b = 0;
  while (n > 1) {
    n >>= 1;
    ++b;
  }
  return b;
}

/// Per-slice nonzero statistics, from one linear scan of an offsets array.
template <typename OffsetsSpan>
NnzDistribution distribution_of(const OffsetsSpan& offsets, std::size_t count) {
  NnzDistribution d;
  if (count == 0) return d;

  d.min = static_cast<std::size_t>(-1);
  std::size_t total = 0;
  std::size_t widest = 0;

  for (std::size_t i = 0; i < count; ++i) {
    const auto nnz = static_cast<std::size_t>(offsets[i + 1] - offsets[i]);
    total += nnz;
    if (nnz < d.min) d.min = nnz;
    if (nnz > d.max) d.max = nnz;
    if (nnz == 0) ++d.empty_count;
    if (nnz == 1) ++d.singleton_count;
    if (nnz > 0) widest = std::max(widest, log2_bucket(nnz));
  }

  d.mean = static_cast<double>(total) / static_cast<double>(count);
  d.histogram.assign(widest + 1, 0);
  for (std::size_t i = 0; i < count; ++i) {
    const auto nnz = static_cast<std::size_t>(offsets[i + 1] - offsets[i]);
    if (nnz > 0) ++d.histogram[log2_bucket(nnz)];
  }
  return d;
}

/// Hash a slice's sparsity pattern *and* its values, so only genuinely
/// identical rows/columns collide. Candidates still need direct comparison
/// before a presolver acts on them -- a hash match is not a proof.
template <typename Mat>
void find_duplicate_candidates(const Mat& m, std::vector<std::pair<Index, Index>>& out) {
  const auto off = m.offsets();
  const auto idx = m.indices();
  const auto val = m.values();

  std::unordered_map<std::uint64_t, Index> seen;
  seen.reserve(m.major_dim());

  for (std::size_t i = 0; i < m.major_dim(); ++i) {
    const auto b = static_cast<std::size_t>(off[i]);
    const auto e = static_cast<std::size_t>(off[i + 1]);
    if (b == e) continue;  // empty slices are the analyzer's own category

    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t k = b; k < e; ++k) {
      h ^= static_cast<std::uint64_t>(idx[k]);
      h *= 1099511628211ULL;
      std::uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(Real));
      std::memcpy(&bits, &val[k], sizeof(bits));
      h ^= bits;
      h *= 1099511628211ULL;
    }

    const auto it = seen.find(h);
    if (it != seen.end()) {
      out.emplace_back(it->second, static_cast<Index>(i));
    } else {
      seen.emplace(h, static_cast<Index>(i));
    }
  }
}

}  // namespace

MatrixAnalysis analyze(const core::SparseMatrixPair<>& matrix,
                       const AnalysisOptions& options) {
  MatrixAnalysis a;
  a.rows = matrix.rows();
  a.cols = matrix.cols();
  a.nnz = matrix.nnz();
  a.density = matrix.csr.density();

  // Row statistics from the CSR view, column statistics from the CSC view.
  // Each is a linear scan of contiguous memory; deriving one from the other
  // would mean a scatter, which is the whole reason both orientations exist.
  a.by_row = distribution_of(matrix.csr.offsets(), a.rows);
  a.by_column = distribution_of(matrix.csc.offsets(), a.cols);

  // -- value magnitudes ----------------------------------------------------
  {
    const auto val = matrix.csr.values();
    Real lo = 0.0;
    Real hi = 0.0;
    bool first = true;
    for (std::size_t k = 0; k < a.nnz; ++k) {
      const Real v = val[k];
      if (std::isnan(v) || std::isinf(v)) {
        a.has_invalid_values = true;
        continue;
      }
      const Real m = std::fabs(v);
      if (m == 0.0) continue;
      if (first) {
        lo = hi = m;
        first = false;
      } else {
        if (m < lo) lo = m;
        if (m > hi) hi = m;
      }
    }
    a.values.min_abs = lo;
    a.values.max_abs = hi;
    a.values.dynamic_range = lo > 0.0 ? hi / lo : 1.0;
  }

  // -- dense columns -------------------------------------------------------
  //
  // The field the KKT builder needs most. One dense column a_j contributes
  // theta_j * a_j * a_j' to A*Theta*A' -- a full rank-1 dense m x m update
  // that makes the whole normal-equations matrix dense no matter how sparse A
  // is. See docs/FORMULATION.md section 10.1.
  {
    const auto sqrt_rule = options.dense_column_factor *
                           std::sqrt(static_cast<double>(a.rows));
    a.dense_column_threshold =
        std::max(options.dense_column_min_nnz,
                 static_cast<std::size_t>(sqrt_rule));

    const auto off = matrix.csc.offsets();
    for (std::size_t j = 0; j < a.cols; ++j) {
      const auto nnz = static_cast<std::size_t>(off[j + 1] - off[j]);
      if (nnz > a.dense_column_threshold) {
        a.dense_columns.push_back(static_cast<Index>(j));
      }
    }
  }

  // -- structure for the presolver -----------------------------------------
  {
    const auto roff = matrix.csr.offsets();
    for (std::size_t i = 0; i < a.rows; ++i) {
      const auto nnz = static_cast<std::size_t>(roff[i + 1] - roff[i]);
      if (nnz == 0) a.empty_rows.push_back(static_cast<Index>(i));
      else if (nnz == 1) a.singleton_rows.push_back(static_cast<Index>(i));
    }
    const auto coff = matrix.csc.offsets();
    for (std::size_t j = 0; j < a.cols; ++j) {
      const auto nnz = static_cast<std::size_t>(coff[j + 1] - coff[j]);
      if (nnz == 0) a.empty_columns.push_back(static_cast<Index>(j));
      else if (nnz == 1) a.singleton_columns.push_back(static_cast<Index>(j));
    }
  }

  if (options.detect_duplicates) {
    find_duplicate_candidates(matrix.csr, a.duplicate_row_candidates);
    find_duplicate_candidates(matrix.csc, a.duplicate_column_candidates);
  }

  // -- symmetry ------------------------------------------------------------
  //
  // Meaningful for Q. Since CSR and CSC hold the same matrix in transposed
  // order, comparing them slice-by-slice tests symmetry without a second
  // traversal of either.
  if (a.rows == a.cols && a.rows > 0) {
    const auto roff = matrix.csr.offsets();
    const auto ridx = matrix.csr.indices();
    const auto rval = matrix.csr.values();
    const auto coff = matrix.csc.offsets();
    const auto cidx = matrix.csc.indices();
    const auto cval = matrix.csc.values();

    bool structural = true;
    bool numerical = true;
    for (std::size_t i = 0; i < a.rows && structural; ++i) {
      if (roff[i] != coff[i] || roff[i + 1] != coff[i + 1]) {
        structural = false;
        break;
      }
      for (auto k = static_cast<std::size_t>(roff[i]);
           k < static_cast<std::size_t>(roff[i + 1]); ++k) {
        if (ridx[k] != cidx[k]) {
          structural = false;
          break;
        }
        if (std::fabs(rval[k] - cval[k]) > options.symmetry_tolerance) {
          numerical = false;
        }
      }
    }
    a.structurally_symmetric = structural;
    a.numerically_symmetric = structural && numerical;
  }

  // -- storage recommendation ----------------------------------------------
  //
  // Returned as data, not as a verdict. module_corrected.txt section 2 is
  // explicit that no universal density threshold should be baked in, and the
  // caller has context this module does not -- in particular that the choice
  // should follow the fill-in of the reduced Newton system rather than the
  // density of A.
  {
    auto& r = a.recommendation;
    r.density = a.density;
    r.sparse_bytes = a.nnz * (sizeof(Real) + sizeof(Index)) +
                     (a.rows + 1) * sizeof(Index);
    r.dense_bytes = a.rows * a.cols * sizeof(Real);
    r.sparse_preferred = r.sparse_bytes < r.dense_bytes;
  }

  return a;
}

BoundClassification classify_bounds(core::HostSpan<const Real> lower,
                                    core::HostSpan<const Real> upper) noexcept {
  BoundClassification b;
  const std::size_t n = std::min(lower.size(), upper.size());
  b.total = n;
  for (std::size_t k = 0; k < n; ++k) {
    const Real lo = lower[k];
    const Real hi = upper[k];
    const bool lf = core::is_finite_bound(lo);
    const bool hf = core::is_finite_bound(hi);
    if (lo > hi) {
      ++b.inconsistent_count;
      continue;
    }
    if (lf && hf) {
      if (lo == hi) {
        ++b.fixed_count;
      } else {
        ++b.boxed_count;
      }
    } else if (lf) {
      ++b.lower_only_count;
    } else if (hf) {
      ++b.upper_only_count;
    } else {
      ++b.free_count;
    }
  }
  return b;
}

}  // namespace sovsolve::analysis
