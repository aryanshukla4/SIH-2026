// Structural and numerical analysis of a constraint matrix.
//
// module_corrected.txt section 2 specifies what the Matrix Analyzer should
// report but defines no output type, so nothing could consume its results.
// This is that type.
//
// Everything here is computed during the reader's CSR/CSC assembly pass, where
// the data is already in cache -- so the analysis is close to free rather than
// a second traversal.
//
// The field that matters most to other modules is `dense_columns`. A single
// dense column a_j contributes theta_j * a_j * a_j' to A*Theta*A' -- a full
// rank-1 dense m x m update that makes the entire normal-equations matrix
// dense no matter how sparse A is. Netlib and MIPLIB contain many such
// instances, and the KKT builder must split them out via Sherman-Morrison.
// It is a first-class field, not a footnote.

#ifndef SOVSOLVE_ANALYSIS_MATRIX_ANALYSIS_HPP
#define SOVSOLVE_ANALYSIS_MATRIX_ANALYSIS_HPP

#include <cstddef>
#include <utility>
#include <vector>

#include "sovsolve/core/SparseMatrix.hpp"
#include "sovsolve/core/Types.hpp"

namespace sovsolve::analysis {

using core::Index;
using core::Real;

/// Distribution of nonzeros over rows or columns.
struct NnzDistribution {
  std::size_t min = 0;
  std::size_t max = 0;
  double mean = 0.0;
  std::size_t empty_count = 0;      ///< slices with no entries
  std::size_t singleton_count = 0;  ///< slices with exactly one entry

  /// Log2-bucketed histogram: bucket k counts slices with nnz in
  /// [2^k, 2^(k+1)). Enough to see the shape without storing per-slice data.
  std::vector<std::size_t> histogram;
};

/// Magnitude range of the stored values. A crude conditioning proxy, available
/// without any factorization.
struct ValueRange {
  Real min_abs = 0.0;   ///< smallest nonzero magnitude
  Real max_abs = 0.0;
  /// max_abs / min_abs. Large values indicate the matrix needs scaling before
  /// the IPM sees it.
  double dynamic_range = 1.0;
};

/// Storage recommendation.
///
/// Returned as *data*, not as a hard-coded decision. module_corrected.txt
/// section 2 is explicit that no universal density threshold should be baked
/// in, and the caller has context this module does not -- notably that the
/// choice should ultimately follow the fill-in of the reduced Newton system,
/// not the density of A.
struct FormatRecommendation {
  bool sparse_preferred = true;
  double density = 0.0;
  /// Bytes required under each layout, so a caller can weigh memory directly.
  std::size_t sparse_bytes = 0;
  std::size_t dense_bytes = 0;
};

struct MatrixAnalysis {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t nnz = 0;
  double density = 0.0;

  NnzDistribution by_row;
  NnzDistribution by_column;
  ValueRange values;
  FormatRecommendation recommendation;

  // -- structure the KKT builder and presolver need ------------------------

  /// Columns dense enough to destroy the normal equations. Detected as
  /// `nnz(col) > max(dense_column_min_nnz, k * sqrt(rows))`.
  std::vector<Index> dense_columns;
  /// Threshold actually used, so the result is reproducible.
  std::size_t dense_column_threshold = 0;

  /// Fed to the presolver -- these are its cheapest and highest-yield rules.
  std::vector<Index> empty_rows;
  std::vector<Index> empty_columns;
  std::vector<Index> singleton_rows;
  std::vector<Index> singleton_columns;

  /// Candidate duplicate rows/columns, found by hashing sparsity patterns.
  /// Candidates, not confirmations: a hash collision must be resolved by
  /// direct comparison before acting.
  std::vector<std::pair<Index, Index>> duplicate_row_candidates;
  std::vector<std::pair<Index, Index>> duplicate_column_candidates;

  /// Structural symmetry, meaningful for Q. `false` for a rectangular matrix.
  bool structurally_symmetric = false;
  /// Numerical symmetry within tolerance.
  bool numerically_symmetric = false;

  /// True when any entry is NaN or infinite -- a corrupt file, caught here
  /// rather than as a mysterious factorization failure twenty minutes later.
  bool has_invalid_values = false;
};

/// Classification of a set of bound pairs -- either the column bounds
/// `l <= x <= u` or the row bounds `l_r <= Ax <= u_r`.
///
/// Two of these counts are consumed by modules downstream rather than merely
/// reported:
///
///   * `free_count` feeds the KKT builder's choice of reduction. A free column
///     contributes neither barrier term, so its entry of `Theta^-1` is exactly
///     zero. In the augmented system that zero is exact and harmless; it only
///     causes trouble when `dx` is eliminated to form the normal equations. A
///     high free share is therefore a reason to take the augmented path,
///     alongside a non-diagonal `Q`. Netlib gas11 is 375 free of 862 columns.
///
///   * `fixed_count` must be ZERO on a canonicalized model. `l == u` makes
///     `x-l > 0` and `u-x > 0` jointly unsatisfiable, so the interior-point
///     method cannot take a first step. A nonzero count here is a violation of
///     the canonicalizer's contract, not a difficult instance.
struct BoundClassification {
  std::size_t total = 0;
  std::size_t free_count = 0;        ///< both bounds infinite
  std::size_t fixed_count = 0;       ///< both finite and equal
  std::size_t boxed_count = 0;       ///< both finite, lower < upper
  std::size_t lower_only_count = 0;  ///< finite lower, infinite upper
  std::size_t upper_only_count = 0;  ///< infinite lower, finite upper
  std::size_t inconsistent_count = 0;  ///< lower > upper: infeasible by itself
};

/// Classify `lower[k] <= . <= upper[k]` for every `k`. The two spans must be
/// the same length; a shorter pair is classified only over the common prefix.
[[nodiscard]] BoundClassification classify_bounds(
    core::HostSpan<const Real> lower, core::HostSpan<const Real> upper) noexcept;

struct AnalysisOptions {
  /// Multiplier k in the `k * sqrt(rows)` dense-column rule.
  double dense_column_factor = 2.0;
  /// Floor, so tiny matrices do not report every column as dense.
  std::size_t dense_column_min_nnz = 32;
  /// Tolerance for the numerical symmetry test.
  Real symmetry_tolerance = 1e-12;
  /// Duplicate detection is O(nnz) but not free; skip it for large instances
  /// when the presolver is not going to use it.
  bool detect_duplicates = true;
};

/// Analyze a matrix held in both orientations.
///
/// Both are required: row statistics come from the CSR view and column
/// statistics from the CSC view, each a linear scan of contiguous memory.
/// Deriving one from the other would mean a scatter.
[[nodiscard]] MatrixAnalysis analyze(const core::SparseMatrixPair<>& matrix,
                                     const AnalysisOptions& options = {});

}  // namespace sovsolve::analysis

#endif  // SOVSOLVE_ANALYSIS_MATRIX_ANALYSIS_HPP
