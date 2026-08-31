// Two-pass assembly of a sparse matrix into sorted CSR + CSC.
//
// ---------------------------------------------------------------------------
// Why two passes rather than triplet (COO) staging
// ---------------------------------------------------------------------------
//
// The obvious approach collects (row, col, value) triples into a growable
// vector and sorts. That costs 16 bytes per nonzero for the staging array, and
// a `std::vector` that doubles peaks at 2x that during reallocation -- so a
// 100 M nonzero instance can spike past 3 GB before the matrix exists.
//
// Counting first gives exact allocation instead: pass one records how many
// entries land in each row and column, a prefix sum turns those counts into
// offsets, and pass two writes each value directly to its final slot. No
// reallocation, no growth spike, and no comparison sort anywhere.
//
// ---------------------------------------------------------------------------
// Why the double transpose
// ---------------------------------------------------------------------------
//
// `finish()` runs staging-CSC -> CSR -> CSC. That looks redundant and is not:
//
//   * A counting-sort transpose visits the source in major order, so entries
//     arrive at each destination slice with **monotonically increasing** minor
//     index. Sorting is a side effect of transposing, not a separate step --
//     which is how invariant I2 gets established for free.
//
//   * Duplicates (which MPS permits) land adjacent in the sorted CSR, so
//     summing them is one linear scan rather than a search.
//
//   * Transposing the compacted CSR back yields a CSC that is sorted and
//     duplicate-free too, and provably describes the same matrix -- the two
//     orientations cannot drift apart because one is computed from the other.
//
// Every step is O(nnz) or O(rows + cols). No sort is ever called.

#ifndef SOVSOLVE_CORE_SPARSE_BUILDER_HPP
#define SOVSOLVE_CORE_SPARSE_BUILDER_HPP

#include <cstddef>
#include <utility>
#include <vector>

#include "sovsolve/core/SparseMatrix.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"

namespace sovsolve::core {

class SparseBuilder {
 public:
  SparseBuilder(std::size_t rows, std::size_t cols)
      : rows_(rows),
        cols_(cols),
        row_counts_(rows + 1, 0),
        col_counts_(cols + 1, 0) {}

  // -- pass 1 -------------------------------------------------------------

  /// Record that an entry exists at (row, col). Values are not needed yet.
  void count(Index row, Index col) noexcept {
    ++row_counts_[static_cast<std::size_t>(row)];
    ++col_counts_[static_cast<std::size_t>(col)];
    ++counted_;
  }

  [[nodiscard]] std::size_t counted() const noexcept { return counted_; }

  /// Prefix-sum the counts and allocate the staging buffers exactly.
  ///
  /// Fails with `IndexOverflow` rather than truncating when the nonzero count
  /// exceeds what `Index` can address -- a silent truncation here would
  /// corrupt the matrix in a way that only surfaces as a wrong answer.
  [[nodiscard]] Status allocate() {
    if (counted_ > static_cast<std::size_t>(kMaxIndex)) {
      return make_error(ErrorCode::IndexOverflow,
                        "nonzero count " + std::to_string(counted_) +
                            " exceeds the 32-bit index limit; rebuild with "
                            "WideIndex");
    }

    stage_offsets_.resize(cols_ + 1);
    std::size_t running = 0;
    for (std::size_t j = 0; j < cols_; ++j) {
      stage_offsets_[j] = static_cast<Index>(running);
      running += col_counts_[j];
    }
    stage_offsets_[cols_] = static_cast<Index>(running);

    stage_rows_.resize(counted_);
    stage_vals_.resize(counted_);

    // Write cursor per column, advanced as entries arrive in pass two.
    cursor_.assign(cols_, 0);
    for (std::size_t j = 0; j < cols_; ++j) {
      cursor_[j] = static_cast<std::size_t>(stage_offsets_[j]);
    }
    allocated_ = true;
    return Status::Ok();
  }

  // -- pass 2 -------------------------------------------------------------

  /// Write an entry to its reserved slot. Must be preceded by a matching
  /// `count()` in pass one, and by `allocate()`.
  void insert(Index row, Index col, Real value) noexcept {
    const std::size_t p = cursor_[static_cast<std::size_t>(col)]++;
    stage_rows_[p] = row;
    stage_vals_[p] = value;
  }

  // -- assembly -----------------------------------------------------------

  /// Transpose to sorted CSR, sum duplicates, transpose back to sorted CSC.
  ///
  /// `sum_duplicates` should stay true: MPS permits a repeated (row, column)
  /// pair and requires the values to be added. Keeping the last value instead
  /// silently changes the model.
  ///
  /// `drop_below` removes entries whose magnitude is below the structural-zero
  /// threshold, after summing -- so a pair that cancels to zero is dropped,
  /// which is the correct reading of the model.
  [[nodiscard]] SparseMatrixPair<> finish(bool sum_duplicates = true,
                                          Real drop_below = 0.0) {
    // ---- staging CSC -> CSR ------------------------------------------------
    // Visiting columns in ascending order means each row receives its entries
    // with increasing column index, so the CSR comes out sorted.
    std::vector<std::size_t> row_cursor(rows_, 0);
    Vector<Index> csr_off(rows_ + 1);
    {
      std::size_t running = 0;
      for (std::size_t i = 0; i < rows_; ++i) {
        csr_off[i] = static_cast<Index>(running);
        row_cursor[i] = running;
        running += row_counts_[i];
      }
      csr_off[rows_] = static_cast<Index>(running);
    }

    Vector<Index> csr_idx(counted_);
    Vector<Real> csr_val(counted_);
    for (std::size_t j = 0; j < cols_; ++j) {
      const std::size_t b = static_cast<std::size_t>(stage_offsets_[j]);
      const std::size_t e = static_cast<std::size_t>(stage_offsets_[j + 1]);
      for (std::size_t k = b; k < e; ++k) {
        const std::size_t i = static_cast<std::size_t>(stage_rows_[k]);
        const std::size_t p = row_cursor[i]++;
        csr_idx[p] = static_cast<Index>(j);
        csr_val[p] = stage_vals_[k];
      }
    }

    // ---- compact: sum duplicates, drop structural zeros -------------------
    // Duplicates are adjacent because the slice is sorted, so this is one
    // linear scan with no search.
    Vector<Index> out_off(rows_ + 1);
    std::vector<Index> keep_idx;
    std::vector<Real> keep_val;
    keep_idx.reserve(counted_);
    keep_val.reserve(counted_);

    for (std::size_t i = 0; i < rows_; ++i) {
      out_off[i] = static_cast<Index>(keep_idx.size());
      const std::size_t b = static_cast<std::size_t>(csr_off[i]);
      const std::size_t e = static_cast<std::size_t>(csr_off[i + 1]);
      std::size_t k = b;
      while (k < e) {
        const Index col = csr_idx[k];
        Real acc = csr_val[k];
        ++k;
        if (sum_duplicates) {
          while (k < e && csr_idx[k] == col) {
            acc += csr_val[k];
            ++k;
          }
        } else {
          while (k < e && csr_idx[k] == col) {
            acc = csr_val[k];  // last wins
            ++k;
          }
        }
        const Real mag = acc < 0 ? -acc : acc;
        if (mag > drop_below) {
          keep_idx.push_back(col);
          keep_val.push_back(acc);
        }
      }
    }
    out_off[rows_] = static_cast<Index>(keep_idx.size());
    const std::size_t final_nnz = keep_idx.size();

    // ---- materialize CSR ---------------------------------------------------
    SparseMatrixPair<> out;
    out.csr = CsrMatrix<>(rows_, cols_, final_nnz);
    {
      auto off = out.csr.offsets();
      auto idx = out.csr.indices();
      auto val = out.csr.values();
      for (std::size_t i = 0; i <= rows_; ++i) off[i] = out_off[i];
      for (std::size_t k = 0; k < final_nnz; ++k) {
        idx[k] = keep_idx[k];
        val[k] = keep_val[k];
      }
    }

    // ---- CSR -> CSC --------------------------------------------------------
    // Recount, because compaction changed the per-column totals. Visiting rows
    // in ascending order leaves each column's row indices sorted.
    out.csc = CscMatrix<>(rows_, cols_, final_nnz);
    {
      std::vector<std::size_t> ccount(cols_, 0);
      for (std::size_t k = 0; k < final_nnz; ++k) {
        ++ccount[static_cast<std::size_t>(keep_idx[k])];
      }
      auto coff = out.csc.offsets();
      std::vector<std::size_t> ccur(cols_, 0);
      std::size_t running = 0;
      for (std::size_t j = 0; j < cols_; ++j) {
        coff[j] = static_cast<Index>(running);
        ccur[j] = running;
        running += ccount[j];
      }
      coff[cols_] = static_cast<Index>(running);

      auto cidx = out.csc.indices();
      auto cval = out.csc.values();
      const auto src_off = out.csr.offsets();
      for (std::size_t i = 0; i < rows_; ++i) {
        const std::size_t b = static_cast<std::size_t>(src_off[i]);
        const std::size_t e = static_cast<std::size_t>(src_off[i + 1]);
        for (std::size_t k = b; k < e; ++k) {
          const std::size_t j = static_cast<std::size_t>(keep_idx[k]);
          const std::size_t p = ccur[j]++;
          cidx[p] = static_cast<Index>(i);
          cval[p] = keep_val[k];
        }
      }
    }

    return out;
  }

  [[nodiscard]] bool allocated() const noexcept { return allocated_; }

 private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t counted_ = 0;
  bool allocated_ = false;

  std::vector<std::size_t> row_counts_;
  std::vector<std::size_t> col_counts_;

  Vector<Index> stage_offsets_;
  Vector<Index> stage_rows_;
  Vector<Real> stage_vals_;
  std::vector<std::size_t> cursor_;
};

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_SPARSE_BUILDER_HPP
