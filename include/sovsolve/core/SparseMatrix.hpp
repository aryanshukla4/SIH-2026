// Compressed sparse matrix storage.
//
// Differences from the handoff datatype doc, and why:
//
//   * CSR *and* CSC. The doc lists CSR only. Every IPM iteration computes both
//     `A*x` (row-oriented) and `A'*y` (column-oriented). Computing `A'*y` from
//     CSR alone requires either atomics on GPU or a cache-hostile scatter on
//     CPU. Holding both costs 2x index+value memory and is what every serious
//     solver does. `SparseMatrixPair` below holds the two together and keeps
//     them consistent.
//
//   * The sorted-index invariant is stated and checked. cuSPARSE *requires*
//     sorted indices within each row for many routines; unsorted input either
//     silently produces wrong results or drops to a slow path. An invariant
//     nobody wrote down is an invariant nobody maintains.
//
//   * Explicit index type. `Index` (int32) by default -- see Types.hpp.
//
//   * No runtime `format` or `memory_location` field. Storage orientation is a
//     type, not a branch taken inside every kernel.

#ifndef SOVSOLVE_CORE_SPARSE_MATRIX_HPP
#define SOVSOLVE_CORE_SPARSE_MATRIX_HPP

#include <cstddef>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"

namespace sovsolve::core {

/// Compressed sparse matrix in CSR or CSC orientation.
///
/// Invariants -- all four are checked by `validate()` and must hold whenever a
/// matrix crosses a module boundary:
///
///   I1. `offsets.size() == major_dim + 1`, `offsets[0] == 0`,
///       `offsets[major_dim] == nnz`, and `offsets` is non-decreasing.
///   I2. Within each major slice, `indices` is **strictly increasing**. Sorted,
///       and therefore also duplicate-free.
///   I3. Every index is in `[0, minor_dim)`.
///   I4. `values.size() == indices.size() == nnz`.
///
/// I2 is the one that gets violated in practice. Duplicates are summed at
/// construction (MPS permits repeated entries), never kept.
///
/// For `Format == CSR`: major = rows, minor = columns.
/// For `Format == CSC`: major = columns, minor = rows.
template <SparseFormat Format, typename T = Real, typename Idx = Index,
          typename Alloc = DefaultAllocator>
class SparseMatrix {
 public:
  using value_type = T;
  using index_type = Idx;
  static constexpr SparseFormat format = Format;
  static constexpr MemorySpace space = Alloc::space;

  SparseMatrix() = default;

  SparseMatrix(std::size_t rows, std::size_t cols, std::size_t nnz)
      : rows_(rows),
        cols_(cols),
        nnz_(nnz),
        offsets_(major_dim_of(rows, cols) + 1),
        indices_(nnz),
        values_(nnz) {}

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }

  /// Cached rather than read from `offsets[major]` -- it is wanted in hot loops
  /// and in every diagnostic line.
  [[nodiscard]] std::size_t nnz() const noexcept { return nnz_; }

  [[nodiscard]] std::size_t major_dim() const noexcept {
    return major_dim_of(rows_, cols_);
  }
  [[nodiscard]] std::size_t minor_dim() const noexcept {
    return Format == SparseFormat::CSR ? cols_ : rows_;
  }

  [[nodiscard]] double density() const noexcept {
    const double denom = static_cast<double>(rows_) * static_cast<double>(cols_);
    return denom > 0.0 ? static_cast<double>(nnz_) / denom : 0.0;
  }

  [[nodiscard]] Span<Idx, space> offsets() noexcept { return offsets_.span(); }
  [[nodiscard]] Span<const Idx, space> offsets() const noexcept { return offsets_.span(); }

  [[nodiscard]] Span<Idx, space> indices() noexcept { return indices_.span(); }
  [[nodiscard]] Span<const Idx, space> indices() const noexcept { return indices_.span(); }

  [[nodiscard]] Span<T, space> values() noexcept { return values_.span(); }
  [[nodiscard]] Span<const T, space> values() const noexcept { return values_.span(); }

  /// Half-open range of entry positions in major slice `i`.
  [[nodiscard]] std::size_t slice_begin(std::size_t i) const noexcept {
    return static_cast<std::size_t>(offsets_[i]);
  }
  [[nodiscard]] std::size_t slice_end(std::size_t i) const noexcept {
    return static_cast<std::size_t>(offsets_[i + 1]);
  }
  [[nodiscard]] std::size_t slice_nnz(std::size_t i) const noexcept {
    return slice_end(i) - slice_begin(i);
  }

  /// Verifies I1-I4. Debug builds and module boundaries; not a hot path.
  ///
  /// Defined inline rather than out-of-line so no explicit instantiation list
  /// has to be maintained as new (Format, T, Idx) combinations appear.
  [[nodiscard]] bool validate() const noexcept {
    static_assert(space == MemorySpace::Host, "validate() is host-only");

    const std::size_t major = major_dim();
    const std::size_t minor = minor_dim();

    // I1: offsets shape, endpoints, monotonicity.
    if (offsets_.size() != major + 1) return false;
    if (offsets_.size() > 0 && offsets_[0] != Idx{0}) return false;
    if (major > 0 && static_cast<std::size_t>(offsets_[major]) != nnz_) return false;
    for (std::size_t i = 0; i < major; ++i) {
      if (offsets_[i] > offsets_[i + 1]) return false;
    }

    // I4: parallel array lengths.
    if (indices_.size() != nnz_ || values_.size() != nnz_) return false;

    for (std::size_t i = 0; i < major; ++i) {
      const std::size_t b = slice_begin(i);
      const std::size_t e = slice_end(i);
      for (std::size_t k = b; k < e; ++k) {
        // I3: index in range.
        if (indices_[k] < Idx{0}) return false;
        if (static_cast<std::size_t>(indices_[k]) >= minor) return false;
        // I2: strictly increasing within the slice -- sorted, so also
        // duplicate-free. This is the invariant that actually gets violated,
        // and the one cuSPARSE depends on.
        if (k > b && indices_[k] <= indices_[k - 1]) return false;
      }
    }
    return true;
  }

 private:
  static constexpr std::size_t major_dim_of(std::size_t rows, std::size_t cols) noexcept {
    return Format == SparseFormat::CSR ? rows : cols;
  }

  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::size_t nnz_ = 0;

  Vector<Idx, Alloc> offsets_;
  Vector<Idx, Alloc> indices_;
  Vector<T, Alloc> values_;
};

template <typename T = Real, typename Idx = Index, typename Alloc = DefaultAllocator>
using CsrMatrix = SparseMatrix<SparseFormat::CSR, T, Idx, Alloc>;

template <typename T = Real, typename Idx = Index, typename Alloc = DefaultAllocator>
using CscMatrix = SparseMatrix<SparseFormat::CSC, T, Idx, Alloc>;

/// A matrix held in both orientations.
///
/// This is what `Problem` stores for `A`. The two views describe the same
/// matrix and are built together, so they cannot drift apart: the CSC is
/// produced directly by the MPS reader (the `COLUMNS` section is
/// column-ordered), and the CSR is a single counting-sort transpose of it,
/// which also establishes the sorted-index invariant I2 for free.
template <typename T = Real, typename Idx = Index, typename Alloc = DefaultAllocator>
struct SparseMatrixPair {
  CsrMatrix<T, Idx, Alloc> csr;  ///< for A*x
  CscMatrix<T, Idx, Alloc> csc;  ///< for A'*y

  [[nodiscard]] std::size_t rows() const noexcept { return csr.rows(); }
  [[nodiscard]] std::size_t cols() const noexcept { return csr.cols(); }
  [[nodiscard]] std::size_t nnz() const noexcept { return csr.nnz(); }
  [[nodiscard]] bool empty() const noexcept { return csr.nnz() == 0; }
};

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_SPARSE_MATRIX_HPP
