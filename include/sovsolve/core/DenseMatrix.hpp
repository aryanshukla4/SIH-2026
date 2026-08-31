// Dense matrix storage.
//
// **Column-major.** The handoff datatype doc does not state a layout, and the
// choice is not free: cuBLAS, LAPACK and every BLAS implementation are
// column-major. A row-major dense matrix means transposing on every call into
// those libraries, which for the reduced Newton system is a per-iteration cost.
//
// Element (i, j) lives at `data[j * ld + i]`.

#ifndef SOVSOLVE_CORE_DENSE_MATRIX_HPP
#define SOVSOLVE_CORE_DENSE_MATRIX_HPP

#include <cassert>
#include <cstddef>

#include "sovsolve/core/AlignedAllocator.hpp"
#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"

namespace sovsolve::core {

template <typename T = Real, typename Alloc = DefaultAllocator>
class DenseMatrix {
 public:
  using value_type = T;
  static constexpr MemorySpace space = Alloc::space;

  DenseMatrix() = default;

  /// `ld` is the leading dimension (column stride). It defaults to `rows` but
  /// is padded up to keep each column 64-byte aligned, which is what lets a
  /// vectorized column traversal use aligned loads.
  DenseMatrix(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), ld_(pad_to_alignment(rows)), data_(ld_ * cols) {}

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] std::size_t ld() const noexcept { return ld_; }

  [[nodiscard]] T* data() noexcept { return data_.data(); }
  [[nodiscard]] const T* data() const noexcept { return data_.data(); }

  [[nodiscard]] Span<T, space> span() noexcept { return data_.span(); }
  [[nodiscard]] Span<const T, space> span() const noexcept { return data_.span(); }

  [[nodiscard]] T& operator()(std::size_t i, std::size_t j) noexcept {
    static_assert(space == MemorySpace::Host, "host-only");
    assert(i < rows_ && j < cols_);
    return data_[j * ld_ + i];
  }
  [[nodiscard]] const T& operator()(std::size_t i, std::size_t j) const noexcept {
    static_assert(space == MemorySpace::Host, "host-only");
    assert(i < rows_ && j < cols_);
    return data_[j * ld_ + i];
  }

  /// Column `j` as a contiguous span -- columns are contiguous under
  /// column-major storage, rows are not.
  [[nodiscard]] Span<T, space> column(std::size_t j) noexcept {
    return data_.span().subspan(j * ld_, rows_);
  }
  [[nodiscard]] Span<const T, space> column(std::size_t j) const noexcept {
    return data_.span().subspan(j * ld_, rows_);
  }

 private:
  static std::size_t pad_to_alignment(std::size_t rows) noexcept {
    constexpr std::size_t elems = kAlignment / sizeof(T);
    if constexpr (elems <= 1) return rows;
    return ((rows + elems - 1) / elems) * elems;
  }

  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::size_t ld_ = 0;
  Vector<T, Alloc> data_;
};

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_DENSE_MATRIX_HPP
