// Non-owning view over contiguous memory.
//
// The handoff datatype doc gives `Vector` no view type, so every function must
// take `Vector&`. That makes it impossible to pass a sub-range, a slice of a
// workspace arena, or a raw device pointer without copying — and copying a
// million doubles inside an IPM iteration is not an acceptable cost.
//
// `Span` is the currency of module boundaries: modules take spans, only
// owners hold `Vector`. It is also the zero-copy handoff point to CUDA, since
// `data()` and `size()` are exactly what a `cudaMemcpy` or a kernel launch
// needs.
//
// This is `std::span` with a `MemorySpace` tag attached. The tag is what stops
// a host loop from silently dereferencing a device pointer: crossing spaces has
// to be written down, because the compiler cannot see the difference.

#ifndef SOVSOLVE_CORE_SPAN_HPP
#define SOVSOLVE_CORE_SPAN_HPP

#include <cassert>
#include <cstddef>

#include "sovsolve/core/Types.hpp"

namespace sovsolve::core {

template <typename T, MemorySpace Space = MemorySpace::Host>
class Span {
 public:
  using value_type = T;
  static constexpr MemorySpace space = Space;

  constexpr Span() noexcept = default;
  constexpr Span(T* data, std::size_t size) noexcept : data_(data), size_(size) {}

  [[nodiscard]] constexpr T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

  [[nodiscard]] constexpr std::size_t size_bytes() const noexcept {
    return size_ * sizeof(T);
  }

  /// Element access. Host spans only — a device pointer is not dereferenceable
  /// from host code, and the static_assert says so at compile time rather than
  /// letting it fault at run time.
  [[nodiscard]] constexpr T& operator[](std::size_t i) const noexcept {
    static_assert(Space == MemorySpace::Host,
                  "cannot dereference a device span from host code");
    assert(i < size_);
    return data_[i];
  }

  [[nodiscard]] constexpr T* begin() const noexcept {
    static_assert(Space == MemorySpace::Host,
                  "cannot iterate a device span from host code");
    return data_;
  }
  [[nodiscard]] constexpr T* end() const noexcept {
    static_assert(Space == MemorySpace::Host,
                  "cannot iterate a device span from host code");
    return data_ + size_;
  }

  /// Sub-range. This is how a workspace arena is carved into named vectors
  /// without allocating.
  [[nodiscard]] constexpr Span subspan(std::size_t offset,
                                       std::size_t count) const noexcept {
    assert(offset + count <= size_);
    return Span{data_ + offset, count};
  }

  /// Implicit const conversion, so a mutable span satisfies a const parameter.
  constexpr operator Span<const T, Space>() const noexcept {  // NOLINT(*-explicit-*)
    return Span<const T, Space>{data_, size_};
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

template <typename T>
using HostSpan = Span<T, MemorySpace::Host>;

template <typename T>
using DeviceSpan = Span<T, MemorySpace::Device>;

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_SPAN_HPP
