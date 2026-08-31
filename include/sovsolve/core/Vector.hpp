// Owning, aligned, 1-D numerical buffer.
//
// Differences from the handoff datatype doc, and why:
//
//   * `precision` is a template parameter, not a runtime field. A runtime
//     precision field puts a branch inside every arithmetic operation on a
//     type whose whole purpose is to be traversed in tight loops.
//
//   * `memory_location` is carried by the allocator policy, resolved at compile
//     time, rather than by a runtime enum. Same reason.
//
//   * Copying is explicit. The doc says nothing about copy semantics, which in
//     practice means an accidental deep copy of a million doubles inside an IPM
//     iteration -- invisible in the source, ruinous in the profile. The copy
//     constructor is deleted; duplication goes through `clone()`.
//
//   * 64-byte alignment, from `AlignedAllocator`.

#ifndef SOVSOLVE_CORE_VECTOR_HPP
#define SOVSOLVE_CORE_VECTOR_HPP

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <utility>

#include "sovsolve/core/AlignedAllocator.hpp"
#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"

namespace sovsolve::core {

template <typename T, typename Alloc = DefaultAllocator>
class Vector {
 public:
  using value_type = T;
  using allocator_type = Alloc;
  static constexpr MemorySpace space = Alloc::space;

  Vector() = default;

  /// Uninitialized allocation. Values are garbage; callers that need zeros ask
  /// for them explicitly. Zeroing a large buffer that is about to be
  /// overwritten is a real cost at load time.
  explicit Vector(std::size_t size)
      : data_(static_cast<T*>(Alloc::allocate(size * sizeof(T)))), size_(size) {}

  Vector(std::size_t size, T fill) : Vector(size) { assign(fill); }

  ~Vector() { Alloc::deallocate(data_); }

  // Move-only. See the header comment.
  Vector(const Vector&) = delete;
  Vector& operator=(const Vector&) = delete;

  Vector(Vector&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}

  Vector& operator=(Vector&& other) noexcept {
    if (this != &other) {
      Alloc::deallocate(data_);
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  /// Explicit deep copy. Host allocators only -- a device copy needs a stream
  /// and belongs in the CUDA backend, not here.
  [[nodiscard]] Vector clone() const {
    static_assert(space == MemorySpace::Host,
                  "clone() is host-only; device copies need a stream");
    Vector out(size_);
    std::memcpy(out.data_, data_, size_ * sizeof(T));
    return out;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  [[nodiscard]] Span<T, space> span() noexcept { return {data_, size_}; }
  [[nodiscard]] Span<const T, space> span() const noexcept { return {data_, size_}; }

  [[nodiscard]] T& operator[](std::size_t i) noexcept {
    static_assert(space == MemorySpace::Host,
                  "cannot dereference device memory from host code");
    return data_[i];
  }
  [[nodiscard]] const T& operator[](std::size_t i) const noexcept {
    static_assert(space == MemorySpace::Host,
                  "cannot dereference device memory from host code");
    return data_[i];
  }

  [[nodiscard]] T* begin() noexcept { return data_; }
  [[nodiscard]] T* end() noexcept { return data_ + size_; }
  [[nodiscard]] const T* begin() const noexcept { return data_; }
  [[nodiscard]] const T* end() const noexcept { return data_ + size_; }

  void assign(T value) {
    static_assert(space == MemorySpace::Host, "host-only");
    std::fill(data_, data_ + size_, value);
  }

  /// Discards contents. Reallocates only when the size actually changes --
  /// re-solving a sequence of related problems should not churn the heap.
  void resize(std::size_t new_size) {
    if (new_size == size_) return;
    Alloc::deallocate(data_);
    data_ = static_cast<T*>(Alloc::allocate(new_size * sizeof(T)));
    size_ = new_size;
  }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

using RealVector = Vector<Real>;
using IndexVector = Vector<Index>;

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_VECTOR_HPP
