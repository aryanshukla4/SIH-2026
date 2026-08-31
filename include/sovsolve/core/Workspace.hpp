// Single-allocation scratch arena for the IPM.
//
// The handoff datatype doc names 14 vectors -- x, s, y, z, dx, ds, dy, dz,
// b, c, rp, rd, rxz, rsy -- with no statement about where they live. Allocated
// individually, and with residuals rebuilt per iteration, that puts `malloc`
// inside the IPM hot loop and fragments the heap across hundreds of iterations.
//
// The IPM's memory requirement is fully known once the problem is parsed. So:
// allocate once, carve `Span`s out of it, and allocate nothing per iteration.
//
// Also the natural place to enforce alignment. Every carved span starts on a
// 64-byte boundary, so vectorized loops over any of them can use aligned loads.
//
// Usage:
//     Workspace ws;
//     auto dx = ws.reserve<Real>(n);      // during setup
//     auto dy = ws.reserve<Real>(m);
//     ws.commit();                        // one allocation happens here
//     // dx, dy are now valid spans; nothing allocates again.

#ifndef SOVSOLVE_CORE_WORKSPACE_HPP
#define SOVSOLVE_CORE_WORKSPACE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sovsolve/core/AlignedAllocator.hpp"
#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"

namespace sovsolve::core {

template <typename Alloc = DefaultAllocator>
class BasicWorkspace {
 public:
  static constexpr MemorySpace space = Alloc::space;

  /// Records a request and returns the byte offset it will occupy.
  ///
  /// The returned handle is resolved to a real span by `get()` after
  /// `commit()`. Two-phase, because the total size is not known until every
  /// module has declared what it needs.
  template <typename T>
  [[nodiscard]] std::size_t reserve(std::size_t count) {
    const std::size_t offset = align_up(cursor_);
    cursor_ = offset + count * sizeof(T);
    return offset;
  }

  /// Performs the single allocation. Every previously returned handle becomes
  /// valid; calling `reserve` afterwards invalidates them and is a bug.
  void commit() {
    buffer_.resize(cursor_);
    committed_ = true;
  }

  template <typename T>
  [[nodiscard]] Span<T, space> get(std::size_t handle, std::size_t count) noexcept {
    return Span<T, space>{
        reinterpret_cast<T*>(buffer_.data() + handle), count};
  }

  [[nodiscard]] std::size_t bytes() const noexcept { return cursor_; }
  [[nodiscard]] bool committed() const noexcept { return committed_; }

  void reset() {
    cursor_ = 0;
    committed_ = false;
    buffer_.resize(0);
  }

 private:
  static constexpr std::size_t align_up(std::size_t n) noexcept {
    return ((n + kAlignment - 1) / kAlignment) * kAlignment;
  }

  Vector<std::uint8_t, Alloc> buffer_;
  std::size_t cursor_ = 0;
  bool committed_ = false;
};

using Workspace = BasicWorkspace<DefaultAllocator>;

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_WORKSPACE_HPP
