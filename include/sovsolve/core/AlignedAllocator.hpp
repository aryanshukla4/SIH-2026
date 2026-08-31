// Allocation policies for solver buffers.
//
// The handoff datatype doc specifies neither alignment nor an allocator, which
// closes off two things this layer needs:
//
//   * Alignment. 64 bytes is one cache line, satisfies AVX2 (32 B) and
//     AVX-512 (64 B) load requirements, and is a divisor of the 128 B GPU
//     coalescing granule.
//
//   * A swappable allocator. Pinned (page-locked) host memory is roughly 2-3x
//     faster on host<->device transfer than pageable memory. That is exactly
//     the cost architecture_corrected.txt section 9 says to avoid, and it can
//     only be obtained if the allocator is a policy rather than a hard-coded
//     `new`. `PinnedAllocator` and `DeviceAllocator` are declared here and
//     defined when CUDA is present; nothing else in this layer changes.

#ifndef SOVSOLVE_CORE_ALIGNED_ALLOCATOR_HPP
#define SOVSOLVE_CORE_ALIGNED_ALLOCATOR_HPP

#include <cstddef>
#include <cstdlib>
#include <new>

// Windows has no C11 aligned_alloc, in either the MSVC CRT or MinGW's UCRT
// runtime. Both provide _aligned_malloc/_aligned_free from <malloc.h>, and
// those must be paired -- memory from _aligned_malloc cannot be freed with
// std::free. Hence the platform split rather than a compiler split.
#if defined(_WIN32)
#include <malloc.h>
#endif

#include "sovsolve/core/Types.hpp"

namespace sovsolve::core {

/// Alignment for all solver buffers: one cache line.
inline constexpr std::size_t kAlignment = 64;

/// Aligned host allocation. Pageable memory.
struct HostAllocator {
  static constexpr MemorySpace space = MemorySpace::Host;

  [[nodiscard]] static void* allocate(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    // Round up: aligned_alloc requires size to be a multiple of alignment.
    const std::size_t padded = ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
#if defined(_WIN32)
    void* p = _aligned_malloc(padded, kAlignment);
#else
    void* p = std::aligned_alloc(kAlignment, padded);
#endif
    if (p == nullptr) throw std::bad_alloc();
    return p;
  }

  static void deallocate(void* p) noexcept {
    if (p == nullptr) return;
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
};

#ifdef SOVSOLVE_WITH_CUDA
/// Page-locked host memory (`cudaHostAlloc`). Same address space as
/// `HostAllocator`, but 2-3x faster to transfer and safe for async copies.
struct PinnedAllocator {
  static constexpr MemorySpace space = MemorySpace::Host;
  [[nodiscard]] static void* allocate(std::size_t bytes);
  static void deallocate(void* p) noexcept;
};

/// Device memory (`cudaMalloc`). Not host-dereferenceable.
struct DeviceAllocator {
  static constexpr MemorySpace space = MemorySpace::Device;
  [[nodiscard]] static void* allocate(std::size_t bytes);
  static void deallocate(void* p) noexcept;
};
#endif  // SOVSOLVE_WITH_CUDA

/// The allocator used unless a type explicitly asks for another.
using DefaultAllocator = HostAllocator;

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_ALIGNED_ALLOCATOR_HPP
