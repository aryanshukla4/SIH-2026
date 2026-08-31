// Flat open-addressing name -> index table.
//
// ---------------------------------------------------------------------------
// Why not std::unordered_map<std::string, int>
// ---------------------------------------------------------------------------
//
// A MIPLIB instance can declare millions of rows and columns, and every one of
// them is looked up at least once per matrix entry. `unordered_map` is the
// wrong shape for that in three separate ways:
//
//   * It is node-based: one heap allocation per entry, and the nodes are
//     scattered, so every probe is a cache miss on a pointer chase.
//   * The `std::string` key copies the name out of the mapped buffer, which
//     for short names means a second allocation each (or at best SSO churn).
//   * Lookup by `string_view` requires the heterogeneous-comparison opt-in,
//     and without it every lookup constructs a temporary `std::string`.
//
// This table stores fixed-size records in one contiguous array and keys them by
// `string_view` pointing **into the memory-mapped file**. No name is ever
// copied, there is one allocation for the whole table, and a probe touches one
// cache line.
//
// Open addressing with linear probing, power-of-two capacity, and a 0.7 load
// factor. Linear probing rather than quadratic because the records are small
// and adjacent probes stay within a cache line.

#ifndef SOVSOLVE_IO_DETAIL_SYMBOL_TABLE_HPP
#define SOVSOLVE_IO_DETAIL_SYMBOL_TABLE_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "sovsolve/core/Types.hpp"

namespace sovsolve::io::detail {

using core::Index;

/// FNV-1a. Chosen over a wide-multiply hash because MPS names are short --
/// 8 characters in fixed format, rarely more than 16 in free format -- so the
/// per-byte loop never runs long enough for a block hash to pay off, and FNV
/// has no setup cost.
[[nodiscard]] inline std::uint64_t hash_name(std::string_view s) noexcept {
  std::uint64_t h = 1469598103934665603ULL;
  for (const char c : s) {
    h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
    h *= 1099511628211ULL;
  }
  // Guarantee a nonzero hash so 0 can mark an empty slot.
  return h == 0 ? 1 : h;
}

class SymbolTable {
 public:
  static constexpr Index kNotFound = -1;

  explicit SymbolTable(std::size_t expected = 1024) { rehash(capacity_for(expected)); }

  /// Insert `name` if absent, returning its index. If present, returns the
  /// existing index and sets `inserted` to false.
  ///
  /// Indices are assigned in insertion order, so they match file order -- which
  /// is what lets the reader use them directly as row/column indices.
  Index insert(std::string_view name, bool& inserted) {
    if (count_ * 10 >= capacity_ * 7) rehash(capacity_ * 2);

    const std::uint64_t h = hash_name(name);
    std::size_t slot = static_cast<std::size_t>(h) & mask_;
    while (slots_[slot].hash != 0) {
      if (slots_[slot].hash == h && slots_[slot].name == name) {
        inserted = false;
        return slots_[slot].index;
      }
      slot = (slot + 1) & mask_;
    }
    const Index index = static_cast<Index>(count_);
    slots_[slot] = Slot{h, name, index};
    ++count_;
    inserted = true;
    return index;
  }

  /// Look up without inserting. Returns `kNotFound` if absent.
  [[nodiscard]] Index find(std::string_view name) const noexcept {
    const std::uint64_t h = hash_name(name);
    std::size_t slot = static_cast<std::size_t>(h) & mask_;
    while (slots_[slot].hash != 0) {
      if (slots_[slot].hash == h && slots_[slot].name == name) {
        return slots_[slot].index;
      }
      slot = (slot + 1) & mask_;
    }
    return kNotFound;
  }

  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

  void reserve(std::size_t expected) {
    const std::size_t want = capacity_for(expected);
    if (want > capacity_) rehash(want);
  }

 private:
  struct Slot {
    std::uint64_t hash = 0;  ///< 0 means empty
    std::string_view name;   ///< points into the mapped file; never owned
    Index index = kNotFound;
  };

  static std::size_t capacity_for(std::size_t expected) noexcept {
    // Power of two at or above expected / 0.7, minimum 16.
    std::size_t want = (expected * 10) / 7 + 1;
    std::size_t cap = 16;
    while (cap < want) cap <<= 1;
    return cap;
  }

  void rehash(std::size_t new_capacity) {
    std::vector<Slot> fresh(new_capacity);
    const std::size_t new_mask = new_capacity - 1;
    for (const auto& s : slots_) {
      if (s.hash == 0) continue;
      std::size_t slot = static_cast<std::size_t>(s.hash) & new_mask;
      while (fresh[slot].hash != 0) slot = (slot + 1) & new_mask;
      fresh[slot] = s;
    }
    slots_.swap(fresh);
    capacity_ = new_capacity;
    mask_ = new_mask;
  }

  std::vector<Slot> slots_;
  std::size_t capacity_ = 0;
  std::size_t mask_ = 0;
  std::size_t count_ = 0;
};

}  // namespace sovsolve::io::detail

#endif  // SOVSOLVE_IO_DETAIL_SYMBOL_TABLE_HPP
