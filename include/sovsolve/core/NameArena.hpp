// Contiguous storage for row and column names.
//
// A MIPLIB instance can carry millions of names. Holding them as
// `std::vector<std::string>` means one heap allocation per name plus 32 bytes
// of `std::string` overhead each -- for 1 M names that is ~32 MB of overhead
// and 1 M allocations before a single character is stored.
//
// Instead: one character blob plus an offset table. Two allocations total,
// names are contiguous (so iteration is cache-friendly), and lookup returns a
// `string_view` into the blob with no copy.
//
// Names are needed and are not optional: solution reporting, error messages
// that identify the offending row, and MPS round-trip testing all require
// them. The handoff `Problem` type omits them entirely.

#ifndef SOVSOLVE_CORE_NAME_ARENA_HPP
#define SOVSOLVE_CORE_NAME_ARENA_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sovsolve::core {

class NameArena {
 public:
  NameArena() { offsets_.push_back(0); }

  /// Appends a name, returning its index. Indices are assigned in insertion
  /// order, so they match row/column indices when names are added in file
  /// order.
  std::size_t add(std::string_view name) {
    chars_.append(name);
    offsets_.push_back(static_cast<std::uint64_t>(chars_.size()));
    return offsets_.size() - 2;
  }

  [[nodiscard]] std::string_view operator[](std::size_t i) const noexcept {
    const auto begin = offsets_[i];
    const auto end = offsets_[i + 1];
    return std::string_view{chars_.data() + begin,
                            static_cast<std::size_t>(end - begin)};
  }

  [[nodiscard]] std::size_t size() const noexcept { return offsets_.size() - 1; }
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  /// Total characters stored, for diagnostics.
  [[nodiscard]] std::size_t bytes() const noexcept {
    return chars_.size() + offsets_.size() * sizeof(std::uint64_t);
  }

  void reserve(std::size_t count, std::size_t avg_len = 8) {
    offsets_.reserve(count + 1);
    chars_.reserve(count * avg_len);
  }

  void clear() {
    chars_.clear();
    offsets_.clear();
    offsets_.push_back(0);
  }

 private:
  std::string chars_;
  std::vector<std::uint64_t> offsets_;  ///< size() + 1 entries
};

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_NAME_ARENA_HPP
