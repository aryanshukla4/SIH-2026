// Locale-independent number parsing for the model readers.
//
// `std::stod` and `std::istringstream` are both disqualified here:
//
//   * They consult the global locale, so a machine set to a comma-decimal
//     locale silently parses "1.5" as 1. That is a wrong answer, not a crash.
//   * They are 10-50x slower than `std::from_chars`. On a 500 MB MPS file with
//     ~50 M numeric fields, that difference is the whole parse time.
//   * `stod` requires a null-terminated string, so using it on a memory-mapped
//     buffer means copying every field into a temporary.
//
// `std::from_chars` has none of those problems: locale-independent by
// specification, operates on a (begin, end) range straight out of the mapped
// buffer, and does not allocate.

#ifndef SOVSOLVE_IO_DETAIL_NUMBER_PARSE_HPP
#define SOVSOLVE_IO_DETAIL_NUMBER_PARSE_HPP

#include <charconv>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

#include "sovsolve/core/Types.hpp"

namespace sovsolve::io::detail {

using core::Real;

/// Parse a floating-point field.
///
/// Returns `nullopt` on malformed input or trailing garbage; the caller turns
/// that into a located `ParseError`.
///
/// Handles Fortran `D` exponents (`1.5D+02`), which appear throughout the
/// older Netlib instances -- those files were produced by Fortran writers and
/// `from_chars` rejects the `D` form outright.
[[nodiscard]] inline std::optional<Real> parse_real(std::string_view text) noexcept {
  if (text.empty()) return std::nullopt;

  const char* first = text.data();
  const char* last = text.data() + text.size();

  Real value{};
  auto [ptr, ec] = std::from_chars(first, last, value);

  if (ec == std::errc{} && ptr == last) return value;

  // A `D` exponent stops from_chars at the 'D'. Rewrite that one character and
  // retry on a small stack buffer -- rare enough that the copy costs nothing,
  // and the buffer is bounded because a numeric field cannot be long.
  if (ptr != last && (*ptr == 'D' || *ptr == 'd')) {
    constexpr std::size_t kMaxNumericField = 64;
    if (text.size() < kMaxNumericField) {
      char buf[kMaxNumericField];
      for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        buf[i] = (c == 'D' || c == 'd') ? 'E' : c;
      }
      Real retry{};
      auto [p2, ec2] = std::from_chars(buf, buf + text.size(), retry);
      if (ec2 == std::errc{} && p2 == buf + text.size()) return retry;
    }
  }

  return std::nullopt;
}

/// Parse a non-negative integer field.
[[nodiscard]] inline std::optional<long long> parse_int(
    std::string_view text) noexcept {
  if (text.empty()) return std::nullopt;
  long long value{};
  auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || ptr != text.data() + text.size()) return std::nullopt;
  return value;
}

/// Case-insensitive comparison against an ASCII keyword.
///
/// Section headers and bound keys are conventionally uppercase but not
/// reliably so, and `std::tolower` is locale-sensitive (in Turkish locales
/// 'I' does not lowercase to 'i'), which is exactly the class of bug this
/// header exists to avoid.
[[nodiscard]] inline bool iequals(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    char ca = a[i];
    char cb = b[i];
    if (ca >= 'a' && ca <= 'z') ca = static_cast<char>(ca - 32);
    if (cb >= 'a' && cb <= 'z') cb = static_cast<char>(cb - 32);
    if (ca != cb) return false;
  }
  return true;
}

}  // namespace sovsolve::io::detail

#endif  // SOVSOLVE_IO_DETAIL_NUMBER_PARSE_HPP
