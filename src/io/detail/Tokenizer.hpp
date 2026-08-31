// Line and field scanning over a memory-mapped buffer.
//
// No iostreams and no allocation anywhere in here: every line and every field
// is a `string_view` into the mapped file. That is what lets the symbol table
// key on names without copying them.

#ifndef SOVSOLVE_IO_DETAIL_TOKENIZER_HPP
#define SOVSOLVE_IO_DETAIL_TOKENIZER_HPP

#include <array>
#include <cstddef>
#include <string_view>

namespace sovsolve::io::detail {

/// MPS data lines carry at most six fields.
inline constexpr std::size_t kMaxFields = 6;

using FieldArray = std::array<std::string_view, kMaxFields>;

[[nodiscard]] inline bool is_space(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\r';
}

/// Iterates lines, tracking 1-based line numbers for error reporting.
///
/// Handles LF and CRLF. Benchmark archives contain both, often within the same
/// download.
class LineScanner {
 public:
  explicit LineScanner(std::string_view text) noexcept : text_(text) {}

  /// Advance to the next line. Returns false at end of input.
  bool next(std::string_view& line, std::size_t& line_number) noexcept {
    if (pos_ >= text_.size()) return false;
    const std::size_t start = pos_;
    std::size_t end = start;
    while (end < text_.size() && text_[end] != '\n') ++end;

    std::size_t trimmed = end;
    while (trimmed > start && text_[trimmed - 1] == '\r') --trimmed;

    line = text_.substr(start, trimmed - start);
    pos_ = end < text_.size() ? end + 1 : text_.size();
    line_number = ++line_no_;
    return true;
  }

  void rewind() noexcept {
    pos_ = 0;
    line_no_ = 0;
  }

 private:
  std::string_view text_;
  std::size_t pos_ = 0;
  std::size_t line_no_ = 0;
};

/// True for lines carrying no data: blank, or a `*` comment in column 1.
[[nodiscard]] inline bool is_skippable(std::string_view line) noexcept {
  if (line.empty()) return true;
  if (line[0] == '*') return true;
  for (const char c : line) {
    if (!is_space(c)) return false;
  }
  return true;
}

/// True when the line starts a section: content begins in column 1.
///
/// This is the one piece of column significance that free-format MPS keeps --
/// data lines are always indented, section headers never are.
[[nodiscard]] inline bool is_section_header(std::string_view line) noexcept {
  return !line.empty() && !is_space(line[0]) && line[0] != '*';
}

/// Split on whitespace. Returns the number of fields found, capped at
/// `kMaxFields`.
[[nodiscard]] inline std::size_t split_free(std::string_view line,
                                            FieldArray& out) noexcept {
  std::size_t count = 0;
  std::size_t i = 0;
  const std::size_t n = line.size();
  while (i < n && count < kMaxFields) {
    while (i < n && is_space(line[i])) ++i;
    if (i >= n) break;
    const std::size_t start = i;
    while (i < n && !is_space(line[i])) ++i;
    out[count++] = line.substr(start, i - start);
  }
  return count;
}

/// Count whitespace-separated fields without storing them. Used by format
/// detection, which only needs the count.
[[nodiscard]] inline std::size_t count_free_fields(std::string_view line) noexcept {
  std::size_t count = 0;
  std::size_t i = 0;
  const std::size_t n = line.size();
  while (i < n) {
    while (i < n && is_space(line[i])) ++i;
    if (i >= n) break;
    ++count;
    while (i < n && !is_space(line[i])) ++i;
  }
  return count;
}

/// Fixed-format field windows, 0-indexed half-open, from the MPS column spec
/// (fields at columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61).
inline constexpr std::array<std::pair<std::size_t, std::size_t>, kMaxFields>
    kFixedWindows{{{1, 3}, {4, 12}, {14, 22}, {24, 36}, {39, 47}, {49, 61}}};

[[nodiscard]] inline std::string_view trim(std::string_view s) noexcept {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && is_space(s[b])) ++b;
  while (e > b && is_space(s[e - 1])) --e;
  return s.substr(b, e - b);
}

/// Extract fields by column position.
///
/// Only fixed format can represent a name containing a space, which is the
/// sole reason this path exists -- see `MPS-FORMAT-NOTES.md` §1.
[[nodiscard]] inline std::size_t split_fixed(std::string_view line,
                                             FieldArray& out) noexcept {
  std::size_t count = 0;
  for (std::size_t f = 0; f < kMaxFields; ++f) {
    const auto [b, e] = kFixedWindows[f];
    if (b >= line.size()) break;
    const std::size_t end = e < line.size() ? e : line.size();
    const std::string_view field = trim(line.substr(b, end - b));
    if (!field.empty()) {
      out[f] = field;
      count = f + 1;
    } else {
      out[f] = {};
    }
  }
  return count;
}

/// Strip surrounding single quotes, as used by `'MARKER'` and `'INTORG'`.
[[nodiscard]] inline std::string_view unquote(std::string_view s) noexcept {
  if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

}  // namespace sovsolve::io::detail

#endif  // SOVSOLVE_IO_DETAIL_TOKENIZER_HPP
