// MPS writer. See Write.hpp for why this exists and what does not round-trip.

#include "sovsolve/io/Write.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <string>
#include <system_error>
#include <vector>

#include "sovsolve/core/Types.hpp"

namespace sovsolve::io {
namespace {

using core::ErrorCode;
using core::Index;
using core::INF;
using core::is_finite_bound;
using core::Real;
using model::VarType;

/// Shortest decimal that reads back bit-identical.
///
/// `to_chars` without a precision argument is specified to produce exactly
/// that. `%.17g` is longer and, depending on the C library, not guaranteed to
/// round-trip -- which would defeat the point of the writer, since a dumped
/// model has to reproduce the failure it was dumped for.
std::string number(Real v) {
  std::array<char, 40> buf{};
  const auto res = std::to_chars(buf.data(), buf.data() + buf.size(), v);
  if (res.ec != std::errc{}) return "0";
  return std::string(buf.data(), res.ptr);
}

/// A name safe to write and read back.
///
/// Free-format MPS separates fields on whitespace, so a name containing a space
/// would silently split into two fields and shift every field after it. Names
/// come from the input file and are normally clean, but a programmatically
/// built model can carry anything, so this is checked rather than assumed.
bool name_is_safe(std::string_view s) noexcept {
  if (s.empty()) return false;
  for (const char c : s) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return false;
  }
  return true;
}

class MpsWriter {
 public:
  MpsWriter(const model::Problem& p, const WriteOptions& o)
      : p_(p), o_(o) {}

  core::Expected<std::string> run();

 private:
  void line(std::string_view s) {
    out_ += s;
    out_ += '\n';
  }

  /// One indented data line, fields padded to `name_width` so the output reads
  /// like fixed format without depending on it.
  void field_line(std::initializer_list<std::string> fields) {
    out_ += "    ";
    std::size_t i = 0;
    for (const auto& f : fields) {
      out_ += f;
      ++i;
      if (i < fields.size()) {
        const auto w = static_cast<std::size_t>(o_.name_width);
        for (std::size_t pad = f.size(); pad < w; ++pad) out_ += ' ';
        out_ += ' ';
      }
    }
    out_ += '\n';
  }

  [[nodiscard]] std::string row_name(std::size_t i) const {
    if (i < p_.row_names.size()) return std::string(p_.row_names[i]);
    return "R" + std::to_string(i + 1);
  }
  [[nodiscard]] std::string col_name(std::size_t j) const {
    if (j < p_.col_names.size()) return std::string(p_.col_names[j]);
    return "C" + std::to_string(j + 1);
  }
  [[nodiscard]] std::string obj_name() const {
    return p_.objective_row_name.empty() ? std::string("COST")
                                         : p_.objective_row_name;
  }

  core::Status check_names() const;
  void write_rows();
  void write_columns();
  void write_rhs();
  void write_ranges();
  void write_bounds();
  void write_quadobj();

  const model::Problem& p_;
  const WriteOptions& o_;
  std::string out_;
};

core::Status MpsWriter::check_names() const {
  if (!name_is_safe(obj_name())) {
    return core::make_error(ErrorCode::ParseError,
                            "objective row name contains whitespace");
  }
  for (std::size_t i = 0; i < p_.num_rows(); ++i) {
    if (!name_is_safe(row_name(i))) {
      return core::make_error(
          ErrorCode::ParseError,
          "row " + std::to_string(i) + " has a name containing whitespace");
    }
  }
  for (std::size_t j = 0; j < p_.num_cols(); ++j) {
    if (!name_is_safe(col_name(j))) {
      return core::make_error(
          ErrorCode::ParseError,
          "column " + std::to_string(j) + " has a name containing whitespace");
    }
  }
  return core::Status{};
}

void MpsWriter::write_rows() {
  line("ROWS");
  field_line({"N", obj_name()});
  for (std::size_t i = 0; i < p_.num_rows(); ++i) {
    const Real lo = p_.row_lower[i];
    const Real hi = p_.row_upper[i];
    const bool lf = is_finite_bound(lo);
    const bool hf = is_finite_bound(hi);

    // The inverse of the reader's sense table. A ranged row is written as `L`
    // with a RANGES entry, which reproduces [hi - |R|, hi] = [lo, hi]; `G` with
    // a range would do equally well, and the choice only has to be consistent
    // with write_rhs and write_ranges.
    const char* sense = nullptr;
    if (lf && hf && lo == hi) {
      sense = "E";
    } else if (hf) {
      sense = "L";
    } else if (lf) {
      sense = "G";
    } else {
      // Both bounds infinite: a free row. `N` is the MPS spelling, and a reader
      // drops N rows after the objective -- see the header comment.
      sense = "N";
    }
    field_line({sense, row_name(i)});
  }
}

void MpsWriter::write_columns() {
  line("COLUMNS");
  const auto off = p_.A.csc.offsets();
  const auto idx = p_.A.csc.indices();
  const auto val = p_.A.csc.values();

  // COLUMNS is column-major, which is exactly the CSC orientation -- so this
  // walks contiguous memory rather than gathering.
  bool in_integer_block = false;
  int marker_id = 0;

  for (std::size_t j = 0; j < p_.num_cols(); ++j) {
    const bool discrete = j < p_.col_type.size() &&
                          p_.col_type[j] != VarType::Continuous;
    if (discrete != in_integer_block) {
      const std::string tag = "MARKER" + std::to_string(marker_id++);
      field_line({tag, "'MARKER'", discrete ? "'INTORG'" : "'INTEND'"});
      in_integer_block = discrete;
    }

    const std::string cname = col_name(j);
    if (j < p_.c.size() && p_.c[j] != 0.0) {
      field_line({cname, obj_name(), number(p_.c[j])});
    }
    for (auto k = static_cast<std::size_t>(off[j]);
         k < static_cast<std::size_t>(off[j + 1]); ++k) {
      const auto i = static_cast<std::size_t>(idx[k]);
      field_line({cname, row_name(i), number(val[k])});
    }
  }
  if (in_integer_block) {
    const std::string tag = "MARKER" + std::to_string(marker_id);
    field_line({tag, "'MARKER'", "'INTEND'"});
  }
}

void MpsWriter::write_rhs() {
  // An objective-row RHS entry is the NEGATED objective constant. This is the
  // format's sharpest trap and the reader applies the same negation, so writer
  // and reader must both carry it or the constant flips sign on every
  // round-trip.
  const bool has_obj_constant = p_.obj_constant != 0.0;

  std::vector<std::size_t> rows_with_rhs;
  for (std::size_t i = 0; i < p_.num_rows(); ++i) {
    const Real lo = p_.row_lower[i];
    const Real hi = p_.row_upper[i];
    const bool lf = is_finite_bound(lo);
    const bool hf = is_finite_bound(hi);
    if (!lf && !hf) continue;               // free row: no RHS
    const Real rhs = hf ? hi : lo;          // matches the sense chosen above
    if (rhs != 0.0) rows_with_rhs.push_back(i);
  }
  if (rows_with_rhs.empty() && !has_obj_constant) return;

  line("RHS");
  if (has_obj_constant) {
    field_line({"RHS", obj_name(), number(-p_.obj_constant)});
  }
  for (const auto i : rows_with_rhs) {
    const Real hi = p_.row_upper[i];
    const Real rhs = is_finite_bound(hi) ? hi : p_.row_lower[i];
    field_line({"RHS", row_name(i), number(rhs)});
  }
}

void MpsWriter::write_ranges() {
  std::vector<std::size_t> ranged;
  for (std::size_t i = 0; i < p_.num_rows(); ++i) {
    const Real lo = p_.row_lower[i];
    const Real hi = p_.row_upper[i];
    if (is_finite_bound(lo) && is_finite_bound(hi) && lo != hi) {
      ranged.push_back(i);
    }
  }
  if (ranged.empty()) return;

  line("RANGES");
  for (const auto i : ranged) {
    // Written as `L` with RHS = hi, so the reader reconstructs
    // [hi - |R|, hi]. R = hi - lo, which is positive by construction here.
    const Real r = p_.row_upper[i] - p_.row_lower[i];
    field_line({"RNG", row_name(i), number(r)});
  }
}

void MpsWriter::write_bounds() {
  std::vector<std::size_t> bounded;
  for (std::size_t j = 0; j < p_.num_cols(); ++j) {
    const Real lo = p_.col_lower[j];
    const Real hi = p_.col_upper[j];
    if (lo != 0.0 || is_finite_bound(hi)) bounded.push_back(j);
  }
  if (bounded.empty()) return;

  line("BOUNDS");
  for (const auto j : bounded) {
    const Real lo = p_.col_lower[j];
    const Real hi = p_.col_upper[j];
    const bool lf = is_finite_bound(lo);
    const bool hf = is_finite_bound(hi);
    const std::string cname = col_name(j);

    if (lf && hf && lo == hi) {
      field_line({"FX", "BND", cname, number(lo)});
      continue;
    }
    if (!lf && !hf) {
      field_line({"FR", "BND", cname});
      continue;
    }
    // An infinite lower bound is written as MI, and it is written BEFORE the
    // UP. That order matters: a reader seeing `UP` with a negative value and no
    // prior lower-bound record treats the lower bound as -infinity (the
    // notorious MPS quirk, see MPS-FORMAT-NOTES). Emitting MI first means a
    // lower bound is always on record before any UP, so the quirk can never
    // fire on our own output whichever convention the reader takes.
    if (!lf) {
      field_line({"MI", "BND", cname});
    } else if (lo != 0.0) {
      field_line({"LO", "BND", cname, number(lo)});
    }
    if (hf) {
      field_line({"UP", "BND", cname, number(hi)});
    }
  }
}

void MpsWriter::write_quadobj() {
  if (!o_.write_quadratic || p_.Q.empty()) return;

  // Q is held full symmetric; QUADOBJ carries the lower triangle, and the
  // reader mirrors it back. Emitting both triangles would double every
  // off-diagonal coefficient on the next read.
  const auto off = p_.Q.csr.offsets();
  const auto idx = p_.Q.csr.indices();
  const auto val = p_.Q.csr.values();

  bool any = false;
  for (std::size_t i = 0; i < p_.Q.rows() && !any; ++i) {
    for (auto k = static_cast<std::size_t>(off[i]);
         k < static_cast<std::size_t>(off[i + 1]); ++k) {
      if (static_cast<std::size_t>(idx[k]) <= i && val[k] != 0.0) any = true;
    }
  }
  if (!any) return;

  line("QUADOBJ");
  for (std::size_t i = 0; i < p_.Q.rows(); ++i) {
    for (auto k = static_cast<std::size_t>(off[i]);
         k < static_cast<std::size_t>(off[i + 1]); ++k) {
      const auto j = static_cast<std::size_t>(idx[k]);
      if (j > i || val[k] == 0.0) continue;
      // QUADOBJ names the column pair as (column, column), so the row index of
      // Q is a COLUMN index of the model.
      field_line({col_name(i), col_name(j), number(val[k])});
    }
  }
}

core::Expected<std::string> MpsWriter::run() {
  if (auto st = check_names(); !st.ok()) return st.error();

  if (!p_.sos_sets.empty()) {
    return core::make_error(
        ErrorCode::UnsupportedFeature,
        "the model carries SOS sets, which this writer cannot emit; writing it "
        "anyway would produce a file that silently differs from the model");
  }

  out_.reserve(p_.nnz() * 32 + 1024);

  line("NAME          " + (p_.problem_name.empty() ? std::string("SOVSOLVE")
                                                   : p_.problem_name));
  if (p_.sense == core::ObjSense::Maximize || o_.always_write_objsense) {
    line("OBJSENSE");
    field_line({p_.sense == core::ObjSense::Maximize ? "MAX" : "MIN"});
  }
  write_rows();
  write_columns();
  write_rhs();
  write_ranges();
  write_bounds();
  write_quadobj();
  line("ENDATA");
  return out_;
}

}  // namespace

core::Expected<std::string> writeMpsToString(const model::Problem& problem,
                                             const WriteOptions& options) {
  MpsWriter w(problem, options);
  return w.run();
}

core::Status writeProblem(const std::string& filepath,
                          const model::Problem& problem, FileFormat format,
                          const WriteOptions& options) {
  FileFormat resolved = format;
  if (resolved == FileFormat::Auto) {
    resolved = filepath.size() >= 3 &&
                       filepath.compare(filepath.size() - 3, 3, ".lp") == 0
                   ? FileFormat::Lp
                   : FileFormat::Mps;
  }
  if (resolved != FileFormat::Mps) {
    return core::make_error(ErrorCode::NotImplemented,
                            "only MPS output is implemented");
  }

  auto text = writeMpsToString(problem, options);
  if (!text.has_value()) return std::move(text).error();

  std::FILE* f = std::fopen(filepath.c_str(), "wb");
  if (f == nullptr) {
    return core::make_error(ErrorCode::FileReadFailed,
                            "cannot open " + filepath + " for writing");
  }
  const auto& s = text.value();
  const std::size_t written = std::fwrite(s.data(), 1, s.size(), f);
  const int closed = std::fclose(f);
  if (written != s.size() || closed != 0) {
    return core::make_error(ErrorCode::FileReadFailed,
                            "short write to " + filepath);
  }
  return core::Status{};
}

}  // namespace sovsolve::io
