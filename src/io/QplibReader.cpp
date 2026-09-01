// QPLIB format reader.
//
// ---------------------------------------------------------------------------
// The format
// ---------------------------------------------------------------------------
//
// Specified in Table 8 of Furini et al., "QPLIB: a library of quadratic
// programming instances" (Math. Prog. Comp. 2019; RAL-P-2017-003 Appendix B).
// It is line-oriented and positional -- there are no section keywords at all,
// so a reader that loses its place produces a plausible but wrong model rather
// than an error. Every read below therefore states what it is reading.
//
// The model it encodes is
//
//     min or max   1/2 x'Q0 x + b0'x + q0
//     s.t.         cl_i <= 1/2 x'Qi x + bi'x <= cu_i     i in M
//                  l_j <= x_j <= u_j                     j in N
//                  x_j integer                           j in Z
//
// The `1/2` on the objective is the same convention `Problem` uses, so `Q0`
// transfers directly. Q is stored as its LOWER TRIANGLE and is symmetric, so an
// off-diagonal entry mirrors -- it is NOT halved. Verified against the paper's
// worked example, which is reproduced verbatim as a test.
//
// ---------------------------------------------------------------------------
// Two things that make it easy to get wrong
// ---------------------------------------------------------------------------
//
// SECTIONS ARE CONDITIONALLY ABSENT, keyed off the three-character type code.
// A linear objective omits Q0 entirely; an unconstrained problem omits `m` and
// every constraint section; all-binary variables omit the bound sections. Read
// a section that is not there and every subsequent field is off by one, with no
// syntax error to notice. The notes in Table 8 are encoded in `Sections` below.
//
// ALMOST EVERY VECTOR IS "DEFAULT PLUS EXCEPTIONS": a default value, a count of
// non-default entries, then that many index/value pairs. Convenient in a file
// where most bounds are identical, but it means a miscounted section silently
// shifts the meaning of everything after it.
//
// Indices in the file are 1-BASED.
//
// ---------------------------------------------------------------------------
// Scope
// ---------------------------------------------------------------------------
//
// Quadratic CONSTRAINTS are refused. `Problem` models linear constraints only,
// which is the team's stated scope (LP and convex QP), and silently dropping
// the `Qi` terms would turn a quadratically-constrained instance into a
// different, easier problem that still looks valid. Types with a third
// character of `D`, `C` or `Q` therefore report UnsupportedFeature.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "io/detail/NumberParse.hpp"
#include "io/detail/Readers.hpp"
#include "io/detail/Tokenizer.hpp"
#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"

namespace sovsolve::io {
namespace {

using core::ErrorCode;
using core::Index;
using core::INF;
using core::Real;
using core::VarType;
using detail::iequals;
using detail::LineScanner;
using detail::parse_int;
using detail::parse_real;

/// Which optional sections the three-character type code says are present.
///
/// Straight from the notes to Table 8. Getting one of these wrong does not
/// produce a parse error -- it produces a silently misaligned file -- so they
/// are named rather than inlined at the point of use.
struct Sections {
  bool objective_quadratic = true;   ///< absent for `L**`            (note 3)
  bool constraints = true;           ///< absent for `**N`, `**B`     (note 2)
  bool constraint_quadratic = true;  ///< absent for `**N/B/L`        (note 4)
  bool variable_types = true;        ///< absent for `*C*`,`*B*`,`*I*`(note 5)
  bool variable_bounds = true;       ///< absent for `*B*`            (note 6)
  bool all_binary = false;           ///< `*B*`: bounds are implicitly [0,1]
  char objective = 'Q';
  char variables = 'C';
  char constraint_kind = 'L';
};

Sections classify(std::string_view type) noexcept {
  Sections s;
  if (type.size() >= 3) {
    s.objective = static_cast<char>(std::toupper(static_cast<unsigned char>(type[0])));
    s.variables = static_cast<char>(std::toupper(static_cast<unsigned char>(type[1])));
    s.constraint_kind = static_cast<char>(std::toupper(static_cast<unsigned char>(type[2])));
  }
  s.objective_quadratic = s.objective != 'L';
  s.constraints = s.constraint_kind != 'N' && s.constraint_kind != 'B';
  s.constraint_quadratic = s.constraints && s.constraint_kind != 'L';
  s.variable_types =
      s.variables != 'C' && s.variables != 'B' && s.variables != 'I';
  s.all_binary = s.variables == 'B';
  s.variable_bounds = !s.all_binary;
  return s;
}

/// Line-oriented cursor.
///
/// A `!` or `#` in column 1 is a comment and blank lines are skipped. On a data
/// line only the leading fields are meaningful: the format's own example
/// annotates records with trailing prose (`3   # variables`), so anything after
/// the fields a record needs is discarded rather than parsed.
class Cursor {
 public:
  explicit Cursor(std::string_view text) : scan_(text) {}

  /// Advance to the next line carrying data. False at end of input.
  bool next() {
    std::string_view line;
    std::size_t no = 0;
    while (scan_.next(line, no)) {
      line_no_ = no;
      const std::size_t first = line.find_first_not_of(" \t\r");
      if (first == std::string_view::npos) continue;
      if (line[first] == '!' || line[first] == '#') continue;
      fields_ = split(line);
      if (fields_.empty()) continue;
      return true;
    }
    return false;
  }

  [[nodiscard]] std::size_t line_no() const noexcept { return line_no_; }
  [[nodiscard]] std::size_t field_count() const noexcept { return fields_.size(); }
  [[nodiscard]] std::string_view field(std::size_t i) const noexcept {
    return i < fields_.size() ? fields_[i] : std::string_view{};
  }

 private:
  static std::vector<std::string_view> split(std::string_view line) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < line.size()) {
      while (i < line.size() && detail::is_space(line[i])) ++i;
      if (i >= line.size()) break;
      const std::size_t start = i;
      while (i < line.size() && !detail::is_space(line[i])) ++i;
      out.push_back(line.substr(start, i - start));
    }
    return out;
  }

  LineScanner scan_;
  std::vector<std::string_view> fields_;
  std::size_t line_no_ = 0;
};

class QplibParser {
 public:
  explicit QplibParser(std::string_view text) : cur_(text) {}

  core::Expected<model::Problem> run();

 private:
  [[nodiscard]] core::Error err(ErrorCode code, std::string what) const {
    core::Error e;
    e.code = code;
    e.message = std::move(what);
    e.line = cur_.line_no();
    if (!reading_.empty()) e.section = reading_;
    return e;
  }

  /// Advance a line, recording WHAT is expected there. The label is what turns
  /// a positional misalignment into a diagnosable error instead of a wrong
  /// model, so every caller supplies one.
  bool advance(const char* what) {
    reading_ = what;
    return cur_.next();
  }

  core::Expected<long long> want_int(const char* what, std::size_t f = 0) {
    if (!advance(what) && f == 0) {
      return err(ErrorCode::ParseError,
                 std::string("unexpected end of file, expected ") + what);
    }
    const auto v = parse_int(cur_.field(f));
    if (!v.has_value()) {
      return err(ErrorCode::ParseError,
                 std::string("expected an integer for ") + what + ", found '" +
                     std::string(cur_.field(f)) + "'");
    }
    return *v;
  }

  core::Expected<Real> want_real(const char* what) {
    if (!advance(what)) {
      return err(ErrorCode::ParseError,
                 std::string("unexpected end of file, expected ") + what);
    }
    const auto v = parse_real(cur_.field(0));
    if (!v.has_value()) {
      return err(ErrorCode::ParseError,
                 std::string("expected a number for ") + what + ", found '" +
                     std::string(cur_.field(0)) + "'");
    }
    return *v;
  }

  /// Read a "default value, then N index/value exceptions" vector.
  core::Status read_sparse_vector(const char* what, std::size_t size,
                                  std::vector<Real>& out);

  Cursor cur_;
  std::string reading_;
  Real file_infinity_ = 1e20;

  /// Map the file's infinity onto ours. Anything at or beyond the file's
  /// declared infinity is infinite, whatever its magnitude.
  [[nodiscard]] Real bound(Real v) const noexcept {
    if (v >= file_infinity_) return INF;
    if (v <= -file_infinity_) return -INF;
    return v;
  }
};

core::Status QplibParser::read_sparse_vector(const char* what, std::size_t size,
                                             std::vector<Real>& out) {
  const std::string label = what;
  auto def = want_real((label + " default").c_str());
  if (!def.has_value()) return std::move(def).error();
  out.assign(size, def.value());

  auto count = want_int((label + " exception count").c_str());
  if (!count.has_value()) return std::move(count).error();
  if (count.value() < 0) {
    return err(ErrorCode::ParseError, label + " has a negative entry count");
  }

  for (long long k = 0; k < count.value(); ++k) {
    if (!advance((label + " entry").c_str())) {
      return err(ErrorCode::ParseError,
                 "unexpected end of file inside " + label);
    }
    const auto idx = parse_int(cur_.field(0));
    const auto val = parse_real(cur_.field(1));
    if (!idx.has_value() || !val.has_value()) {
      return err(ErrorCode::ParseError,
                 label + " entry is not an index/value pair");
    }
    // 1-based in the file.
    const long long j = *idx - 1;
    if (j < 0 || static_cast<std::size_t>(j) >= size) {
      return err(ErrorCode::ParseError,
                 label + " index " + std::to_string(*idx) + " is out of range");
    }
    out[static_cast<std::size_t>(j)] = *val;
  }
  return core::Status{};
}

core::Expected<model::Problem> QplibParser::run() {
  // -- header --------------------------------------------------------------
  if (!advance("problem name")) {
    return err(ErrorCode::ParseError, "file is empty");
  }
  const std::string problem_name(cur_.field(0));

  if (!advance("problem type")) {
    return err(ErrorCode::ParseError, "missing problem type");
  }
  const std::string type(cur_.field(0));
  if (type.size() < 3) {
    return err(ErrorCode::ParseError,
               "problem type '" + type + "' is not three characters");
  }
  const Sections sec = classify(type);

  if (sec.constraint_quadratic) {
    return err(ErrorCode::UnsupportedFeature,
               "problem type '" + type +
                   "' has quadratic constraints, which this model does not "
                   "represent; dropping the quadratic terms would silently "
                   "turn it into a different, easier problem");
  }

  if (!advance("objective sense")) {
    return err(ErrorCode::ParseError, "missing objective sense");
  }
  const std::string_view sense_word = cur_.field(0);
  const bool maximize = iequals(sense_word, "maximize") ||
                        iequals(sense_word, "maximise") ||
                        iequals(sense_word, "max");
  if (!maximize && !iequals(sense_word, "minimize") &&
      !iequals(sense_word, "minimise") && !iequals(sense_word, "min")) {
    return err(ErrorCode::ParseError,
               "objective sense must be minimize or maximize, found '" +
                   std::string(sense_word) + "'");
  }

  auto n_var = want_int("number of variables");
  if (!n_var.has_value()) return std::move(n_var).error();
  if (n_var.value() <= 0) {
    return err(ErrorCode::ParseError, "variable count must be positive");
  }
  const auto n = static_cast<std::size_t>(n_var.value());

  std::size_t m = 0;
  if (sec.constraints) {
    auto n_con = want_int("number of constraints");
    if (!n_con.has_value()) return std::move(n_con).error();
    if (n_con.value() < 0) {
      return err(ErrorCode::ParseError, "constraint count is negative");
    }
    m = static_cast<std::size_t>(n_con.value());
  }

  model::Problem p;
  p.sense = maximize ? core::ObjSense::Maximize : core::ObjSense::Minimize;
  p.problem_name = problem_name;
  p.objective_row_name = "obj";

  // -- objective Hessian ---------------------------------------------------
  //
  // Lower triangle of a symmetric matrix, so an off-diagonal entry mirrors into
  // both positions at full value. Halving it here is the classic error: the
  // paper's own example has Q0_21 = -1 producing the term -x1*x2 under the
  // model's 1/2 factor, which only works if both (2,1) and (1,2) hold -1.
  if (sec.objective_quadratic) {
    auto nnz = want_int("objective Hessian nonzero count");
    if (!nnz.has_value()) return std::move(nnz).error();
    if (nnz.value() < 0) {
      return err(ErrorCode::ParseError, "negative Hessian nonzero count");
    }
    struct Entry {
      Index r;
      Index c;
      Real v;
    };
    std::vector<Entry> entries;
    entries.reserve(static_cast<std::size_t>(nnz.value()) * 2);

    for (long long k = 0; k < nnz.value(); ++k) {
      if (!advance("objective Hessian entry")) {
        return err(ErrorCode::ParseError,
                   "unexpected end of file inside the objective Hessian");
      }
      const auto h = parse_int(cur_.field(0));
      const auto c = parse_int(cur_.field(1));
      const auto v = parse_real(cur_.field(2));
      if (!h.has_value() || !c.has_value() || !v.has_value()) {
        return err(ErrorCode::ParseError,
                   "objective Hessian entry is not a row/column/value triple");
      }
      const long long hi = *h - 1;
      const long long ci = *c - 1;
      if (hi < 0 || ci < 0 || static_cast<std::size_t>(hi) >= n ||
          static_cast<std::size_t>(ci) >= n) {
        return err(ErrorCode::ParseError,
                   "objective Hessian index out of range at (" +
                       std::to_string(*h) + "," + std::to_string(*c) + ")");
      }
      entries.push_back({static_cast<Index>(hi), static_cast<Index>(ci), *v});
      if (hi != ci) {
        entries.push_back({static_cast<Index>(ci), static_cast<Index>(hi), *v});
      }
    }
    if (!entries.empty()) {
      core::SparseBuilder qb(n, n);
      for (const auto& e : entries) qb.count(e.r, e.c);
      if (auto st = qb.allocate(); !st.ok()) return st.error();
      for (const auto& e : entries) qb.insert(e.r, e.c, e.v);
      p.Q = qb.finish(true, 0.0);
    }
  }

  // -- linear objective and constant ---------------------------------------
  std::vector<Real> c_vec;
  if (auto st = read_sparse_vector("linear objective", n, c_vec); !st.ok()) {
    return st.error();
  }
  auto q0 = want_real("objective constant");
  if (!q0.has_value()) return std::move(q0).error();
  p.obj_constant = q0.value();

  p.c = core::RealVector(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) p.c[j] = c_vec[j];

  // -- constraint matrix ---------------------------------------------------
  //
  // Given as a flat list of (constraint, variable, value) triples rather than
  // the default-plus-exceptions shape, since a constraint matrix has no
  // meaningful default.
  std::vector<Real> row_lo(m, 0.0);
  std::vector<Real> row_hi(m, 0.0);
  {
    struct Entry {
      Index r;
      Index c;
      Real v;
    };
    std::vector<Entry> entries;

    if (sec.constraints) {
      auto nnz = want_int("constraint matrix nonzero count");
      if (!nnz.has_value()) return std::move(nnz).error();
      if (nnz.value() < 0) {
        return err(ErrorCode::ParseError, "negative constraint nonzero count");
      }
      entries.reserve(static_cast<std::size_t>(nnz.value()));
      for (long long k = 0; k < nnz.value(); ++k) {
        if (!advance("constraint matrix entry")) {
          return err(ErrorCode::ParseError,
                     "unexpected end of file inside the constraint matrix");
        }
        const auto i = parse_int(cur_.field(0));
        const auto j = parse_int(cur_.field(1));
        const auto v = parse_real(cur_.field(2));
        if (!i.has_value() || !j.has_value() || !v.has_value()) {
          return err(ErrorCode::ParseError,
                     "constraint entry is not a row/column/value triple");
        }
        const long long ri = *i - 1;
        const long long cj = *j - 1;
        if (ri < 0 || static_cast<std::size_t>(ri) >= m || cj < 0 ||
            static_cast<std::size_t>(cj) >= n) {
          return err(ErrorCode::ParseError,
                     "constraint entry index out of range at (" +
                         std::to_string(*i) + "," + std::to_string(*j) + ")");
        }
        entries.push_back({static_cast<Index>(ri), static_cast<Index>(cj), *v});
      }
    }

    core::SparseBuilder builder(m, n);
    for (const auto& e : entries) builder.count(e.r, e.c);
    if (auto st = builder.allocate(); !st.ok()) return st.error();
    for (const auto& e : entries) builder.insert(e.r, e.c, e.v);
    p.A = builder.finish(true, 0.0);
  }

  // -- infinity, then all four bound vectors -------------------------------
  if (sec.constraints) {
    auto inf = want_real("infinity value");
    if (!inf.has_value()) return std::move(inf).error();
    file_infinity_ = inf.value();

    if (auto st = read_sparse_vector("constraint lower bound", m, row_lo);
        !st.ok()) {
      return st.error();
    }
    if (auto st = read_sparse_vector("constraint upper bound", m, row_hi);
        !st.ok()) {
      return st.error();
    }
  }

  std::vector<Real> col_lo(n, 0.0);
  std::vector<Real> col_hi(n, sec.all_binary ? 1.0 : INF);
  if (sec.variable_bounds) {
    if (!sec.constraints) {
      // The infinity line belongs to the constraint block; with no constraints
      // it has not been read yet, and the bound sections still need it.
      auto inf = want_real("infinity value");
      if (!inf.has_value()) return std::move(inf).error();
      file_infinity_ = inf.value();
    }
    if (auto st = read_sparse_vector("variable lower bound", n, col_lo);
        !st.ok()) {
      return st.error();
    }
    if (auto st = read_sparse_vector("variable upper bound", n, col_hi);
        !st.ok()) {
      return st.error();
    }
  }

  p.row_lower = core::RealVector(m, 0.0);
  p.row_upper = core::RealVector(m, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    p.row_lower[i] = bound(row_lo[i]);
    p.row_upper[i] = bound(row_hi[i]);
  }
  p.col_lower = core::RealVector(n, 0.0);
  p.col_upper = core::RealVector(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    p.col_lower[j] = bound(col_lo[j]);
    p.col_upper[j] = bound(col_hi[j]);
  }

  // -- variable types ------------------------------------------------------
  p.col_type.assign(n, VarType::Continuous);
  if (sec.all_binary) {
    p.col_type.assign(n, VarType::Binary);
  } else if (sec.variables == 'I') {
    // Note 5: an all-integer problem states binaries as integers bounded 0..1
    // rather than carrying a type section.
    p.col_type.assign(n, VarType::Integer);
  } else if (sec.variable_types) {
    std::vector<Real> types;
    if (auto st = read_sparse_vector("variable type", n, types); !st.ok()) {
      return st.error();
    }
    for (std::size_t j = 0; j < n; ++j) {
      const int t = static_cast<int>(types[j]);
      if (t == 1) {
        p.col_type[j] = VarType::Integer;
      } else if (t == 2) {
        p.col_type[j] = VarType::Binary;
        // Note: an explicit binary type overrides whatever the bound sections
        // said, per the paper -- binaries are [0,1] however they were declared.
        p.col_lower[j] = 0.0;
        p.col_upper[j] = 1.0;
      } else if (t != 0) {
        return err(ErrorCode::ParseError,
                   "variable type " + std::to_string(t) +
                       " is not 0 (continuous), 1 (integer) or 2 (binary)");
      }
    }
  }

  // Starting points for x, y and z, and the name overrides, all follow. They
  // are read only far enough to be skipped: `Problem` is the model, not a
  // solver state, and a warm start belongs to whoever produced it.
  std::vector<Real> discard;
  if (auto st = read_sparse_vector("starting point x", n, discard); !st.ok()) {
    return st.error();
  }
  if (sec.constraints) {
    if (auto st = read_sparse_vector("starting point y", m, discard); !st.ok()) {
      return st.error();
    }
  }
  if (auto st = read_sparse_vector("starting point z", n, discard); !st.ok()) {
    return st.error();
  }

  // -- names ---------------------------------------------------------------
  //
  // Only the exceptions are listed; a variable with no override is named by its
  // own index, as a decimal string.
  std::vector<std::string> col_names(n);
  std::vector<std::string> row_names(m);
  for (std::size_t j = 0; j < n; ++j) col_names[j] = std::to_string(j + 1);
  for (std::size_t i = 0; i < m; ++i) row_names[i] = std::to_string(i + 1);

  const auto read_names = [&](const char* what, std::size_t size,
                              std::vector<std::string>& into) -> core::Status {
    auto count = want_int(what);
    if (!count.has_value()) return std::move(count).error();
    if (count.value() < 0) {
      return err(ErrorCode::ParseError,
                 std::string(what) + " count is negative");
    }
    for (long long k = 0; k < count.value(); ++k) {
      if (!advance(what)) {
        return err(ErrorCode::ParseError,
                   std::string("unexpected end of file inside ") + what);
      }
      const auto idx = parse_int(cur_.field(0));
      if (!idx.has_value() || cur_.field_count() < 2) {
        return err(ErrorCode::ParseError,
                   std::string(what) + " entry is not an index/name pair");
      }
      const long long i = *idx - 1;
      if (i < 0 || static_cast<std::size_t>(i) >= size) {
        return err(ErrorCode::ParseError,
                   std::string(what) + " index out of range");
      }
      into[static_cast<std::size_t>(i)] = std::string(cur_.field(1));
    }
    return core::Status{};
  };

  if (auto st = read_names("variable name count", n, col_names); !st.ok()) {
    return st.error();
  }
  if (sec.constraints) {
    if (auto st = read_names("constraint name count", m, row_names); !st.ok()) {
      return st.error();
    }
  }

  p.col_names.reserve(n);
  for (const auto& s : col_names) p.col_names.add(s);
  p.row_names.reserve(m);
  for (const auto& s : row_names) p.row_names.add(s);

  return p;
}

}  // namespace

core::Expected<model::Problem> parse_qplib(std::string_view content,
                                           const model::ReaderOptions& options) {
  (void)options;  // no ambiguous constructs in this format to police
  QplibParser parser(content);
  return parser.run();
}

}  // namespace sovsolve::io
