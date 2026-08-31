// MPS reader.
//
// Structure: one full pass builds every symbol table and reads every section
// except the matrix values, then a second pass over COLUMNS alone fills the
// pre-sized buffers. See SparseBuilder.hpp for why counting first beats triplet
// staging.
//
// Every format rule implemented here is documented in docs/MPS-FORMAT-NOTES.md,
// including the ones that are silent-wrong-answer traps rather than parse
// errors: the negated objective constant, the four RANGES cases, the
// negative-UP bound quirk, and integer markers.

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "io/detail/MappedFile.hpp"
#include "io/detail/NumberParse.hpp"
#include "io/detail/SymbolTable.hpp"
#include "io/detail/Tokenizer.hpp"
#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/io/Load.hpp"

namespace sovsolve::io {
namespace {

using core::Error;
using core::ErrorCode;
using core::Expected;
using core::Index;
using core::INF;
using core::ObjSense;
using core::Real;
using core::RowType;
using core::VarType;
using detail::FieldArray;
using detail::iequals;
using detail::parse_real;
using model::Problem;
using model::ReaderOptions;

enum class Section {
  None, Name, ObjSense, Rows, Columns, Rhs, Ranges, Bounds,
  QuadObj, QMatrix, Sos, EndData, Unknown,
};

Section classify_section(std::string_view token) noexcept {
  if (iequals(token, "NAME")) return Section::Name;
  if (iequals(token, "OBJSENSE") || iequals(token, "OBJSENSE:")) return Section::ObjSense;
  if (iequals(token, "ROWS")) return Section::Rows;
  if (iequals(token, "COLUMNS")) return Section::Columns;
  if (iequals(token, "RHS")) return Section::Rhs;
  if (iequals(token, "RANGES")) return Section::Ranges;
  if (iequals(token, "BOUNDS")) return Section::Bounds;
  if (iequals(token, "QUADOBJ") || iequals(token, "QSECTION")) return Section::QuadObj;
  if (iequals(token, "QMATRIX")) return Section::QMatrix;
  if (iequals(token, "SOS")) return Section::Sos;
  if (iequals(token, "ENDATA")) return Section::EndData;
  return Section::Unknown;
}

Error located(ErrorCode code, std::string message, std::size_t line,
              std::string section) {
  return Error{code, std::move(message), line, std::nullopt, std::move(section)};
}

/// Detect fixed vs free format.
///
/// The only thing fixed format can express that free cannot is a name
/// containing a space, and that shows up as a ROWS data line splitting into
/// more than two whitespace fields. Cheap, and it catches the case that
/// actually matters.
bool detect_fixed_format(std::string_view text) noexcept {
  detail::LineScanner scan(text);
  std::string_view line;
  std::size_t no = 0;
  Section current = Section::None;
  while (scan.next(line, no)) {
    if (detail::is_skippable(line)) continue;
    if (detail::is_section_header(line)) {
      FieldArray f{};
      const std::size_t n = detail::split_free(line, f);
      current = n > 0 ? classify_section(f[0]) : Section::Unknown;
      if (current == Section::Columns) break;  // past ROWS; nothing more to learn
      continue;
    }
    if (current == Section::Rows && detail::count_free_fields(line) > 2) {
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

class MpsParser {
 public:
  MpsParser(std::string_view text, const ReaderOptions& options)
      : text_(text), options_(options) {}

  Expected<Problem> run() {
    use_fixed_ = options_.auto_detect_format ? detect_fixed_format(text_)
                                             : options_.fixed_format;

    if (auto st = pass_one(); !st.ok()) return st.error();
    if (auto st = finalize_rows(); !st.ok()) return st.error();
    if (auto st = pass_two(); !st.ok()) return st.error();
    return assemble();
  }

 private:
  std::size_t split(std::string_view line, FieldArray& out) const noexcept {
    return use_fixed_ ? detail::split_fixed(line, out)
                      : detail::split_free(line, out);
  }

  // -- pass one: everything except matrix values --------------------------

  core::Status pass_one() {
    detail::LineScanner scan(text_);
    std::string_view line;
    std::size_t no = 0;
    Section current = Section::None;
    bool in_integer_block = false;

    while (scan.next(line, no)) {
      if (detail::is_skippable(line)) continue;

      if (detail::is_section_header(line)) {
        FieldArray f{};
        const std::size_t n = detail::split_free(line, f);
        current = n > 0 ? classify_section(f[0]) : Section::Unknown;

        if (current == Section::Name && n > 1) problem_name_ = std::string(f[1]);
        // OBJSENSE may sit on the header line or on the following data line.
        if (current == Section::ObjSense && n > 1) read_objsense(f[1]);
        if (current == Section::EndData) break;
        if (current == Section::Unknown) {
          return located(ErrorCode::UnknownSection,
                         "unrecognized section '" + std::string(f[0]) + "'", no, "");
        }
        continue;
      }

      FieldArray f{};
      const std::size_t n = split(line, f);
      if (n == 0) continue;

      switch (current) {
        case Section::ObjSense:
          read_objsense(f[0]);
          break;

        case Section::Rows: {
          if (n < 2) {
            return located(ErrorCode::ParseError,
                           "ROWS entry needs a type and a name", no, "ROWS");
          }
          if (auto st = add_row(f[0], f[1], no); !st.ok()) return st;
          break;
        }

        case Section::Columns: {
          if (is_marker_line(f, n)) {
            const bool org = marker_is_intorg(f, n);
            in_integer_block = org;
            break;
          }
          if (auto st = count_column_entries(f, n, no, in_integer_block); !st.ok()) {
            return st;
          }
          break;
        }

        case Section::Rhs:
          if (auto st = read_rhs(f, n, no); !st.ok()) return st;
          break;

        case Section::Ranges:
          if (auto st = read_ranges(f, n, no); !st.ok()) return st;
          break;

        case Section::Bounds:
          if (auto st = read_bounds(f, n, no); !st.ok()) return st;
          break;

        case Section::QuadObj:
        case Section::QMatrix:
          if (auto st = read_quad(f, n, no, current == Section::QMatrix); !st.ok()) {
            return st;
          }
          break;

        case Section::Sos:
          // Recorded, not silently dropped -- ignoring an SOS set changes the
          // model and yields a confidently wrong answer.
          saw_sos_ = true;
          break;

        case Section::Name:
          if (problem_name_.empty()) problem_name_ = std::string(f[0]);
          break;

        default:
          break;
      }
    }

    if (objective_row_ < 0) {
      return core::make_error(ErrorCode::ParseError,
                              "no objective row: the ROWS section declared no N row");
    }
    return core::Status::Ok();
  }

  void read_objsense(std::string_view token) noexcept {
    if (iequals(token, "MAX") || iequals(token, "MAXIMIZE")) {
      sense_ = ObjSense::Maximize;
    } else if (iequals(token, "MIN") || iequals(token, "MINIMIZE")) {
      sense_ = ObjSense::Minimize;
    }
  }

  core::Status add_row(std::string_view type, std::string_view name,
                       std::size_t no) {
    RowType kind{};
    if (iequals(type, "N")) kind = RowType::Free;
    else if (iequals(type, "L")) kind = RowType::Less;
    else if (iequals(type, "G")) kind = RowType::Greater;
    else if (iequals(type, "E")) kind = RowType::Equal;
    else {
      return located(ErrorCode::ParseError,
                     "unknown row type '" + std::string(type) + "'", no, "ROWS");
    }

    bool inserted = false;
    const Index id = rows_.insert(name, inserted);
    if (!inserted) {
      return located(ErrorCode::DuplicateName,
                     "row '" + std::string(name) + "' declared twice", no, "ROWS");
    }
    row_kind_.push_back(kind);
    row_name_.push_back(name);

    if (kind == RowType::Free) {
      // The FIRST N row is the objective. Every later N row is a free row and
      // is dropped -- files carrying alternative objectives are common.
      if (objective_row_ < 0) {
        objective_row_ = id;
        objective_name_ = name;
      } else {
        ++free_rows_dropped_;
      }
      row_to_constraint_.push_back(-1);
    } else {
      row_to_constraint_.push_back(static_cast<Index>(constraint_count_++));
    }
    return core::Status::Ok();
  }

  static bool is_marker_line(const FieldArray& f, std::size_t n) noexcept {
    // The marker's own name is arbitrary, so match on the literal 'MARKER'
    // field rather than on field position.
    for (std::size_t i = 0; i < n; ++i) {
      if (iequals(detail::unquote(f[i]), "MARKER")) return true;
    }
    return false;
  }

  static bool marker_is_intorg(const FieldArray& f, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
      if (iequals(detail::unquote(f[i]), "INTORG")) return true;
    }
    return false;
  }

  /// Register the column and count its entries. Values are read in pass two.
  core::Status count_column_entries(const FieldArray& f, std::size_t n,
                                    std::size_t no, bool integer_block) {
    if (n < 3) {
      return located(ErrorCode::ParseError,
                     "COLUMNS entry needs a column, a row and a value", no,
                     "COLUMNS");
    }

    bool inserted = false;
    const Index col = cols_.insert(f[0], inserted);
    if (inserted) {
      col_name_.push_back(f[0]);
      col_type_.push_back(integer_block ? VarType::Integer : VarType::Continuous);
      col_lower_.push_back(0.0);
      col_upper_.push_back(INF);
      col_saw_lower_.push_back(false);
      col_saw_upper_.push_back(false);
    }

    // Pairs at fields (1,2) and, when present, (3,4).
    for (std::size_t p = 1; p + 1 < n; p += 2) {
      const Index row = rows_.find(f[p]);
      if (row < 0) {
        return located(ErrorCode::UndefinedName,
                       "row '" + std::string(f[p]) + "' was never declared", no,
                       "COLUMNS");
      }
      if (!parse_real(f[p + 1]).has_value()) {
        return located(ErrorCode::ParseError,
                       "'" + std::string(f[p + 1]) + "' is not a number", no,
                       "COLUMNS");
      }
      if (row == objective_row_) continue;                 // objective coefficient
      const Index cons = row_to_constraint_[static_cast<std::size_t>(row)];
      if (cons < 0) continue;                              // free row: dropped
      pending_counts_.push_back({cons, col});
    }
    return core::Status::Ok();
  }

  /// RHS and RANGES are read during pass one, before `finalize_rows()` runs.
  /// The ROWS section is complete by then, so `constraint_count_` is final and
  /// these can be sized on first use.
  void ensure_row_arrays() {
    if (rhs_.size() < constraint_count_) {
      rhs_.resize(constraint_count_, 0.0);
      range_.resize(constraint_count_, 0.0);
      has_range_.resize(constraint_count_, false);
    }
  }

  core::Status read_rhs(const FieldArray& f, std::size_t n, std::size_t no) {
    ensure_row_arrays();
    // Field 0 is a set name, conventionally ignored, but some writers omit it.
    // Disambiguate by checking whether field 0 names a known row.
    const std::size_t base = (rows_.find(f[0]) >= 0) ? 0 : 1;
    for (std::size_t p = base; p + 1 < n; p += 2) {
      const Index row = rows_.find(f[p]);
      if (row < 0) {
        return located(ErrorCode::UndefinedName,
                       "row '" + std::string(f[p]) + "' was never declared", no,
                       "RHS");
      }
      const auto v = parse_real(f[p + 1]);
      if (!v) {
        return located(ErrorCode::ParseError,
                       "'" + std::string(f[p + 1]) + "' is not a number", no, "RHS");
      }
      if (row == objective_row_) {
        // An RHS on the objective row is the objective constant, NEGATED.
        // Near-universal convention, rarely written down; getting the sign
        // wrong shifts every reported objective value.
        obj_constant_ = -*v;
        continue;
      }
      const Index cons = row_to_constraint_[static_cast<std::size_t>(row)];
      if (cons < 0) continue;
      rhs_[static_cast<std::size_t>(cons)] = *v;
    }
    return core::Status::Ok();
  }

  core::Status read_ranges(const FieldArray& f, std::size_t n, std::size_t no) {
    ensure_row_arrays();
    const std::size_t base = (rows_.find(f[0]) >= 0) ? 0 : 1;
    for (std::size_t p = base; p + 1 < n; p += 2) {
      const Index row = rows_.find(f[p]);
      if (row < 0) {
        return located(ErrorCode::UndefinedName,
                       "row '" + std::string(f[p]) + "' was never declared", no,
                       "RANGES");
      }
      const auto v = parse_real(f[p + 1]);
      if (!v) {
        return located(ErrorCode::ParseError,
                       "'" + std::string(f[p + 1]) + "' is not a number", no,
                       "RANGES");
      }
      const Index cons = row_to_constraint_[static_cast<std::size_t>(row)];
      if (cons < 0) continue;
      range_[static_cast<std::size_t>(cons)] = *v;
      has_range_[static_cast<std::size_t>(cons)] = true;
    }
    return core::Status::Ok();
  }

  core::Status read_bounds(const FieldArray& f, std::size_t n, std::size_t no) {
    if (n < 2) {
      return located(ErrorCode::ParseError, "BOUNDS entry is too short", no,
                     "BOUNDS");
    }
    // Layout is: KEY [setname] column [value]. The set name is optional.
    const std::string_view key = f[0];
    std::size_t ci = 1;
    if (n >= 3 && cols_.find(f[1]) < 0 && cols_.find(f[2]) >= 0) ci = 2;

    const Index col = cols_.find(f[ci]);
    if (col < 0) {
      // A bound on a column that never appeared in COLUMNS is legal but means
      // the column has no matrix entries. Nothing to attach it to.
      return core::Status::Ok();
    }
    const auto j = static_cast<std::size_t>(col);

    std::optional<Real> value;
    if (n > ci + 1) value = parse_real(f[ci + 1]);

    const bool needs_value =
        iequals(key, "LO") || iequals(key, "UP") || iequals(key, "FX") ||
        iequals(key, "LI") || iequals(key, "UI") || iequals(key, "SC");
    if (needs_value && !value) {
      return located(ErrorCode::ParseError,
                     "bound key '" + std::string(key) + "' requires a value", no,
                     "BOUNDS");
    }

    if (iequals(key, "LO")) {
      col_lower_[j] = *value;
      col_saw_lower_[j] = true;
    } else if (iequals(key, "UP")) {
      col_upper_[j] = *value;
      col_saw_upper_[j] = true;
      // The negative-UP quirk: an upper bound below the default lower bound of
      // zero. Readers disagree; CPLEX/Gurobi/HiGHS free the lower bound.
      // Logged, because a discrepancy against another solver should be
      // traceable rather than mysterious.
      if (*value < 0.0 && !col_saw_lower_[j]) {
        if (options_.negative_upper_implies_free_lower) col_lower_[j] = -INF;
        ++negative_upper_count_;
      }
    } else if (iequals(key, "FX")) {
      col_lower_[j] = *value;
      col_upper_[j] = *value;
      col_saw_lower_[j] = true;
      col_saw_upper_[j] = true;
    } else if (iequals(key, "FR")) {
      col_lower_[j] = -INF;
      col_upper_[j] = INF;
      col_saw_lower_[j] = true;
      col_saw_upper_[j] = true;
    } else if (iequals(key, "MI")) {
      // Modern convention leaves the upper bound alone; some historical
      // readers set it to zero.
      col_lower_[j] = -INF;
      col_saw_lower_[j] = true;
    } else if (iequals(key, "PL")) {
      col_upper_[j] = INF;
      col_saw_upper_[j] = true;
    } else if (iequals(key, "BV")) {
      col_lower_[j] = 0.0;
      col_upper_[j] = 1.0;
      col_type_[j] = VarType::Binary;
      col_saw_lower_[j] = true;
      col_saw_upper_[j] = true;
    } else if (iequals(key, "LI")) {
      col_lower_[j] = *value;
      col_type_[j] = VarType::Integer;
      col_saw_lower_[j] = true;
    } else if (iequals(key, "UI")) {
      col_upper_[j] = *value;
      col_type_[j] = VarType::Integer;
      col_saw_upper_[j] = true;
    } else if (iequals(key, "SC")) {
      col_upper_[j] = *value;
      col_type_[j] = VarType::SemiContinuous;
      saw_semicontinuous_ = true;
    } else {
      return located(ErrorCode::ParseError,
                     "unknown bound key '" + std::string(key) + "'", no, "BOUNDS");
    }
    return core::Status::Ok();
  }

  core::Status read_quad(const FieldArray& f, std::size_t n, std::size_t no,
                         bool full_matrix) {
    if (n < 3) {
      return located(ErrorCode::ParseError, "quadratic entry is too short", no,
                     "QUADOBJ");
    }
    const Index i = cols_.find(f[0]);
    const Index j = cols_.find(f[1]);
    if (i < 0 || j < 0) {
      return located(ErrorCode::UndefinedName,
                     "quadratic entry references an undeclared column", no,
                     "QUADOBJ");
    }
    const auto v = parse_real(f[2]);
    if (!v) {
      return located(ErrorCode::ParseError,
                     "'" + std::string(f[2]) + "' is not a number", no, "QUADOBJ");
    }

    // QUADOBJ gives one triangle, so an off-diagonal entry stands for both
    // (i,j) and (j,i). QMATRIX gives both explicitly. Storing full symmetric
    // either way keeps the solver's access pattern uniform.
    quad_.push_back({i, j, *v});
    if (!full_matrix && i != j) quad_.push_back({j, i, *v});
    return core::Status::Ok();
  }

  // -- between passes -----------------------------------------------------

  core::Status finalize_rows() {
    const std::size_t m = constraint_count_;
    const std::size_t n = cols_.size();
    if (n == 0) {
      return core::make_error(ErrorCode::ParseError,
                              "model declares no columns");
    }

    // rhs_/range_ were sized lazily by read_rhs; make sure they are complete.
    rhs_.resize(m, 0.0);
    range_.resize(m, 0.0);
    has_range_.resize(m, false);

    builder_.emplace(m, n);
    for (const auto& [row, col] : pending_counts_) {
      builder_->count(row, col);
    }
    pending_counts_.clear();
    pending_counts_.shrink_to_fit();
    return builder_->allocate();
  }

  // -- pass two: matrix values only ---------------------------------------

  core::Status pass_two() {
    detail::LineScanner scan(text_);
    std::string_view line;
    std::size_t no = 0;
    Section current = Section::None;

    obj_.assign(cols_.size(), 0.0);

    while (scan.next(line, no)) {
      if (detail::is_skippable(line)) continue;
      if (detail::is_section_header(line)) {
        FieldArray f{};
        const std::size_t k = detail::split_free(line, f);
        current = k > 0 ? classify_section(f[0]) : Section::Unknown;
        if (current == Section::EndData) break;
        continue;
      }
      if (current != Section::Columns) continue;

      FieldArray f{};
      const std::size_t k = split(line, f);
      if (k < 3 || is_marker_line(f, k)) continue;

      const Index col = cols_.find(f[0]);
      for (std::size_t p = 1; p + 1 < k; p += 2) {
        const Index row = rows_.find(f[p]);
        const Real v = *parse_real(f[p + 1]);
        if (row == objective_row_) {
          // Duplicates on the objective row sum, same as matrix entries.
          obj_[static_cast<std::size_t>(col)] += v;
          continue;
        }
        const Index cons = row_to_constraint_[static_cast<std::size_t>(row)];
        if (cons < 0) continue;
        builder_->insert(cons, col, v);
      }
    }
    return core::Status::Ok();
  }

  // -- assembly -----------------------------------------------------------

  Expected<Problem> assemble() {
    const std::size_t m = constraint_count_;
    const std::size_t n = cols_.size();

    Problem p;
    p.sense = sense_;
    p.obj_constant = obj_constant_;
    p.problem_name = problem_name_;
    p.objective_row_name = std::string(objective_name_);

    p.A = builder_->finish(options_.sum_duplicate_entries, 0.0);

    p.c = core::RealVector(n);
    for (std::size_t j = 0; j < n; ++j) p.c[j] = obj_[j];

    // Row bounds from type + rhs + ranges. See MPS-FORMAT-NOTES.md section 6:
    // the sign of the range matters only for E rows.
    p.row_lower = core::RealVector(m);
    p.row_upper = core::RealVector(m);
    for (std::size_t i = 0; i < row_kind_.size(); ++i) {
      const Index cons = row_to_constraint_[i];
      if (cons < 0) continue;
      const auto k = static_cast<std::size_t>(cons);
      const Real b = rhs_[k];
      Real lo = 0.0;
      Real hi = 0.0;
      switch (row_kind_[i]) {
        case RowType::Less:    lo = -INF; hi = b; break;
        case RowType::Greater: lo = b;    hi = INF; break;
        case RowType::Equal:   lo = b;    hi = b; break;
        case RowType::Free:    lo = -INF; hi = INF; break;
      }
      if (has_range_[k]) {
        const Real r = range_[k];
        const Real mag = r < 0 ? -r : r;
        switch (row_kind_[i]) {
          case RowType::Less:    lo = b - mag; hi = b; break;
          case RowType::Greater: lo = b;       hi = b + mag; break;
          case RowType::Equal:
            if (r >= 0) { lo = b; hi = b + r; }
            else        { lo = b + r; hi = b; }
            break;
          case RowType::Free: break;
        }
      }
      p.row_lower[k] = lo;
      p.row_upper[k] = hi;
    }

    p.col_lower = core::RealVector(n);
    p.col_upper = core::RealVector(n);
    p.col_type.resize(n);
    for (std::size_t j = 0; j < n; ++j) {
      p.col_lower[j] = col_lower_[j];
      p.col_upper[j] = col_upper_[j];
      p.col_type[j] = col_type_[j];
      if (col_lower_[j] > col_upper_[j]) {
        return core::make_error(
            ErrorCode::InconsistentBounds,
            "column '" + std::string(col_name_[j]) + "' has lower bound " +
                std::to_string(col_lower_[j]) + " above upper bound " +
                std::to_string(col_upper_[j]));
      }
    }

    p.row_names.reserve(m);
    for (std::size_t i = 0; i < row_name_.size(); ++i) {
      if (row_to_constraint_[i] >= 0) p.row_names.add(row_name_[i]);
    }
    p.col_names.reserve(n);
    for (const auto& name : col_name_) p.col_names.add(name);

    if (!quad_.empty()) {
      core::SparseBuilder qb(n, n);
      for (const auto& e : quad_) qb.count(e.row, e.col);
      if (auto st = qb.allocate(); !st.ok()) return st.error();
      for (const auto& e : quad_) qb.insert(e.row, e.col, e.value);
      p.Q = qb.finish(true, 0.0);
    }

    if (saw_sos_ || saw_semicontinuous_) {
      // Reported rather than silently dropped: ignoring an SOS set or a
      // semi-continuous variable changes the model, and a confidently wrong
      // answer is worse than a refusal.
      return core::make_error(
          ErrorCode::UnsupportedFeature,
          std::string(saw_sos_ ? "model contains SOS sets"
                               : "model contains semi-continuous variables") +
              ", which the solver core does not yet handle");
    }

    return p;
  }

  // -- state --------------------------------------------------------------

  struct Pair { Index row; Index col; };
  struct QuadEntry { Index row; Index col; Real value; };

  std::string_view text_;
  ReaderOptions options_;
  bool use_fixed_ = false;

  detail::SymbolTable rows_{4096};
  detail::SymbolTable cols_{4096};

  std::vector<RowType> row_kind_;
  std::vector<std::string_view> row_name_;
  std::vector<Index> row_to_constraint_;
  std::size_t constraint_count_ = 0;

  std::vector<std::string_view> col_name_;
  std::vector<VarType> col_type_;
  std::vector<Real> col_lower_;
  std::vector<Real> col_upper_;
  std::vector<bool> col_saw_lower_;
  std::vector<bool> col_saw_upper_;

  std::vector<Real> rhs_;
  std::vector<Real> range_;
  std::vector<bool> has_range_;
  std::vector<Real> obj_;

  std::vector<Pair> pending_counts_;
  std::vector<QuadEntry> quad_;
  std::optional<core::SparseBuilder> builder_;

  Index objective_row_ = -1;
  std::string_view objective_name_;
  std::string problem_name_;
  ObjSense sense_ = ObjSense::Minimize;
  Real obj_constant_ = 0.0;

  std::size_t free_rows_dropped_ = 0;
  std::size_t negative_upper_count_ = 0;
  bool saw_sos_ = false;
  bool saw_semicontinuous_ = false;
};

}  // namespace

Expected<Problem> parse_mps(std::string_view content,
                            const ReaderOptions& options) {
  MpsParser parser(content, options);
  return parser.run();
}

}  // namespace sovsolve::io
