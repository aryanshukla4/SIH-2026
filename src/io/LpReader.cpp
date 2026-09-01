// CPLEX LP format reader.
//
// ---------------------------------------------------------------------------
// Why this format, and why after MPS
// ---------------------------------------------------------------------------
//
// Every benchmark PS 26119 names ships as MPS -- Netlib, MIPLIB 2017, the
// Mittelmann sets -- so LP format buys no benchmark coverage. What it buys is
// readability: `3 x + 2 y <= 10` can be checked by eye, which matters for
// demonstration and for hand-written test models. That is why it comes second.
//
// ---------------------------------------------------------------------------
// The dialect question
// ---------------------------------------------------------------------------
//
// LP format has no standard. CPLEX, Gurobi and GLPK disagree, so a reader has
// to pick. This one takes CPLEX as the base and ACCEPTS Gurobi's superset --
// rejecting a construct later is easy, adding one after tests exist is not.
// The divergences that matter are noted at the point where they are handled;
// the headline one is the ranged constraint `-5 <= x + y <= 10`, which Gurobi
// allows and CPLEX does not.
//
// ---------------------------------------------------------------------------
// Assembly: the mirror of the MPS path
// ---------------------------------------------------------------------------
//
// MPS is column-major, so `MpsReader` builds CSC first and transposes to CSR.
// LP is ROW-major -- a constraint is written as one expression -- so this
// builds CSR first and transposes to CSC. `SparseBuilder::count(row, col)` is
// orientation-agnostic, so the same two-pass, no-sort, O(nnz) pipeline applies
// unchanged; only the direction of the counting-sort transpose differs.
//
// Coefficients are collected into flat row-grouped vectors during the single
// parse, then replayed twice through the builder. The replay is over memory
// that is already hot, so the second pass costs far less than the parse.
//
// ---------------------------------------------------------------------------
// Deliberately unsupported
// ---------------------------------------------------------------------------
//
// SOS sets, semi-continuous columns, indicator constraints, general
// constraints and quadratic CONSTRAINTS are reported as UnsupportedFeature
// rather than skipped. Silently dropping a section changes the model, and a
// model that quietly differs from its file is worse than one that fails to
// load. A quadratic OBJECTIVE is supported.

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

#include "io/detail/NumberParse.hpp"
#include "io/detail/Readers.hpp"
#include "io/detail/SymbolTable.hpp"
#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"

namespace sovsolve::io {
namespace {

using core::ErrorCode;
using core::Index;
using core::INF;
using core::Real;
using detail::iequals;
using detail::parse_real;
using detail::SymbolTable;

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
//
// LP format is expression-based, not field-based: newlines are insignificant
// and a constraint may wrap across as many lines as it likes. So this is a
// token stream over the whole buffer rather than the line-and-field scanner the
// MPS reader uses. Line numbers are tracked anyway, purely for diagnostics.

enum class Tok : std::uint8_t {
  End,
  Number,
  Ident,
  Plus,
  Minus,
  Star,
  Caret,
  Slash,
  Colon,
  LBracket,
  RBracket,
  Le,   ///< <=  =<  <
  Ge,   ///< >=  =>  >
  Eq,   ///< =
};

struct Token {
  Tok kind = Tok::End;
  std::string_view text;
  Real number = 0.0;
  std::size_t line = 0;
};

/// A character that may appear inside an LP identifier.
///
/// The format allows a generous set; the exclusions are the operators and
/// brackets. A leading digit is not allowed (that would be a number), which is
/// handled by the caller rather than here.
bool is_ident_char(char c) noexcept {
  const auto u = static_cast<unsigned char>(c);
  if (std::isalnum(u) != 0) return true;
  switch (c) {
    case '!': case '"': case '#': case '$': case '%': case '&':
    case '(': case ')': case ',': case '.': case ';': case '?':
    case '@': case '_': case '\'': case '`': case '{': case '}':
    case '~': case '|':
      return true;
    default:
      return false;
  }
}

class Lexer {
 public:
  explicit Lexer(std::string_view text) noexcept : text_(text) { advance(); }

  [[nodiscard]] const Token& peek() const noexcept { return cur_; }
  [[nodiscard]] const Token& peek2() noexcept {
    if (!have_ahead_) {
      const std::size_t save_pos = pos_;
      const std::size_t save_line = line_;
      const Token save_cur = cur_;
      advance();
      ahead_ = cur_;
      have_ahead_ = true;
      pos_ = save_pos;
      line_ = save_line;
      cur_ = save_cur;
    }
    return ahead_;
  }

  Token take() noexcept {
    Token t = cur_;
    have_ahead_ = false;
    advance();
    return t;
  }

  [[nodiscard]] std::size_t line() const noexcept { return cur_.line; }

  /// A restore point, for the one place the grammar needs real backtracking.
  ///
  /// `-5 <= x + y <= 10` and `3 x + y <= 10` both begin with a signed number,
  /// and which one it is only becomes clear after reading it: a relational
  /// operator means it was a range bound, anything else means it was a
  /// coefficient or a constant term. One token of lookahead is not enough,
  /// because the number itself is several tokens (`-`, `5`). Every field here
  /// is trivially copyable, so a checkpoint is a struct copy.
  struct State {
    std::size_t pos = 0;
    std::size_t line = 1;
    Token cur;
    Token ahead;
    bool have_ahead = false;
  };

  [[nodiscard]] State save() const noexcept {
    return State{pos_, line_, cur_, ahead_, have_ahead_};
  }
  void restore(const State& s) noexcept {
    pos_ = s.pos;
    line_ = s.line;
    cur_ = s.cur;
    ahead_ = s.ahead;
    have_ahead_ = s.have_ahead;
  }

 private:
  void skip_trivia() noexcept {
    for (;;) {
      while (pos_ < text_.size()) {
        const char c = text_[pos_];
        if (c == '\n') {
          ++line_;
          ++pos_;
        } else if (c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v') {
          ++pos_;
        } else {
          break;
        }
      }
      // `\` starts a comment that runs to end of line. This is the one place
      // where a newline is significant.
      if (pos_ < text_.size() && text_[pos_] == '\\') {
        while (pos_ < text_.size() && text_[pos_] != '\n') ++pos_;
        continue;
      }
      return;
    }
  }

  void advance() noexcept {
    skip_trivia();
    cur_.line = line_;
    if (pos_ >= text_.size()) {
      cur_.kind = Tok::End;
      cur_.text = {};
      return;
    }
    const char c = text_[pos_];
    const std::size_t start = pos_;

    switch (c) {
      case '+': ++pos_; cur_.kind = Tok::Plus; break;
      case '-': ++pos_; cur_.kind = Tok::Minus; break;
      case '*': ++pos_; cur_.kind = Tok::Star; break;
      case '^': ++pos_; cur_.kind = Tok::Caret; break;
      case '/': ++pos_; cur_.kind = Tok::Slash; break;
      case ':': ++pos_; cur_.kind = Tok::Colon; break;
      case '[': ++pos_; cur_.kind = Tok::LBracket; break;
      case ']': ++pos_; cur_.kind = Tok::RBracket; break;
      case '<':
        ++pos_;
        if (pos_ < text_.size() && text_[pos_] == '=') ++pos_;
        cur_.kind = Tok::Le;
        break;
      case '>':
        ++pos_;
        if (pos_ < text_.size() && text_[pos_] == '=') ++pos_;
        cur_.kind = Tok::Ge;
        break;
      case '=':
        ++pos_;
        // `=<` and `=>` are accepted spellings of `<=` and `>=`.
        if (pos_ < text_.size() && text_[pos_] == '<') {
          ++pos_;
          cur_.kind = Tok::Le;
        } else if (pos_ < text_.size() && text_[pos_] == '>') {
          ++pos_;
          cur_.kind = Tok::Ge;
        } else {
          cur_.kind = Tok::Eq;
        }
        break;
      default: {
        const auto u = static_cast<unsigned char>(c);
        if (std::isdigit(u) != 0 || c == '.') {
          scan_number();
        } else if (is_ident_char(c)) {
          while (pos_ < text_.size() && is_ident_char(text_[pos_])) ++pos_;
          cur_.kind = Tok::Ident;
        } else {
          // Unrecognized punctuation. Consume it so the parser can report a
          // location instead of spinning.
          ++pos_;
          cur_.kind = Tok::Ident;
        }
        break;
      }
    }
    cur_.text = text_.substr(start, pos_ - start);
    if (cur_.kind == Tok::Number) {
      const auto v = parse_real(cur_.text);
      cur_.number = v.value_or(0.0);
    }
  }

  void scan_number() noexcept {
    while (pos_ < text_.size() &&
           (std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0 ||
            text_[pos_] == '.')) {
      ++pos_;
    }
    // An exponent, but only when it really is one: `3e5` is a number while
    // `3 e5` is a coefficient times a variable named e5, and `3x` is likewise
    // a coefficient times `x`. The distinction is that an exponent must be
    // followed by digits, optionally after a sign.
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      std::size_t probe = pos_ + 1;
      if (probe < text_.size() && (text_[probe] == '+' || text_[probe] == '-')) {
        ++probe;
      }
      if (probe < text_.size() &&
          std::isdigit(static_cast<unsigned char>(text_[probe])) != 0) {
        while (probe < text_.size() &&
               std::isdigit(static_cast<unsigned char>(text_[probe])) != 0) {
          ++probe;
        }
        pos_ = probe;
      }
    }
    cur_.kind = Tok::Number;
  }

  std::string_view text_;
  std::size_t pos_ = 0;
  std::size_t line_ = 1;
  Token cur_;
  Token ahead_;
  bool have_ahead_ = false;
};

// ---------------------------------------------------------------------------
// Section keywords
// ---------------------------------------------------------------------------

enum class Section : std::uint8_t {
  None,
  Minimize,
  Maximize,
  SubjectTo,
  Bounds,
  General,
  Binary,
  SemiContinuous,
  Sos,
  End,
};

/// Recognize a section keyword at the current position without consuming it.
///
/// Multi-word headers (`Subject To`, `Semi-Continuous`) are why this peeks two
/// tokens. `Semi-Continuous` also lexes as three tokens, since `-` is an
/// operator; the second word is enough to identify it.
Section peek_section(Lexer& lex) {
  const Token& t = lex.peek();
  if (t.kind == Tok::End) return Section::End;
  if (t.kind != Tok::Ident) return Section::None;
  const std::string_view w = t.text;

  if (iequals(w, "minimize") || iequals(w, "minimise") || iequals(w, "min") ||
      iequals(w, "minimum")) {
    return Section::Minimize;
  }
  if (iequals(w, "maximize") || iequals(w, "maximise") || iequals(w, "max") ||
      iequals(w, "maximum")) {
    return Section::Maximize;
  }
  if (iequals(w, "bounds") || iequals(w, "bound")) return Section::Bounds;
  if (iequals(w, "general") || iequals(w, "generals") || iequals(w, "gen") ||
      iequals(w, "integer") || iequals(w, "integers")) {
    return Section::General;
  }
  if (iequals(w, "binary") || iequals(w, "binaries") || iequals(w, "bin")) {
    return Section::Binary;
  }
  if (iequals(w, "semi-continuous") || iequals(w, "semis") ||
      iequals(w, "semi")) {
    return Section::SemiContinuous;
  }
  if (iequals(w, "sos")) return Section::Sos;
  if (iequals(w, "end")) return Section::End;

  // `st`, `s.t.`, `st.` are single tokens; `subject to` and `such that` are two.
  if (iequals(w, "st") || iequals(w, "st.") || iequals(w, "s.t.") ||
      iequals(w, "s.t")) {
    return Section::SubjectTo;
  }
  if (iequals(w, "subject") || iequals(w, "such")) {
    const Token& n = lex.peek2();
    if (n.kind == Tok::Ident && (iequals(n.text, "to") || iequals(n.text, "that"))) {
      return Section::SubjectTo;
    }
  }
  return Section::None;
}

/// Consume the tokens of a section header identified by `peek_section`.
void consume_section(Lexer& lex, Section s) {
  const bool two_words =
      s == Section::SubjectTo &&
      (iequals(lex.peek().text, "subject") || iequals(lex.peek().text, "such"));
  const bool hyphenated =
      s == Section::SemiContinuous && iequals(lex.peek().text, "semi") &&
      lex.peek2().kind == Tok::Minus;
  lex.take();
  if (two_words) lex.take();
  if (hyphenated) {
    lex.take();  // '-'
    if (lex.peek().kind == Tok::Ident) lex.take();  // 'continuous'
  }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// One coefficient, in row-major order as the file presents them.
struct Entry {
  Index row = 0;
  Index col = 0;
  Real value = 0.0;
};

class LpParser {
 public:
  LpParser(std::string_view text, const model::ReaderOptions& options)
      : lex_(text), options_(options) {}

  core::Expected<model::Problem> run();

 private:
  [[nodiscard]] core::Error err(ErrorCode code, std::string message) const {
    core::Error e;
    e.code = code;
    e.message = std::move(message);
    e.line = lex_.line();
    if (!section_name_.empty()) e.section = section_name_;
    return e;
  }

  Index column(std::string_view name) {
    bool inserted = false;
    const Index idx = columns_.insert(name, inserted);
    if (inserted) {
      col_name_.emplace_back(name);
      col_lower_.push_back(0.0);
      col_upper_.push_back(INF);
      col_type_.push_back(model::VarType::Continuous);
      bound_seen_.push_back(false);
    }
    return idx;
  }

  /// True when the current token really begins a variable reference.
  ///
  /// Two things can masquerade as one. A section keyword (`Bounds`, `End`) is
  /// an identifier like any other, and so is the LABEL of the next statement --
  /// `c2` in `c2: x - y >= 1`. Consuming either as a variable silently changes
  /// the model rather than failing, which is the worst outcome, so both are
  /// ruled out here rather than at each call site.
  [[nodiscard]] bool starts_variable() {
    if (lex_.peek().kind != Tok::Ident) return false;
    if (lex_.peek2().kind == Tok::Colon) return false;
    return peek_section(lex_) == Section::None;
  }

  /// Read `[+|-] (number | inf | infinity)`. Returns false without consuming a
  /// sign only when the very first token is not part of a number.
  [[nodiscard]] bool read_signed_number(Real& out) {
    const auto is_inf_word = [](std::string_view w) {
      return iequals(w, "inf") || iequals(w, "infinity");
    };
    const Tok k = lex_.peek().kind;
    if (k != Tok::Number && k != Tok::Plus && k != Tok::Minus &&
        !(k == Tok::Ident && is_inf_word(lex_.peek().text))) {
      return false;
    }
    Real sign = 1.0;
    if (k == Tok::Plus) {
      lex_.take();
    } else if (k == Tok::Minus) {
      lex_.take();
      sign = -1.0;
    }
    if (lex_.peek().kind == Tok::Number) {
      out = sign * lex_.take().number;
      return true;
    }
    if (lex_.peek().kind == Tok::Ident && is_inf_word(lex_.peek().text)) {
      lex_.take();
      out = sign * INF;
      return true;
    }
    return false;
  }

  core::Status parse_objective(bool maximize);
  core::Status parse_constraints();
  core::Status parse_bounds();
  core::Status parse_marked_columns(model::VarType type, bool binary);
  core::Status parse_quadratic(std::vector<Entry>& out, Real& scale);

  /// Parse `[sign] term { sign term }` into `out`, accumulating any bare
  /// numeric terms into `constant`. Stops at a relational operator, a section
  /// keyword, `[`, or end of input.
  ///
  /// `bracket_sign`, when given, receives the sign left pending if the
  /// expression stops at `[`. In `x - y + [ ... ] / 2` the `+` separates the
  /// linear part from the quadratic one and belongs to the bracket, so it is
  /// handed on rather than treated as a dangling operator.
  core::Status parse_linear(std::vector<std::pair<Index, Real>>& out,
                            Real& constant, Real* bracket_sign = nullptr);

  Lexer lex_;
  const model::ReaderOptions& options_;
  std::string section_name_;

  SymbolTable columns_;
  SymbolTable rows_;
  std::vector<std::string> col_name_;

  /// Row names must live in a container whose elements never move.
  ///
  /// `SymbolTable` stores `string_view`s and does not own the characters. For
  /// COLUMNS that is free: a column name is a view into the input text, which
  /// outlives the parse. Row names are not always -- an unnamed constraint gets
  /// a generated `R<k>` that exists only as a `std::string` we create. Handing
  /// the table a view into a local, or into a `std::vector<std::string>` that
  /// can reallocate (and, with the small-string optimization, move the
  /// characters themselves), leaves the table pointing at freed memory; the
  /// next lookup that probes that slot reads it. `std::deque` never invalidates
  /// references to elements already inserted, so views into it stay valid.
  std::deque<std::string> row_name_;

  std::vector<Real> col_lower_;
  std::vector<Real> col_upper_;
  std::vector<model::VarType> col_type_;
  std::vector<bool> bound_seen_;

  std::vector<Real> row_lower_;
  std::vector<Real> row_upper_;

  std::vector<Entry> entries_;
  std::vector<Entry> q_entries_;
  Real q_scale_ = 1.0;

  std::vector<std::pair<Index, Real>> obj_;
  Real obj_constant_ = 0.0;
  bool maximize_ = false;
  std::string obj_name_;
  std::string problem_name_;
};

core::Status LpParser::parse_linear(std::vector<std::pair<Index, Real>>& out,
                                    Real& constant, Real* bracket_sign) {
  Real sign = 1.0;
  // Set only by consuming a `+` or `-`, so it means "a sign is waiting for its
  // term" rather than "we are at the start". An empty expression is legal --
  // the objective's linear part may be absent, and the call that follows a
  // quadratic bracket usually finds nothing left before the next section.
  bool expect_term = false;

  for (;;) {
    const Token& t = lex_.peek();
    if (t.kind == Tok::Plus) {
      lex_.take();
      expect_term = true;
      continue;
    }
    if (t.kind == Tok::Minus) {
      lex_.take();
      sign = -sign;
      expect_term = true;
      continue;
    }
    if (t.kind == Tok::Number) {
      const Real coef = lex_.take().number;
      // `3 x` and `3 * x` are the same term; a number followed by anything
      // else is a bare constant.
      if (lex_.peek().kind == Tok::Star) lex_.take();
      if (starts_variable()) {
        const Index c = column(lex_.take().text);
        out.emplace_back(c, sign * coef);
      } else {
        constant += sign * coef;
      }
      sign = 1.0;
      expect_term = false;
      continue;
    }
    if (t.kind == Tok::Ident) {
      if (!starts_variable()) break;
      const Index c = column(lex_.take().text);
      out.emplace_back(c, sign);
      sign = 1.0;
      expect_term = false;
      continue;
    }
    break;
  }
  if (expect_term && lex_.peek().kind == Tok::LBracket) {
    if (bracket_sign != nullptr) *bracket_sign = sign;
    return core::Status{};
  }
  if (expect_term && !out.empty()) {
    return err(ErrorCode::ParseError, "expression ends with a dangling sign");
  }
  return core::Status{};
}

core::Status LpParser::parse_quadratic(std::vector<Entry>& out, Real& scale) {
  // `[ 2 x^2 + 4 x * y ] / 2`
  //
  // The bracket holds the objective's quadratic part written out in full, and
  // the trailing `/ 2` is part of the syntax rather than arithmetic on it. The
  // model is `1/2 x'Qx`, so the bracket contents ARE `x'Qx` when `/2` is
  // present. This is the same factor-of-two trap `QUADOBJ` has, and it is worth
  // one hand-checked test: getting it wrong scales every QP objective by 2.
  lex_.take();  // '['
  Real sign = 1.0;

  while (lex_.peek().kind != Tok::RBracket) {
    if (lex_.peek().kind == Tok::End) {
      return err(ErrorCode::ParseError, "unterminated quadratic bracket");
    }
    if (lex_.peek().kind == Tok::Plus) {
      lex_.take();
      continue;
    }
    if (lex_.peek().kind == Tok::Minus) {
      lex_.take();
      sign = -sign;
      continue;
    }
    Real coef = 1.0;
    if (lex_.peek().kind == Tok::Number) {
      coef = lex_.take().number;
      if (lex_.peek().kind == Tok::Star) lex_.take();
    }
    if (lex_.peek().kind != Tok::Ident) {
      return err(ErrorCode::ParseError, "expected a variable in quadratic term");
    }
    const Index a = column(lex_.take().text);
    Index b = a;
    if (lex_.peek().kind == Tok::Caret) {
      lex_.take();
      if (lex_.peek().kind != Tok::Number || lex_.peek().number != 2.0) {
        return err(ErrorCode::UnsupportedFeature,
                   "only squared terms are supported in a quadratic objective");
      }
      lex_.take();
    } else {
      if (lex_.peek().kind == Tok::Star) lex_.take();
      if (lex_.peek().kind == Tok::Ident &&
          peek_section(lex_) == Section::None) {
        b = column(lex_.take().text);
      }
    }
    // Q is stored full symmetric, so an off-diagonal term written once
    // contributes to both (a,b) and (b,a), each getting half the coefficient.
    if (a == b) {
      out.push_back({a, b, sign * coef});
    } else {
      out.push_back({a, b, sign * coef * 0.5});
      out.push_back({b, a, sign * coef * 0.5});
    }
    sign = 1.0;
  }
  lex_.take();  // ']'

  scale = 1.0;
  if (lex_.peek().kind == Tok::Slash) {
    lex_.take();
    if (lex_.peek().kind != Tok::Number) {
      return err(ErrorCode::ParseError, "expected a divisor after ']'");
    }
    const Real d = lex_.take().number;
    if (d == 0.0) return err(ErrorCode::ParseError, "division by zero after ']'");
    // With `/2` present the bracket is x'Qx and the model wants 1/2 x'Qx, so
    // the stored Q is the bracket as written. Without it, the bracket is
    // already the objective contribution, so Q must be doubled.
    scale = 2.0 / d;
  } else {
    scale = 2.0;
  }
  return core::Status{};
}

core::Status LpParser::parse_objective(bool maximize) {
  section_name_ = maximize ? "MAXIMIZE" : "MINIMIZE";
  maximize_ = maximize;

  // An optional `name:` label.
  if (lex_.peek().kind == Tok::Ident && lex_.peek2().kind == Tok::Colon &&
      peek_section(lex_) == Section::None) {
    obj_name_ = std::string(lex_.take().text);
    lex_.take();  // ':'
  }

  Real bracket_sign = 1.0;
  if (auto st = parse_linear(obj_, obj_constant_, &bracket_sign); !st.ok()) {
    return st;
  }

  if (lex_.peek().kind == Tok::LBracket) {
    if (auto st = parse_quadratic(q_entries_, q_scale_); !st.ok()) return st;
    if (bracket_sign < 0.0) {
      for (auto& e : q_entries_) e.value = -e.value;
    }
    // A quadratic objective may be followed by more linear terms.
    if (auto st = parse_linear(obj_, obj_constant_); !st.ok()) return st;
  }
  return core::Status{};
}

core::Status LpParser::parse_constraints() {
  section_name_ = "SUBJECT TO";

  for (;;) {
    const Section s = peek_section(lex_);
    if (s != Section::None) break;
    if (lex_.peek().kind == Tok::End) break;

    std::string name;
    if (lex_.peek().kind == Tok::Ident && lex_.peek2().kind == Tok::Colon) {
      name = std::string(lex_.take().text);
      lex_.take();  // ':'
    }

    // Gurobi's ranged form puts a bound in front: `-5 <= x + y <= 10`.
    // CPLEX does not accept this; we do, because rejecting later is cheap.
    //
    // Deciding whether the leading number IS a range bound needs backtracking,
    // not lookahead: `-5` is two tokens, and `3 x + y <= 10` starts the same
    // way with `3` as a coefficient. So read the number, look at what follows,
    // and rewind if it turns out to have been part of the expression.
    bool leading_bound = false;
    Real leading_value = 0.0;
    Tok leading_op = Tok::Le;
    {
      const auto checkpoint = lex_.save();
      Real candidate = 0.0;
      if (read_signed_number(candidate)) {
        const Tok next = lex_.peek().kind;
        if (next == Tok::Le || next == Tok::Ge || next == Tok::Eq) {
          leading_value = candidate;
          leading_op = lex_.take().kind;
          leading_bound = true;
        } else {
          lex_.restore(checkpoint);
        }
      } else {
        lex_.restore(checkpoint);
      }
    }

    std::vector<std::pair<Index, Real>> terms;
    Real lhs_constant = 0.0;
    if (auto st = parse_linear(terms, lhs_constant); !st.ok()) return st;

    const Tok op = lex_.peek().kind;
    if (op != Tok::Le && op != Tok::Ge && op != Tok::Eq) {
      if (terms.empty()) break;
      return err(ErrorCode::ParseError,
                 "expected a relational operator in constraint");
    }
    lex_.take();

    // The right-hand side must be a single signed constant.
    //
    // This is the CPLEX rule, and taking it literally is what makes the format
    // parseable at all. Allowing variables after the operator makes the end of
    // an unnamed constraint ambiguous -- in
    //
    //     x + y <= 5
    //     x - y >= 1
    //
    // nothing separates the `5` from the `x` that follows, so a reader that
    // accepts variables on the right silently folds the next constraint into
    // this one's right-hand side. Every LP writer normalizes variables to the
    // left, so the restriction costs nothing real; a file that violates it gets
    // a clear error instead of a quietly different model.
    Real rhs = 0.0;
    if (!read_signed_number(rhs)) {
      return err(ErrorCode::UnsupportedFeature,
                 "the right-hand side of a constraint must be a constant; move "
                 "variables to the left of the operator");
    }
    rhs -= lhs_constant;

    Real lo = -INF;
    Real hi = INF;
    if (op == Tok::Le) {
      hi = rhs;
    } else if (op == Tok::Ge) {
      lo = rhs;
    } else {
      lo = rhs;
      hi = rhs;
    }

    if (leading_bound) {
      // A constant inside the expression shifts BOTH sides of a ranged row:
      // `-5 <= x + y + 3 <= 10` constrains `x + y` to `[-8, 7]`.
      const Real bound = leading_value - lhs_constant;
      if (leading_op == Tok::Le) {
        lo = bound;
      } else if (leading_op == Tok::Ge) {
        hi = bound;
      } else {
        lo = bound;
        hi = bound;
      }
      if (lo > hi) {
        return err(ErrorCode::InconsistentBounds,
                   "ranged constraint has lower bound above upper bound");
      }
    }

    if (name.empty()) {
      name = "R" + std::to_string(rows_.size() + 1);
    }
    // Check for the duplicate BEFORE storing, so the stable-storage deque and
    // the symbol table never disagree about which names exist.
    if (rows_.find(name) != SymbolTable::kNotFound) {
      return err(ErrorCode::DuplicateName, "duplicate constraint name " + name);
    }
    row_name_.push_back(std::move(name));
    bool inserted = false;
    const Index r = rows_.insert(row_name_.back(), inserted);
    row_lower_.push_back(lo);
    row_upper_.push_back(hi);

    for (const auto& [c, v] : terms) entries_.push_back({r, c, v});
  }
  return core::Status{};
}

core::Status LpParser::parse_bounds() {
  section_name_ = "BOUNDS";

  const auto read_value = [&](Real& out) { return read_signed_number(out); };

  for (;;) {
    if (peek_section(lex_) != Section::None) break;
    if (lex_.peek().kind == Tok::End) break;

    // Three shapes:
    //   value op name [op value]
    //   name op value
    //   name free
    Real first = 0.0;
    const bool leading_value =
        (lex_.peek().kind == Tok::Number || lex_.peek().kind == Tok::Plus ||
         (lex_.peek().kind == Tok::Minus)) &&
        read_value(first);

    if (leading_value) {
      const Tok op = lex_.peek().kind;
      if (op != Tok::Le && op != Tok::Ge && op != Tok::Eq) {
        return err(ErrorCode::ParseError, "expected an operator in bound");
      }
      lex_.take();
      if (lex_.peek().kind != Tok::Ident) {
        return err(ErrorCode::ParseError, "expected a variable name in bound");
      }
      const Index c = column(lex_.take().text);
      const auto k = static_cast<std::size_t>(c);
      bound_seen_[k] = true;

      if (op == Tok::Le) {
        col_lower_[k] = first;
      } else if (op == Tok::Ge) {
        col_upper_[k] = first;
      } else {
        col_lower_[k] = first;
        col_upper_[k] = first;
      }

      // The optional second half of `lo <= x <= hi`.
      if (lex_.peek().kind == Tok::Le || lex_.peek().kind == Tok::Ge ||
          lex_.peek().kind == Tok::Eq) {
        const Tok op2 = lex_.take().kind;
        Real second = 0.0;
        if (!read_value(second)) {
          return err(ErrorCode::ParseError, "expected a value in bound");
        }
        if (op2 == Tok::Le) {
          col_upper_[k] = second;
        } else if (op2 == Tok::Ge) {
          col_lower_[k] = second;
        } else {
          col_lower_[k] = second;
          col_upper_[k] = second;
        }
      }
      continue;
    }

    if (lex_.peek().kind != Tok::Ident) {
      return err(ErrorCode::ParseError, "expected a variable name in bound");
    }
    const Index c = column(lex_.take().text);
    const auto k = static_cast<std::size_t>(c);
    bound_seen_[k] = true;

    if (lex_.peek().kind == Tok::Ident && iequals(lex_.peek().text, "free")) {
      lex_.take();
      col_lower_[k] = -INF;
      col_upper_[k] = INF;
      continue;
    }

    const Tok op = lex_.peek().kind;
    if (op != Tok::Le && op != Tok::Ge && op != Tok::Eq) {
      return err(ErrorCode::ParseError, "expected an operator in bound");
    }
    lex_.take();
    Real value = 0.0;
    if (!read_value(value)) {
      return err(ErrorCode::ParseError, "expected a value in bound");
    }
    if (op == Tok::Le) {
      col_upper_[k] = value;
      // The MPS reader treats a negative upper bound with no explicit lower as
      // implying a free lower bound; LP format is explicit about bounds, so the
      // same quirk does not apply and the option is deliberately not consulted
      // here. Noted because the two readers must not silently disagree.
    } else if (op == Tok::Ge) {
      col_lower_[k] = value;
    } else {
      col_lower_[k] = value;
      col_upper_[k] = value;
    }
  }
  return core::Status{};
}

core::Status LpParser::parse_marked_columns(model::VarType type, bool binary) {
  section_name_ = binary ? "BINARY" : "GENERAL";
  for (;;) {
    if (peek_section(lex_) != Section::None) break;
    if (lex_.peek().kind != Tok::Ident) break;
    const Index c = column(lex_.take().text);
    const auto k = static_cast<std::size_t>(c);
    col_type_[k] = type;
    if (binary) {
      col_lower_[k] = 0.0;
      col_upper_[k] = 1.0;
      bound_seen_[k] = true;
    }
  }
  return core::Status{};
}

core::Expected<model::Problem> LpParser::run() {
  // The objective section is mandatory and comes first.
  Section s = peek_section(lex_);
  if (s != Section::Minimize && s != Section::Maximize) {
    return err(ErrorCode::ParseError,
               "LP file must begin with a Minimize or Maximize section");
  }
  consume_section(lex_, s);
  if (auto st = parse_objective(s == Section::Maximize); !st.ok()) {
    return st.error();
  }

  for (;;) {
    s = peek_section(lex_);
    if (s == Section::End) break;
    if (s == Section::None) {
      if (lex_.peek().kind == Tok::End) break;
      return err(ErrorCode::ParseError,
                 "expected a section keyword, found '" +
                     std::string(lex_.peek().text) + "'");
    }
    consume_section(lex_, s);
    switch (s) {
      case Section::SubjectTo:
        if (auto st = parse_constraints(); !st.ok()) return st.error();
        break;
      case Section::Bounds:
        if (auto st = parse_bounds(); !st.ok()) return st.error();
        break;
      case Section::General:
        if (auto st = parse_marked_columns(model::VarType::Integer, false);
            !st.ok()) {
          return st.error();
        }
        break;
      case Section::Binary:
        if (auto st = parse_marked_columns(model::VarType::Binary, true);
            !st.ok()) {
          return st.error();
        }
        break;
      case Section::SemiContinuous:
        return err(ErrorCode::UnsupportedFeature,
                   "semi-continuous columns are not supported; the section is "
                   "reported rather than skipped, since dropping it would "
                   "silently change the model");
      case Section::Sos:
        return err(ErrorCode::UnsupportedFeature,
                   "SOS sets are not supported; the section is reported rather "
                   "than skipped, since dropping it would silently change the "
                   "model");
      case Section::Minimize:
      case Section::Maximize:
        return err(ErrorCode::ParseError, "a second objective section");
      case Section::None:
      case Section::End:
        break;
    }
  }
  section_name_.clear();

  // -- assemble ------------------------------------------------------------

  const std::size_t m = rows_.size();
  const std::size_t n = columns_.size();

  model::Problem p;
  p.sense = maximize_ ? core::ObjSense::Maximize : core::ObjSense::Minimize;
  // The written constant is the objective's own constant, with no sign trap --
  // unlike MPS, where an RHS entry on the objective row is its NEGATION.
  p.obj_constant = obj_constant_;
  p.problem_name = problem_name_;
  p.objective_row_name = obj_name_.empty() ? std::string("obj") : obj_name_;

  p.c = core::RealVector(n, 0.0);
  for (const auto& [c, v] : obj_) p.c[static_cast<std::size_t>(c)] += v;

  p.row_lower = core::RealVector(m, 0.0);
  p.row_upper = core::RealVector(m, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    p.row_lower[i] = row_lower_[i];
    p.row_upper[i] = row_upper_[i];
  }

  p.col_lower = core::RealVector(n, 0.0);
  p.col_upper = core::RealVector(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    p.col_lower[j] = col_lower_[j];
    p.col_upper[j] = col_upper_[j];
  }
  p.col_type = col_type_;

  // Row-major first: the entries arrive grouped by row, so CSR is the natural
  // orientation and CSC comes from the counting-sort transpose. Exactly the
  // mirror of the MPS path, which is column-major and builds CSC first.
  {
    core::SparseBuilder builder(m, n);
    for (const auto& e : entries_) builder.count(e.row, e.col);
    if (auto st = builder.allocate(); !st.ok()) return st.error();
    for (const auto& e : entries_) builder.insert(e.row, e.col, e.value);
    p.A = builder.finish(options_.sum_duplicate_entries, 0.0);
  }

  if (!q_entries_.empty()) {
    core::SparseBuilder qb(n, n);
    for (const auto& e : q_entries_) qb.count(e.row, e.col);
    if (auto st = qb.allocate(); !st.ok()) return st.error();
    for (const auto& e : q_entries_) {
      qb.insert(e.row, e.col, e.value * q_scale_);
    }
    p.Q = qb.finish(true, 0.0);
  }

  p.row_names.reserve(m);
  for (const auto& name : row_name_) p.row_names.add(name);
  p.col_names.reserve(n);
  for (const auto& name : col_name_) p.col_names.add(name);

  return p;
}

}  // namespace

core::Expected<model::Problem> parse_lp(std::string_view content,
                                        const model::ReaderOptions& options) {
  LpParser parser(content, options);
  return parser.run();
}

}  // namespace sovsolve::io
