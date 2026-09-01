// Error reporting across module boundaries.
//
// `loadProblem` in the handoff docs returns `Problem` by value with no error
// channel. On a 500 MB MPS file, "parse failed" without a line number is not a
// usable diagnostic. Every fallible entry point in this layer returns
// `Expected<T>` carrying a message, and a line/column when the failure came
// from a file.
//
// C++20, so `std::expected` (C++23) is unavailable. This is a minimal
// stand-in with the same shape, so migration is mechanical if the project
// later moves to C++23.

#ifndef SOVSOLVE_CORE_STATUS_HPP
#define SOVSOLVE_CORE_STATUS_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace sovsolve::core {

enum class ErrorCode : std::uint8_t {
  Ok,
  FileNotFound,
  FileReadFailed,
  ParseError,           ///< malformed syntax
  UnknownSection,
  DuplicateName,
  UndefinedName,        ///< reference to a row/column never declared
  DimensionMismatch,
  InconsistentBounds,   ///< lower > upper after all sections applied
  /// The model is infeasible by inspection -- a verdict, not a malfunction.
  /// Raised when canonicalization finds a row whose activity is identically
  /// zero over the remaining columns while its right-hand side excludes zero.
  /// Reported here rather than handed to the solver, where it would surface as
  /// an unexplained factorization breakdown instead of an answer.
  PrimalInfeasible,
  UnsupportedFeature,   ///< e.g. SOS sets — recorded, never silently dropped
  IndexOverflow,        ///< nonzero count exceeds the Index type
  OutOfMemory,
  NumericalError,
  NotImplemented,
};

[[nodiscard]] const char* to_string(ErrorCode code) noexcept;

/// A failure, with enough context to locate it in a large input file.
struct Error {
  ErrorCode code = ErrorCode::Ok;
  std::string message;

  /// 1-based source line, when the error originated in a file.
  std::optional<std::size_t> line;
  /// 1-based column within that line.
  std::optional<std::size_t> column;
  /// Section being parsed when the failure occurred, e.g. "BOUNDS".
  std::optional<std::string> section;

  /// Human-readable rendering: "file.mps:412:15 [BOUNDS] message".
  [[nodiscard]] std::string format() const;
};

/// Convenience constructors.
[[nodiscard]] inline Error make_error(ErrorCode code, std::string message) {
  return Error{code, std::move(message), std::nullopt, std::nullopt, std::nullopt};
}

[[nodiscard]] inline Error make_parse_error(std::string message,
                                            std::size_t line,
                                            std::string section) {
  return Error{ErrorCode::ParseError, std::move(message), line, std::nullopt,
               std::move(section)};
}

// --------------------------------------------------------------------------
// Expected<T>
// --------------------------------------------------------------------------

/// Either a value or an `Error`. Modelled on `std::expected` (C++23).
///
/// Exceptions are not thrown across module boundaries; failures are values.
template <typename T>
class Expected {
 public:
  Expected(T value) : storage_(std::move(value)) {}          // NOLINT(*-explicit-*)
  Expected(Error error) : storage_(std::move(error)) {}      // NOLINT(*-explicit-*)

  [[nodiscard]] bool has_value() const noexcept {
    return std::holds_alternative<T>(storage_);
  }
  explicit operator bool() const noexcept { return has_value(); }

  [[nodiscard]] T& value() & { return std::get<T>(storage_); }
  [[nodiscard]] const T& value() const& { return std::get<T>(storage_); }
  [[nodiscard]] T&& value() && { return std::get<T>(std::move(storage_)); }

  [[nodiscard]] const Error& error() const& { return std::get<Error>(storage_); }
  [[nodiscard]] Error&& error() && { return std::get<Error>(std::move(storage_)); }

  [[nodiscard]] T* operator->() { return &std::get<T>(storage_); }
  [[nodiscard]] const T* operator->() const { return &std::get<T>(storage_); }

  [[nodiscard]] T& operator*() & { return std::get<T>(storage_); }
  [[nodiscard]] const T& operator*() const& { return std::get<T>(storage_); }

 private:
  std::variant<T, Error> storage_;
};

/// Void-returning fallible operation.
class Status {
 public:
  Status() = default;
  Status(Error error) : error_(std::move(error)) {}  // NOLINT(*-explicit-*)

  [[nodiscard]] bool ok() const noexcept { return !error_.has_value(); }
  explicit operator bool() const noexcept { return ok(); }

  [[nodiscard]] const Error& error() const { return *error_; }

  static Status Ok() { return Status{}; }

 private:
  std::optional<Error> error_;
};

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_STATUS_HPP
