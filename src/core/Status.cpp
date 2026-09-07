#include "sovsolve/core/Status.hpp"

#include <string>

namespace sovsolve::core {

const char* to_string(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::Ok: return "ok";
    case ErrorCode::FileNotFound: return "file not found";
    case ErrorCode::FileReadFailed: return "file read failed";
    case ErrorCode::ParseError: return "parse error";
    case ErrorCode::UnknownSection: return "unknown section";
    case ErrorCode::DuplicateName: return "duplicate name";
    case ErrorCode::UndefinedName: return "undefined name";
    case ErrorCode::DimensionMismatch: return "dimension mismatch";
    case ErrorCode::InconsistentBounds: return "inconsistent bounds";
    case ErrorCode::PrimalInfeasible: return "primal infeasible";
    case ErrorCode::Unbounded: return "unbounded";
    case ErrorCode::UnsupportedFeature: return "unsupported feature";
    case ErrorCode::IndexOverflow: return "index overflow";
    case ErrorCode::OutOfMemory: return "out of memory";
    case ErrorCode::NumericalError: return "numerical error";
    case ErrorCode::NotImplemented: return "not implemented";
  }
  return "unrecognized error";
}

std::string Error::format() const {
  // Shape: "412:15 [BOUNDS] parse error: message". The caller prepends the
  // filename, since Error is also produced when parsing an in-memory buffer.
  std::string out;
  if (line.has_value()) {
    out += std::to_string(*line);
    if (column.has_value()) {
      out += ':';
      out += std::to_string(*column);
    }
    out += ' ';
  }
  if (section.has_value() && !section->empty()) {
    out += '[';
    out += *section;
    out += "] ";
  }
  out += to_string(code);
  if (!message.empty()) {
    out += ": ";
    out += message;
  }
  return out;
}

}  // namespace sovsolve::core
