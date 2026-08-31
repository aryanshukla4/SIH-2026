// Public loading entry point: format dispatch and file access.

#include "sovsolve/io/Load.hpp"

#include <string>
#include <string_view>

#include "io/detail/MappedFile.hpp"
#include "io/detail/Readers.hpp"
#include "sovsolve/core/Status.hpp"

namespace sovsolve::io {
namespace {

using core::ErrorCode;

bool ends_with(std::string_view s, std::string_view suffix) noexcept {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Format from the filename.
///
/// A `.gz` suffix is stripped first, since MIPLIB ships `problem.mps.gz` and
/// the interesting extension is the one underneath.
FileFormat format_from_path(std::string_view path) noexcept {
  if (ends_with(path, ".gz")) path.remove_suffix(3);
  if (ends_with(path, ".mps")) return FileFormat::Mps;
  if (ends_with(path, ".lp")) return FileFormat::Lp;
  if (ends_with(path, ".qplib")) return FileFormat::Qplib;
  // Netlib instances are frequently extensionless or carry an instance name as
  // the extension, and they are all MPS.
  return FileFormat::Mps;
}

}  // namespace

core::Expected<model::Problem> parseProblem(std::string_view content,
                                            FileFormat format,
                                            const model::ReaderOptions& options) {
  switch (format) {
    case FileFormat::Mps:
    case FileFormat::Auto:
      return parse_mps(content, options);
    case FileFormat::Lp:
      return core::make_error(ErrorCode::NotImplemented,
                              "CPLEX LP format reader is not implemented yet");
    case FileFormat::Qplib:
      return core::make_error(ErrorCode::NotImplemented,
                              "QPLIB reader is not implemented yet");
  }
  return core::make_error(ErrorCode::NotImplemented, "unrecognized file format");
}

core::Expected<model::Problem> loadProblem(const std::string& filepath,
                                           FileFormat format,
                                           const model::ReaderOptions& options) {
  detail::MappedFile file;
  if (auto status = file.open(filepath); !status.ok()) return status.error();

  const FileFormat resolved =
      format == FileFormat::Auto ? format_from_path(filepath) : format;

  auto result = parseProblem(file.contents(), resolved, options);

  // Prefix the filename onto the diagnostic, so it reads as
  // "afiro.mps:412 [BOUNDS] ...". The parser has no idea what it was handed --
  // parseProblem also serves in-memory buffers.
  if (!result.has_value()) {
    auto err = std::move(result).error();
    err.message = filepath + ": " + err.message;
    return err;
  }
  return result;
}

}  // namespace sovsolve::io
