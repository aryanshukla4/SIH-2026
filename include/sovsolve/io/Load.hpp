// Public model-loading interface.
//
// The handoff signature is
//
//     Problem loadProblem(const std::string& filepath);
//
// which has no error channel. On a 500 MB MPS file "parse failed" without a
// line number is not a usable diagnostic, so the return type is
// `Expected<Problem>` carrying a message, a line, and the section being parsed.
// The shape is otherwise unchanged: one entry point, format detected from the
// file, parsers internal to this module.

#ifndef SOVSOLVE_IO_LOAD_HPP
#define SOVSOLVE_IO_LOAD_HPP

#include <cstdint>
#include <string>
#include <string_view>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"

namespace sovsolve::io {

enum class FileFormat : std::uint8_t {
  Auto,   ///< detect from extension, then from content
  Mps,    ///< fixed or free MPS; auto-detected between the two
  Lp,     ///< CPLEX LP format
  Qplib,  ///< QPLIB
};

/// Load a model from disk.
///
/// Handles gzip-compressed input transparently **when built with zlib**, which
/// MIPLIB needs since it ships `.mps.gz`. zlib is a compression library rather
/// than a solver library, so it is clean under the "from scratch" constraint.
///
/// It is OPTIONAL, and the CMake configure step reports which way it went. With
/// zlib absent -- as on the current development machine -- a `.gz` input
/// returns `UnsupportedFeature` naming the cause, rather than being mis-parsed
/// as text. So MIPLIB cannot be loaded on such a build without decompressing
/// first, and the corpus tests here cover Netlib only.
///
/// The returned `Problem` is the model exactly as written -- no presolve, no
/// scaling, no canonicalization. See model/Canonical.hpp for the transform to
/// the solver's working form.
[[nodiscard]] core::Expected<model::Problem> loadProblem(
    const std::string& filepath, FileFormat format = FileFormat::Auto,
    const model::ReaderOptions& options = {});

/// Parse from an in-memory buffer. Used by the test suite, which builds tiny
/// hand-checkable instances inline rather than as files on disk.
[[nodiscard]] core::Expected<model::Problem> parseProblem(
    std::string_view content, FileFormat format,
    const model::ReaderOptions& options = {});

}  // namespace sovsolve::io

#endif  // SOVSOLVE_IO_LOAD_HPP
