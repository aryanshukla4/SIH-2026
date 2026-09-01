// Model writing -- the inverse of Load.hpp.
//
// ---------------------------------------------------------------------------
// Why a writer earns its place before another reader
// ---------------------------------------------------------------------------
//
// Three jobs, and none of them is served by a second input format:
//
//   1. ROUND-TRIP TESTING. `parse -> write -> parse` and compare gives the
//      readers a check that does not depend on my expectations being right. A
//      reader tested only against hand-written assertions is tested against
//      the same misunderstanding that produced it.
//
//   2. MODULE-BOUNDARY DEBUGGING. The solver is split across four people. When
//      the IPM fails at iteration 12 of 80bau3b, its owner should not have to
//      run my parser, canonicalizer, presolve and scaler to reproduce it. With
//      a writer I hand over ONE file holding the exact model their module saw,
//      and they debug it in isolation.
//
//   3. BENCHMARK EVIDENCE. Comparing against a reference solver means feeding
//      both the same model. Writing it out is how you prove they got one.
//
// ---------------------------------------------------------------------------
// Fidelity
// ---------------------------------------------------------------------------
//
// Numbers are written with `std::to_chars`, which produces the SHORTEST decimal
// that reads back bit-identical. Not `%.17g`: that is both longer and, on some
// implementations, not exactly round-tripping. Exactness is the whole point
// here -- a writer that perturbs the model cannot be used to reproduce a
// numerical failure.
//
// One construct deliberately does not round-trip: a row with both bounds
// infinite is written as an `N` row, and MPS defines `N` rows after the first
// as free rows that a reader drops. That is correct MPS semantics rather than a
// defect -- such a row constrains nothing -- but it means a programmatically
// built model containing one will come back with fewer rows. Parsed models
// never contain one, since the reader already dropped them.

#ifndef SOVSOLVE_IO_WRITE_HPP
#define SOVSOLVE_IO_WRITE_HPP

#include <string>
#include <string_view>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/io/Load.hpp"
#include "sovsolve/model/Problem.hpp"

namespace sovsolve::io {

struct WriteOptions {
  /// Emit `OBJSENSE` even for a minimization. Off by default: minimization is
  /// the MPS default and older readers reject the section.
  bool always_write_objsense = false;

  /// Emit the `QUADOBJ` section when the model has a quadratic objective.
  /// Turning it off writes the LP relaxation of a QP, which is occasionally
  /// what you want when isolating a failure.
  bool write_quadratic = true;

  /// Column width for names. Free-format MPS ignores it; it exists so output
  /// stays readable next to a fixed-format file.
  int name_width = 10;
};

/// Render `problem` as MPS into a string.
///
/// Free-format MPS: fields are whitespace-separated, section headers start in
/// column 1 and data lines are indented. Every reader accepts it, including the
/// fixed-format ones, because fixed format is a column-aligned special case.
[[nodiscard]] core::Expected<std::string> writeMpsToString(
    const model::Problem& problem, const WriteOptions& options = {});

/// Write `problem` to `filepath`.
///
/// The format follows the extension when `format` is `Auto`, matching
/// `loadProblem`. Only `Mps` is implemented; the others report
/// `NotImplemented` rather than silently writing the wrong thing.
[[nodiscard]] core::Status writeProblem(const std::string& filepath,
                                        const model::Problem& problem,
                                        FileFormat format = FileFormat::Auto,
                                        const WriteOptions& options = {});

}  // namespace sovsolve::io

#endif  // SOVSOLVE_IO_WRITE_HPP
