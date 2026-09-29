// Module 20: presentation only. Reads an iteration record and writes concise
// human-readable output. Kept independent of both Diagnostics (the data
// record) and the solver math, per docs/spec/module.txt Module 20.

#ifndef SOVSOLVE_SOLVER_LOGGING_HPP
#define SOVSOLVE_SOLVER_LOGGING_HPP

#include <cstddef>
#include <cstdio>

#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/Diagnostics.hpp"
#include "sovsolve/solver/LpPresolve.hpp"

namespace sovsolve::solver {

using model::LogOptions;

/// Writes one line for `record` to `out`, honoring `options.level`.
/// `Level::Silent` writes nothing; `Level::Summary` and above write one line
/// per iteration; `Level::Debug` additionally includes a timing breakdown.
void log_iteration(const IterationRecord& record, const LogOptions& options,
                    std::FILE* out = stdout);

/// Writes one line reporting what `solver::presolve` (Presolver.hpp) did to
/// the problem's dimensions -- the "Presolve reductions: rows X(-Y); columns
/// Z(-W)" line HiGHS prints and this project's own presolve never has,
/// making its effect visible only via a manual `mpsinfo`/`solve` comparison.
/// `Level::Silent` writes nothing; anything else writes this one line
/// (unlike `log_iteration`, which repeats every iteration, this is a
/// one-time summary and prints at every other level equally).
void log_presolve_summary(std::size_t rows_before, std::size_t cols_before,
                          std::size_t nnz_before, std::size_t rows_after,
                          std::size_t cols_after, std::size_t nnz_after,
                          const LogOptions& options, std::FILE* out = stdout);

/// The LP presolve's (LpPresolve.hpp) sizes and per-reduction counts.
void log_lp_presolve(const LpPresolveStats& stats, const LogOptions& options,
                     std::FILE* out = stdout);

/// Writes one line per entrant in a Module 30 concurrent solve: the engine,
/// its verdict, how long it ran and which one won. Declared here rather than
/// in ConcurrentSolve.hpp so that module stays free of presentation, the
/// same split Module 20 keeps from Diagnostics.
///
/// The times are also the per-engine runtime record an algorithm-selection
/// model would be trained on, which is why every entrant is printed and not
/// just the winner. `Level::Silent` writes nothing.
struct ConcurrentReport;
void log_concurrent_race(const ConcurrentReport& report, const LogOptions& options,
                         std::FILE* out = stdout);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_LOGGING_HPP
