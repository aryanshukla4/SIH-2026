// Module 20: presentation only. Reads an iteration record and writes concise
// human-readable output. Kept independent of both Diagnostics (the data
// record) and the solver math, per module.txt Module 20.

#ifndef SOVSOLVE_SOLVER_LOGGING_HPP
#define SOVSOLVE_SOLVER_LOGGING_HPP

#include <cstdio>

#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/Diagnostics.hpp"

namespace sovsolve::solver {

using model::LogOptions;

/// Writes one line for `record` to `out`, honoring `options.level`.
/// `Level::Silent` writes nothing; `Level::Summary` and above write one line
/// per iteration; `Level::Debug` additionally includes a timing breakdown.
void log_iteration(const IterationRecord& record, const LogOptions& options,
                    std::FILE* out = stdout);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_LOGGING_HPP
