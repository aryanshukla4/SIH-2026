// Internal reader entry points.
//
// Private to the io module: `Load.hpp` is the public surface, and the
// per-format parsers stay behind it so a new format can be added without any
// caller changing.

#ifndef SOVSOLVE_IO_DETAIL_READERS_HPP
#define SOVSOLVE_IO_DETAIL_READERS_HPP

#include <string_view>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"

namespace sovsolve::io {

/// Fixed or free MPS, auto-detected between the two.
[[nodiscard]] core::Expected<model::Problem> parse_mps(
    std::string_view content, const model::ReaderOptions& options);

/// CPLEX LP format, accepting Gurobi's superset.
[[nodiscard]] core::Expected<model::Problem> parse_lp(
    std::string_view content, const model::ReaderOptions& options);

}  // namespace sovsolve::io

#endif  // SOVSOLVE_IO_DETAIL_READERS_HPP
