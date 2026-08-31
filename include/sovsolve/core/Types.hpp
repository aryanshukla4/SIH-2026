// Fundamental scalar types, constants and enumerations.
//
// See docs/FORMULATION.md for the mathematics these types encode.
//
// Note on `Scalar`: the handoff datatype doc proposes a `Scalar` struct holding
// a value plus a runtime `precision` field. That type is deliberately NOT
// provided here. A wrapper around `double` adds nothing, and a runtime precision
// field puts a branch inside every arithmetic operation. Use `Real`; if mixed
// precision is ever wanted, template on the scalar type and resolve it at
// compile time.

#ifndef SOVSOLVE_CORE_TYPES_HPP
#define SOVSOLVE_CORE_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace sovsolve::core {

// --------------------------------------------------------------------------
// Scalar and index types
// --------------------------------------------------------------------------

/// Working precision. FP64 throughout, per prompts_corrected.txt.
using Real = double;

/// Default index type.
///
/// int32 rather than int64: it halves index bandwidth in CSR/CSC traversal
/// (which is memory-bound, so bandwidth is the cost that matters) and matches
/// the path cuSPARSE prefers. Instances exceeding 2^31 nonzeros promote to
/// `WideIndex`; the reader checks for overflow at load time rather than
/// truncating silently.
using Index = std::int32_t;

/// Fallback index type for instances beyond the int32 nonzero limit.
using WideIndex = std::int64_t;

/// Largest nonzero count representable by `Index`.
inline constexpr std::int64_t kMaxIndex =
    static_cast<std::int64_t>(std::numeric_limits<Index>::max());

// --------------------------------------------------------------------------
// Infinity
// --------------------------------------------------------------------------

/// Infinity threshold.
///
/// This is the MPS convention: any bound whose magnitude is >= 1e20 *is*
/// infinity, and is treated as absent. It is NOT
/// `std::numeric_limits<double>::infinity()` — files contain literal 1e20 and
/// 1e30 values that must compare equal to infinity, and arithmetic on a real
/// IEEE infinity produces NaN where 1e20 produces a large finite number.
///
/// Test with `is_infinite()` / `is_finite_bound()`, never with `==`.
inline constexpr Real INF = 1e20;

[[nodiscard]] inline constexpr bool is_pos_infinite(Real v) noexcept {
  return v >= INF;
}

[[nodiscard]] inline constexpr bool is_neg_infinite(Real v) noexcept {
  return v <= -INF;
}

[[nodiscard]] inline constexpr bool is_infinite(Real v) noexcept {
  return v >= INF || v <= -INF;
}

[[nodiscard]] inline constexpr bool is_finite_bound(Real v) noexcept {
  return v > -INF && v < INF;
}

// --------------------------------------------------------------------------
// Model enumerations
// --------------------------------------------------------------------------

/// Objective direction.
///
/// Present because MPS files carry `OBJSENSE`, and a maximization model parsed
/// as minimization solves backwards without any error being raised.
enum class ObjSense : std::uint8_t {
  Minimize,
  Maximize,
};

/// Row type as declared in the MPS `ROWS` section.
///
/// Retained for diagnostics and round-trip writing only. The authoritative
/// representation of a constraint is the `[row_lower, row_upper]` pair — see
/// `model::Problem`. Row senses and `RANGES` are both encodings of that pair,
/// and collapsing them removes an entire class of sign bugs.
enum class RowType : std::uint8_t {
  Less,       ///< `L` : a'x <= rhs
  Greater,    ///< `G` : a'x >= rhs
  Equal,      ///< `E` : a'x  = rhs
  Free,       ///< `N` : unconstrained; first one is the objective
};

/// Variable domain.
///
/// `Integer` and `Binary` come from `MARKER INTORG`/`INTEND` in the `COLUMNS`
/// section. Without this field MILP is impossible — integrality appears nowhere
/// else in an MPS file.
enum class VarType : std::uint8_t {
  Continuous,
  Integer,
  Binary,
  SemiContinuous,
};

/// Problem class, derived from the parsed model rather than declared.
enum class ProblemType : std::uint8_t {
  LP,     ///< Q empty, all columns continuous
  QP,     ///< Q non-empty, all columns continuous
  MILP,   ///< Q empty, at least one discrete column
  MIQP,   ///< Q non-empty, at least one discrete column
};

/// Storage orientation for sparse matrices.
enum class SparseFormat : std::uint8_t {
  CSR,  ///< compressed sparse row    — efficient for A*x
  CSC,  ///< compressed sparse column — efficient for A'*y
};

/// Where a buffer lives.
///
/// A tag, resolved at compile time by the allocator policy on `Vector`. It is
/// deliberately not a runtime field on the matrix type: a runtime
/// (format x location) pair puts a four-way branch inside every operation.
enum class MemorySpace : std::uint8_t {
  Host,
  Device,
};

// --------------------------------------------------------------------------
// Solver status
// --------------------------------------------------------------------------

/// Terminal classification of a solve. See docs/FORMULATION.md section 11.
enum class SolverStatus : std::uint8_t {
  NotConverged,    ///< still running, or stalled without reliable classification
  Optimal,
  Infeasible,      ///< primal infeasibility reliably detected
  Unbounded,       ///< dual infeasibility reliably detected
  Nonconvex,       ///< Q not PSD; detected at factorization, not by an up-front test
  MaxIterations,
  TimeLimit,
  NumericalError,
};

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_TYPES_HPP
