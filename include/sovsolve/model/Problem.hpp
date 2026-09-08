// The faithful optimization model: what the input file actually said.
//
// This is layer one of two. `Problem` preserves the model as written; the
// Canonicalizer (Canonical.hpp) produces the solver's working form and records
// how to invert the transformation. See docs/FORMULATION.md sections 1-2.
//
// ---------------------------------------------------------------------------
// Why this differs from the handoff datatype doc
// ---------------------------------------------------------------------------
//
// The doc's `Problem` is
//
//     problem_type, A, Q, b, c, lower_bounds, upper_bounds, constraint_sense
//
// against the assumed model `Ax <= b, x >= 0`. Real benchmark instances are
// not of that form, and a loader restricted to it cannot usefully read a single
// Netlib or MIPLIB file. Eight fields are added:
//
//   col_type        integrality from MARKER INTORG/INTEND. Without it MILP is
//                   impossible -- integrality appears nowhere else in an MPS
//                   file.
//   sense           OBJSENSE. A maximization model parsed as minimization
//                   solves backwards with no error raised.
//   obj_constant    the RHS entry on the objective row. Omitting it makes every
//                   reported objective wrong by a constant.
//   row_lower/upper replaces `b` + `constraint_sense`, so RANGES rows
//                   (l <= a'x <= u) are representable at all.
//   row_names       solution reporting, error messages, round-trip tests.
//   col_names       likewise.
//   objective_row   MPS designates the objective by name among the N rows.
//   sos / semi-continuous markers, so they can be reported rather than
//                   silently dropped.
//
// The doc also contains a contradiction worth noting: it lists
// `lower_bounds`/`upper_bounds` on Problem while stating the formulation uses
// `x >= 0`. Where general bounds live was undefined. Here they live on
// `Problem`, and the Canonicalizer is what turns them into `x >= 0`.

#ifndef SOVSOLVE_MODEL_PROBLEM_HPP
#define SOVSOLVE_MODEL_PROBLEM_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "sovsolve/core/NameArena.hpp"
#include "sovsolve/core/SparseMatrix.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"

namespace sovsolve::model {

using core::Index;
using core::NameArena;
using core::ObjSense;
using core::ProblemType;
using core::Real;
using core::RealVector;
using core::VarType;

/// A special ordered set, recorded so it can be reported rather than ignored.
///
/// Silently dropping an SOS set changes the model and yields a confidently
/// wrong answer, which is worse than refusing the file. The loader returns
/// `UnsupportedFeature` when these are present.
struct SosSet {
  int type = 1;                    ///< 1 or 2
  std::vector<Index> columns;
  std::vector<Real> weights;
};

/// The model exactly as parsed.
///
///     minimize/maximize   1/2 x'Qx + c'x + obj_constant
///     subject to          row_lower <= A x <= row_upper
///                         col_lower <=   x <= col_upper
///                         x_j integral for col_type[j] != Continuous
///
/// Bounds use the MPS infinity convention: magnitude >= core::INF *is*
/// infinity. Test with `core::is_infinite`, never `==`.
struct Problem {
  // -- objective ----------------------------------------------------------
  ObjSense sense = ObjSense::Minimize;
  Real obj_constant = 0.0;
  RealVector c;                          ///< length n

  /// Quadratic objective, stored as a **full symmetric** matrix rather than a
  /// triangle. Triangle storage halves memory but makes SpMV need atomics on
  /// GPU; full storage is the right trade for the solver's access pattern.
  /// Empty for LP.
  ///
  /// The `1/2` factor belongs to the objective definition above -- see
  /// docs/MPS-FORMAT-NOTES.md section 8, where getting this wrong scales every
  /// quadratic objective by exactly 2.
  core::SparseMatrixPair<> Q;

  // -- constraints --------------------------------------------------------

  /// Both orientations. CSR serves `A*x`, CSC serves `A'*y`; the IPM needs
  /// both every iteration.
  core::SparseMatrixPair<> A;

  /// Row bounds, length m. This single representation covers L/G/E rows and
  /// RANGES uniformly:
  ///     L:  (-INF, rhs]      G:  [rhs, +INF)      E:  [rhs, rhs]
  /// See docs/MPS-FORMAT-NOTES.md section 6 for the RANGES translation.
  RealVector row_lower;
  RealVector row_upper;

  // -- variables ----------------------------------------------------------
  RealVector col_lower;                  ///< length n, may be -INF
  RealVector col_upper;                  ///< length n, may be +INF
  std::vector<VarType> col_type;         ///< length n

  // -- identification -----------------------------------------------------
  std::string problem_name;
  NameArena row_names;                   ///< m entries, excludes the objective row
  NameArena col_names;                   ///< n entries
  std::string objective_row_name;

  // -- features recorded but not solved -----------------------------------
  std::vector<SosSet> sos_sets;

  // -- derived ------------------------------------------------------------

  [[nodiscard]] std::size_t num_rows() const noexcept { return A.rows(); }
  [[nodiscard]] std::size_t num_cols() const noexcept { return A.cols(); }
  [[nodiscard]] std::size_t nnz() const noexcept { return A.nnz(); }

  [[nodiscard]] bool has_quadratic() const noexcept { return !Q.empty(); }

  /// Explicit deep copy, host-only -- Problem is move-only (RealVector and
  /// SparseMatrixPair members are) for the same reason Vector itself is: an
  /// implicit copy of a multi-million-entry matrix must never happen by
  /// accident. Module 22 (branch-and-bound) is the one caller that
  /// genuinely needs an independent Problem per search -- one clone per
  /// `solve()` call, with only the small `col_lower`/`col_upper` vectors
  /// re-cloned per node, never `A`/`Q`/`c` again after this.
  [[nodiscard]] Problem clone() const {
    Problem out;
    out.sense = sense;
    out.obj_constant = obj_constant;
    out.c = c.clone();
    out.Q = Q.clone();
    out.A = A.clone();
    out.row_lower = row_lower.clone();
    out.row_upper = row_upper.clone();
    out.col_lower = col_lower.clone();
    out.col_upper = col_upper.clone();
    out.col_type = col_type;
    out.problem_name = problem_name;
    out.row_names = row_names;
    out.col_names = col_names;
    out.objective_row_name = objective_row_name;
    out.sos_sets = sos_sets;
    return out;
  }

  [[nodiscard]] bool has_discrete() const noexcept {
    for (const auto t : col_type) {
      if (t != VarType::Continuous) return true;
    }
    return false;
  }

  [[nodiscard]] std::size_t num_discrete() const noexcept {
    std::size_t count = 0;
    for (const auto t : col_type) {
      if (t != VarType::Continuous) ++count;
    }
    return count;
  }

  /// Classification is derived from the parsed model, never declared by the
  /// file. A `problem_type` field that can disagree with the data is a bug
  /// waiting to happen.
  [[nodiscard]] ProblemType type() const noexcept {
    const bool q = has_quadratic();
    const bool d = has_discrete();
    if (q && d) return ProblemType::MIQP;
    if (q) return ProblemType::QP;
    if (d) return ProblemType::MILP;
    return ProblemType::LP;
  }

  /// Checks internal consistency: array lengths agree with the matrix
  /// dimensions, no lower bound exceeds its upper, Q is square and symmetric
  /// when present, and both matrix orientations satisfy their invariants.
  [[nodiscard]] bool validate() const noexcept;
};

}  // namespace sovsolve::model

#endif  // SOVSOLVE_MODEL_PROBLEM_HPP
