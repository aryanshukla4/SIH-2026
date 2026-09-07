// Reversible model transformations.
//
// ---------------------------------------------------------------------------
// Why this type exists
// ---------------------------------------------------------------------------
//
// module_corrected.txt section 3 requires presolve transformations to be
// reversible, and section 4 requires the scaler to retain inverse scaling
// information. Both are correct requirements -- but no module in the handoff
// list ever *consumes* that information, and no type was defined to hold it.
// As specified, the solver computes an answer in transformed space with no
// defined path back to the user's variables.
//
// `TransformStack` is that type. Canonicalization, presolve and scaling all
// push onto the same stack, in order; recovering a solution replays it in
// reverse.
//
// ---------------------------------------------------------------------------
// Duals are not an afterthought
// ---------------------------------------------------------------------------
//
// Each transform defines *two* inverses: one for the primal point and one for
// the duals. Retrofitting dual recovery onto a stack designed only for primal
// recovery means revisiting every transform, so both are declared from the
// start -- even where the dual inverse is currently the identity.

#ifndef SOVSOLVE_MODEL_TRANSFORM_HPP
#define SOVSOLVE_MODEL_TRANSFORM_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"

namespace sovsolve::model {

using core::Index;
using core::Real;

enum class TransformKind : std::uint8_t {
  // -- canonicalization ---------------------------------------------------
  //
  // In the bounded-variable form, variable bounds stay native, so no column is
  // shifted, reflected or split. The only column operations are "keep" and
  // "substitute out". Every original column pushes exactly ONE of the two, and
  // every original row exactly one of MapRow / RemoveEmptyRow / DropFreeRow.
  //
  // Records are keyed by the ORIGINAL index in `primary`, never by position in
  // the stack. Position-keyed recovery breaks the moment one column pushes no
  // record: every later column then recovers the wrong value.
  NegateObjective,      ///< maximize -> minimize; c and Q negated
  KeepColumn,           ///< original column `primary` -> canonical `secondary`
  RemoveFixedVariable,  ///< l == u; substituted out, `value` = the fixed value
  MapRow,               ///< original row `primary` -> canonical `secondary`
  NegateRow,            ///< a'x >= b  ->  -a'x <= -b, so a slack can be added
  AddSlack,             ///< inequality row; `secondary` = index within s
  BoundedSlack,         ///< ranged row; `secondary` = index in s, `value` = width
  RemoveEmptyRow,       ///< all-zero row over the kept columns, RHS consistent
  DropFreeRow,          ///< row with both bounds infinite; vacuous

  // -- presolve (owned by the presolve module, Presolver.cpp) --------------
  RemoveEmptyColumn,
  RemoveSingletonRow,
  RemoveRedundantRow,
  TightenBound,
  ShiftVariable,   ///< x' = x - t
  NegateVariable,  ///< x' = t - x

  /// A FREE column that is a SINGLETON in row `primary` (`secondary` = the
  /// column), substituted out exactly: the row determines the column's
  /// value, so both disappear together. Not one of the six kinds above --
  /// those each remove either a row or a column, never both at once. See
  /// `Presolver.hpp`'s doc comment for the derivation, and
  /// `Canonicalizer.cpp::recover_solution` for the dual/primal recovery.
  RemoveFreeSingleton,

  /// Two columns with an identical `A` pattern (same rows, same values) AND
  /// identical cost were merged into one: `primary` = the DROPPED column
  /// (original index), `secondary` = the SURVIVING column (original index),
  /// whose bounds were widened to the Minkowski sum of the pair's own bounds.
  /// See `Presolver.hpp`'s doc comment for the merge derivation, and
  /// `Canonicalizer.cpp::recover_solution` for the primal split and shared-
  /// stationarity dual recovery.
  MergeDuplicateColumn,

  // -- scaling ------------------------------------------------------------
  RowScaling,
  ColumnScaling,
};

/// One recorded transformation.
///
/// Deliberately a flat struct with a kind tag rather than a class hierarchy:
/// the stack is replayed in a tight loop, the set of kinds is closed and known
/// here, and a flat record keeps the whole stack contiguous.
struct TransformRecord {
  TransformKind kind = TransformKind::NegateObjective;

  /// Row or column the transform applies to. `kIndexNone` when not applicable.
  Index primary = kIndexNone;
  /// Second index, e.g. the partner column of a free-variable split or the row
  /// added by `AddBoundRow`.
  Index secondary = kIndexNone;

  /// Shift, scale factor, or bound value, depending on `kind`.
  Real value = 0.0;
  /// Second scalar where one is needed (e.g. a range width).
  Real value2 = 0.0;

  static constexpr Index kIndexNone = -1;
};

/// An ordered sequence of transformations from the original model to the
/// solver's working model.
///
/// Applied front to back; inverted back to front.
class TransformStack {
 public:
  void push(const TransformRecord& record) { records_.push_back(record); }

  [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
  [[nodiscard]] bool empty() const noexcept { return records_.empty(); }

  [[nodiscard]] const TransformRecord& operator[](std::size_t i) const noexcept {
    return records_[i];
  }

  [[nodiscard]] const std::vector<TransformRecord>& records() const noexcept {
    return records_;
  }

  /// Dimensions of the original model, so recovery can size its output without
  /// consulting the original `Problem`.
  std::size_t original_rows = 0;
  std::size_t original_cols = 0;

  void clear() { records_.clear(); }

 private:
  std::vector<TransformRecord> records_;
};

}  // namespace sovsolve::model

#endif  // SOVSOLVE_MODEL_TRANSFORM_HPP
