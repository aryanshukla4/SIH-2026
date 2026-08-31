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
  // Every column transform is an instance of the affine map x = d*x' + t with
  // d in {+1, -1}. Keeping the two kinds distinct rather than collapsing them
  // into one record makes the inverse readable at the point of use.
  NegateObjective,     ///< maximize -> minimize; c and Q negated
  ShiftVariable,       ///< x >= l  ->  x' = x - l          (d = +1, t = l)
  NegateVariable,      ///< x <= u, no lower  ->  x' = u - x (d = -1, t = u)
  SplitFreeVariable,   ///< x free  ->  x = xp - xm, both >= 0
  AddBoundRow,         ///< x' <= w  ->  x' + t = w, t >= 0
  NegateRow,           ///< a'x >= b  ->  -a'x <= -b, so a slack can be added
  AddSlack,            ///< a'x <= b  ->  a'x + s = b, s >= 0
  BoundedSlack,        ///< l <= a'x <= u  ->  slack with its own bound row
  DropFreeRow,         ///< N rows after the objective

  // -- presolve (reserved; owned by the presolve module) -------------------
  RemoveEmptyRow,
  RemoveEmptyColumn,
  RemoveFixedVariable,
  RemoveSingletonRow,
  RemoveRedundantRow,
  TightenBound,

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
