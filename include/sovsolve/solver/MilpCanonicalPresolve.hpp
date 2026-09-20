// Module 29: MIP presolve on the canonical model.
//
// SOURCES.
//   [CIP] T. Achterberg, "Constraint Integer Programming", PhD thesis, TU
//         Berlin (2007), chapter 10. Its measurements (Table 10.1):
//         "disabling presolving almost leads to a doubling of the runtime".
//   [AGH] T. Achterberg, R. E. Bixby, Z. Gu, E. Rothberg, D. Weninger,
//         "Presolve Reductions in Mixed Integer Programming", ZIB Report
//         16-44 (2016) -- the Gurobi presolve engine, with one impact table
//         per reduction. Section numbers below are its.
//
// ---------------------------------------------------------------------------
// Stage A -- reductions that keep every column
// ---------------------------------------------------------------------------
//
// Per row, repeated at most 10 times while it changes ([CIP] 10.1 step 1g):
//   - infeasibility and redundancy from the activity bounds (step 1d): a row
//     that can never hold proves the model infeasible; an inequality that can
//     never be violated is deleted;
//   - bound tightening by domain propagation, Algorithm 7.1 (step 1c);
//   - coefficient tightening for integer columns (step 1f): if the row is
//     redundant whenever x_j is off its bound, shrink a_j and the side so the
//     integer points are the same and fewer fractional ones remain.
// Then dual fixing (Algorithm 10.14): a column whose objective pushes it
// toward a bound no row resists is fixed there.
//
// Stage A never removes a column (a fixed column stays, with lower == upper),
// so the canonical column space is unchanged and a solution maps back through
// the unpresolved canonical model with no bookkeeping at all. That is why it
// came first: it bought [AGH]'s bound strengthening (1.09) and coefficient
// strengthening (1.06) without a postsolve stack.
//
// ---------------------------------------------------------------------------
// Stage B -- reductions that REMOVE columns
// ---------------------------------------------------------------------------
//
// [AGH] measures the substitution of implied free variables at 1.42 in the
// ">= 10 sec" bracket (Table 13) -- "the most important presolve algorithm
// within Gurobi's set of single-column reductions", and the largest entry
// reachable without node presolve. It removes a column, so it needs an
// explicit inverse: that is `CanonicalPostsolve`.
//
//   1. SUBSTITUTE IMPLIED FREE VARIABLES ([AGH] 4.5). Let [lo_j, hi_j] be the
//      bounds row i alone implies for x_j, computed from the OTHER columns'
//      bounds. When that range sits inside x_j's own bounds, those bounds can
//      never bind, x_j is "implied free", and an equality row containing it
//
//          A_iS x_S + a_ij x_j = b_i
//
//      determines it: x_j := (b_i - A_iS x_S) / a_ij. Substituting that into
//      every other row and into the objective removes the column AND the row.
//      The check uses row i alone, which is weaker than [AGH]'s (it propagates
//      the whole model first) and therefore safe: the reduced problem keeps
//      the bounds of S, so any point of it recovers an x_j inside [lo_j, hi_j].
//
//      A FREE COLUMN SINGLETON -- a free column whose only non-zero is in one
//      equality -- is this reduction's degenerate case: the implied range sits
//      inside (-inf, inf) for free, and no other row contains the column, so
//      the substitution is exact and creates no fill-in.
//
//      ONLY CONTINUOUS COLUMNS ARE SUBSTITUTED. [AGH] 4.5 permits an integer
//      x_j when S is integer and a_ik/a_ij is integral for every k in S;
//      that side condition is not checked here, so integer columns are left
//      alone. It also keeps every IntegerColumn of MilpSolve.cpp alive across
//      presolve, needing only an index remap.
//
//      Two guards from [AGH] 4.5, both required in practice:
//        - A MARKOWITZ THRESHOLD on the pivot, |a_ij| >= 0.01 * max|a_.j| or
//          |a_ij| >= 0.01 * max|a_i.|. Dividing by a tiny pivot is "one of the
//          main sources for numerical issues in the presolving step".
//        - A FILL-IN LIMIT. The substitution writes row i into every other row
//          holding the column, so it can densify A. See `kMaxSubstitutionFill`.
//
//   2. PARALLEL COLUMNS ([AGH] 6.3). Columns with A_.k = lambda A_.j and
//      c_k = lambda c_j contribute to every row and to the objective only
//      through x_j + lambda x_k, so the pair merges into one column
//
//          y := x_j + lambda x_k
//
//      carrying A_.j and c_j, with the Minkowski-sum bounds of [AGH] (6.1).
//      Postsolve splits y back, which is where the bounds are needed.
//
//      ONLY CONTINUOUS PAIRS ARE MERGED, for the same reason as above and for
//      one more: [AGH] notes that merging integer columns is valid only when
//      the image of x_j + lambda x_k is an interval of integers with no holes,
//      a condition that has to be re-derived in original units because our
//      canonical columns are scaled. [AGH] Table 25 puts parallel columns at
//      1.09, so the continuous-only case is the whole of the cheap part.
//
// NOT DONE HERE, ranked by [AGH]'s ">= 10 sec" ratios: probing (1.17, 7.2),
// multi-row reductions (1.34, section 5), implied integer detection (1.13,
// 7.6), dominated columns (6.4), node presolve (1.51, section 8).
//
// ---------------------------------------------------------------------------
// Invariants this file must not break
// ---------------------------------------------------------------------------
//
// EQUALITY ROWS COME FIRST. `num_equality` names a contiguous leading block
// (Canonical.hpp), so the matrix is rebuilt with the surviving equalities
// ahead of the surviving inequalities.
//
// RANGE COLUMNS ARE NEVER REMOVED. They occupy the trailing `num_range`
// positions and the count, not a per-column mark, is what identifies them.
//
// EQUALITY ROWS ARE NEVER COEFFICIENT-TIGHTENED: step 1f would give them two
// different sides, which the canonical form cannot hold without a range
// column.
//
// INTEGRALITY IN ORIGINAL UNITS. A canonical column is x_original / s
// (Scaler.cpp); integrality, integer rounding and the unit steps of step 1f
// are applied in original units.

#ifndef SOVSOLVE_SOLVER_MILP_CANONICAL_PRESOLVE_HPP
#define SOVSOLVE_SOLVER_MILP_CANONICAL_PRESOLVE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"

namespace sovsolve::solver {

struct CanonicalPresolveStats {
  std::size_t rounds = 0;
  std::size_t bounds_tightened = 0;
  std::size_t coefficients_tightened = 0;
  std::size_t rows_removed = 0;
  std::size_t columns_fixed = 0;       ///< by dual fixing (stage A)
  std::size_t columns_substituted = 0;  ///< implied free substitution (stage B)
  std::size_t columns_merged = 0;       ///< parallel columns (stage B)
};

/// One removed column, and everything needed to put its value back.
///
/// Indices are in PLAIN canonical column space -- the space the problem had
/// when `presolve_canonical` was entered, which is also the space of the
/// unpresolved model MilpSolve.cpp keeps for rebuilding solutions. Records are
/// replayed in reverse, so a record may safely refer to a column that a LATER
/// record removed: that one has already been restored by then.
struct PostsolveColumn {
  enum class Kind : std::uint8_t {
    /// `column` came out of an equality row: it is `(rhs - row . x) / pivot`.
    SubstituteEquality,
    /// `column` was merged into `partner`, which now holds
    /// `y = x_partner + pivot * x_column`. Splitting y needs both bound pairs.
    MergeParallel,
  };

  Kind kind = Kind::SubstituteEquality;
  std::size_t column = 0;   ///< the column that was REMOVED
  std::size_t partner = 0;  ///< MergeParallel: the column that SURVIVED
  core::Real pivot = 1.0;   ///< SubstituteEquality: a_ij. MergeParallel: lambda.
  core::Real rhs = 0.0;     ///< SubstituteEquality: b_i.

  core::Real removed_lower = 0.0, removed_upper = 0.0;
  core::Real partner_lower = 0.0, partner_upper = 0.0;  ///< MergeParallel only

  /// SubstituteEquality: row i without column j, as (column, coefficient).
  std::vector<std::pair<std::size_t, core::Real>> row;
};

/// The inverse of stage B: presolved canonical space -> plain canonical space.
struct CanonicalPostsolve {
  static constexpr std::size_t kRemoved = std::numeric_limits<std::size_t>::max();

  std::size_t plain_columns = 0;
  /// Plain column index -> its index in the presolved problem, or `kRemoved`.
  std::vector<std::size_t> new_of_old;
  /// Applied in this order; `expand_canonical_point` replays them backwards.
  std::vector<PostsolveColumn> undo;

  [[nodiscard]] bool removes_columns() const noexcept { return !undo.empty(); }
};

/// Rebuild a plain-canonical point from a presolved one.
///
/// `presolved_x` holds the presolved problem's structural columns (trailing
/// logicals, if the caller has any, are ignored). `plain_x` comes back sized
/// `post.plain_columns`. With nothing removed this is a straight copy.
void expand_canonical_point(const CanonicalPostsolve& post,
                            core::HostSpan<const core::Real> presolved_x,
                            std::vector<core::Real>& plain_x);

/// Presolve `problem` in place. `integer_scale[j]` is `s` for an integer
/// column (`x_original = s x_j`) and 0 for a continuous one; it is indexed in
/// PLAIN canonical space and is not remapped. `postsolve` comes back with the
/// column map and the undo stack, and is the only way to interpret the result
/// when columns were removed. Returns `PrimalInfeasible` when a row provably
/// cannot be satisfied.
///
/// `allow_column_removal` false runs stage A only, leaving `postsolve` an
/// identity map -- the shape MilpSolve.cpp's cut and branching code sees when
/// stage B is switched off.
[[nodiscard]] core::Status presolve_canonical(model::CanonicalProblem& problem,
                                              const std::vector<core::Real>& integer_scale,
                                              std::size_t max_rounds,
                                              bool allow_column_removal,
                                              CanonicalPostsolve& postsolve,
                                              CanonicalPresolveStats& stats);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_MILP_CANONICAL_PRESOLVE_HPP
