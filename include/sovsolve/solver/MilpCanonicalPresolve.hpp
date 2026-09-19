// Module 29, stage A: MIP presolve on the canonical model.
//
// SOURCE: T. Achterberg, "Constraint Integer Programming", PhD thesis, TU
// Berlin (2007), chapter 10 -- [CIP]. Its measurements (Table 10.1):
// "disabling presolving almost leads to a doubling of the runtime". This
// stage transcribes the linear-constraint presolving of Algorithm 10.1 and the
// dual fixing of Algorithm 10.14; probing (10.6), restarts (10.9) and
// equation aggregation (10.1.1) are later stages.
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
// toward a bound no row resists is fixed there. Rounds repeat until nothing
// changes.
//
// WHAT CHANGES. Column bounds, coefficients and right-hand sides of inequality
// rows, and deleted redundant rows. Columns are never removed (a fixed column
// stays, with lower == upper), so the canonical column space is unchanged and
// a solution maps back through the unpresolved canonical model. Equality rows
// are never coefficient-tightened: step 1f would give them two different
// sides, which the canonical form (Canonical.hpp) cannot hold without a range
// column.
//
// INTEGRALITY IN ORIGINAL UNITS. A canonical column is x_original / s
// (Scaler.cpp); integrality, integer rounding and the unit steps of step 1f
// are applied in original units.

#ifndef SOVSOLVE_SOLVER_MILP_CANONICAL_PRESOLVE_HPP
#define SOVSOLVE_SOLVER_MILP_CANONICAL_PRESOLVE_HPP

#include <cstddef>
#include <vector>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"

namespace sovsolve::solver {

struct CanonicalPresolveStats {
  std::size_t rounds = 0;
  std::size_t bounds_tightened = 0;
  std::size_t coefficients_tightened = 0;
  std::size_t rows_removed = 0;
  std::size_t columns_fixed = 0;  ///< by dual fixing
};

/// Presolve `problem` in place. `integer_scale[j]` is `s` for an integer
/// column (`x_original = s x_j`) and 0 for a continuous one. Returns
/// `PrimalInfeasible` when a row provably cannot be satisfied.
[[nodiscard]] core::Status presolve_canonical(model::CanonicalProblem& problem,
                                              const std::vector<core::Real>& integer_scale,
                                              std::size_t max_rounds,
                                              CanonicalPresolveStats& stats);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_MILP_CANONICAL_PRESOLVE_HPP
