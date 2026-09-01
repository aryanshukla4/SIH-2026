// Canonicalization: faithful model -> solver working form.
//
// Layer two of the two-layer design. This module does not exist in the handoff
// docs; its absence is why they could assume `Ax <= b, x >= 0` -- the
// transformation from what files contain to what the solver wants was never
// assigned to anyone.
//
// ---------------------------------------------------------------------------
// The canonical form, precisely
// ---------------------------------------------------------------------------
//
//     minimize    1/2 x'Qx + c'x
//     subject to  A_E x       = b_E
//                 A_I x + s   = b_I,   s >= 0
//                 l <= x <= u
//
// This is the bounded-variable form the team settled on. Two properties of it
// matter more than they look:
//
//   * Equality rows carry NO slack. A form that appends `s >= 0` to every row
//     cannot express an equality, and equalities are the majority of rows in
//     real instances -- 516 of 821 in 25fv47, all 459 in gas11.
//
//   * Finite variable bounds stay NATIVE. Turning each `x <= u` into a
//     constraint row grows `A*Theta*A'` by one row per boxed column, measured
//     at +750% on rgn and +648% on gt2. Two complementarity pairs per variable
//     costs one extra vector instead.
//
// `A_E` and `A_I` are contiguous ROW BLOCKS of one matrix, not two matrices:
// rows [0, num_equality) are `A_E` and the rest are `A_I`. One matrix keeps
// `A*x` and `A'*y` as single kernel calls, which is what the IPM actually does
// with them every iteration; the fill-reducing ordering permutes the rows again
// downstream anyway, so preserving file order buys nothing.
//
// ---------------------------------------------------------------------------
// The startability contract
// ---------------------------------------------------------------------------
//
// The output is either startable by the interior-point method, or a definite
// infeasibility verdict. Nothing structurally unsolvable reaches the solver.
// This is a contract rather than an optimization, so it cannot be delegated to
// the presolver, which callers may switch off.
//
// Two conditions, both measured on the Netlib corpus and both common:
//
//   * No column has `l == u`. The IPM needs `x-l > 0` and `u-x > 0` at once;
//     their sum is `u-l`, so a fixed column admits no strictly interior point
//     and the method cannot take a first step. 1067 fixed columns appear across
//     8 of our 19 instances. They are substituted out here.
//
//   * No row of `A` is entirely zero. An all-zero row makes that row and column
//     of `A*Theta*A'` identically zero for every `Theta` -- an exact zero pivot
//     no regularization of `Theta` can repair. 80bau3b has 25, greenbea 3.
//     Consistent ones are dropped; an inconsistent one (`0 = 5`) returns
//     INFEASIBLE rather than being handed on to fail numerically later.
//
// Note that substituting fixed columns out can CREATE an empty row, when a row
// met the kept columns nowhere. Emptiness is therefore tested after the
// substitution, not before.
//
// Free variables are NOT split into `x = xp - xm`. The split doubles the column
// count and makes the pair structurally dependent, and interior-point methods
// converge poorly on it. Free columns stay free; the KKT builder floors their
// zero entry of `Theta^-1`.

#ifndef SOVSOLVE_MODEL_CANONICAL_HPP
#define SOVSOLVE_MODEL_CANONICAL_HPP

#include <cstddef>

#include "sovsolve/core/SparseMatrix.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Problem.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/model/Transform.hpp"

namespace sovsolve::model {

/// The solver's working model. See the header comment for the exact form.
struct CanonicalProblem {
  RealVector c;                ///< length num_cols()
  core::SparseMatrixPair<> A;  ///< num_rows() x num_cols()
  core::SparseMatrixPair<> Q;  ///< empty for LP
  RealVector b;                ///< length num_rows()

  /// Native variable bounds. Either may be infinite; a free column has both
  /// infinite. No entry satisfies `col_lower[j] == col_upper[j]` -- fixed
  /// columns were substituted out.
  RealVector col_lower;
  RealVector col_upper;

  /// Number of trailing columns that are RANGE columns rather than original
  /// variables. They occupy `[num_cols() - num_range, num_cols())`.
  ///
  /// A ranged row `lo <= a'x <= hi` becomes an EQUALITY plus one bounded
  /// column:
  ///
  ///     a'x + t = hi,     0 <= t <= hi - lo
  ///
  /// which is exactly `lo <= a'x <= hi`. The alternative -- keeping the row an
  /// inequality and giving its slack a finite upper bound -- looks cheaper but
  /// is not: `s` would then need a second dual, an extra complementarity
  /// residual, an extra step-length test and an extra term in `mu`, touching
  /// the initializer, residual calculator, predictor-corrector, KKT builder,
  /// step-length calculator and barrier controller. A bounded column needs
  /// none of that, because `l <= x <= u` is already the form every variable is
  /// in. One column against six modules.
  ///
  /// The other textbook option, splitting the row into `a'x <= hi` and
  /// `-a'x <= -lo`, is worse than both: the two rows are negatives, so their
  /// 2x2 contribution to `A*Theta*A'` has determinant `t^2 - t^2 = 0` and the
  /// system is singular except for the slack diagonal -- which vanishes exactly
  /// at convergence, when a satisfied range makes both sides tight.
  ///
  /// No instance in our corpus has a ranged row, so this path is unit-tested
  /// but not exercised by the benchmark set.
  std::size_t num_range = 0;

  /// Row layout: equalities first, then inequalities.
  /// `A_E` is rows `[0, num_equality)`; `A_I` is rows `[num_equality, m)`.
  /// The slack vector `s` has length `num_inequality_rows()` and its element
  /// `k` belongs to canonical row `num_equality + k`.
  std::size_t num_equality = 0;

  /// Constant folded out of the objective by substituting fixed columns, in
  /// canonical (minimization) space. Added back during recovery.
  Real obj_offset = 0.0;

  /// True when the original model was a maximization and `c`/`Q` were negated.
  bool objective_negated = false;

  [[nodiscard]] std::size_t num_rows() const noexcept { return A.rows(); }
  [[nodiscard]] std::size_t num_cols() const noexcept { return A.cols(); }
  [[nodiscard]] std::size_t num_inequality_rows() const noexcept {
    return num_rows() - num_equality;
  }

  /// Objective value of a canonical point, in canonical (minimization) space.
  [[nodiscard]] Real objective(core::HostSpan<const Real> x) const noexcept;

  [[nodiscard]] bool validate() const noexcept;

  /// Verify the startability contract: no fixed column, no all-zero row.
  ///
  /// This is what Module 6 (Initializer) runs before constructing a
  /// SolverState. It is O(m + n + nnz) and exists because the canonicalizer is
  /// not the last stage to touch the model -- presolve reductions create new
  /// empty rows -- so establishing the contract once is not the same as it
  /// holding when the IPM finally sees the model.
  ///
  /// Returns the offending index in `bad_index` and whether it names a row.
  [[nodiscard]] bool is_ipm_startable(std::size_t* bad_index = nullptr,
                                      bool* bad_is_row = nullptr) const noexcept;
};

/// Result of canonicalizing: the working model plus its inverse map.
struct CanonicalResult {
  CanonicalProblem problem;
  TransformStack transforms;
};

/// Canonicalize `problem`. The input is not modified.
///
/// Fails on a model that is infeasible by inspection: a bound pair with lower
/// above upper, or an all-zero row whose right-hand side excludes zero. Both
/// are reported as definite verdicts rather than propagated into a model that
/// cannot be solved.
[[nodiscard]] core::Expected<CanonicalResult> canonicalize(
    const Problem& problem, const Options& options = {});

/// Map a canonical-space solution back to the original model.
///
/// Recovers primal values, row duals and both bound duals. Row duals pick up a
/// sign flip wherever a `>=` row was negated, and are then normalized to the
/// reporting convention CPLEX, Gurobi and HiGHS use -- a positive dual for a
/// `<=` row in a minimization. This is the single exit point of the pipeline,
/// so it is the one place that normalization belongs.
///
/// A column substituted out as fixed gets its value back directly and its
/// reduced cost computed from the recovered duals against the ORIGINAL data.
[[nodiscard]] core::Expected<Solution> recover_solution(
    const Problem& original, const CanonicalProblem& canonical,
    const TransformStack& transforms, const Solution& canonical_solution);

}  // namespace sovsolve::model

#endif  // SOVSOLVE_MODEL_CANONICAL_HPP
