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
//     subject to  A x = b
//                 x >= 0
//
// Every row is an equality and every variable is non-negative. The handoff
// form `Ax + s = b, x >= 0, s >= 0` is exactly this with the slacks appended
// as columns: `A_canonical = [A_structural | I_slack]` and
// `x_canonical = [x ; s]`.
//
// Writing it that way rather than as a separate `s` vector matters, because
// the handoff form cannot express an equality row -- there is nowhere to put a
// slack that must be zero. Real models are full of equality rows (516 of the
// 821 rows in Netlib 25fv47), so a formulation that cannot represent them is
// not usable. Appending slack columns only to inequality rows handles both
// kinds uniformly.
//
// ---------------------------------------------------------------------------
// Column transforms are one affine map
// ---------------------------------------------------------------------------
//
// Every variable transform is an instance of `x = d*x' + t` with `d` in
// {+1, -1}, so the whole objective and matrix update is uniform:
//
//     A_new  = A D                         (scale column j by d_j)
//     bounds shift by  -A t
//     Q_new  = D Q D                       (Q_new[i][j] = d_i d_j Q[i][j])
//     c_new  = D (Q t + c)
//     offset = 1/2 t'Q t + c't
//
// Free variables are the one case outside this map: they are split into
// `x = xp - xm` with both parts non-negative.

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
  RealVector c;                    ///< length num_cols()
  core::SparseMatrixPair<> A;      ///< num_rows() x num_cols(), all equalities
  core::SparseMatrixPair<> Q;      ///< empty for LP
  RealVector b;                    ///< length num_rows()

  /// Column layout: `[structural | slack | bound]`.
  ///
  /// - `structural` are the transformed original variables, plus one extra
  ///   column for each free variable that was split.
  /// - `slack` are the `s` of `Ax + s = b`, one per inequality row.
  /// - `bound` are the auxiliary variables `t` introduced by bound rows, one
  ///   per finite upper bound that had to become a constraint.
  std::size_t num_structural = 0;
  std::size_t num_slack = 0;
  std::size_t num_bound = 0;

  /// Constant folded out of the objective by the shifts, in canonical
  /// (minimization) space. Added back during recovery.
  Real obj_offset = 0.0;

  /// True when the original model was a maximization and `c`/`Q` were negated.
  bool objective_negated = false;

  [[nodiscard]] std::size_t num_rows() const noexcept { return A.rows(); }
  [[nodiscard]] std::size_t num_cols() const noexcept { return A.cols(); }

  /// First index of the slack block, i.e. where `s` begins inside `x`.
  [[nodiscard]] std::size_t slack_begin() const noexcept { return num_structural; }
  [[nodiscard]] std::size_t bound_begin() const noexcept {
    return num_structural + num_slack;
  }

  /// Objective value of a canonical point, in canonical (minimization) space.
  [[nodiscard]] Real objective(core::HostSpan<const Real> x) const noexcept;

  [[nodiscard]] bool validate() const noexcept;
};

/// Result of canonicalizing: the working model plus its inverse map.
struct CanonicalResult {
  CanonicalProblem problem;
  TransformStack transforms;
};

/// Canonicalize `problem`. The input is not modified.
///
/// Fails only on a model that is already infeasible by inspection (a bound
/// pair with lower above upper), which is reported rather than propagated into
/// a canonical model that cannot be solved.
[[nodiscard]] core::Expected<CanonicalResult> canonicalize(
    const Problem& problem, const Options& options = {});

/// Map a canonical-space solution back to the original model.
///
/// Recovers primal values, row duals and reduced costs. Row duals pick up a
/// sign flip wherever a `>=` row was negated; reduced costs pick up the dual
/// of any bound row that was introduced for that column, since the bound row
/// is carrying what was originally a bound multiplier.
[[nodiscard]] core::Expected<Solution> recover_solution(
    const Problem& original, const CanonicalProblem& canonical,
    const TransformStack& transforms, const Solution& canonical_solution);

}  // namespace sovsolve::model

#endif  // SOVSOLVE_MODEL_CANONICAL_HPP
