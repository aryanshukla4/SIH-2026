// Module 28, stage 8b: cutting plane separation at the root, in canonical
// space, for the simplex branch-and-bound.
//
// SOURCES, transcribed:
//
//   [W]    K. Wolter, "Implementation of Cutting Plane Separators for Mixed
//          Integer Programs", diploma thesis, TU Berlin (2006) -- [CIP]'s
//          reference [218]; SCIP's separators.
//   [CIP]  T. Achterberg, "Constraint Integer Programming", PhD thesis,
//          TU Berlin (2007), chapter 8 and section 3.3.8.
//
//   c-MIR     [W] chapter 3, the "resulting algorithm (fast version)":
//             Algorithm 3.1 with the aggregation heuristic of Algorithm 3.2
//             under Score Type 3, the bound substitution of Algorithm 3.3
//             (Criteria F3 and S3), and the cut generation of Algorithm 3.4
//             (Procedure 1 with the extended candidate set for delta);
//             MAXAGGR = 5, MAXFAILS = 150, MAXCONTS = 20, MAXCUTS = 100,
//             MAXTESTDELTA = 10; section 3.3.4's numerics (aggregation factors
//             within a ratio of 10^4, f_beta in [0.05, 0.95]).
//   GMI       [CIP] (8.4)-(8.5) -- the MIR of Proposition 8.3 applied to a
//             simplex tableau row -- with [W] section 6.1's safeguards: skip
//             rows whose fractional part is below 0.05, scale each cut so its
//             integer-variable coefficients are integral (denominators at
//             most 1000) and discard it if that scalar exceeds 1000.
//   selection [CIP] Algorithm 3.2: efficacy + 0.1 objective parallelism +
//             orthogonality, minimum orthogonality 0.5.
//
// WHERE INTEGRALITY LIVES. A canonical column is `x_original / s` (scaling,
// Scaler.cpp), so an integer column is integral in ORIGINAL units. The
// separators work in those units -- v_j = s_j x_j -- and translate every cut
// back to canonical x before it leaves this module.

#ifndef SOVSOLVE_SOLVER_MILP_SEPARATORS_HPP
#define SOVSOLVE_SOLVER_MILP_SEPARATORS_HPP

#include <cstddef>
#include <utility>
#include <vector>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/simplex/Basis.hpp"
#include "sovsolve/solver/simplex/SimplexResult.hpp"

namespace sovsolve::solver {

using core::Real;

/// `sum_j terms[j].second * x_{terms[j].first} <= rhs`, over canonical
/// structural columns.
struct Cut {
  std::vector<std::pair<std::size_t, Real>> terms;
  Real rhs = 0.0;
};

/// What the separators read. The pointers must outlive the call.
struct SeparationInput {
  const model::CanonicalProblem* problem = nullptr;
  /// Per canonical column: `s` for an integer column (`x_original = s x`),
  /// 0 for a continuous one.
  const std::vector<Real>* integer_scale = nullptr;
  /// Global canonical column bounds.
  const std::vector<Real>* lower = nullptr;
  const std::vector<Real>* upper = nullptr;
  /// MilpOptions::cut_violation_margin.
  Real violation_margin = 1e-6;
};

/// [W] Algorithm 3.1 bookkeeping carried across rounds: how often each row has
/// been aggregated (the `l` of Score Type 3).
struct CmirState {
  std::vector<std::size_t> aggregations;
};

/// Gomory mixed integer cuts from the optimal basis of `lp`, violated by it.
[[nodiscard]] std::vector<Cut> separate_gomory(const SeparationInput& in,
                                               const simplex::SimplexResult& lp);

/// Complemented MIR cuts violated by `lp`'s point. `round` is the number of
/// separation rounds already performed (MAXFAILS grows in early rounds).
[[nodiscard]] std::vector<Cut> separate_cmir(const SeparationInput& in,
                                             const simplex::SimplexResult& lp,
                                             std::size_t round, CmirState& state);

/// [CIP] Algorithm 3.2 over `cuts`, at `x` (canonical structural values).
[[nodiscard]] std::vector<Cut> select_cuts(std::vector<Cut> cuts,
                                           const model::CanonicalProblem& problem,
                                           const std::vector<Real>& x, std::size_t max_cuts);

/// Append `cuts` as inequality rows at the end of `problem` -- equality rows
/// stay first, every existing row, column and logical keeps its index -- and,
/// if given, extend `basis` with the new rows' logicals as basic.
[[nodiscard]] core::Status append_cuts(model::CanonicalProblem& problem,
                                       const std::vector<Cut>& cuts, simplex::Basis* basis);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_MILP_SEPARATORS_HPP
