// Module 6: builds the strictly-interior starting SolverState, and owns the
// re-verification of the startability contract -- Module 2 (Canonicalizer)
// establishes it, but presolve reductions can create new empty rows, so it
// must be re-checked here, immediately before it is relied on.

#ifndef SOVSOLVE_SOLVER_INITIALIZER_HPP
#define SOVSOLVE_SOLVER_INITIALIZER_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver {

using core::Expected;
using model::CanonicalProblem;
using model::Options;

/// Builds a strictly-interior `SolverState` for `problem`.
///
/// Calls `CanonicalProblem::is_ipm_startable()` first and fails with the
/// offending row/column named in the error, rather than constructing a state
/// that cannot take a first Newton step.
///
/// The point itself is the textbook bound-midpoint heuristic, not full
/// Mehrotra initialization (Mehrotra's method solves two least-squares
/// systems for x0/y0, which needs a working linear solver -- Module 12 is
/// still a stub, see gpu/LinearSolver.hpp):
///
///     boxed column      x_j = (l_j + u_j) / 2
///     lower-only        x_j = l_j + 1
///     upper-only        x_j = u_j - 1
///     free               x_j = 0
///     z_j = 1 where l_j finite, else 0; v_j = 1 where u_j finite, else 0
///     s_k = 1 for every inequality row
///     y_i = 0 on equality rows (unrestricted in sign); y_i = -1 on
///           inequality rows -- FORMULATION.md 3 requires -y_I > 0 strictly,
///           and 0 would also make the KKT builder's D_s = s/(-y_I) divide
///           by zero on iteration 0
///
/// This is strictly interior for any startable model (no fixed column, no
/// all-zero row) but is not feasible -- `gpu::compute_residuals` on the
/// result will show nonzero `rp`. `mu` is seeded from the same point via the
/// Module 16 formula (`FORMULATION.md`), so the first residual call has a
/// real target rather than an arbitrary constant.
[[nodiscard]] Expected<SolverState> initialize(const CanonicalProblem& problem,
                                                const Options& options);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_INITIALIZER_HPP
