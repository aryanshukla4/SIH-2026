// Incomplete Cholesky, IC(0), preconditioner for the matrix-free CG solve
// (LinearSolver.cu's solve_spd_cg) of M = A*T*A' + D_s + delta_d*I.
//
// One level up from Jacobi (solve_spd_cg's original preconditioner, still the
// fallback here) in both power and cost: Jacobi treats every row in
// isolation, this captures how rows interact through columns they share, at
// the cost of two triangular solves per CG iteration instead of one
// elementwise divide.
//
// cuSPARSE provides the actual factorization and triangular-solve primitives
// (`cusparseDcsric02`, `cusparseDcsrsv2_*`) -- primitive OPERATIONS (factor a
// matrix I hand it, solve a triangular system), the same tier as the SpMV
// calls solve_spd_cg already relies on, not a solver library. The choice to
// precondition CG this way, and the dense-column handling below, are not
// library-provided.
//
// -----------------------------------------------------------------------
// Dense columns, faced directly
// -----------------------------------------------------------------------
//
// M's true sparsity pattern is that of A*A' -- and a single DENSE column
// (MatrixAnalysis::dense_columns) contributes a fill entry between every
// pair of rows it touches, an O((rows touching it)^2) blowup exactly as
// FORMULATION.md's section 10.1 caveat already documents for forming A*T*A'
// explicitly. This preconditioner EXCLUDES dense columns from the fill
// computation entirely -- their off-diagonal contribution to M is dropped,
// but their DIAGONAL contribution (cheap and exact regardless of density)
// is still folded in, the same value the Jacobi preconditioner already
// computes. This makes the preconditioner strictly worse than a complete
// (Sherman-Morrison-Woodbury-corrected) version on dense-column-heavy
// instances (`israel`, 19 dense columns) and strictly better than plain
// Jacobi everywhere else -- an honest, bounded improvement, not a claimed-
// complete one. SMW correction is a real, deferred follow-up, not something
// this file silently gets wrong.

#ifndef SOVSOLVE_SOLVER_GPU_PRECONDITIONER_HPP
#define SOVSOLVE_SOLVER_GPU_PRECONDITIONER_HPP

#include <cstddef>
#include <vector>

#include "sovsolve/core/SparseMatrix.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"

namespace sovsolve::solver::gpu {

using core::Real;
using core::RealVector;
using core::Status;

/// (Re)builds the IC(0) factorization of `M = A*T*A' + D_s + delta_d*I` for
/// this Newton solve. `a` is the SAME sparse `A` `solve_spd_cg` already has
/// via `NormalEquationsSystem::a`; `theta`/`diag_add` are that struct's own
/// fields. `is_dense_column` (size `a.cols()`) marks which columns to
/// exclude from the fill computation -- build it once from
/// `analysis::analyze(a).dense_columns` and reuse it; this function does not
/// call `analyze()` itself; it does not know */"dense"/* means anything, only
/// which columns to skip.
///
/// Internally caches everything keyed by `(a.rows(), a.nnz())`, matching the
/// `PersistentSolverContext`-style idiom already used throughout
/// `LinearSolver.cu` -- device allocation and the (expensive) symbolic fill
/// pattern only happen when the problem's dimensions actually change; every
/// call still redoes the numeric accumulation and the incomplete
/// factorization itself, since `theta`/`diag_add` change every Newton solve.
///
/// Returns `core::ErrorCode::NumericalError` if IC(0) hits a structural zero
/// pivot -- a known, real IC(0) failure mode (the dropped fill can leave an
/// approximate factor less well-behaved than the true matrix regularization
/// already made positive definite), NOT a bug to chase. The caller
/// (`solve_spd_cg`) falls back to the Jacobi preconditioner for that one
/// solve rather than treating it as fatal.
[[nodiscard]] Status ic0_build(const core::SparseMatrixPair<>& a, const RealVector& theta,
                               const RealVector& diag_add,
                               const std::vector<bool>& is_dense_column);

/// `z = M_ic0^-1 * r` via two triangular solves against the factor the most
/// recent successful `ic0_build` produced. `r`/`z` are DEVICE pointers,
/// length `a.rows()` from that call (may alias: `z` is fully overwritten
/// before being read back for the second solve internally uses its own
/// scratch buffer, not `r`/`z` themselves, so aliasing `r == z` is safe).
[[nodiscard]] Status ic0_apply(const Real* r_device, Real* z_device, std::size_t m);

/// True when this build links cuDSS (SOVSOLVE_HAVE_CUDSS).
[[nodiscard]] bool cudss_available() noexcept;

/// EXACT sparse Cholesky of the same matrix `ic0_build` approximates --
/// `A_S T A_S^T + D` with dense columns kept to the diagonal -- through
/// NVIDIA cuDSS, a sparse direct linear solver (a linear-algebra library, not
/// an LP solver). Used as `solve_spd_cg`'s preconditioner, so CG converges in
/// a few iterations and its dense-column and refinement duties stay with CG.
/// The reordering and symbolic analysis are cached while the pattern is
/// unchanged, which it is for every Newton solve of one interior-point run.
///
/// Returns an error when cuDSS is not linked or the factorization reports a
/// non-positive pivot; the caller then falls back to IC(0), then Jacobi.
[[nodiscard]] Status cudss_build(const core::SparseMatrixPair<>& a, const RealVector& theta,
                                 const RealVector& diag_add,
                                 const std::vector<bool>& is_dense_column);

/// `z = M^-1 r` with the factor the last successful `cudss_build` produced.
/// DEVICE pointers of length m; `r == z` is allowed.
[[nodiscard]] Status cudss_apply(const Real* r_device, Real* z_device, std::size_t m);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_PRECONDITIONER_HPP
