// Module 12. Solves the KktSystem Module 9 assembled.
//
// PRODUCTION IMPLEMENTATION: matrix-free Krylov methods, never factoring
// anything. `solve_spd_cg` (Conjugate Gradient) handles the SPD
// normal-equations reduction; `solve_minres` (MINRES) handles the symmetric
// quasi-definite augmented KKT system. Both apply their matrix only as a
// sequence of sparse operations (cuSPARSE SpMV + a couple of small
// hand-written elementwise kernels, VectorOps.hpp) -- O(nnz) per Krylov
// iteration, no O(k^2)/O(k^3) term anywhere. Chosen specifically because a
// direct sparse factorization (the traditional alternative) needs ordering +
// symbolic + numeric factorization machinery that duplicates most of what
// cuDSS would provide, and PS 26119 rules out linking a solver library --
// cuBLAS/cuSPARSE stay in bounds as primitive *operations* (see README.md's
// "On the 'from scratch' constraint"), a complete sparse solver does not.
//
// `solve_dense` (renamed from the original `solve`) and `solve_spd_dense`
// (renamed from the original `solve_spd`) are KEPT, but only as correctness
// oracles for `tests/unit/solver_gpu_algorithms_test.cpp` -- dense LU/Cholesky
// factorization is verified-correct machinery with no iterative-convergence
// question attached, which is exactly what makes it useful for checking that
// the derived matrix-free algebra (dx-recovery, the reduced RHS, the operator
// itself) is right. Neither is called from `PredictorCorrector.cu` anymore.
//
// Iterative refinement is NOT implemented for any of the four paths
// (`refinement_passes` is always 0): docs/spec/module.txt Module 12 / FORMULATION.md
// 10.3 specify refinement against the UNREGULARIZED residual specifically,
// which needs the true (non-regularized) Newton system's residual -- a real
// piece of design deferred, not silently approximated. For the Krylov paths
// this matters less than it did for the dense ones: CG/MINRES already
// minimize the residual of whatever system they're handed, iteration by
// iteration, which is a related but not identical guarantee.

#ifndef SOVSOLVE_SOLVER_GPU_LINEAR_SOLVER_HPP
#define SOVSOLVE_SOLVER_GPU_LINEAR_SOLVER_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/solver/KktSystem.hpp"
#include "sovsolve/solver/gpu/KktBuilder.hpp"
#include "sovsolve/solver/gpu/Ordering.hpp"

namespace sovsolve::solver::gpu {

using core::Expected;
using core::Real;
using core::RealVector;
using core::Status;

struct LinearSolveResult {
  RealVector solution;
  std::size_t refinement_passes = 0;

  /// On the dense paths (`solve_dense`/`solve_spd_dense`): max|diag|/min|diag|
  /// off the factor -- the standard cheap ill-conditioning proxy (see the
  /// comment on `Options::IpmOptions::max_pivot_ratio`).
  ///
  /// On the Krylov paths (`solve_spd_cg`/`solve_minres`), which have no
  /// factorization and therefore no pivot to read: `1.0` if the solve
  /// converged within its tolerance, or `+infinity` if it hit its iteration
  /// cap without converging. `+infinity` always exceeds `max_pivot_ratio`
  /// regardless of its configured value, which is exactly the signal
  /// `PredictorCorrector.cu`'s existing escalate/refactor loop already knows
  /// how to act on -- "too ill-conditioned to solve at this regularization
  /// level" reuses the same downstream handling whether it came from a bad
  /// pivot or a non-converging Krylov solve.
  Real pivot_ratio = 1.0;
};

/// Dense LU (`cusolverDnDgetrf`/`Dgetrs`) solve of the full augmented KKT
/// system. Test-only correctness oracle for `solve_minres` -- see the file
/// comment. `symbolic` is accepted but unused; pass `nullptr`.
[[nodiscard]] Expected<LinearSolveResult> solve_dense(const KktSystem& system,
                                                       const SymbolicFactorization* symbolic,
                                                       int max_refinement_steps);

/// Dense Cholesky (`cusolverDnDpotrf`/`Dpotrs`) solve of the SPD
/// normal-equations system. Test-only correctness oracle for `solve_spd_cg`
/// -- see the file comment.
[[nodiscard]] Expected<LinearSolveResult> solve_spd_dense(const NormalEquationsSystem& system);

/// Matrix-free Conjugate Gradient solve of the SPD normal-equations system
/// (`NormalEquationsSystem`, KktBuilder.hpp) -- dimension `m`, never formed
/// as a matrix (dense or sparse). Applies `A T A^T + D_s + delta_d*I` as an
/// operator: `A^T * p` (cuSPARSE SpMV on `system.a->csc`, used directly as
/// CSR-of-`A^T`), elementwise-scaled by `theta` (VectorOps.hpp's `hadamard`),
/// then `A * (...)` (cuSPARSE SpMV on `system.a->csr`), then the diagonal
/// regularization folded in (`hadamard_add`) -- two sparse matrix-vector
/// products and two elementwise kernels per CG iteration, `O(nnz)` total,
/// zero `O(m^2)`/`O(m^3)` anywhere.
///
/// Preconditioned with Jacobi (diagonal): `diag_i = diag_add_i + sum_j
/// A_ij^2 * theta_j`, built once per call from `system.a`'s CSR rows,
/// `O(nnz)`. `LinearSolveResult::solution` is `dy` alone -- the caller
/// (`PredictorCorrector.cu`) recovers `dx` from it before calling
/// `recover_newton_direction`, same division of responsibility as the
/// former dense path.
///
/// `cg_tolerance`/`cg_max_iterations` are `Options::IpmOptions`'s
/// `cg_tolerance`/`cg_max_iterations` fields, passed as plain scalars rather
/// than the whole `Options` object -- matches this file's existing
/// minimal-parameter convention (`build_kkt` takes `delta_p`/`delta_d`
/// individually the same way).
/// `direct`: 0 IC(0)/Jacobi, 1 the exact cuDSS factor, 2 the exact in-house
/// SparseLdl factor (IpmOptions::direct); a failed factor falls back to IC(0).
[[nodiscard]] Expected<LinearSolveResult> solve_spd_cg(const NormalEquationsSystem& system,
                                                        Real cg_tolerance, int cg_max_iterations,
                                                        int direct = 2);

/// Matrix-free MINRES solve of the symmetric quasi-definite augmented KKT
/// system (`KktSystem`, already sparse -- `build_kkt` never densifies it,
/// only this function's dense predecessor `solve_dense` did). One cuSPARSE
/// SpMV per iteration on `system.matrix.csr` directly: both triangles are
/// stored explicitly (`build_kkt` inserts `(2,1)=A` and `(1,2)=A^T`
/// separately), so a plain non-transposed SpMV over the whole stored matrix
/// already computes the correct full product -- no operator composition
/// needed here the way CG's rectangular `A` needs one.
///
/// Preconditioned with block-Jacobi, using `system.precond_diag` (built by
/// `build_kkt` from the same `theta_inv`/`D_s` scalars it already computes
/// for the matrix's own diagonal -- see that function's doc comment).
///
/// CG doesn't apply here: the matrix is indefinite (one block negative
/// definite, one positive definite), not SPD. MINRES is CG's
/// indefinite-safe sibling -- same per-iteration cost class (one SpMV, a
/// handful of `Ddot`/`Daxpy` calls), different (Lanczos-based) recurrence.
[[nodiscard]] Expected<LinearSolveResult> solve_minres(const KktSystem& system,
                                                        Real minres_tolerance,
                                                        int minres_max_iterations);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_LINEAR_SOLVER_HPP
