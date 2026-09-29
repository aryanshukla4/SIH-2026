// Module 9. Builds Theta^-1 = X_L^-1 Z + (U-X)^-1 V first, floors it
// elementwise at delta_p, only then inverts -- never inverts a possibly-zero
// diagonal directly (docs/spec/module.txt Module 9).

#ifndef SOVSOLVE_SOLVER_GPU_KKT_BUILDER_HPP
#define SOVSOLVE_SOLVER_GPU_KKT_BUILDER_HPP

#include "sovsolve/analysis/MatrixAnalysis.hpp"
#include "sovsolve/core/SparseMatrix.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/KktSystem.hpp"
#include "sovsolve/solver/Residuals.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Real;
using core::RealVector;
using core::Status;
using model::CanonicalProblem;

/// Builds the augmented/quasi-definite KKT system (FORMULATION.md 10.2):
///
///     | -(Q + T^-1 + delta_p I)          A^T             | | dx |   | rhs1 |
///     |            A                (D_s + delta_d I)    | | dy | = | rhs2 |
///
/// derived from the six-block Newton system (FORMULATION.md 7) by
/// eliminating dz, dv (via the bound-complementarity rows) and ds (via the
/// slack-complementarity row):
///
///     T^-1_j = z_j/(x_j-l_j) + v_j/(u_j-x_j)      (term omitted, infinite bound)
///     D_s_k  = s_k / (-y_I_k)                     (0 on equality rows)
///     rhs1_j = rd_j + rxz_j/(x_j-l_j) - ruv_j/(u_j-x_j)
///     rhs2_i = -rp_E_i                            (equality rows)
///     rhs2_k = -(rp_I_k + rsy_k / y_I_k)           (inequality rows)
///
/// `T^-1` is NOT floored at `delta_p` here -- on this path it is never
/// inverted, so the 10.1 floor does not apply; `delta_p` is added to the
/// diagonal directly instead, which is why it appears in both places above.
///
/// `mat_analysis` must be the analysis of the model actually being
/// factorized (post-presolve, post-scaling): presolve changes
/// free_columns/dense_columns/empty_rows. It is currently unused --
/// `ReductionType::QpAugmentedKkt` is selected unconditionally, because the
/// alternative (`LpNormalEquationsDy`, FORMULATION.md 10.1) requires a
/// sparse `A * Theta * A^T` product this pass does not build. See
/// `KktSystem::descriptor.reason` on the result for the recorded justification.
///
/// `out.precond_diag` (length n+m) is filled alongside the matrix itself: the
/// (1,1) block's diagonal magnitude `|theta_inv_j + delta_p|` for the first
/// `n` entries, the (2,2) block's diagonal `D_s_i + delta_d` for the last
/// `m` -- exactly the scalar quantities this function already computes for
/// the matrix's own diagonal, exposed separately because `LinearSolver.cu`'s
/// `solve_minres` needs them as a block-Jacobi preconditioner without
/// re-deriving them or extracting them back out of the sparse matrix.
[[nodiscard]] Status build_kkt(const CanonicalProblem& problem, const SolverState& state,
                                const Residuals& residuals,
                                const analysis::MatrixAnalysis& mat_analysis,
                                Real delta_p, Real delta_d, KktSystem& out);

/// Matrix-free normal-equations reduction (FORMULATION.md 10.1), LP only (`Q`
/// empty -- caller's responsibility, not re-checked here, same as `build_kkt`
/// does not re-check that `mat_analysis` is fresh). Eliminates `dx` to leave
/// an SPD `m x m` system for `dy`:
///
///     (A T A^T + D_s + delta_d I) dy  =  rhs2 + A T rhs1
///
/// derived from the augmented system (KktBuilder.hpp's `build_kkt` doc
/// comment) by substituting its block-1 equation, `dx = T*(A^T dy - rhs1)`,
/// into its block-2 equation. Unlike the augmented path, `T^-1` here IS
/// floored before inverting (FORMULATION.md 10.1 -- this is the one place the
/// two paths' diagonal handling genuinely differs, not just a relabeling):
///
///     T^-1_j = z_j/(x_j-l_j) + v_j/(u_j-x_j)      (term omitted, infinite bound)
///     T^-1_j <- max(T^-1_j, delta_p)               floor, unconditional
///     T_j    <- 1 / T^-1_j
///
/// `A T A^T` is never formed, dense OR sparse -- `LinearSolver.cu`'s
/// `solve_spd_cg` applies it as an operator (two sparse matrix-vector
/// products via cuSPARSE) inside a matrix-free Conjugate Gradient solve, so
/// there is no `O(m^2)`/`O(mn)` construction cost and no `O(m^3)`
/// factorization -- see that function's doc comment.
struct NormalEquationsSystem {
  /// Non-owning: `build_normal_equations`'s caller (`PredictorCorrector.cu`)
  /// holds `problem` for the entire Newton solve this struct is scoped to,
  /// which outlives every use of this pointer. Needed because `solve_spd_cg`
  /// applies `A` and `A^T` directly via cuSPARSE (never densifies), so the
  /// sparse matrix itself has to travel with the rest of the system.
  const core::SparseMatrixPair<>* a = nullptr;
  RealVector theta;      ///< length n, T (already floored & inverted)
  RealVector rhs1;       ///< length n, identical formula to build_kkt's rhs1
  RealVector rhs;        ///< length m, the REDUCED system's right-hand side
  RealVector diag_add;   ///< length m, D_s + delta_d*I, added after A*T*A^T
  ReductionDescriptor descriptor;
};

[[nodiscard]] Status build_normal_equations(const CanonicalProblem& problem,
                                             const SolverState& state,
                                             const Residuals& residuals, Real delta_p,
                                             Real delta_d, NormalEquationsSystem& out);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_KKT_BUILDER_HPP
