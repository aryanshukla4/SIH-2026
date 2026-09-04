// Module 9. Builds Theta^-1 = X_L^-1 Z + (U-X)^-1 V first, floors it
// elementwise at delta_p, only then inverts -- never inverts a possibly-zero
// diagonal directly (module.txt Module 9).

#ifndef SOVSOLVE_SOLVER_GPU_KKT_BUILDER_HPP
#define SOVSOLVE_SOLVER_GPU_KKT_BUILDER_HPP

#include "sovsolve/analysis/MatrixAnalysis.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/KktSystem.hpp"
#include "sovsolve/solver/Residuals.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Real;
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
[[nodiscard]] Status build_kkt(const CanonicalProblem& problem, const SolverState& state,
                                const Residuals& residuals,
                                const analysis::MatrixAnalysis& mat_analysis,
                                Real delta_p, Real delta_d, KktSystem& out);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_KKT_BUILDER_HPP
