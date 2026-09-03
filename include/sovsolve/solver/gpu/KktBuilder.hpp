// Module 9. Builds Theta^-1 = X_L^-1 Z + (U-X)^-1 V first, floors it
// elementwise at delta_p, only then inverts -- never inverts a possibly-zero
// diagonal directly (module.txt Module 9).

#ifndef SOVSOLVE_SOLVER_GPU_KKT_BUILDER_HPP
#define SOVSOLVE_SOLVER_GPU_KKT_BUILDER_HPP

#include "sovsolve/analysis/MatrixAnalysis.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/KktSystem.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Real;
using core::Status;
using model::CanonicalProblem;

/// `mat_analysis` must be the analysis of the model actually being
/// factorized (post-presolve, post-scaling), not the ingestion-time
/// analysis: presolve changes free_columns/dense_columns/empty_rows.
///
/// STUB: signature and reduction-path selection only. Returns
/// ReductionType::LpNormalEquationsDy unconditionally for now; the
/// augmented-path selection rule (free-column share, non-diagonal Q) and the
/// actual Theta^-1 floor/assembly land next pass.
[[nodiscard]] Status build_kkt(const CanonicalProblem& problem, const SolverState& state,
                                const analysis::MatrixAnalysis& mat_analysis,
                                Real delta_p, Real delta_d, KktSystem& out);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_KKT_BUILDER_HPP
