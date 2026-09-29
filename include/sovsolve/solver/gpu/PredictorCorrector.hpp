// Module 8: orchestrates one Mehrotra predictor-corrector iteration.
//
// Lives under gpu/, not src/solver/ directly: it calls Modules 7, 9, 10, 12,
// 13, 14, 15 and 16, all of which are GPU-boundary functions, so it belongs
// in sovsolve_solver_gpu for the same reason they do (see the module list in
// docs/spec/architecture.txt "## GPU boundary" and Regularization.hpp's header
// comment on why this avoids a link cycle between sovsolve_solver and
// sovsolve_solver_gpu).
//
// docs/spec/module.txt Module 8: "affine solve, affine primal/dual step lengths,
// mu_aff, sigma = clamp((mu_aff/mu)^3, 0, 1), corrector RHS including cross
// terms for lower/upper/slack pairs, corrector solve, final directions,
// separate primal/dual step lengths." Sequence, when
// Options::IpmOptions::predictor_corrector is on (the default):
//
//   1. update_mu and compute_residuals at the CURRENT mu -- done by the
//      CALLER now, not here (see this file's doc comment on `residuals`):
//      the outer loop already needs exactly this pair for its own
//      convergence check immediately before calling run_iteration, so
//      recomputing it again here was pure waste.
//   2. (record `residuals` for diagnostics -- the caller's value, unchanged.)
//   3. compute_residuals at mu=0 -- Mehrotra's "affine"/prediction target.
//   4. build_kkt + solve + recover_newton_direction on that -- the affine
//      direction, cloned into state.dx_aff/ds_aff/dy_aff/dz_aff/dv_aff.
//   5. compute_step_lengths with eta=1 (no safety factor -- this step is
//      never taken, only used to measure how far the affine direction could
//      go) -- then compute_mu_at_trial_point at that trial point -> mu_aff,
//      and sigma = clamp((mu_aff/mu)^3, 0, 1).
//   6. compute_residuals at sigma*mu -- the corrector target -- then add
//      Mehrotra's second-order cross terms by hand (not something
//      compute_residuals does, since it is a Module-8-specific correction,
//      not part of the six-block Newton system itself):
//
//          rxz += dx_aff .* dz_aff        (finite lower bound only)
//          ruv -= dx_aff .* dv_aff        (finite upper bound only)
//          rsy -= ds_aff .* dy_aff_I
//
//      Derived by linearizing (x-l+dx)(z+dz) = sigma*mu (and the equivalent
//      pairs for u-x,v and s,-y_I) around the CURRENT point, substituting
//      the already-computed dx_aff/dz_aff/etc. for the second-order term
//      instead of dropping it -- that substitution is exactly what turns a
//      plain Newton step into Mehrotra's predictor-corrector step. Verified
//      empirically in solver_gpu_algorithms_test.cpp against the corrector
//      Newton system, the same way NewtonRecovery.hpp's gap is verified.
//   7. build_kkt + solve + recover_newton_direction again, on the corrector
//      system -- this OVERWRITES state.dx/ds/dy/dz/dv with the final
//      direction. NOT reused from step 4's factorization (docs/spec/module.txt
//      Module 8's "reuse structure where valid" is not implemented -- the
//      dense cuSOLVER stopgap in LinearSolver.hpp factorizes from scratch
//      both times; a real factorization-reuse path needs the sparse solver
//      this project does not have yet).
//   8. compute_step_lengths with the real `eta` (Options::IpmOptions::eta,
//      0.995 default) -- the step actually taken.
//   9. apply_step.
//
// When predictor_corrector is off, steps 3-5 are skipped and sigma is the
// fixed Options::IpmOptions::sigma; step 6 has no cross terms to add.

#ifndef SOVSOLVE_SOLVER_GPU_PREDICTOR_CORRECTOR_HPP
#define SOVSOLVE_SOLVER_GPU_PREDICTOR_CORRECTOR_HPP

#include "sovsolve/core/Status.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/Diagnostics.hpp"
#include "sovsolve/solver/Regularization.hpp"
#include "sovsolve/solver/Residuals.hpp"
#include "sovsolve/solver/SolverState.hpp"

namespace sovsolve::solver::gpu {

using core::Status;
using model::CanonicalProblem;
using model::Options;
using solver::IterationRecord;
using solver::RegularizationController;
using solver::Residuals;

/// Runs one predictor-corrector iteration in place on `state`, recording the
/// result into `record`. `residuals` is the caller's ALREADY-COMPUTED
/// residual at the state's CURRENT mu -- the outer loop (gpu::solve_problem,
/// Solve.hpp) computes exactly this, at exactly this state, one line before
/// calling here, to run Module 18's convergence check. This used to be
/// recomputed a second time internally (a real, wasted SpMV-based pass every
/// single iteration of every solve); passing it in instead removes that
/// redundancy. Only used for `record`'s diagnostic fields -- the affine and
/// corrector residuals this function needs for the actual Newton systems
/// (at mu=0 and sigma*mu respectively) are distinct values, still computed
/// fresh here, since neither is ever available from the caller.
///
/// Does not check convergence -- that is Module 18 (Convergence Checker),
/// driven by the outer loop. DOES escalate/refactor `regularization`
/// internally on breakdown (PredictorCorrector.cu's solve_newton_system) and
/// decay it on a clean solve, once per affine/corrector KKT solve -- see the
/// doc comment on Options::IpmOptions::regularization_escalation.
[[nodiscard]] Status run_iteration(const CanonicalProblem& problem, const Options& options,
                                    RegularizationController& regularization,
                                    SolverState& state, const Residuals& residuals,
                                    IterationRecord& record);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_PREDICTOR_CORRECTOR_HPP
