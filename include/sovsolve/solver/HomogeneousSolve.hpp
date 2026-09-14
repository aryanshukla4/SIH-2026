// Module 25 stage 4: the homogeneous algorithm's iteration loop.
//
// Stages 1-3 built the pieces; this runs them in the order [AA] section 1.4
// specifies. One iteration is:
//
//   1. residuals and mu                          stage 1  (Homogeneous.hpp)
//   2. progress measures, stopping tests         stage 2  (HomogeneousStep.hpp)
//   3. the border quantities                     stage 3  (HomogeneousNewton.hpp)
//   4. K (p; q) = (h_x; -b), ONCE                stage 3  refresh_border_solve
//   5. predictor: gamma = 0, eta = 1             [AA] section 1.4.1
//   6. gamma from the predictor's step length    [AA] (1.12)
//   7. corrector: eta = 1 - gamma, Mehrotra term [AA] (1.13)
//   8. step length, centrality-tested            [AA] (1.20), (1.21)
//   9. update
//
// Steps 5 and 7 share step 4's solve, which is the whole reason the embedding
// costs one extra solve per ITERATION rather than per direction.
//
// WHY THIS IS A SEPARATE ENGINE RATHER THAN A FLAG ON THE IPM. The existing
// interior-point path is GPU-resident, in `sovsolve_solver_gpu`, and the
// gpu -> solver library edge is one-way (src/solver/CMakeLists.txt). A
// host-only module cannot call into it. So this is `Method::Hsd`, a fourth
// engine selected at the command line, sharing the canonicalize -> presolve ->
// scale -> reconstruct pipeline with the other three and injecting its own
// `KktSolver`. When the GPU backend is ready it supplies a different
// `KktSolver` and nothing here changes.
//
// WHAT TO EXPECT, stated up front so a weak first number is not read as a bug.
// The linear algebra underneath is `HostKkt.hpp`'s matrix-free CG on
// `A Theta A'`, whose condition number grows like `1/mu^2` (FORMULATION.md
// section 10.1). The late iterations are where that bites, and this engine
// carries no crossover, no Gondzio correctors and no elaborate starting point.
// The claim being tested by this stage is NOT "fastest engine"; it is "the
// interior-point family can now return Infeasible and Unbounded instead of
// running away", which is the gap README.md has recorded since Version 4.

#ifndef SOVSOLVE_SOLVER_HOMOGENEOUS_SOLVE_HPP
#define SOVSOLVE_SOLVER_HOMOGENEOUS_SOLVE_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/solver/Homogeneous.hpp"
#include "sovsolve/solver/HomogeneousStep.hpp"

namespace sovsolve::solver {

using core::Real;

struct HsdResult {
  core::SolverStatus status = core::SolverStatus::NotConverged;

  /// The iterate, already divided through by `tau` when the verdict is
  /// `Optimal`. On a certificate it is the RAY, undivided -- dividing by a
  /// `tau` near zero would amplify it into nonsense, and the ray is what the
  /// caller wants in that case anyway.
  core::RealVector x, s, y, z, v;

  Real objective = 0.0;

  Real tau = 1.0;
  Real kappa = 0.0;
  HomogeneousVerdict verdict = HomogeneousVerdict::Indeterminate;

  std::size_t iterations = 0;
  /// Solves of `K`. Expect `3 * iterations`: one border solve plus a predictor
  /// and a corrector. Reported so the [AA] section 1.5 cost claim is checkable
  /// on a real run rather than only in a unit test.
  std::size_t kkt_solves = 0;
  /// Total inner CG iterations. The honest work measure for this engine.
  std::size_t cg_iterations = 0;
  /// A CG solve gave up at its iteration cap. Not fatal -- an inexact Newton
  /// direction is still a direction -- but it is why a slow run is slow.
  bool inexact_solves = false;

  /// The stopping measures at the returned iterate, so a run that did not
  /// converge says how far it got.
  HomogeneousProgress progress;
};

/// Runs the homogeneous algorithm on an already canonicalized, presolved and
/// scaled problem.
[[nodiscard]] core::Expected<HsdResult> solve_hsd(const model::CanonicalProblem& problem,
                                                  const model::Options& options);

/// Canonical-space `Solution`, with quality metrics from the same shared
/// routine the simplex and PDLP use (solver/SolutionQuality.hpp), so the four
/// engines' reported numbers are comparable rather than merely similarly named.
[[nodiscard]] model::Solution to_canonical_solution(const model::CanonicalProblem& problem,
                                                    const HsdResult& result);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_HOMOGENEOUS_SOLVE_HPP
