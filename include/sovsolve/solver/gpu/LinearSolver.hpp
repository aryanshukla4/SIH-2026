// Module 12. Solves the KktSystem Module 9 assembled.
//
// CURRENT IMPLEMENTATION: dense, via cuSOLVER's classic LU path
// (cusolverDnDgetrf/Dgetrs) -- a deliberate stopgap, not the intended final
// design. Two real gaps, both documented rather than hidden:
//
//   * Dense, not sparse. No cuDSS is installed on the development machine
//     (README.md "Solver core"), and a from-scratch sparse LDL^T
//     factorization (ordering, elimination tree, numerical factorization) is
//     substantial separate work. Converting the (n+m)x(n+m) KKT matrix to a
//     dense buffer costs O(dim^2) memory and O(dim^3) time per solve --
//     fine for small hand-built test problems, prohibitive for a real
//     Netlib/MIPLIB instance. `SymbolicFactorization` (Ordering.hpp) exists
//     for the eventual sparse path and is accepted here as an optional,
//     currently-ignored pointer rather than removed from the signature.
//
//   * General LU, not symmetric-indefinite LDL^T. The KKT matrix is
//     symmetric quasi-definite (FORMULATION.md 10.2), which cusolverDnDsytrf
//     could exploit for roughly half the factorization cost -- but
//     cusolverDnDgetrf/Dgetrs are the longest-standing, most stable dense
//     solve pair in cuSOLVER's classic API, which matters more than the 2x
//     for a stopgap that is going to be replaced by a sparse factorization
//     anyway.
//
// Iterative refinement is NOT implemented this pass (`refinement_passes` is
// always 0): module.txt Module 12 / FORMULATION.md 10.3 specify refinement
// against the UNREGULARIZED residual specifically, which needs the true
// (non-regularized) Newton system's residual, not just the factored matrix's
// own residual -- a real piece of design deferred along with the sparse
// path, not silently approximated.

#ifndef SOVSOLVE_SOLVER_GPU_LINEAR_SOLVER_HPP
#define SOVSOLVE_SOLVER_GPU_LINEAR_SOLVER_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/solver/KktSystem.hpp"
#include "sovsolve/solver/gpu/Ordering.hpp"

namespace sovsolve::solver::gpu {

using core::Expected;
using core::Real;
using core::RealVector;
using core::Status;

struct LinearSolveResult {
  RealVector solution;
  std::size_t refinement_passes = 0;

  /// max|U_ii| / min|U_ii| from the Dgetrf factor -- read off for free
  /// (no extra solve), the standard cheap proxy for how ill-conditioned the
  /// factorization was. cuSOLVER's `info` only catches exact singularity;
  /// this catches the "solved without error but the answer is noise" case
  /// (see the comment on Options::IpmOptions::max_pivot_ratio).
  Real pivot_ratio = 1.0;
};

/// `symbolic` is accepted but unused by the current dense implementation --
/// pass `nullptr` until the sparse path exists. `max_refinement_steps` is
/// likewise unused (refinement is deferred, see the file comment).
[[nodiscard]] Expected<LinearSolveResult> solve(const KktSystem& system,
                                                  const SymbolicFactorization* symbolic,
                                                  int max_refinement_steps);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_LINEAR_SOLVER_HPP
