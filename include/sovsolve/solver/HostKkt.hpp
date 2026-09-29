// Module 25 stage 4a: a host `KktSolver`, so the homogeneous algorithm can
// actually run.
//
// `HomogeneousNewton.hpp` takes the solve of
//
//     K = [ -Theta^-1   A'  ]
//         [   A         D_s ]
//
// as an injected interface, which left the question of who supplies one. This
// is the host answer, and it is MATRIX-FREE: eliminating `dx` from the first
// block row,
//
//     dx = Theta (A' dy - rx)
//     (A Theta A' + D_s) dy = ry + A Theta rx
//
// gives a symmetric positive definite system (`Theta > 0`, `D_s >= 0`), solved
// by conjugate gradients. Each CG iteration is one `A'v` and one `Av` and
// nothing else -- `A Theta A'` is never formed, which matters because a single
// dense column of `A` would make it dense (FORMULATION.md section 10.1; Netlib
// `israel` has 19).
//
// This is section 10.1's normal-equations reduction, and the GPU interior-point
// path already solves exactly this system the same way (`solve_spd_cg` in
// LinearSolver.cu). Reusing the shape rather than inventing one means the
// eventual GPU `KktSolver` is a port, not a redesign.
//
// The products go through `pdlp::MatVec`, which already exists for precisely
// this reason and already counts itself, so a CPU-versus-GPU comparison later
// reads one instrumentation rather than two.
//
// WHAT THIS INHERITS, and it is not small. Section 10.1 measures
// `cond(A Theta A') ~ 1/mu^2`: at `mu = 1e-10` the diagonal spans roughly 1e20
// while double precision resolves about 1e16. CG's iteration count grows with
// the square root of the condition number, so the late iterations are where
// this will struggle, and a Jacobi preconditioner (below) only flattens the
// diagonal, not the underlying spread. That limit is real and is reported
// rather than hidden -- `HsdResult` carries the CG iteration count so a run
// that stalls says why.

#ifndef SOVSOLVE_SOLVER_HOST_KKT_HPP
#define SOVSOLVE_SOLVER_HOST_KKT_HPP

#include <cstddef>
#include <vector>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/HomogeneousNewton.hpp"
#include "sovsolve/solver/pdlp/MatVec.hpp"

namespace sovsolve::solver {

using core::Real;

/// Controls for the inner CG solve.
struct HostKktOptions {
  /// Relative residual at which CG stops. Looser than the outer tolerance on
  /// purpose: an inexact Newton direction costs iterations, an over-solved one
  /// costs time on every iteration including the early ones where the direction
  /// barely matters.
  Real tolerance = 1e-10;

  /// Cap on CG iterations per solve. Hit routinely near convergence -- see the
  /// conditioning note in this header -- so it is a budget, not an assertion.
  std::size_t max_iterations = 5000;

  /// Floor on `Theta^-1` before inverting, FORMULATION.md section 10.1. Not
  /// optional: a free column contributes neither bound term, so its entry is
  /// exactly zero and `Theta` would be `inf`. The floor covers the whole
  /// diagonal, not only free columns, because any variable that ends strictly
  /// between its bounds drives its entry to zero too.
  Real theta_inv_floor = 1e-12;

  /// Dual regularization added to the normal-equations diagonal. Covers rank
  /// deficiency in `A`, which no floor on `Theta` can reach: if `A` has a
  /// dependent row then `A Theta A'` is singular for every `Theta`.
  Real delta_d = 1e-10;

  /// Factor retries with delta_d multiplied by 10 each time ([AG99] section 5).
  std::size_t regularization_retries = 6;
};

class NormalFactor;

/// Matrix-free `KktSolver` over the canonical problem's existing CSR/CSC pair.
///
/// Rebuilt each outer iteration, because `Theta` and `D_s` change every time
/// the iterate moves. Construction is O(nnz) -- it forms the Jacobi diagonal
/// and nothing else.
class HostKktSolver final : public KktSolver {
 public:
  /// With `factor`, the normal equations are factored here (its symbolic
  /// analysis is shared across iterations) and the factor preconditions CG;
  /// without one, or if factoring fails at every regularization level, CG
  /// runs with the Jacobi diagonal as before.
  HostKktSolver(const model::CanonicalProblem& problem, const HomogeneousBorder& border,
                const HostKktOptions& options, NormalFactor* factor = nullptr);

  /// True if this iteration's solves are preconditioned by a direct factor.
  [[nodiscard]] bool direct() const noexcept { return factor_ != nullptr; }
  /// The dual regularization actually used, after any escalation.
  [[nodiscard]] Real delta_d() const noexcept { return options_.delta_d; }

  [[nodiscard]] core::Status solve(core::HostSpan<const Real> rhs_x,
                                   core::HostSpan<const Real> rhs_y,
                                   core::HostSpan<Real> dx,
                                   core::HostSpan<Real> dy) override;

  /// Total CG iterations across every solve on this object. The honest cost
  /// measure for this engine, in the same spirit as PDLP's KKT passes.
  [[nodiscard]] std::size_t cg_iterations() const noexcept { return cg_iterations_; }

  /// True if any solve exhausted `max_iterations` without reaching tolerance.
  /// The direction is still returned -- an inexact Newton step is usable and
  /// the step-length test will judge it -- but the caller is told.
  [[nodiscard]] bool hit_iteration_cap() const noexcept { return hit_cap_; }

 private:
  void apply(const core::RealVector& in, core::RealVector& out);
  void precondition();  ///< zvec_ <- P^-1 r_

  const model::CanonicalProblem* problem_;
  pdlp::HostMatVec matvec_;
  HostKktOptions options_;
  NormalFactor* factor_ = nullptr;  ///< null: Jacobi preconditioning
  std::vector<Real> precond_;       ///< m, scratch for the factor solve
  std::size_t factor_failures_ = 0; ///< factor solves that overflowed

  core::RealVector theta_;      ///< n, the inverted and floored diagonal
  core::RealVector d_slack_;    ///< m, zero on equality rows
  core::RealVector jacobi_;     ///< m, 1 / diag(A Theta A' + D_s + delta_d)

  // CG working vectors, allocated once.
  core::RealVector r_, p_, ap_, zvec_, scratch_n_, rhs_;

  std::size_t cg_iterations_ = 0;
  bool hit_cap_ = false;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_HOST_KKT_HPP
