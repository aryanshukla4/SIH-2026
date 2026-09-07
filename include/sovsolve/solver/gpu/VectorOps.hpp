// Small elementwise device-vector kernels shared by the matrix-free Krylov
// solvers (LinearSolver.cu's solve_spd_cg / solve_minres).
//
// Neither cuBLAS nor cuSPARSE has a vector Hadamard-product primitive (BLAS
// predates the need for one) -- applying a diagonal scaling (e.g. Theta, or a
// regularization diagonal) to a vector without ever materializing that
// diagonal as a matrix needs exactly this, and nothing more: one thread per
// element, no shared memory, no atomics, no synchronization. This is the one
// piece of hand-written CUDA the Krylov solvers need.

#ifndef SOVSOLVE_SOLVER_GPU_VECTOR_OPS_HPP
#define SOVSOLVE_SOLVER_GPU_VECTOR_OPS_HPP

#include <cstddef>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"

namespace sovsolve::solver::gpu {

using core::Real;
using core::Status;

/// `y[i] = a[i] * b[i]` for `i` in `[0, n)`. All three are DEVICE pointers.
[[nodiscard]] Status hadamard(const Real* a, const Real* b, Real* y, std::size_t n);

/// `y[i] += a[i] * b[i]` for `i` in `[0, n)` -- the fused multiply-add
/// sibling, used to fold a diagonal term (e.g. `D_s + delta_d*I`) into an
/// already-computed vector without a temporary buffer or a second launch.
[[nodiscard]] Status hadamard_add(const Real* a, const Real* b, Real* y, std::size_t n);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_VECTOR_OPS_HPP
