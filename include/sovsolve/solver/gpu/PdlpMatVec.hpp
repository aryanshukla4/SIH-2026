// Module 24 stage E: PDLP's `K` and `K'` on the GPU, via cuSPARSE.
//
// WHY THIS IS THE ONLY GPU FILE PDLP NEEDS.
//
// The other two engines would each need a pile of ported machinery -- the
// interior-point path has a KKT builder, an ordering, a preconditioner and a
// linear solver; the simplex has a basis factorization with FTRAN/BTRAN and a
// product-form update, and does not parallelize anyway. PDLP has none of that.
// Its entire contact with the matrix is `K x` and `K' y`, so the GPU port IS
// this file, and everything else in `Pdlp.cpp` runs unchanged.
//
// WHY IT IS THE SHAPE THAT FITS THIS HARDWARE. docs/ARCHITECTURE-REVIEW.md 3.5
// records the target as an RTX 3050 Laptop (GA107, compute 8.6) -- confirmed by
// nvidia-smi: 4096 MiB, compute 8.6. Consumer Ampere carries 2 FP64 units per
// SM against 128 FP32, so FP64 runs at 1/64 of FP32 throughput. Put in roofline
// terms, with FP64 peak near 95 GFLOPS against roughly 192 GB/s of bandwidth,
// machine balance sits around 0.5 flop/byte:
//
//     SpMV (CSR)      8 B value + 4 B index per nonzero, 2 flops
//                     ~= 0.17 flop/byte   -> BELOW balance, memory-bound
//     factorization   O(n^3) flops on O(n^2) data
//                     -> far ABOVE balance, compute-bound
//
// A memory-bound kernel leaves those two crippled FP64 units idle waiting on
// DRAM regardless, so the 1/64 penalty never bites; a compute-bound one runs
// straight into the 95 GFLOPS ceiling, which is below what this laptop's CPU
// achieves with AVX2. That is the whole argument for PDLP on this machine, and
// this file is where it gets tested instead of asserted.
//
// WHAT 3.5's TWO NUMBERS ACTUALLY MEAN, since they are easy to conflate. Its
// "~2x (8 bytes vs 4)" is the FP64-versus-FP32 cost ON THE GPU -- doubles move
// twice the bytes, so a bandwidth-bound kernel is ~2x slower in FP64. It is NOT
// a GPU-versus-CPU speedup. That is a different quantity entirely (~192 GB/s
// against a laptop's ~51 GB/s of DDR4, so under 4x as a ceiling), and NOBODY
// HAS MEASURED IT HERE. Both entries in that table are spec sheet plus roofline
// reasoning. Producing the real number is the point of this stage.
//
// RESIDENCY IS THE DESIGN CONSTRAINT, not the kernel. `cusparseSpMV` is a
// library call; the way a port like this fails is by copying vectors back to
// the host every iteration, which costs more than the kernel saves. So:
//
//   - the matrix is uploaded ONCE, in the constructor, and never re-sent;
//   - both orientations are uploaded, because PDLP needs `K` and `K'` and a
//     transposed SpMV on CSR is far slower than a second CSR of the transpose
//     (which the canonicalizer already maintains as the CSC);
//   - the cuSPARSE workspace buffer is sized once per orientation and reused.
//
// Vectors still cross the bus each call, which is the honest limitation of
// entering through `MatVec`: the interface hands us host spans. That bounds
// what this stage can show, and the header on `products()` explains what the
// next step would be. Measuring the bounded version first is deliberate -- it
// is the change that touches nothing else, so a regression here is this file.
//
// 4 GB VRAM is not a constraint for PDLP, which is worth stating because it is
// a constraint for everything else: matrix-free means there is no fill-in and
// no factor to store, so the footprint is the matrix plus a handful of vectors.
// `datt256` at 95 MB has room to spare; a Cholesky factor of it would not.

#ifndef SOVSOLVE_SOLVER_GPU_PDLP_MAT_VEC_HPP
#define SOVSOLVE_SOLVER_GPU_PDLP_MAT_VEC_HPP

#include <cstddef>
#include <memory>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/pdlp/MatVec.hpp"

namespace sovsolve::solver::gpu {

using core::Real;

/// cuSPARSE-backed `pdlp::MatVec`.
///
/// Declared with a plain C++ interface and an opaque implementation pointer so
/// this header compiles under any C++ compiler -- `tests/unit/header_compile_check`
/// includes it in the default `release` preset with no CUDA toolkit present,
/// which is how the other GPU-boundary headers here are arranged too.
class CusparseMatVec final : public pdlp::MatVec {
 public:
  /// Uploads both orientations of `problem.A` to the device. Fails rather than
  /// throwing if no device is available, if allocation fails, or if the nonzero
  /// count overflows the index type cuSPARSE was configured with.
  /// `detailed_timing` attributes time between the kernel and the copies, at
  /// the cost of two `cudaDeviceSynchronize` per product. PDLP applies `K`
  /// tens of thousands of times, so that instrumentation is a large fraction
  /// of what the run then appears to cost: the breakdown and an honest wall
  /// time cannot come from the same run. Default off.
  [[nodiscard]] static core::Expected<std::unique_ptr<CusparseMatVec>> create(
      const model::CanonicalProblem& problem, bool detailed_timing = false);

  ~CusparseMatVec() override;
  CusparseMatVec(const CusparseMatVec&) = delete;
  CusparseMatVec& operator=(const CusparseMatVec&) = delete;

  void multiply(core::HostSpan<const Real> x, core::HostSpan<Real> out) override;
  void multiply_transpose(core::HostSpan<const Real> y,
                          core::HostSpan<Real> out) override;

  [[nodiscard]] std::size_t num_rows() const override;
  [[nodiscard]] std::size_t num_cols() const override;

  /// Bytes resident on the device: both matrix orientations, the vectors, and
  /// the cuSPARSE workspaces. Reported so the 4 GB limit is a measured number
  /// in the A/B rather than an assumption.
  [[nodiscard]] std::size_t device_bytes() const;

  /// Seconds spent inside `cusparseSpMV` itself, excluding the host/device
  /// copies around it. Zero unless `detailed_timing` was set.
  ///
  /// Kept SEPARATE from wall time on purpose. The kernel time is what the
  /// roofline argument above predicts; the copies are an artefact of entering
  /// through a host-span interface. Reporting one number would let a good
  /// kernel hide behind bad transfers, or the reverse -- and the difference
  /// between them is exactly what decides whether a device-resident iterate is
  /// worth building next.
  [[nodiscard]] double kernel_seconds() const;
  /// Seconds spent copying vectors across the bus, for the same reason.
  [[nodiscard]] double transfer_seconds() const;

 private:
  struct Impl;
  explicit CusparseMatVec(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_PDLP_MAT_VEC_HPP
