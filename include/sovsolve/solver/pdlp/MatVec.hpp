// Module 24: the only expensive operation PDLP has.
//
// PDHG touches the constraint matrix exactly twice per iteration -- once as
// `K x` and once as `K' y` -- and never factorizes it. That is the entire
// reason this engine exists alongside the other two: on this project's
// hardware FP64 factorization carries a ~64x penalty and is slower than the
// CPU, while sparse matrix-vector products are bandwidth-bound at ~2x
// (docs/ARCHITECTURE-REVIEW.md section 3.5). An algorithm whose inner loop is
// nothing but SpMV is the one shape that can actually use this GPU.
//
// So the product goes behind an interface from the start. The CPU
// implementation lives here; a cuSPARSE implementation belongs in
// sovsolve_solver_gpu and is INJECTED by the caller, because the
// gpu -> solver library edge is one-way and must stay that way
// (src/solver/CMakeLists.txt).
//
// `K` is this project's canonical `A` with no reordering: PDLP's published
// form puts inequality rows first, ours puts equality rows first, and rather
// than permute the matrix the difference is carried in the dual's feasible
// set (see Pdlp.hpp). Permuting would mean a second copy of the largest
// object in the problem, to save a branch.

#ifndef SOVSOLVE_SOLVER_PDLP_MAT_VEC_HPP
#define SOVSOLVE_SOLVER_PDLP_MAT_VEC_HPP

#include <cstddef>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"

namespace sovsolve::solver::pdlp {

using core::Real;

/// Applies `K` and `K'`. Implementations must not retain the spans they are
/// handed; they are caller-owned working vectors reused every iteration.
class MatVec {
 public:
  MatVec() = default;
  MatVec(const MatVec&) = delete;
  MatVec& operator=(const MatVec&) = delete;
  virtual ~MatVec() = default;

  /// `out = K x`. `x` has `num_cols()` entries, `out` has `num_rows()`.
  virtual void multiply(core::HostSpan<const Real> x, core::HostSpan<Real> out) = 0;

  /// `out = K' y`. `y` has `num_rows()` entries, `out` has `num_cols()`.
  virtual void multiply_transpose(core::HostSpan<const Real> y,
                                  core::HostSpan<Real> out) = 0;

  [[nodiscard]] virtual std::size_t num_rows() const = 0;
  [[nodiscard]] virtual std::size_t num_cols() const = 0;

  /// Total products applied, `K` and `K'` counted separately.
  ///
  /// PDLP reports work in "KKT passes" rather than seconds because the metric
  /// is noise-free and comparable across machines -- and because for a
  /// matrix-free method it genuinely IS the cost. One pass is one `K` plus
  /// one `K'`, so `products() / 2`. Counting here rather than in the solver
  /// means a CPU-vs-GPU A/B compares the same number, not two different
  /// instrumentations.
  [[nodiscard]] std::size_t products() const noexcept { return products_; }
  void reset_products() noexcept { products_ = 0; }

 protected:
  std::size_t products_ = 0;
};

/// Host implementation over the canonical problem's existing CSR/CSC pair.
///
/// Both orientations are already built and kept consistent by the
/// canonicalizer and the scaler, so `K x` reads the CSR and `K' y` reads the
/// CSC -- each a contiguous sweep with no indirection through the other
/// orientation, and no transpose ever materialized.
class HostMatVec final : public MatVec {
 public:
  explicit HostMatVec(const model::CanonicalProblem& problem) : problem_(&problem) {}

  void multiply(core::HostSpan<const Real> x, core::HostSpan<Real> out) override;
  void multiply_transpose(core::HostSpan<const Real> y,
                          core::HostSpan<Real> out) override;

  [[nodiscard]] std::size_t num_rows() const override { return problem_->num_rows(); }
  [[nodiscard]] std::size_t num_cols() const override { return problem_->num_cols(); }

 private:
  const model::CanonicalProblem* problem_;
};

}  // namespace sovsolve::solver::pdlp

#endif  // SOVSOLVE_SOLVER_PDLP_MAT_VEC_HPP
