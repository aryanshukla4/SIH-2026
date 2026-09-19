// Module 24F: where PDLP's iterate LIVES.
//
// Stage E (gpu/PdlpMatVec.hpp) moved `K x` and `K' y` onto the GPU and measured
// the result honestly: on datt256 the kernel took 1.63 s and the host<->device
// copies around it took 2.94 s -- 1.8x the kernel. The copies were not a cost of
// the GPU. They were a cost of the INTERFACE: `MatVec` hands over host spans, so
// every product had to ship its input down and its output back.
//
// This header moves the seam up one level. Instead of abstracting the matrix,
// it abstracts the ITERATE -- the handful of vectors PDHG touches on every
// single iteration -- so an implementation can keep them wherever it likes and
// the solver only ever sees scalars.
//
// THE SPLIT, read directly off Pdlp.cpp rather than designed in the abstract:
//
//   HOT, every iteration -> this interface
//       Algorithm 2's trial loop, the fixed step of equation (3), the
//       iterate-difference and iterate-sum sequences of the infeasibility
//       detector, and the step-weighted restart average.
//
//   COLD, every `check_interval` iterations (40) -> the host, unchanged
//       Termination (6a)-(6c), the restart decision and its trust-region gap
//       evaluations, the certificate tests. These need the whole iterate on the
//       host anyway, and running them 1 time in 40 is the paper's own design
//       (section 3: "we only evaluate the restart or termination criteria every
//       40 iterations").
//
// So a device implementation ships vectors across the bus once per 40
// iterations instead of twice per matrix product, and per TRIAL it returns
// exactly three doubles -- the quantities Algorithm 2 line 6 needs to decide
// whether to accept the step.
//
// WHY AN INTERFACE AND NOT A SECOND SOLVER. The alternative -- a separate
// device PDLP -- would be two implementations of one algorithm that could drift
// apart, which this project has refused everywhere else. Here there is one
// algorithm (Pdlp.cpp) and two places for its state to live. The HOST
// implementation below is the old inner loop moved verbatim, and the test that
// this refactor changed nothing is that the whole corpus produces BIT-IDENTICAL
// objectives and product counts before and after.

#ifndef SOVSOLVE_SOLVER_PDLP_ITERATION_BACKEND_HPP
#define SOVSOLVE_SOLVER_PDLP_ITERATION_BACKEND_HPP

#include <cstddef>
#include <cstdint>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/pdlp/MatVec.hpp"

namespace sovsolve::solver::pdlp {

using core::Real;

/// What Algorithm 2 line 6 needs from one trial step, and nothing more:
///
///     eta_bar = (omega ||dx||^2 + ||dy||^2 / omega) / (2 |dy' K dx|)
///
/// Three doubles, so a trial costs 24 bytes of host traffic however large the
/// model is.
struct TrialMetrics {
  Real interaction = 0.0;  ///< `dy' K dx`
  Real dx_sq = 0.0;        ///< `||x' - x||^2`
  Real dy_sq = 0.0;        ///< `||y' - y||^2`
};

/// Vectors the cold path reads back.
enum class BackendVector : std::uint8_t {
  X,
  Y,
  AverageX,     ///< step-weighted SUM since the last restart (not yet divided)
  AverageY,
  IterateSumX,  ///< sum of every iterate since iteration 0
  IterateSumY,
  DifferenceX,  ///< `z^{k+1} - z^k` from the most recent step
  DifferenceY,
};

class IterationBackend {
 public:
  IterationBackend() = default;
  IterationBackend(const IterationBackend&) = delete;
  IterationBackend& operator=(const IterationBackend&) = delete;
  virtual ~IterationBackend() = default;

  // ---- state transfer (cold path only) ----------------------------------

  /// Replaces the iterate. Called at start-up and after a restart moves it.
  virtual void set_iterate(core::HostSpan<const Real> x, core::HostSpan<const Real> y) = 0;

  /// Copies one vector out. Only the cold path calls this.
  virtual void download(BackendVector which, core::HostSpan<Real> out) = 0;

  // ---- Algorithm 2 --------------------------------------------------------

  /// `K' y` and `K x` at the current iterate. Both are fixed across the whole
  /// trial sequence, because `x` and `y` do not move until a trial is accepted
  /// -- so this is ONCE per iteration, not once per trial. Two products.
  virtual void begin_step() = 0;

  /// Lines 4-5 and the numerator/denominator of line 6, for one trial size.
  /// One product, `K(2x' - x)`. Does not move the iterate.
  [[nodiscard]] virtual TrialMetrics trial(Real tau, Real sigma) = 0;

  /// Makes the most recent trial the iterate.
  virtual void accept_trial() = 0;

  // ---- equation (3), the fixed-step ablation -----------------------------

  /// One plain PDHG step. Two products.
  virtual void fixed_step(Real tau, Real sigma) = 0;

  // ---- infeasibility sequences (arXiv 2102.04592 section 1.1) ------------

  /// `diff <- z`, taken BEFORE a step.
  virtual void snapshot_iterate() = 0;
  /// `diff <- z - diff` and `sum <- sum + z`, taken AFTER it.
  virtual void finish_difference() = 0;

  // ---- restart average (Algorithm 1 line 7) ------------------------------

  /// `avg <- avg + weight * z`. The weight sum itself stays on the host; it is
  /// a scalar.
  virtual void accumulate_average(Real weight) = 0;
  virtual void reset_average() = 0;

  /// Matrix products this backend applied itself, over and above whatever the
  /// cold-path `MatVec` counted. Zero for an implementation that routes its
  /// products THROUGH that `MatVec`, so nothing is counted twice.
  [[nodiscard]] virtual std::size_t own_products() const = 0;
};

/// The pre-refactor inner loop, moved verbatim. Products go through the
/// injected `MatVec`, which is also the cold path's -- so the host path and the
/// Stage E cuSPARSE path both keep working exactly as they did.
class HostIterationBackend final : public IterationBackend {
 public:
  HostIterationBackend(const model::CanonicalProblem& problem, MatVec& matvec);

  void set_iterate(core::HostSpan<const Real> x, core::HostSpan<const Real> y) override;
  void download(BackendVector which, core::HostSpan<Real> out) override;
  void begin_step() override;
  [[nodiscard]] TrialMetrics trial(Real tau, Real sigma) override;
  void accept_trial() override;
  void fixed_step(Real tau, Real sigma) override;
  void snapshot_iterate() override;
  void finish_difference() override;
  void accumulate_average(Real weight) override;
  void reset_average() override;
  [[nodiscard]] std::size_t own_products() const override { return 0; }

 private:
  const model::CanonicalProblem& problem_;
  MatVec& matvec_;
  std::size_t n_;
  std::size_t m_;

  core::RealVector x_, y_;
  core::RealVector x_trial_, y_trial_;
  core::RealVector extrapolated_;    ///< `2x' - x`
  core::RealVector kt_y_;            ///< `K' y`, fixed across a trial sequence
  core::RealVector k_x_current_;     ///< `K x`, likewise
  core::RealVector k_extrapolated_;  ///< `K(2x' - x)`, per trial
  core::RealVector x_prev_;          ///< fixed-step scratch
  core::RealVector k_x_;             ///< fixed-step scratch
  core::RealVector diff_x_, diff_y_;
  core::RealVector sum_x_, sum_y_;
  core::RealVector avg_x_, avg_y_;
};

}  // namespace sovsolve::solver::pdlp

#endif  // SOVSOLVE_SOLVER_PDLP_ITERATION_BACKEND_HPP
