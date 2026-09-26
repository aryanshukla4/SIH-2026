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
#include <optional>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/solver/pdlp/HalpernControl.hpp"
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

/// How far the iterate travelled from the Halpern anchor, per block.
///
/// The two numbers the PID primal-weight controller needs and nothing else:
/// `||x_{n,t} - x_{n,0}||_2` and `||y_{n,t} - y_{n,0}||_2` (cuPDLPx section 3).
/// Returned as scalars for the same reason `TrialMetrics` is -- a restart on
/// the device path must not have to ship two vectors home to update one
/// weight.
struct AnchorDistance {
  Real dx = 0.0;
  Real dy = 0.0;
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
  /// `PDHG(z^k)` from the most recent `halpern_step`, i.e. `T(z)` before the
  /// reflection and the anchor blend. Halpern mode reads the SOLUTION here
  /// rather than from `X`/`Y`, and that is a correctness requirement, not a
  /// preference: with `gamma > 0` the blended iterate is an EXTRAPOLATION and
  /// may sit outside the box `X`, whereas `T(z)` is a projection and never
  /// does. Reporting `x` would hand back a point violating its own bounds.
  PdhgX,
  PdhgY,
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

  // ---- Module 31: reflected Halpern (cuPDLPx, arXiv 2507.14051) ----------
  //
  // WHY THESE ARE NOT `fixed_step` PLUS HOST ARITHMETIC. The Halpern blend
  //
  //     z^{k+1} = lambda [ (1+gamma) T(z^k) - gamma z^k ] + (1-lambda) z^{n,0}
  //
  // is linear in `z`, so `K` applied to it is the same blend of `K T(z^k)`,
  // `K z^k` and `K z^{n,0}`. Carrying `K x` and `K' y` through the recurrence
  // instead of recomputing them keeps the iteration at TWO products -- the
  // same as vanilla PDHG -- where the obvious implementation needs three.
  // That only works if the blend and the products live in the same place,
  // which is here.
  //
  // The recurrence is exact linear algebra, not an approximation, but it does
  // accumulate rounding the way any recurrence does. `begin_halpern()`
  // recomputes both from scratch, and every restart goes through a point
  // whose products were just computed directly, so the drift is bounded by
  // one epoch rather than by the whole run.

  /// `||K||_2` by power iteration, computed WHERE THE MATRIX LIVES -- or
  /// `nullopt`, meaning "use the host path through the cold `MatVec`".
  ///
  /// Measured before this existed, `--method=pdlpx --gpu-resident=1` on
  /// datt256 (262k columns) spent 1.04 s here against 1.06 s for the entire
  /// iteration loop: ~200 products through the cold path, each shipping a
  /// full vector down and another back across PCIe. The iterations were
  /// already 5x faster than the host's; the step-size estimate in front of
  /// them was eating half of it. A device implementation keeps the vector
  /// resident and reads back one scalar per power step.
  ///
  /// Same seed, same update, same stopping rule as the host routine, so the
  /// two agree to rounding. The default suits the host backend, whose
  /// `MatVec` has no transfer to avoid.
  [[nodiscard]] virtual std::optional<Real> spectral_norm(std::size_t /*iterations*/,
                                                          Real /*tolerance*/) {
    return std::nullopt;
  }

  /// Whether this backend implements the three operations below.
  ///
  /// A capability flag rather than three more pure virtuals, so that adding
  /// Module 31 does not break every existing backend at once. `solve_pdlp`
  /// refuses a Halpern run on a backend that answers false, with a named
  /// error -- which is the failure mode to want, because the alternative is
  /// three no-op overrides somewhere that make the iterate silently stop
  /// moving and the run report `MaxIterations` on a solvable model.
  [[nodiscard]] virtual bool supports_halpern() const { return false; }

  /// Recompute `K x` and `K' y` at the current iterate and make it the anchor
  /// `z^{n,0}`. Two products. Called once at start-up; a restart uses
  /// `restart_at_pdhg_point()` instead, which needs none.
  virtual void begin_halpern() {}

  /// One reflected-Halpern iteration, with `lambda = (k+1)/(k+2)`.
  ///
  /// Two products, and it leaves `T(z^k)` readable as `BackendVector::Pdhg*`.
  ///
  /// Returns the three pieces of the fixed-point error AT `z^k`, measured
  /// before the move -- `dx = x^k - T(z^k)_x`, `dy = y^k - T(z^k)_y`, and
  /// `dy' K dx` -- from which the caller forms
  ///
  ///     r(z)^2 = (omega/eta) ||dx||^2 + 2 dy' K dx + (1/(omega eta)) ||dy||^2
  ///
  /// because `P_{eta,omega} = [[omega/eta I, A'], [A, 1/(eta omega) I]]`. This
  /// is the same three scalars `trial()` returns and for the same reason: the
  /// restart criterion is evaluated EVERY iteration (cuPDLPx section 3), so it
  /// must not cost a transfer.
  [[nodiscard]] virtual TrialMetrics halpern_step(Real /*eta*/, Real /*omega*/,
                                                  Real /*gamma*/, Real /*lambda*/) {
    return {};
  }

  /// Restart: `z <- T(z^k)` from the most recent `halpern_step`, then the
  /// anchor follows it. This is arXiv 2407.16144 Algorithm 2 line 6 --
  /// the new epoch starts at a PDHG step of the last inner iterate, not at
  /// the inner iterate itself.
  ///
  /// ZERO products: `halpern_step` already computed `T(z)` and, between them,
  /// its two matrix images. Returns the anchor distances the PID controller
  /// needs, measured against the anchor being replaced.
  [[nodiscard]] virtual AnchorDistance restart_at_pdhg_point() { return {}; }

  // ---- the controller, where the iterate lives ---------------------------
  //
  // `count` whole Halpern iterations -- step, restart decision, primal-weight
  // update, restart move -- with NO host involvement between them. The
  // decisions are HalpernControl.hpp's, executed wherever this backend keeps
  // its state; the solver reads that state only at a termination check.
  //
  // This is what removes the per-iteration synchronization on the device:
  // the host enqueues `count` iterations of kernels and waits once. The host
  // backend implements it as the obvious loop over `halpern_step` and
  // `restart_at_pdhg_point`, so both backends run one algorithm.
  //
  // `first_iteration` is the global count before the chunk, which condition
  // (iii) needs. `track_differences` wraps each step in the infeasibility
  // detector's `snapshot_iterate` / `finish_difference`, in the same place the
  // per-iteration loop used to.

  virtual void write_halpern_state(const HalpernState& /*state*/) {}
  [[nodiscard]] virtual HalpernState read_halpern_state() { return {}; }
  virtual void run_halpern(std::size_t /*count*/, std::uint64_t /*first_iteration*/,
                           const HalpernParams& /*params*/, bool /*track_differences*/) {}

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

  /// A STICKY error: the first failure the backend hit, or OK.
  ///
  /// The hot-path methods return nothing, because checking a status after
  /// every kernel launch would put a host decision back in the middle of the
  /// loop this interface exists to keep clear. So a device implementation
  /// records its first failure here and carries on, and the solver checks this
  /// at every sync point. Without it a failed launch mid-solve would not stop
  /// anything: the iterate would simply stop changing, and the run would report
  /// MaxIterations on a solvable problem. The host implementation cannot fail
  /// and keeps the default.
  [[nodiscard]] virtual core::Status status() const { return core::Status::Ok(); }

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
  [[nodiscard]] bool supports_halpern() const override { return true; }
  void begin_halpern() override;
  [[nodiscard]] TrialMetrics halpern_step(Real eta, Real omega, Real gamma,
                                          Real lambda) override;
  [[nodiscard]] AnchorDistance restart_at_pdhg_point() override;
  void write_halpern_state(const HalpernState& state) override { control_ = state; }
  [[nodiscard]] HalpernState read_halpern_state() override { return control_; }
  void run_halpern(std::size_t count, std::uint64_t first_iteration,
                   const HalpernParams& params, bool track_differences) override;
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

  // Module 31. `x_trial_`/`y_trial_` above double as `T(z^k)`; these are the
  // pieces the two-product recurrence needs on top of them.
  core::RealVector kt_y_trial_;   ///< `K' T(z)_y`, the second product
  core::RealVector k_x_trial_;    ///< `K T(z)_x`, from `(K ext + K x) / 2`
  core::RealVector x_anchor_, y_anchor_;        ///< `z^{n,0}`
  core::RealVector kt_y_anchor_, k_x_anchor_;   ///< its matrix images
  HalpernState control_;                         ///< HalpernControl.hpp
};

}  // namespace sovsolve::solver::pdlp

#endif  // SOVSOLVE_SOLVER_PDLP_ITERATION_BACKEND_HPP
