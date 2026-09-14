#include "sovsolve/solver/pdlp/Pdlp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "sovsolve/solver/pdlp/DualityGap.hpp"
#include "sovsolve/solver/pdlp/MatVec.hpp"

namespace sovsolve::solver::pdlp {

namespace {

using core::ErrorCode;
using core::SolverStatus;
using model::CanonicalProblem;
using model::Options;

constexpr std::size_t kDefaultMaxIterations = 100000;

[[nodiscard]] Real dot(const core::RealVector& a, const core::RealVector& b) {
  Real acc = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) acc += a[i] * b[i];
  return acc;
}

[[nodiscard]] Real euclidean_norm(const core::RealVector& v) {
  Real acc = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) acc += v[i] * v[i];
  return std::sqrt(acc);
}

core::HostSpan<const Real> in(const core::RealVector& v) {
  return core::HostSpan<const Real>(v.data(), v.size());
}

core::HostSpan<Real> out(core::RealVector& v) {
  return core::HostSpan<Real>(v.data(), v.size());
}

/// `proj_Lambda` for one column: which signs its reduced cost may take is
/// decided entirely by which of its bounds are finite. See Pdlp.hpp.
[[nodiscard]] Real project_reduced_cost(Real value, Real lower, Real upper) {
  const bool has_lower = core::is_finite_bound(lower);
  const bool has_upper = core::is_finite_bound(upper);
  if (!has_lower && !has_upper) return 0.0;       // free column: must vanish
  if (!has_lower) return std::min(value, 0.0);    // only an upper bound: R-
  if (!has_upper) return std::max(value, 0.0);    // only a lower bound: R+
  return value;                                   // boxed: any sign is fine
}

/// `||K||_2` by power iteration on `K'K`, whose largest eigenvalue is the
/// squared largest singular value.
///
/// Only the BASELINE needs this. PDLP's Algorithm 1 starts from
/// `1/||K||_inf`, a single sweep with no iteration at all, and then adapts --
/// so this whole routine disappears when the adaptive step size lands.
[[nodiscard]] Real estimate_spectral_norm(MatVec& matvec, const Options& options) {
  const std::size_t m = matvec.num_rows();
  const std::size_t n = matvec.num_cols();
  core::RealVector v(n, 0.0);
  core::RealVector kv(m, 0.0);

  // A deterministic non-uniform start: a constant vector is exactly the wrong
  // seed when the dominant singular vector is orthogonal to it, which happens
  // on structured matrices.
  for (std::size_t j = 0; j < n; ++j) {
    v[j] = 1.0 + static_cast<Real>(j % 7) * 0.1;
  }
  Real norm = euclidean_norm(v);
  if (norm == 0.0) return 0.0;
  for (std::size_t j = 0; j < n; ++j) v[j] /= norm;

  Real sigma = 0.0;
  for (std::size_t it = 0; it < options.pdlp.power_iterations; ++it) {
    matvec.multiply(in(v), out(kv));
    matvec.multiply_transpose(in(kv), out(v));
    const Real next = euclidean_norm(v);
    if (next <= 0.0) return 0.0;
    for (std::size_t j = 0; j < n; ++j) v[j] /= next;
    // `v` now holds a unit vector and `next` is the Rayleigh quotient of
    // `K'K`, so the singular value is its square root.
    const Real candidate = std::sqrt(next);
    if (sigma > 0.0 && std::fabs(candidate - sigma) <= options.pdlp.power_tolerance * sigma) {
      sigma = candidate;
      break;
    }
    sigma = candidate;
  }
  return sigma;
}

/// `||K||_inf`, the induced infinity norm: the largest absolute row sum.
///
/// PDLP's Algorithm 1 initializes its step size from `1/||K||_inf` rather than
/// `1/||K||_2`, and the difference is not a detail. The spectral norm needs
/// power iteration -- tens of matrix products before the first real step --
/// while this is one sweep of the values already in memory. The adaptive rule
/// then corrects whatever the initial guess was, so paying for an accurate
/// starting point buys nothing.
[[nodiscard]] Real infinity_norm(const CanonicalProblem& problem) {
  const auto& csr = problem.A.csr;
  Real worst = 0.0;
  for (std::size_t i = 0; i < problem.num_rows(); ++i) {
    Real row_sum = 0.0;
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      row_sum += std::fabs(csr.values()[k]);
    }
    worst = std::max(worst, row_sum);
  }
  return worst;
}

/// Below this, a norm is treated as structurally zero rather than small.
/// Algorithm 3 and InitializePrimalWeight both guard divisions with it.
constexpr Real kZeroTolerance = 1e-12;

/// `InitializePrimalWeight(c, q) = ||c||_2 / ||q||_2` (paper section 3.3).
///
/// This is a statement about UNITS. `omega` splits the step between the primal
/// and dual (`tau = eta/omega`, `sigma = eta*omega`), and the natural scale of
/// the primal side is set by `c` while the dual's is set by `q` -- so starting
/// from their ratio makes the two halves comparable before a single iteration
/// has run. The paper proves this makes the whole method scale-invariant:
/// multiply the objective, the constraints or the right-hand side by a scalar
/// and the iterates are identical up to that scaling.
[[nodiscard]] Real initialize_primal_weight(const CanonicalProblem& problem) {
  const Real c_norm = euclidean_norm(problem.c);
  const Real q_norm = euclidean_norm(problem.b);
  if (c_norm > kZeroTolerance && q_norm > kZeroTolerance) return c_norm / q_norm;
  return 1.0;
}

/// Everything equations (6a)-(6c) need, measured at the current iterate.
struct Convergence {
  Real gap = 0.0;
  Real primal = 0.0;
  Real dual = 0.0;
  Real primal_objective = 0.0;
  [[nodiscard]] bool converged(Real tol) const {
    return gap <= tol && primal <= tol && dual <= tol;
  }
};

class PdlpSolver {
 public:
  PdlpSolver(const CanonicalProblem& problem, const Options& options)
      : problem_(problem),
        opt_(options),
        matvec_(problem),
        m_(problem.num_rows()),
        n_(problem.num_cols()),
        gap_(problem) {}

  core::Expected<PdlpResult> run();

 private:
  void evaluate(Convergence& conv);
  [[nodiscard]] PdlpResult pack(SolverStatus status, const Convergence& conv) const;

  const CanonicalProblem& problem_;
  const Options& opt_;
  HostMatVec matvec_;
  std::size_t m_;
  std::size_t n_;

  core::RealVector x_;
  core::RealVector y_;
  core::RealVector x_prev_;
  core::RealVector extrapolated_;  ///< `2 x^{k+1} - x^k`
  core::RealVector kt_y_;          ///< `K' y`
  core::RealVector k_x_;           ///< `K x`
  core::RealVector reduced_cost_;
  core::RealVector primal_residual_;

  // Adaptive step size (Algorithm 2) scratch. `x_trial_`/`y_trial_` hold the
  // candidate point of a trial that may yet be rejected, so `x_`/`y_` are only
  // overwritten once a step is accepted.
  core::RealVector x_trial_;
  core::RealVector y_trial_;
  core::RealVector k_extrapolated_;  ///< `K(2x' - x)`, recomputed per trial
  core::RealVector k_x_current_;     ///< `K x`, fixed across a trial sequence

  Real omega_ = 1.0;  ///< primal weight; adapts in C3, fixed at 1 until then
  std::size_t step_rejections_ = 0;

  // ---- adaptive restarts (paper section 3.2) ----------------------------
  NormalizedDualityGap gap_;
  core::RealVector x_restart_;  ///< `z^{n,0}`, the current outer loop's anchor
  core::RealVector y_restart_;
  /// Step-size-weighted running sums for `z-bar` (Algorithm 1 line 7). Held
  /// as sums rather than as the average so a restart is two `assign(0)` calls
  /// and no division.
  core::RealVector avg_x_;
  core::RealVector avg_y_;
  Real weight_sum_ = 0.0;
  /// Scratch for the candidate `z-bar`, materialized only when a restart
  /// check actually runs.
  core::RealVector bar_x_;
  core::RealVector bar_y_;
  core::RealVector scratch_kt_y_;
  core::RealVector scratch_k_x_;

  /// `mu_n(z^{n,0}, z^{n-1,0})`: the gap at the current anchor, measured
  /// against the PREVIOUS anchor. Conditions (i) and (ii) compare against it.
  /// Infinite for the first outer loop, where there is no previous anchor --
  /// so only condition (iii) can fire, which is correct rather than a
  /// special case: with no history there is nothing to have decayed from.
  /// The PREVIOUS outer loop's anchor, `z^{n-1,0}`. Algorithm 3 measures how
  /// far the anchor moved between restarts, so both are needed.
  core::RealVector x_prev_restart_;
  core::RealVector y_prev_restart_;
  bool have_prev_restart_ = false;

  Real reference_gap_ = std::numeric_limits<Real>::infinity();
  /// `mu_n(z_c^{n,t}, z^{n,0})` from the previous check, for condition (ii)'s
  /// "no local progress" half.
  Real last_candidate_gap_ = std::numeric_limits<Real>::infinity();
  std::size_t inner_iterations_ = 0;  ///< `t`
  std::size_t restarts_ = 0;          ///< `n`

  /// Algorithm 2. `eta` enters as this iteration's trial size and leaves as
  /// `eta'`, the starting point for the next one.
  void adaptive_step(std::size_t total_iterations, Real& eta);
  void fixed_step(Real tau, Real sigma);

  void accumulate_average(Real eta);
  /// Section 3.2. Returns true when the outer loop restarted.
  [[nodiscard]] bool maybe_restart(std::size_t total_iterations);
};

void PdlpSolver::evaluate(Convergence& conv) {
  // `K'y` gives the reduced costs; `Kx` gives the primal residual.
  matvec_.multiply_transpose(in(y_), out(kt_y_));
  matvec_.multiply(in(x_), out(k_x_));

  // (6c): the part of `c - K'y` that no bound can absorb.
  Real dual_residual_sq = 0.0;
  Real bound_term = 0.0;
  for (std::size_t j = 0; j < n_; ++j) {
    const Real raw = problem_.c[j] - kt_y_[j];
    const Real lambda =
        project_reduced_cost(raw, problem_.col_lower[j], problem_.col_upper[j]);
    reduced_cost_[j] = lambda;
    const Real leftover = raw - lambda;
    dual_residual_sq += leftover * leftover;

    // `l'lambda^+ - u'lambda^-`, skipping infinite bounds. The projection
    // above guarantees an infinite bound pairs with a zero `lambda`, so the
    // skipped terms are `inf * 0` -- NaN if it were evaluated.
    if (lambda > 0.0 && core::is_finite_bound(problem_.col_lower[j])) {
      bound_term += problem_.col_lower[j] * lambda;
    } else if (lambda < 0.0 && core::is_finite_bound(problem_.col_upper[j])) {
      bound_term += problem_.col_upper[j] * lambda;
    }
  }

  // (6b): equality rows are two-sided, `<=` rows can only be violated upward.
  Real primal_residual_sq = 0.0;
  for (std::size_t i = 0; i < m_; ++i) {
    const Real slack = k_x_[i] - problem_.b[i];
    const Real violation = i < problem_.num_equality ? slack : std::max(slack, 0.0);
    primal_residual_[i] = violation;
    primal_residual_sq += violation * violation;
  }

  const Real primal_objective = dot(problem_.c, x_);
  const Real dual_objective = dot(problem_.b, y_) + bound_term;

  conv.primal_objective = primal_objective;
  conv.gap = std::fabs(dual_objective - primal_objective) /
             (1.0 + std::fabs(dual_objective) + std::fabs(primal_objective));
  conv.primal = std::sqrt(primal_residual_sq) / (1.0 + euclidean_norm(problem_.b));
  conv.dual = std::sqrt(dual_residual_sq) / (1.0 + euclidean_norm(problem_.c));
}

PdlpResult PdlpSolver::pack(SolverStatus status, const Convergence& conv) const {
  PdlpResult r;
  r.status = status;
  // `core::Vector` deletes copy assignment on purpose (Vector.hpp): an
  // accidental deep copy of a million doubles inside an iteration is
  // invisible in the source and ruinous in the profile. This one is
  // deliberate and happens once, at exit.
  r.x = x_.clone();
  r.y = y_.clone();
  r.reduced_cost = reduced_cost_.clone();
  r.objective = conv.primal_objective;
  r.matrix_products = matvec_.products();
  r.step_rejections = step_rejections_;
  r.restarts = restarts_;
  r.relative_duality_gap = conv.gap;
  r.relative_primal_residual = conv.primal;
  r.relative_dual_residual = conv.dual;
  return r;
}

/// One fixed-step PDHG iteration -- paper equation (3) verbatim. Kept so the
/// adaptive rule can be switched off and compared against, which is how the
/// paper's own ablation (figure 1) is structured.
void PdlpSolver::fixed_step(Real tau, Real sigma) {
  matvec_.multiply_transpose(in(y_), out(kt_y_));
  for (std::size_t j = 0; j < n_; ++j) {
    x_prev_[j] = x_[j];
    const Real step = x_[j] - tau * (problem_.c[j] - kt_y_[j]);
    x_[j] = std::clamp(step, problem_.col_lower[j], problem_.col_upper[j]);
  }
  // The extrapolation `2x^{k+1} - x^k` is what makes this PDHG rather than
  // Arrow-Hurwicz, and it is what the convergence proof needs; using `x^k`
  // here converges only under far stronger conditions.
  for (std::size_t j = 0; j < n_; ++j) {
    extrapolated_[j] = 2.0 * x_[j] - x_prev_[j];
  }
  matvec_.multiply(in(extrapolated_), out(k_x_));
  for (std::size_t i = 0; i < m_; ++i) {
    const Real step = y_[i] + sigma * (problem_.b[i] - k_x_[i]);
    y_[i] = i < problem_.num_equality ? step : std::min(step, 0.0);
  }
}

/// Algorithm 2: one PDHG step whose size is chosen by trial.
///
/// The convergence analysis of PDHG needs the step to satisfy equation (5),
///
///     eta <= ||z^{k+1} - z^k||^2_omega / ( 2 (y^{k+1} - y^k)' K (x^{k+1} - x^k) )
///
/// Classically one guarantees that by taking `eta = 1/||K||_2`, which is both
/// pessimistic -- the bound is global, while the ratio above is local and
/// usually much larger -- and expensive, since `||K||_2` needs estimating.
/// Instead: take the step, measure the ratio it actually produced, and accept
/// only if (5) held. If it did not, shrink and retry the same iteration.
///
/// The two exponents in line 7 are what stop this oscillating. The shrink
/// factor `(1 - (k+1)^-0.3)` and the growth cap `(1 + (k+1)^-0.6)` both decay
/// with the iteration count, so early steps move aggressively and later ones
/// settle -- and because the growth cap decays faster than the shrink factor,
/// the step size cannot keep re-inflating into the same rejection.
///
/// Matrix-product accounting, which is the whole cost model here: `y` does NOT
/// change across trials, so `K'y` is computed ONCE per iteration rather than
/// per trial. `K x` likewise. Only `K(2x' - x)` is per-trial. So an iteration
/// costs 2 + (number of trials) products, against the fixed rule's 2 --
/// measured at ~1.53 KKT passes per iteration against ~1.03 on this corpus,
/// which is why the rule must be judged in passes and not in iterations.
///
/// One deviation from the paper's literal statement of (5), found by
/// measurement rather than by reading it; the argument and the evidence are
/// at the `denominator` line below.
void PdlpSolver::adaptive_step(std::size_t total_iterations, Real& eta_inout) {
  Real eta = eta_inout;
  // Fixed across the whole trial sequence: `x_` and `y_` do not move until a
  // trial is accepted.
  matvec_.multiply_transpose(in(y_), out(kt_y_));
  matvec_.multiply(in(x_), out(k_x_current_));

  const Real k_plus_one = static_cast<Real>(total_iterations + 1);
  const Real shrink = 1.0 - std::pow(k_plus_one, -0.3);
  const Real growth = 1.0 + std::pow(k_plus_one, -0.6);

  // A cap, not part of the algorithm: Algorithm 2's loop is stated as
  // `for i = 1..infinity`, and it terminates because `eta` shrinks
  // geometrically until (5) must hold. On degenerate data (a zero row, a
  // denominator at the rounding floor) that argument can fail numerically, and
  // an unbounded loop inside a solver is worse than a slightly-too-large step.
  constexpr int kMaxTrials = 60;

  for (int trial = 0; trial < kMaxTrials; ++trial) {
    const Real tau = eta / omega_;
    const Real sigma = eta * omega_;

    // line 4: x' = proj_X(x - (eta/omega)(c - K'y))
    for (std::size_t j = 0; j < n_; ++j) {
      const Real step = x_[j] - tau * (problem_.c[j] - kt_y_[j]);
      x_trial_[j] = std::clamp(step, problem_.col_lower[j], problem_.col_upper[j]);
      extrapolated_[j] = 2.0 * x_trial_[j] - x_[j];
    }

    // line 5: y' = proj_Y(y + eta*omega (q - K(2x' - x)))
    matvec_.multiply(in(extrapolated_), out(k_extrapolated_));
    for (std::size_t i = 0; i < m_; ++i) {
      const Real step = y_[i] + sigma * (problem_.b[i] - k_extrapolated_[i]);
      y_trial_[i] = i < problem_.num_equality ? step : std::min(step, 0.0);
    }

    // line 6: eta_bar = ||dz||^2_omega / (2 dy' K dx).
    //
    // `K dx` needs no product of its own. With `u = 2x' - x` we have
    // `u - x = 2(x' - x)`, so `K dx = (K u - K x) / 2` -- and both terms are
    // already in hand.
    Real interaction = 0.0;
    for (std::size_t i = 0; i < m_; ++i) {
      const Real k_dx = 0.5 * (k_extrapolated_[i] - k_x_current_[i]);
      interaction += (y_trial_[i] - y_[i]) * k_dx;
    }

    Real dx_dy_norm_sq = 0.0;
    {
      Real px = 0.0;
      for (std::size_t j = 0; j < n_; ++j) {
        const Real d = x_trial_[j] - x_[j];
        px += d * d;
      }
      Real py = 0.0;
      for (std::size_t i = 0; i < m_; ++i) {
        const Real d = y_trial_[i] - y_[i];
        py += d * d;
      }
      dx_dy_norm_sq = omega_ * px + py / omega_;
    }

    // DEVIATION from the paper, found by measurement -- see the header block
    // on this function. Equation (5) and Algorithm 2 line 6 write this
    // denominator WITHOUT an absolute value, which makes `eta_bar` infinite
    // whenever the cross term is negative, and the step unconditionally
    // acceptable. That is true of the single iteration -- the descent
    // inequality then holds with room to spare -- and unsafe across
    // iterations, because nothing else bounds `eta` and the growth cap
    // ratchets it up every time the sign happens to come out negative.
    //
    // Taking the magnitude instead is the reading that matches what makes
    // `eta = 1/||K||_2` admissible in the first place:
    //
    //     2 |dy' K dx|  <=  ||K|| ( omega||dx||^2 + ||dy||^2/omega )
    //                    =  ||K|| ||dz||^2_omega
    //
    // so `||dz||^2_omega / (2|dy' K dx|) >= 1/||K||`, which is exactly the
    // paper's own stated property that `eta_bar >= 1/||K||_2` always holds.
    // With the signed denominator that property is vacuous rather than
    // informative.
    const Real denominator = 2.0 * std::fabs(interaction);
    const Real eta_bar = denominator > 0.0
                             ? dx_dy_norm_sq / denominator
                             : std::numeric_limits<Real>::infinity();

    const Real eta_next = std::min(shrink * eta_bar, growth * eta);

    if (eta <= eta_bar) {
      // Accepted. `x_prev_` keeps the pre-step primal point because the
      // restart machinery (C2) needs the iterate difference.
      for (std::size_t j = 0; j < n_; ++j) {
        x_prev_[j] = x_[j];
        x_[j] = x_trial_[j];
      }
      for (std::size_t i = 0; i < m_; ++i) y_[i] = y_trial_[i];
      eta_inout = std::isfinite(eta_next) && eta_next > 0.0 ? eta_next : eta;
      return;
    }

    ++step_rejections_;
    if (!std::isfinite(eta_next) || eta_next <= 0.0 || eta_next >= eta) {
      // Not shrinking, so retrying cannot help. Take the step anyway rather
      // than spin: a slightly oversized step degrades convergence, a hang
      // does not degrade, it stops.
      for (std::size_t j = 0; j < n_; ++j) {
        x_prev_[j] = x_[j];
        x_[j] = x_trial_[j];
      }
      for (std::size_t i = 0; i < m_; ++i) y_[i] = y_trial_[i];
      eta_inout = eta;
      return;
    }
    eta = eta_next;
  }

  eta_inout = eta;
}

/// Algorithm 1 line 7: the step-size-weighted average of the inner loop's
/// iterates. Kept as running sums so a restart costs two `assign(0)` calls.
void PdlpSolver::accumulate_average(Real eta) {
  weight_sum_ += eta;
  for (std::size_t j = 0; j < n_; ++j) avg_x_[j] += eta * x_[j];
  for (std::size_t i = 0; i < m_; ++i) avg_y_[i] += eta * y_[i];
}

/// Paper section 3.2: choose a restart candidate, test three conditions, and
/// restart the outer loop from the candidate if any holds.
///
/// Why restarts matter more than the other enhancements. PDHG's convergence
/// guarantee is on the ERGODIC iterate -- the running average -- which
/// converges at a good rate but keeps the early, bad iterates in the average
/// forever. The last iterate has no such guarantee but is usually far better
/// late in a run. Restarting takes whichever is currently better, makes it the
/// new starting point, and discards the history: the average stops being
/// polluted, and the guarantee is re-established from a strictly better point.
/// That is why the paper's ablation ranks this first, and why it is the
/// enhancement that turns a method which tails off into one that does not.
///
/// The candidate is `z^{n,t+1}` or `z-bar^{n,t+1}`, whichever has the smaller
/// normalized duality gap measured from the anchor (`GetRestartCandidate`).
/// Evaluating that costs two matrix products per point, which is why this runs
/// on the same schedule as the termination check rather than every iteration.
bool PdlpSolver::maybe_restart(std::size_t total_iterations) {
  // --- condition (iii), long inner loop: `t >= beta_artificial * k`.
  //
  // Checked first because it needs no gap evaluation at all. It is not a
  // fallback: primal weights are updated only at a restart (Algorithm 1 line
  // 12), so without this an unlucky early weight could never be corrected.
  const bool long_inner_loop =
      static_cast<Real>(inner_iterations_) >=
      opt_.pdlp.restart_artificial * static_cast<Real>(total_iterations);

  // `z-bar`, the step-size-weighted average since the anchor.
  const bool have_average = weight_sum_ > 0.0;
  if (have_average) {
    for (std::size_t j = 0; j < n_; ++j) bar_x_[j] = avg_x_[j] / weight_sum_;
    for (std::size_t i = 0; i < m_; ++i) bar_y_[i] = avg_y_[i] / weight_sum_;
  }

  // GetRestartCandidate: the current iterate, or the average, whichever has
  // the smaller gap from the anchor.
  matvec_.multiply_transpose(in(y_), out(scratch_kt_y_));
  matvec_.multiply(in(x_), out(scratch_k_x_));
  auto current = gap_.evaluate(x_, y_, scratch_kt_y_, scratch_k_x_, x_restart_,
                               y_restart_, omega_);
  if (!current.has_value()) return false;

  Real candidate_gap = *current;
  bool use_average = false;
  if (have_average) {
    matvec_.multiply_transpose(in(bar_y_), out(scratch_kt_y_));
    matvec_.multiply(in(bar_x_), out(scratch_k_x_));
    auto averaged = gap_.evaluate(bar_x_, bar_y_, scratch_kt_y_, scratch_k_x_,
                                  x_restart_, y_restart_, omega_);
    if (averaged.has_value() && *averaged < candidate_gap) {
      candidate_gap = *averaged;
      use_average = true;
    }
  }

  // --- conditions (i) and (ii), against the previous outer loop's gap.
  const bool sufficient =
      candidate_gap <= opt_.pdlp.restart_sufficient * reference_gap_;
  const bool necessary =
      candidate_gap <= opt_.pdlp.restart_necessary * reference_gap_ &&
      candidate_gap > last_candidate_gap_;

  last_candidate_gap_ = candidate_gap;
  if (!sufficient && !necessary && !long_inner_loop) return false;

  // Restart: the candidate becomes both the new iterate and the new anchor.
  if (use_average) {
    for (std::size_t j = 0; j < n_; ++j) x_[j] = bar_x_[j];
    for (std::size_t i = 0; i < m_; ++i) y_[i] = bar_y_[i];
  }
  // Algorithm 1 line 12 updates the primal weight AFTER the restart, from how
  // far the anchor moved. Snapshot the outgoing anchor before it is replaced.
  if (opt_.pdlp.primal_weight_update && have_prev_restart_) {
    Real dx = 0.0;
    for (std::size_t j = 0; j < n_; ++j) {
      const Real d = x_[j] - x_restart_[j];
      dx += d * d;
    }
    Real dy = 0.0;
    for (std::size_t i = 0; i < m_; ++i) {
      const Real d = y_[i] - y_restart_[i];
      dy += d * d;
    }
    dx = std::sqrt(dx);
    dy = std::sqrt(dy);

    // Algorithm 3. Equalizing the primal and dual distances to optimality in
    // the omega-norm means `sqrt(omega)||dx|| = ||dy||/sqrt(omega)`, i.e.
    // `omega = ||dy||/||dx||`. That raw estimate swings hard between
    // restarts, so it is smoothed against the previous weight in LOG space,
    // where the weight is symmetric (`log(1/omega) = -log(omega)`); with
    // `theta = 0.5` that is their geometric mean.
    if (dx > kZeroTolerance && dy > kZeroTolerance) {
      const Real theta = opt_.pdlp.primal_weight_smoothing;
      const Real updated =
          std::exp(theta * std::log(dy / dx) + (1.0 - theta) * std::log(omega_));
      if (std::isfinite(updated) && updated > 0.0) omega_ = updated;
    }
    // Otherwise keep the previous weight: an anchor that did not move in one
    // of the two blocks carries no information about their balance, and
    // dividing by it would manufacture some.
  }
  have_prev_restart_ = true;

  for (std::size_t j = 0; j < n_; ++j) x_restart_[j] = x_[j];
  for (std::size_t i = 0; i < m_; ++i) y_restart_[i] = y_[i];

  avg_x_.assign(0.0);
  avg_y_.assign(0.0);
  weight_sum_ = 0.0;
  inner_iterations_ = 0;
  last_candidate_gap_ = std::numeric_limits<Real>::infinity();
  reference_gap_ = candidate_gap;
  ++restarts_;
  return true;
}

core::Expected<PdlpResult> PdlpSolver::run() {
  x_.resize(n_);
  x_.assign(0.0);
  x_prev_.resize(n_);
  x_prev_.assign(0.0);
  extrapolated_.resize(n_);
  extrapolated_.assign(0.0);
  kt_y_.resize(n_);
  kt_y_.assign(0.0);
  reduced_cost_.resize(n_);
  reduced_cost_.assign(0.0);
  y_.resize(m_);
  y_.assign(0.0);
  k_x_.resize(m_);
  k_x_.assign(0.0);
  primal_residual_.resize(m_);
  primal_residual_.assign(0.0);
  x_trial_.resize(n_);
  x_trial_.assign(0.0);
  y_trial_.resize(m_);
  y_trial_.assign(0.0);
  k_extrapolated_.resize(m_);
  k_extrapolated_.assign(0.0);
  k_x_current_.resize(m_);
  k_x_current_.assign(0.0);
  x_restart_.resize(n_);
  x_restart_.assign(0.0);
  y_restart_.resize(m_);
  y_restart_.assign(0.0);
  avg_x_.resize(n_);
  avg_x_.assign(0.0);
  avg_y_.resize(m_);
  avg_y_.assign(0.0);
  bar_x_.resize(n_);
  bar_x_.assign(0.0);
  bar_y_.resize(m_);
  bar_y_.assign(0.0);
  scratch_kt_y_.resize(n_);
  scratch_kt_y_.assign(0.0);
  scratch_k_x_.resize(m_);
  scratch_k_x_.assign(0.0);
  x_prev_restart_.resize(n_);
  x_prev_restart_.assign(0.0);
  y_prev_restart_.resize(m_);
  y_prev_restart_.assign(0.0);

  // Paper section 4.1: "All first-order methods use all-zero vectors as the
  // initial starting points." Zero is not interior and does not need to be --
  // PDHG projects, it does not follow a barrier.
  for (std::size_t j = 0; j < n_; ++j) {
    x_[j] = std::clamp(0.0, problem_.col_lower[j], problem_.col_upper[j]);
    x_prev_[j] = x_[j];
    x_restart_[j] = x_[j];
  }

  omega_ = opt_.pdlp.primal_weight_update ? initialize_primal_weight(problem_) : 1.0;

  const bool adaptive = opt_.pdlp.adaptive_step_size;

  // Algorithm 1 line 2 starts the adaptive rule from `1/||K||_inf`, one sweep.
  // The fixed rule needs the far more expensive `0.9/||K||_2` because nothing
  // downstream will correct a bad guess.
  const Real scale = adaptive ? infinity_norm(problem_)
                              : estimate_spectral_norm(matvec_, opt_);
  if (!(scale > 0.0) || !std::isfinite(scale)) {
    // A zero matrix has no coupling between primal and dual; there is nothing
    // for PDHG to iterate on and the caller should not be told it converged.
    Convergence conv;
    evaluate(conv);
    return pack(conv.converged(opt_.pdlp.termination_tolerance)
                    ? SolverStatus::Optimal
                    : SolverStatus::NotConverged,
                conv);
  }

  Real eta = adaptive ? 1.0 / scale : opt_.pdlp.step_size_fraction / scale;
  const Real fixed_tau = eta / omega_;
  const Real fixed_sigma = eta * omega_;

  const std::size_t budget =
      opt_.pdlp.max_iterations != 0 ? opt_.pdlp.max_iterations : kDefaultMaxIterations;
  const std::size_t interval = std::max<std::size_t>(opt_.pdlp.check_interval, 1);

  const auto start = std::chrono::steady_clock::now();
  const bool has_time_limit = opt_.limits.time_limit_seconds > 0.0;

  const bool restarts_enabled = opt_.pdlp.adaptive_restart;

  Convergence conv;
  evaluate(conv);
  if (conv.converged(opt_.pdlp.termination_tolerance)) {
    return pack(SolverStatus::Optimal, conv);
  }

  std::size_t iteration = 0;
  SolverStatus outcome = SolverStatus::MaxIterations;

  while (iteration < budget) {
    const Real step_taken = eta;
    if (adaptive) {
      adaptive_step(iteration, eta);
    } else {
      fixed_step(fixed_tau, fixed_sigma);
    }
    if (restarts_enabled) {
      // Algorithm 1 line 7 weights each iterate by the step size that
      // produced it, so a long step counts for more in the average than a
      // short one -- which is what makes the average meaningful when the
      // adaptive rule is varying the step by an order of magnitude.
      accumulate_average(adaptive ? step_taken : fixed_tau);
      ++inner_iterations_;
    }

    ++iteration;

    if (iteration % interval == 0) {
      evaluate(conv);
      if (conv.converged(opt_.pdlp.termination_tolerance)) {
        outcome = SolverStatus::Optimal;
        break;
      }
      if (!std::isfinite(conv.primal) || !std::isfinite(conv.dual) ||
          !std::isfinite(conv.gap)) {
        outcome = SolverStatus::NumericalError;
        break;
      }
      if (has_time_limit) {
        const std::chrono::duration<double> elapsed =
            std::chrono::steady_clock::now() - start;
        if (elapsed.count() >= opt_.limits.time_limit_seconds) {
          outcome = SolverStatus::TimeLimit;
          break;
        }
      }
      // Same schedule as the termination check, and for the same reason: the
      // gap evaluations cost matrix products that do not advance the iterate
      // (paper section 3, "we only evaluate the restart or termination
      // criteria every 40 iterations").
      if (restarts_enabled) {
        (void)maybe_restart(iteration);
      }
    }
  }

  // The loop may have exited on the iteration budget between two checks, in
  // which case `conv` is stale by up to `interval` iterations. Re-measure so
  // the reported residuals describe the iterate actually returned.
  evaluate(conv);
  if (conv.converged(opt_.pdlp.termination_tolerance)) outcome = SolverStatus::Optimal;

  PdlpResult result = pack(outcome, conv);
  result.iterations = iteration;
  return result;
}

}  // namespace

core::Expected<PdlpResult> solve_pdlp(const CanonicalProblem& problem,
                                      const Options& options) {
  if (!problem.Q.empty()) {
    return core::make_error(ErrorCode::UnsupportedFeature,
                            "solve_pdlp: PDLP solves linear programs; this model has "
                            "a quadratic objective");
  }
  PdlpSolver solver(problem, options);
  return solver.run();
}

}  // namespace sovsolve::solver::pdlp
