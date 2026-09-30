#include "sovsolve/solver/simplex/PrimalSimplex.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "detail/SimplexEngine.hpp"

namespace sovsolve::solver::simplex {
namespace {

using core::ErrorCode;
using core::Index;
using core::is_finite_bound;
using detail::kMaxStuckRefactorizations;
using detail::kTimeCheckInterval;
using detail::SimplexEngine;

/// "No bound stops the step." Distinguished from a very large finite ratio so
/// that unboundedness is decided by the absence of a blocking event, not by a
/// magnitude comparison.
constexpr Real kNoLimit = std::numeric_limits<Real>::max();

// --------------------------------------------------------------------------
// EXPAND: Gill, Murray, Saunders & Wright, "A practical anti-cycling procedure
// for linearly constrained optimization", Math. Prog. 45 (1989)
// --------------------------------------------------------------------------
//
// On a degenerate vertex the textbook ratio test returns a step of zero: the
// basis changes, the point does not, and nothing stops the same bases from
// repeating. EXPAND makes every step positive instead. A working feasibility
// tolerance `delta` grows by `tau` at every iteration (section 4.2); each
// variable was within the previous tolerance, so each is strictly inside the
// new one and a step of at least `tau / |pivot|` is always possible (4.3).
// With the objective strictly falling at every step no basis can repeat.
//
// The price is that a leaving variable may become nonbasic slightly OFF its
// bound (by at most `delta`). It keeps that value until a reset (section 4.3)
// puts every nonbasic back on its bound: after `K` iterations, and on each
// tentative verdict.
//
// This replaced Bland's rule after 100 zero steps, which cannot cycle but
// picks such poor pivots that Netlib `truss` had not finished after
// 491,000 iterations.

/// K, the iterations in one expanding sequence (section 4.2: K = eps^(-1/4)
/// = 10^4 for double precision).
constexpr std::size_t kExpandIterations = 10000;

/// delta_0 and delta_K as fractions of the master tolerance delta_f
/// (section 4.2: 0.5 and 0.99). delta_f is OUR `primal_feasibility_tolerance`
/// (1e-7) rather than the paper's eps^(3/8) = 1e-6, so a verdict means what it
/// meant before EXPAND.
constexpr Real kExpandStart = 0.5;
constexpr Real kExpandEnd = 0.99;

/// R, the resets allowed on a tentative verdict (section 4.3 suggests 1, or 2
/// for badly conditioned problems). After R, the point is accepted as it
/// stands: every variable is within delta < delta_f of its bounds.
constexpr std::size_t kTerminationResets = 2;

enum class Step : std::uint8_t {
  Pivoted,      ///< a basis change, or a bound flip, happened
  Optimal,      ///< primal feasible with no improving column
  Infeasible,   ///< phase 1 minimized the total violation and it is positive
  Unbounded,    ///< an improving column with nothing to block it: a ray
  Refactorize,  ///< the step was not numerically usable; rebuild and retry
};

class PrimalSolver : public SimplexEngine {
 public:
  PrimalSolver(const model::CanonicalProblem& problem, const model::Options& options)
      : SimplexEngine(problem, options) {
    d1_.assign(total_, 0.0);
    y1_.assign(m_, 0.0);
    preserve_nonbasic_values_ = true;
    const Real master = opt_.primal_feasibility_tolerance;
    delta_start_ = kExpandStart * master;
    tau_ = (kExpandEnd - kExpandStart) * master / static_cast<Real>(kExpandIterations);
    delta_ = delta_start_;
  }

  [[nodiscard]] core::Expected<SimplexResult> run(const Basis* warm_start);

 private:
  /// The primal simplex protects the PRIMAL point, so unlike the dual it must
  /// not re-place nonbasic columns to tidy up reduced costs -- moving a
  /// nonbasic is a change to the very thing being protected. If a basis repair
  /// has cost primal feasibility, the next iteration simply finds a positive
  /// total infeasibility and re-enters phase 1, which is the correct response.
  /// A repair re-placed every nonbasic on its bound, which is an EXPAND reset,
  /// so the expanding sequence restarts with it.
  void on_refactorized(bool repaired) override {
    if (repaired) restart_expanding_sequence();
  }

  /// A basic variable counts as infeasible when it is beyond `delta`, the
  /// working tolerance (section 7.1).
  [[nodiscard]] Real total_infeasibility() const;
  [[nodiscard]] Real max_infeasibility() const;
  void compute_phase1_duals();
  [[nodiscard]] std::size_t choose_entering(Real& direction) const;
  [[nodiscard]] Step iterate();

  void restart_expanding_sequence() {
    delta_ = delta_start_;
    expand_iterations_ = 0;
  }
  /// Section 4.3: every nonbasic back on its bound, the basic variables
  /// recomputed from them, and a new expanding sequence.
  [[nodiscard]] Status reset();
  [[nodiscard]] bool nonbasic_off_bounds() const;

  std::vector<Real> d1_;  ///< phase-1 reduced costs, w space
  std::vector<Real> y1_;  ///< phase-1 row duals, row space
  bool phase1_ = false;

  Real delta_ = 0.0;        ///< the working feasibility tolerance
  Real delta_start_ = 0.0;  ///< delta_0
  Real tau_ = 0.0;          ///< its growth per iteration
  std::size_t expand_iterations_ = 0;
  std::size_t termination_resets_ = 0;
};

// --------------------------------------------------------------------------
// Phase 1: the sum of infeasibilities, and its gradient
// --------------------------------------------------------------------------

Real PrimalSolver::total_infeasibility() const {
  const Real tol = delta_;
  Real sum = 0.0;
  for (std::size_t r = 0; r < m_; ++r) {
    const auto bw = static_cast<std::size_t>(basis_.basic[r]);
    const Real x = x_basic_[r];
    if (x < lower_[bw] - tol) {
      sum += lower_[bw] - x;
    } else if (x > upper_[bw] + tol) {
      sum += x - upper_[bw];
    }
  }
  return sum;
}

Real PrimalSolver::max_infeasibility() const {
  Real worst = 0.0;
  for (std::size_t r = 0; r < m_; ++r) {
    const auto bw = static_cast<std::size_t>(basis_.basic[r]);
    const Real x = x_basic_[r];
    worst = std::max(worst, std::max(lower_[bw] - x, x - upper_[bw]));
  }
  return worst;
}

bool PrimalSolver::nonbasic_off_bounds() const {
  for (std::size_t w = 0; w < total_; ++w) {
    const VarStatus st = basis_.status[w];
    if (st == VarStatus::Basic || st == VarStatus::Free) continue;
    if (value_[w] != working_value(w)) return true;
  }
  return false;
}

Status PrimalSolver::reset() {
  reset_nonbasic_values();
  restart_expanding_sequence();
  return refactorize();
}

void PrimalSolver::compute_phase1_duals() {
  // The phase-1 objective is `sum of bound violations`. Its gradient with
  // respect to a basic variable is -1 where that variable is below its lower
  // bound (increasing it reduces the violation) and +1 where it is above its
  // upper. A nonbasic variable sits exactly on a bound, so it contributes
  // nothing and its own phase-1 cost is zero -- which is why `d1` below is
  // `0 - Ahat_j' y1` rather than `c1_j - Ahat_j' y1`. (Under EXPAND a
  // nonbasic may be off its bound, but never by more than `delta`, so it is
  // never counted infeasible.)
  const Real tol = delta_;
  for (std::size_t r = 0; r < m_; ++r) {
    const auto bw = static_cast<std::size_t>(basis_.basic[r]);
    const Real x = x_basic_[r];
    if (x < lower_[bw] - tol) {
      y1_[r] = -1.0;
    } else if (x > upper_[bw] + tol) {
      y1_[r] = 1.0;
    } else {
      y1_[r] = 0.0;
    }
  }
  if (m_ > 0) lu_.btran(core::HostSpan<Real>(y1_.data(), y1_.size()));

  for (std::size_t w = 0; w < total_; ++w) {
    if (basis_.status[w] == VarStatus::Basic) {
      d1_[w] = 0.0;
      continue;
    }
    Real dot = 0.0;
    matrix_.for_each_in_column(w, [&](std::size_t i, Real a) { dot += a * y1_[i]; });
    d1_[w] = -dot;
  }
}

// --------------------------------------------------------------------------
// The iteration
// --------------------------------------------------------------------------

std::size_t PrimalSolver::choose_entering(Real& direction) const {
  const std::vector<Real>& d = phase1_ ? d1_ : dj_;
  const Real threshold = opt_.dual_feasibility_tolerance;

  std::size_t best = total_;
  Real best_score = threshold;
  Real best_direction = 0.0;

  for (std::size_t w = 0; w < total_; ++w) {
    const VarStatus st = basis_.status[w];
    if (st == VarStatus::Basic || st == VarStatus::Fixed) continue;
    // A column pinned between equal bounds cannot move, so it is not a
    // candidate however attractive its reduced cost looks.
    if (is_finite_bound(lower_[w]) && is_finite_bound(upper_[w]) &&
        lower_[w] == upper_[w]) {
      continue;
    }

    Real score = 0.0;
    Real dir = 0.0;
    if (st == VarStatus::AtLower) {
      score = -d[w];  // improving iff d < 0
      dir = 1.0;
    } else if (st == VarStatus::AtUpper) {
      score = d[w];  // improving iff d > 0
      dir = -1.0;
    } else {
      // Free, resting at zero: it may move either way, so any nonzero reduced
      // cost is an improving direction. The dual simplex has to box such a
      // column before it can start; here it is an ordinary candidate.
      score = std::fabs(d[w]);
      dir = d[w] < 0.0 ? 1.0 : -1.0;
    }
    if (score <= threshold) continue;
    if (score > best_score) {
      best_score = score;
      best = w;
      best_direction = dir;
    }
  }

  direction = best_direction;
  return best;
}

Step PrimalSolver::iterate() {
  // Section 4.2: the working tolerance grows at the start of every iteration,
  // so every variable -- within the previous tolerance -- is strictly inside
  // the new one.
  delta_ += tau_;
  ++expand_iterations_;

  const bool was_phase1 = phase1_;
  phase1_ = total_infeasibility() > 0.0;
  if (phase1_) {
    compute_phase1_duals();
  } else if (was_phase1) {
    // Crossing into phase 2: the true duals have been maintained incrementally
    // through phase 1, but this is a cheap once-per-solve chance to take them
    // from the original data instead of from an accumulated sum.
    compute_dual();
  }

  Real direction = 0.0;
  std::size_t entering = choose_entering(direction);
  if (entering == total_ && phase1_ &&
      max_infeasibility() <= opt_.primal_feasibility_tolerance) {
    // Phase 1 is stuck on violations the working tolerance counts but the
    // caller's tolerance accepts: `delta` restarts at half of it after every
    // reset. Before EXPAND this point was feasible, so it must not become a
    // proof of infeasibility now. Widen `delta` to cover it and go on in
    // phase 2; the next reset brings `delta` back down.
    delta_ = std::max(delta_, max_infeasibility());
    phase1_ = false;
    compute_dual();
    entering = choose_entering(direction);
  }
  if (entering == total_) {
    // Nothing improves the current objective. In phase 2 that is optimality;
    // in phase 1 it means the MINIMUM total bound violation over the whole
    // polytope is positive, which is a proof that no feasible point exists.
    return phase1_ ? Step::Infeasible : Step::Optimal;
  }

  load_and_ftran_column(entering);

  // `x_q` moves by `t >= 0` in `direction`, so basic variable `r` moves at
  // `rate = -alpha[r] * direction` per unit of `t`. Each basic variable is
  // classified against the working tolerance: beyond it (phase 1 only) the
  // variable blocks where it BECOMES feasible, a breakpoint of the phase-1
  // objective; within it, the variable blocks at the bound it moves toward.
  const Real delta = delta_;

  // Pass 1 (section 4.1, Harris's first pass): the largest step that keeps
  // every basic variable within its bounds widened by `delta`.
  Real max_step = kNoLimit;
  for (std::size_t r = 0; r < m_; ++r) {
    const Real rate = -column_[r] * direction;
    if (std::fabs(rate) < opt_.pivot_floor) continue;

    const auto bw = static_cast<std::size_t>(basis_.basic[r]);
    const Real x = x_basic_[r];
    const Real lo = lower_[bw];
    const Real up = upper_[bw];

    Real t = kNoLimit;
    if (x < lo - delta) {
      // Moving down it just gets worse, and blocks nothing.
      if (rate <= 0.0) continue;
      t = (lo - x) / rate;
    } else if (x > up + delta) {
      if (rate >= 0.0) continue;
      t = (up - x) / rate;
    } else if (rate > 0.0) {
      if (!is_finite_bound(up)) continue;
      t = (up + delta - x) / rate;
    } else {
      if (!is_finite_bound(lo)) continue;
      t = (lo - delta - x) / rate;
    }
    // Negative only if a refactorization moved a variable past the tolerance.
    max_step = std::min(max_step, std::max(t, 0.0));
  }

  // The entering column's own opposite bound. Reaching it within the pass-1
  // step means a bound flip: the column crosses its range, lands exactly on
  // the other bound, and no basis change happens at all.
  const bool entering_boxed =
      is_finite_bound(lower_[entering]) && is_finite_bound(upper_[entering]);
  if (entering_boxed) {
    const Real flip = direction > 0.0 ? upper_[entering] - value_[entering]
                                      : value_[entering] - lower_[entering];
    if (flip <= max_step) {
      for (std::size_t i = 0; i < m_; ++i) x_basic_[i] -= column_[i] * direction * flip;
      if (basis_.status[entering] == VarStatus::AtLower) {
        basis_.status[entering] = VarStatus::AtUpper;
        value_[entering] = upper_[entering];
      } else {
        basis_.status[entering] = VarStatus::AtLower;
        value_[entering] = lower_[entering];
      }
      ++bound_flips_;
      ++iterations_;
      if (phase1_) ++phase1_iterations_;
      return Step::Pivoted;
    }
  }

  if (max_step >= kNoLimit) {
    if (phase1_) {
      // The phase-1 objective is bounded below by zero, so it cannot improve
      // without limit. Reaching here means the column or the values are stale.
      return Step::Refactorize;
    }
    // Phase 2, an improving column, and nothing anywhere stops it: the point
    // runs to infinity along a genuine ray of the feasible region and the
    // objective falls forever. A direct certificate, produced by the ratio
    // test itself.
    return Step::Unbounded;
  }

  // Pass 2 (section 4.1): among the variables whose step to their EXACT bound
  // is within the pass-1 step, the one with the largest pivot leaves.
  std::size_t leaving_slot = m_;
  Real pivot = 0.0;
  Real exact_step = 0.0;
  bool leaving_to_upper = false;
  for (std::size_t r = 0; r < m_; ++r) {
    const Real rate = -column_[r] * direction;
    if (std::fabs(rate) < opt_.pivot_floor) continue;

    const auto bw = static_cast<std::size_t>(basis_.basic[r]);
    const Real x = x_basic_[r];
    const Real lo = lower_[bw];
    const Real up = upper_[bw];

    Real t = kNoLimit;
    bool to_upper = false;
    if (x < lo - delta) {
      if (rate <= 0.0) continue;
      t = (lo - x) / rate;
    } else if (x > up + delta) {
      if (rate >= 0.0) continue;
      t = (up - x) / rate;
      to_upper = true;
    } else if (rate > 0.0) {
      if (!is_finite_bound(up)) continue;
      t = (up - x) / rate;
      to_upper = true;
    } else {
      if (!is_finite_bound(lo)) continue;
      t = (lo - x) / rate;
    }
    if (t > max_step || std::fabs(rate) <= pivot) continue;
    pivot = std::fabs(rate);
    exact_step = t;
    leaving_slot = r;
    leaving_to_upper = to_upper;
  }
  if (leaving_slot == m_) return Step::Refactorize;

  // The step is at least `tau / |pivot|` (section 4.1, `alpha_min`), so it is
  // never zero. It may leave the blocking variable past its bound by at most
  // `delta`; it becomes nonbasic at that value. Capped at the pass-1 step,
  // which is smaller only after a refactorization moved a variable past the
  // tolerance.
  const Real step = std::max(exact_step, std::min(tau_ / pivot, max_step));

  // Case 2: an ordinary pivot.
  const Real alpha = column_[leaving_slot];
  if (std::fabs(alpha) < opt_.pivot_floor) return Step::Refactorize;

  const auto leaving = static_cast<std::size_t>(basis_.basic[leaving_slot]);

  // The dual update needs the pivot row, and it must be taken BEFORE the basis
  // changes underneath it.
  compute_pivot_row(leaving_slot);
  const Real theta = dj_[entering] / alpha;

  // The leaving variable keeps the value the step gives it -- on its bound, or
  // past it by at most `delta` -- rather than being moved onto the bound: that
  // move would break `A x = b` by up to `delta` (section 3.3).
  const Real leaving_value = x_basic_[leaving_slot] - alpha * direction * step;
  for (std::size_t i = 0; i < m_; ++i) {
    if (i == leaving_slot) continue;
    x_basic_[i] -= column_[i] * direction * step;
  }
  const Real entering_value = value_[entering] + direction * step;

  value_[leaving] = leaving_value;
  if (is_finite_bound(lower_[leaving]) && is_finite_bound(upper_[leaving]) &&
      lower_[leaving] == upper_[leaving]) {
    basis_.status[leaving] = VarStatus::Fixed;
  } else {
    basis_.status[leaving] = leaving_to_upper ? VarStatus::AtUpper : VarStatus::AtLower;
  }

  // `y' = y + (d_q / alpha_r) * rho` is the dual update for replacing slot r
  // with column q. It holds for ANY such basis change, so it is correct during
  // phase 1 too even though phase 1 pivots on a different objective -- which
  // is why the true duals need no reconstruction when phase 2 begins.
  if (theta != 0.0) {
    for (const Index j : arow_nz_) {
      const auto w = static_cast<std::size_t>(j);
      if (basis_.status[w] == VarStatus::Basic) continue;
      dj_[w] -= theta * arow_[w];
    }
    for (std::size_t i = 0; i < m_; ++i) y_[i] += theta * rho_[i];
  }
  // `B^-1 Ahat_leaving` is exactly `e_r`, so `arow_[leaving] == 1`.
  dj_[leaving] = -theta;
  dj_[entering] = 0.0;

  basis_.basic[leaving_slot] = static_cast<Index>(entering);
  basis_.status[entering] = VarStatus::Basic;
  x_basic_[leaving_slot] = entering_value;

  clear_pivot_row();
  ++iterations_;
  if (phase1_) ++phase1_iterations_;

  const Status updated = lu_.update(
      leaving_slot, core::HostSpan<const Real>(column_.data(), column_.size()));
  if (!updated.ok()) force_refactor_ = true;
  return Step::Pivoted;
}

// --------------------------------------------------------------------------
// Driver
// --------------------------------------------------------------------------

core::Expected<SimplexResult> PrimalSolver::run(const Basis* warm_start) {
  load_true_bounds();
  install_basis(warm_start);

  Status status = refactorize();
  if (!status.ok()) {
    basis_ = make_logical_basis(matrix_);
    reset_nonbasic_values();
    status = refactorize();
    if (!status.ok()) return status.error();
  }

  core::SolverStatus outcome = core::SolverStatus::NotConverged;
  std::size_t stuck = 0;

  for (;;) {
    if (iterations_ >= max_iterations_) {
      outcome = core::SolverStatus::MaxIterations;
      break;
    }
    if (iterations_ % kTimeCheckInterval == 0 && time_exhausted()) {
      outcome = core::SolverStatus::TimeLimit;
      break;
    }
    // Another engine already won the race -- see DualSimplex.cpp.
    if (iterations_ % kTimeCheckInterval == 0 && cancelled()) {
      outcome = core::SolverStatus::NotConverged;
      break;
    }
    if (expand_iterations_ >= kExpandIterations) {
      // The end of an expanding sequence (section 4.3).
      status = reset();
      if (!status.ok()) {
        outcome = core::SolverStatus::NumericalError;
        break;
      }
    } else if (force_refactor_ || lu_.num_updates() >= opt_.refactor_interval) {
      status = refactorize();
      if (!status.ok()) {
        outcome = core::SolverStatus::NumericalError;
        break;
      }
    }

    const Step step = iterate();
    if (step == Step::Pivoted) {
      stuck = 0;
      continue;
    }

    if (step == Step::Refactorize) {
      if (++stuck > kMaxStuckRefactorizations) {
        outcome = core::SolverStatus::NumericalError;
        break;
      }
      status = refactorize();
      if (!status.ok()) {
        outcome = core::SolverStatus::NumericalError;
        break;
      }
      continue;
    }

    // Optimal, Infeasible and Unbounded are all terminal verdicts about the
    // model, so none may be taken from a factorization that has drifted since
    // its last rebuild -- same rule the dual simplex applies, for the same
    // reason.
    if (lu_.num_updates() > 0) {
      status = refactorize();
      if (!status.ok()) {
        outcome = core::SolverStatus::NumericalError;
        break;
      }
      continue;
    }

    // A tentative verdict with nonbasic variables still off their bounds: put
    // them back and check the verdict again (section 4.3), at most R times.
    // On a well-conditioned model the verdict holds straight after the reset.
    // The same goes for a working tolerance widened past the caller's (see
    // the phase-1 case in `iterate`): the reset brings it back to delta_0.
    if (termination_resets_ < kTerminationResets &&
        (nonbasic_off_bounds() || delta_ > opt_.primal_feasibility_tolerance)) {
      ++termination_resets_;
      status = reset();
      if (!status.ok()) {
        outcome = core::SolverStatus::NumericalError;
        break;
      }
      continue;
    }

    switch (step) {
      case Step::Optimal:
        outcome = core::SolverStatus::Optimal;
        break;
      case Step::Infeasible:
        outcome = core::SolverStatus::Infeasible;
        break;
      case Step::Unbounded:
        outcome = core::SolverStatus::Unbounded;
        break;
      case Step::Pivoted:
      case Step::Refactorize:
        break;
    }
    break;
  }

  return pack_result(outcome);
}

}  // namespace

core::Expected<SimplexResult> solve_primal_simplex(const model::CanonicalProblem& problem,
                                                   const model::Options& options,
                                                   const Basis* warm_start) {
  if (!problem.validate()) {
    return core::make_error(ErrorCode::DimensionMismatch,
                            "canonical problem failed its own validate()");
  }
  PrimalSolver solver(problem, options);
  return solver.run(warm_start);
}

}  // namespace sovsolve::solver::simplex
