#include "sovsolve/solver/simplex/PrimalSimplex.hpp"

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

/// Consecutive zero-length steps before the pricing rule switches to Bland's.
///
/// Dantzig pricing can cycle on a degenerate vertex -- the basis changes, the
/// point does not, and the same sequence of bases can repeat forever. Bland's
/// rule (always the lowest-index eligible column, and the lowest-index tie on
/// the way out) is provably non-cycling but picks poor pivots, so it is held
/// back until degeneracy actually persists rather than paid for throughout.
constexpr std::size_t kDegenerateStepsBeforeBland = 100;

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
  }

  [[nodiscard]] core::Expected<SimplexResult> run(const Basis* warm_start);

 private:
  /// The primal simplex protects the PRIMAL point, so unlike the dual it must
  /// not re-place nonbasic columns to tidy up reduced costs -- moving a
  /// nonbasic is a change to the very thing being protected. If a basis repair
  /// has cost primal feasibility, the next iteration simply finds a positive
  /// total infeasibility and re-enters phase 1, which is the correct response
  /// and needs no special handling here.
  void on_refactorized(bool /*repaired*/) override {}

  [[nodiscard]] Real total_infeasibility() const;
  void compute_phase1_duals();
  [[nodiscard]] std::size_t choose_entering(Real& direction) const;
  [[nodiscard]] Step iterate();

  std::vector<Real> d1_;  ///< phase-1 reduced costs, w space
  std::vector<Real> y1_;  ///< phase-1 row duals, row space
  bool phase1_ = false;
  bool bland_ = false;
  std::size_t degenerate_steps_ = 0;
};

// --------------------------------------------------------------------------
// Phase 1: the sum of infeasibilities, and its gradient
// --------------------------------------------------------------------------

Real PrimalSolver::total_infeasibility() const {
  const Real tol = opt_.primal_feasibility_tolerance;
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

void PrimalSolver::compute_phase1_duals() {
  // The phase-1 objective is `sum of bound violations`. Its gradient with
  // respect to a basic variable is -1 where that variable is below its lower
  // bound (increasing it reduces the violation) and +1 where it is above its
  // upper. A nonbasic variable sits exactly on a bound, so it contributes
  // nothing and its own phase-1 cost is zero -- which is why `d1` below is
  // `0 - Ahat_j' y1` rather than `c1_j - Ahat_j' y1`.
  const Real tol = opt_.primal_feasibility_tolerance;
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

    if (bland_) {
      // Bland's rule: the LOWEST eligible index, unconditionally. Scanning in
      // increasing `w` means the first eligible column is it.
      direction = dir;
      return w;
    }
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
  const bool was_phase1 = phase1_;
  phase1_ = total_infeasibility() > opt_.primal_feasibility_tolerance;
  if (phase1_) {
    compute_phase1_duals();
  } else if (was_phase1) {
    // Crossing into phase 2: the true duals have been maintained incrementally
    // through phase 1, but this is a cheap once-per-solve chance to take them
    // from the original data instead of from an accumulated sum.
    compute_dual();
  }

  Real direction = 0.0;
  const std::size_t entering = choose_entering(direction);
  if (entering == total_) {
    // Nothing improves the current objective. In phase 2 that is optimality;
    // in phase 1 it means the MINIMUM total bound violation over the whole
    // polytope is positive, which is a proof that no feasible point exists.
    return phase1_ ? Step::Infeasible : Step::Optimal;
  }

  load_and_ftran_column(entering);

  // Ratio test. `x_q` moves by `t >= 0` in `direction`, so basic variable `r`
  // moves at `rate = -alpha[r] * direction` per unit of `t`.
  const Real tol = opt_.primal_feasibility_tolerance;
  Real best_t = kNoLimit;
  std::size_t leaving_slot = m_;
  Real leaving_target = 0.0;
  bool leaving_to_upper = false;

  // The entering column's own opposite bound. Reaching it first means a bound
  // flip: the column crosses its range and no basis change happens at all.
  const bool entering_boxed =
      is_finite_bound(lower_[entering]) && is_finite_bound(upper_[entering]);
  if (entering_boxed) best_t = upper_[entering] - lower_[entering];

  for (std::size_t r = 0; r < m_; ++r) {
    const Real rate = -column_[r] * direction;
    if (std::fabs(rate) < opt_.pivot_floor) continue;

    const auto bw = static_cast<std::size_t>(basis_.basic[r]);
    const Real x = x_basic_[r];
    const Real lo = lower_[bw];
    const Real up = upper_[bw];

    Real t = kNoLimit;
    Real target = 0.0;
    bool to_upper = false;

    if (x < lo - tol) {
      // Infeasible below. Moving up, it becomes feasible on reaching `lo` --
      // a breakpoint of the phase-1 objective, so the step stops there.
      // Moving down it just gets worse, and blocks nothing.
      if (rate <= 0.0) continue;
      t = (lo - x) / rate;
      target = lo;
    } else if (x > up + tol) {
      if (rate >= 0.0) continue;
      t = (up - x) / rate;
      target = up;
      to_upper = true;
    } else if (rate > 0.0) {
      if (!is_finite_bound(up)) continue;
      t = (up - x) / rate;
      target = up;
      to_upper = true;
    } else {
      if (!is_finite_bound(lo)) continue;
      t = (lo - x) / rate;
      target = lo;
    }

    if (t < 0.0) t = 0.0;  // already at or just past the bound: a degenerate step
    const bool better =
        t < best_t ||
        (bland_ && t <= best_t && leaving_slot != m_ &&
         basis_.basic[r] < basis_.basic[leaving_slot]);
    if (!better) continue;
    best_t = t;
    leaving_slot = r;
    leaving_target = target;
    leaving_to_upper = to_upper;
  }

  if (best_t >= kNoLimit) {
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

  if (best_t > 0.0) {
    // Real movement: the degenerate run is over, so Dantzig pricing comes back.
    // Leaving Bland's rule latched on is a correctness-preserving disaster --
    // it keeps picking the lowest-index eligible column for the rest of the
    // solve, and measured on Netlib `greenbea` that left phase 1 still 1.7
    // away from feasible after 600,000 pivots. Bland's is an escape from
    // cycling, not a pricing rule to settle into.
    degenerate_steps_ = 0;
    bland_ = false;
  } else {
    ++degenerate_steps_;
    if (degenerate_steps_ > kDegenerateStepsBeforeBland) bland_ = true;
  }

  // Case 1: the entering column reached its own opposite bound first.
  if (leaving_slot == m_) {
    for (std::size_t i = 0; i < m_; ++i) x_basic_[i] -= column_[i] * direction * best_t;
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

  // Case 2: an ordinary pivot.
  const Real alpha = column_[leaving_slot];
  if (std::fabs(alpha) < opt_.pivot_floor) return Step::Refactorize;

  const auto leaving = static_cast<std::size_t>(basis_.basic[leaving_slot]);

  // The dual update needs the pivot row, and it must be taken BEFORE the basis
  // changes underneath it.
  compute_pivot_row(leaving_slot);
  const Real theta = dj_[entering] / alpha;

  for (std::size_t i = 0; i < m_; ++i) {
    if (i == leaving_slot) continue;
    x_basic_[i] -= column_[i] * direction * best_t;
  }
  const Real entering_value = value_[entering] + direction * best_t;

  value_[leaving] = leaving_target;
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
    if (force_refactor_ || lu_.num_updates() >= opt_.refactor_interval) {
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
