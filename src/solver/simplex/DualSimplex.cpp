#include "sovsolve/solver/simplex/DualSimplex.hpp"

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

/// One eligible column in the dual ratio test.
struct Candidate {
  Index w = 0;
  Real ratio = 0.0;
  Real arow = 0.0;
};

enum class Step : std::uint8_t {
  Pivoted,      ///< a basis change happened
  Optimal,      ///< no primal infeasibility remains under the working bounds
  Infeasible,   ///< dual objective unbounded: a primal infeasibility proof
  Refactorize,  ///< the step was not numerically usable; rebuild and retry
};

class DualSolver : public SimplexEngine {
 public:
  DualSolver(const model::CanonicalProblem& problem, const model::Options& options)
      : SimplexEngine(problem, options) {
    // Set before the first refactorize(), which may re-establish dual
    // feasibility and therefore install artificial bounds of this size.
    current_bound_ = opt_.artificial_bound;
  }

  [[nodiscard]] core::Expected<SimplexResult> run(const Basis* warm_start);

 private:
  void on_refactorized(bool repaired) override;

  [[nodiscard]] std::size_t restore_dual_feasibility(Real bound);
  void reflip_dual_infeasible();
  void escalate_artificial_bounds(Real bound);
  void collect_artificial_offenders();
  [[nodiscard]] bool any_artificial_installed() const;

  [[nodiscard]] Step iterate();
  [[nodiscard]] std::size_t choose_leaving(Real& delta, Real& sigma) const;
  [[nodiscard]] bool collect_candidates(Real sigma);
  void apply_flips();

  [[nodiscard]] bool ray_is_unbounded(const std::vector<Index>& moving);
  [[nodiscard]] bool any_unbounded_ray();

  std::vector<Candidate> candidates_;
  std::vector<Index> flips_;
  std::vector<Index> offenders_;
  std::vector<Index> single_;
  Real current_bound_ = 0.0;  ///< artificial bound magnitude currently installed
};

// --------------------------------------------------------------------------
// Dual feasibility -- the invariant this algorithm protects
// --------------------------------------------------------------------------

void DualSolver::reflip_dual_infeasible() {
  // The cheap half of dual-feasibility repair: a column whose WORKING bounds
  // are both finite and which is resting on the dual-infeasible side simply
  // moves across. No bound changes, so the only primal effect is the column's
  // own value moving between two bounds it already had.
  //
  // Deliberately does NOT install artificial bounds, which the full
  // `restore_dual_feasibility` does. Installing one mid-solve teleports a
  // nonbasic from a finite bound to +/- `artificial_bound`, a jump of 1e7 in
  // the primal values of every basic variable its column touches. Measured:
  // doing that at every refactorization took Netlib `stair` from Optimal at
  // -251.267 in 3357 pivots to MaxIterations at -4.2e7. Phase 1 installs
  // boxes; refactorization only re-places columns inside them.
  for (std::size_t w = 0; w < total_; ++w) {
    const VarStatus st = basis_.status[w];
    if (st == VarStatus::Basic || st == VarStatus::Fixed) continue;
    if (!is_finite_bound(lower_[w]) || !is_finite_bound(upper_[w])) continue;
    if (st == VarStatus::AtLower && dj_[w] < -opt_.dual_feasibility_tolerance) {
      basis_.status[w] = VarStatus::AtUpper;
      value_[w] = upper_[w];
    } else if (st == VarStatus::AtUpper && dj_[w] > opt_.dual_feasibility_tolerance) {
      basis_.status[w] = VarStatus::AtLower;
      value_[w] = lower_[w];
    }
  }
}

std::size_t DualSolver::restore_dual_feasibility(Real bound) {
  std::size_t installed = 0;
  for (std::size_t w = 0; w < total_; ++w) {
    const VarStatus st = basis_.status[w];
    if (st == VarStatus::Basic || st == VarStatus::Fixed) continue;

    const bool lo_finite = is_finite_bound(true_lower_[w]);
    const bool up_finite = is_finite_bound(true_upper_[w]);
    const Real cost = dj_[w];

    if (lo_finite && up_finite) {
      // A boxed column always has a dual-feasible side; if it is resting on
      // the wrong one, moving it across costs nothing and needs no artificial
      // bound. A no-op at startup (make_logical_basis already places boxed
      // columns by the sign of their cost) and load-bearing after a basis
      // repair, which moves the duals under columns already placed.
      if (st == VarStatus::AtLower && cost < -opt_.dual_feasibility_tolerance) {
        basis_.status[w] = VarStatus::AtUpper;
        value_[w] = upper_[w];
      } else if (st == VarStatus::AtUpper && cost > opt_.dual_feasibility_tolerance) {
        basis_.status[w] = VarStatus::AtLower;
        value_[w] = lower_[w];
      }
      continue;
    }

    if (!lo_finite && !up_finite) {
      // A free column is dual feasible only at `d_j == 0` exactly, so it has
      // no bound to rest on and the iteration cannot start with it nonbasic.
      // Boxing it gives it one; the box is removed by the check at optimality.
      //
      // (The primal simplex needs none of this -- a free column is an ordinary
      // entering candidate there, which is one reason it makes a good cleanup
      // phase for the cases this machinery cannot close.)
      lower_[w] = -bound;
      upper_[w] = bound;
      artificial_[w] = 3;
      basis_.status[w] = cost >= 0.0 ? VarStatus::AtLower : VarStatus::AtUpper;
      value_[w] = working_value(w);
      ++installed;
      continue;
    }

    if (lo_finite && cost < -opt_.dual_feasibility_tolerance) {
      // Dual feasibility wants this column at an upper bound it does not have.
      upper_[w] = true_lower_[w] + bound;
      artificial_[w] = 2;
      basis_.status[w] = VarStatus::AtUpper;
      value_[w] = upper_[w];
      ++installed;
    } else if (up_finite && cost > opt_.dual_feasibility_tolerance) {
      lower_[w] = true_upper_[w] - bound;
      artificial_[w] = 1;
      basis_.status[w] = VarStatus::AtLower;
      value_[w] = lower_[w];
      ++installed;
    }
  }
  return installed;
}

void DualSolver::escalate_artificial_bounds(Real bound) {
  for (std::size_t w = 0; w < total_; ++w) {
    if (artificial_[w] == 0) continue;
    if ((artificial_[w] & 1) != 0) {
      lower_[w] = is_finite_bound(true_upper_[w]) ? true_upper_[w] - bound : -bound;
    }
    if ((artificial_[w] & 2) != 0) {
      upper_[w] = is_finite_bound(true_lower_[w]) ? true_lower_[w] + bound : bound;
    }
    if (basis_.status[w] != VarStatus::Basic) value_[w] = working_value(w);
  }
}

bool DualSolver::any_artificial_installed() const {
  for (std::size_t w = 0; w < total_; ++w) {
    if (artificial_[w] != 0) return true;
  }
  return false;
}

void DualSolver::collect_artificial_offenders() {
  offenders_.clear();
  for (std::size_t w = 0; w < total_; ++w) {
    if (artificial_[w] == 0) continue;
    const VarStatus st = basis_.status[w];
    if (st == VarStatus::Basic) continue;
    const bool on_fake_upper = (artificial_[w] & 2) != 0 && st == VarStatus::AtUpper;
    const bool on_fake_lower = (artificial_[w] & 1) != 0 && st == VarStatus::AtLower;
    if (!on_fake_upper && !on_fake_lower) continue;
    // A zero reduced cost means the objective does not care where this column
    // sits, so resting on a made-up bound is harmless -- the point is still
    // optimal for the true model.
    if (std::fabs(dj_[w]) <= opt_.dual_feasibility_tolerance) continue;
    offenders_.push_back(static_cast<Index>(w));
  }
}

void DualSolver::on_refactorized(bool repaired) {
  // Dual feasibility is this algorithm's standing invariant: the ratio test's
  // `d_j / arow_j >= 0` holds only while it does, and every termination
  // verdict is derived from that ratio. The incremental dual update
  // (`d_j -= theta_d * arow_j`) accumulates rounding, so the invariant is
  // re-established here on freshly computed reduced costs rather than assumed
  // to have survived -- but only by re-placing columns within the bounds they
  // already have.
  //
  // A basis repair is the one case that needs the full treatment: it moves the
  // duals outright, and a column left dual infeasible on a side it has no
  // bound for has nowhere to go without one.
  if (repaired) {
    (void)restore_dual_feasibility(current_bound_);
  } else {
    reflip_dual_infeasible();
  }
}

// --------------------------------------------------------------------------
// The iteration
// --------------------------------------------------------------------------

std::size_t DualSolver::choose_leaving(Real& delta, Real& sigma) const {
  // Dantzig pricing: the largest bound violation, unweighted. Dual steepest
  // edge divides this by a reference-framework norm so the choice is violation
  // per unit of dual movement; it needs weight updates threaded through every
  // pivot, and a provably correct iteration is worth more first than a fast
  // one.
  std::size_t best = m_;
  Real worst = opt_.primal_feasibility_tolerance;
  for (std::size_t r = 0; r < m_; ++r) {
    const auto bw = static_cast<std::size_t>(basis_.basic[r]);
    const Real xv = x_basic_[r];
    const Real below = lower_[bw] - xv;
    if (below > worst) {
      worst = below;
      best = r;
      delta = below;
      sigma = -1.0;
    }
    const Real above = xv - upper_[bw];
    if (above > worst) {
      worst = above;
      best = r;
      delta = above;
      sigma = 1.0;
    }
  }
  return best;
}

bool DualSolver::collect_candidates(Real sigma) {
  candidates_.clear();
  for (const Index j : arow_nz_) {
    const auto w = static_cast<std::size_t>(j);
    const VarStatus st = basis_.status[w];
    if (st == VarStatus::Basic || st == VarStatus::Fixed) continue;

    // `arow_` is the unsigned pivot row; `sigma` folds the two infeasibility
    // cases (basic below its lower bound, basic above its upper) into one set
    // of eligibility rules. See DualSimplex.hpp.
    const Real a = sigma * arow_[w];
    bool eligible = false;
    if (st == VarStatus::AtLower) {
      eligible = a > opt_.pivot_floor;
    } else if (st == VarStatus::AtUpper) {
      eligible = a < -opt_.pivot_floor;
    }
    if (!eligible) continue;

    // `d_j / arow_j` is non-negative whenever the basis is dual feasible: the
    // eligibility rule pairs `d_j >= 0` with `a > 0` and `d_j <= 0` with
    // `a < 0`. A small negative is rounding on a reduced cost that is really
    // zero, and is clamped rather than allowed to produce a negative dual
    // step, which would move the dual objective the wrong way.
    Real ratio = dj_[w] / a;
    if (ratio < 0.0) ratio = 0.0;
    candidates_.push_back(Candidate{static_cast<Index>(w), ratio, a});
  }
  return !candidates_.empty();
}

void DualSolver::apply_flips() {
  if (flips_.empty()) return;
  for (std::size_t i = 0; i < m_; ++i) aggregate_[i] = 0.0;

  for (const Index j : flips_) {
    const auto w = static_cast<std::size_t>(j);
    Real shift = 0.0;
    if (basis_.status[w] == VarStatus::AtLower) {
      shift = upper_[w] - lower_[w];
      basis_.status[w] = VarStatus::AtUpper;
      value_[w] = upper_[w];
    } else {
      shift = lower_[w] - upper_[w];
      basis_.status[w] = VarStatus::AtLower;
      value_[w] = lower_[w];
    }
    if (shift == 0.0) continue;
    matrix_.for_each_in_column(
        w, [&](std::size_t i, Real a) { aggregate_[i] += a * shift; });
    ++bound_flips_;
  }

  // One FTRAN for every flip together: each flipped column shifts `x_B` by
  // `B^-1 Ahat_w * shift`, and the shifts are additive, so accumulating them
  // in row space first turns a solve per flip into a solve per iteration.
  lu_.ftran(core::HostSpan<Real>(aggregate_.data(), aggregate_.size()));
  for (std::size_t i = 0; i < m_; ++i) x_basic_[i] -= aggregate_[i];
}

Step DualSolver::iterate() {
  Real delta = 0.0;
  Real sigma = 0.0;
  const std::size_t leaving_slot = choose_leaving(delta, sigma);
  if (leaving_slot == m_) return Step::Optimal;

  const auto leaving = static_cast<std::size_t>(basis_.basic[leaving_slot]);

  compute_pivot_row(leaving_slot);
  if (!collect_candidates(sigma)) {
    // No column can absorb the dual step: the dual objective improves without
    // limit, which is a proof that the primal is infeasible.
    clear_pivot_row();
    return Step::Infeasible;
  }

  flips_.clear();
  std::size_t entering = total_;
  Real theta_d = 0.0;

  if (opt_.bound_flipping) {
    std::sort(candidates_.begin(), candidates_.end(),
              [](const Candidate& a, const Candidate& b) { return a.ratio < b.ratio; });
    Real remaining = delta;
    for (const Candidate& cand : candidates_) {
      const auto w = static_cast<std::size_t>(cand.w);
      const bool boxed = is_finite_bound(lower_[w]) && is_finite_bound(upper_[w]);
      if (boxed) {
        // Flipping this column to its other bound absorbs `|arow| * range` of
        // the violation at no pivot cost. Its reduced cost crosses zero at
        // this ratio anyway, so the flip is what KEEPS it dual feasible -- not
        // an optimization bolted onto the ratio test, but the ratio test's own
        // consequence.
        const Real reduction = std::fabs(cand.arow) * (upper_[w] - lower_[w]);
        if (reduction < remaining) {
          remaining -= reduction;
          flips_.push_back(cand.w);
          continue;
        }
      }
      entering = w;
      theta_d = cand.ratio;
      break;
    }
    if (entering == total_) {
      // Every eligible column was flippable and all of them together still do
      // not close the violation: nothing blocks the dual step at all.
      clear_pivot_row();
      flips_.clear();
      return Step::Infeasible;
    }
  } else {
    Real best_ratio = std::numeric_limits<Real>::max();
    Real best_arow = 0.0;
    for (const Candidate& cand : candidates_) {
      const bool better = cand.ratio < best_ratio ||
                          (cand.ratio <= best_ratio &&
                           std::fabs(cand.arow) > std::fabs(best_arow));
      if (!better) continue;
      best_ratio = cand.ratio;
      best_arow = cand.arow;
      entering = static_cast<std::size_t>(cand.w);
    }
    theta_d = best_ratio;
  }

  // Flips change `x_B`, so the violation the pivot still has to close must be
  // measured after them, not before.
  apply_flips();

  const Real target = sigma > 0.0 ? upper_[leaving] : lower_[leaving];
  const Real signed_residual =
      sigma > 0.0 ? x_basic_[leaving_slot] - target : target - x_basic_[leaving_slot];
  const Real residual = std::fmax(signed_residual, 0.0);

  load_and_ftran_column(entering);

  const Real alpha = column_[leaving_slot];
  if (std::fabs(alpha) < opt_.pivot_floor) {
    // The pivot row said this column was usable and the freshly solved column
    // says it is not. The factorization is the stale party; rebuild it.
    clear_pivot_row();
    return Step::Refactorize;
  }

  const Real step = residual / (sigma * alpha);

  for (std::size_t i = 0; i < m_; ++i) {
    if (i == leaving_slot) continue;
    x_basic_[i] -= column_[i] * step;
  }
  const Real entering_value = value_[entering] + step;

  value_[leaving] = target;
  if (is_finite_bound(lower_[leaving]) && is_finite_bound(upper_[leaving]) &&
      lower_[leaving] == upper_[leaving]) {
    basis_.status[leaving] = VarStatus::Fixed;
  } else {
    basis_.status[leaving] = sigma > 0.0 ? VarStatus::AtUpper : VarStatus::AtLower;
  }

  if (theta_d != 0.0) {
    const Real signed_theta = theta_d * sigma;
    for (const Index j : arow_nz_) {
      const auto w = static_cast<std::size_t>(j);
      if (basis_.status[w] == VarStatus::Basic) continue;
      dj_[w] -= signed_theta * arow_[w];
    }
    for (std::size_t i = 0; i < m_; ++i) y_[i] += signed_theta * rho_[i];
  }
  // `B^-1 Ahat_leaving` is exactly `e_r`, so the unsigned `arow_[leaving]` is
  // 1 and the leaving column's new reduced cost is `-theta_d * sigma` -- which
  // carries the sign its new status requires, with no separate repair step.
  dj_[leaving] = -theta_d * sigma;
  dj_[entering] = 0.0;

  basis_.basic[leaving_slot] = static_cast<Index>(entering);
  basis_.status[entering] = VarStatus::Basic;
  x_basic_[leaving_slot] = entering_value;

  clear_pivot_row();
  ++iterations_;

  const Status updated = lu_.update(
      leaving_slot, core::HostSpan<const Real>(column_.data(), column_.size()));
  if (!updated.ok()) force_refactor_ = true;
  return Step::Pivoted;
}

// --------------------------------------------------------------------------
// Unboundedness, via the artificial bounds phase 1 installed
// --------------------------------------------------------------------------

bool DualSolver::ray_is_unbounded(const std::vector<Index>& moving) {
  // Move every column in `moving` off its artificial bound simultaneously,
  // each in its own objective-improving direction. The point traces
  //
  //     x_N(t) = x_N + t * dir,   x_B(t) = x_B - t * B^-1 (sum_w Ahat_w dir_w)
  //
  // with a strictly decreasing objective (every term is `d_w * dir_w < 0` by
  // construction). It is a genuine unbounded ray of the MODEL exactly when
  // nothing along it ever hits a TRUE bound -- one FTRAN of the aggregated
  // column plus an O(m) scan. That is a proof, not the inference "the
  // artificial bound kept growing, so it is probably unbounded".
  if (moving.empty()) return false;

  for (std::size_t i = 0; i < m_; ++i) column_[i] = 0.0;
  for (const Index j : moving) {
    const auto w = static_cast<std::size_t>(j);
    const Real direction = dj_[w] < 0.0 ? 1.0 : -1.0;
    if (direction > 0.0 && is_finite_bound(true_upper_[w])) return false;
    if (direction < 0.0 && is_finite_bound(true_lower_[w])) return false;
    matrix_.for_each_in_column(
        w, [&](std::size_t i, Real a) { column_[i] += a * direction; });
  }
  if (m_ > 0) lu_.ftran(core::HostSpan<Real>(column_.data(), column_.size()));

  for (std::size_t r = 0; r < m_; ++r) {
    const Real rate = -column_[r];  // d(x_B[r])/dt
    if (rate == 0.0) continue;
    const auto bw = static_cast<std::size_t>(basis_.basic[r]);
    if (rate > 0.0 && is_finite_bound(true_upper_[bw])) return false;
    if (rate < 0.0 && is_finite_bound(true_lower_[bw])) return false;
  }
  return true;
}

bool DualSolver::any_unbounded_ray() {
  // The joint direction first, because a real ray often needs several columns
  // moving TOGETHER: on `min -x-y s.t. x-y <= 1, -x+y <= 1` neither column is
  // a ray on its own (each drives a slack to zero) while `x = y = t` keeps
  // both rows untouched forever. Testing columns one at a time misses that
  // whole class.
  if (ray_is_unbounded(offenders_)) return true;

  // Then each column alone, since the joint direction can cancel a ray that
  // one column does have.
  for (const Index j : offenders_) {
    single_.assign(1, j);
    if (ray_is_unbounded(single_)) return true;
  }
  return false;
}

// --------------------------------------------------------------------------
// Driver
// --------------------------------------------------------------------------

core::Expected<SimplexResult> DualSolver::run(const Basis* warm_start) {
  load_true_bounds();
  install_basis(warm_start);

  Status status = refactorize();
  if (!status.ok()) {
    // A warm start that no longer factorizes is an expected outcome, not an
    // error: the caller handed over a neighbouring problem's basis and cannot
    // know in advance that a bound change left it singular.
    basis_ = make_logical_basis(matrix_);
    reset_nonbasic_values();
    status = refactorize();
    if (!status.ok()) return status.error();
  }

  // Phase 1, run exactly once. Every column that dual feasibility wants on a
  // side it has no bound for -- and every free column, which has no bound at
  // all -- gets a temporary finite one here. `refactorize()` deliberately does
  // NOT do this on its own (see `reflip_dual_infeasible`): installing a box
  // mid-solve moves a nonbasic by the bound's whole magnitude, which is a
  // perturbation of the primal values, not a repair of them.
  current_bound_ = opt_.artificial_bound;
  if (restore_dual_feasibility(current_bound_) > 0) compute_primal();

  core::SolverStatus outcome = core::SolverStatus::NotConverged;

  for (std::size_t round = 0;; ++round) {
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
      if (step == Step::Optimal || step == Step::Infeasible) {
        // Both are TERMINAL verdicts about the model, so neither may be taken
        // from a product-form factorization that has drifted since its last
        // rebuild. Refactorizing recomputes the primal values, the duals and
        // the dual-feasible placement from the original data; if the verdict
        // survives that, it is a statement about the model rather than about
        // accumulated rounding.
        if (lu_.num_updates() > 0) {
          status = refactorize();
          if (!status.ok()) {
            outcome = core::SolverStatus::NumericalError;
            break;
          }
          continue;
        }
        outcome = step == Step::Optimal ? core::SolverStatus::Optimal
                                        : core::SolverStatus::Infeasible;
        break;
      }

      // Step::Refactorize -- no progress was made, so this must be bounded or
      // an unusable pivot spins forever.
      if (++stuck > kMaxStuckRefactorizations) {
        outcome = core::SolverStatus::NumericalError;
        break;
      }
      status = refactorize();
      if (!status.ok()) {
        outcome = core::SolverStatus::NumericalError;
        break;
      }
    }

    // An `Infeasible` verdict is a statement about the problem the iteration
    // actually solved -- and while phase 1 has artificial bounds installed,
    // that is the BOXED problem, not the model. A box is a real constraint: a
    // model whose feasible points all lie outside it is infeasible for the box
    // and perfectly feasible for itself. Netlib `greenbea` is exactly that
    // case, and reporting the boxed verdict as the model's made the solver
    // claim, confidently, that a model with a published optimum had no
    // feasible point at all.
    //
    // So the box is widened and the question asked again. Only an
    // infeasibility found with NO artificial bound in play is a verdict about
    // the model.
    if (outcome == core::SolverStatus::Infeasible && any_artificial_installed()) {
      if (round >= opt_.max_artificial_rounds) {
        outcome = core::SolverStatus::NotConverged;
        break;
      }
      current_bound_ *= opt_.artificial_bound_growth;
      escalate_artificial_bounds(current_bound_);
      compute_primal();
      outcome = core::SolverStatus::NotConverged;
      continue;
    }

    if (outcome != core::SolverStatus::Optimal) break;

    collect_artificial_offenders();
    if (offenders_.empty()) break;  // optimal for the true model, not just the boxed one

    if (any_unbounded_ray()) {
      outcome = core::SolverStatus::Unbounded;
      break;
    }
    if (round >= opt_.max_artificial_rounds) {
      // The box is still binding and no ray proves unboundedness. Reporting
      // Optimal here would be a wrong answer on a model with a better point
      // outside the box, so this reports "no verdict" instead -- and the
      // composite path (SolveSimplex.hpp) hands the basis to the primal
      // simplex, which has no boxes to be trapped by.
      outcome = core::SolverStatus::NotConverged;
      break;
    }
    current_bound_ *= opt_.artificial_bound_growth;
    escalate_artificial_bounds(current_bound_);
    compute_primal();
    outcome = core::SolverStatus::NotConverged;
  }

  return pack_result(outcome);
}

}  // namespace

core::Expected<SimplexResult> solve_dual_simplex(const model::CanonicalProblem& problem,
                                                 const model::Options& options,
                                                 const Basis* warm_start) {
  if (!problem.validate()) {
    return core::make_error(ErrorCode::DimensionMismatch,
                            "canonical problem failed its own validate()");
  }
  DualSolver solver(problem, options);
  return solver.run(warm_start);
}

}  // namespace sovsolve::solver::simplex
