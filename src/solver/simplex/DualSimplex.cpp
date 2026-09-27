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
  DualSolver(const model::CanonicalProblem& problem, const model::Options& options,
             const std::vector<Real>* costs)
      : SimplexEngine(problem, options, costs) {
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
  [[nodiscard]] bool certificate_holds_for_true_bounds() const;

  [[nodiscard]] Step iterate();
  /// The Farkas certificate, captured at the pivot that proved infeasibility.
  /// Empty unless this run ended with `Step::Infeasible`.
  std::vector<Real> certificate_;

  [[nodiscard]] std::size_t choose_leaving(Real& delta, Real& sigma);
  void init_weights();
  [[nodiscard]] Real weight(std::size_t slot);
  void update_weights(std::size_t leaving_slot, std::size_t entering);
  void export_weights();
  [[nodiscard]] bool collect_candidates(Real sigma);
  void apply_flips();

  [[nodiscard]] bool ray_is_unbounded(const std::vector<Index>& moving);
  [[nodiscard]] bool any_unbounded_ray();

  std::vector<Candidate> candidates_;
  std::vector<Index> flips_;
  std::vector<Index> offenders_;
  std::vector<Index> single_;
  Real current_bound_ = 0.0;  ///< artificial bound magnitude currently installed

  // Dual steepest edge (Koberstein section 3.3), per SLOT. `weight_owner_[r]`
  // is the variable slot r's weight belongs to: a basis repair swaps a
  // logical into a slot without a pivot, and a weight must not be read for a
  // variable it was not computed for -- such a slot restarts at 1.
  std::vector<Real> weights_;
  std::vector<Index> weight_owner_;
  std::vector<Real> tau_;  ///< B^-1 rho_r, the one extra FTRAN per iteration
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

bool DualSolver::certificate_holds_for_true_bounds() const {
  // An infeasibility found while artificial boxes are installed is a proof
  // about the BOXED problem -- unless the certificate never leans on a box.
  // Checked directly, by Farkas: every point with `Ahat v = b` satisfies
  // `y'b = sum_w g_w v_w` with `g = Ahat' y`, so if `y'b` lies outside the
  // range that sum can take over the TRUE bounds, no such point exists. A
  // column with `g_w != 0` toward an infinite true bound makes the range
  // unbounded on that side, and the proof does not transfer.
  //
  // Both sides are tested, so the sign convention of `certificate_` does not
  // matter. Without this, a model like Netlib BGPRTR -- infeasible, with
  // columns the phase 1 had to box -- widened its boxes until the round limit
  // and ended NotConverged, which left `compute_iis` with nothing to explain.
  if (certificate_.size() != m_ || m_ == 0) return false;
  Real y_scale = 0.0;
  Real yb = 0.0;
  for (std::size_t i = 0; i < m_; ++i) {
    y_scale = std::fmax(y_scale, std::fabs(certificate_[i]));
    yb += certificate_[i] * matrix_.problem().b[i];
  }
  if (!(y_scale > 0.0)) return false;

  // `g_w` below this is rounding in `B^-T e_r`, not a column the proof uses.
  const Real zero = 1e-9 * y_scale;
  Real low = 0.0;
  Real high = 0.0;
  bool low_finite = true;
  bool high_finite = true;
  Real magnitude = std::fabs(yb);
  for (std::size_t w = 0; w < total_; ++w) {
    Real g = 0.0;
    matrix_.for_each_in_column(w, [&](std::size_t i, Real a) { g += a * certificate_[i]; });
    if (std::fabs(g) <= zero) continue;
    const Real lo = true_lower_[w];
    const Real up = true_upper_[w];
    // g * v over [lo, up]: the minimum sits at lo when g > 0, at up when g < 0.
    const Real at_min = g > 0.0 ? lo : up;
    const Real at_max = g > 0.0 ? up : lo;
    if (is_finite_bound(at_min)) {
      low += g * at_min;
      magnitude += std::fabs(g * at_min);
    } else {
      low_finite = false;
    }
    if (is_finite_bound(at_max)) {
      high += g * at_max;
      magnitude += std::fabs(g * at_max);
    } else {
      high_finite = false;
    }
  }

  // The violation must be material, not rounding: the same primal tolerance
  // that decides whether a row is violated at all, scaled to the certificate,
  // plus a relative guard for the sums' own cancellation.
  const Real tol = opt_.primal_feasibility_tolerance * y_scale + 1e-9 * magnitude;
  return (high_finite && yb > high + tol) || (low_finite && yb < low - tol);
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

// --------------------------------------------------------------------------
// Dual steepest edge weights
// --------------------------------------------------------------------------

/// The floor Koberstein section 8.2.2.1 applies to every updated weight, "as
/// recommended in [26]" (Forrest and Goldfarb): the update formula can round a
/// small weight negative, and a non-positive weight would make its row's score
/// infinite or meaningless.
constexpr Real kMinWeight = 1e-4;

void DualSolver::init_weights() {
  // "If we start the dual simplex method with an all logical basis, then the
  // dual steepest edge weights can be initialized with 1.0, which is the
  // correct value" (chapter 7, on crash bases) -- B = I, so B^-T e_r = e_r. For any other
  // basis the exact values cost up to m BTRANs; the thesis's branch-and-bound
  // default is to "reuse the weights of the last LP-iteration" instead, which
  // is what a warm start's `dse_weights` carries. A variable with no carried
  // weight starts at 1.
  weights_.assign(m_, 1.0);
  weight_owner_.assign(basis_.basic.begin(), basis_.basic.end());
  tau_.assign(m_, 0.0);
  if (basis_.dse_weights.size() != total_) return;
  for (std::size_t r = 0; r < m_; ++r) {
    const Real w = basis_.dse_weights[static_cast<std::size_t>(basis_.basic[r])];
    if (std::isfinite(w) && w >= kMinWeight) weights_[r] = w;
  }
}

Real DualSolver::weight(std::size_t slot) {
  if (weight_owner_[slot] != basis_.basic[slot]) {
    weight_owner_[slot] = basis_.basic[slot];
    weights_[slot] = 1.0;
  }
  return weights_[slot];
}

void DualSolver::update_weights(std::size_t leaving_slot, std::size_t entering) {
  // Koberstein section 8.2.2.1, in its order:
  //
  //  1. "recompute the value of the weight beta_r = rho_r' rho_r by its
  //     definition" -- rho_r = B^-T e_r is already in `rho_` from the pivot
  //     row, and every other weight's update uses beta_r, so an exact one
  //     stops error from accumulating through it;
  //  2. beta_r' = beta_r / alpha_qr^2, (3.47a), with alpha_qr taken from the
  //     FTRAN'd column -- "generally of higher numerical precision" than the
  //     BTRAN version;
  //  3. tau = B^-1 rho_r, (3.49), one extra FTRAN on the OLD basis;
  //  4. every other weight by the reorganized (8.1),
  //         beta_i' = beta_i + alpha_qi (alpha_qi beta_r' + kappa tau_i),
  //         kappa = -2 / alpha_qr,
  //     which the thesis found "worked substantially better" than (3.50)
  //     because it divides before it squares;
  //  5. floor at 1e-4.
  const Real alpha_r = column_[leaving_slot];
  Real beta_r = 0.0;
  for (std::size_t i = 0; i < m_; ++i) beta_r += rho_[i] * rho_[i];

  tau_.assign(rho_.begin(), rho_.end());
  lu_.ftran(core::HostSpan<Real>(tau_.data(), tau_.size()));

  const Real beta_r_new = std::max(beta_r / (alpha_r * alpha_r), kMinWeight);
  const Real kappa = -2.0 / alpha_r;
  for (std::size_t i = 0; i < m_; ++i) {
    if (i == leaving_slot) continue;
    const Real alpha_i = column_[i];
    if (alpha_i == 0.0) continue;
    const Real w = weight(i) + alpha_i * (alpha_i * beta_r_new + kappa * tau_[i]);
    weights_[i] = std::max(w, kMinWeight);
  }
  weights_[leaving_slot] = beta_r_new;
  weight_owner_[leaving_slot] = static_cast<Index>(entering);
}

void DualSolver::export_weights() {
  basis_.dse_weights.assign(total_, 1.0);
  for (std::size_t r = 0; r < m_; ++r) {
    basis_.dse_weights[static_cast<std::size_t>(basis_.basic[r])] = weight(r);
  }
}

std::size_t DualSolver::choose_leaving(Real& delta, Real& sigma) {
  // Among the rows violated by more than the primal feasibility tolerance:
  //
  //   Dantzig             the largest violation;
  //   dual steepest edge  the largest  violation^2 / beta_r,  Koberstein
  //                       (3.53) -- violation per unit length of the dual
  //                       edge it moves along, so the choice no longer
  //                       depends on how the rows happen to be scaled
  //                       (section 3.3, (3.51)).
  //
  // With bounds present the weights are those of the standard-form edges, not
  // the exact bounded ones; the thesis keeps them anyway ("for most problems
  // this additional effort does not pay off"), which makes (3.53) a heuristic.
  const bool dse = opt_.dual_steepest_edge;
  std::size_t best = m_;
  Real best_score = 0.0;
  for (std::size_t r = 0; r < m_; ++r) {
    const auto bw = static_cast<std::size_t>(basis_.basic[r]);
    const Real xv = x_basic_[r];
    Real violation = lower_[bw] - xv;
    Real side = -1.0;
    const Real above = xv - upper_[bw];
    if (above > violation) {
      violation = above;
      side = 1.0;
    }
    if (violation <= opt_.primal_feasibility_tolerance) continue;
    const Real score = dse ? violation * violation / weight(r) : violation;
    if (best == m_ || score > best_score) {
      best_score = score;
      best = r;
      delta = violation;
      sigma = side;
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
    //
    // Keep the proof. `rho_` is `B^-T e_r`, already computed for the ratio
    // test, and `sigma * rho_` is the direction the dual objective improves
    // along -- a Farkas certificate. Discarding it here and reconstructing it
    // later would mean re-deriving from a basis that has since moved.
    certificate_.assign(m_, 0.0);
    for (std::size_t i = 0; i < m_; ++i) certificate_[i] = sigma * rho_[i];
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
        //
        // Pass the breakpoint only if the violation stays MATERIALLY open
        // after the flip -- open by more than the primal feasibility
        // tolerance, the same threshold `choose_leaving` uses to call a row
        // violated at all. Comparing against zero instead declared a FEASIBLE
        // model infeasible: when the flips close the violation EXACTLY -- a
        // row feasible only at its boundary, as in a MILP relaxation with a
        // single feasible point -- scaling leaves a residual of ~1e-16, the
        // last column is flipped rather than entered, no candidate is left,
        // and the exit below reports an infeasibility that is pure rounding.
        // Found by milp_test's enumeration oracle (seed 23); the unscaled
        // model, whose integer data cancels exactly, solved correctly.
        const Real reduction = std::fabs(cand.arow) * (upper_[w] - lower_[w]);
        if (remaining - reduction > opt_.primal_feasibility_tolerance) {
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
      //
      // This is the SECOND place infeasibility is proved, and it produces the
      // same certificate as the first -- `sigma * rho_` is still the direction
      // the dual objective improves along. Capturing it in only one of the two
      // exits leaves `compute_iis` unable to explain a whole class of models:
      // `x >= 5` against a bound of `x <= 3` reaches infeasibility HERE, by
      // flipping `x` to its upper bound and finding the violation still open,
      // never through the other exit at all.
      certificate_.assign(m_, 0.0);
      for (std::size_t i = 0; i < m_; ++i) certificate_[i] = sigma * rho_[i];
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

  if (opt_.dual_steepest_edge) update_weights(leaving_slot, entering);

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
  if (opt_.dual_steepest_edge) init_weights();

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
      // Another engine already won the race. Stop where we are and report
      // NotConverged -- we did not prove anything, and the driver discards
      // this result anyway.
      if (iterations_ % kTimeCheckInterval == 0 && cancelled()) {
        outcome = core::SolverStatus::NotConverged;
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
    //
    // Unless the certificate itself never uses a box: then it proves the
    // MODEL infeasible, and widening further would only throw the proof away.
    if (outcome == core::SolverStatus::Infeasible && any_artificial_installed() &&
        certificate_holds_for_true_bounds()) {
      break;
    }
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

  if (opt_.dual_steepest_edge) export_weights();
  SimplexResult packed = pack_result(outcome);
  if (outcome == core::SolverStatus::Infeasible) {
    packed.infeasibility_certificate = certificate_;
  }
  return packed;
}

}  // namespace

core::Expected<SimplexResult> solve_dual_simplex(const model::CanonicalProblem& problem,
                                                 const model::Options& options,
                                                 const Basis* warm_start,
                                                 const std::vector<Real>* costs) {
  if (!problem.validate()) {
    return core::make_error(ErrorCode::DimensionMismatch,
                            "canonical problem failed its own validate()");
  }
  if (costs != nullptr && costs->size() != problem.num_cols()) {
    return core::make_error(ErrorCode::DimensionMismatch,
                            "solve_dual_simplex: cost override has the wrong length");
  }
  DualSolver solver(problem, options, costs);
  return solver.run(warm_start);
}

}  // namespace sovsolve::solver::simplex
