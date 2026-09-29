#include "sovsolve/solver/Crossover.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "sovsolve/solver/simplex/LuFactor.hpp"

namespace sovsolve::solver {

namespace {

using core::INF;
using core::Real;
using simplex::AugmentedMatrix;
using simplex::Basis;
using simplex::VarStatus;

/// A nonbasic variable this close to a bound (relative to the bound's size)
/// is put exactly on it instead of being pushed. OURS: the tolerance engines
/// stop at 1e-8 relative, so a variable that belongs on a bound usually sits
/// within a few 1e-9 of it; snapping at 1e-9 moves A x by less than the
/// engine's own residual.
constexpr Real kSnapTolerance = 1e-9;

/// Ratio-test pivot threshold: entries of B^-1 a_j smaller than this cannot
/// block a push. Absolute, on the scaled model, like kMinEtaPivot's 1e-11 in
/// LuFactor.hpp but larger, because a push that pivots on a tiny entry puts a
/// badly conditioned column into the basis.
constexpr Real kPushPivotTolerance = 1e-9;

/// Harris's relaxation of bounds in the first ratio-test pass (Harris,
/// "Pivot selection methods of the Devex LP code", Math. Programming 5, 1973):
/// among blocking candidates within this much of the tightest, take the one
/// with the largest pivot.
constexpr Real kHarrisTolerance = 1e-9;

/// A basic variable whose reduced cost in the point's duals exceeds this is a
/// dual superbasic, and the dual push tries to move it out of the basis.
/// OURS: the tolerance engines' duals are accurate to about 1e-8 relative on
/// the scaled model, so anything below 1e-9 is noise, not a signal.
constexpr Real kDualSuperbasicTolerance = 1e-9;

/// Product-form updates between refactorizations, the same trade-off as
/// SimplexOptions::refactor_interval's default.
constexpr std::size_t kRefactorInterval = 100;

[[nodiscard]] Real snap_tol(Real bound) { return kSnapTolerance * (1.0 + std::abs(bound)); }

}  // namespace

Basis crossover_crash_basis(const model::CanonicalProblem& problem,
                            const model::Solution& point) {
  const AugmentedMatrix matrix(problem);
  const std::size_t n = matrix.num_structural();
  const std::size_t m = matrix.num_rows();
  const std::size_t total = n + m;

  std::vector<Real> activity(m, 0.0);
  const auto& csc = problem.A.csc;
  for (std::size_t j = 0; j < n; ++j) {
    const Real xj = point.x[j];
    if (xj == 0.0) continue;
    for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      activity[static_cast<std::size_t>(csc.indices()[k])] += csc.values()[k] * xj;
    }
  }

  const auto dual_or_zero = [](const core::RealVector& d, std::size_t j) {
    return j < d.size() ? std::abs(d[j]) : 0.0;
  };
  // Andersen & Ye's indicator g / (g + d). 0/0 -- on a bound with a zero
  // dual, degenerate both ways -- ranks with the nonbasic ones.
  const auto indicator = [](Real g, Real d) {
    g = std::max(g, 0.0);
    return g + d > 0.0 ? g / (g + d) : 0.0;
  };

  std::vector<Real> value(total);
  std::vector<Real> score(total);  // larger = more basic
  for (std::size_t w = 0; w < total; ++w) {
    const Real v = w < n ? point.x[w] : problem.b[w - n] - activity[w - n];
    const Real lo = matrix.lower(w);
    const Real hi = matrix.upper(w);
    value[w] = v;
    if (lo == hi) {
      score[w] = -INF;
    } else if (lo == -INF && hi == INF) {
      score[w] = INF;
    } else if (w >= n) {
      // An inequality row's slack; its dual is -y_i (docs/FORMULATION.md).
      score[w] = indicator(v - lo, dual_or_zero(point.y, w - n));
    } else if (hi == INF || (lo != -INF && v - lo <= hi - v)) {
      score[w] = indicator(v - lo, dual_or_zero(point.z, w));
    } else {
      score[w] = indicator(hi - v, dual_or_zero(point.v, w));
    }
  }

  // Ties break on index, so the same point always gives the same basis.
  std::vector<std::size_t> order(total);
  for (std::size_t w = 0; w < total; ++w) order[w] = w;
  std::stable_sort(order.begin(), order.end(),
                   [&score](std::size_t a, std::size_t b) { return score[a] > score[b]; });

  Basis basis;
  basis.status.assign(total, VarStatus::AtLower);
  for (std::size_t w = 0; w < total; ++w) {
    const Real lo = matrix.lower(w);
    const Real hi = matrix.upper(w);
    if (lo == hi) {
      basis.status[w] = VarStatus::Fixed;
    } else if (lo == -INF && hi == INF) {
      basis.status[w] = VarStatus::Free;
    } else if (lo == -INF || (hi != INF && hi - value[w] < value[w] - lo)) {
      basis.status[w] = VarStatus::AtUpper;
    }
  }
  std::vector<std::size_t> chosen(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(m));
  std::sort(chosen.begin(), chosen.end());
  basis.basic.reserve(m);
  for (const std::size_t w : chosen) {
    basis.status[w] = VarStatus::Basic;
    basis.basic.push_back(static_cast<core::Index>(w));
  }
  return basis;
}

core::Expected<Basis> crossover_basis(const model::CanonicalProblem& problem,
                                      const model::Solution& point, Real pivot_tolerance,
                                      double time_limit_seconds, CrossoverStats* stats) {
  const auto start = std::chrono::steady_clock::now();
  const auto elapsed = [&start]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };

  const AugmentedMatrix matrix(problem);
  const std::size_t n = matrix.num_structural();
  const std::size_t m = matrix.num_rows();
  const std::size_t total = n + m;
  if (point.x.size() != n) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "crossover: the point does not match the problem");
  }
  CrossoverStats local;
  CrossoverStats& st = stats != nullptr ? *stats : local;
  st = CrossoverStats{};

  // The point in augmented space: structurals, then each row's logical
  // xi = b - A x, all clipped into their bounds. What clipping and the
  // engine's own residual leave of A w = b is left alone: every push keeps
  // A w unchanged, and the cleanup simplex recomputes the basic values from
  // the nonbasic ones exactly.
  std::vector<Real> w(total, 0.0);
  {
    std::vector<Real> activity(m, 0.0);
    const auto& csc = problem.A.csc;
    for (std::size_t j = 0; j < n; ++j) {
      w[j] = point.x[j];
      if (w[j] == 0.0) continue;
      for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
        activity[static_cast<std::size_t>(csc.indices()[k])] += csc.values()[k] * w[j];
      }
    }
    for (std::size_t i = 0; i < m; ++i) w[n + i] = problem.b[i] - activity[i];
    for (std::size_t v = 0; v < total; ++v) {
      w[v] = std::clamp(w[v], matrix.lower(v), matrix.upper(v));
    }
  }

  Basis basis = crossover_crash_basis(problem, point);
  simplex::LuFactorization lu;
  {
    std::size_t repairs = 0;
    const core::Status factored = lu.factorize_repairing(matrix, basis, pivot_tolerance, &repairs);
    if (!factored.ok()) return factored.error();
    st.repairs += repairs;
  }
  st.crash_seconds = elapsed();

  // Put every nonbasic variable that is on (or within kSnapTolerance of) a
  // bound exactly there, and collect the rest: the superbasics.
  std::vector<std::size_t> superbasic;
  const auto classify = [&](std::size_t v) {
    const Real lo = matrix.lower(v);
    const Real hi = matrix.upper(v);
    if (lo == hi) {
      w[v] = lo;
      basis.status[v] = VarStatus::Fixed;
    } else if (lo == -INF && hi == INF) {
      if (std::abs(w[v]) <= kSnapTolerance) {
        w[v] = 0.0;
        basis.status[v] = VarStatus::Free;
      } else {
        superbasic.push_back(v);
      }
    } else if (lo != -INF && w[v] - lo <= snap_tol(lo)) {
      w[v] = lo;
      basis.status[v] = VarStatus::AtLower;
    } else if (hi != INF && hi - w[v] <= snap_tol(hi)) {
      w[v] = hi;
      basis.status[v] = VarStatus::AtUpper;
    } else {
      superbasic.push_back(v);
    }
  };
  for (std::size_t v = 0; v < total; ++v) {
    if (basis.status[v] != VarStatus::Basic) classify(v);
  }
  st.superbasic = superbasic.size();

  std::vector<Real> column(m, 0.0);
  std::size_t since_refactor = 0;
  const auto refactor = [&]() -> core::Status {
    // A repair displaces columns and gives them a model bound as status; any
    // of them left off its bound goes back on the superbasic list.
    std::size_t repairs = 0;
    const core::Status factored = lu.factorize_repairing(matrix, basis, pivot_tolerance, &repairs);
    if (!factored.ok()) return factored;
    ++st.refactorizations;
    st.repairs += repairs;
    since_refactor = 0;
    if (repairs > 0) {
      for (std::size_t v = 0; v < total; ++v) {
        if (basis.status[v] == VarStatus::Basic) continue;
        const Real lo = matrix.lower(v);
        const Real hi = matrix.upper(v);
        const bool on_bound = (lo != -INF && w[v] == lo) || (hi != INF && w[v] == hi) ||
                              (lo == -INF && hi == INF && w[v] == 0.0);
        if (!on_bound) classify(v);
      }
    }
    return factored;
  };

  // Every push removes one superbasic, and a repair can add at most one per
  // displaced column, so this bound is generous.
  const std::size_t max_steps = 4 * (total + 1);
  std::size_t steps = 0;
  for (std::size_t next = 0; next < superbasic.size(); ++next) {
    if (++steps > max_steps) {
      return core::make_error(core::ErrorCode::NumericalError,
                              "crossover: primal push did not terminate");
    }
    if ((steps & 63u) == 0 && elapsed() > time_limit_seconds) {
      return core::make_error(core::ErrorCode::NumericalError, "crossover: time limit");
    }
    const std::size_t j = superbasic[next];
    if (basis.status[j] == VarStatus::Basic) continue;  // entered through an exchange
    {
      // Listed twice (a repair re-lists what it displaces) and already done.
      const Real lo = matrix.lower(j);
      const Real hi = matrix.upper(j);
      if ((lo != -INF && w[j] == lo) || (hi != INF && w[j] == hi)) continue;
    }

    // alpha = B^-1 a_j: moving w_j by +1 moves the basic variables by -alpha.
    std::fill(column.begin(), column.end(), 0.0);
    matrix.for_each_in_column(j, [&column](std::size_t row, Real value) { column[row] = value; });
    if (m > 0) lu.ftran(core::HostSpan<Real>(column.data(), column.size()));

    // Reduced cost d_j = c_j - c_B' alpha picks the direction that does not
    // make the objective worse.
    Real d = matrix.cost(j);
    for (std::size_t r = 0; r < m; ++r) {
      d -= matrix.cost(static_cast<std::size_t>(basis.basic[r])) * column[r];
    }
    const Real lo = matrix.lower(j);
    const Real hi = matrix.upper(j);
    int first = d < 0.0 ? +1 : d > 0.0 ? -1 : 0;
    if (first == 0) {
      // Indifferent: head for the nearer finite bound.
      const Real to_lo = lo == -INF ? INF : w[j] - lo;
      const Real to_hi = hi == INF ? INF : hi - w[j];
      first = to_hi < to_lo ? +1 : -1;
    }

    bool moved = false;
    for (const int dir : {first, -first}) {
      const Real own = dir > 0 ? (hi == INF ? INF : hi - w[j]) : (lo == -INF ? INF : w[j] - lo);

      // Harris pass 1: the largest step no basic variable overshoots its
      // bound by more than kHarrisTolerance.
      Real relaxed = own;
      for (std::size_t r = 0; r < m; ++r) {
        const Real a = column[r];
        if (std::abs(a) <= kPushPivotTolerance) continue;
        const auto b = static_cast<std::size_t>(basis.basic[r]);
        const Real rate = -static_cast<Real>(dir) * a;  // d w_b / d t
        if (rate < 0.0 && matrix.lower(b) != -INF) {
          relaxed = std::min(relaxed, (w[b] - matrix.lower(b) + kHarrisTolerance) / -rate);
        } else if (rate > 0.0 && matrix.upper(b) != INF) {
          relaxed = std::min(relaxed, (matrix.upper(b) - w[b] + kHarrisTolerance) / rate);
        }
      }
      if (relaxed == INF) continue;  // unbounded this way; try the other

      // Pass 2: of the basic variables that block within `relaxed`, the one
      // with the largest pivot.
      std::size_t leave = m;
      Real leave_step = INF;
      Real best_pivot = 0.0;
      for (std::size_t r = 0; r < m; ++r) {
        const Real a = column[r];
        if (std::abs(a) <= kPushPivotTolerance) continue;
        const auto b = static_cast<std::size_t>(basis.basic[r]);
        const Real rate = -static_cast<Real>(dir) * a;
        Real step = INF;
        if (rate < 0.0 && matrix.lower(b) != -INF) {
          step = (w[b] - matrix.lower(b)) / -rate;
        } else if (rate > 0.0 && matrix.upper(b) != INF) {
          step = (matrix.upper(b) - w[b]) / rate;
        }
        if (step <= relaxed && std::abs(a) > best_pivot) {
          best_pivot = std::abs(a);
          leave = r;
          leave_step = std::max(step, 0.0);
        }
      }

      const bool to_own_bound = leave == m || own <= leave_step;
      const Real t = to_own_bound ? own : leave_step;
      for (std::size_t r = 0; r < m; ++r) {
        if (column[r] != 0.0) {
          w[static_cast<std::size_t>(basis.basic[r])] -= static_cast<Real>(dir) * t * column[r];
        }
      }
      if (to_own_bound) {
        w[j] = dir > 0 ? hi : lo;
        basis.status[j] = dir > 0 ? VarStatus::AtUpper : VarStatus::AtLower;
        ++st.pushed_to_bound;
      } else {
        w[j] += static_cast<Real>(dir) * t;
        const auto b = static_cast<std::size_t>(basis.basic[leave]);
        const bool at_lower = -static_cast<Real>(dir) * column[leave] < 0.0;
        w[b] = at_lower ? matrix.lower(b) : matrix.upper(b);
        basis.status[b] = matrix.lower(b) == matrix.upper(b) ? VarStatus::Fixed
                          : at_lower                          ? VarStatus::AtLower
                                                              : VarStatus::AtUpper;
        basis.basic[leave] = static_cast<core::Index>(j);
        basis.status[j] = VarStatus::Basic;
        ++st.pivots;
        const core::Status updated =
            lu.update(leave, core::HostSpan<const Real>(column.data(), column.size()));
        if (!updated.ok() || ++since_refactor >= kRefactorInterval) {
          const core::Status factored = refactor();
          if (!factored.ok()) return factored.error();
        }
      }
      moved = true;
      break;
    }

    if (!moved) {
      // No bound blocks in either direction: a free variable on a direction
      // of zero cost. Exchange it into the basis without moving it, for the
      // basic variable with the largest pivot; that one, if it is off its
      // bounds, becomes a superbasic in turn.
      std::size_t leave = m;
      Real best_pivot = kPushPivotTolerance;
      for (std::size_t r = 0; r < m; ++r) {
        if (std::abs(column[r]) > best_pivot) {
          best_pivot = std::abs(column[r]);
          leave = r;
        }
      }
      if (leave == m) {
        // A column the basis cannot reach at all -- only possible for an
        // all-zero column, which canonicalization removes. Rest it where it is
        // allowed to rest.
        w[j] = lo == -INF && hi == INF ? 0.0 : lo != -INF ? lo : hi;
        basis.status[j] = lo == -INF && hi == INF ? VarStatus::Free
                          : lo != -INF             ? VarStatus::AtLower
                                                   : VarStatus::AtUpper;
        continue;
      }
      const auto b = static_cast<std::size_t>(basis.basic[leave]);
      basis.basic[leave] = static_cast<core::Index>(j);
      basis.status[j] = VarStatus::Basic;
      basis.status[b] = VarStatus::AtLower;  // provisional; classify() places it
      ++st.pivots;
      const core::Status updated =
          lu.update(leave, core::HostSpan<const Real>(column.data(), column.size()));
      if (!updated.ok() || ++since_refactor >= kRefactorInterval) {
        const core::Status factored = refactor();
        if (!factored.ok()) return factored.error();
      }
      if (basis.status[b] != VarStatus::Basic) classify(b);
    }
  }

  // ---------------------------------------------------------------------------
  // Dual push. The basis is now primal feasible, but a basic variable whose
  // reduced cost in the point's duals is clearly nonzero belongs on a bound:
  // at a basic dual solution every basic reduced cost is zero. For each such
  // DUAL SUPERBASIC j in slot r, move the duals along rho = B^-T e_r, which
  // changes the reduced costs by  d_k -= theta * alpha_rk  (alpha_r the
  // tableau row; alpha_rj = 1, alpha_r = 0 on the other basic columns). Going
  // toward d_j = 0, the first nonbasic k whose reduced cost would cross zero
  // blocks; then k enters in slot r and j leaves onto the bound it already
  // sits on -- a primal-degenerate pivot, so the point does not move. If
  // nothing blocks, d_j reaches zero and j stays basic, now consistent.
  // ---------------------------------------------------------------------------
  st.primal_push_seconds = elapsed() - st.crash_seconds;
  if (point.y.size() == m) {
    // Reduced costs from the point's own duals: d = c - A'y, and -y_i for a
    // row's logical (its column is e_i and its cost is zero).
    std::vector<Real> d(total, 0.0);
    const auto& csc = problem.A.csc;
    for (std::size_t j = 0; j < n; ++j) {
      Real dj = matrix.cost(j);
      for (std::size_t k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
        dj -= csc.values()[k] * point.y[static_cast<std::size_t>(csc.indices()[k])];
      }
      d[j] = dj;
    }
    for (std::size_t i = 0; i < m; ++i) d[n + i] = -point.y[i];

    // Dual feasibility of a nonbasic reduced cost, by where it rests.
    const auto blocks = [&](std::size_t k, Real rate) {
      // `rate` = d(d_k)/d(tau). Returns the tau at which d_k reaches zero from
      // its feasible side, or INF when it cannot block.
      const VarStatus s = basis.status[k];
      if (s == VarStatus::Basic || s == VarStatus::Fixed) return INF;
      if (std::abs(rate) <= kPushPivotTolerance) return INF;
      if (s == VarStatus::AtLower) {  // needs d_k >= 0
        return rate < 0.0 ? std::max(d[k], 0.0) / -rate : INF;
      }
      if (s == VarStatus::AtUpper) {  // needs d_k <= 0
        return rate > 0.0 ? std::max(-d[k], 0.0) / rate : INF;
      }
      return 0.0;  // Free: any movement of d_k leaves zero at once
    };

    std::vector<Real> rho(m, 0.0);
    std::vector<Real> row(total, 0.0);  // alpha_r over every variable; zero outside `touched`
    std::vector<std::size_t> touched;   // variables with a (possibly) nonzero alpha_rk
    std::vector<char> seen(total, 0);
    const auto& csr = problem.A.csr;
    for (std::size_t r = 0; r < m; ++r) {
      if ((r & 63u) == 0 && elapsed() > time_limit_seconds) {
        return core::make_error(core::ErrorCode::NumericalError, "crossover: time limit");
      }
      const auto j = static_cast<std::size_t>(basis.basic[r]);
      const Real dj = d[j];
      if (std::abs(dj) <= kDualSuperbasicTolerance) continue;
      // j must leave onto a bound it already occupies, on the side its
      // reduced cost is feasible for: d_j > 0 at lower, d_j < 0 at upper.
      const Real target = dj > 0.0 ? matrix.lower(j) : matrix.upper(j);
      if (target == -INF || target == INF || std::abs(w[j] - target) > snap_tol(target)) {
        continue;  // not on that bound: moving it would move the point
      }
      ++st.dual_superbasic;

      // Tableau row r: rho = B^-T e_r, alpha_rk = rho . Ahat_k, built row-wise
      // from the nonzeros of rho only -- rho is usually sparse, and a
      // column-wise pass would touch all of A for every dual superbasic.
      std::fill(rho.begin(), rho.end(), 0.0);
      rho[r] = 1.0;
      lu.btran(core::HostSpan<Real>(rho.data(), rho.size()));
      for (const std::size_t k : touched) {
        row[k] = 0.0;
        seen[k] = 0;
      }
      touched.clear();
      const auto touch = [&](std::size_t k, Real a) {
        if (!seen[k]) {
          seen[k] = 1;
          touched.push_back(k);
        }
        row[k] += a;
      };
      for (std::size_t i = 0; i < m; ++i) {
        const Real ri = rho[i];
        if (ri == 0.0) continue;
        touch(n + i, ri);
        for (std::size_t e = csr.slice_begin(i); e < csr.slice_end(i); ++e) {
          touch(static_cast<std::size_t>(csr.indices()[e]), ri * csr.values()[e]);
        }
      }

      // tau >= 0 along sign(d_j): d_k(tau) = d_k - sign * tau * alpha_rk.
      const Real sign = dj > 0.0 ? 1.0 : -1.0;
      // Harris pass 1 with a relaxed zero, pass 2 the largest |alpha|.
      Real relaxed = std::abs(dj);
      for (const std::size_t k : touched) {
        const Real rate = -sign * row[k];
        const Real tau = blocks(k, rate);
        if (tau == INF) continue;
        relaxed = std::min(relaxed, tau + kHarrisTolerance / std::abs(rate));
      }
      std::size_t enter = total;
      Real enter_tau = INF;
      Real best = 0.0;
      for (const std::size_t k : touched) {
        const Real rate = -sign * row[k];
        const Real tau = blocks(k, rate);
        if (tau <= relaxed && std::abs(row[k]) > best) {
          best = std::abs(row[k]);
          enter = k;
          enter_tau = tau;
        }
      }

      const Real tau = enter == total ? std::abs(dj) : std::min(enter_tau, std::abs(dj));
      for (const std::size_t k : touched) d[k] -= sign * tau * row[k];
      if (enter == total || enter_tau >= std::abs(dj)) {
        d[j] = 0.0;  // nothing blocked: j stays basic, its reduced cost now zero
        continue;
      }

      // k enters in slot r, j leaves onto `target`. The primal step is zero
      // because x_j is already there, so only the factor changes.
      std::fill(column.begin(), column.end(), 0.0);
      matrix.for_each_in_column(enter,
                                [&column](std::size_t i, Real value) { column[i] = value; });
      lu.ftran(core::HostSpan<Real>(column.data(), column.size()));
      if (std::abs(column[r]) <= kPushPivotTolerance) continue;  // row and column disagree
      w[j] = target;
      basis.status[j] = matrix.lower(j) == matrix.upper(j) ? VarStatus::Fixed
                        : dj > 0.0                          ? VarStatus::AtLower
                                                            : VarStatus::AtUpper;
      basis.basic[r] = static_cast<core::Index>(enter);
      basis.status[enter] = VarStatus::Basic;
      d[enter] = 0.0;
      ++st.dual_pivots;
      const core::Status updated =
          lu.update(r, core::HostSpan<const Real>(column.data(), column.size()));
      if (!updated.ok() || ++since_refactor >= kRefactorInterval) {
        const core::Status factored = refactor();
        if (!factored.ok()) return factored.error();
      }
    }
  }

  st.dual_push_seconds = elapsed() - st.crash_seconds - st.primal_push_seconds;
  basis.dse_weights.clear();
  if (!basis.validate()) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "crossover: produced an invalid basis");
  }
  return basis;
}

}  // namespace sovsolve::solver
