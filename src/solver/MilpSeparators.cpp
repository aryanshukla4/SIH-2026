#include "sovsolve/solver/MilpSeparators.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

#include "sovsolve/core/SparseBuilder.hpp"
#include "sovsolve/solver/simplex/LuFactor.hpp"

namespace sovsolve::solver {

namespace {

using core::is_finite_bound;
using simplex::VarStatus;

constexpr Real kInf = std::numeric_limits<Real>::infinity();
constexpr Real kZero = 1e-9;       // a coefficient below this is zero
constexpr Real kBoundTol = 1e-6;   // "strictly between its bounds"

// [W] section 3.4, the fast version, and section 3.3.4.
constexpr std::size_t kMaxAggr = 5;
constexpr std::size_t kMaxFails = 150;
constexpr std::size_t kMaxConts = 20;
constexpr std::size_t kMaxCuts = 100;
constexpr std::size_t kMaxTestDelta = 10;
constexpr Real kMinFrac = 0.05;
constexpr Real kMaxFrac = 0.95;
constexpr Real kMaxAggrRatio = 1e4;

Real frac(Real d) { return d - std::floor(d); }

/// [W] Remark 3.2: F_f(d) = floor(d) + max(f_d - f, 0)/(1 - f).
Real mir_function(Real d, Real f) {
  const Real fd = frac(d);
  if (fd < kZero || fd > 1.0 - kZero) return std::round(d);
  return std::floor(d) + std::max(fd - f, 0.0) / (1.0 - f);
}

/// A cut is "violated" when its left side exceeds the right by more than
/// `margin` (MilpOptions::cut_violation_margin, OURS -- [CIP] and [W] compare
/// against zero), relative to max(1, |rhs|).
bool violated(Real lhs, Real rhs, Real margin) {
  return lhs - rhs > margin * std::max(1.0, std::fabs(rhs));
}

/// Integer bounds of an integer column in ORIGINAL units, or infinite.
Real int_lower(Real lb, Real s) {
  return is_finite_bound(lb) ? std::ceil(s * lb - kBoundTol) : -kInf;
}
Real int_upper(Real ub, Real s) {
  return is_finite_bound(ub) ? std::floor(s * ub + kBoundTol) : kInf;
}

/// Accumulates a cut in canonical structural space.
struct CutBuilder {
  std::vector<Real> coef;
  std::vector<std::size_t> nz;
  std::vector<char> mark;
  Real rhs = 0.0;

  explicit CutBuilder(std::size_t n) : coef(n, 0.0), mark(n, 0) {}
  void add(std::size_t j, Real a) {
    if (a == 0.0) return;
    if (mark[j] == 0) {
      mark[j] = 1;
      nz.push_back(j);
    }
    coef[j] += a;
  }
  /// `+ kappa * slack_i`, slack_i = b_i - A_i x: moves `kappa b_i` right.
  void add_slack(const model::CanonicalProblem& p, std::size_t i, Real kappa) {
    const auto& csr = p.A.csr;
    for (std::size_t q = csr.slice_begin(i); q < csr.slice_end(i); ++q) {
      add(static_cast<std::size_t>(csr.indices()[q]), -kappa * csr.values()[q]);
    }
    rhs -= kappa * p.b[i];
  }
  Cut finish() {
    Cut c;
    for (std::size_t j : nz) {
      if (std::fabs(coef[j]) > kZero) c.terms.emplace_back(j, coef[j]);
      coef[j] = 0.0;
      mark[j] = 0;
    }
    nz.clear();
    c.rhs = rhs;
    rhs = 0.0;
    std::sort(c.terms.begin(), c.terms.end());
    return c;
  }
};

Real activity(const Cut& c, const std::vector<Real>& x) {
  Real a = 0.0;
  for (const auto& [j, v] : c.terms) a += v * x[j];
  return a;
}

/// [W] section 6.1's GMI safeguard: find p/q with q <= 1000 for every
/// integer-variable coefficient (original units), multiply the cut by
/// lcm(q)/gcd(scaled coefficients), reject if that scalar exceeds 1000.
bool scale_integral(Cut& cut, const std::vector<Real>& scale) {
  auto approximate = [](Real v, std::int64_t& p, std::int64_t& q) {
    // Continued fractions, denominators up to 1000.
    const Real target = v;
    std::int64_t h0 = 0, h1 = 1, k0 = 1, k1 = 0;
    Real x = v;
    for (int it = 0; it < 64; ++it) {
      const Real a = std::floor(x);
      if (std::fabs(a) > 1e15) return false;
      const auto ai = static_cast<std::int64_t>(a);
      const std::int64_t h2 = ai * h1 + h0;
      const std::int64_t k2 = ai * k1 + k0;
      if (k2 > 1000) return false;
      h0 = h1; h1 = h2; k0 = k1; k1 = k2;
      if (std::fabs(static_cast<Real>(h1) / static_cast<Real>(k1) - target) <=
          1e-9 * std::max(1.0, std::fabs(target))) {
        p = h1;
        q = k1;
        return true;
      }
      const Real rest = x - a;
      if (rest < 1e-15) return false;
      x = 1.0 / rest;
    }
    return false;
  };
  std::int64_t lcm = 1;
  std::vector<std::pair<std::int64_t, std::int64_t>> fractions;
  for (const auto& [j, v] : cut.terms) {
    if (scale[j] == 0.0) continue;
    std::int64_t p = 0, q = 1;
    if (!approximate(v / scale[j], p, q)) return false;
    fractions.emplace_back(p, q);
    lcm = std::lcm(lcm, q);
    if (lcm > 1000) return false;
  }
  if (fractions.empty()) return true;
  std::int64_t g = 0;
  for (const auto& [p, q] : fractions) g = std::gcd(g, std::abs(p * (lcm / q)));
  if (g == 0) return true;
  const Real scalar = static_cast<Real>(lcm) / static_cast<Real>(g);
  if (scalar > 1000.0) return false;
  for (auto& term : cut.terms) term.second *= scalar;
  cut.rhs *= scalar;
  return true;
}

}  // namespace

// ==========================================================================
// Gomory mixed integer cuts
// ==========================================================================

std::vector<Cut> separate_gomory(const SeparationInput& in, const simplex::SimplexResult& lp) {
  std::vector<Cut> out;
  const model::CanonicalProblem& p = *in.problem;
  const std::size_t n = p.num_cols();
  const std::size_t m = p.num_rows();
  const auto& scale = *in.integer_scale;
  const auto& lower = *in.lower;
  const auto& upper = *in.upper;
  if (m == 0) return out;

  // The optimal basis, factorized afresh. A basis that needs repair is not
  // the one the LP point came from; no tableau rows are read from it.
  simplex::Basis basis = lp.basis;
  const simplex::AugmentedMatrix matrix(p);
  simplex::LuFactorization lu;
  std::size_t repairs = 0;
  if (!lu.factorize_repairing(matrix, basis, 0.1, &repairs).ok() || repairs != 0) return out;

  const auto& csc = p.A.csc;
  std::vector<Real> rho(m);
  CutBuilder builder(n);
  for (std::size_t r = 0; r < m; ++r) {
    const auto bi = static_cast<std::size_t>(basis.basic[r]);
    if (bi >= n || scale[bi] == 0.0) continue;  // basic integer structurals only
    const Real si = scale[bi];
    const Real vi = si * lp.x[bi];
    const Real fi = frac(vi);
    if (fi < kMinFrac || fi > 1.0 - kZero) continue;  // [W] 6.1: "except f < 0.05"

    std::fill(rho.begin(), rho.end(), 0.0);
    rho[r] = 1.0;
    lu.btran(core::HostSpan<Real>(rho.data(), rho.size()));

    // The tableau row in v_i units, every nonbasic shifted onto the bound it
    // rests on (w = bound + w' or bound - w'), so the right side is v_i*:
    //     v_i + sum_int c_j v'_j + sum_cont g_j y'_j = v_i*.
    struct Term {
      std::size_t w;
      Real c;
      bool integer;
      bool at_upper;
    };
    std::vector<Term> terms;
    bool usable = true;
    for (std::size_t w = 0; w < n + m && usable; ++w) {
      const VarStatus st = basis.status[w];
      if (st == VarStatus::Basic || st == VarStatus::Fixed) continue;
      Real a = 0.0;
      if (w < n) {
        for (std::size_t q = csc.slice_begin(w); q < csc.slice_end(w); ++q) {
          a += rho[static_cast<std::size_t>(csc.indices()[q])] * csc.values()[q];
        }
      } else {
        a = rho[w - n];
      }
      a *= si;
      if (std::fabs(a) <= kZero) continue;
      if (st == VarStatus::Free) {
        usable = false;  // no bound to substitute on
        break;
      }
      const bool at_upper = st == VarStatus::AtUpper;
      const Real c = at_upper ? -a : a;
      bool integer = false;
      Real coef = c;
      if (w < n && scale[w] != 0.0) {
        const Real bound = at_upper ? scale[w] * upper[w] : scale[w] * lower[w];
        if (is_finite_bound(at_upper ? upper[w] : lower[w]) &&
            std::fabs(bound - std::round(bound)) <= kBoundTol) {
          integer = true;
          coef = c / scale[w];  // per original unit
        }
      }
      terms.push_back(Term{w, coef, integer, at_upper});
    }
    if (!usable) continue;

    // v_i itself: shifted by its integral lower bound when it has one; with
    // coefficient 1, F(1) = 1, so the shift changes nothing but the right side.
    const Real Li = int_lower(lower[bi], si);
    const Real beta = std::isfinite(Li) ? vi - Li : vi;
    const Real f0 = frac(beta);
    if (f0 < kMinFrac || f0 > 1.0 - kZero) continue;

    // MIR of [CIP] Proposition 8.3 / (8.5) with delta = 1:
    //   v'_i + sum_int F(c_j) v'_j + sum_{g_j<0} g_j/(1-f0) y'_j <= floor(beta)
    // then every shifted variable back to canonical x.
    builder.rhs = std::floor(beta);
    if (std::isfinite(Li)) {
      builder.add(bi, si);
      builder.rhs += Li;
    } else {
      builder.add(bi, si);
    }
    for (const Term& t : terms) {
      const Real k = t.integer ? mir_function(t.c, f0) : std::min(t.c, 0.0) / (1.0 - f0);
      if (k == 0.0) continue;
      if (t.w < n) {
        // integer: v' = s(x - lb) or s(ub - x); continuous: y' = x - lb or ub - x
        const Real unit = t.integer ? scale[t.w] : 1.0;
        if (t.at_upper) {
          builder.add(t.w, -k * unit);
          builder.rhs -= k * unit * upper[t.w];
        } else {
          builder.add(t.w, k * unit);
          builder.rhs += k * unit * lower[t.w];
        }
      } else {
        builder.add_slack(p, t.w - n, k);  // the logical IS the row's slack
      }
    }
    Cut cut = builder.finish();
    if (cut.terms.empty()) continue;
    if (!scale_integral(cut, scale)) continue;
    if (violated(activity(cut, lp.x), cut.rhs, in.violation_margin)) out.push_back(std::move(cut));
  }
  return out;
}

// ==========================================================================
// Complemented MIR cuts -- [W] Algorithms 3.1-3.4
// ==========================================================================

namespace {

class CmirSeparator {
 public:
  CmirSeparator(const SeparationInput& in, const simplex::SimplexResult& lp, CmirState& state)
      : p_(*in.problem),
        scale_(*in.integer_scale),
        lower_(*in.lower),
        upper_(*in.upper),
        x_(lp.x),
        y_(lp.y),
        margin_(in.violation_margin),
        state_(state),
        n_(p_.num_cols()),
        m_(p_.num_rows()),
        agg_(n_, 0.0),
        in_agg_(n_, 0),
        builder_(n_) {
    if (state_.aggregations.size() != m_) state_.aggregations.assign(m_, 0);
    slack_.assign(m_, 0.0);
    const auto& csr = p_.A.csr;
    for (std::size_t i = 0; i < m_; ++i) {
      Real act = 0.0;
      for (std::size_t q = csr.slice_begin(i); q < csr.slice_end(i); ++q) {
        act += csr.values()[q] * x_[static_cast<std::size_t>(csr.indices()[q])];
      }
      slack_[i] = i < p_.num_equality ? 0.0 : std::max(p_.b[i] - act, 0.0);
    }
    for (std::size_t j = 0; j < n_; ++j) c_norm_ += p_.c[j] * p_.c[j];
    c_norm_ = std::sqrt(c_norm_);
  }

  std::vector<Cut> run(std::size_t round) {
    std::vector<Cut> cuts;
    // Starting constraints by non-increasing CONSSCORE ([W] section 3.4).
    std::vector<std::pair<Real, std::size_t>> order;
    for (std::size_t i = 0; i < m_; ++i) {
      if (continuous_between(i) > kMaxConts) continue;  // MAXCONTS
      order.emplace_back(-score(i), i);
    }
    std::sort(order.begin(), order.end());
    // MAXFAILS, doubled in early rounds: MAXFAILS + (MAXFAILS - 2k)+.
    const std::size_t bonus = 2 * round < kMaxFails ? kMaxFails - 2 * round : 0;
    const std::size_t max_fails = kMaxFails + bonus;
    std::size_t fails = 0;
    for (const auto& [neg, start] : order) {
      (void)neg;
      if (cuts.size() >= kMaxCuts || fails >= max_fails) break;
      if (separate_from(start, cuts)) {
        fails = 0;
      } else {
        ++fails;
      }
    }
    return cuts;
  }

 private:
  /// [W] Score Type 3 / CONSSCORE:
  ///   0.9^l ( max{min{|y_i|/||c||, 1}, 1e-4} + 1e-4 (1 - dens_i)
  ///           + 1e-3 (1 - max{slack_i/||a_i||, 0.1}) ).
  /// READING: the extracted text shows "max{ max{ db_i/||(c,d)||, 1.0},
  /// 0.0001}", in which the outer 1e-4 floor could never bind; the inner is
  /// read as a min, the only reading in which every constant matters.
  Real score(std::size_t i) const {
    const auto& csr = p_.A.csr;
    const std::size_t nnz = csr.slice_end(i) - csr.slice_begin(i);
    Real row_norm = 0.0;
    for (std::size_t q = csr.slice_begin(i); q < csr.slice_end(i); ++q) {
      row_norm += csr.values()[q] * csr.values()[q];
    }
    row_norm = std::sqrt(row_norm);
    const Real dual =
        c_norm_ > 0.0 ? std::max(std::min(std::fabs(y_[i]) / c_norm_, 1.0), 1e-4) : 1e-4;
    const Real dens = n_ > 0 ? static_cast<Real>(nnz) / static_cast<Real>(n_) : 0.0;
    const Real tight = row_norm > 0.0 ? std::max(slack_[i] / row_norm, 0.1) : 0.1;
    return std::pow(0.9, static_cast<Real>(state_.aggregations[i])) *
           (dual + 1e-4 * (1.0 - dens) + 1e-3 * (1.0 - tight));
  }

  bool strictly_between(std::size_t j) const {
    return x_[j] - lower_[j] > kBoundTol && upper_[j] - x_[j] > kBoundTol;
  }

  std::size_t continuous_between(std::size_t i) const {
    const auto& csr = p_.A.csr;
    std::size_t count = i >= p_.num_equality && slack_[i] > kBoundTol ? 1 : 0;
    for (std::size_t q = csr.slice_begin(i); q < csr.slice_end(i); ++q) {
      const auto j = static_cast<std::size_t>(csr.indices()[q]);
      if (scale_[j] == 0.0 && strictly_between(j)) ++count;
    }
    return count;
  }

  void add_row(std::size_t r, Real lambda) {
    const auto& csr = p_.A.csr;
    for (std::size_t q = csr.slice_begin(r); q < csr.slice_end(r); ++q) {
      const auto j = static_cast<std::size_t>(csr.indices()[q]);
      if (in_agg_[j] == 0) {
        in_agg_[j] = 1;
        agg_nz_.push_back(j);
      }
      agg_[j] += lambda * csr.values()[q];
    }
    if (r >= p_.num_equality) slacks_.emplace_back(r, lambda);
    beta_ += lambda * p_.b[r];
    rows_.push_back(r);
    factors_.push_back(lambda);
  }

  void clear() {
    for (std::size_t j : agg_nz_) {
      agg_[j] = 0.0;
      in_agg_[j] = 0;
    }
    agg_nz_.clear();
    slacks_.clear();
    rows_.clear();
    factors_.clear();
    beta_ = 0.0;
  }

  /// [W] Algorithm 3.1 for one starting constraint.
  bool separate_from(std::size_t start, std::vector<Cut>& cuts) {
    clear();
    add_row(start, 1.0);
    ++state_.aggregations[start];
    bool found = false;
    for (;;) {
      Cut cut;
      if (knapsack_cut(cut)) {
        cuts.push_back(std::move(cut));
        found = true;
        break;
      }
      if (rows_.size() > kMaxAggr) break;  // MAXAGGR rows added to the start
      if (!aggregate()) break;
    }
    clear();
    return found;
  }

  /// [W] Algorithm 3.2 with Score Type 3: eliminate the continuous column
  /// with the greatest bound distance, using the best-scoring row not yet in
  /// the aggregation, subject to section 3.3.4's factor ratio and MAXCONTS.
  bool aggregate() {
    const auto& csc = p_.A.csc;
    std::vector<std::pair<Real, std::size_t>> candidates;
    for (std::size_t j : agg_nz_) {
      if (scale_[j] != 0.0 || std::fabs(agg_[j]) <= kZero) continue;
      const Real d = std::min(x_[j] - lower_[j], upper_[j] - x_[j]);  // infinite side -> inf
      if (!(d > 0.0)) continue;
      candidates.emplace_back(-(is_finite_bound(d) ? d : kInf), j);
    }
    std::sort(candidates.begin(), candidates.end());
    Real min_factor = kInf, max_factor = 0.0;
    for (Real f : factors_) {
      min_factor = std::min(min_factor, std::fabs(f));
      max_factor = std::max(max_factor, std::fabs(f));
    }
    for (const auto& [negd, k] : candidates) {
      (void)negd;
      std::size_t best_row = m_;
      Real best_score = -kInf;
      Real best_lambda = 0.0;
      for (std::size_t q = csc.slice_begin(k); q < csc.slice_end(k); ++q) {
        const auto r = static_cast<std::size_t>(csc.indices()[q]);
        if (std::find(rows_.begin(), rows_.end(), r) != rows_.end()) continue;
        const Real a = csc.values()[q];
        if (std::fabs(a) <= kZero) continue;
        const Real lambda = -agg_[k] / a;
        if (max_factor / std::fabs(lambda) > kMaxAggrRatio ||
            std::fabs(lambda) / min_factor > kMaxAggrRatio) {
          continue;
        }
        const Real s = score(r);
        if (s > best_score) {
          best_score = s;
          best_row = r;
          best_lambda = lambda;
        }
      }
      if (best_row == m_) continue;
      add_row(best_row, best_lambda);
      agg_[k] = 0.0;  // eliminated exactly
      ++state_.aggregations[best_row];
      // MAXCONTS on the new aggregated constraint.
      std::size_t conts = 0;
      for (std::size_t j : agg_nz_) {
        if (scale_[j] == 0.0 && std::fabs(agg_[j]) > kZero && strictly_between(j)) ++conts;
      }
      for (const auto& [row, lam] : slacks_) {
        if (std::fabs(lam) > kZero && slack_[row] > kBoundTol) ++conts;
      }
      return conts <= kMaxConts;
    }
    return false;
  }

  /// [W] Algorithms 3.3 and 3.4 on the current aggregated constraint
  ///     sum_j agg_j x_j + sum_{i in Q} lambda_i s_i = beta.
  bool knapsack_cut(Cut& cut) {
    // ---- Algorithm 3.3: bound substitution, Criteria F3 (only simple bounds
    // exist here) and S3 (the closer bound). ----
    struct Cont {
      std::size_t j;       // canonical column, or m_+... for a slack (below)
      bool slack;
      bool upper;          // ybar = ub - x   (else x - lb, or the slack itself)
      Real coef;           // coefficient of ybar in the equation
    };
    struct Int {
      std::size_t j;
      Real alpha;          // per ORIGINAL unit
      Real L, U, v;        // integral bounds and LP value, original units
    };
    std::vector<Cont> conts;
    std::vector<Int> ints;
    Real beta = beta_;
    for (std::size_t j : agg_nz_) {
      const Real a = agg_[j];
      if (std::fabs(a) <= kZero) continue;
      if (scale_[j] != 0.0) {
        const Real s = scale_[j];
        const Real L = int_lower(lower_[j], s);
        const Real U = int_upper(upper_[j], s);
        if (!std::isfinite(L) && !std::isfinite(U)) return false;
        ints.push_back(Int{j, a / s, L, U, s * x_[j]});
        continue;
      }
      const bool has_l = is_finite_bound(lower_[j]);
      const bool has_u = is_finite_bound(upper_[j]);
      bool use_upper = false;
      if (!has_l && !has_u) return false;
      if (!has_u) {
        use_upper = false;
      } else if (!has_l) {
        use_upper = true;
      } else {
        use_upper = upper_[j] - x_[j] < x_[j] - lower_[j];
      }
      if (use_upper) {
        beta -= a * upper_[j];
        conts.push_back(Cont{j, false, true, -a});
      } else {
        beta -= a * lower_[j];
        conts.push_back(Cont{j, false, false, a});
      }
    }
    for (const auto& [row, lam] : slacks_) {
      if (std::fabs(lam) > kZero) conts.push_back(Cont{row, true, false, lam});
    }
    if (ints.empty()) return false;
    // Relax to the mixed knapsack sum alpha v <= beta + s, s = -sum_{coef<0} coef ybar.
    Real s_star = 0.0;
    for (const Cont& c : conts) {
      if (c.coef >= 0.0) continue;
      const Real ybar = c.slack ? slack_[c.j]
                                : (c.upper ? upper_[c.j] - x_[c.j] : x_[c.j] - lower_[c.j]);
      s_star -= c.coef * ybar;
    }

    // ---- Algorithm 3.4, Procedure 1 with the extended delta candidates. ----
    const std::size_t k = ints.size();
    std::vector<char> in_u(k, 0);
    auto between = [&](const Int& t) { return t.v - t.L > kBoundTol && t.U - t.v > kBoundTol; };
    for (std::size_t q = 0; q < k; ++q) {
      const Int& t = ints[q];
      if (!std::isfinite(t.L)) {
        in_u[q] = 1;
      } else if (std::isfinite(t.U) && t.v - t.L >= 0.5 * (t.U - t.L)) {
        in_u[q] = 1;
      }
    }
    std::vector<Real> deltas;
    Real max_alpha = 0.0;
    for (const Int& t : ints) {
      max_alpha = std::max(max_alpha, std::fabs(t.alpha));
      if (!between(t) || std::fabs(t.alpha) <= kZero) continue;
      const Real d = std::fabs(t.alpha);
      if (deltas.size() < kMaxTestDelta &&
          std::none_of(deltas.begin(), deltas.end(),
                       [d](Real e) { return std::fabs(e - d) <= kZero * std::max(1.0, d); })) {
        deltas.push_back(d);
      }
    }
    deltas.push_back(1.0 + max_alpha);  // the extended candidate, not counted

    // Violation of the c-MIR for (T, U, delta), in delta-scaled units.
    auto evaluate = [&](Real delta, Real& f0_out) {
      Real bt = beta;
      for (std::size_t q = 0; q < k; ++q) {
        bt -= ints[q].alpha * (in_u[q] != 0 ? ints[q].U : ints[q].L);
      }
      const Real b = bt / delta;
      const Real f0 = frac(b);
      f0_out = f0;
      if (f0 < kMinFrac || f0 > kMaxFrac) return -kInf;  // [W] MINFRAC / MAXFRAC
      Real lhs = 0.0;
      for (std::size_t q = 0; q < k; ++q) {
        const Int& t = ints[q];
        lhs += in_u[q] != 0 ? mir_function(-t.alpha / delta, f0) * (t.U - t.v)
                            : mir_function(t.alpha / delta, f0) * (t.v - t.L);
      }
      return lhs - std::floor(b) - s_star / (delta * (1.0 - f0));
    };

    // Integer columns in T strictly between bounds, by non-increasing
    // x_j - b_j/2 (closest to the upper half first).
    auto t_order = [&]() {
      std::vector<std::pair<Real, std::size_t>> o;
      for (std::size_t q = 0; q < k; ++q) {
        if (in_u[q] != 0 || !between(ints[q]) || !std::isfinite(ints[q].U)) continue;
        o.emplace_back(-((ints[q].v - ints[q].L) - 0.5 * (ints[q].U - ints[q].L)), q);
      }
      std::sort(o.begin(), o.end());
      return o;
    };

    Real best_delta = 0.0;
    Real best_v = -kInf;
    Real f0 = 0.0;
    {
      const auto order = t_order();
      for (std::size_t l = 0; l <= order.size(); ++l) {
        if (l > 0) in_u[order[l - 1].second] = 1;  // complement an additional one
        bool any = false;
        for (Real d : deltas) {
          const Real v = evaluate(d, f0);
          if (v == -kInf) continue;
          any = true;
          if (v > best_v) {
            best_v = v;
            best_delta = d;
          }
        }
        if (any) break;
      }
    }
    if (best_v == -kInf) return false;
    for (Real div : {2.0, 4.0, 8.0}) {
      const Real d = best_delta / div;
      const Real v = evaluate(d, f0);
      if (v > best_v) {
        best_v = v;
        best_delta = d;
      }
    }
    {
      const auto order = t_order();
      for (const auto& [neg, q] : order) {
        (void)neg;
        if (ints[q].v - ints[q].L <= 0.0) continue;
        in_u[q] = 1;
        const Real v = evaluate(best_delta, f0);
        if (v > best_v) {
          best_v = v;
        } else {
          in_u[q] = 0;
        }
      }
    }
    const Real v_final = evaluate(best_delta, f0);
    if (v_final == -kInf || v_final <= 0.0) return false;

    // ---- Back to canonical x. ----
    //   sum_T F(a/d)(v - L) + sum_U F(-a/d)(U - v) - s/(d(1-f0)) <= floor(bt/d)
    Real bt = beta;
    for (std::size_t q = 0; q < k; ++q) bt -= ints[q].alpha * (in_u[q] != 0 ? ints[q].U : ints[q].L);
    builder_.rhs = std::floor(bt / best_delta);
    for (std::size_t q = 0; q < k; ++q) {
      const Int& t = ints[q];
      const Real s = scale_[t.j];
      if (in_u[q] != 0) {
        const Real g = mir_function(-t.alpha / best_delta, f0);
        builder_.add(t.j, -g * s);
        builder_.rhs -= g * t.U;
      } else {
        const Real g = mir_function(t.alpha / best_delta, f0);
        builder_.add(t.j, g * s);
        builder_.rhs += g * t.L;
      }
    }
    for (const Cont& c : conts) {
      if (c.coef >= 0.0) continue;
      const Real kappa = c.coef / (best_delta * (1.0 - f0));  // coefficient of ybar
      if (c.slack) {
        builder_.add_slack(p_, c.j, kappa);
      } else if (c.upper) {
        builder_.add(c.j, -kappa);
        builder_.rhs -= kappa * upper_[c.j];
      } else {
        builder_.add(c.j, kappa);
        builder_.rhs += kappa * lower_[c.j];
      }
    }
    cut = builder_.finish();
    if (cut.terms.empty()) return false;
    return violated(activity(cut, x_), cut.rhs, margin_);
  }

  const model::CanonicalProblem& p_;
  const std::vector<Real>& scale_;
  const std::vector<Real>& lower_;
  const std::vector<Real>& upper_;
  const std::vector<Real>& x_;
  const std::vector<Real>& y_;
  Real margin_;
  CmirState& state_;
  std::size_t n_;
  std::size_t m_;
  std::vector<Real> slack_;
  Real c_norm_ = 0.0;

  std::vector<Real> agg_;
  std::vector<char> in_agg_;
  std::vector<std::size_t> agg_nz_;
  std::vector<std::pair<std::size_t, Real>> slacks_;
  std::vector<std::size_t> rows_;
  std::vector<Real> factors_;
  Real beta_ = 0.0;
  CutBuilder builder_;
};

}  // namespace

std::vector<Cut> separate_cmir(const SeparationInput& in, const simplex::SimplexResult& lp,
                               std::size_t round, CmirState& state) {
  CmirSeparator sep(in, lp, state);
  return sep.run(round);
}

// ==========================================================================
// Selection -- [CIP] Algorithm 3.2
// ==========================================================================

std::vector<Cut> select_cuts(std::vector<Cut> cuts, const model::CanonicalProblem& p,
                             const std::vector<Real>& x, std::size_t max_cuts) {
  constexpr Real kWe = 1.0, kWp = 0.1, kWo = 1.0, kMinOrtho = 0.5;
  const std::size_t n = p.num_cols();
  Real c_norm = 0.0;
  for (std::size_t j = 0; j < n; ++j) c_norm += p.c[j] * p.c[j];
  c_norm = std::sqrt(c_norm);

  struct Entry {
    Real efficacy, parallelism, ortho, score, norm;
    bool alive;
  };
  std::vector<Entry> e(cuts.size());
  for (std::size_t r = 0; r < cuts.size(); ++r) {
    Real norm = 0.0, dot_c = 0.0;
    for (const auto& [j, v] : cuts[r].terms) {
      norm += v * v;
      dot_c += v * p.c[j];
    }
    norm = std::sqrt(norm);
    const Real eff = norm > 0.0 ? (activity(cuts[r], x) - cuts[r].rhs) / norm : 0.0;
    const Real par = norm > 0.0 && c_norm > 0.0 ? std::fabs(dot_c) / (norm * c_norm) : 0.0;
    e[r] = Entry{eff, par, 1.0, kWe * eff + kWp * par + kWo * 1.0, norm, norm > 0.0};
  }
  auto dot = [](const Cut& a, const Cut& b) {  // both sorted by column
    Real s = 0.0;
    std::size_t i = 0, j = 0;
    while (i < a.terms.size() && j < b.terms.size()) {
      if (a.terms[i].first < b.terms[j].first) {
        ++i;
      } else if (a.terms[i].first > b.terms[j].first) {
        ++j;
      } else {
        s += a.terms[i].second * b.terms[j].second;
        ++i;
        ++j;
      }
    }
    return s;
  };
  std::vector<Cut> chosen;
  while (chosen.size() < max_cuts) {
    std::size_t best = cuts.size();
    for (std::size_t r = 0; r < cuts.size(); ++r) {
      if (e[r].alive && (best == cuts.size() || e[r].score > e[best].score)) best = r;
    }
    if (best == cuts.size()) break;
    e[best].alive = false;
    for (std::size_t r = 0; r < cuts.size(); ++r) {
      if (!e[r].alive) continue;
      const Real o = 1.0 - std::fabs(dot(cuts[best], cuts[r])) / (e[best].norm * e[r].norm);
      e[r].ortho = std::min(e[r].ortho, o);
      if (e[r].ortho < kMinOrtho) {
        e[r].alive = false;
      } else {
        e[r].score = kWe * e[r].efficacy + kWp * e[r].parallelism + kWo * e[r].ortho;
      }
    }
    chosen.push_back(std::move(cuts[best]));
  }
  return chosen;
}

// ==========================================================================
// Appending rows
// ==========================================================================

core::Status append_cuts(model::CanonicalProblem& p, const std::vector<Cut>& cuts,
                         simplex::Basis* basis) {
  if (cuts.empty()) return core::Status::Ok();
  const std::size_t n = p.num_cols();
  const std::size_t m = p.num_rows();
  const std::size_t rows = m + cuts.size();
  const auto& csr = p.A.csr;
  auto enumerate = [&](auto&& emit) {
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t q = csr.slice_begin(i); q < csr.slice_end(i); ++q) {
        emit(static_cast<core::Index>(i), csr.indices()[q], csr.values()[q]);
      }
    }
    for (std::size_t c = 0; c < cuts.size(); ++c) {
      for (const auto& [j, v] : cuts[c].terms) {
        emit(static_cast<core::Index>(m + c), static_cast<core::Index>(j), v);
      }
    }
  };
  core::SparseBuilder builder(rows, n);
  enumerate([&](core::Index r, core::Index c, Real) { builder.count(r, c); });
  if (auto st = builder.allocate(); !st.ok()) return st;
  enumerate([&](core::Index r, core::Index c, Real v) { builder.insert(r, c, v); });
  p.A = builder.finish(true, 0.0);
  core::RealVector b(rows);
  for (std::size_t i = 0; i < m; ++i) b[i] = p.b[i];
  for (std::size_t c = 0; c < cuts.size(); ++c) b[m + c] = cuts[c].rhs;
  p.b = std::move(b);

  if (basis != nullptr) {
    // Logicals are w = n + row, so the new rows' logicals come at the end.
    for (std::size_t c = 0; c < cuts.size(); ++c) {
      basis->status.push_back(simplex::VarStatus::Basic);
      basis->basic.push_back(static_cast<core::Index>(n + m + c));
      if (!basis->dse_weights.empty()) basis->dse_weights.push_back(1.0);
    }
  }
  return core::Status::Ok();
}

}  // namespace sovsolve::solver
