#include "sovsolve/solver/MilpCanonicalPresolve.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "sovsolve/core/SparseBuilder.hpp"

namespace sovsolve::solver {

namespace {

using core::is_finite_bound;
using core::Real;

constexpr Real kTol = 1e-6;  // [CIP] (7.2)'s feasibility tolerance epsilon-hat

/// [CIP] (7.3): accept a bound change only if it removes at least 5% of the
/// domain (or of the bound's magnitude, at least 1), or makes it finite.
Real min_change(Real lo, Real hi, Real bound) {
  return 0.05 * std::max(std::min(hi - lo, std::fabs(bound)), 1.0);
}

struct Row {
  std::vector<std::pair<std::size_t, Real>> entries;
  Real b = 0.0;
  bool equality = false;
  bool alive = true;
};

struct Activity {
  Real min = 0.0, max = 0.0;
  std::size_t min_inf = 0, max_inf = 0;
};

}  // namespace

core::Status presolve_canonical(model::CanonicalProblem& p,
                                const std::vector<Real>& integer_scale, std::size_t max_rounds,
                                CanonicalPresolveStats& stats) {
  const std::size_t n = p.num_cols();
  const std::size_t m = p.num_rows();
  const auto infeasible = [](const char* why) {
    return core::make_error(core::ErrorCode::PrimalInfeasible, why);
  };

  std::vector<Real> lo(p.col_lower.data(), p.col_lower.data() + n);
  std::vector<Real> hi(p.col_upper.data(), p.col_upper.data() + n);
  // Integer bounds are integral in original units.
  for (std::size_t j = 0; j < n; ++j) {
    const Real s = integer_scale[j];
    if (s == 0.0) continue;
    if (is_finite_bound(lo[j])) lo[j] = std::ceil(s * lo[j] - kTol) / s;
    if (is_finite_bound(hi[j])) hi[j] = std::floor(s * hi[j] + kTol) / s;
    if (lo[j] > hi[j] + kTol) return infeasible("presolve: an integer column has no integer value");
  }

  std::vector<Row> rows(m);
  const auto& csr = p.A.csr;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t q = csr.slice_begin(i); q < csr.slice_end(i); ++q) {
      rows[i].entries.emplace_back(static_cast<std::size_t>(csr.indices()[q]), csr.values()[q]);
    }
    rows[i].b = p.b[i];
    rows[i].equality = i < p.num_equality;
  }

  auto activity = [&](const Row& r) {
    Activity a;
    for (const auto& [j, v] : r.entries) {
      const Real l = v > 0.0 ? lo[j] : hi[j];
      const Real h = v > 0.0 ? hi[j] : lo[j];
      if (is_finite_bound(l)) a.min += v * l; else ++a.min_inf;
      if (is_finite_bound(h)) a.max += v * h; else ++a.max_inf;
    }
    return a;
  };

  bool any_change = true;
  for (std::size_t round = 0; round < max_rounds && any_change; ++round) {
    any_change = false;
    ++stats.rounds;

    for (Row& row : rows) {
      if (!row.alive) continue;
      // [CIP] 10.1 step 1g: at most 10 passes over one row while it changes.
      for (int pass = 0; pass < 10; ++pass) {
        bool row_changed = false;
        const Real rho = row.b;
        const Real lambda = row.equality ? row.b : -1e300;
        const bool has_lambda = row.equality;
        Activity a = activity(row);

        // Step 1d: infeasibility, then redundancy.
        if (a.min_inf == 0 && a.min > rho + kTol * (1.0 + std::fabs(rho))) {
          return infeasible("presolve: a row's minimum activity exceeds its right-hand side");
        }
        if (has_lambda && a.max_inf == 0 && a.max < lambda - kTol * (1.0 + std::fabs(lambda))) {
          return infeasible("presolve: an equality's maximum activity is below its right-hand side");
        }
        if (!row.equality && a.max_inf == 0 && a.max <= rho + kTol * (1.0 + std::fabs(rho))) {
          row.alive = false;  // can never be violated
          ++stats.rows_removed;
          any_change = true;
          break;
        }

        // Step 1c: bound tightening, Algorithm 7.1.
        for (const auto& [j, v] : row.entries) {
          const Real lpart = v > 0.0 ? lo[j] : hi[j];
          const Real hpart = v > 0.0 ? hi[j] : lo[j];
          Real alpha = -1e300, beta = 1e300;
          bool alpha_ok = false, beta_ok = false;
          if (is_finite_bound(lpart)) {
            if (a.min_inf == 0) { alpha = a.min - v * lpart; alpha_ok = true; }
          } else if (a.min_inf == 1) {
            alpha = a.min;
            alpha_ok = true;
          }
          if (is_finite_bound(hpart)) {
            if (a.max_inf == 0) { beta = a.max - v * hpart; beta_ok = true; }
          } else if (a.max_inf == 1) {
            beta = a.max;
            beta_ok = true;
          }
          Real new_lo = -1e300, new_hi = 1e300;
          bool lo_set = false, hi_set = false;
          if (alpha_ok) {
            const Real bnd = (rho - alpha) / v;
            if (v > 0.0) { new_hi = bnd; hi_set = true; } else { new_lo = bnd; lo_set = true; }
          }
          if (has_lambda && beta_ok) {
            const Real bnd = (lambda - beta) / v;
            if (v > 0.0) { new_lo = bnd; lo_set = true; } else { new_hi = bnd; hi_set = true; }
          }
          const Real s = integer_scale[j];
          if (hi_set && is_finite_bound(new_hi)) {
            new_hi = 1e-5 * std::ceil(1e5 * new_hi - kTol);
            if (s != 0.0) new_hi = std::floor(s * new_hi + kTol) / s;
            if (new_hi < hi[j] &&
                (!is_finite_bound(hi[j]) || new_hi < hi[j] - min_change(lo[j], hi[j], hi[j]))) {
              hi[j] = new_hi;
              ++stats.bounds_tightened;
              row_changed = true;
            }
          }
          if (lo_set && is_finite_bound(new_lo)) {
            new_lo = 1e-5 * std::floor(1e5 * new_lo + kTol);
            if (s != 0.0) new_lo = std::ceil(s * new_lo - kTol) / s;
            if (new_lo > lo[j] &&
                (!is_finite_bound(lo[j]) || new_lo > lo[j] + min_change(lo[j], hi[j], lo[j]))) {
              lo[j] = new_lo;
              ++stats.bounds_tightened;
              row_changed = true;
            }
          }
          if (lo[j] > hi[j] + kTol * (1.0 + std::fabs(hi[j]))) {
            return infeasible("presolve: a column's bounds crossed");
          }
          if (lo[j] > hi[j]) lo[j] = hi[j];
          // No early exit: activities made stale by a tightening earlier in
          // this scan are looser, so later deductions from them stay valid.
        }
        if (row_changed) {
          any_change = true;
          continue;
        }

        // Step 1f: coefficient tightening, inequality rows (lambda = -inf)
        // and integer columns with finite integral bounds. In original units
        // v = s x, a column steps by 1 and its coefficient is a/s; beta is the
        // maximum activity. Condition "the row is redundant whenever x_j is
        // off its bound": beta - a_v <= rho for a_v > 0 (off the upper
        // bound), beta + a_v <= rho for a_v < 0 (off the lower bound). Then
        //   a_v > 0:  a'_v = beta - rho,  rho -= (a_v - a'_v) U
        //   a_v < 0:  a'_v = rho - beta,  rho -= (a_v - a'_v) L
        // -- [CIP] step 1f with lambda = -inf, whose other side then drops.
        if (!row.equality && a.max_inf == 0) {
          for (auto& [j, v] : row.entries) {
            const Real s = integer_scale[j];
            if (s == 0.0 || !is_finite_bound(lo[j]) || !is_finite_bound(hi[j])) continue;
            const Real av = v / s;
            const Real L = s * lo[j];
            const Real U = s * hi[j];
            const Real tol = kTol * (1.0 + std::fabs(rho));
            if (av > 0.0 && a.max - av <= rho + tol) {
              const Real anew = a.max - rho;
              if (anew > kTol && anew < av - kTol) {
                row.b -= (av - anew) * U;
                v = anew * s;
                ++stats.coefficients_tightened;
                row_changed = true;
                break;
              }
            } else if (av < 0.0 && a.max + av <= rho + tol) {
              const Real anew = rho - a.max;
              if (anew < -kTol && anew > av + kTol) {
                row.b -= (av - anew) * L;
                v = anew * s;
                ++stats.coefficients_tightened;
                row_changed = true;
                break;
              }
            }
          }
        }
        if (!row_changed) break;
        any_change = true;
      }
    }

    // Dual fixing, [CIP] Algorithm 10.14: with no row resisting a move in the
    // direction the objective favours, the column goes to that bound. Locks
    // as in Example 3.4: an equality locks both ways, `a'x <= b` locks up for
    // a > 0 and down for a < 0.
    std::vector<std::size_t> down(n, 0), up(n, 0);
    for (const Row& row : rows) {
      if (!row.alive) continue;
      for (const auto& [j, v] : row.entries) {
        if (row.equality) {
          ++down[j];
          ++up[j];
        } else if (v > 0.0) {
          ++up[j];
        } else if (v < 0.0) {
          ++down[j];
        }
      }
    }
    for (std::size_t j = 0; j < n; ++j) {
      if (lo[j] == hi[j]) continue;
      const Real c = p.c[j];
      if (c >= 0.0 && down[j] == 0 && is_finite_bound(lo[j])) {
        hi[j] = lo[j];
      } else if (c <= 0.0 && up[j] == 0 && is_finite_bound(hi[j])) {
        lo[j] = hi[j];
      } else {
        continue;  // an infinite target bound: left for the LP to report
      }
      ++stats.columns_fixed;
      any_change = true;
    }
  }

  // Write back: bounds, and the surviving rows -- equalities first, as the
  // canonical form requires (Canonical.hpp).
  for (std::size_t j = 0; j < n; ++j) {
    p.col_lower[j] = lo[j];
    p.col_upper[j] = hi[j];
  }
  std::vector<std::size_t> keep;
  std::size_t equalities = 0;
  for (std::size_t i = 0; i < m; ++i) {
    if (rows[i].alive && rows[i].equality) {
      keep.push_back(i);
      ++equalities;
    }
  }
  for (std::size_t i = 0; i < m; ++i) {
    if (rows[i].alive && !rows[i].equality) keep.push_back(i);
  }
  auto enumerate = [&](auto&& emit) {
    for (std::size_t r = 0; r < keep.size(); ++r) {
      for (const auto& [j, v] : rows[keep[r]].entries) {
        if (v != 0.0) emit(static_cast<core::Index>(r), static_cast<core::Index>(j), v);
      }
    }
  };
  core::SparseBuilder builder(keep.size(), n);
  enumerate([&](core::Index r, core::Index c, Real) { builder.count(r, c); });
  if (auto st = builder.allocate(); !st.ok()) return st;
  enumerate([&](core::Index r, core::Index c, Real v) { builder.insert(r, c, v); });
  p.A = builder.finish(true, 0.0);
  core::RealVector b(keep.size());
  for (std::size_t r = 0; r < keep.size(); ++r) b[r] = rows[keep[r]].b;
  p.b = std::move(b);
  p.num_equality = equalities;
  return core::Status::Ok();
}

}  // namespace sovsolve::solver
