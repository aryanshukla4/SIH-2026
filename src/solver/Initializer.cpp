#include "sovsolve/solver/Initializer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "sovsolve/solver/NormalFactor.hpp"

namespace sovsolve::solver {

namespace {

/// Offset from a single finite bound for an otherwise one-sided column.
/// Arbitrary but conventional -- any positive constant keeps the point
/// strictly interior; 1 is the usual textbook choice.
constexpr Real kUnboundedOffset = 1.0;

/// S. Mehrotra, "On the implementation of a primal-dual interior point
/// method", SIAM J. Optim. 2 (1992), section 7, (7.1)-(7.3):
///
///     pi~ = (AA')^-1 A c,   s~ = c - A'pi~,   x~ = A'(AA')^-1 b
///     dx  = max(-1.5 min x~, 0),   ds = max(-1.5 min s~, 0)
///     dx^ = dx + .5 (x~+dx)'(s~+ds) / sum(s~+ds)
///     ds^ = ds + .5 (x~+dx)'(s~+ds) / sum(x~+dx)
///     x0 = x~ + dx^,   s0 = s~ + ds^
///
/// The paper's x >= 0 is our set of PRIMAL GAPS -- x - l, u - x and the
/// inequality slacks s -- and its s is their DUALS z, v and -y_I, so the
/// shifts apply to those. The inequality rows' slack columns join A as an
/// identity block, which also makes AA' nonsingular on those rows. A box
/// column cannot move both gaps up at once; it is placed at least the shift
/// inside each bound, or at the box's middle when the box is narrower.
///
/// Mehrotra measured the point to take "a significantly smaller number of
/// iterations" than ad hoc ones; this replaces the fixed start above, which
/// left Netlib greenbea taking steps of ~0.001 for 180 iterations.
bool mehrotra_start(const CanonicalProblem& problem, SolverState& state) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;
  const std::size_t m_i = m - m_e;
  if (m == 0) return false;
  const auto& csr = problem.A.csr;
  const auto& csc = problem.A.csc;

  // AA' + [0; I] (+ a tiny regularization for dependent equality rows).
  NormalFactor factor(problem);
  core::RealVector theta(n, 1.0);
  core::RealVector diag(m, 1e-8);
  for (std::size_t i = m_e; i < m; ++i) diag[i] += 1.0;
  if (!factor.factorize(theta, diag).ok()) return false;

  // x~ = A'w, s_row~ = w_I with (AA' + [0;I]) w = b.
  std::vector<Real> w(problem.b.data(), problem.b.data() + m);
  factor.solve(w);
  std::vector<Real> xt(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    Real acc = 0.0;
    for (auto k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      acc += csc.values()[k] * w[static_cast<std::size_t>(csc.indices()[k])];
    }
    xt[j] = acc;
  }
  std::vector<Real> st(m_i);
  for (std::size_t k = 0; k < m_i; ++k) st[k] = w[m_e + k];

  // pi~ = (AA' + [0;I])^-1 A c  (the slack columns cost nothing).
  std::vector<Real> pi(m, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    Real acc = 0.0;
    for (auto k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      acc += csr.values()[k] * problem.c[static_cast<std::size_t>(csr.indices()[k])];
    }
    pi[i] = acc;
  }
  factor.solve(pi);
  std::vector<Real> d(n);  // reduced costs c - A'pi
  for (std::size_t j = 0; j < n; ++j) {
    Real acc = problem.c[j];
    for (auto k = csc.slice_begin(j); k < csc.slice_end(j); ++k) {
      acc -= csc.values()[k] * pi[static_cast<std::size_t>(csc.indices()[k])];
    }
    d[j] = acc;
  }

  // Primal gaps and their duals, pair by pair.
  Real min_gap = core::INF, min_dual = core::INF;
  const auto lower = [&](std::size_t j) { return core::is_finite_bound(problem.col_lower[j]); };
  const auto upper = [&](std::size_t j) { return core::is_finite_bound(problem.col_upper[j]); };
  for (std::size_t j = 0; j < n; ++j) {
    if (lower(j)) {
      min_gap = std::min(min_gap, xt[j] - problem.col_lower[j]);
      min_dual = std::min(min_dual, upper(j) ? std::max(d[j], 0.0) : d[j]);
    }
    if (upper(j)) {
      min_gap = std::min(min_gap, problem.col_upper[j] - xt[j]);
      min_dual = std::min(min_dual, lower(j) ? std::max(-d[j], 0.0) : -d[j]);
    }
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    min_gap = std::min(min_gap, st[k]);
    min_dual = std::min(min_dual, -pi[m_e + k]);  // slack column dual: 0 - pi_I
  }
  if (!std::isfinite(min_gap)) return false;  // no complementarity pair at all
  const Real dx = std::max(-1.5 * min_gap, 0.0);
  const Real ds = std::max(-1.5 * min_dual, 0.0);

  // (7.2)-(7.3) over every pair.
  Real prod = 0.0, sum_gap = 0.0, sum_dual = 0.0;
  const auto pair = [&](Real gap, Real dual) {
    prod += (gap + dx) * (dual + ds);
    sum_gap += gap + dx;
    sum_dual += dual + ds;
  };
  for (std::size_t j = 0; j < n; ++j) {
    if (lower(j)) pair(xt[j] - problem.col_lower[j], upper(j) ? std::max(d[j], 0.0) : d[j]);
    if (upper(j)) pair(problem.col_upper[j] - xt[j], lower(j) ? std::max(-d[j], 0.0) : -d[j]);
  }
  for (std::size_t k = 0; k < m_i; ++k) pair(st[k], -pi[m_e + k]);
  if (!(sum_gap > 0.0) || !(sum_dual > 0.0)) return false;
  const Real dxh = dx + 0.5 * prod / sum_dual;
  const Real dsh = ds + 0.5 * prod / sum_gap;
  if (!(dxh > 0.0) || !(dsh > 0.0) || !std::isfinite(dxh) || !std::isfinite(dsh)) return false;

  for (std::size_t j = 0; j < n; ++j) {
    const Real lo = problem.col_lower[j], hi = problem.col_upper[j];
    Real x = xt[j];
    if (lower(j) && upper(j)) {
      const Real margin = std::min(dxh, 0.5 * (hi - lo));
      x = std::min(std::max(x, lo + margin), hi - margin);
      if (margin == 0.5 * (hi - lo)) x = 0.5 * (lo + hi);
    } else if (lower(j)) {
      x = xt[j] + dxh;
    } else if (upper(j)) {
      x = xt[j] - dxh;
    }
    state.x[j] = x;
    state.z[j] = lower(j) ? (upper(j) ? std::max(d[j], 0.0) : d[j]) + dsh : 0.0;
    state.v[j] = upper(j) ? (lower(j) ? std::max(-d[j], 0.0) : -d[j]) + dsh : 0.0;
  }
  for (std::size_t i = 0; i < m_e; ++i) state.y[i] = pi[i];
  for (std::size_t k = 0; k < m_i; ++k) {
    state.s[k] = st[k] + dxh;
    state.y[m_e + k] = -(-pi[m_e + k] + dsh);
  }

  // Everything must be strictly interior, or the caller falls back.
  for (std::size_t j = 0; j < n; ++j) {
    if (lower(j) && !(state.x[j] - problem.col_lower[j] > 0.0 && state.z[j] > 0.0)) return false;
    if (upper(j) && !(problem.col_upper[j] - state.x[j] > 0.0 && state.v[j] > 0.0)) return false;
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    if (!(state.s[k] > 0.0 && -state.y[m_e + k] > 0.0)) return false;
  }
  return true;
}

}  // namespace

Expected<SolverState> initialize(const CanonicalProblem& problem, const Options& options,
                                  const core::RealVector* warm_start) {
  std::size_t bad_index = 0;
  bool bad_is_row = false;
  if (!problem.is_ipm_startable(&bad_index, &bad_is_row)) {
    return core::make_error(
        core::ErrorCode::NumericalError,
        std::string("model is not IPM-startable: ") +
            (bad_is_row ? "row " : "column ") + std::to_string(bad_index));
  }

  SolverState state;
  const std::size_t n = problem.num_cols();
  const std::size_t m_i = problem.num_inequality_rows();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;

  state.x = core::RealVector(n);
  state.z = core::RealVector(n);
  state.v = core::RealVector(n);
  state.s = core::RealVector(m_i);
  state.y = core::RealVector(m, 0.0);
  // -y_I > 0 is a strict interiority requirement (FORMULATION.md 3), not
  // just a convenience -- equality-row y is unrestricted and stays 0, but
  // y_I = 0 both violates that contract and makes D_s = s/(-y_I) divide by
  // zero the moment the KKT builder runs on iteration 0.
  for (std::size_t i = m_e; i < m; ++i) state.y[i] = -1.0;

  if (options.ipm.mehrotra_start && warm_start == nullptr) {
    if (mehrotra_start(problem, state)) {
      Real sum = 0.0;
      std::size_t pairs = 0;
      for (std::size_t j = 0; j < n; ++j) {
        if (core::is_finite_bound(problem.col_lower[j])) {
          sum += (state.x[j] - problem.col_lower[j]) * state.z[j];
          ++pairs;
        }
        if (core::is_finite_bound(problem.col_upper[j])) {
          sum += (problem.col_upper[j] - state.x[j]) * state.v[j];
          ++pairs;
        }
      }
      for (std::size_t k = 0; k < m_i; ++k) {
        sum += state.s[k] * -state.y[m_e + k];
        ++pairs;
      }
      state.mu = pairs > 0 ? sum / static_cast<Real>(pairs) : 1.0;
      return state;
    }
    // Not strictly interior: back to the fixed start below, from a clean y.
    for (std::size_t i = 0; i < m; ++i) state.y[i] = i < m_e ? 0.0 : -1.0;
  }

  Real sum_lower = 0.0;
  Real sum_upper = 0.0;
  std::size_t active_pairs = m_i;

  for (std::size_t j = 0; j < n; ++j) {
    const Real lo = problem.col_lower[j];
    const Real hi = problem.col_upper[j];
    const bool has_lower = core::is_finite_bound(lo);
    const bool has_upper = core::is_finite_bound(hi);

    const bool has_hint = warm_start != nullptr && j < warm_start->size() &&
                          core::is_finite_bound((*warm_start)[j]);
    if (has_hint) {
      // A real hint is clamped strictly inside the column's own bounds --
      // never used against a finite bound as-is -- the margin scales with
      // the box's own width so a binary column's [0,1] box (extremely
      // common in Module 22's branch-and-bound) is not distorted by a
      // margin sized for a wide one.
      Real x_hint = (*warm_start)[j];
      if (has_lower && has_upper) {
        const Real margin = std::min({1e-6, 0.001 * (hi - lo), 0.25 * (hi - lo)});
        x_hint = std::min(std::max(x_hint, lo + margin), hi - margin);
      } else if (has_lower) {
        x_hint = std::max(x_hint, lo + kUnboundedOffset * 1e-6);
      } else if (has_upper) {
        x_hint = std::min(x_hint, hi - kUnboundedOffset * 1e-6);
      }
      state.x[j] = x_hint;
    } else if (has_lower && has_upper) {
      state.x[j] = 0.5 * (lo + hi);
    } else if (has_lower) {
      state.x[j] = lo + kUnboundedOffset;
    } else if (has_upper) {
      state.x[j] = hi - kUnboundedOffset;
    } else {
      state.x[j] = 0.0;
    }

    state.z[j] = has_lower ? 1.0 : 0.0;
    state.v[j] = has_upper ? 1.0 : 0.0;

    if (has_lower) {
      sum_lower += (state.x[j] - lo) * state.z[j];
      ++active_pairs;
    }
    if (has_upper) {
      sum_upper += (hi - state.x[j]) * state.v[j];
      ++active_pairs;
    }
  }

  // `s_k = b_i - (Ax)_i` -- the slack that makes row `i` EXACTLY satisfied by
  // the x just chosen above, rather than a flat 1.0 oblivious to the row's
  // own scale and RHS. On a large, tightly-coupled instance (80bau3b: 2262
  // inequality rows, 9799 columns) a flat slack left the primal residual at
  // ~4e5 from iteration 0 -- not because the bound-midpoint x was a
  // uniformly bad guess, but because s never even tried to match what x
  // already implied, so nearly every row started out "wrong" by whatever
  // its own RHS happened to be. Falls back to the textbook 1.0 only when the
  // implied slack isn't safely positive (the row is already tight or
  // violated at this x) -- interiority must never be sacrificed for a
  // closer-but-boundary-touching start.
  {
    const auto& csr = problem.A.csr;
    for (std::size_t k = 0; k < m_i; ++k) {
      const std::size_t i = m_e + k;
      Real activity = 0.0;
      for (auto t = csr.slice_begin(i); t < csr.slice_end(i); ++t) {
        activity += csr.values()[t] * state.x[static_cast<std::size_t>(csr.indices()[t])];
      }
      const Real implied_slack = problem.b[i] - activity;
      state.s[k] = implied_slack > kUnboundedOffset ? implied_slack : kUnboundedOffset;
    }
  }

  // mu = [sum((x-l).*z) + sum((u-x).*v) + sum(-s.*y_I)] / active_pair_count
  // (Module 16's formula, FORMULATION.md 6). Guards the degenerate case of a
  // model with no bounds and no inequality rows at all -- unreachable in
  // practice, but division by zero is not a graceful way to find out.
  Real sum_slack = 0.0;
  for (std::size_t k = 0; k < m_i; ++k) {
    sum_slack += -state.s[k] * state.y[m_e + k];
  }
  state.mu = active_pairs > 0
                 ? (sum_lower + sum_upper + sum_slack) / static_cast<Real>(active_pairs)
                 : 1.0;

  return state;
}

}  // namespace sovsolve::solver
