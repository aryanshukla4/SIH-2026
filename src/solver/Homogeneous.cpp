#include "sovsolve/solver/Homogeneous.hpp"

#include <algorithm>
#include <cmath>

namespace sovsolve::solver {

namespace {

using core::is_finite_bound;

Real inf_norm(const core::RealVector& v) {
  Real worst = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) worst = std::fmax(worst, std::fabs(v[i]));
  return worst;
}

}  // namespace

core::Status compute_homogeneous_residuals(const model::CanonicalProblem& problem,
                                           const SolverState& state, Real mu,
                                           HomogeneousResiduals& out) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_i = problem.num_inequality_rows();
  const std::size_t m_e = problem.num_equality;

  if (state.x.size() != n || state.s.size() != m_i || state.y.size() != m ||
      state.z.size() != n || state.v.size() != n) {
    return core::make_error(
        core::ErrorCode::DimensionMismatch,
        "compute_homogeneous_residuals: SolverState size does not match problem");
  }

  const Real tau = state.tau;

  out.rp = core::RealVector(m);
  out.rd = core::RealVector(n);
  out.rxz = core::RealVector(n, 0.0);
  out.ruv = core::RealVector(n, 0.0);
  out.rsy = core::RealVector(m_i);

  // rp = A x + [0; s] - b*tau. Identical to the direct form except for the
  // `tau` on `b` -- which is the whole homogenization, applied one constant at
  // a time.
  const auto& A_csr = problem.A.csr;
  for (std::size_t i = 0; i < m; ++i) {
    Real ax = 0.0;
    for (std::size_t k = A_csr.slice_begin(i); k < A_csr.slice_end(i); ++k) {
      ax += A_csr.values()[k] * state.x[static_cast<std::size_t>(A_csr.indices()[k])];
    }
    Real rp_i = ax - problem.b[i] * tau;
    if (i >= m_e) rp_i += state.s[i - m_e];
    out.rp[i] = rp_i;
  }

  core::RealVector Qx(n, 0.0);
  if (!problem.Q.empty()) {
    const auto& Q_csr = problem.Q.csr;
    for (std::size_t i = 0; i < n; ++i) {
      Real qi = 0.0;
      for (std::size_t k = Q_csr.slice_begin(i); k < Q_csr.slice_end(i); ++k) {
        qi += Q_csr.values()[k] * state.x[static_cast<std::size_t>(Q_csr.indices()[k])];
      }
      Qx[i] = qi;
    }
  }

  core::RealVector ATy(n, 0.0);
  const auto& A_csc = problem.A.csc;
  for (std::size_t j = 0; j < n; ++j) {
    Real aty = 0.0;
    for (std::size_t k = A_csc.slice_begin(j); k < A_csc.slice_end(j); ++k) {
      aty += A_csc.values()[k] * state.y[static_cast<std::size_t>(A_csc.indices()[k])];
    }
    ATy[j] = aty;
  }

  // rd = Qx + c*tau - A'y - z + v. The sign convention is the DIRECT form's,
  // not the embedding's own `A'y + z - v - c tau = 0` -- they differ by an
  // overall negation, and matching the existing one is what lets the same
  // recovery and refinement code be reused unchanged.
  for (std::size_t j = 0; j < n; ++j) {
    out.rd[j] = Qx[j] + problem.c[j] * tau - ATy[j] - state.z[j] + state.v[j];
  }

  // Bound complementarity, with the bounds themselves scaled by tau.
  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(problem.col_lower[j])) {
      out.rxz[j] = (state.x[j] - problem.col_lower[j] * tau) * state.z[j] - mu;
    }
    if (is_finite_bound(problem.col_upper[j])) {
      out.ruv[j] = (problem.col_upper[j] * tau - state.x[j]) * state.v[j] - mu;
    }
  }

  for (std::size_t k = 0; k < m_i; ++k) {
    out.rsy[k] = -state.s[k] * state.y[m_e + k] - mu;
  }

  // The gap row: c'x - b'y - l'z + u'v + kappa.
  //
  // The finite-bound guards are not cosmetic. An infinite bound has no
  // complementarity pair, so its `z` or `v` is identically zero and the term
  // is absent -- but `inf * 0` is NaN, so it must be skipped rather than
  // computed and found to vanish.
  Real cx = 0.0;
  for (std::size_t j = 0; j < n; ++j) cx += problem.c[j] * state.x[j];
  Real by = 0.0;
  for (std::size_t i = 0; i < m; ++i) by += problem.b[i] * state.y[i];
  Real lz = 0.0;
  Real uv = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(problem.col_lower[j])) lz += problem.col_lower[j] * state.z[j];
    if (is_finite_bound(problem.col_upper[j])) uv += problem.col_upper[j] * state.v[j];
  }
  out.rg = cx - by - lz + uv + state.kappa;
  out.rtk = state.tau * state.kappa - mu;

  out.rp_inf = inf_norm(out.rp);
  out.rd_inf = inf_norm(out.rd);
  out.complementarity_inf = std::max(
      {inf_norm(out.rxz), inf_norm(out.ruv), inf_norm(out.rsy), std::fabs(out.rtk)});

  return core::Status::Ok();
}

Real homogeneous_mu(const model::CanonicalProblem& problem, const SolverState& state) {
  const std::size_t n = problem.num_cols();
  const std::size_t m_i = problem.num_inequality_rows();
  const std::size_t m_e = problem.num_equality;
  const Real tau = state.tau;

  Real total = 0.0;
  std::size_t pairs = 0;
  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(problem.col_lower[j])) {
      total += (state.x[j] - problem.col_lower[j] * tau) * state.z[j];
      ++pairs;
    }
    if (is_finite_bound(problem.col_upper[j])) {
      total += (problem.col_upper[j] * tau - state.x[j]) * state.v[j];
      ++pairs;
    }
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    total += -state.s[k] * state.y[m_e + k];
    ++pairs;
  }

  // The embedding's own pair. It counts in BOTH the numerator and the
  // denominator -- omitting it from the denominator would inflate mu on a
  // model with few bounds, which is the same error FORMULATION.md section 6
  // records for dividing by `2n + m_I` instead of the active count.
  total += state.tau * state.kappa;
  ++pairs;

  return pairs == 0 ? 0.0 : total / static_cast<Real>(pairs);
}

HomogeneousVerdict classify_homogeneous(const model::CanonicalProblem& problem,
                                        const SolverState& state, Real tau_floor) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();

  // `tau` and `kappa` are compared against EACH OTHER, not against an absolute
  // threshold. Both scale with the problem's data, so "tau is small" only
  // means anything relative to the quantity it is complementary to.
  const Real scale = std::fmax(state.tau, state.kappa);
  if (!(scale > 0.0)) return HomogeneousVerdict::Indeterminate;
  if (state.tau > tau_floor * scale) return HomogeneousVerdict::Optimal;

  // tau has collapsed, so the iterate is a ray. Which kind is decided by the
  // two objective terms of section 13.2.
  Real cx = 0.0;
  for (std::size_t j = 0; j < n; ++j) cx += problem.c[j] * state.x[j];
  Real dual_objective = 0.0;
  for (std::size_t i = 0; i < m; ++i) dual_objective += problem.b[i] * state.y[i];
  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(problem.col_lower[j])) {
      dual_objective += problem.col_lower[j] * state.z[j];
    }
    if (is_finite_bound(problem.col_upper[j])) {
      dual_objective -= problem.col_upper[j] * state.v[j];
    }
  }

  // A margin, scaled to the magnitudes involved, on both tests. Neither is a
  // tie-break: if both fire, or neither does, the honest answer is that this
  // iterate does not decide it.
  const Real reference = std::fmax(std::fabs(cx), std::fabs(dual_objective));
  const Real margin = tau_floor * (1.0 + reference);
  const bool primal_infeasible = dual_objective > margin;
  const bool dual_infeasible = cx < -margin;

  if (primal_infeasible && !dual_infeasible) return HomogeneousVerdict::PrimalInfeasible;
  if (dual_infeasible && !primal_infeasible) return HomogeneousVerdict::DualInfeasible;

  // Both or neither. `Indeterminate` is kept as its own outcome rather than
  // folded into a guess: a model can be BOTH primal and dual infeasible, in
  // which case naming one is wrong, and a `tau` that collapsed without either
  // objective moving means the run stalled rather than proved anything.
  return HomogeneousVerdict::Indeterminate;
}

core::Status recover_from_homogeneous(SolverState& state) {
  if (!(state.tau > 0.0) || !std::isfinite(state.tau)) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "recover_from_homogeneous: tau is not positive, so the "
                            "iterate is a ray rather than a scaled solution");
  }
  const Real inv = 1.0 / state.tau;
  for (std::size_t j = 0; j < state.x.size(); ++j) state.x[j] *= inv;
  for (std::size_t k = 0; k < state.s.size(); ++k) state.s[k] *= inv;
  for (std::size_t i = 0; i < state.y.size(); ++i) state.y[i] *= inv;
  for (std::size_t j = 0; j < state.z.size(); ++j) state.z[j] *= inv;
  for (std::size_t j = 0; j < state.v.size(); ++j) state.v[j] *= inv;
  state.kappa *= inv;
  state.tau = 1.0;
  return core::Status::Ok();
}

}  // namespace sovsolve::solver
