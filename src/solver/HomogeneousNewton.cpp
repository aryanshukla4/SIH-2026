#include "sovsolve/solver/HomogeneousNewton.hpp"

#include <cmath>

namespace sovsolve::solver {

namespace {

using core::is_finite_bound;

/// Sizes a workspace vector without reallocating when it is already right.
/// An interior-point iteration must not allocate.
void ensure(core::RealVector& v, std::size_t n) {
  if (v.size() != n) v = core::RealVector(n);
}

core::HostSpan<const Real> in(const core::RealVector& v) {
  return {v.data(), v.size()};
}

core::HostSpan<Real> out_span(core::RealVector& v) { return {v.data(), v.size()}; }

/// Every quantity the border divides by. Checked once, up front, so a failure
/// names the cause instead of surfacing as a NaN four modules downstream --
/// `0 * inf` is `NaN` in IEEE754, and `NaN` compares false against every
/// tolerance, so the solve would run silently to the iteration limit and report
/// MaxIterations on a solvable problem. FORMULATION.md section 10.1 records
/// the same failure for the unfloored `Theta^-1`.
bool interior(Real value) { return value > 0.0 && std::isfinite(value); }

}  // namespace

core::Status compute_homogeneous_border(const model::CanonicalProblem& problem,
                                        const SolverState& state,
                                        HomogeneousBorder& out) {
  const std::size_t n = problem.num_cols();
  const std::size_t m_i = problem.num_inequality_rows();
  const std::size_t m_e = problem.num_equality;
  const Real tau = state.tau;

  if (state.x.size() != n || state.z.size() != n || state.v.size() != n ||
      state.s.size() != m_i || state.y.size() != problem.num_rows()) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "compute_homogeneous_border: SolverState does not match "
                            "problem");
  }
  if (!interior(tau)) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "compute_homogeneous_border: tau has left the interior");
  }

  ensure(out.theta_inv, n);
  ensure(out.h_x, n);
  ensure(out.g_x, n);
  ensure(out.d_slack, m_i);
  out.w = 0.0;

  for (std::size_t j = 0; j < n; ++j) {
    // `theta_l` and `theta_u` are the two halves of section 10.1's diagonal,
    // with `tau` scaling the bounds. A bound that is infinite has no
    // complementarity pair, so it contributes to NEITHER the diagonal nor the
    // border -- which is exactly why a free column's border entry is already
    // [AA]'s `-c_j` and needs no generalization.
    Real theta_l = 0.0;
    Real theta_u = 0.0;
    Real bound_term = 0.0;

    if (is_finite_bound(problem.col_lower[j])) {
      const Real l = problem.col_lower[j];
      const Real gap = state.x[j] - l * tau;
      if (!interior(gap)) {
        return core::make_error(
            core::ErrorCode::NumericalError,
            "compute_homogeneous_border: x - l*tau has left the interior");
      }
      theta_l = state.z[j] / gap;
      bound_term += theta_l * l;
      out.w += theta_l * l * l;
    }
    if (is_finite_bound(problem.col_upper[j])) {
      const Real u = problem.col_upper[j];
      const Real gap = u * tau - state.x[j];
      if (!interior(gap)) {
        return core::make_error(
            core::ErrorCode::NumericalError,
            "compute_homogeneous_border: u*tau - x has left the interior");
      }
      theta_u = state.v[j] / gap;
      bound_term += theta_u * u;
      out.w += theta_u * u * u;
    }

    out.theta_inv[j] = theta_l + theta_u;
    // The column and the row differ by `2c`, not by a sign. See the header.
    out.h_x[j] = bound_term - problem.c[j];
    out.g_x[j] = bound_term + problem.c[j];
  }

  for (std::size_t k = 0; k < m_i; ++k) {
    const Real sigma = slack_dual(state.y[m_e + k]);
    if (!interior(sigma) || !interior(state.s[k])) {
      return core::make_error(
          core::ErrorCode::NumericalError,
          "compute_homogeneous_border: an inequality pair has left the interior");
    }
    out.d_slack[k] = state.s[k] / sigma;
  }

  out.trailing = -(out.w + state.kappa / tau);
  return core::Status::Ok();
}

core::Status refresh_border_solve(const model::CanonicalProblem& problem,
                                  const HomogeneousBorder& border, KktSolver& solver,
                                  HomogeneousNewtonWorkspace& work) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();

  if (border.h_x.size() != n) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "refresh_border_solve: border does not match problem");
  }

  ensure(work.p_x, n);
  ensure(work.p_y, m);
  ensure(work.rp_hat, m);
  work.p_valid = false;

  // The border column is `(h_x; -b)`. Reuse rp_hat as scratch for `-b` rather
  // than holding a fourth vector: it is overwritten per right-hand side anyway.
  for (std::size_t i = 0; i < m; ++i) work.rp_hat[i] = -problem.b[i];

  core::Status st = solver.solve(in(border.h_x), in(work.rp_hat), out_span(work.p_x),
                                 out_span(work.p_y));
  if (!st.ok()) return st;
  work.p_valid = true;
  return core::Status::Ok();
}

core::Status solve_homogeneous_newton(const model::CanonicalProblem& problem,
                                      const SolverState& state,
                                      const HomogeneousBorder& border,
                                      const HomogeneousNewtonRhs& rhs,
                                      KktSolver& solver,
                                      HomogeneousNewtonWorkspace& work, bool affine,
                                      SolverState& out) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_i = problem.num_inequality_rows();
  const std::size_t m_e = problem.num_equality;
  const Real tau = state.tau;

  if (!work.p_valid) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "solve_homogeneous_newton: refresh_border_solve has not "
                            "been called for this iterate");
  }
  if (rhs.rp.size() != m || rhs.rd.size() != n || rhs.rxz.size() != n ||
      rhs.ruv.size() != n || rhs.rsy.size() != m_i) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "solve_homogeneous_newton: right-hand side does not match "
                            "problem");
  }
  if (!interior(tau)) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "solve_homogeneous_newton: tau has left the interior");
  }

  ensure(work.rd_hat, n);
  ensure(work.rp_hat, m);
  ensure(work.u_x, n);
  ensure(work.u_y, m);

  // Eliminate dz and dv:
  //     dz = (rxz - Z dx + Z l dtau) / (x - l tau)
  //     dv = (ruv + V dx - V u dtau) / (u tau - x)
  // which moves `-rxz/(x - l tau) + ruv/(u tau - x)` onto the dual row and
  // `+l'rxz/(x - l tau) - u'ruv/(u tau - x)` onto the gap row.
  Real rg_hat = rhs.rg;
  for (std::size_t j = 0; j < n; ++j) {
    Real shift = 0.0;
    if (is_finite_bound(problem.col_lower[j])) {
      const Real term = rhs.rxz[j] / (state.x[j] - problem.col_lower[j] * tau);
      shift -= term;
      rg_hat += problem.col_lower[j] * term;
    }
    if (is_finite_bound(problem.col_upper[j])) {
      const Real term = rhs.ruv[j] / (problem.col_upper[j] * tau - state.x[j]);
      shift += term;
      rg_hat -= problem.col_upper[j] * term;
    }
    work.rd_hat[j] = rhs.rd[j] + shift;
  }

  // Eliminate ds: `ds = (rsy + S dy_I) / Sigma_I`, which moves
  // `-rsy/Sigma_I` onto the inequality primal rows and nothing onto the gap
  // row -- the gap row does not involve `s` at all. Checked separately against
  // a dense Jacobian: inequality rows touch only K's (2,2) block.
  for (std::size_t i = 0; i < m_e; ++i) work.rp_hat[i] = rhs.rp[i];
  for (std::size_t k = 0; k < m_i; ++k) {
    const Real sigma = slack_dual(state.y[m_e + k]);
    work.rp_hat[m_e + k] = rhs.rp[m_e + k] - rhs.rsy[k] / sigma;
  }

  // Eliminate dkappa: `dkappa = (rtk - kappa dtau) / tau`.
  const Real rho = rg_hat - rhs.rtk / tau;

  core::Status st = solver.solve(in(work.rd_hat), in(work.rp_hat), out_span(work.u_x),
                                 out_span(work.u_y));
  if (!st.ok()) return st;

  // Scalar Schur complement. [AA] (1.28)/(1.29) in our coordinates: with border
  // column `hc`, border row `gr` and trailing `border.trailing`,
  //
  //     dtau = (gr'u - rho) / (gr'p - trailing)
  //     (dx; dy) = u - dtau * p
  Real gr_u = 0.0;
  Real gr_p = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    gr_u += border.g_x[j] * work.u_x[j];
    gr_p += border.g_x[j] * work.p_x[j];
  }
  for (std::size_t i = 0; i < m; ++i) {
    gr_u -= problem.b[i] * work.u_y[i];
    gr_p -= problem.b[i] * work.p_y[i];
  }

  const Real denominator = gr_p - border.trailing;
  if (!std::isfinite(denominator) || denominator == 0.0) {
    return core::make_error(core::ErrorCode::NumericalError,
                            "solve_homogeneous_newton: the Schur complement is "
                            "singular, so dtau is not determined");
  }
  const Real dtau = (gr_u - rho) / denominator;

  core::RealVector& dx = affine ? out.dx_aff : out.dx;
  core::RealVector& dy = affine ? out.dy_aff : out.dy;
  core::RealVector& dz = affine ? out.dz_aff : out.dz;
  core::RealVector& dv = affine ? out.dv_aff : out.dv;
  core::RealVector& ds = affine ? out.ds_aff : out.ds;
  ensure(dx, n);
  ensure(dy, m);
  ensure(dz, n);
  ensure(dv, n);
  ensure(ds, m_i);

  for (std::size_t j = 0; j < n; ++j) dx[j] = work.u_x[j] - dtau * work.p_x[j];
  for (std::size_t i = 0; i < m; ++i) dy[i] = work.u_y[i] - dtau * work.p_y[i];

  // Back-substitute. A bound that does not exist gets a zero direction rather
  // than being left uninitialized -- the step length and mu both iterate over
  // pairs that exist, but a NaN here would propagate through any caller that
  // does not.
  for (std::size_t j = 0; j < n; ++j) {
    dz[j] = 0.0;
    dv[j] = 0.0;
    if (is_finite_bound(problem.col_lower[j])) {
      const Real l = problem.col_lower[j];
      dz[j] = (rhs.rxz[j] - state.z[j] * dx[j] + state.z[j] * l * dtau) /
              (state.x[j] - l * tau);
    }
    if (is_finite_bound(problem.col_upper[j])) {
      const Real u = problem.col_upper[j];
      dv[j] = (rhs.ruv[j] + state.v[j] * dx[j] - state.v[j] * u * dtau) /
              (u * tau - state.x[j]);
    }
  }
  for (std::size_t k = 0; k < m_i; ++k) {
    const Real sigma = slack_dual(state.y[m_e + k]);
    ds[k] = (rhs.rsy[k] + state.s[k] * dy[m_e + k]) / sigma;
  }

  if (affine) {
    out.dtau_aff = dtau;
    out.dkappa_aff = (rhs.rtk - state.kappa * dtau) / tau;
  } else {
    out.dtau = dtau;
    out.dkappa = (rhs.rtk - state.kappa * dtau) / tau;
  }
  return core::Status::Ok();
}

}  // namespace sovsolve::solver
