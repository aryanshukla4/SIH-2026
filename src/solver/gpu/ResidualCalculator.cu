#include "sovsolve/solver/gpu/ResidualCalculator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sovsolve::solver::gpu {

namespace {

Real inf_norm(const core::RealVector& v) {
  Real result = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) result = std::max(result, std::fabs(v[i]));
  return result;
}

}  // namespace

Status compute_residuals(const CanonicalProblem& problem, const SolverState& state,
                          Real mu, Residuals& out) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_i = problem.num_inequality_rows();
  const std::size_t m_e = problem.num_equality;

  if (state.x.size() != n || state.s.size() != m_i || state.y.size() != m ||
      state.z.size() != n || state.v.size() != n) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "compute_residuals: SolverState size does not match problem");
  }

  out.rp = core::RealVector(m);
  out.rd = core::RealVector(n);
  out.rxz = core::RealVector(n, 0.0);
  out.ruv = core::RealVector(n, 0.0);
  out.rsy = core::RealVector(m_i);

  // rp_E = A_E x - b_E ; rp_I = A_I x + s - b_I. One pass over A's CSR: rows
  // [0, m_e) are equality, [m_e, m) are inequality (CanonicalProblem's row
  // layout, see Canonical.hpp).
  const auto& A_csr = problem.A.csr;
  for (std::size_t i = 0; i < m; ++i) {
    Real ax = 0.0;
    for (std::size_t k = A_csr.slice_begin(i); k < A_csr.slice_end(i); ++k) {
      ax += A_csr.values()[k] * state.x[static_cast<std::size_t>(A_csr.indices()[k])];
    }
    Real rp_i = ax - problem.b[i];
    if (i >= m_e) rp_i += state.s[i - m_e];
    out.rp[i] = rp_i;
  }

  // rd = Qx + c - A'y - z + v. Q is full symmetric storage (Problem.hpp), so
  // a plain CSR SpMV gives Qx directly, no triangle-mirroring needed. A'y
  // uses the CSC orientation -- exactly what it exists for (SparseMatrix.hpp).
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

  for (std::size_t j = 0; j < n; ++j) {
    out.rd[j] = Qx[j] + problem.c[j] - ATy[j] - state.z[j] + state.v[j];
  }

  // rxz = (x-l).*z - mu ; ruv = (u-x).*v - mu, omitted (left at 0) where the
  // corresponding bound is infinite -- there is no barrier term there.
  for (std::size_t j = 0; j < n; ++j) {
    if (core::is_finite_bound(problem.col_lower[j])) {
      out.rxz[j] = (state.x[j] - problem.col_lower[j]) * state.z[j] - mu;
    }
    if (core::is_finite_bound(problem.col_upper[j])) {
      out.ruv[j] = (problem.col_upper[j] - state.x[j]) * state.v[j] - mu;
    }
  }

  // rsy = -s.*y_I - mu, over the inequality block of y (rows [m_e, m)).
  for (std::size_t k = 0; k < m_i; ++k) {
    out.rsy[k] = -state.s[k] * state.y[m_e + k] - mu;
  }

  out.rp_inf = inf_norm(out.rp);
  out.rd_inf = inf_norm(out.rd);
  out.complementarity_inf =
      std::max({inf_norm(out.rxz), inf_norm(out.ruv), inf_norm(out.rsy)});

  return Status::Ok();
}

}  // namespace sovsolve::solver::gpu
