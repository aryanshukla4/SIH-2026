#include "sovsolve/solver/HostKkt.hpp"

#include <cmath>

#include "sovsolve/solver/NormalFactor.hpp"

namespace sovsolve::solver {

namespace {

core::HostSpan<const Real> in(const core::RealVector& v) { return {v.data(), v.size()}; }
core::HostSpan<Real> out_span(core::RealVector& v) { return {v.data(), v.size()}; }

Real dot(const core::RealVector& a, const core::RealVector& b) {
  Real acc = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) acc += a[i] * b[i];
  return acc;
}

}  // namespace

HostKktSolver::HostKktSolver(const model::CanonicalProblem& problem,
                             const HomogeneousBorder& border,
                             const HostKktOptions& options, NormalFactor* factor)
    : problem_(&problem), matvec_(problem), options_(options) {
  const std::size_t n = problem.num_cols();
  const std::size_t m = problem.num_rows();
  const std::size_t m_e = problem.num_equality;

  theta_ = core::RealVector(n);
  for (std::size_t j = 0; j < n; ++j) {
    // Floor THEN invert -- the other order divides by zero on a free column.
    theta_[j] = 1.0 / std::fmax(border.theta_inv[j], options_.theta_inv_floor);
  }

  d_slack_ = core::RealVector(m, 0.0);
  for (std::size_t k = 0; k < problem.num_inequality_rows(); ++k) {
    d_slack_[m_e + k] = border.d_slack[k];
  }

  // Jacobi preconditioner: `diag(A Theta A')_i = sum_j A_ij^2 Theta_j`, one
  // sweep of the CSR. Cheap, and it is the difference between CG converging in
  // tens of iterations and hundreds on a badly scaled row.
  jacobi_ = core::RealVector(m);
  const auto& csr = problem.A.csr;
  for (std::size_t i = 0; i < m; ++i) {
    Real acc = d_slack_[i] + options_.delta_d;
    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      const Real a = csr.values()[k];
      acc += a * a * theta_[static_cast<std::size_t>(csr.indices()[k])];
    }
    jacobi_[i] = acc > 0.0 ? 1.0 / acc : 1.0;
  }

  // Direct factor of A_S Theta A_S' + D_s + delta_d I, [AG99] section 5: on a
  // breakdown the dual regularization goes up by 10 and the factorization is
  // retried. The CG operator uses the SAME delta_d as the factor, so CG
  // converges to the regularized (proximal) Newton direction -- whose
  // right-hand side the regularization does not change, [AG99] (45).
  if (factor != nullptr) {
    core::RealVector diag(m);
    for (std::size_t attempt = 0; attempt <= options_.regularization_retries; ++attempt) {
      for (std::size_t i = 0; i < m; ++i) diag[i] = d_slack_[i] + options_.delta_d;
      if (factor->factorize(theta_, diag).ok()) {
        factor_ = factor;
        break;
      }
      options_.delta_d *= 10.0;
    }
    if (factor_ != nullptr) {
      precond_.assign(m, 0.0);
    } else {
      options_.delta_d = options.delta_d;  // Jacobi fallback, as without a factor
    }
    for (std::size_t i = 0; i < m; ++i) {
      Real acc = d_slack_[i] + options_.delta_d;
      for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
        const Real a = csr.values()[k];
        acc += a * a * theta_[static_cast<std::size_t>(csr.indices()[k])];
      }
      jacobi_[i] = acc > 0.0 ? 1.0 / acc : 1.0;
    }
  }

  r_ = core::RealVector(m);
  p_ = core::RealVector(m);
  ap_ = core::RealVector(m);
  zvec_ = core::RealVector(m);
  rhs_ = core::RealVector(m);
  scratch_n_ = core::RealVector(n);
}

void HostKktSolver::precondition() {
  const std::size_t m = r_.size();
  if (factor_ == nullptr) {
    for (std::size_t i = 0; i < m; ++i) zvec_[i] = jacobi_[i] * r_[i];
    return;
  }
  for (std::size_t i = 0; i < m; ++i) precond_[i] = r_[i];
  factor_->solve(precond_);
  // A factor can pass the backend's own positive-definiteness test with a
  // pivot so small that the solve overflows (CHOLMOD's supernodal code has no
  // per-pivot rule, unlike SparseLdl). An overflowed preconditioner would
  // poison CG with NaN, so this solver drops back to Jacobi for the rest of
  // its life instead.
  bool finite = true;
  for (std::size_t i = 0; i < m && finite; ++i) finite = std::isfinite(precond_[i]);
  if (!finite) {
    factor_ = nullptr;
    ++factor_failures_;
    for (std::size_t i = 0; i < m; ++i) zvec_[i] = jacobi_[i] * r_[i];
    return;
  }
  for (std::size_t i = 0; i < m; ++i) zvec_[i] = precond_[i];
}

void HostKktSolver::apply(const core::RealVector& input, core::RealVector& output) {
  // `A Theta A' v + D_s v + delta_d v`, never forming the product.
  matvec_.multiply_transpose(in(input), out_span(scratch_n_));
  for (std::size_t j = 0; j < scratch_n_.size(); ++j) scratch_n_[j] *= theta_[j];
  matvec_.multiply(in(scratch_n_), out_span(output));
  for (std::size_t i = 0; i < output.size(); ++i) {
    output[i] += (d_slack_[i] + options_.delta_d) * input[i];
  }
}

core::Status HostKktSolver::solve(core::HostSpan<const Real> rhs_x,
                                  core::HostSpan<const Real> rhs_y,
                                  core::HostSpan<Real> dx, core::HostSpan<Real> dy) {
  const std::size_t n = problem_->num_cols();
  const std::size_t m = problem_->num_rows();
  if (rhs_x.size() != n || rhs_y.size() != m || dx.size() != n || dy.size() != m) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "HostKktSolver::solve: span sizes do not match the problem");
  }

  // rhs = ry + A Theta rx
  for (std::size_t j = 0; j < n; ++j) scratch_n_[j] = theta_[j] * rhs_x[j];
  matvec_.multiply(in(scratch_n_), out_span(rhs_));
  for (std::size_t i = 0; i < m; ++i) rhs_[i] += rhs_y[i];

  // Preconditioned CG from a zero start, so the initial residual IS the
  // right-hand side and the first apply() is saved.
  for (std::size_t i = 0; i < m; ++i) {
    dy[i] = 0.0;
    r_[i] = rhs_[i];
  }
  precondition();
  for (std::size_t i = 0; i < m; ++i) p_[i] = zvec_[i];
  Real rz = dot(r_, zvec_);
  const Real rhs_norm = std::sqrt(dot(rhs_, rhs_));
  const Real target = options_.tolerance * std::fmax(rhs_norm, 1.0);

  std::size_t iteration = 0;
  if (rhs_norm > 0.0) {
    for (; iteration < options_.max_iterations; ++iteration) {
      apply(p_, ap_);
      const Real pap = dot(p_, ap_);
      // A nonpositive curvature means the operator is not positive definite at
      // working precision -- the regularization was not enough. Stop with what
      // we have rather than dividing by it; the step-length test downstream is
      // what decides whether the direction is usable.
      if (!(pap > 0.0) || !std::isfinite(pap)) break;

      const Real alpha = rz / pap;
      for (std::size_t i = 0; i < m; ++i) {
        dy[i] += alpha * p_[i];
        r_[i] -= alpha * ap_[i];
      }
      if (std::sqrt(dot(r_, r_)) <= target) {
        ++iteration;
        break;
      }
      precondition();
      const Real rz_next = dot(r_, zvec_);
      const Real beta = rz_next / rz;
      rz = rz_next;
      for (std::size_t i = 0; i < m; ++i) p_[i] = zvec_[i] + beta * p_[i];
    }
  }
  cg_iterations_ += iteration;
  if (iteration >= options_.max_iterations) hit_cap_ = true;

  // Back out dx = Theta (A'dy - rx).
  core::RealVector dy_copy(m);
  for (std::size_t i = 0; i < m; ++i) dy_copy[i] = dy[i];
  matvec_.multiply_transpose(in(dy_copy), out_span(scratch_n_));
  for (std::size_t j = 0; j < n; ++j) dx[j] = theta_[j] * (scratch_n_[j] - rhs_x[j]);

  ++solves_;
  return core::Status::Ok();
}

}  // namespace sovsolve::solver
