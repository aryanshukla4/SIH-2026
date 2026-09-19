#include "sovsolve/solver/pdlp/IterationBackend.hpp"

#include <algorithm>

namespace sovsolve::solver::pdlp {

namespace {

core::HostSpan<const Real> in(const core::RealVector& v) { return {v.data(), v.size()}; }
core::HostSpan<Real> out(core::RealVector& v) { return {v.data(), v.size()}; }

void zeroed(core::RealVector& v, std::size_t n) {
  v.resize(n);
  v.assign(0.0);
}

void copy_into(const core::RealVector& from, core::HostSpan<Real> to) {
  for (std::size_t i = 0; i < from.size(); ++i) to[i] = from[i];
}

}  // namespace

HostIterationBackend::HostIterationBackend(const model::CanonicalProblem& problem,
                                           MatVec& matvec)
    : problem_(problem),
      matvec_(matvec),
      n_(problem.num_cols()),
      m_(problem.num_rows()) {
  for (auto* v : {&x_, &x_trial_, &extrapolated_, &kt_y_, &x_prev_, &diff_x_, &sum_x_,
                  &avg_x_}) {
    zeroed(*v, n_);
  }
  for (auto* v : {&y_, &y_trial_, &k_x_current_, &k_extrapolated_, &k_x_, &diff_y_,
                  &sum_y_, &avg_y_}) {
    zeroed(*v, m_);
  }
}

void HostIterationBackend::set_iterate(core::HostSpan<const Real> x,
                                       core::HostSpan<const Real> y) {
  for (std::size_t j = 0; j < n_; ++j) x_[j] = x[j];
  for (std::size_t i = 0; i < m_; ++i) y_[i] = y[i];
}

void HostIterationBackend::download(BackendVector which, core::HostSpan<Real> dst) {
  switch (which) {
    case BackendVector::X: copy_into(x_, dst); break;
    case BackendVector::Y: copy_into(y_, dst); break;
    case BackendVector::AverageX: copy_into(avg_x_, dst); break;
    case BackendVector::AverageY: copy_into(avg_y_, dst); break;
    case BackendVector::IterateSumX: copy_into(sum_x_, dst); break;
    case BackendVector::IterateSumY: copy_into(sum_y_, dst); break;
    case BackendVector::DifferenceX: copy_into(diff_x_, dst); break;
    case BackendVector::DifferenceY: copy_into(diff_y_, dst); break;
  }
}

void HostIterationBackend::begin_step() {
  matvec_.multiply_transpose(in(y_), out(kt_y_));
  matvec_.multiply(in(x_), out(k_x_current_));
}

// The loop bodies below are the pre-refactor Pdlp.cpp code, including the ORDER
// of every accumulation. That is not fussiness: floating-point addition is not
// associative, so reordering a sum changes the last bits, the adaptive step
// then accepts a borderline trial differently, and the run diverges -- which is
// exactly the drift measured between the CPU and cuSPARSE paths in section 24E.
// Keeping the order is what lets "the refactor changed nothing" be checked as
// bit-identical output rather than argued.
TrialMetrics HostIterationBackend::trial(Real tau, Real sigma) {
  // line 4: x' = proj_X(x - tau (c - K'y))
  for (std::size_t j = 0; j < n_; ++j) {
    const Real step = x_[j] - tau * (problem_.c[j] - kt_y_[j]);
    x_trial_[j] = std::clamp(step, problem_.col_lower[j], problem_.col_upper[j]);
    extrapolated_[j] = 2.0 * x_trial_[j] - x_[j];
  }

  // line 5: y' = proj_Y(y + sigma (q - K(2x' - x)))
  matvec_.multiply(in(extrapolated_), out(k_extrapolated_));
  for (std::size_t i = 0; i < m_; ++i) {
    const Real step = y_[i] + sigma * (problem_.b[i] - k_extrapolated_[i]);
    y_trial_[i] = i < problem_.num_equality ? step : std::min(step, 0.0);
  }

  // line 6. `K dx = (K(2x'-x) - Kx) / 2`, so it costs no product of its own.
  TrialMetrics metrics;
  for (std::size_t i = 0; i < m_; ++i) {
    const Real k_dx = 0.5 * (k_extrapolated_[i] - k_x_current_[i]);
    metrics.interaction += (y_trial_[i] - y_[i]) * k_dx;
  }
  for (std::size_t j = 0; j < n_; ++j) {
    const Real d = x_trial_[j] - x_[j];
    metrics.dx_sq += d * d;
  }
  for (std::size_t i = 0; i < m_; ++i) {
    const Real d = y_trial_[i] - y_[i];
    metrics.dy_sq += d * d;
  }
  return metrics;
}

void HostIterationBackend::accept_trial() {
  for (std::size_t j = 0; j < n_; ++j) x_[j] = x_trial_[j];
  for (std::size_t i = 0; i < m_; ++i) y_[i] = y_trial_[i];
}

void HostIterationBackend::fixed_step(Real tau, Real sigma) {
  matvec_.multiply_transpose(in(y_), out(kt_y_));
  for (std::size_t j = 0; j < n_; ++j) {
    x_prev_[j] = x_[j];
    const Real step = x_[j] - tau * (problem_.c[j] - kt_y_[j]);
    x_[j] = std::clamp(step, problem_.col_lower[j], problem_.col_upper[j]);
  }
  // `2x^{k+1} - x^k` is what makes this PDHG rather than Arrow-Hurwicz.
  for (std::size_t j = 0; j < n_; ++j) extrapolated_[j] = 2.0 * x_[j] - x_prev_[j];
  matvec_.multiply(in(extrapolated_), out(k_x_));
  for (std::size_t i = 0; i < m_; ++i) {
    const Real step = y_[i] + sigma * (problem_.b[i] - k_x_[i]);
    y_[i] = i < problem_.num_equality ? step : std::min(step, 0.0);
  }
}

void HostIterationBackend::snapshot_iterate() {
  for (std::size_t j = 0; j < n_; ++j) diff_x_[j] = x_[j];
  for (std::size_t i = 0; i < m_; ++i) diff_y_[i] = y_[i];
}

void HostIterationBackend::finish_difference() {
  for (std::size_t j = 0; j < n_; ++j) {
    diff_x_[j] = x_[j] - diff_x_[j];
    sum_x_[j] += x_[j];
  }
  for (std::size_t i = 0; i < m_; ++i) {
    diff_y_[i] = y_[i] - diff_y_[i];
    sum_y_[i] += y_[i];
  }
}

void HostIterationBackend::accumulate_average(Real weight) {
  for (std::size_t j = 0; j < n_; ++j) avg_x_[j] += weight * x_[j];
  for (std::size_t i = 0; i < m_; ++i) avg_y_[i] += weight * y_[i];
}

void HostIterationBackend::reset_average() {
  avg_x_.assign(0.0);
  avg_y_.assign(0.0);
}

}  // namespace sovsolve::solver::pdlp
