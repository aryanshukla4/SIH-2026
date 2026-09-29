#include "SimplexEngine.hpp"

#include <algorithm>
#include <cstddef>

namespace sovsolve::solver::simplex::detail {

using core::is_finite_bound;

SimplexEngine::SimplexEngine(const model::CanonicalProblem& problem,
                             const model::Options& options,
                             const std::vector<Real>* costs)
    : matrix_(problem),
      opt_(options.simplex),
      cancel_(options.cancel),
      time_limit_(options.limits.time_limit_seconds),
      m_(problem.num_rows()),
      n_(problem.num_cols()),
      total_(problem.num_cols() + problem.num_rows()) {
  matrix_.set_costs(costs);
  lower_.assign(total_, 0.0);
  upper_.assign(total_, 0.0);
  value_.assign(total_, 0.0);
  artificial_.assign(total_, 0);
  x_basic_.assign(m_, 0.0);
  y_.assign(m_, 0.0);
  dj_.assign(total_, 0.0);
  rho_.assign(m_, 0.0);
  column_.assign(m_, 0.0);
  aggregate_.assign(m_, 0.0);
  arow_.assign(total_, 0.0);
  arow_mark_.assign(total_, 0);
  max_iterations_ =
      opt_.max_iterations != 0 ? opt_.max_iterations : 50 * (m_ + n_) + 1000;
}

void SimplexEngine::load_true_bounds() {
  for (std::size_t w = 0; w < total_; ++w) {
    lower_[w] = matrix_.lower(w);
    upper_[w] = matrix_.upper(w);
  }
  true_lower_ = lower_;
  true_upper_ = upper_;
  std::fill(artificial_.begin(), artificial_.end(), static_cast<char>(0));
}

void SimplexEngine::reset_nonbasic_values() {
  for (std::size_t w = 0; w < total_; ++w) {
    if (basis_.status[w] == VarStatus::Basic) continue;
    value_[w] = working_value(w);
  }
}

void SimplexEngine::install_basis(const Basis* warm_start) {
  if (warm_start != nullptr && warm_start->status.size() == total_ &&
      warm_start->basic.size() == m_ && warm_start->validate()) {
    basis_ = *warm_start;
  } else {
    basis_ = make_logical_basis(matrix_);
  }
  reset_nonbasic_values();
}

Status SimplexEngine::refactorize() {
  std::size_t repairs = 0;
  const Status status =
      lu_.factorize_repairing(matrix_, basis_, opt_.pivot_tolerance, &repairs);
  if (!status.ok()) return status;
  ++refactorizations_;
  force_refactor_ = false;

  if (repairs > 0) {
    // A repair swapped columns in and out of the basis, and the displaced
    // column's status was chosen from the MODEL's bounds by a class that
    // cannot see a phase 1's artificial ones. Re-place it against the WORKING
    // bounds before anything reads its value.
    repairs_ += repairs;
    for (std::size_t w = 0; w < total_; ++w) {
      if (basis_.status[w] == VarStatus::Basic) continue;
      const bool lo = is_finite_bound(lower_[w]);
      const bool up = is_finite_bound(upper_[w]);
      if (basis_.status[w] == VarStatus::AtLower && !lo) {
        basis_.status[w] = up ? VarStatus::AtUpper : VarStatus::Free;
      } else if (basis_.status[w] == VarStatus::AtUpper && !up) {
        basis_.status[w] = lo ? VarStatus::AtLower : VarStatus::Free;
      } else if (basis_.status[w] == VarStatus::Free && (lo || up)) {
        basis_.status[w] = lo ? VarStatus::AtLower : VarStatus::AtUpper;
      }
    }
  }

  if (repairs > 0 || !preserve_nonbasic_values_) reset_nonbasic_values();
  compute_dual();
  on_refactorized(repairs > 0);
  compute_primal();
  return Status::Ok();
}

void SimplexEngine::compute_primal() {
  if (m_ == 0) return;
  for (std::size_t i = 0; i < m_; ++i) x_basic_[i] = matrix_.problem().b[i];
  for (std::size_t w = 0; w < total_; ++w) {
    if (basis_.status[w] == VarStatus::Basic) continue;
    const Real v = value_[w];
    if (v == 0.0) continue;
    matrix_.for_each_in_column(w, [&](std::size_t i, Real a) { x_basic_[i] -= a * v; });
  }
  lu_.ftran(core::HostSpan<Real>(x_basic_.data(), x_basic_.size()));
}

void SimplexEngine::compute_dual() {
  for (std::size_t r = 0; r < m_; ++r) {
    y_[r] = matrix_.cost(static_cast<std::size_t>(basis_.basic[r]));
  }
  if (m_ > 0) lu_.btran(core::HostSpan<Real>(y_.data(), y_.size()));

  for (std::size_t w = 0; w < total_; ++w) {
    if (basis_.status[w] == VarStatus::Basic) {
      dj_[w] = 0.0;
      continue;
    }
    Real dot = 0.0;
    matrix_.for_each_in_column(w, [&](std::size_t i, Real a) { dot += a * y_[i]; });
    dj_[w] = matrix_.cost(w) - dot;
  }
}

void SimplexEngine::compute_pivot_row(std::size_t leaving_slot) {
  for (std::size_t i = 0; i < m_; ++i) rho_[i] = 0.0;
  rho_[leaving_slot] = 1.0;
  lu_.btran(core::HostSpan<Real>(rho_.data(), rho_.size()));

  arow_nz_.clear();
  const auto& csr = matrix_.problem().A.csr;
  for (std::size_t i = 0; i < m_; ++i) {
    const Real weight = rho_[i];
    if (weight == 0.0) continue;

    for (std::size_t k = csr.slice_begin(i); k < csr.slice_end(i); ++k) {
      const auto j = static_cast<std::size_t>(csr.indices()[k]);
      if (arow_mark_[j] == 0) {
        arow_mark_[j] = 1;
        arow_[j] = 0.0;
        arow_nz_.push_back(static_cast<Index>(j));
      }
      arow_[j] += weight * csr.values()[k];
    }

    // Row i's logical is the unit column e_i, so its pivot-row entry is rho[i]
    // exactly -- no traversal, and no materialized identity block.
    const std::size_t logical = matrix_.logical_of_row(i);
    if (arow_mark_[logical] == 0) {
      arow_mark_[logical] = 1;
      arow_[logical] = 0.0;
      arow_nz_.push_back(static_cast<Index>(logical));
    }
    arow_[logical] += weight;
  }
}

void SimplexEngine::clear_pivot_row() {
  for (const Index j : arow_nz_) {
    const auto w = static_cast<std::size_t>(j);
    arow_mark_[w] = 0;
    arow_[w] = 0.0;
  }
  arow_nz_.clear();
}

void SimplexEngine::load_and_ftran_column(std::size_t w) {
  for (std::size_t i = 0; i < m_; ++i) column_[i] = 0.0;
  matrix_.for_each_in_column(w, [&](std::size_t i, Real a) { column_[i] = a; });
  if (m_ > 0) lu_.ftran(core::HostSpan<Real>(column_.data(), column_.size()));
}

SimplexResult SimplexEngine::pack_result(core::SolverStatus outcome) const {
  SimplexResult result;
  result.status = outcome;
  result.basis = basis_;
  result.x.assign(total_, 0.0);
  for (std::size_t w = 0; w < total_; ++w) result.x[w] = value_[w];
  for (std::size_t r = 0; r < m_; ++r) {
    result.x[static_cast<std::size_t>(basis_.basic[r])] = x_basic_[r];
  }
  result.y.assign(y_.begin(), y_.end());
  result.reduced_cost.assign(dj_.begin(), dj_.end());

  Real objective = 0.0;
  for (std::size_t j = 0; j < n_; ++j) objective += matrix_.cost(j) * result.x[j];
  result.objective = objective;

  result.iterations = iterations_;
  result.refactorizations = refactorizations_;
  result.basis_repairs = repairs_;
  result.bound_flips = bound_flips_;
  result.phase1_iterations = phase1_iterations_;
  return result;
}

}  // namespace sovsolve::solver::simplex::detail
