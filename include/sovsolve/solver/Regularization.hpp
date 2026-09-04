// Module 11: escalation/decay bookkeeping ONLY. Does NOT own the Module 9
// Theta^-1/delta_p floor -- that's applied unconditionally, every iteration,
// inside the KKT builder. This module raises delta above that floor on
// breakdown and decays it back down on a clean solve.

#ifndef SOVSOLVE_SOLVER_REGULARIZATION_HPP
#define SOVSOLVE_SOLVER_REGULARIZATION_HPP

#include <algorithm>
#include <cstddef>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Options.hpp"

namespace sovsolve::solver {

using core::Real;
using model::Options;

/// Tracks delta_p/delta_d across iterations.
///
///     on breakdown:   delta <- min(delta * escalation, delta_max), refactor
///     on clean solve: delta <- max(delta / decay, floor)
///
/// At delta_max with factorization still failing, the caller must report
/// NumericalError rather than escalate further (module.txt Module 11).
///
/// Header-only (not paired with a .cpp): gpu::run_iteration (PredictorCorrector,
/// in sovsolve_solver_gpu) is the only real caller, and it must not create a
/// link dependency on sovsolve_solver -- that library already conditionally
/// links sovsolve_solver_gpu the other way when SOVSOLVE_ENABLE_CUDA is on,
/// and a cycle between the two static libraries is not something CMake/the
/// linker will sort out for a few lines of arithmetic.
class RegularizationController {
 public:
  explicit RegularizationController(const Options& options)
      : delta_p_(options.ipm.primal_regularization_floor),
        delta_d_(options.ipm.dual_regularization_floor),
        floor_p_(options.ipm.primal_regularization_floor),
        floor_d_(options.ipm.dual_regularization_floor),
        escalation_(options.ipm.regularization_escalation),
        decay_(options.ipm.regularization_decay),
        max_(options.ipm.delta_max) {}

  [[nodiscard]] Real delta_p() const noexcept { return delta_p_; }
  [[nodiscard]] Real delta_d() const noexcept { return delta_d_; }

  /// Returns false when delta is already at delta_max on both sides and
  /// cannot escalate further -- the caller should report NumericalError.
  [[nodiscard]] bool escalate() {
    if (delta_p_ >= max_ && delta_d_ >= max_) return false;
    delta_p_ = std::min(delta_p_ * escalation_, max_);
    delta_d_ = std::min(delta_d_ * escalation_, max_);
    ++events_;
    return true;
  }

  void decay() {
    delta_p_ = std::max(delta_p_ / decay_, floor_p_);
    delta_d_ = std::max(delta_d_ / decay_, floor_d_);
  }

  [[nodiscard]] std::size_t escalation_events() const noexcept { return events_; }

 private:
  Real delta_p_;
  Real delta_d_;
  Real floor_p_;
  Real floor_d_;
  Real escalation_;
  Real decay_;
  Real max_;
  std::size_t events_ = 0;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_REGULARIZATION_HPP
