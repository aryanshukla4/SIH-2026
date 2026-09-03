// Module 11: escalation/decay bookkeeping ONLY. Does NOT own the Module 9
// Theta^-1/delta_p floor -- that's applied unconditionally, every iteration,
// inside the KKT builder. This module raises delta above that floor on
// breakdown and decays it back down on a clean solve.

#ifndef SOVSOLVE_SOLVER_REGULARIZATION_HPP
#define SOVSOLVE_SOLVER_REGULARIZATION_HPP

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
class RegularizationController {
 public:
  explicit RegularizationController(const Options& options);

  [[nodiscard]] Real delta_p() const noexcept { return delta_p_; }
  [[nodiscard]] Real delta_d() const noexcept { return delta_d_; }

  /// Returns false when delta is already at delta_max on both sides and
  /// cannot escalate further -- the caller should report NumericalError.
  [[nodiscard]] bool escalate();
  void decay();

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
