#include "sovsolve/solver/Regularization.hpp"

#include <algorithm>

namespace sovsolve::solver {

RegularizationController::RegularizationController(const Options& options)
    : delta_p_(options.ipm.primal_regularization_floor),
      delta_d_(options.ipm.dual_regularization_floor),
      floor_p_(options.ipm.primal_regularization_floor),
      floor_d_(options.ipm.dual_regularization_floor),
      escalation_(options.ipm.regularization_escalation),
      decay_(options.ipm.regularization_decay),
      max_(options.ipm.delta_max) {}

bool RegularizationController::escalate() {
  if (delta_p_ >= max_ && delta_d_ >= max_) return false;
  delta_p_ = std::min(delta_p_ * escalation_, max_);
  delta_d_ = std::min(delta_d_ * escalation_, max_);
  ++events_;
  return true;
}

void RegularizationController::decay() {
  delta_p_ = std::max(delta_p_ / decay_, floor_p_);
  delta_d_ = std::max(delta_d_ / decay_, floor_d_);
}

}  // namespace sovsolve::solver
