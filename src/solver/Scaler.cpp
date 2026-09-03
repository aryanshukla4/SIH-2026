#include "sovsolve/solver/Scaler.hpp"

namespace sovsolve::solver {

Status scale(CanonicalProblem& /*problem*/, const Options& /*options*/,
             TransformStack& /*transforms*/) {
  return Status::Ok();
}

}  // namespace sovsolve::solver
