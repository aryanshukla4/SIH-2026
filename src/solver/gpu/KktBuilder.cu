#include "sovsolve/solver/gpu/KktBuilder.hpp"

namespace sovsolve::solver::gpu {

Status build_kkt(const CanonicalProblem& /*problem*/, const SolverState& /*state*/,
                  const analysis::MatrixAnalysis& /*mat_analysis*/, Real /*delta_p*/,
                  Real /*delta_d*/, KktSystem& out) {
  out.descriptor.type = ReductionType::LpNormalEquationsDy;
  out.descriptor.reason = "stub: reduction-path selection not implemented";
  return core::make_error(core::ErrorCode::NotImplemented,
                           "build_kkt: Theta^-1 assembly not yet written");
}

}  // namespace sovsolve::solver::gpu
