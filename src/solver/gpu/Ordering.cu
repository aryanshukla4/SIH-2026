#include "sovsolve/solver/gpu/Ordering.hpp"

namespace sovsolve::solver::gpu {

Status analyze_sparsity(const KktSystem& /*system*/,
                         std::unique_ptr<SymbolicFactorization>& out) {
  out.reset();
  return core::make_error(
      core::ErrorCode::NotImplemented,
      "analyze_sparsity: cuDSS analysis pass-through not yet wired");
}

}  // namespace sovsolve::solver::gpu
