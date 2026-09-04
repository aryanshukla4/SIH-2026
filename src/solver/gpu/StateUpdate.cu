#include "sovsolve/solver/gpu/StateUpdate.hpp"

#include <cstddef>

namespace sovsolve::solver::gpu {

Status apply_step(SolverState& state) {
  if (state.dx.size() != state.x.size() || state.ds.size() != state.s.size() ||
      state.dy.size() != state.y.size() || state.dz.size() != state.z.size() ||
      state.dv.size() != state.v.size()) {
    return core::make_error(core::ErrorCode::DimensionMismatch,
                            "apply_step: a direction field's size does not match its "
                            "corresponding state field");
  }

  const auto ap = state.alpha_primal;
  const auto ad = state.alpha_dual;

  for (std::size_t j = 0; j < state.x.size(); ++j) {
    state.x[j] += ap * state.dx[j];
    state.z[j] += ad * state.dz[j];
    state.v[j] += ad * state.dv[j];
  }
  for (std::size_t k = 0; k < state.s.size(); ++k) {
    state.s[k] += ap * state.ds[k];
  }
  for (std::size_t i = 0; i < state.y.size(); ++i) {
    state.y[i] += ad * state.dy[i];
  }

  return Status::Ok();
}

}  // namespace sovsolve::solver::gpu
