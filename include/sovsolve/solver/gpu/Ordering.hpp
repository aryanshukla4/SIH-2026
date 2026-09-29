// Module 10. Fill-reducing ordering + symbolic factorization, computed once
// per sparsity pattern and reused every IPM iteration (the pattern is
// constant; only values change -- FORMULATION.md section 10.3).

#ifndef SOVSOLVE_SOLVER_GPU_ORDERING_HPP
#define SOVSOLVE_SOLVER_GPU_ORDERING_HPP

#include <memory>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/solver/KktSystem.hpp"

namespace sovsolve::solver::gpu {

using core::Status;

/// Opaque handle: on the GPU path this is expected to be a thin pass-through
/// to cuDSS's own analysis phase rather than a from-scratch symbolic
/// factorization (docs/spec/module.txt Module 10's explicit deferral clause), so its
/// contents are backend-defined.
class SymbolicFactorization {
 public:
  virtual ~SymbolicFactorization() = default;
};

/// STUB: always produces an empty (unimplemented) handle.
[[nodiscard]] Status analyze_sparsity(const KktSystem& system,
                                       std::unique_ptr<SymbolicFactorization>& out);

}  // namespace sovsolve::solver::gpu

#endif  // SOVSOLVE_SOLVER_GPU_ORDERING_HPP
