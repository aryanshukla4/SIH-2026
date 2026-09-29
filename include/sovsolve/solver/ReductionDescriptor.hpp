// What the KKT builder actually built, so the linear solver and Newton
// direction recovery know what they are looking at.
//
// docs/spec/module.txt Module 9 mandates this explicitly: "never hardcode dy
// universally." The KKT builder picks one of several reductions depending on
// Q's structure and the free-column share (see KktSystem.hpp); every module
// downstream must read that choice from here rather than assuming it.

#ifndef SOVSOLVE_SOLVER_REDUCTION_DESCRIPTOR_HPP
#define SOVSOLVE_SOLVER_REDUCTION_DESCRIPTOR_HPP

#include <cstddef>
#include <string>

namespace sovsolve::solver {

/// Which linear system the KKT builder actually formed.
enum class ReductionType {
  /// LP, Q empty: eliminated to `(A Theta A' + D_s + delta_d I) dy = rhs`.
  /// Newton Direction Recovery must solve for `dy` first, then back out
  /// `dx, ds, dz, dv` from it.
  LpNormalEquationsDy,

  /// General QP, or an LP where the free-column share made the augmented
  /// path preferable: the 2x2 quasi-definite block system. `dx` and `dy`
  /// come out of the same solve; `dy` is not privileged.
  QpAugmentedKkt,

  /// QP with a diagonal (or cheaply-invertible) Q: dy-reduction is still
  /// possible, selected only when it is actually advantageous.
  QpSchurDy,
};

/// Metadata the KKT builder records alongside `KktSystem::matrix`, so later
/// modules can recover directions and duals without re-deriving the choice.
struct ReductionDescriptor {
  ReductionType type = ReductionType::LpNormalEquationsDy;

  /// Diagonal/Theta^-1 entries actually clamped to `delta_p` this iteration.
  /// Feeds Diagnostics::IterationRecord::theta_floor_activations.
  std::size_t theta_floor_activations = 0;

  /// Human-readable justification, e.g. "44% free columns -> augmented path"
  /// or "diagonal Q -> normal equations". Required by docs/spec/module.txt Module 9:
  /// the chosen reduction must be recorded, not just applied.
  std::string reason;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_REDUCTION_DESCRIPTOR_HPP
