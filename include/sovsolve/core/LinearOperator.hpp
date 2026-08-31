// Abstract "apply a matrix to a vector" interface.
//
// Not in the handoff docs, and cheap to add now while it is one header rather
// than an edit to every module.
//
// Three things it buys:
//
//   1. `A*Theta*A'` need never be materialized. In an IPM the normal-equations
//      operator changes its *values* every iteration but not its *structure*;
//      an iterative solver only ever needs to apply it. Materializing it is a
//      choice, and with a dense column present it is a bad one -- a single
//      dense column makes `A*Theta*A'` fully dense (see docs/FORMULATION.md
//      section 10.1).
//
//   2. Iterative linear solvers (CG on the normal equations, MINRES on the
//      augmented system) become possible without changing the callers.
//
//   3. First-order / matrix-free methods, which are the credible GPU path on
//      consumer hardware where FP64 factorization runs at 1/64 speed but
//      bandwidth-bound SpMV does not.
//
// Cost: one virtual call per apply, against millions of floating-point
// operations inside it. Immaterial.

#ifndef SOVSOLVE_CORE_LINEAR_OPERATOR_HPP
#define SOVSOLVE_CORE_LINEAR_OPERATOR_HPP

#include <cstddef>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Types.hpp"

namespace sovsolve::core {

/// An `m x n` linear map.
class LinearOperator {
 public:
  virtual ~LinearOperator() = default;

  [[nodiscard]] virtual std::size_t rows() const noexcept = 0;
  [[nodiscard]] virtual std::size_t cols() const noexcept = 0;

  /// `y <- alpha * A * x + beta * y`
  virtual void apply(Real alpha, HostSpan<const Real> x, Real beta,
                     HostSpan<Real> y) const = 0;

  /// `y <- alpha * A' * x + beta * y`
  ///
  /// Declared separately rather than derived, because the efficient
  /// implementation reads a different storage orientation -- CSC for the
  /// transpose, CSR for the forward direction. This is why `Problem` holds
  /// both (see SparseMatrix.hpp).
  virtual void apply_transpose(Real alpha, HostSpan<const Real> x, Real beta,
                               HostSpan<Real> y) const = 0;

  /// True when the operator is symmetric, so `apply_transpose == apply`.
  /// Lets callers skip a redundant traversal for `Q`.
  [[nodiscard]] virtual bool is_symmetric() const noexcept { return false; }
};

}  // namespace sovsolve::core

#endif  // SOVSOLVE_CORE_LINEAR_OPERATOR_HPP
