// The six Newton-system residuals. See docs/FORMULATION.md section 7 for the
// derivation -- this is just the type that holds them.

#ifndef SOVSOLVE_SOLVER_RESIDUALS_HPP
#define SOVSOLVE_SOLVER_RESIDUALS_HPP

#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"

namespace sovsolve::solver {

using core::Real;
using core::RealVector;

/// rp_E = A_E x - b_E            rd   = Qx + c - A'y - z + v
/// rp_I = A_I x + s - b_I        rxz  = (x-l).*z  - mu*1
/// rsy  = -s.*y_I    - mu*1      ruv  = (u-x).*v  - mu*1
///
/// A pair is omitted wherever the corresponding bound is infinite --
/// counting it unconditionally understates mu on a model with many free
/// columns (gas11 is 44% free). See SolutionQuality::complementarity in
/// Solution.hpp for the same rule applied to the reported quality measure.
struct Residuals {
  RealVector rp;   ///< length m: rp_E then rp_I, one vector (rows are one block)
  RealVector rd;   ///< length n
  RealVector rxz;  ///< length n, zero where col_lower is infinite
  RealVector ruv;  ///< length n, zero where col_upper is infinite
  RealVector rsy;  ///< length m_I

  Real rp_inf = 0.0;
  Real rd_inf = 0.0;
  Real complementarity_inf = 0.0;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_RESIDUALS_HPP
