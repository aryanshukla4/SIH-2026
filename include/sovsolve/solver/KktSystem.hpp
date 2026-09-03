// Module 9's output: the assembled Newton system, ready for Module 12 (the
// linear solver) to factorize.
//
// Reuses SparseMatrixPair/RealVector rather than introducing a new unifying
// Matrix type. datatypes.txt section 3 asks for a single public Matrix type
// with no separate CSR/CSC classes; the codebase already deviates from that
// (see core/SparseMatrix.hpp's header comment) because CSR-only forces GPU
// atomics or a cache-hostile scatter for A'*y. A second, parallel matrix
// convention just for the KKT system would only duplicate that deviation,
// not undo it.

#ifndef SOVSOLVE_SOLVER_KKT_SYSTEM_HPP
#define SOVSOLVE_SOLVER_KKT_SYSTEM_HPP

#include "sovsolve/core/SparseMatrix.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/solver/ReductionDescriptor.hpp"

namespace sovsolve::solver {

using core::RealVector;
using core::SparseMatrixPair;

/// The linear system Module 12 must solve this iteration.
///
/// `matrix` is either the SPD normal-equations matrix
/// (`ReductionType::LpNormalEquationsDy`) or the symmetric quasi-definite
/// augmented KKT matrix (`QpAugmentedKkt`/`QpSchurDy`) -- `descriptor.type`
/// says which, and `rhs` is sized accordingly (`m` for the normal equations,
/// `n+m` for the augmented system).
struct KktSystem {
  SparseMatrixPair<> matrix;
  RealVector rhs;
  ReductionDescriptor descriptor;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_KKT_SYSTEM_HPP
