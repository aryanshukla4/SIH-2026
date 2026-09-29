// Module 9's output: the assembled Newton system, ready for Module 12 (the
// linear solver) to factorize.
//
// Reuses SparseMatrixPair/RealVector rather than introducing a new unifying
// Matrix type. docs/spec/datatypes.txt section 3 asks for a single public Matrix type
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

  /// Length `n+m`, filled only on the `QpAugmentedKkt` path (empty
  /// otherwise): the matrix's own diagonal magnitude, block by block --
  /// `|theta_inv_j + delta_p|` for the first `n` entries, `D_s_i + delta_d`
  /// for the last `m`. A block-Jacobi preconditioner for `LinearSolver.cu`'s
  /// matrix-free `solve_minres`, built from quantities `build_kkt` (gpu/
  /// KktBuilder.cu) already computes for the matrix's own diagonal -- not a
  /// second computation, just exposed instead of being thrown away.
  RealVector precond_diag;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_KKT_SYSTEM_HPP
