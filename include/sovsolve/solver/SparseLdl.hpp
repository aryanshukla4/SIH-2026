// Sparse LDL' factorization of a symmetric matrix, with the regularized pivots
// an interior-point method needs.
//
// SOURCES.
//   [AG99] A. Altman, J. Gondzio, "Regularized symmetric indefinite systems in
//          interior point methods for linear and quadratic optimization",
//          Optimization Methods and Software 11 (1999) 275-302, section 5.
//   [W99]  S. J. Wright, "Modified Cholesky factorizations in interior-point
//          algorithms for linear programming", SIAM J. Optim. 9 (1999).
//   T. A. Davis, "Direct Methods for Sparse Linear Systems" (SIAM 2006),
//   chapter 4: the elimination tree and the up-looking factorization used
//   here are the textbook ones.
//
// WHY IN-HOUSE. The interior-point normal equations `A Theta A' + D` are SPD
// on paper and singular in practice: a linearly dependent row of A (every
// network model has one) gives an exactly zero pivot, and near the optimum
// Theta spans twenty orders of magnitude. [AG99] section 5's cure acts INSIDE
// the factorization -- a pivot found too small while eliminating is corrected
// on the spot -- which a black-box Cholesky does not expose.
//
// THE PIVOT RULE. The MECHANISM is [W99]'s (section 3, algorithm modchol): a
// pivot too small to trust is skipped, "simulated, as in LIPSOL and PCx, by
// inserting a huge element in the pivot position" (section 6), so its
// component of the solution is dropped rather than amplified by 1/d_k.
// The THRESHOLD is OURS: skip when `d_k <= pivot_tolerance * a_kk`, the
// pivot's own entry before elimination -- i.e. when cancellation destroyed it.
// [W99]'s own threshold, 1e-13 times the LARGEST diagonal of the matrix, was
// measured here and rejected: this factor preconditions CG on a system whose
// diagonal spans ~1e20, so the global threshold skipped legitimate pivots and
// the GPU interior point lost 25fv47 (3 s -> 17 s), fffff800 and 80bau3b
// (Optimal -> NotConverged). [W99] analyses a factor used directly. That is the limit of [AG99]'s dynamic dual
// regularization for a pivot that is zero up to rounding; the uniform part of
// their regularization (the small R_p, R_d on every pivot) is added by the
// caller to the matrix itself, before factorizing.
//
// NUMERIC FACTORIZATION. Left-looking SUPERNODAL: consecutive columns whose
// below-diagonal patterns nest (the parent in the elimination tree is the
// next column and the column count drops by exactly one) form a supernode,
// assembled and factored as a dense block, with every update from an earlier
// supernode applied as a dense block too -- the organisation of the
// supernodal Cholesky in [AA00] section 1.5.2 and of CHOLMOD. Written from
// the standard method, not transcribed; the result is checked against the
// column-by-column factor it replaced (same fill, same pivots skipped).
//
// ORDERING. Approximate minimum degree on the quotient graph (SparseLdl.cpp),
// written from general knowledge of Amestoy, Davis & Duff (1996) rather than
// transcribed, and checked against CHOLMOD's fill. Deterministic run to run.

#ifndef SOVSOLVE_SOLVER_SPARSE_LDL_HPP
#define SOVSOLVE_SOLVER_SPARSE_LDL_HPP

#include <cstddef>
#include <vector>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"

namespace sovsolve::solver {

using core::Real;

class SparseLdl {
 public:
  /// Symbolic analysis of an n x n symmetric pattern given in FULL compressed
  /// column form (both triangles, diagonal present). Computes the ordering,
  /// the elimination tree and the column counts of L. Reusable for every
  /// matrix with this pattern.
  [[nodiscard]] core::Status analyze(std::size_t n, const std::vector<std::size_t>& col_ptr,
                                     const std::vector<std::size_t>& row_idx);

  /// Numeric factorization of values aligned with the analysed pattern.
  [[nodiscard]] core::Status factorize(const std::vector<Real>& values, Real pivot_tolerance);

  /// Solve (L D L') x = b in place, in the ORIGINAL (unpermuted) numbering.
  void solve(std::vector<Real>& x) const;

  [[nodiscard]] std::size_t size() const noexcept { return n_; }
  [[nodiscard]] std::size_t nnz_l() const noexcept { return li_.size(); }
  /// Pivots replaced in the last factorization.
  [[nodiscard]] std::size_t modified_pivots() const noexcept { return modified_; }

 private:
  std::size_t n_ = 0;
  std::vector<std::size_t> col_ptr_, row_idx_;  ///< the analysed pattern
  std::vector<std::size_t> perm_, pinv_;        ///< new -> old, old -> new
  std::vector<std::ptrdiff_t> parent_;          ///< elimination tree (permuted)
  std::vector<std::size_t> lp_;                 ///< column pointers of L
  std::vector<std::size_t> li_;                 ///< row indices of L
  std::vector<Real> lx_, d_;
  std::size_t modified_ = 0;

  // Supernodes: columns [super_[s], super_[s+1]) share one below-diagonal
  // pattern and are factored as one dense block (column-major, rows =
  // {first column} u li_ of the first column) at blk_off_[s] in blk_.
  std::vector<std::size_t> super_, super_of_, blk_off_;
  std::vector<Real> blk_;

  // Factorization workspace, kept to avoid reallocating each iteration.
  mutable std::vector<Real> work_;
  std::vector<std::size_t> flag_, lnz_, relmap_, next_row_;
  std::vector<Real> diag_, update_;
  std::vector<std::vector<std::size_t>> pending_;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_SPARSE_LDL_HPP
