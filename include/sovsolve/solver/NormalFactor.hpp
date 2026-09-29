// Direct factorization of the interior-point normal equations
//
//     M = A_S Theta A_S' + D,      D = D_s + r_d I   (diagonal)
//
// used as the PRECONDITIONER of the CG solve in HostKkt.cpp. CG then solves
// the full system (every column, including the dense ones left out of M) and
// converges in a handful of iterations -- which is also iterative refinement
// of the factored solve, for free.
//
// SOURCES.
//   [AG99] A. Altman, J. Gondzio, "Regularized symmetric indefinite systems in
//          interior point methods for linear and quadratic optimization",
//          Optim. Methods Softw. 11 (1999), section 5: the proximal-point
//          regularization R_p, R_d; defaults r_p = eps^(3/4), r_d = eps^(1/2);
//          regularization multiplied by 10 when a solve needs it.
//   [AA95] section 6 / [G97]: dense columns make A Theta A' dense, so they are
//          kept OUT of the factor; the CG iteration around it absorbs them in
//          about as many extra iterations as there are dense columns.
//
// BACKENDS. CHOLMOD's supernodal Cholesky when the build found SuiteSparse
// (SOVSOLVE_HAVE_CHOLMOD) -- a linear-algebra library, not an LP solver --
// otherwise the in-house SparseLdl. Both see the same matrix; the symbolic
// analysis (ordering, elimination tree) is done once and reused every
// interior-point iteration, since the pattern of M never changes.

#ifndef SOVSOLVE_SOLVER_NORMAL_FACTOR_HPP
#define SOVSOLVE_SOLVER_NORMAL_FACTOR_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/core/Vector.hpp"
#include "sovsolve/model/Canonical.hpp"

namespace sovsolve::solver {

using core::Real;

class NormalFactor {
 public:
  explicit NormalFactor(const model::CanonicalProblem& problem);
  ~NormalFactor();
  NormalFactor(const NormalFactor&) = delete;
  NormalFactor& operator=(const NormalFactor&) = delete;

  /// Factor A_S diag(theta) A_S' + diag(diag_add). Fails (and leaves the
  /// factor unusable) only when the backend reports a breakdown the pivot
  /// rule could not absorb; the caller raises the regularization and retries.
  [[nodiscard]] core::Status factorize(const core::RealVector& theta,
                                       const core::RealVector& diag_add);

  /// x <- M^-1 x.
  void solve(std::vector<Real>& x) const;

  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] std::size_t dense_columns() const noexcept { return num_dense_; }
  [[nodiscard]] std::size_t nnz_l() const noexcept;
  [[nodiscard]] std::size_t modified_pivots() const noexcept;
  [[nodiscard]] const char* backend() const noexcept;

  /// Columns treated as dense, flagged per column.
  [[nodiscard]] const std::vector<char>& dense() const noexcept { return is_dense_; }

 private:
  struct Backend;

  const model::CanonicalProblem* problem_;
  std::size_t m_ = 0;
  std::vector<char> is_dense_;
  std::size_t num_dense_ = 0;
  // Full symmetric pattern of M, compressed by column (== by row).
  std::vector<std::size_t> col_ptr_, row_idx_;
  std::vector<Real> values_, work_;
  std::unique_ptr<Backend> backend_;
  bool ready_ = false;
};

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_NORMAL_FACTOR_HPP
