// Module 23: sparse LU factorization of the basis, with FTRAN/BTRAN solves and
// a product-form update.
//
// Derived from the definition of Gaussian elimination with a Markowitz pivot
// order (Markowitz 1957) and threshold partial pivoting; the product-form
// update is Dantzig's, stated directly from `B_new = B_old * E`. Nothing here
// is read from or ported out of another solver -- see docs/HIGHS-COMPARISON.md
// for why that constraint exists and how it is checked.
//
// --------------------------------------------------------------------------
// Index spaces -- getting these backwards is the bug this comment exists to
// prevent
// --------------------------------------------------------------------------
//
// The basis matrix `B` is m x m. Its ROWS are canonical constraint rows. Its
// COLUMNS are basic SLOTS: column r of B is `Ahat` column `basis.basic[r]`.
// Those are two different numberings of `0..m-1` and they are NOT
// interchangeable.
//
//     ftran(v):  v enters in ROW space,  leaves in SLOT space   (B d = v)
//     btran(v):  v enters in SLOT space, leaves in ROW space    (B' rho = v)
//
// Both are what the dual simplex actually needs: the entering column
// `Ahat_q` is a row-space vector and `B^-1 Ahat_q` indexes basic slots, while
// the leaving row's unit vector `e_r` picks a basic SLOT and `B^-T e_r` is
// dotted against `Ahat`'s columns, which are row-space.
//
// --------------------------------------------------------------------------
// Factored form
// --------------------------------------------------------------------------
//
// Elimination step `k` picks pivot `(pivot_row[k], pivot_col[k])` and stores
//
//     L_k : the multipliers applied to rows below the pivot
//     U_k : the pivot row's surviving entries, over columns of rank > k
//
// so that `B^-1 v = U_op(L_op(v))` with
//
//     L_op: for k = 0..m-1:  for (i, mult) in L_k:  v[i] -= mult * v[p_k]
//     U_op: for k = m-1..0:  d[q_k] = (v[p_k] - sum U_k) / u_diag[k]
//
// and therefore `B = L U`, which is what makes the transposed solves below
// `L^-T (U^-T v)` rather than the other order.

#ifndef SOVSOLVE_SOLVER_SIMPLEX_LU_FACTOR_HPP
#define SOVSOLVE_SOLVER_SIMPLEX_LU_FACTOR_HPP

#include <cstddef>
#include <vector>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/solver/simplex/Basis.hpp"

namespace sovsolve::solver::simplex {

using core::Index;
using core::Real;
using core::Status;

/// Entries whose magnitude falls below this during elimination are dropped
/// rather than stored. The canonical model is geometric-mean scaled before the
/// solver sees it (Scaler.hpp), so an absolute floor here is meaningful: a
/// coefficient this small against scaled data is fill-in noise, and keeping it
/// costs a nonzero in every subsequent update that touches it.
inline constexpr Real kLuDropTolerance = 1e-14;

/// Columns examined per pivot search before taking the best candidate found.
///
/// The full Markowitz criterion minimizes `(r_i - 1)(c_j - 1)` over every
/// remaining entry, which is quadratic per step and is not what any practical
/// implementation does. Scanning columns in increasing-count order and
/// stopping after a few candidates gets nearly the same fill for a bounded
/// cost -- and a count-1 column has cost 0, so the all-logical starting basis
/// (every column a singleton) is factorized in one pass with no search at all.
inline constexpr std::size_t kMarkowitzCandidates = 4;

/// Smallest product-form pivot the update will accept.
///
/// Independent of the factorization's `pivot_tolerance`, which is relative to
/// a column's largest remaining entry: this one bounds absolute growth in the
/// eta file, where a tiny pivot multiplies every later solve's rounding error
/// by its reciprocal. Below this the caller refactorizes, which recomputes the
/// column from the original data rather than compounding the existing factor.
inline constexpr Real kMinEtaPivot = 1e-11;

class LuFactorization {
 public:
  /// Factorize the basis matrix implied by `basis` over `matrix`.
  ///
  /// `pivot_tolerance` is the threshold-pivoting factor: an entry is eligible
  /// only if its magnitude is at least this fraction of the largest remaining
  /// magnitude in its column. Lower values admit sparser pivots and a less
  /// stable factor; 0.1 is the usual compromise and the default in
  /// `SimplexOptions`.
  ///
  /// Returns `ErrorCode::NumericalError` when the basis is structurally or
  /// numerically singular. That is a real, expected outcome (a bound-tightened
  /// branch-and-bound node can produce one), not a malfunction -- the caller
  /// repairs the basis and retries rather than failing the solve.
  [[nodiscard]] Status factorize(const AugmentedMatrix& matrix, const Basis& basis,
                                 Real pivot_tolerance);

  /// Factorize, repairing `basis` in place if it turns out to be singular.
  ///
  /// A singular basis is not a hypothetical. Two routine things produce one:
  /// accumulated product-form drift, which lets the ratio test accept a pivot
  /// that the exact arithmetic says is zero (measured on Netlib `greenbea`,
  /// where every refactorization interval reached the same dead end); and a
  /// warm start carried across a bound change, which Stage 3 does on every
  /// branch-and-bound node.
  ///
  /// The repair is exact, not a heuristic. When elimination stalls with rows
  /// `R` and slots `C` still active, pick any `p` in `R` and put row `p`'s
  /// LOGICAL into one of those slots. That logical cannot already be basic:
  /// its column is the unit vector `e_p`, so if it were basic its slot would
  /// be a singleton in the active submatrix, Markowitz scores a singleton at
  /// cost zero, and it would have been pivoted already -- which would have
  /// made row `p` inactive. So the substitution always introduces a genuinely
  /// new column, and that column is a unit vector, which pivots immediately.
  ///
  /// `repairs`, when non-null, receives the number of columns replaced. A
  /// nonzero count is worth surfacing: it means the caller's basis was not the
  /// one it thought it had.
  [[nodiscard]] Status factorize_repairing(const AugmentedMatrix& matrix, Basis& basis,
                                           Real pivot_tolerance,
                                           std::size_t* repairs = nullptr);

  /// Solve `B d = v` in place. `v` enters in row space, leaves in slot space.
  void ftran(core::HostSpan<Real> v) const;

  /// Solve `B' rho = v` in place. `v` enters in slot space, leaves in row
  /// space.
  void btran(core::HostSpan<Real> v) const;

  /// Replace the variable in `leaving_slot` with one whose FTRAN'd column is
  /// `entering_ftran` (slot space, i.e. the output of `ftran` on the entering
  /// column), by appending a product-form eta.
  ///
  /// `B_new = B_old * E` where `E` is the identity with column `leaving_slot`
  /// replaced by `entering_ftran`, so `B_new^-1 = E^-1 B_old^-1` -- the eta
  /// composes onto the existing factor without touching it.
  ///
  /// Returns `NumericalError` when the pivot element `entering_ftran[
  /// leaving_slot]` is too small to divide by; the caller must refactorize
  /// (which recomputes the column exactly) rather than proceed on a basis
  /// whose inverse it can no longer apply.
  [[nodiscard]] Status update(std::size_t leaving_slot,
                              core::HostSpan<const Real> entering_ftran);

  /// Product-form etas appended since the last `factorize`. The caller
  /// refactorizes on a count threshold: eta application cost grows linearly
  /// with this, and so does accumulated rounding error.
  [[nodiscard]] std::size_t num_updates() const noexcept { return eta_start_.size() - 1; }

  /// Nonzeros held by the triangular factors, excluding the eta file.
  [[nodiscard]] std::size_t factor_nnz() const noexcept {
    return l_row_.size() + u_index_.size() + u_diag_.size();
  }

  /// Nonzeros held by the eta file.
  [[nodiscard]] std::size_t eta_nnz() const noexcept { return eta_index_.size(); }

  [[nodiscard]] std::size_t dimension() const noexcept { return dim_; }

  /// True once `factorize` has succeeded. A zero-row basis factorizes
  /// successfully and is still valid, so this cannot be inferred from `dim_`
  /// or from the factor sizes.
  [[nodiscard]] bool valid() const noexcept { return factorized_; }

  void clear();

 private:
  /// The factorization proper. When elimination stalls and the out-params are
  /// given, they receive a row and a slot that were still active at the stall
  /// -- the information `factorize_repairing` needs to choose a substitution,
  /// and the reason the repair loop does not have to guess.
  [[nodiscard]] Status factorize_impl(const AugmentedMatrix& matrix, const Basis& basis,
                                      Real pivot_tolerance, std::size_t* stall_row,
                                      std::size_t* stall_slot);

  void apply_etas_forward(core::HostSpan<Real> v) const;
  void apply_etas_reverse(core::HostSpan<Real> v) const;

  std::size_t dim_ = 0;
  bool factorized_ = false;

  // Pivot sequence. `pivot_row_[k]` is a row index, `pivot_col_[k]` a slot.
  std::vector<Index> pivot_row_;
  std::vector<Index> pivot_col_;

  // L, step-major: multipliers applied to rows below pivot k.
  std::vector<Index> l_start_;  // dim_ + 1
  std::vector<Index> l_row_;
  std::vector<Real> l_value_;

  // U, step-major: pivot row k's entries over columns of rank > k.
  std::vector<Index> u_start_;  // dim_ + 1
  std::vector<Index> u_index_;  // slot indices
  std::vector<Real> u_value_;
  std::vector<Real> u_diag_;  // dim_

  // Product-form eta file, in creation order. Eta t replaces slot
  // `eta_slot_[t]` with the column stored in `[eta_start_[t], eta_start_[t+1])`.
  std::vector<Index> eta_slot_;
  std::vector<Index> eta_start_{0};
  std::vector<Index> eta_index_;  // slot indices
  std::vector<Real> eta_value_;
  std::vector<Real> eta_pivot_;  // the eta's element at its own slot

  /// Scratch for the solves. `ftran`/`btran` permute between two index spaces,
  /// so they cannot run in place on the caller's buffer: back-substitution
  /// writes `d[q_k]` while a later step still needs to read `v[p_j]`, and
  /// `q_k == p_j` is perfectly possible. Held as a member so a solve does not
  /// allocate -- there are two per simplex iteration.
  ///
  /// This makes a single `LuFactorization` non-reentrant, which is correct for
  /// how it is used: one factorization belongs to one simplex instance.
  mutable std::vector<Real> work_;
};

}  // namespace sovsolve::solver::simplex

#endif  // SOVSOLVE_SOLVER_SIMPLEX_LU_FACTOR_HPP
