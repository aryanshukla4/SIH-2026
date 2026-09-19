// Module 23 (added post-v3, see module.txt): the simplex working form and the
// basis that indexes it.
//
// The canonical model (docs/FORMULATION.md section 2) is
//
//     min c'x  s.t.  A_E x = b_E,  A_I x + s = b_I, s >= 0,  l <= x <= u
//
// which is already bounded-variable, but it has no square coefficient matrix
// for a basis to be a submatrix OF: the slack block is deliberately not stored
// (Canonicalizer.cpp -- "the reduced system never needs it formed", which is
// true for the IPM and false for simplex).
//
// This header adds that block back as a VIEW rather than as data:
//
//     w = (x, xi) in R^(n+m),   Ahat = [A | I_m],   Ahat w = b
//
//       xi_i in [0, 0]      for i <  num_equality   (fixed logical)
//       xi_i in [0, +INF)   for i >= num_equality   (this IS the canonical s)
//       x_j  in [col_lower_j, col_upper_j]
//
// `I_m` is never materialized -- column `n + i` is the unit vector `e_i`, and
// `AugmentedMatrix::for_each_in_column` simply yields that one entry. The cost
// of materializing it would be m extra columns of storage plus an m-entry
// permutation of A.csc, to represent a matrix whose every entry is known.
//
// One logical per row, not one per inequality row: equality rows get a logical
// fixed at zero. That costs `num_equality` structurally-inert columns and buys
// a starting basis that is exactly the identity -- factorizable without any
// work, and a valid starting point for every instance. The alternative (a
// basis over structural columns) requires a crash procedure before the solver
// can take its first step.

#ifndef SOVSOLVE_SOLVER_SIMPLEX_BASIS_HPP
#define SOVSOLVE_SOLVER_SIMPLEX_BASIS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"

namespace sovsolve::solver::simplex {

using core::Index;
using core::Real;

/// Where a variable sits. Every variable is either basic or resting on a
/// bound; this is the whole state a simplex basis carries, and it is what
/// makes a warm start cheap -- an m-vector plus an (n+m)-vector of these,
/// against an IPM's five dense Real vectors.
///
/// `Fixed` is distinguished from `AtLower` even though the numeric value is
/// the same, because a fixed variable can never be chosen to enter the basis:
/// its ratio test result is meaningless (it cannot move) and admitting it
/// produces a pivot that changes nothing while consuming an iteration. Equality
/// logicals are the common case, one per equality row.
///
/// `Free` (both bounds infinite, resting at zero) is distinguished for the
/// opposite reason: it has no bound to rest on, so it is dual-feasible only
/// when its reduced cost is exactly zero, and dual phase 1 must give it a
/// temporary artificial bound before the iteration can proceed.
enum class VarStatus : std::uint8_t {
  Basic,
  AtLower,
  AtUpper,
  Fixed,
  Free,
};

/// The simplex working matrix `Ahat = [A | I_m]`, as a view over a
/// `CanonicalProblem` that is NOT copied and must outlive this object.
///
/// Bounds are exposed through `lower()` / `upper()` in the same w-index space
/// as the columns, so no caller ever has to remember whether index `j` means a
/// structural column or a row's logical -- getting that wrong silently swaps a
/// variable's bounds for an unrelated one's.
class AugmentedMatrix {
 public:
  explicit AugmentedMatrix(const model::CanonicalProblem& problem) noexcept
      : problem_(&problem),
        num_rows_(problem.num_rows()),
        num_structural_(problem.num_cols()) {}

  [[nodiscard]] std::size_t num_rows() const noexcept { return num_rows_; }
  [[nodiscard]] std::size_t num_structural() const noexcept { return num_structural_; }
  [[nodiscard]] std::size_t num_total() const noexcept {
    return num_structural_ + num_rows_;
  }

  [[nodiscard]] bool is_logical(std::size_t w) const noexcept {
    return w >= num_structural_;
  }
  /// Row owning logical `w`. Precondition: `is_logical(w)`.
  [[nodiscard]] std::size_t logical_row(std::size_t w) const noexcept {
    return w - num_structural_;
  }
  [[nodiscard]] std::size_t logical_of_row(std::size_t i) const noexcept {
    return num_structural_ + i;
  }

  /// Lower bound of variable `w` in the augmented space.
  [[nodiscard]] Real lower(std::size_t w) const noexcept {
    return is_logical(w) ? 0.0 : problem_->col_lower[w];
  }

  /// Upper bound of variable `w` in the augmented space.
  ///
  /// An equality row's logical is fixed at zero (`[0, 0]`); an inequality
  /// row's logical is the canonical slack `s >= 0`, unbounded above.
  [[nodiscard]] Real upper(std::size_t w) const noexcept {
    if (!is_logical(w)) return problem_->col_upper[w];
    return logical_row(w) < problem_->num_equality ? 0.0 : core::INF;
  }

  /// Objective coefficient. Logicals carry no cost -- the canonical objective
  /// is over `x` alone, and `s` appears in no term of it.
  ///
  /// Reads the override when one is set: the cost PERTURBATION of Koberstein's
  /// thesis section 6.3.1 (SolveSimplex.cpp) runs the dual simplex on slightly
  /// changed costs, and pointing the one accessor every cost read goes through
  /// at a different vector is what keeps that from needing a copy of the
  /// problem -- which branch-and-bound would otherwise pay at every node.
  [[nodiscard]] Real cost(std::size_t w) const noexcept {
    if (is_logical(w)) return 0.0;
    return costs_ != nullptr ? (*costs_)[w] : problem_->c[w];
  }

  /// Replaces the costs read by `cost()`, or restores the problem's own with
  /// `nullptr`. Length `num_structural()`; not owned.
  void set_costs(const std::vector<Real>* costs) noexcept { costs_ = costs; }

  [[nodiscard]] const model::CanonicalProblem& problem() const noexcept {
    return *problem_;
  }

  /// Visit every nonzero of column `w` as `fn(row_index, value)`.
  ///
  /// Templated on the callable rather than taking a `std::function` because
  /// this is called once per column per refactorization and once per entering
  /// column per iteration; an indirect call per NONZERO is not a cost worth
  /// paying for type erasure here.
  template <typename Fn>
  void for_each_in_column(std::size_t w, Fn&& fn) const {
    if (is_logical(w)) {
      fn(logical_row(w), 1.0);
      return;
    }
    const auto& csc = problem_->A.csc;
    for (std::size_t k = csc.slice_begin(w); k < csc.slice_end(w); ++k) {
      fn(static_cast<std::size_t>(csc.indices()[k]), csc.values()[k]);
    }
  }

  /// Number of nonzeros in column `w`, without traversing it.
  [[nodiscard]] std::size_t column_nnz(std::size_t w) const noexcept {
    return is_logical(w) ? 1u : problem_->A.csc.slice_nnz(w);
  }

 private:
  const model::CanonicalProblem* problem_;
  const std::vector<Real>* costs_ = nullptr;
  std::size_t num_rows_;
  std::size_t num_structural_;
};

/// A basis: which variables are basic, and where every nonbasic one rests.
///
/// Deliberately `std::vector`-backed and copyable, unlike `core::Vector` which
/// is move-only. Branch-and-bound copies a parent's basis into each child --
/// that is the entire point of Stage 3 -- and a move-only type would force
/// every such copy through an explicit clone() that exists only to work around
/// the container choice. The buffers are m int32s and (n+m) bytes; this is a
/// control structure, not a numerical one.
struct Basis {
  /// Status of every variable, indexed in augmented (w) space, length n+m.
  std::vector<VarStatus> status;
  /// w-index of the variable occupying each basic slot, length m.
  /// `basic[r]` is the variable whose value is row r of the FTRAN/BTRAN space.
  std::vector<Index> basic;
  /// Dual steepest edge weights `beta = ||B^-T e_r||^2` of the basic
  /// variables, indexed in augmented (w) space -- length n+m, or empty when
  /// none are known. Koberstein section 8.2.2.1 keeps them in w space "in such
  /// a way that weight i is assigned to the position corresponding to the
  /// index B(i) of the associated basic variable", which is what lets them
  /// survive a change of slot order -- and here, a primal simplex cleanup that
  /// never reads them, and a branch-and-bound child that inherits its parent's
  /// basis ("the default is to reuse the weights of the last LP-iteration").
  /// Entries of nonbasic variables are meaningless. A heuristic's state, not
  /// part of the basis's identity: `validate()` ignores it.
  std::vector<Real> dse_weights;

  [[nodiscard]] std::size_t num_rows() const noexcept { return basic.size(); }
  [[nodiscard]] std::size_t num_total() const noexcept { return status.size(); }
  [[nodiscard]] bool empty() const noexcept { return basic.empty(); }

  /// Check the invariants a factorization silently depends on: every
  /// `basic[r]` in range, marked `Basic`, and named by exactly one slot, with
  /// the `Basic` count equal to the number of slots. A basis violating any of
  /// them produces a structurally singular or wrong-dimensioned matrix, which
  /// surfaces as an unexplained factorization failure several calls away from
  /// the mistake that caused it.
  ///
  /// Not `noexcept`: it allocates a seen-marker to catch a variable listed in
  /// two slots. Counting `Basic` statuses alone cannot catch that -- slots
  /// `[5, 5]` against statuses marking 5 and 7 basic gives a matching count
  /// and a rank-deficient basis.
  [[nodiscard]] bool validate() const;
};

/// The all-logical starting basis: every row's logical is basic, every
/// structural column rests on a bound.
///
/// The basis matrix is exactly `I_m`, so it factorizes without arithmetic and
/// is never singular. Nonbasic placement is the dual-feasible one where a
/// choice exists (see `DualSimplex`'s phase 1 for the columns where it does
/// not): at lower bound when `c_j >= 0`, at upper when `c_j < 0`, since the
/// dual feasibility condition is `d_j >= 0` at lower and `d_j <= 0` at upper,
/// and `d_j == c_j` for every structural column under an all-logical basis
/// (`y = B^-T c_B = 0`).
[[nodiscard]] Basis make_logical_basis(const AugmentedMatrix& matrix);

/// Value of a nonbasic variable, given where it rests.
///
/// A `Free` nonbasic sits at zero: it has no bound to sit on, and zero is the
/// only choice that does not depend on an arbitrary scale.
[[nodiscard]] Real nonbasic_value(const AugmentedMatrix& matrix, std::size_t w,
                                  VarStatus status) noexcept;

}  // namespace sovsolve::solver::simplex

#endif  // SOVSOLVE_SOLVER_SIMPLEX_BASIS_HPP
