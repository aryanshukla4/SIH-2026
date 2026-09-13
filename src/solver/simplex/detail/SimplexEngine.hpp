// Module 23: the revised-simplex machinery both algorithms run on.
//
// "Revised simplex" names an IMPLEMENTATION, not an algorithm: carry a
// factorized basis and answer questions about the tableau by solving with it
// (FTRAN/BTRAN), instead of carrying the tableau itself. Primal and dual
// simplex are two algorithms over that same machinery, and essentially all of
// it is common to them:
//
//     the basis and its LU                      LuFactor.hpp
//     x_B = B^-1 (b - N x_N)                    compute_primal
//     y = B^-T c_B,  d = c - Ahat' y            compute_dual
//     the pivot row  rho' Ahat,  rho = B^-T e_r compute_pivot_row
//     refactorization, repair, drift control    refactorize
//     bounds, statuses, nonbasic placement      load_true_bounds / install_basis
//
// What actually differs between them is two things: which invariant is held
// while moving, and consequently the order of the two choices.
//
//     dual   keeps DUAL feasibility, chases primal:  leaving row, then column
//     primal keeps PRIMAL feasibility, chases dual:  entering column, then row
//
// So this is a base class, not a utility namespace: the derived solver holds
// the state across its own iteration, and `refactorize()` calls back into it
// through `on_refactorized` because re-establishing "the invariant" means
// different things to the two of them. The dual re-places nonbasic columns to
// restore dual feasibility; the primal must NOT, because moving a nonbasic is
// a change to the primal point, which is the thing IT is protecting.
//
// This header is internal to src/solver/simplex, following the same pattern as
// src/io/detail.

#ifndef SOVSOLVE_SRC_SOLVER_SIMPLEX_DETAIL_SIMPLEX_ENGINE_HPP
#define SOVSOLVE_SRC_SOLVER_SIMPLEX_DETAIL_SIMPLEX_ENGINE_HPP

#include <chrono>
#include <cstddef>
#include <vector>

#include "sovsolve/core/Span.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/solver/simplex/Basis.hpp"
#include "sovsolve/solver/simplex/LuFactor.hpp"
#include "sovsolve/solver/simplex/SimplexResult.hpp"

namespace sovsolve::solver::simplex::detail {

using core::Index;
using core::Real;
using core::Status;

/// Consecutive no-progress refactorizations tolerated before the solve is
/// declared numerically stuck. Without a bound, a pivot that keeps coming back
/// unusable after a rebuild spins forever instead of reporting.
inline constexpr std::size_t kMaxStuckRefactorizations = 5;

/// Iterations between wall-clock checks. `steady_clock::now()` is not free and
/// a simplex iteration on a small model is very cheap, so checking every pivot
/// would show up in the profile of exactly the instances that need the time
/// least.
inline constexpr std::size_t kTimeCheckInterval = 64;

class SimplexEngine {
 public:
  using Clock = std::chrono::steady_clock;

  /// Virtual, though nothing in this module deletes through a base pointer:
  /// the class is already polymorphic for `on_refactorized`, so the vtable
  /// exists either way and this costs one slot to make the warning set's
  /// -Wnon-virtual-dtor requirement true for every derived solver rather than
  /// suppressing it at each one.
  virtual ~SimplexEngine() = default;

  SimplexEngine(const SimplexEngine&) = delete;
  SimplexEngine& operator=(const SimplexEngine&) = delete;

 protected:
  SimplexEngine(const model::CanonicalProblem& problem, const model::Options& options);

  // --- setup -------------------------------------------------------------

  /// Copy the model's own bounds into the working bounds and clear every
  /// artificial marker.
  void load_true_bounds();

  /// Adopt `warm_start` when it fits and validates, otherwise the all-logical
  /// basis. A warm start that does not fit is not an error: a caller handing
  /// over a neighbouring problem's basis cannot always know it still applies.
  void install_basis(const Basis* warm_start);

  /// Re-place every nonbasic variable on the bound its status names. Nonbasic
  /// variables always sit exactly on a bound, so this is idempotent -- it
  /// exists to re-derive values after the BOUNDS moved underneath them.
  void reset_nonbasic_values();

  [[nodiscard]] Real working_value(std::size_t w) const {
    switch (basis_.status[w]) {
      case VarStatus::AtLower:
      case VarStatus::Fixed:
        return lower_[w];
      case VarStatus::AtUpper:
        return upper_[w];
      case VarStatus::Free:
      case VarStatus::Basic:
        break;
    }
    return 0.0;
  }

  // --- linear algebra ----------------------------------------------------

  /// Rebuild the factorization from the original data, repairing the basis if
  /// it is singular, then recompute the primal values and the duals.
  ///
  /// This is the drift control for the whole solve: the product-form update
  /// and the incremental dual update both accumulate rounding, and everything
  /// either algorithm concludes is read off those numbers.
  [[nodiscard]] Status refactorize();

  /// Called at the end of `refactorize()`, after the values and duals are
  /// fresh and before the primal values are recomputed. `repaired` is true
  /// when the basis had to be changed to factorize at all.
  virtual void on_refactorized(bool repaired) = 0;

  /// `x_B = B^-1 (b - N x_N)`, in slot space.
  void compute_primal();

  /// `y = B^-T c_B` in row space and `d = c - Ahat' y` in w space.
  void compute_dual();

  /// Row `leaving_slot` of `B^-1 Ahat`, scattered into `arow_` with its
  /// nonzero positions listed in `arow_nz_`. `rho_` is left holding
  /// `B^-T e_r`, which the dual update needs afterwards.
  ///
  /// Unsigned: the dual simplex's `sigma` flip is the dual simplex's business.
  void compute_pivot_row(std::size_t leaving_slot);

  /// Reset `arow_`/`arow_mark_` over the recorded nonzeros only -- an O(m+n)
  /// clear per iteration would dominate a sparse row.
  void clear_pivot_row();

  /// FTRAN of `Ahat_w` into `column_`, leaving it in slot space.
  void load_and_ftran_column(std::size_t w);

  // --- bookkeeping -------------------------------------------------------

  [[nodiscard]] bool time_exhausted() const {
    if (time_limit_ <= 0.0) return false;
    const std::chrono::duration<double> elapsed = Clock::now() - start_;
    return elapsed.count() >= time_limit_;
  }

  /// Assemble the answer from the current state.
  [[nodiscard]] SimplexResult pack_result(core::SolverStatus outcome) const;

  // --- state -------------------------------------------------------------

  AugmentedMatrix matrix_;
  const model::SimplexOptions& opt_;
  double time_limit_;
  std::size_t m_;
  std::size_t n_;
  std::size_t total_;
  std::size_t max_iterations_ = 0;

  Basis basis_;
  LuFactorization lu_;

  /// Working bounds: the model's own, except where a phase 1 installed an
  /// artificial one. `artificial_[w]` is a bitmask -- 1 for a replaced lower
  /// bound, 2 for a replaced upper -- so the true bound stays recoverable and
  /// an artificial bound still active at optimality is recognizable.
  std::vector<Real> lower_;
  std::vector<Real> upper_;
  std::vector<char> artificial_;
  std::vector<Real> true_lower_;
  std::vector<Real> true_upper_;

  std::vector<Real> value_;    ///< nonbasic primal values, w space
  std::vector<Real> x_basic_;  ///< basic primal values, slot space
  std::vector<Real> y_;        ///< row duals, row space
  std::vector<Real> dj_;       ///< reduced costs, w space

  std::vector<Real> rho_;        ///< B^-T e_r, row space
  std::vector<Real> column_;     ///< scratch: row space in, slot space out
  std::vector<Real> aggregate_;  ///< accumulated multi-column shift, row space
  std::vector<Real> arow_;       ///< pivot row, w space
  std::vector<char> arow_mark_;
  std::vector<Index> arow_nz_;

  Clock::time_point start_ = Clock::now();
  std::size_t iterations_ = 0;
  std::size_t refactorizations_ = 0;
  std::size_t bound_flips_ = 0;
  std::size_t repairs_ = 0;
  std::size_t phase1_iterations_ = 0;
  bool force_refactor_ = false;
};

}  // namespace sovsolve::solver::simplex::detail

#endif  // SOVSOLVE_SRC_SOLVER_SIMPLEX_DETAIL_SIMPLEX_ENGINE_HPP
