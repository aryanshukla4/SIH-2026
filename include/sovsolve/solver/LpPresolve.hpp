// LP presolve with primal AND dual postsolve.
//
// SOURCES.
//   [AA95] E. D. Andersen, K. D. Andersen, "Presolving in linear programming",
//          Math. Programming 71 (1995) 221-245. Section numbers below are its.
//   [G97]  J. Gondzio, "Presolve analysis of linear programs prior to applying
//          an interior point method", INFORMS J. Computing 9 (1997) 73-91.
//
// ---------------------------------------------------------------------------
// Why a second presolver
// ---------------------------------------------------------------------------
//
// `Presolver.hpp` (Module 4) writes its reductions onto the canonicalizer's
// TransformStack and recovers them in one pass inside `recover_solution`. That
// shape cannot express a CHAIN of reductions whose dual recovery depends on
// order -- a forcing row that fixes a column whose bound an earlier singleton
// row tightened, for example -- so it stopped at the reductions that do not
// interact. This module keeps its own stack and replays it strictly in
// reverse, the postsolve of [AA95] section 4, and works for LP only (`Q`
// empty). Module 4 stays in place for QP and for the warm-started GPU path.
//
// It sits between canonicalization and scaling:
//
//     canonicalize -> lp_presolve -> scale -> engine -> unscale
//                  -> LpPostsolve::expand -> recover_solution
//
// so the model it sees and the model it hands back are both CanonicalProblems
// (equalities first, inequalities one-sided `a'x <= b`, native bounds, no
// fixed column, no empty row), and `expand` returns a Solution in the
// pre-presolve canonical space that `recover_solution` already understands.
//
// ---------------------------------------------------------------------------
// Reductions ([AA95] section 3 numbering)
// ---------------------------------------------------------------------------
//
//   (i)    empty row                   -> removed, or PrimalInfeasible
//   (ii)   empty column                -> fixed at its cost-improving bound
//   (iv)   fixed column                -> substituted out
//   (v)    singleton row               -> equality: fixes the column;
//                                         inequality: becomes a column bound
//   (vi)   free column singleton       -> substituted out with its row
//   (vii)  doubleton equation with a column singleton
//   (viii) implied free column singleton
//   (ix)   infeasible row (activity bounds) -> PrimalInfeasible
//   (x)    forcing row                 -> every column fixed at a bound
//          redundant inequality        -> removed
//   (xi)   dominated column            -> fixed at a bound
//   (xvii) parallel (duplicate) columns -> merged
//
// Not done: duplicate rows ((xiv), a linear transformation of the duals) and
// the weakly dominated / forcing columns of (xii)-(xiii).
//
// ---------------------------------------------------------------------------
// Dual recovery
// ---------------------------------------------------------------------------
//
// Postsolve keeps a running reduced-cost vector `d = cc - A'y` over the whole
// pre-presolve matrix, where `cc` is the cost vector AS IT STOOD when each
// reduction was made. Every record that assigns a row dual updates `d` along
// that row; every record that modified costs puts them back. At the end
// `cc` is the original cost again, so `d` is the true reduced cost and the
// bound duals are its positive and negative parts. See LpPresolve.cpp.
//
// ---------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------
//
// PrimalInfeasible is returned only on a violation above `verdict_tolerance`
// relative to the row, far looser than the tolerance reductions act on, so
// that a row in between is simply left for the engine. Dual infeasibility
// (an unbounded cost direction) is never reported here: it proves only that
// the model is unbounded OR infeasible, and presolve cannot tell which.

#ifndef SOVSOLVE_SOLVER_LP_PRESOLVE_HPP
#define SOVSOLVE_SOLVER_LP_PRESOLVE_HPP

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "sovsolve/core/SparseMatrix.hpp"
#include "sovsolve/core/Status.hpp"
#include "sovsolve/core/Types.hpp"
#include "sovsolve/model/Canonical.hpp"
#include "sovsolve/model/Options.hpp"
#include "sovsolve/model/Solution.hpp"
#include "sovsolve/model/Transform.hpp"

namespace sovsolve::solver {

struct LpPresolveStats {
  std::size_t rows_before = 0, cols_before = 0, nnz_before = 0;
  std::size_t rows_after = 0, cols_after = 0, nnz_after = 0;
  std::size_t rounds = 0;

  std::size_t empty_rows = 0;
  std::size_t redundant_rows = 0;
  std::size_t singleton_rows = 0;
  std::size_t forcing_rows = 0;
  std::size_t fixed_columns = 0;
  std::size_t empty_columns = 0;
  std::size_t free_singletons = 0;
  std::size_t implied_free_singletons = 0;
  std::size_t doubletons = 0;
  std::size_t dominated_columns = 0;
  std::size_t parallel_columns = 0;
};

/// One reduction, as postsolve needs it. Flat, like TransformRecord.
struct LpPostsolveRecord {
  enum class Kind : std::uint8_t {
    FixColumn,        ///< col = value
    DropRow,          ///< row removed with dual 0 (empty or redundant)
    SingletonEq,      ///< equality singleton row; its column is fixed by a FixColumn
    SingletonIneq,    ///< inequality singleton row turned into a bound on col
    Forcing,          ///< row forced all its columns to a bound; entries = the row
    Substitute,       ///< singleton col eliminated with its row; entries = the rest
    Doubleton,        ///< singleton col eliminated with a doubleton equation
    Parallel,         ///< col merged into col2: x_col2' = x_col2 + scale * x_col
  };

  Kind kind = Kind::FixColumn;
  std::size_t row = 0;
  std::size_t col = 0;
  std::size_t col2 = 0;
  core::Real a = 0.0;       ///< a_{row,col}, or the parallel scale
  core::Real a2 = 0.0;      ///< a_{row,col2} (Doubleton)
  core::Real value = 0.0;   ///< fixed value / rhs / cost multiplier, by kind
  core::Real value2 = 0.0;  ///< cost multiplier t (Substitute/Doubleton)
  std::uint8_t flags = 0;   ///< kind-specific bits, see LpPresolve.cpp
  core::Real lo = 0.0, hi = 0.0, lo2 = 0.0, hi2 = 0.0;  ///< bounds (Parallel)
  std::size_t entries_begin = 0, entries_end = 0;      ///< into `entries`
};

/// Everything needed to map a solution of the presolved problem back to the
/// canonical problem it came from. Empty (and `expand` an identity) when
/// presolve was skipped or removed nothing.
class LpPostsolve {
 public:
  [[nodiscard]] bool active() const noexcept { return active_; }

  /// Presolved-space canonical solution -> pre-presolve canonical space.
  /// `reduced` must already be UNSCALED (see `unscale_canonical_solution`).
  [[nodiscard]] model::Solution expand(const model::Solution& reduced) const;

  /// The pre-presolve problem, for `recover_solution`.
  [[nodiscard]] const model::CanonicalProblem& original() const noexcept { return original_; }

 private:
  friend core::Status lp_presolve(model::CanonicalProblem&, const model::Options&,
                                  LpPostsolve&, LpPresolveStats*);

  bool active_ = false;
  model::CanonicalProblem original_;
  std::vector<core::Real> final_cost_;       ///< working cost at the end, every column
  std::vector<std::size_t> kept_row_;        ///< reduced row -> original row
  std::vector<std::size_t> kept_col_;        ///< reduced col -> original col
  std::vector<LpPostsolveRecord> records_;
  std::vector<std::pair<std::size_t, core::Real>> entries_;
};

/// Presolve `problem` in place. No-op (and `post.active()` false) when
/// presolve is disabled or the model has a quadratic objective.
/// Returns `PrimalInfeasible` on a proven-infeasible row.
[[nodiscard]] core::Status lp_presolve(model::CanonicalProblem& problem,
                                       const model::Options& options, LpPostsolve& post,
                                       LpPresolveStats* stats = nullptr);

/// Undo the RowScaling/ColumnScaling records of `scaling` on a canonical
/// solution: x = S x', y = R y', z = z'/S, v = v'/S. For the path where
/// scaling is recorded on its own stack rather than on the canonicalizer's.
void unscale_canonical_solution(const model::TransformStack& scaling, model::Solution& solution);

}  // namespace sovsolve::solver

#endif  // SOVSOLVE_SOLVER_LP_PRESOLVE_HPP
