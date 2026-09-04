// Solver tolerances and control parameters.
//
// The handoff docs mention tolerances, eta, sigma and iteration limits in prose
// across three files but define no type to hold them. In practice that means
// each module invents its own constant, and the solver's behaviour is
// distributed across a dozen unrelated literals.
//
// Defaults follow docs/FORMULATION.md sections 8-9.

#ifndef SOVSOLVE_MODEL_OPTIONS_HPP
#define SOVSOLVE_MODEL_OPTIONS_HPP

#include <cstddef>
#include <cstdint>

#include "sovsolve/core/Types.hpp"

namespace sovsolve::model {

using core::Real;

/// Convergence thresholds.
///
/// All three are **relative** and measured in the **infinity norm**.
/// Absolute tests fail in both directions: with ||b|| ~ 1e6 they never
/// terminate, and with ||b|| ~ 1e-6 they report optimality at a garbage point.
struct Tolerances {
  /// ||rp||_inf / (1 + ||b||_inf)
  Real primal_feasibility = 1e-8;
  /// ||rd||_inf / (1 + ||c||_inf)
  Real dual_feasibility = 1e-8;
  /// |c'x - b'y| / (1 + |c'x|)
  Real relative_gap = 1e-8;

  /// Matrix entries below this magnitude are treated as structural zeros at
  /// load time.
  Real matrix_zero = 1e-12;

  /// Bound violation tolerated when classifying a point as feasible.
  Real bound_violation = 1e-9;
};

/// Interior-point algorithm controls.
struct IpmOptions {
  /// Step-length safety factor, `alpha = eta * alpha_max`, 0 < eta < 1.
  /// Applied separately to the primal and dual steps -- see FORMULATION.md
  /// section 8. A single shared step length costs 20-30% more iterations.
  Real eta = 0.995;

  /// Fixed centering parameter, used only when `predictor_corrector` is off.
  Real sigma = 0.1;

  /// Mehrotra predictor-corrector. On by default: it is the difference between
  /// roughly 15-25 iterations and ~50, and the second solve reuses the
  /// factorization, so it is nearly free.
  bool predictor_corrector = true;

  /// Primal and dual regularization -- FLOORS, not the values themselves.
  ///
  /// The working deltas are derived from the magnitudes of the SCALED data and
  /// then clamped below by these; an absolute constant is meaningless when the
  /// matrix dynamic range reaches 2.6e11 before scaling, as it does on gas11.
  ///
  /// Not optional, on either path, and they do different jobs:
  ///
  ///   normal equations   Theta^-1 <- max(Theta^-1, delta_p)   then invert
  ///                      (A Theta A' + D_s + delta_d I) dy = rhs
  ///
  ///   augmented KKT      [ -(Q + Theta^-1 + delta_p I)        A'          ]
  ///                      [        A                  (D_s + delta_d I)    ]
  ///
  /// On the augmented path `Theta^-1` is never inverted, so the floor does not
  /// apply -- `delta_p` replaces it. Both remain mandatory there: `Q` is only
  /// positive SEMIdefinite, so a free column where `Q` is zero leaves the (1,1)
  /// block with a zero row; and `D_s` is zero on every EQUALITY row by
  /// construction, so the (2,2) block is singular across the whole equality
  /// block before rank deficiency in `A` is even considered. Netlib gas11 is
  /// 459 of 459 equality rows.
  ///
  /// Together they make the matrix quasi-definite, which is what guarantees an
  /// LDL' factorization for ANY symmetric permutation -- the property that lets
  /// the ordering be fixed from the sparsity pattern once and reused with no
  /// numerical pivoting.
  Real primal_regularization_floor = 1e-8;
  Real dual_regularization_floor = 1e-8;

  /// Escalation on breakdown, and decay back toward the floor on success:
  ///
  ///     breakdown    delta <- min(delta * escalation, delta_max), refactor
  ///     clean solve  delta <- max(delta / decay, floor)
  ///
  /// The decay is not optional. A pure ratchet leaves every later iteration
  /// solving a system perturbed more than it needs, and iterative refinement
  /// pays for it in extra passes. At `delta_max` with a still-failing
  /// factorization the direction is too perturbed to be a Newton step, so the
  /// solve reports NumericalError rather than escalating further.
  Real regularization_escalation = 100.0;
  Real regularization_decay = 10.0;
  Real delta_max = 1e-2;

  /// Breakdown trigger for the dense LU path: the ratio of the factored
  /// matrix's largest to smallest diagonal magnitude (the pivot growth
  /// ratio -- read off the U factor for free, no extra solve). cuSOLVER's
  /// `info` from Dgetrf only flags EXACT singularity; a quasi-definite KKT
  /// matrix with Theta^-1 dominating one diagonal entry factors "successfully"
  /// long before that, at a condition number past double precision's ~1e16
  /// noise floor -- confirmed on afiro.mps, where the augmented matrix reached
  /// cond(A) = 1.65e15 (pivot ratio in the same range) at iteration 8 while
  /// Dgetrf's `info` stayed 0 throughout, and the returned direction was
  /// numerically meaningless (magnitude ~1e11) from that point on. Above this
  /// ratio, `solve_newton_system` (PredictorCorrector.cu) treats it as
  /// breakdown: escalate and refactor, same as an exact-singular pivot.
  Real max_pivot_ratio = 1e10;

  /// Maximum iterative-refinement passes per linear solve.
  ///
  /// Refinement is measured against the UNREGULARIZED residual: the regularized
  /// system is a nearby problem, and refining against its own residual
  /// converges accurately to the wrong one.
  int max_refinement_steps = 3;
};

/// Termination limits.
struct Limits {
  std::size_t max_iterations = 200;
  double time_limit_seconds = 3600.0;

  /// Consecutive iterations without meaningful progress before declaring a
  /// stall.
  ///
  /// Consumed by the convergence checker, which on stall returns the BEST
  /// ITERATE seen -- not the last one -- with `MaxIterations` or
  /// `NotConverged`, and **never** `Infeasible`. Stagnation is evidence that
  /// this run stopped improving, not evidence about the model; a positive
  /// infeasibility verdict comes only from the canonicalizer finding an
  /// inconsistent empty row or a crossed bound pair.
  std::size_t stall_iterations = 10;
};

/// Reader behaviour for ambiguous MPS constructs.
///
/// Every field here corresponds to a real disagreement between existing
/// readers. Defaults match CPLEX/Gurobi/HiGHS. See docs/MPS-FORMAT-NOTES.md.
struct ReaderOptions {
  /// `UP` with a negative value and no prior `LO`: set the lower bound to
  /// -INF (true, the common convention) or leave it at 0, making the column
  /// infeasible (false).
  bool negative_upper_implies_free_lower = true;

  /// Log every occurrence of the above, so a discrepancy against another
  /// solver is traceable rather than mysterious.
  bool log_ambiguous_bounds = true;

  /// Sum duplicate (row, column) entries, as the format requires. Disabling
  /// this keeps the last value, which silently changes the model; the option
  /// exists only for comparison against readers that get it wrong.
  bool sum_duplicate_entries = true;

  /// Auto-detect fixed vs free format. When false, `fixed_format` decides.
  bool auto_detect_format = true;
  bool fixed_format = false;
};

/// Diagnostics verbosity.
struct LogOptions {
  enum class Level : std::uint8_t { Silent, Summary, Iteration, Debug };
  Level level = Level::Iteration;

  /// Per-iteration record keeping. Disable for clean benchmark timings.
  bool collect_history = true;
};

/// Everything, in one object.
struct Options {
  Tolerances tolerances;
  IpmOptions ipm;
  Limits limits;
  ReaderOptions reader;
  LogOptions log;
};

}  // namespace sovsolve::model

#endif  // SOVSOLVE_MODEL_OPTIONS_HPP
