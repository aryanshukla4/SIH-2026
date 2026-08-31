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

  /// Primal and dual regularization added to the reduced system.
  ///
  /// Not optional. cond(A*Theta*A') ~ 1/mu^2 by construction, so the system
  /// becomes catastrophically ill-conditioned precisely as the method
  /// converges. Regularization plus iterative refinement is the standard
  /// mitigation, and the event counts are the evidence for any robustness
  /// claim.
  Real primal_regularization = 1e-8;
  Real dual_regularization = 1e-8;

  /// Maximum iterative-refinement passes per linear solve.
  int max_refinement_steps = 3;
};

/// Termination limits.
struct Limits {
  std::size_t max_iterations = 200;
  double time_limit_seconds = 3600.0;

  /// Consecutive iterations without meaningful progress before declaring a
  /// stall. On stall the **best iterate seen** is returned, not the last one.
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
