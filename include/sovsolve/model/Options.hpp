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
  /// |c'x - b'y|, checked as an OR alongside relative_gap -- when the true
  /// optimum is at/near zero, relative_gap's `1 + |c'x|` denominator
  /// collapses to ~1 and the "relative" test silently becomes an absolute
  /// one anyway, capped at whatever noise floor the KKT solve's own
  /// regularization leaves in mu (confirmed on markshare_4_0: primal/dual
  /// residuals both < 1e-12, mu frozen at 2.4e-10, yet relative_gap sat at
  /// 1.5e-8 -- just over the default tol -- and never moved regardless of
  /// pfloor/dfloor). Every production solver (HiGHS, CPLEX, Gurobi) pairs a
  /// relative gap tolerance with an absolute one for exactly this reason.
  Real absolute_gap = 1e-8;

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

  /// Use the normal-equations reduction (FORMULATION.md 10.1,
  /// `ReductionType::LpNormalEquationsDy`, solved by matrix-free `solve_spd_cg`
  /// in LinearSolver.cu) instead of the augmented KKT system (solved by
  /// matrix-free `solve_minres`) for LP (`Q` empty) -- QP always uses the
  /// augmented path regardless of this flag, since the reduction is only
  /// valid for `Q = 0`. Neither production path factors anything anymore
  /// (see `cg_tolerance`/`minres_tolerance` below); this flag chooses which
  /// SYSTEM to solve, not whether to factor it.
  ///
  /// The reduced system is `m x m` instead of `(n+m) x (n+m)` and SPD, so it
  /// needs only plain CG rather than MINRES. It also squares the system's
  /// conditioning relative to the augmented path (`cond(A T A') ~ 1/mu^2` vs.
  /// the quasi-definite augmented system's better-behaved conditioning near
  /// convergence) -- a real, known tradeoff, confirmed on `gas11` (already
  /// unbounded/undetected before any of this, but measurably worse under
  /// this reduction) -- which is why this is not an unconditional win.
  ///
  /// Defaults ON (true) despite that, on MEASURED evidence: `solve_minres`
  /// (the augmented path) is correct but, with only a diagonal (Jacobi)
  /// preconditioner, needs thousands of iterations to converge on
  /// indefinite systems at ordinary problem sizes -- confirmed on this
  /// corpus (`egout` alone went from 0.3s to 24s under the augmented path
  /// once `minres_max_iterations` was raised enough to stop it returning
  /// silently wrong answers; several other instances did not finish within
  /// a minute). `solve_spd_cg`'s SPD reduction converges fast under the same
  /// Jacobi preconditioner because CG on an SPD system is simply a much
  /// easier problem than MINRES on an indefinite one. Until the augmented
  /// path has a real (non-diagonal) preconditioner, defaulting to the
  /// normal-equations path is the faster AND more reliable choice for the
  /// LP case it applies to -- QP is unaffected either way, since it always
  /// uses the augmented path regardless of this flag.
  bool use_normal_equations = true;

  /// Relative-residual stopping tolerance for the matrix-free Conjugate
  /// Gradient solve (`solve_spd_cg`, normal-equations path only). Tighter
  /// than the outer `Tolerances::primal_feasibility`/`dual_feasibility`
  /// (1e-8) because CG's own solve error propagates directly into the Newton
  /// direction on top of whatever error the outer IPM iteration already
  /// tolerates -- solving the linear system to the SAME tolerance as the
  /// outer loop would add a comparable second source of error on top of it.
  Real cg_tolerance = 1e-10;

  /// Safety cap on Conjugate Gradient iterations per linear solve, so a
  /// poorly preconditioned system (see the Jacobi-preconditioner doc comment
  /// on `solve_spd_cg`) degrades to a slow-but-bounded solve rather than an
  /// unbounded loop. Hitting this cap without reaching `cg_tolerance` is
  /// reported via `LinearSolveResult::pivot_ratio` exceeding
  /// `max_pivot_ratio` above -- the existing escalate/refactor loop
  /// (`PredictorCorrector.cu`) already knows what to do with that signal.
  int cg_max_iterations = 500;

  /// Same role as `cg_tolerance`, for the augmented path's matrix-free MINRES
  /// solve (`solve_minres`) -- see that function's doc comment.
  Real minres_tolerance = 1e-10;

  /// Same role as `cg_max_iterations`, for `solve_minres` -- but MEASURED
  /// (not guessed) to need a much higher default than CG's. The indefinite
  /// augmented system is materially harder for a plain diagonal (Jacobi)
  /// preconditioner than the SPD normal-equations system CG solves: at the
  /// original default of 500, `solve_minres` was hitting the cap without
  /// converging on ordinary-sized instances (confirmed on `adlittle.mps`),
  /// and "accept the direction anyway" -- correct for a merely-noisy-but-
  /// exact LU/Cholesky factorization -- is NOT safe for a Krylov solve that
  /// stopped without actually solving the system: unlike a factorization,
  /// a capped-out Krylov iterate is not even approximately a solution, so
  /// feeding it into the Newton step corrupts the whole IPM trajectory
  /// (confirmed: adlittle's objective came back with the wrong SIGN at
  /// iteration cap 500, and matched the correct answer at cap 5000). A
  /// better (non-diagonal) preconditioner is the real fix and is not
  /// attempted this pass -- see LinearSolver.cu's solve_minres doc comment;
  /// this default is a measured, honest stopgap, not a tuned optimum.
  int minres_max_iterations = 5000;
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

/// Module 4 controls. Presolve is safe and reversible by construction (see
/// solver/Presolver.hpp), but module.txt is explicit that startability
/// (the canonicalizer's contract) "must not depend on the Presolver, which
/// can be switched off" -- this is that switch.
struct PresolveOptions {
  bool enabled = true;
};

/// Module 22 controls: branch-and-bound over the existing LP/QP relaxation
/// solver (solve_problem, Solve.hpp), activated automatically whenever
/// Problem::has_discrete() is true -- see solver/gpu/BranchAndBound.hpp.
struct MilpOptions {
  /// A column's relaxation value counts as integral once it is within this
  /// distance of the nearest integer.
  Real integer_tolerance = 1e-6;

  /// Search stops once this many nodes have been explored, reporting the
  /// best incumbent found so far (status NotConverged, never Infeasible --
  /// same "stagnation is not a verdict about the model" principle Module 18
  /// applies to the continuous solver's own stall detection) rather than a
  /// proof of optimality.
  std::size_t node_limit = 100000;

  double time_limit_seconds = 3600.0;

  /// Search stops early, reporting Optimal, once the gap between the best
  /// integer-feasible solution found (the incumbent) and the best remaining
  /// relaxation bound anywhere in the tree closes below this -- an accepted,
  /// provably-bounded suboptimality, the same "mip gap" every production
  /// MILP solver (CPLEX, Gurobi, HiGHS) exposes rather than insisting on an
  /// exact zero gap, which is often not worth the remaining nodes it costs.
  Real gap_tolerance = 1e-9;
};

/// Everything, in one object.
struct Options {
  Tolerances tolerances;
  IpmOptions ipm;
  Limits limits;
  ReaderOptions reader;
  LogOptions log;
  PresolveOptions presolve;
  MilpOptions milp;
};

}  // namespace sovsolve::model

#endif  // SOVSOLVE_MODEL_OPTIONS_HPP
