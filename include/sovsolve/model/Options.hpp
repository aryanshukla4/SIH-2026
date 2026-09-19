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

  /// Gondzio (1996) multiple centrality correctors, run after the Mehrotra
  /// corrector direction (only when `predictor_corrector` is on). The
  /// fraction-to-boundary ratio test in StepLength.cu takes a single GLOBAL
  /// min over every dual coordinate, so one tightly-bound/degenerate pair
  /// throttles the step length for every other variable too -- confirmed on
  /// bandm/grow15/scfxm1.mps, where the dual step length collapsed to ~0 for
  /// many consecutive iterations while the residual it was supposed to fix
  /// stayed exactly flat. Each corrector pass re-centers any complementarity
  /// pair that an extended trial step would push outside [0.1, 10] * mu by
  /// solving one more Newton system with a residual that is zero everywhere
  /// except that shortfall/excess, then adds the result onto the accumulated
  /// direction -- kept only if it does not shrink either step length. 0
  /// disables it.
  int max_centrality_correctors = 2;

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

/// How branch-and-bound picks the variable to branch on (solver/MilpSolve.hpp).
///
/// Only the HOST branch-and-bound (Module 28, dual simplex) reads this. The
/// GPU one (Module 22) always branches most-fractional, because the rules
/// below need warm-started simplex probes that an interior-point method cannot
/// provide.
enum class BranchingRule : std::uint8_t {
  /// Closest to one-half. Achterberg, Koch and Martin, "Branching rules
  /// revisited" (2005), section 2.1: "in general not better than selecting the
  /// variable randomly." Kept as the baseline every other rule is measured
  /// against.
  MostFractional,
  /// History only: the average objective gain per unit of rounding observed on
  /// past branchings. Reliability branching with `reliability = 0`.
  Pseudocost,
  /// Pseudocosts, with strong branching until each is reliable. The default
  /// rule of SIP and SCIP, and the best performer in the paper's study.
  Reliability,
};

/// Module 22 controls: branch-and-bound over the existing LP/QP relaxation
/// solver (solve_problem, Solve.hpp), activated automatically whenever
/// Problem::has_discrete() is true -- see solver/gpu/BranchAndBound.hpp.
struct MilpOptions {
  /// Module 28, the host branch-and-bound only. Every default below is the
  /// value Achterberg's thesis "Constraint Integer Programming" (2007)
  /// sections 5.2-5.7 gives for SCIP, which the thesis reports as tuned.
  BranchingRule branching = BranchingRule::Reliability;

  /// `eta_rel`: pseudocosts count as reliable once both directions have been
  /// observed at least this often. Thesis section 5.7: 8. Zero turns
  /// reliability branching into plain pseudocost branching.
  Real reliability = 8.0;

  /// `lambda`: strong branching stops once the best score has not changed for
  /// this many consecutive candidates. Thesis section 5.4: 8.
  std::size_t lookahead = 8;

  /// `kappa`: at most this many strong-branching candidates per node, a
  /// safeguard. Thesis section 5.4: 100.
  std::size_t max_strong_candidates = 100;

  /// `gamma`, the dual simplex iteration limit per strong-branching probe:
  /// twice the average node LP's iterations, clamped to this range. Thesis
  /// section 5.4: [10, 500].
  std::size_t strong_iterations_min = 10;
  std::size_t strong_iterations_max = 500;

  /// `epsilon` in the product score `max{q-, eps} * max{q+, eps}`, thesis
  /// equation (5.2). Keeps a zero gain in one direction from zeroing out the
  /// comparison. Thesis: 1e-6.
  Real score_epsilon = 1e-6;

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

/// Which algorithm solves the continuous relaxation.
///
/// These are not interchangeable on every model and are not meant to be:
/// the interior-point method converges in a near-constant iteration count and
/// runs its heavy work on the GPU, while the dual simplex terminates at an
/// exact vertex, produces primal-infeasibility and unboundedness verdicts the
/// IPM's convergence test structurally cannot, and warm-starts across a bound
/// change -- which is the only reason a branch-and-bound node is cheap. PDLP
/// is a first-order method: no factorization at all, so its per-iteration cost
/// is a pair of sparse matrix-vector products and it reaches sizes the other
/// two cannot, at lower accuracy per iteration.
enum class Method : std::uint8_t {
  InteriorPoint,
  DualSimplex,
  PrimalSimplex,
  Pdlp,
  /// Module 25: the homogeneous self-dual embedding, host-side.
  ///
  /// A separate engine rather than a flag on `InteriorPoint` because that path
  /// is GPU-resident and the gpu -> solver library edge is one-way, so a
  /// host-only module cannot be called from inside it. The two share the
  /// interior-point FAMILY and nothing else; see solver/HomogeneousSolve.hpp.
  Hsd,
};

/// Module 25 controls: the homogeneous self-dual embedding
/// (solver/HomogeneousSolve.hpp).
///
/// The first nine fields are Andersen & Andersen (2000) Table 1.1 VERBATIM --
/// MOSEK's shipped defaults, which that paper records as unchanged across its
/// whole computational study. They are exposed rather than hard-coded so the
/// A/B against the other engines can be run at matched tolerances, not so they
/// need tuning.
///
/// Plain fields rather than the solver's own `HomogeneousParameters`, because
/// `model` sits below `solver` and must not include from it -- the same reason
/// `PdlpOptions` restates its numbers. `solve_hsd` does the conversion.
struct HsdOptions {
  Real beta1 = 0.1;      ///< centering cap, [AA] (1.12)
  Real beta2 = 1.0e-8;   ///< centrality floor, [AA] (1.20)
  Real beta3 = 0.9999;   ///< fraction to boundary, [AA] section 1.4.3
  Real rho_p = 1.0e-8;   ///< relative primal infeasibility tolerance
  Real rho_d = 1.0e-8;   ///< relative dual infeasibility tolerance
  Real rho_a = 1.0e-10;  ///< relative objective-gap tolerance, [AA] (1.24)
  Real rho_mu = 1.0e-10; ///< floor on mu/mu_0 in the ill-posed test
  Real rho_i = 1.0e-10;  ///< threshold at which tau counts as collapsed
  /// Relative tolerance on the gap-row residual. NOT in [AA] Table 1.1 although
  /// its section 1.4.5 uses it; set to match its two siblings in that same
  /// criterion. The one number here with no citation behind it.
  Real rho_g = 1.0e-8;

  std::size_t max_iterations = 200;

  /// Inner CG controls (solver/HostKkt.hpp). The linear algebra is matrix-free
  /// conjugate gradients on `A Theta A'`, whose condition number grows like
  /// `1/mu^2` (FORMULATION.md section 10.1) -- so `cg_max_iterations` is a
  /// budget that the late iterations are expected to hit, not an assertion.
  Real cg_tolerance = 1.0e-10;
  /// MEASURED, not guessed. At 500 the corpus gave 14/19 with three instances
  /// hitting the outer iteration limit and two stalling outright; at 5000 those
  /// five became Optimal, and in FEWER outer iterations -- 25fv47 went from 200
  /// (unconverged) to 26, e226 from a stall at 65 to 17, stair from a stall at
  /// 64 to 19. An inexact Newton direction does not merely slow the method
  /// down: it produces a step the centrality condition (1.20) rejects, so the
  /// run stalls rather than degrading gracefully. `IpmOptions` records the same
  /// lesson for its MinRes budget.
  std::size_t cg_max_iterations = 5000;
  /// Floor on `Theta^-1` before inverting. Mandatory: a free column's entry is
  /// exactly zero and would invert to infinity.
  Real theta_inv_floor = 1.0e-12;
  /// Dual regularization on the normal-equations diagonal, covering rank
  /// deficiency in `A` that no floor on `Theta` can reach.
  Real delta_d = 1.0e-10;
};

/// Module 24 controls: PDLP (solver/pdlp/Pdlp.hpp).
struct PdlpOptions {
  /// Relative tolerance for all three termination criteria (paper equations
  /// 6a-6c). The paper uses 1e-8 for "high quality" and 1e-4 for "moderately
  /// accurate"; 1e-8 is the default here so PDLP is held to the same standard
  /// as the other two engines rather than being flattered by a looser one.
  Real termination_tolerance = 1e-8;

  /// Adaptive restarts (paper section 3.2). The inner loop restarts from a
  /// candidate point -- either the current iterate or the step-size-weighted
  /// average since the last restart, whichever has the smaller normalized
  /// duality gap -- and the outer loop counter advances.
  ///
  /// This is the enhancement the paper's own ablation ranks first, and the
  /// reason is structural rather than empirical: PDHG's ergodic (averaged)
  /// iterate converges at a good rate but averages in the early, bad
  /// iterates forever. Restarting throws that history away once it has served
  /// its purpose.
  bool adaptive_restart = true;

  /// Section 3.2's three restart constants. `sufficient` fires on a decisive
  /// decay of the normalized duality gap; `necessary` fires on a smaller
  /// decay that has additionally stopped making local progress; `artificial`
  /// caps how long an inner loop may run relative to the total iteration
  /// count, which is what guarantees the primal weight (updated only at a
  /// restart) keeps being updated at all.
  Real restart_sufficient = 0.9;
  Real restart_necessary = 0.1;
  Real restart_artificial = 0.5;

  /// Update the primal weight `omega` at each restart (paper Algorithm 3).
  ///
  /// `omega` is the one parameter that decides how the step is SPLIT between
  /// the primal and dual (`tau = eta/omega`, `sigma = eta*omega`), while
  /// `eta` decides its size. Left at 1 the two halves are weighted equally
  /// regardless of their actual scales, which is wrong whenever `c` and `q`
  /// differ in magnitude -- and initializing it to `||c||_2/||q||_2` is
  /// exactly the statement that they should not be.
  bool primal_weight_update = true;

  /// Exponential smoothing in log space, Algorithm 3 line 4. The raw estimate
  /// `||dy||/||dx||` swings wildly from one restart to the next; `theta = 0.5`
  /// (the paper's value) takes the geometric mean of the estimate and the
  /// previous weight. Log space is the right place for it because the weight
  /// is symmetric there: `log(1/omega) = -log(omega)`.
  Real primal_weight_smoothing = 0.5;

  /// Detect infeasibility and unboundedness from the iterates themselves
  /// (arXiv 2102.04592). Without it PDLP simply runs a diverging model to the
  /// iteration limit -- the objective on `gas11` runs away to -1.9e11 and no
  /// verdict is produced.
  bool infeasibility_detection = true;

  /// Margin on every certificate condition.
  ///
  /// The asymmetry decides this: too strict only costs a missed detection,
  /// reported honestly as `MaxIterations`, while too loose declares a
  /// slowly-converging FEASIBLE model infeasible -- a confidently wrong
  /// answer, and the exact failure `greenbea` produced in Module 23.
  ///
  /// So it was measured rather than chosen. Across the 18 feasible, bounded
  /// Netlib instances, plus `gas11` which is genuinely unbounded:
  ///
  ///     1e-8   no false verdicts, but MISSES gas11 (459 of its 459 rows are
  ///            equalities, each needing |K v_x|_i <= tol, and the residual
  ///            sits between 1e-8 and 1e-6)
  ///     1e-6   no false verdicts, and catches gas11 at iteration 40
  ///     1e-4   FOUR false verdicts -- 25fv47, israel and stair reported
  ///            Infeasible and 80bau3b Unbounded, all of them actually Optimal
  ///
  /// The cliff between 1e-6 and 1e-4 is sharp, so the default sits two orders
  /// below it.
  Real certificate_tolerance = 1e-6;

  /// Iterations between termination checks. Each check costs a `K'y` product
  /// and, once restarts land, a normalized duality gap evaluation -- real work
  /// that does not advance the iterate. The paper uses 40.
  std::size_t check_interval = 40;

  /// Choose each step's size by trial (paper Algorithm 2) instead of fixing
  /// it at `0.9/||K||_2`. Off is the textbook rule; keep the switch because
  /// the paper's own ablation compares exactly these two, and because a
  /// suspected step-size bug should be isolatable without rebuilding the rest
  /// of the iteration.
  bool adaptive_step_size = true;

  /// Fraction of `1/||K||_2` used as the baseline step size. PDHG converges
  /// for `eta <= 1/||K||_2`; the paper's baseline backs off to 0.9 of it.
  /// Superseded by the adaptive rule once Algorithm 2 lands.
  Real step_size_fraction = 0.9;

  /// Power-iteration budget and relative tolerance for estimating `||K||_2`.
  /// Only the baseline needs this: PDLP proper starts from `1/||K||_inf`,
  /// which is one sweep.
  std::size_t power_iterations = 100;
  Real power_tolerance = 1e-6;

  /// Iteration cap. `0` means automatic: `100000`, matching the paper's own
  /// KKT-pass limit for its baseline comparisons.
  std::size_t max_iterations = 0;
};

/// How Module 5 equilibrates `A` before anything downstream sees it.
enum class ScalingMode : std::uint8_t {
  /// Alternating geometric mean of the smallest and largest magnitude in each
  /// row/column: `1/sqrt(min*max)`. The long-standing default here, and what
  /// the IPM and both simplex engines were tuned against.
  GeometricMean,
  /// Ruiz equilibration followed by one Pock-Chambolle pass, which is what
  /// PDLP specifies (NeurIPS 2021 paper, section 3.5). Only meaningful for
  /// `Method::Pdlp` -- a first-order method has no factorization to stabilize,
  /// so its conditioning depends far more directly on the scaling than a
  /// factorization-based method's does.
  RuizPockChambolle,
};

/// Module 5 controls (solver/Scaler.hpp).
struct ScalingOptions {
  /// Left at `GeometricMean` unless the caller opts in, so adding PDLP does
  /// not silently re-tune the IPM and simplex paths.
  ScalingMode mode = ScalingMode::GeometricMean;

  /// Ruiz iterations before the Pock-Chambolle pass. PDLP section 3.5 uses
  /// 10. Ruiz proves the row and column infinity norms converge to 1 under
  /// iteration, so this is a convergence budget, not a tuning knob.
  std::size_t ruiz_iterations = 10;

  /// Whether the Pock-Chambolle pass runs after the Ruiz iterations. PDLP's
  /// default is both; its own ablation (paper appendix C.5) compares each
  /// alone, and turning this off is also the only way to observe Ruiz's
  /// defining property -- the Pock-Chambolle pass deliberately moves the
  /// row/column norms away from the fixed point Ruiz drives them to, so a
  /// test of "did Ruiz equilibrate" has to look before it runs.
  bool pock_chambolle = true;

  /// Pock-Chambolle exponent. With `alpha = 1` the row pass uses the
  /// `2 - alpha = 1` norm and the column pass the `alpha = 1` norm, i.e. both
  /// are l1. PDLP uses 1 as its baseline; the paper also reports testing 0
  /// and 2.
  Real pock_chambolle_alpha = 1.0;
};

/// Module 23 controls: the dual simplex (solver/simplex/DualSimplex.hpp).
struct SimplexOptions {
  Method method = Method::InteriorPoint;

  /// Threshold-pivoting factor for the basis LU: an entry is an eligible
  /// pivot only at this fraction or more of the largest remaining magnitude
  /// in its column. Lower admits sparser factors and less stability. 0.1 is
  /// the long-standing compromise value.
  Real pivot_tolerance = 0.1;

  /// A reduced cost outside `[-tol, tol]` on the wrong side of its bound is a
  /// real dual infeasibility rather than rounding.
  Real dual_feasibility_tolerance = 1e-7;

  /// A basic variable outside its bounds by more than this is genuinely
  /// primal infeasible, and is what dual pricing selects on.
  Real primal_feasibility_tolerance = 1e-7;

  /// Smallest `|alpha|` in the pivot row that the ratio test will accept as
  /// an entering candidate. A pivot below this is a near-parallel column
  /// whose reciprocal amplifies rounding through every later solve.
  Real pivot_floor = 1e-9;

  /// Product-form etas appended before the basis is refactorized from the
  /// original data. Both eta application cost and accumulated rounding grow
  /// linearly with the eta file, so this bounds them together.
  std::size_t refactor_interval = 100;

  /// Temporary finite bound given to a dual-infeasible nonbasic column during
  /// phase 1. Escalated by `artificial_bound_growth` when the solved
  /// artificially-bounded problem leaves a variable resting on one and the
  /// unbounded-ray test does not fire.
  Real artificial_bound = 1e7;
  Real artificial_bound_growth = 100.0;
  std::size_t max_artificial_rounds = 5;

  /// Bound-flipping (long-step) dual ratio test. Off is the textbook
  /// single-candidate test; keep the switch so a suspected ratio-test bug can
  /// be isolated without rebuilding the rest of the iteration.
  bool bound_flipping = true;

  /// Finish a dual simplex run that ended without a verdict by handing its
  /// basis to the primal simplex.
  ///
  /// The two algorithms fail in different places, which is the whole point of
  /// having both. The dual's phase 1 boxes a dual-infeasible column in a
  /// temporary bound; when that box is still binding at the end and no ray
  /// proves unboundedness, the dual has no verdict to give. The primal has no
  /// boxes to be trapped by, and the dual's endpoint is primal feasible (it is
  /// feasible for the narrower boxed problem, and the true bounds are wider),
  /// so the primal starts in phase 2 and simply finishes the job.
  bool primal_cleanup = true;

  /// Pivot budget. `0` means automatic: `50 * (m + n) + 1000`.
  ///
  /// Deliberately NOT `Limits::max_iterations`, whose default of 200 is an
  /// interior-point budget -- an IPM converges in tens of iterations and a
  /// simplex takes on the order of the problem's own dimension, so one number
  /// cannot mean both. Sharing the field would silently cut every simplex
  /// solve off at 200 pivots and report `MaxIterations` on models that were
  /// solving perfectly well.
  ///
  /// The multiplier is deliberately loose. A simplex usually finishes in a
  /// small multiple of `m + n`, but degeneracy is not rare and the cost of
  /// guessing low is a `MaxIterations` on a model that was converging:
  /// measured, Netlib `greenbea` needs 112,421 pivots against `m + n = 7797`,
  /// which a 10x budget cuts off at 79,970. `Limits::time_limit_seconds` is
  /// the backstop that actually bounds a runaway solve.
  std::size_t max_iterations = 0;
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
  SimplexOptions simplex;
  ScalingOptions scaling;
  PdlpOptions pdlp;
  HsdOptions hsd;
};

}  // namespace sovsolve::model

#endif  // SOVSOLVE_MODEL_OPTIONS_HPP
