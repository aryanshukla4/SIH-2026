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

#include "sovsolve/core/Cancel.hpp"
#include "sovsolve/core/Span.hpp"
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
  /// Mehrotra (1992) section 7's starting point, adapted to bounds and
  /// inequality slacks (Initializer.cpp), instead of the fixed bound-midpoint
  /// start. Ignored when a warm-start hint is given. OFF by default on
  /// measurement: on the GPU interior point it helped boeing1 and removed
  /// greenbea's wrong optimum, but lost bnl1 (Optimal -> NotConverged) -- the
  /// paper's "significantly smaller number of iterations" did not hold here.
  bool mehrotra_start = false;

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

  /// Preconditioner for the normal-equations CG (solver/gpu/Preconditioner.hpp):
  ///   0  IC(0), then Jacobi -- no exact factor
  ///   1  exact sparse Cholesky through cuDSS, when the build links it
  ///   2  exact factor from the in-house SparseLdl on the host -- the same
  ///      matrix, factored by our own code, so the two can be compared
  /// A failed factor falls back to IC(0), then Jacobi.
  int direct = 2;

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
/// solver/Presolver.hpp), but docs/spec/module.txt is explicit that startability
/// (the canonicalizer's contract) "must not depend on the Presolver, which
/// can be switched off" -- this is that switch.
struct PresolveOptions {
  bool enabled = true;

  /// The LP presolve of solver/LpPresolve.hpp (Andersen & Andersen 1995) in
  /// place of Module 4's, wherever the model is an LP and the pipeline has
  /// no warm start to forward-map. Off gives Module 4 alone (`--presolve=1`).
  bool lp_reductions = true;
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
/// How Module 28's branch-and-bound picks the next open node. Achterberg,
/// "Constraint Integer Programming" (2007), chapter 6.
enum class NodeSelection : std::uint8_t {
  /// Always the open node with the smallest dual bound, [CIP] section 6.2.
  /// Fewest nodes for a fixed branching rule (Proposition 6.1), but its nodes'
  /// LP solutions are "usually far away from integrality", so it rarely finds
  /// an incumbent. The rule this engine shipped with; kept for comparison.
  BestFirst,
  /// [CIP] section 6.6, SCIP's default and the best overall strategy in
  /// Table 6.1: best estimate search with plunging, and every `best_frequency`
  /// plunges a best-bound node instead.
  Interleaved,
};

struct MilpOptions {
  /// Module 28, the host branch-and-bound only. Every default below is the
  /// value Achterberg's thesis "Constraint Integer Programming" (2007)
  /// sections 5.2-5.7 gives for SCIP, which the thesis reports as tuned.
  BranchingRule branching = BranchingRule::Reliability;

  /// [CIP] chapter 6. See NodeSelection.
  NodeSelection node_selection = NodeSelection::Interleaved;

  /// `bestfreq`: every this-many plunges, the next node is the best-BOUND
  /// leaf rather than the best-estimate one. [CIP] section 6.6: 10.
  std::size_t best_frequency = 10;

  /// A plunge takes at least `plunge_min_depth_fraction * dmax` and at most
  /// `plunge_max_depth_fraction * dmax` steps, `dmax` the deepest node
  /// processed so far. [CIP] section 6.3: 0.1 and 0.5.
  Real plunge_min_depth_fraction = 0.1;
  Real plunge_max_depth_fraction = 0.5;

  /// Past the minimum, a plunge is abandoned once the next node's local
  /// relative gap `(c_Q - c_lower) / (c_upper - c_lower)` exceeds this.
  /// [CIP] section 6.3: 0.25.
  Real plunge_max_gap = 0.25;

  /// Primal heuristics, [CIP] chapter 9: simple rounding after every node LP
  /// (9.1.2) and diving (Algorithm 9.1, section 9.2).
  bool heuristics = true;

  /// Diving's LP iterations may not exceed this fraction of the node-LP
  /// iterations: "we demand that this number must stay below 5% of the
  /// current total number of simplex iterations used for solving the regular
  /// LP relaxations" ([CIP] section 9.2).
  Real dive_quota = 0.05;

  /// Added to the quota above. NOT from the thesis: at the root the node-LP
  /// total is one LP's worth, so a pure 5% would forbid the first dive
  /// outright. The per-heuristic offsets SCIP uses are in Berthold's diploma
  /// thesis ([CIP] reference [41]), which this code has not seen.
  std::size_t dive_allowance = 1000;

  /// The objective feasibility pump at the root ([CIP] 9.3.3; Berthold,
  /// "Primal Heuristics for Mixed Integer Programs", ZIB 2006, section 3.1.2
  /// and Algorithm 3). Run only while there is no incumbent: it is a START
  /// heuristic. Parameters are Berthold's (his "FP095", the setting he
  /// recommends): alpha_0 = 1 reduced by `fp_alpha_factor` per round, cycle
  /// test on alpha within 0.005, stage limits 10000/2000 rounds and 70/600
  /// stalls (a stall: fractionality not reduced by 10%), T in [10, 30]
  /// columns flipped on a 1-cycle.
  bool feasibility_pump = true;
  Real fp_alpha_factor = 0.95;
  std::size_t fp_max_rounds_stage1 = 10000;
  std::size_t fp_max_rounds_stage2 = 2000;
  std::size_t fp_max_stalls_stage1 = 70;
  std::size_t fp_max_stalls_stage2 = 600;

  /// RENS at the root ([CIP] 9.1.1; Berthold section 3.2.1): fix the integral
  /// integer columns of the root LP, bound the fractional ones to
  /// [floor, ceil], and solve that sub-MIP with this same branch-and-bound.
  /// [CIP]: "abort the sub-MIP solving process after either a total of 5000
  /// nodes has been processed or no improvement of the sub-MIP incumbent has
  /// been found for 500 consecutive nodes"; skip if more than half the
  /// integer columns are fractional; proceed only if presolving the sub-MIP
  /// removes at least 25% of the columns.
  bool rens = true;
  std::size_t rens_node_limit = 5000;
  std::size_t rens_stall_nodes = 500;
  Real rens_max_fractional_ratio = 0.5;
  Real rens_min_reduction = 0.25;

  /// Domain propagation, [CIP] chapter 7: linear constraint propagation
  /// (Algorithm 7.1) at EVERY node and after every dive bound change -- the
  /// thesis's "aggr linear" setting, which Table 7.1 measures better than its
  /// every-fifth-depth default on almost all test sets -- plus objective
  /// propagation (7.6) and root reduced cost strengthening (7.7). Also gates
  /// [CIP] 8.8's LOCAL reduced cost strengthening at every node, which SCIP
  /// files under separators but which here, as there, only tightens bounds.
  bool propagation = true;
  /// Each row is visited at most this many times per propagation call.
  /// [CIP] states no limit -- its 5% minimum-change rule (7.3) is what bounds
  /// the work -- so this cap is OURS; stopping early is always safe.
  std::size_t propagation_row_visits = 20;

  /// MIP presolve on the canonical model ([CIP] chapter 10, stage A:
  /// Algorithm 10.1's linear-constraint presolving and Algorithm 10.14's dual
  /// fixing). See MilpCanonicalPresolve.hpp.
  bool presolve = true;
  /// Presolve rounds, stopping earlier when a round changes nothing. [CIP]
  /// runs its outer loop until no reduction is found; the cap is OURS.
  std::size_t presolve_rounds = 20;
  /// Stage B: the reductions that REMOVE a column -- substitution of implied
  /// free variables ([AGH] 4.5, its largest single-column entry at 1.42) and
  /// parallel column merging ([AGH] 6.3, 1.09). Needs `presolve`. Separate
  /// from it because these are the reductions that need a postsolve stack,
  /// so switching them off returns the solver to stage A's simpler shape.
  bool presolve_columns = true;

  /// Root cutting planes in canonical space, cut-and-branch as [CIP] 8.10
  /// measures SCIP's default: Gomory mixed integer cuts ([CIP] 8.3 with
  /// Wolter's section 6.1 safeguards) and complemented MIR cuts (Wolter
  /// chapter 3, the fast version), selected by [CIP] Algorithm 3.2, for at
  /// most `cut_rounds` separation rounds -- Wolter's MAXROUNDS = 15.
  bool gomory_cuts = true;
  bool cmir_cuts = true;
  std::size_t cut_rounds = 15;
  /// A cut enters only if the LP point violates it by more than this,
  /// relative to max(1, |rhs|). [CIP] and Wolter compare against zero; the
  /// margin, which drops cuts violated only by rounding, is OURS.
  Real cut_violation_margin = 1e-6;

  /// Conflict analysis, [CIP] chapter 11: learn a conflict constraint from
  /// every node proven empty by propagation or by an infeasible LP
  /// (Algorithm 11.1), resolved through the bound-change trail to one FUIP
  /// constraint per depth level (at most 10 per conflict, [CIP] 11.3), kept as
  /// bound disjunctions (11.14) and used in domain propagation. Needs
  /// `propagation`.
  bool conflict_analysis = true;
  /// A conflict constraint considered this many times without a deduction is
  /// discarded -- [CIP] 11.3's aging. Neither [CIP] nor Witzig, Berthold and
  /// Heinz, "Experiments with Conflict Analysis in Mixed Integer Programming"
  /// (CPAIOR 2017, ZIB report 16-63) states the threshold: OURS.
  std::size_t conflict_max_age = 1000;
  /// Witzig et al. section 3: the pool holds "at least 1 000 and at most
  /// 50 000 conflict constraints at the same time", sized by "the number of
  /// variables and constraints"; when full, "the oldest conflict constraints
  /// are removed". Their sizing formula is not given; here the size is
  /// columns + rows, clamped to those bounds (OURS).
  std::size_t conflict_pool_min = 1000;
  std::size_t conflict_pool_max = 50000;
  /// Witzig et al. section 3: a conflict whose proof used the objective
  /// cutoff is deleted once a new incumbent improves on the one it was
  /// derived with by more than this fraction -- "a threshold of 5%".
  Real conflict_cutoff_drop = 0.05;

  /// Stop once this many nodes pass without an improved incumbent (0 = off).
  /// The RENS sub-MIP's stalling limit; available to any run.
  std::size_t stall_node_limit = 0;

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

  /// Separate root cover/GCD cuts before the search (MilpCuts.hpp). Module 28
  /// only; Module 22 always does. A switch because cuts are not free -- every
  /// cut is a row every node LP carries -- and whether they pay is a
  /// per-instance measurement, not an assumption.
  bool root_cuts = true;

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
  /// Module 31: the same engine running cuPDLPx's reflected-Halpern scheme
  /// (arXiv 2507.14051) instead of averaged PDHG. A separate `Method` rather
  /// than only a flag on `Pdlp` because the two are worth racing against each
  /// other in `Concurrent`, and because a benchmark row needs a name.
  PdlpX,
  /// Module 25: the homogeneous self-dual embedding, host-side.
  ///
  /// A separate engine rather than a flag on `InteriorPoint` because that path
  /// is GPU-resident and the gpu -> solver library edge is one-way, so a
  /// host-only module cannot be called from inside it. The two share the
  /// interior-point FAMILY and nothing else; see solver/HomogeneousSolve.hpp.
  Hsd,
  /// Module 30: run several of the engines above at once, on separate cores,
  /// and keep whichever finishes first. See solver/ConcurrentSolve.hpp.
  Concurrent,
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
  /// deficiency in `A` that no floor on `Theta` can reach. Altman & Gondzio
  /// (1999) section 5's default r_d = eps^(1/2): measured on the 94 feasible
  /// Netlib models with the direct factor, 92 optimal against 91 at the old
  /// 1e-10 with SparseLdl, and 93 either way with CHOLMOD.
  Real delta_d = 1.49e-8;

  /// Factor the normal equations directly (solver/NormalFactor.hpp) and use
  /// the factor as CG's preconditioner, instead of the Jacobi diagonal. Off
  /// reproduces the matrix-free engine exactly.
  bool direct = true;
  /// Altman & Gondzio (1999) section 5: when a solve needs it, the
  /// regularization is "multiplied by 10" and the factorization retried. The
  /// number of retries before falling back to the Jacobi diagonal is OURS.
  std::size_t regularization_retries = 6;
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
  ///     1e-4   FOUR false verdicts -- 25fv47, israel and stair reported
  ///            Infeasible and 80bau3b Unbounded, all of them actually Optimal
  ///     1e-6 .. 1e-9   no false verdicts, gas11 detected at every one
  ///
  /// (An earlier version of this comment recorded "1e-8 MISSES gas11". That was
  /// measured against the PRE-Module-24G test, which normalized by the
  /// candidate's own size; once the reference's (50)/(51) replaced it, gas11 is
  /// detected across that whole range. Re-swept and corrected.)
  ///
  /// The default is 1e-8 rather than 1e-6 because of `dfl001` -- feasible,
  /// optimum 1.1266396047e+07, and reported Unbounded at iteration 480 under
  /// 1e-6. Its bogus certificate and gas11's real one are separated by nearly
  /// four orders of magnitude in the very quantity (51) tests, so this is a
  /// measured gap and not a threshold nudged until a symptom disappeared:
  ///
  ///     instance   cone violation per unit of improvement
  ///     dfl001     8.8e-7    bogus   -- rejected at 1e-8 with 88x to spare
  ///     gas11      9.2e-11   real    -- accepted at 1e-8 with 108x to spare
  ///
  /// The asymmetry above still applies and still points this way: tightening
  /// costs at worst a missed detection reported honestly as `MaxIterations`,
  /// and buys the elimination of a confidently wrong answer.
  Real certificate_tolerance = 1e-8;

  /// Iterations between termination checks. Each check costs a `K'y` product
  /// and, once restarts land, a normalized duality gap evaluation -- real work
  /// that does not advance the iterate. The paper uses 40.
  std::size_t check_interval = 40;

  /// Evaluate the termination criteria where the iterate lives, when the
  /// backend can (cuPDLPx, arXiv 2507.14051, section 4, computes them on the
  /// GPU): five sums come back instead of the whole iterate, and the host
  /// re-measures in full only when they say "converged" and at exit, so every
  /// reported number is computed exactly as before. Off restores the
  /// download-every-check path.
  bool resident_check = true;

  /// With `resident_check`, the infeasibility certificates -- which need the
  /// iterate and its difference sequences on the host -- are tested at every
  /// this-many-th check instead of every check. OURS: no paper sets it; 10
  /// delays a certificate by at most 10 * check_interval iterations.
  std::size_t certificate_check_every = 10;

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

  /// Iteration cap. `0` means automatic: `1000000` (measured, see Pdlp.cpp).
  ///
  /// The cuPDLPx paper benchmarks with a time limit ALONE (section 4, "Time
  /// limit": 3600 s for small and medium instances, no iteration cap), and a
  /// cap on top of it reports as a failure a run the paper counts as solved.
  /// But that protocol is requested EXPLICITLY -- `scripts/benchmark.py
  /// --paper` passes a huge `--pdlp-max-iter` -- rather than inferred from
  /// the time limit, because `Limits::time_limit_seconds` is never unset: it
  /// defaults to 3600. An earlier version lifted the cap whenever a time
  /// limit was present, and so lifted it on EVERY run, turning a stalled
  /// model's ~10 s MaxIterations into an hour-long wait.
  std::size_t max_iterations = 0;

  /// Evaluate the termination criteria on the ORIGINAL problem rather than
  /// the preconditioned one the iteration runs on.
  ///
  /// cuPDLPx section 4: "The termination criteria are checked for the
  /// original LP instance, not the preconditioned ones, so that the
  /// preconditioning does not impact the termination." Without this, `1e-8`
  /// means "1e-8 in whatever units Ruiz and Pock-Chambolle happened to
  /// produce", which is not comparable across scalings, across engines, or to
  /// the paper -- and a scaling that shrinks the residuals would make the
  /// method look converged when it is not.
  ///
  /// It needs the preconditioner's factors, `R` and `S` in `A~ = R A S`, which
  /// only the caller has (they live in the transform stack). They go in
  /// `original_row_scale` / `original_col_scale` below; when those are empty
  /// the criteria fall back to the problem as given, which is also what a
  /// caller that did not scale should get.
  bool terminate_on_original = true;

  /// The preconditioner's composite row and column factors, `R` and `S`.
  /// NON-OWNING, like `Options::cancel`: set by the caller for the duration of
  /// one solve (LpSolve.cpp), empty otherwise. Only read when
  /// `terminate_on_original` is set and the lengths match the problem.
  core::HostSpan<const Real> original_row_scale;
  core::HostSpan<const Real> original_col_scale;

  // ---- Module 31: the cuPDLPx scheme (arXiv 2507.14051) ------------------
  //
  // Four changes, all driven by the restarted-Halpern theory of arXiv
  // 2407.16144. They are one switch rather than four because they are not
  // independent: constant step size is only safe because reflection already
  // takes a longer effective step, and the fixed-point restart criterion is
  // only defined for the Halpern scheme. See solver/pdlp/Pdlp.hpp.

  /// Run the reflected-Halpern iteration instead of averaged PDHG.
  ///
  /// Off by default, so `--method=pdlp` keeps measuring the engine the corpus
  /// numbers in docs/ were taken with. `--method=pdlpx` turns it on.
  bool halpern = false;

  /// Reflection `gamma` in `(1 + gamma) PDHG - gamma * id`.
  ///
  /// cuPDLPx allows `gamma` in [0, 1] and does not say which it uses. 1 is
  /// the value the theory is stated for -- arXiv 2407.16144 equation (26) is
  /// `2T(z) - z`, i.e. exactly `gamma = 1` -- and it is the setting that earns
  /// the factor-of-2 complexity improvement its Corollary 1 proves. 0
  /// recovers plain Halpern, which is the useful ablation.
  Real reflection = 1.0;

  /// `eta = halpern_step_fraction / ||A||_2`, cuPDLPx's constant step size.
  ///
  /// The paper's own 0.998. It replaces the adaptive trial loop entirely:
  /// PDHG needs `eta <= 1/||A||_2`, and 0.998 of it is the largest step that
  /// still leaves the canonical norm `P` positive definite by a margin. That
  /// margin is load-bearing here in a way it is not for baseline PDHG,
  /// because `r(z) = ||z - PDHG(z)||_P` is measured in that norm and a `P`
  /// that is merely semidefinite would let the restart criterion read zero at
  /// a non-fixed point.
  Real halpern_step_fraction = 0.998;

  /// The three fixed-point restart constants -- PUBLISHED, not tuned here.
  ///
  /// cuPDLPx names the three conditions but publishes no values. They come
  /// instead from HPR-LP (Chen, Sun, Yuan, Zhang, Zhao, arXiv 2408.12179),
  /// section 4, "Initialization and parameter setting": "the restart
  /// criteria are based on conditions (10), (11), and (12), with parameters
  /// alpha_1 = 0.2, alpha_2 = 0.6, and alpha_3 = 0.2."
  ///
  /// Borrowing them is licensed, not merely convenient. arXiv 2509.23903
  /// (same authors) Proposition 3.1 proves that cuPDLPx with `gamma = 1`
  /// generates EXACTLY the iterates of the HPR method when `sigma = eta /
  /// omega` and `lambda_A = 1 / eta^2` -- and HPR-LP's merit function
  /// `||w - w-hat||_M` is the same fixed-point residual as cuPDLPx's
  /// `||z - PDHG(z)||_P`. Same method, same quantity, same three conditions.
  ///
  /// HISTORY, kept because it is the reason for the rule above: these were
  /// once tuned on 18 Netlib instances (0.2 / 0.8 / 0.36), and on the full
  /// 99 that tuning did not generalize. The one that survived, 0.2, happens
  /// to be HPR-LP's alpha_1. The theory's own `1/e` (arXiv 2407.16144 eq.
  /// 10) remains one flag away: `--pdlp-restart-sufficient=0.3679`.
  Real halpern_restart_sufficient = 0.2;
  Real halpern_restart_necessary = 0.6;
  Real halpern_restart_artificial = 0.2;

  /// How often conditions (i) and (ii) are EVALUATED, in iterations.
  ///
  /// 1 is cuPDLPx ("evaluates potential restart conditions at each
  /// iteration"), which the fixed-point residual makes free. HPR-LP checks
  /// "termination and restart criteria every 150 iterations", and its alpha
  /// constants above were published under that schedule. Default 1, the
  /// cuPDLPx design this module implements; 150 is the configuration the
  /// constants were chosen for, and the two are measured against each other
  /// rather than assumed equivalent.
  std::size_t halpern_restart_check_every = 1;

  /// Which rule moves the primal weight at a restart.
  enum class WeightRule : std::uint8_t {
    /// HPR-LP Algorithm 3 ("SigmaUpdate"), translated through Proposition 3.1:
    /// `omega = ||dy|| / ||dx||`, applied only under its safeguards (17) --
    /// both distances in (1e-16, 1e12) -- and (18) -- the ratio of relative
    /// dual to primal infeasibility at the restart point in (1e-8, 1e8) --
    /// and otherwise reset to the starting weight. NO free constant: every
    /// number in it is published.
    HprLp,
    /// cuPDLPx section 3's PID controller. Its three coefficients are NOT
    /// published (see below), which is why it is not the default.
    Pid,
  };
  WeightRule weight_rule = WeightRule::HprLp;

  /// PID coefficients, used only with `WeightRule::Pid`.
  ///
  /// cuPDLPx publishes none of the three. With `K_I = K_D = 0` the update
  /// collapses onto the original PDLP paper's Algorithm 3 with `theta = K_P`
  /// (derivation in Pdlp.hpp), so `(0.5, 0, 0)` reproduces a published rule;
  /// the values below were this project's own tuning on 18 Netlib instances,
  /// kept for comparison, and are NOT a claim that they generalize -- on the
  /// full 99-instance set they did not.
  Real pid_kp = 0.3;
  Real pid_ki = 0.01;
  Real pid_kd = 0.05;

  /// Anti-windup clamp on the PID's accumulated integral term, in log units.
  Real pid_integral_clamp = 10.0;
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
  /// The concurrent race by default: it is the configuration every published
  /// benchmark of this solver measures, and it works on every build. The
  /// previous default, the GPU interior point, made a plain `solve model.mps`
  /// on the CPU-only build fail with "no CUDA" -- the first thing a new user
  /// would run. Branch-and-bound is unaffected: MilpSolve sets its own node
  /// and relaxation method explicitly.
  Method method = Method::Concurrent;

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

  /// Cost perturbation against DUAL DEGENERACY, Koberstein, "The dual simplex
  /// method, techniques for a fast and stable implementation" (thesis, 2005),
  /// section 6.3.1 -- see SolveSimplex.cpp. Applied only when the structural
  /// costs take fewer than n/4 distinct values, the thesis's own test for a
  /// significantly dual-degenerate problem.
  bool cost_perturbation = true;

  /// Dual steepest edge pricing in the dual simplex: choose the leaving row by
  /// `violation^2 / beta_r` rather than by the largest violation (Dantzig).
  /// Koberstein, "The dual simplex method, techniques for a fast and stable
  /// implementation" (2005), section 3.3 and 8.2.2.1 -- Forrest and Goldfarb's
  /// "Dual algorithm I". The thesis's section 9.4 measures it as its single
  /// largest improvement: -43.9% iterations, -47.6% time against Dantzig.
  bool dual_steepest_edge = true;

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
/// Module 30, the concurrent optimizer.
///
/// DELIBERATELY NOT TUNED TO ANY ONE MACHINE. The engine line-up is chosen at
/// run time from `std::thread::hardware_concurrency()`, so the same binary
/// races two engines on a small laptop and every engine it has on a large
/// workstation, with no rebuild and no per-machine constant. A deployment
/// target with dozens of cores should also raise `max_threads` above the
/// number of engines only once there is intra-engine parallelism to spend it
/// on -- racing five engines cannot use fifty cores.
struct ConcurrentOptions {
  /// Upper bound on racing threads. 0 means "decide from the hardware".
  /// A value of 1 makes the concurrent method run a single engine, which is
  /// how it behaves on a single-core machine.
  std::size_t max_threads = 0;

  /// Race the GPU interior-point engine too, where a build has one.
  ///
  /// OFF by default, and that default is a statement about CONSUMER cards,
  /// not about the algorithm: on a consumer Ampere part FP64 runs at 1/64 of
  /// FP32 (docs/ARCHITECTURE-REVIEW.md 3.5), so a factorization-based GPU
  /// engine loses to the CPU simplex and only burns a thread supervising it.
  /// On a datacentre card (A100 and later: 1:2 FP64, HBM bandwidth) that
  /// reverses, and the GPU engine costs no CPU core to run -- it should be in
  /// the race there. Turn it on when the deployment hardware warrants it.
  bool include_gpu_interior_point = false;

  /// Crossover: when the race is won by an engine that stops at a tolerance
  /// rather than at a vertex -- cuPDLPx or the interior point -- finish its
  /// answer with the primal simplex, warm-started from a basis read off its
  /// point. The race then returns an exact vertex whichever engine won, so the
  /// objective no longer depends on thread timing (measured: cuPDLPx won 5 of
  /// 60 runs of a two-row LP on a busy machine and returned -2.7999999914
  /// where the simplex engines return -2.8). Same idea as Gurobi's concurrent
  /// LP, which crosses a barrier win over to a basis.
  ///
  /// OFF by default, because it is not yet cheap. Measured 2026-09-30 on the
  /// 31 Netlib LPs where cuPDLPx or HSD wins the race (i5-12450H, best of 3):
  /// SGM10 0.60 s off, 1.21 s with the primal finish, 1.40 s with the dual;
  /// 10 and 7 of the 31 ran out of budget. From the Andersen-Ye crash basis
  /// the simplex still needs about as long as a cold solve on models such as
  /// 25fv47 (0.07 s -> 0.58 s). A crossover that is cheap enough to be the
  /// default needs the primal and dual "push" phases (Megiddo 1991; Andersen
  /// & Ye 1996) rather than handing the crash basis straight to the simplex.
  /// Turn it on (`--concurrent-crossover=1`) when an exact vertex on every run
  /// matters more than speed.
  bool crossover = false;

  /// Crossover time budget, as a multiple of the race's own wall time, with a
  /// floor of `crossover_min_seconds`. If the simplex has not finished by then
  /// the tolerance answer is returned as it was -- the crossover can make a
  /// solve at most this much slower, never wrong. OURS: 1.0 caps the worst
  /// case at twice the race; the floor keeps it from starving on tiny models
  /// where the race takes microseconds.
  Real crossover_time_factor = 1.0;
  Real crossover_min_seconds = 1.0;

  /// Which simplex finishes the crossover from the crash basis:
  /// `PrimalSimplex`, or `DualSimplex` (with its primal cleanup). OURS: the
  /// primal, measured faster on the 31 models above (SGM10 1.21 s vs 1.40 s),
  /// though the dual reached a vertex more often (21 vs 18).
  Method crossover_method = Method::PrimalSimplex;
};

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
  ConcurrentOptions concurrent;

  /// Cooperative cancellation, set by the concurrent optimizer so an engine
  /// that has already lost the race stops instead of running to completion.
  /// Null in a normal solve, and then every engine's check costs one
  /// predictable null comparison per iteration.
  ///
  /// It lives here rather than in each engine's signature because every
  /// engine already takes `const Options&`, so this adds a capability to all
  /// four without touching four different call sites and their overloads.
  /// NOT OWNED -- the token must outlive the solve, which it does: the
  /// concurrent driver keeps it on its own stack and joins every thread
  /// before returning.
  const core::CancelToken* cancel = nullptr;
};

}  // namespace sovsolve::model

#endif  // SOVSOLVE_MODEL_OPTIONS_HPP
