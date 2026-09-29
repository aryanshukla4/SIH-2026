# How this differs from HiGHS

PS 26119 requires that the solver "shall not be built upon any existing open
source solver library." HiGHS appears exactly once in this repository, in
`scripts/oracle_check.py`, as a **development-time correctness oracle** — it
checks our parser/canonicalizer/solver output against an independent
implementation on Netlib instances with published optima. It is never linked,
never vendored, not on any path that produces the `solve` binary, and the
check skips cleanly when SciPy (which wraps HiGHS) is absent. See
`README.md`'s "On the 'from scratch' constraint" section. Using it to check
our own answers is the opposite of building on it.

This note exists because using the same *names* HiGHS (and every other
solver) uses — `nnz`, `presolve`, `iteration`, `Optimal`/`NotConverged` — can
look like copying to someone skimming logs side by side. It isn't: these are
standard terms in sparse linear algebra and interior-point methods, decades
older than either codebase (`nnz` = "number of non-zeros," a term you'll find
in any sparse-matrix paper; presolve reduction techniques trace to Andersen &
Andersen's 1995 paper, used by HiGHS, CPLEX, Gurobi, and this project alike,
each with its own independent implementation). What follows is where the
actual architecture and algorithms genuinely differ.

## 1. Both algorithm families, each derived independently

*(Rewritten 2026-09-13, extended 2026-09-14. Until Module 23 this section
said "there is no simplex code anywhere in this project." That was true when
written and is not true now; the honest statement is below.)*

**HiGHS** defaults to **dual simplex** for LP, with its own IPM available as
an option and a MIP branch-and-bound layer on top.

**This solver** now has three engines from three algorithm families, and where
the *category* overlaps with HiGHS that is worth stating plainly rather than
hiding:

- A **primal-dual interior-point method** (`src/solver/gpu/`), GPU-resident
  and matrix-free — a predictor-corrector (Mehrotra-style) Newton iteration
  moving through the *interior* of the feasible region.
- A **revised simplex** engine (`src/solver/simplex/`, `docs/spec/module.txt` §23),
  host-only, in both its **dual** and **primal** forms, sharing one
  factorized-basis core (`detail/SimplexEngine`) and selected at the command
  line with `--method=dual-simplex` / `--method=primal-simplex`.
- **PDLP**, a **first-order** method (`src/solver/pdlp/`, `docs/spec/module.txt` §24),
  `--method=pdlp`. This one has **no counterpart in HiGHS at all** — it
  factors nothing, ever, and its entire inner loop is a pair of sparse
  matrix-vector products.

That third engine is also where the vendored-source question comes up a second
time, and it is answered the same way. `or-tools/` contains Google's own PDLP,
by the same authors as the paper. **It was not read.** The evidence is the same
kind as for the simplex: the three things our implementation had to get right
that the *papers themselves* state wrongly or leave out — an acceptance bound
that diverges as literally written, a dual objective whose printed sign
contradicts the paper's own notation section, and a trust-region reduction that
silently drops finite upper bounds. All three are recorded in `docs/spec/module.txt`
§24 with the measurements that exposed them. Code copied from a working
implementation does not reproduce a paper's errata and then fix them.

What matters for PS 26119 is not whether the *category* is shared — "revised
simplex with a factorized basis" is a 1953 result (Dantzig–Orchard-Hays), and
every solver that implements it produces code that rhymes — but whether the
implementation was **derived or copied**. This project's was derived, and the
derivation is written down where it can be checked:

| Component | Textbook-level definition it was built from | Where this project's own derivation is written down |
|---|---|---|
| Sparse LU, Markowitz ordering + threshold partial pivoting | Markowitz's pivot-cost criterion `(r_i-1)(c_j-1)` | `src/solver/simplex/LuFactor.cpp` header comment, including the argument for why a stalled elimination can always be repaired with a logical |
| Product-form basis update (PFI) | One elementary eta column per pivot | `LuFactor.cpp::update` |
| Bounded-variable augmented working form | Derived here from this project's own canonical form | `docs/FORMULATION.md` §12 and `Basis.hpp` |
| Dual ratio test, sigma-folded pivot row, bound flipping | The bounded-variable dual ratio test | the sign derivation in `DualSimplex.hpp`'s header, worked out symbol by symbol from `y = B^-T c_B` |
| Dual phase 1 by artificial bounds | Bound the free directions, solve, escalate | `DualSimplex.cpp::restore_dual_feasibility` |
| Primal phase 1 by sum of infeasibilities | Piecewise-linear composite objective | `PrimalSimplex.hpp` header |
| EXPAND anti-degeneracy ratio test | Gill, Murray, Saunders & Wright, Math. Prog. 45 (1989), sections 4 and 7 | `PrimalSimplex.cpp::iterate` |

These are the *definitions* — the mathematical statements you would find in
any linear-programming text. The code that implements them here was written
against those statements, not against another implementation of them.

The vendored `HiGHS/` tree contains its own dual simplex at
`HiGHS/highs/simplex/HEkkDual.*`. **It was not read while writing any of the
above**, deliberately and for exactly this reason — the same standard §3
below describes for the presolve rules. The evidence that this is an
independent derivation is the shape of the bugs it produced, all recorded in
`docs/spec/module.txt` §23: a sigma sign that had to be re-derived from scratch, a
basis-validation check that counted statuses instead of detecting a
duplicated slot, an artificial-bound escalation that was initially mistaken
for a genuine `Infeasible` verdict, and a Bland's-rule latch that never
released. Ported code does not fail that way.

**What the second engine bought, measured:** the IPM reached `Optimal` on
6–7 of the 19 local Netlib instances; both simplex engines reach `Optimal` on
19/19 with objectives matching `scripts/oracle_check.py`'s published table.
`SolverStatus::Infeasible` and `SolverStatus::Unbounded` became reachable
from the solver for the first time — `gas11` now returns `Unbounded` as a
certificate instead of running away to `-7.5e10`.

## 2. A GPU-native, matrix-free *interior-point* path

HiGHS is a CPU solver. Its linear algebra (sparse LU for simplex, or a
sparse Cholesky/LDL^T for its own IPM option) factors matrices explicitly.

This project's **simplex** path is also CPU and also factors explicitly — a
sparse LU is what the method requires, and claiming otherwise would be
dishonest. The difference described in this section is about the **IPM**
path, which is GPU-resident and factors nothing.

**PDLP** (§1) factors nothing either, and for a sharper reason: not because it
avoids a factorization it could have used, but because the algorithm has no
place to put one. Its entire per-iteration cost is `K x` and `K' y`. That is
also the one operation this project's hardware does well — `docs/ARCHITECTURE-
REVIEW.md` §3.5 measures FP64 factorization on this GA107 at roughly **64×**
penalised, slower than the CPU, against SpMV at **~2×** because it is
bandwidth-bound rather than FLOP-bound. Whether that theoretical fit turns into
a measured win is **still unanswered**: the cuSPARSE backend is not written.
It is listed as not-done in `docs/spec/module.txt` §24 rather than claimed here.

The IPM's linear algebra is **matrix-free and GPU-resident**
(`src/solver/gpu/LinearSolver.cu`): every Newton system is solved by an
iterative Krylov method that only ever *applies* the matrix as a sequence of
sparse matrix-vector products (cuSPARSE SpMV) — the KKT matrix itself is
never factored.

- **`solve_spd_cg`** — Conjugate Gradient on the normal-equations reduction
  `A*T*A' + D_s + delta_d*I`, used whenever the problem is a pure LP (`Q`
  empty), preconditioned by a **from-scratch incomplete Cholesky, IC(0)**
  factorization (`src/solver/gpu/Preconditioner.cu`) built directly against
  cuSPARSE's `csric02`/`csrsv2` *primitive* factor/triangular-solve calls —
  explicitly not a complete sparse solver, which is what the "from scratch"
  constraint rules out linking. IC(0) drops fill from dense columns
  deliberately (documented, bounded tradeoff) rather than forming the O(k²)
  blowup a dense column would otherwise cause in `A*A'`.
- **`solve_minres`** — MINRES on the full symmetric quasi-definite augmented
  KKT system, used whenever `Q` is non-diagonal (QP) or the augmented path is
  otherwise required, preconditioned with block-Jacobi.

Why this matters architecturally: a direct factorization needs ordering +
symbolic + numeric factorization machinery that would essentially reimplement
what a library like cuDSS provides — exactly what PS 26119 rules out linking.
Matrix-free Krylov methods sidestep that requirement entirely while still
running natively on the GPU, which a CPU-only sparse-factorization solver
like HiGHS was never designed to do.

## 3. Our own presolve module, built by direct derivation

HiGHS ships a large, mature presolve engine developed over years. This
project's presolve (`src/solver/Presolver.cpp`, `docs/spec/module.txt` §4) was built
from scratch this project cycle, rule by rule, each one algebraically derived
and independently unit-tested (`tests/property/presolver_test.cpp`) before
being trusted on the real corpus — not ported from any reference
implementation:

| Rule | What it does | Derived from |
|---|---|---|
| Empty row / column | Drop vacuous rows; fix or detect unboundedness on inert columns | First-principles feasibility check on zero activity |
| Free-column singleton | Substitute a free variable that is the sole entry of an equality row | Solving the row's equation for that one variable |
| Singleton-row tightening | Turn one entry of a row into a bound on its column, fixing it when the bound collapses | Isolating the row's single unknown |
| Duplicate-column merging | Merge two columns with identical constraint pattern *and* identical cost into one, Minkowski-summing their bounds | Recognizing `x_j`/`x_k` are objective- and constraint-interchangeable |
| General singleton-column elimination | Fix or eliminate a bounded (not just free) singleton column in an inequality row, case-split on cost/coefficient sign | Slack-maximizing / cost-improving-direction analysis, generalizing the free-column rule |

Each of these pushes onto a `TransformStack` with its own recovery math
(`src/model/Canonicalizer.cpp::recover_solution`) — e.g., duplicate-column
recovery has to split one merged canonical value back into two original
variables and share one stationarity value between two separate dual
variables. That recovery derivation, and the two real bugs it surfaced along
the way (a stale dual mapping on any Presolver-dropped column, and a fold
that skipped a singleton row's own residual), came from tracing this
project's own math against this project's own test corpus, not from reading
HiGHS's presolve source.

## 4. Our own initialization and step strategy

HiGHS's IPM option (and most production IPMs) use Mehrotra's full
least-squares starting-point heuristic. This solver currently uses a simpler,
explicitly-documented bound-midpoint heuristic
(`src/solver/Initializer.cpp`) — with one refinement made directly from
observing this project's own solver's behavior on the Netlib corpus: the
initial slack for each inequality row is chosen to satisfy that row *exactly*
given the already-chosen `x` (`s_i = b_i - (Ax)_i`) rather than a flat
constant, because the flat constant was measured (not assumed) to leave the
primal residual at ~4×10⁵ on `80bau3b` from iteration 0. That fix, the
predictor-corrector Mehrotra cross-term safeguard in
`src/solver/gpu/PredictorCorrector.cu`, and the escalate/decay regularization
controller (`include/sovsolve/solver/Regularization.hpp`) were all written
and tuned against this project's own convergence logs — none of it is HiGHS
code or a port of HiGHS's internal heuristics.

## 5. What's intentionally the same

Some things are the same on purpose, because they're not implementation
choices at all:

- **The reporting convention** (positive dual for a `<=` row in a
  minimization) matches HiGHS/CPLEX/Gurobi deliberately, documented in
  `Solution.hpp` — so a user comparing output side-by-side isn't confused by
  an arbitrary sign flip. Matching a *convention* is not the same as reusing
  an *algorithm*.
- **Vocabulary** (`nnz`, `presolve`, `iteration`, `primal`/`dual
  infeasibility`, `Optimal`/`NotConverged` style status names) is standard
  across the entire field — HiGHS, CPLEX, Gurobi, SCIP, and the Netlib LP
  literature all use it. Every solver's log looks superficially similar for
  the same reason every calculator's `+` looks the same.

## 6. Honest scope of the comparison

This is not a claim that this solver is faster or more robust than HiGHS
today — HiGHS is a mature, heavily-optimized production solver, and this
project's IPM is not yet at that level on the harder Netlib instances
(`README.md`'s benchmarks and this project's own corpus sweeps are explicit
about which instances still don't converge). The claim is narrower and
verifiable directly from the source: every algorithm that produces this
solver's answer — presolve, the matrix-free Krylov linear algebra, the IC(0)
preconditioner, initialization, the predictor-corrector iteration — was
derived from its mathematical definition and tested against this project's
own corpus, with HiGHS used only, and only ever, to check the answer was
right.
