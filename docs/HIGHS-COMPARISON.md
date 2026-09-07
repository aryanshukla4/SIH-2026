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

## 1. Different algorithm family, not just a different implementation

**HiGHS** defaults to **dual simplex** for LP (with its own IPM available as
an option, plus a MIP branch-and-bound layer on top). Simplex moves between
vertices of the feasible polytope one pivot at a time.

**This solver** is a **primal-dual interior-point method (IPM)** exclusively
— a predictor-corrector (Mehrotra-style) Newton iteration that moves through
the *interior* of the feasible region, staying strictly inside every bound
until convergence. There is no simplex code anywhere in this project. The
two families have different iteration counts, different failure modes, and
different reasons for needing presolve — none of that is shared code, it's a
different mathematical approach entirely.

## 2. GPU-native, matrix-free by construction

HiGHS is a CPU solver. Its linear algebra (sparse LU for simplex, or a
sparse Cholesky/LDL^T for its own IPM option) factors matrices explicitly.

This solver's linear algebra is **matrix-free and GPU-resident**
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
project's presolve (`src/solver/Presolver.cpp`, `module.txt` §4) was built
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
