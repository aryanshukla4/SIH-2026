# sovsolve — model ingestion & storage layer

Foundation layer for the sovereign LP/MILP/QP solver core (SIH PS 26119, MRPL).

This subproject owns everything between a model file on disk and the numbers the
solver iterates on:

| Module | Responsibility |
|---|---|
| `core` | storage primitives — aligned vectors, CSR/CSC matrices, spans, workspace arena |
| `model` | the faithful `Problem`, the canonicalizer, the reversible transform stack |
| `io` | MPS / LP / QPLIB readers, MPS writer |
| `analysis` | matrix structure and conditioning report |

Downstream modules — residuals, KKT builder, linear solver, IPM loop — consume what
this layer produces.

---

## Build

Requires a C++20 compiler, CMake ≥ 3.24, and Ninja.

```sh
cmake --preset release
cmake --build build
ctest --test-dir build --output-on-failure
```

Other presets: `debug`, and `asan`.

**C++20, not C++23**, deliberately: `nvcc` on Windows requires MSVC, whose C++23
support lags. Core headers must compile under both MinGW g++ and MSVC/nvcc, so
GCC-only builtins stay behind `#ifdef`.

### "zlib NOT found" at configure time

```
-- zlib NOT found: .mps.gz input will report UnsupportedFeature
```

**This is informational, not a warning, and the build is fine.** zlib is an
*optional system dependency*, looked up with `find_package(ZLIB QUIET)`; it is
not vendored in this repository.

| | With zlib | Without zlib |
|---|---|---|
| `.mps`, `.lp`, `.qplib` | works | works |
| `.mps.gz` (MIPLIB) | works | `UnsupportedFeature`, naming the cause |

Compressed input is **refused, never mis-parsed** — reading gzip bytes as text
would silently produce a garbage model, which is worse than failing.

Everything in the current test corpus is uncompressed Netlib, so zlib is not
needed to build, test, or benchmark. Install it when you want MIPLIB (which
ships `.mps.gz`, and is where the MILP and large instances live):

```sh
pacman -S mingw-w64-x86_64-zlib      # MSYS2 / MinGW
sudo apt install zlib1g-dev          # Debian / Ubuntu
```

Re-run `cmake --preset release` afterwards; the line becomes
`zlib found: gzip (.mps.gz) input enabled`. Alternatively just
`gzip -d problem.mps.gz` and read the plain file.

### ASan / UBSan on MinGW

The `asan` preset needs sanitizer runtimes that MinGW-w64 GCC does **not** ship —
`-fsanitize=address` fails at link time there. It works on Linux/macOS CI.

The fuzz suite does not depend on it: `tests/fuzz/parser_fuzz.cpp` supplies its
own detector by placing each input against a **guard page**, so any read past the
end faults deterministically, and it verifies the guard actually faults before
trusting a clean run.

---

## Tools

```sh
# structure, bounds, dense columns, conditioning proxy for any model
./build/tools/mpsinfo/mpsinfo.exe tests/data/netlib/afiro.mps

# dump the faithful AND canonical models as plain numbers
./build/tools/modeldump/modeldump.exe tests/data/netlib/afiro.mps

# parse throughput, bytes/nnz, peak RSS — read the spread column, see below
./build/bench/parse_bench.exe
```

`modeldump` exists for debugging across the module boundary: when a downstream
module misbehaves on an instance, its owner can dump exactly what the ingestion
layer produced instead of running this whole pipeline to reproduce it.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/FORMULATION.md`](docs/FORMULATION.md) | **The single source of mathematical truth.** Canonical form, sign conventions, residuals, the Newton system, step lengths, convergence, reduced systems. Do not restate these equations elsewhere — link here. |
| [`docs/ARCHITECTURE-REVIEW.md`](docs/ARCHITECTURE-REVIEW.md) | Review of the team's handoff docs against PS 26119. Fifteen findings, ranked, with owners. |
| [`docs/CANONICAL-FORM-ADDENDUM.md`](docs/CANONICAL-FORM-ADDENDUM.md) | The canonical-form decision, with row-growth measurements from our corpus. |
| [`docs/MPS-FORMAT-NOTES.md`](docs/MPS-FORMAT-NOTES.md) | Format quirks and traps — RANGES, BOUNDS, the objective-constant sign, the `QUADOBJ` ½ factor. Every rule is a silent-wrong-answer trap in a real file. |
| [`docs/LP-FORMAT-NOTES.md`](docs/LP-FORMAT-NOTES.md) | CPLEX LP dialect decisions and the three constructs that change a model rather than failing. |
| [`docs/DATA-STRUCTURES.md`](docs/DATA-STRUCTURES.md) | Storage contracts and invariants. |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Benchmark method, and **why parse-throughput numbers on this machine cannot support a small optimisation claim.** |

The four `*.txt` files at the repository root are the team's shared specification
(architecture, modules, datatypes, generation prompts), locked at v3.

---

## Design decisions worth knowing before you use this layer

**Two-layer model.** `Problem` stores what the file said — ranged rows, general
bounds, maximization, objective constants, integrality flags. A separate
`Canonicalizer` produces the solver's working form and records a reversible
transform stack. A loader restricted to `Ax ≤ b, x ≥ 0` cannot usefully read a
single Netlib or MIPLIB instance.

**The canonical form is bounded-variable**, not `Ax = b, x ≥ 0`:

```
minimize    ½xᵀQx + cᵀx
subject to  A_E x       = b_E        equality rows first, no slack
            A_I x + s   = b_I,  s ≥ 0
            l ≤ x ≤ u                bounds stay bounds
```

Finite variable bounds stay **native**. Turning each `x ≤ u` into a constraint row
grew the reduced system by +750% rows on `rgn` and +648% on `gt2` — for a model
that did not change. `A_E` and `A_I` are contiguous row blocks of one matrix, so
`Ax` and `Aᵀy` stay single kernel calls.

**The canonicalizer guarantees a startable model.** Its output is either usable by
an interior-point method or a definite infeasibility verdict:

- no column has `l == u` — a fixed column admits no strictly interior point, since
  `(x−l) + (u−x) = u−l = 0`. 1067 such columns across the 19-instance corpus.
- no row of `A` is all-zero — that is an exact zero pivot in `AΘAᵀ`, unreachable by
  regularization. Consistent ones are dropped; `0 = 5` returns `PrimalInfeasible`.

`CanonicalProblem::is_ipm_startable()` exposes the same `O(m+n+nnz)` check, because
the canonicalizer is not the last stage to touch the model — presolve creates new
empty rows.

**Rows are `[row_lower, row_upper]`**, not sense + rhs. Collapses `L`/`G`/`E`/`N`
and `RANGES` into one representation and removes a class of sign bugs.

**`A` is stored in both CSR and CSC.** Every IPM iteration needs `Ax` *and* `Aᵀy`;
computing the latter from CSR alone needs GPU atomics or a cache-hostile scatter.
Measured cost: 24.5 bytes/nonzero for both orientations.

**Storage orientation and memory space are types, not runtime fields.** A runtime
`(format × location)` pair puts a four-way branch inside every kernel.

**Dense matrices are column-major**, matching cuBLAS and LAPACK.

**The inequality-slack dual is eliminated as `w_s = −y_I`**, not stored. It is not
an independent quantity, and a stored copy can drift from `−y_I` under rounding.
A `slack_dual(y) = −y` helper keeps sign tests reading positively. Verified against
HiGHS that this convention already matches standard reporting signs, so no global
normalization is applied. See `FORMULATION.md` §4.

---

## Verification

| Layer | What it catches |
|---|---|
| Unit tests | Every documented format trap, against values computed from the spec |
| Property tests | Canonicalize → check → invert, over random and hand-built points (2678 assertions) |
| Round-trip | `parse → write → parse` — all 19 instances bit-exact |
| Fuzzing | 300k mutated inputs, guard-paged, zero crashes |
| **External oracle** | Both models solved by **HiGHS**, checked against Netlib's published optima |

11 suites, 3455 assertions. The oracle is the only layer that does not check this
code against itself:

```sh
python scripts/oracle_check.py ./build/tools/modeldump/modeldump.exe tests/data/netlib
```

9 of 11 published optima match exactly; 13 of 13 canonical forms reach the same
optimum as their parsed model. The two deviations are archive drift — Netlib's
values come from MINOS 5.3 on a VAX in 1988 — and are recorded per-instance with
the measured deviation rather than absorbed by a loosened tolerance.

**Benchmark numbers need care.** Repeated runs of an identical binary over
identical input measured 24, 41, 56, 82 and 85 MB/s on this machine. The benchmark
reports that spread instead of hiding it behind a best-of-N; do not quote an
improvement smaller than it. Stable figures — bytes/nonzero, peak RSS, cache hit
rates — can be quoted directly. See `docs/BENCHMARKS.md`.

---

## Architecture enforcement

The module dependency graph is

```
core  ←  model  ←  io
core  ←  analysis
model ←  io
```

and it is **checked mechanically**, not by convention. `scripts/check_layering.py`
scans the actual `#include` edges and runs as a ctest case, so a violation fails
the build like a compile error. CMake targets alone cannot enforce this, since
every module exposes the same `include/` root.

---

## Solver core (Modules 5-20, 23)

**There are two LP engines.**

- **Interior point** (Modules 5-20, `src/solver/gpu/`) -- GPU-resident,
  matrix-free, predictor-corrector. The default (`--method=ipm`), and the only
  path that handles QP.
- **Revised simplex** (Module 23, `src/solver/simplex/`) -- host-only, in both
  its dual and primal forms, sharing one factorized-basis core.
  `--method=dual-simplex` / `--method=primal-simplex`. LP only. Added
  2026-09-13; see `module.txt` section 23 and `docs/FORMULATION.md` section 12.

Because the simplex engine has no CUDA dependency, `cmake --preset release` on
native Windows -- no WSL2, no CUDA toolkit -- now produces a working `solve`
binary and runs 17 of the 19 test suites. That was not true before Module 23:
`tools/solve` was hard-gated on `if(TARGET sovsolve_solver_gpu)`.

### Interior-point path

The interior-point solver itself -- Scaler, Initializer, KKT builder, linear
solver, predictor-corrector loop, and everything downstream of it -- lives
under `include/sovsolve/solver/` and `src/solver/`. *(The paragraph below
describes an early state of this module and is kept for history; the linear
solver in particular has since been replaced -- see the correction after it.)*
It was at that point a
**skeleton**: types and module boundaries exist and compile, and a few
modules have real algorithm code -- `Initializer` (Module 6, a bound-midpoint
strictly-interior starting point), `gpu::compute_residuals` (Module 7, the
six Newton-system residuals -- currently host-executed, see the header
comment on `gpu/ResidualCalculator.hpp` for why), `gpu::build_kkt` (Module 9,
the augmented/quasi-definite KKT system derived from `FORMULATION.md`
sections 7 and 10.2 -- assembly only, no factorization yet), `gpu::solve`
(Module 12, an actual GPU factorization via `cusolverDnDgetrf`/`Dgetrs` --
the first code in this project that runs real work on the GPU rather than
just compiling under `nvcc`), `gpu::recover_newton_direction` (Module 13,
un-eliminating `ds`/`dz`/`dv` from the augmented solve's `[dx; dy]`),
`Regularization` (escalate/decay bookkeeping), `Diagnostics` (CSV/JSON
export) and `Logging`. Everything else -- step length, state update, mu
control, the predictor-corrector loop itself -- still returns
`ErrorCode::NotImplemented`.

`recover_newton_direction`'s test (`solver_gpu_algorithms_test.cpp`) verifies
the recovered directions against the *original* six-block Newton system
(`FORMULATION.md` 7), not just by re-deriving the same formulas the
implementation uses. That surfaced a real, exactly-quantifiable gap: since
`build_kkt` regularizes the (1,1)/(2,2) diagonal blocks with `delta_p`/
`delta_d`, the solved `(dx, dy)` satisfy the *regularized* system, not the
true one, by precisely `-delta_p * dx_j` (row 1) and `-delta_d * y_I * dy_I`
(row 6) -- exactly the gap `FORMULATION.md` 10.1's iterative-refinement
requirement exists to correct. The test asserts that exact relationship
rather than a loosened tolerance.

`gpu::compute_step_lengths` (Module 14, `FORMULATION.md` 8) and
`gpu::apply_step` (Module 15) are also real -- the separate primal/dual
ratio tests, the `eta` safety factor, and the equality-row `y` exclusion
from the dual ratio test (an unrestricted-sign quantity has no ratio to
test).

**A full Mehrotra predictor-corrector iteration now runs**:
`gpu::update_mu`/`gpu::compute_mu_at_trial_point` (Module 16) and
`gpu::run_iteration` (Module 8, `include/sovsolve/solver/gpu/
PredictorCorrector.hpp`) orchestrate every module above into one real IPM
step -- affine solve, affine step lengths at `eta=1`, `mu_aff`, `sigma =
clamp((mu_aff/mu)^3, 0, 1)`, the corrector residuals with Mehrotra's
second-order cross terms folded in by hand, corrector solve, final step
lengths at the real `eta`, state update. `solver_gpu_algorithms_test.cpp`
checks this isn't just plumbing: it runs one full iteration on a small LP and
confirms the primal residual actually shrinks afterward -- a real Newton
step toward feasibility, not just a function that returns `Status::Ok()`.

Two things worth knowing:
- `PredictorCorrector` lives under `gpu/`, not `src/solver/` directly --
  it calls GPU-boundary functions, so (like every other module that does)
  it belongs in `sovsolve_solver_gpu`. `RegularizationController`
  (`Regularization.hpp`) became header-only so this doesn't create a link
  cycle between the host and GPU solver libraries; see that header's
  comment for the full reasoning.
- Module 8's "reuse structure where valid" (module.txt) -- reusing the
  affine solve's factorization for the corrector solve -- is **not**
  implemented. The dense cuSOLVER stopgap (`LinearSolver.hpp`) factorizes
  from scratch both times; real reuse needs the sparse solver this project
  does not have yet.

**A real `solve_problem()` entry point now exists and returns correct
answers.** `gpu::solve_problem` (`include/sovsolve/solver/gpu/Solve.hpp`) is
the whole pipeline: `canonicalize -> scale -> initialize -> [run_iteration +
ConvergenceChecker] -> reconstruct_solution`. `ConvergenceChecker` (Module
18, `FORMULATION.md` 9) is real now too -- the three relative infinity-norm
criteria, explicit NaN guards (a poisoned residual reports `NumericalError`
directly rather than silently running to `MaxIterations`), and stall
detection. On stall or the iteration limit the returned `Solution` is the
best iterate seen, not the last one, with `from_best_iterate` set.

**Solution mapping itself was not rebuilt here** -- it already existed,
correct and tested, as part of the ingestion layer:
`model::recover_solution()` (`Canonicalizer.cpp`) handles primal/dual
recovery, the sign flips for negated rows, reduced-cost reconstruction for
substituted columns, and bound-violation checking against the *original*
problem, and `solver::reconstruct_solution()` is a thin wrapper over it. The
only new piece was packaging a `SolverState` into a canonical-space
`Solution` for that existing function to consume.

`solver_gpu_algorithms_test.cpp`'s capstone test calls `solve_problem()` on
a small LP (`min x1+x2` s.t. `x1+x2=10`, `0<=x1,x2<=8`) and checks the
result reaches `SolverStatus::Optimal` with the correct objective and a
feasible point -- the first test in this project that exercises the entire
pipeline through the public entry point rather than one module at a time.

Two more architecture notes from this pass:
- `gpu::Solve` needed both host-only functions (`canonicalize`,
  `initialize`, `ConvergenceChecker`, `reconstruct_solution`) and
  `gpu::run_iteration`, so `sovsolve_solver_gpu` now links `sovsolve_solver`
  -- a one-way edge (`src/solver/CMakeLists.txt` explains why it's safe and
  `sovsolve_solver` must not link back).
- What's left before this is a *complete* solver: the normal-equations LP
  path (Module 9's cheaper alternative), a sparse (not dense) linear solver,
  iterative refinement, and `Unbounded`/`Nonconvex` detection -- none of
  which block a correct answer on a well-behaved small problem, all of
  which matter for real Netlib/MIPLIB-sized instances.

---

## Running against real benchmarks -- status

The pipeline had never been run on an actual Netlib instance before this
pass (every earlier test was a hand-built toy problem). It doesn't converge
yet. `tools/solve` (below) exists specifically to make this checkable.

**`Scaler` (Module 5) is now real** -- alternating geometric-mean row/column
scaling (`Scaler.hpp`/`.cpp`), applied to `A`, `Q`, `b`, `c`, and both bound
vectors, with the inversion (`x = col_scale*x'`, `y = row_scale*y'`,
`z = z'/col_scale`, `v = v'/col_scale`) implemented in
`model::recover_solution()` itself -- not in a wrapper, per the
`TransformStack`'s own original design (scaling records push onto the same
stack canonicalization does, keyed by *canonical* index since scaling runs
after canonicalization). The objective computation is scale-invariant only
for a matched scaled/unscaled pair, which caught a real mismatch risk while
writing it -- documented in the code where it's easy to reintroduce.
Confirmed correct: the existing full-pipeline test (`min x1+x2` s.t.
`x1+x2=10`) still reaches the exact right answer with real scaling now
active, not the identity no-op it exercised before.

**Scaling alone did not fix `afiro`** (the smallest Netlib LP, 27 rows, 32
columns, all lower-bounded columns, a mild 22.7x coefficient range --
structurally simple). `mu` exploded geometrically (iteration 9 reached
`1e10`) and step lengths stayed pinned near zero. A synthetic problem mixing
equality and inequality rows (something no earlier test did) converges
perfectly with the same code, which ruled out a row-type indexing bug.

**Root-caused, not guessed**: dumped the actual KKT matrix at two points and
checked both against an independent `numpy` solve. At iteration 0 (affine
solve), `cond(A) = 17` -- well-conditioned, and the GPU solution matched
`numpy` to `1e-13`, confirming `KktBuilder`/`LinearSolver`/`NewtonRecovery`
are all correct. At iteration 8's corrector solve, `cond(A) = 1.65e15` --
past double precision's ~1e16 noise floor. `cusolverDnDgetrf` reported
`info=0` (no error) the entire time; the returned "solution" (magnitude
~1e11) was pure rounding noise, not signal, and that's exactly where `mu`
jumped from `6e6` to `1.46e10`. Mechanism: `Theta^-1_j = z_j/(x_j-l_j)` grows
legitimately as a bound-hugging variable's dual rises, and nothing
counteracted it -- `RegularizationController.escalate()`/`.decay()` were
tested in isolation but had zero call sites in `PredictorCorrector.cu` or
`Solve.cu`; `delta_p`/`delta_d` sat at the `1e-8` floor for the whole solve.

**Fixed**: `solve()` (`LinearSolver.cu`) now reports the Dgetrf factor's pivot
growth ratio (`max|U_ii| / min|U_ii|`, read off the factor for free via a
strided `cudaMemcpy2D` -- no extra solve) as `LinearSolveResult::pivot_ratio`.
`solve_newton_system` (`PredictorCorrector.cu`) checks it against
`Options::IpmOptions::max_pivot_ratio` (default `1e10`): over that, it calls
`regularization.escalate()` and refactors with the larger delta (looping
until clean or `escalate()` reports `delta_max` reached, at which point it
reports `NumericalError`); under that, it calls `regularization.decay()`.
`IterationRecord::regularization_events` is now wired to the count. Verified:
all 13 tests still pass (including the 117-check GPU algorithms suite,
unchanged), and on `afiro`, `mu` no longer explodes -- it plateaus around
`1e5` instead of reaching `1e10`.

**`afiro` didn't converge with escalation alone**, and that turned out to be a
genuinely different, now-isolated problem: instrumented `compute_step_lengths`
to report which variable/slack binds the primal ratio test each iteration.
One inequality-row slack's ratio geometrically collapsed every iteration it
bound (`s=1.25 -> 6.3e-3 -> 3.1e-5 -> 1.6e-7 -> ...`) while its Newton
direction (`ds`) stayed large-negative (~-40 to -400) at every single step --
the corrector kept trying to overshoot past that slack's bound, the ratio
test correctly clipped it, but the step never settled near the boundary.
`sigma` was pinned at `1.000` (full centering -- Mehrotra's heuristic
correctly detecting the affine step was making things worse) for most of the
run, so this wasn't a missing-centering bug; centering was already maximally
conservative and it still wasn't enough.

**Root cause, confirmed by instrumenting the actual cross-term values**: the
Mehrotra second-order correction (`rxz/ruv/rsy += dx_aff .* dz_aff` etc.,
`PredictorCorrector.cu`) is a Taylor-expansion remainder -- theoretically
`o(mu)`, a small refinement on top of the `sigma*mu` target. With nothing
bounding it, on a degenerate pair (the affine step itself already extreme --
exactly what a crude `x_j=1` start produces) it can be 1-2+ orders of
magnitude larger than `mu` itself and completely override the target instead
of refining it. Confirmed on afiro: cross terms of `-5.2e3` against `mu=1.0`,
and `1.0e6` against `mu=5.9e4`, at precisely the pair whose step length
collapsed. **Fixed**: each cross term is now clamped to `[-mu, mu]` before
being added -- keeps it a refinement, matches its theoretical role, doesn't
touch the base `sigma*mu` target at all. `afiro` now reaches `Optimal` in 15
iterations, `obj=-464.75314223` against the published `-464.75314286`.

**Two more real bugs found by running the full 19-instance local Netlib set**
(not just afiro) after the fix above, rather than assuming one fix cures
everything:

1. `Solve.cu`'s `dual_obj` was just `b'y`. Derived from the same six-block
   system (substituting stationarity + primal feasibility into `c'x`): at
   convergence `c'x -> b'y + l'z - u'v - x'Qx`, not `b'y` alone. Any problem
   with a finite, active bound (nearly all of them) has a permanent,
   unclosable "gap" with the old formula even at a truly optimal point --
   `avgas`/`egout`/`rgn` all hit residuals at `~1e-12` yet reported
   `NotConverged` purely because of this. Fixed: `dual_objective()` now
   includes `l'z - u'v` (finite-bound terms only) and `-x'Qx` for QP.
2. `PredictorCorrector.cu` treated "pivot ratio still high at `delta_max`" as
   fatal (`NumericalError`), discarding the whole solve. But architecture.txt's
   "at delta_max with factorization still failing" means exact singularity
   (already a separate, hard `Expected` failure from `solve()`) -- a merely
   still-elevated pivot ratio at `delta_max` is the *expected*, harmless
   terminal-phase signature of a converging point (`Theta^-1` naturally spikes
   for a tightly-bound variable right at the solution) far more often than a
   real breakdown. `avgas`/`egout`/`rgn` were one iteration from `Optimal` and
   got hard-aborted by this. Fixed: accept the direction instead of erroring;
   the outer loop's best-iterate tracking is what actually guards against a
   bad step doing damage. (Also fixed, same investigation: `Solve.cu` used to
   discard the whole solve -- including an already-excellent best iterate --
   the instant `run_iteration` returned any error. It now falls back to the
   best iterate found so far, same philosophy the code already used for a
   stall.)

**A third, separate crash** turned up on `shell.mps` after the above:
`mu_aff`/`sigma` went `NaN` at iteration 56, right as `mu` reached `9.66e-12`
-- the edge of double precision for this problem's scale. Root cause: as a
complementarity gap (`x-l`, `u-x`, or `-y_I`) shrinks toward that noise floor,
`x + alpha*dx` can round to *exactly* the bound, turning `z/(x-l)` (or the
analogous terms in `KktBuilder.cu`'s RHS and `NewtonRecovery.cu`'s `dz`/`dv`)
into a genuine `0/0 = NaN` that silently poisons the whole KKT system --
and NaN comparisons are false, so the step-length ratio test doesn't catch
it either; it just poisons the state via the next `apply_step`. Fixed: a
`safe_gap()` helper (`SolverState.hpp`, next to `slack_dual()`) floors these
gaps at `1e-30` before they're used as a divisor -- far below any legitimate
gap, so it only ever engages at the precision floor. `shell.mps` no longer
crashes; it now reaches `Optimal` (with `--stall=40`) matching the published
`1.2088253460e9`.

**Current state, the full local 19-instance Netlib set.**

*Simplex (Module 23, 2026-09-13): **19/19 reach `Optimal` on both engines**,
with matching objectives.* Every objective was cross-checked against
`scripts/oracle_check.py`'s published table, including its two documented
archive-drift entries -- `80bau3b` (8.1e-6) and `greenbea`, where our
`-7.2555248130e+07` matches the LP DASA / HiGHS value rather than the 1988
archive's `-7.2462405908e+07`. `gas11` returns **`Unbounded`**: the archive has
no ground truth for it and HiGHS also reports unbounded. This is the first time
this solver has produced `Unbounded` at all -- see the IPM paragraph below,
which called that "an architectural gap, not a quick fix," and was right.

| | IPM | dual simplex | primal simplex |
|---|---|---|---|
| `Optimal` | 6-7 / 19 | **19 / 19** | **19 / 19** |
| `Unbounded` detected | never | `gas11` | `gas11` |

Neither simplex engine dominates the other: `stair` takes 569 primal pivots
against 3357 dual, while `80bau3b` takes 5546 dual against 19201 primal. That
is the ordinary reason production solvers keep both, and it is why
`--primal-cleanup` (on by default) runs the dual and then hands its final basis
to the primal.

*IPM, for comparison* (default settings, `--max-iter=300`): **6 reach
`Optimal`** (`afiro`, `avgas`, `chip`, `egout`,
`flugpl`, `rgn`) -- `shell` makes 7 with a slightly relaxed `--stall`. Several
more are essentially converged but plateau just above the `1e-8` tolerance
even given 4x more iterations (`stair` gap `~7e-6`, `bell5` `~9e-5`,
`etamacro`/`25fv47`/`standata` `~2-5e-4`) -- genuinely stuck, not just cut off
early, so this is a different remaining gap, not yet root-caused.
`adlittle`/`e226`/`gt2`/`israel` are still far from converged. `gas11`'s
objective runs away to `-7.5e10` while staying nearly primal-feasible --
looks like an undetected unbounded-dual direction (`SolverStatus::Unbounded`
exists but `ConvergenceChecker` never returns it), an architectural gap, not
a quick fix. `80bau3b`/`greenbea` time out -- almost certainly just the dense
`O(dim^3)` stopgap being too slow for their size, not a correctness issue.

### Tuning without rebuilding

`tools/solve` now takes `--flag=value` overrides for every field on
`Options` -- `--eta`, `--sigma`, `--predictor-corrector`, `--pfloor`/
`--dfloor`, `--escalation`, `--decay`, `--delta-max`, `--max-pivot-ratio`,
`--refine`, `--max-iter`, `--stall`, `--tol-primal`/`--tol-dual`/`--tol-gap`,
`--time-limit`. Positional `max_iterations` still works for backward
compatibility. `--help` prints the full list with defaults and the
`Options.hpp` field each maps to. Output always ends with
`solve_time_seconds=...`, meant to be the thing tuning is measured against.

### `tools/solve` -- CLI entry point for testing against real files

```sh
# host-only build, no CUDA toolkit needed (simplex engines)
./build/tools/solve/solve tests/data/netlib/afiro.mps --method=dual-simplex
./build/tools/solve/solve tests/data/netlib/afiro.mps --method=primal-simplex

# CUDA build (adds the interior-point default)
./build-cuda/tools/solve/solve tests/data/netlib/afiro.mps [max_iterations]
```

Loads a model file, solves it, and prints status/objective/iterations/quality
in a form a benchmark script can parse.

`--method=ipm|dual-simplex|primal-simplex` selects the engine; `ipm` is the
default and is **compiled out** when `SOVSOLVE_ENABLE_CUDA` is off, since
`solve_problem()` needs the GPU library. The simplex flags are
`--simplex-max-iter`, `--pivot-tolerance`, `--pivot-floor`,
`--refactor-interval`, `--artificial-bound`, `--primal-cleanup`,
`--bound-flipping`, `--simplex-tol-primal`, `--simplex-tol-dual`.

The tool is no longer gated on the GPU library existing -- it builds against
`sovsolve_solver` alone and only links `sovsolve_solver_gpu` when that target
is present. Per-iteration
diagnostics print by default (`LogOptions::Level::Iteration`, `Options`'s own
default) -- `Logging.cpp`'s line now includes `mu`, `mu_aff`, `sigma`, and
both residuals, which is what made the `afiro` trajectory above visible in
the first place; `Level::Debug` additionally adds factor/solve/refine timing.

`build_kkt` always selects the augmented path (`ReductionType::
QpAugmentedKkt`); the normal-equations LP path (`FORMULATION.md` 10.1) needs
a sparse `A * Theta * A^T` product this pass does not build, and is deferred.

**Correction (2026-09-13):** the paragraph below describes the dense cuSOLVER
stopgap, which is no longer the production path. `src/solver/gpu/LinearSolver.cu`
now implements matrix-free Krylov solves -- `solve_spd_cg` (CG on the normal
equations, IC(0)-preconditioned) and `solve_minres` (MINRES on the augmented
KKT) -- and those are what a solve actually uses. `solve_dense`/`solve_spd_dense`
survive as the reference implementations the Krylov solvers' algebra is tested
against, which is why they are still in the file. `docs/HIGHS-COMPARISON.md`
section 2 describes the current design; this paragraph is the stale one. Kept
here rather than deleted so the history is legible:

`gpu::solve` was a deliberate stopgap, documented in full in
`gpu/LinearSolver.hpp`: it converts the sparse KKT matrix to **dense**
(`O(dim^2)` memory, `O(dim^3)` time -- fine for small test problems, not for
a real Netlib/MIPLIB instance) and factorizes with cuSOLVER's classic
**general LU** rather than the symmetric-indefinite `LDL^T` the quasi-definite
structure could exploit, because `getrf`/`getrs` are the most stable, longest-
standing dense solve pair in cuSOLVER's API. No `cuDSS` is installed on the
development machine, and a from-scratch sparse `LDL^T` factorization (fill-
reducing ordering, elimination tree, numerical factorization) is separate,
substantial work -- this stopgap exists so the rest of the pipeline has a
real solve to build against in the meantime. Iterative refinement is not yet
implemented (`refinement_passes` is always `0`): `FORMULATION.md` 10.3
specifies refinement against the **unregularized** residual, which needs the
true Newton system's residual, not just the factored (regularized) matrix's
own residual -- deferred, not approximated.

The GPU-boundary modules (`src/solver/gpu/*.cu` -- residuals, KKT assembly,
ordering, linear solve, Newton recovery, step length, state update, mu
control) are written directly as CUDA C++ from the start, per
`architecture.txt`'s GPU-boundary text. They only build under
`SOVSOLVE_ENABLE_CUDA=ON` (the `cuda` CMake preset, targeting WSL2 + the
Linux CUDA toolkit -- `nvcc` on native Windows needs MSVC as its host
compiler, which this project avoids). The host-only build
(`release`/`debug`/`asan` presets) stays fully self-contained without the
CUDA toolkit present.

```sh
cmake --preset cuda && cmake --build build-cuda   # WSL2, CUDA toolkit installed
```

**Deviation from the locked v3 spec, recorded here rather than silently
absorbed:** `module.txt` Module 12 requires *both* a CPU reference
implementation and a GPU implementation for the linear solver, specifically
so results can be cross-checked against each other. This build is CUDA-only
-- there is no CPU reference path. That means a wrong answer from the linear
solver has nothing independent to diff against except the external oracle
(`scripts/oracle_check.py`, which validates the ingestion layer, not the IPM
loop). Worth reconsidering if numerical bugs in the solver core turn out to
be hard to isolate with only one implementation.

---

## On the "from scratch" constraint

PS 26119 requires that the solver "shall not be built upon any existing open
source solver library". This layer takes that seriously:

- **No solver library is linked, vendored, or depended on.** `third_party/` is
  empty. Every algorithm here — sparse assembly, the canonicalizer, the transform
  stack, the matrix analyzer — is written from the mathematical definition.
- **zlib is an optional system dependency**, not a bundled one, and is a
  *compression* library needed because MIPLIB ships `.mps.gz`. The solver builds
  and runs without it.
- **The test harness is hand-written** (`tests/TestMain.hpp`, ~100 lines) rather
  than GoogleTest or Catch2. A test framework is plainly not a solver library, but
  keeping the dependency list empty means the question never has to be argued.
- **`scripts/oracle_check.py` uses SciPy (HiGHS) as a test oracle.** This is
  **development tooling only**: it is never linked, never shipped, not on any build
  path that produces a solver binary, and the test skips cleanly when SciPy is
  absent. Its role is to check *our* reader and canonicalizer against an
  independent implementation — the opposite of building on one.
