# sovsolve

**A from-scratch optimization solver for linear, mixed-integer and quadratic
programs — CPU and GPU — built for SIH 2026, problem statement 26119 (MRPL).**

sovsolve reads a model from a standard file, solves it with one of several
engines that were each implemented here from their published papers, and
returns the answer in the model's own terms. No open-source or commercial
solver library is linked, vendored or called at solve time — see
[On the "from scratch" constraint](#on-the-from-scratch-constraint).

```text
$ solve tests/data/netlib/afiro.mps --method=concurrent
concurrent: 4 engines on 4 threads
  dual-simplex     NotConverged        0.001s  iters=13
  pdlpx            Optimal             0.001s  iters=320      <-- winner
  primal-simplex   Optimal             0.001s  iters=16
  hsd              NotConverged        0.001s  iters=9
status=Optimal
objective=-4.6475314202e+02
...
```

(Engines that lose the race are stopped early, so `NotConverged` next to a
loser means "cancelled", not "failed".)

---

## Contents

- [What it can do](#what-it-can-do)
- [Quick start](#quick-start)
- [Which method should I use?](#which-method-should-i-use)
- [How it works](#how-it-works)
- [Results](#results)
- [How the results are checked](#how-the-results-are-checked)
- [Benchmarking it yourself](#benchmarking-it-yourself)
- [On the "from scratch" constraint](#on-the-from-scratch-constraint)
- [Repository map](#repository-map)
- [References](#references)

---

## What it can do

| Problem | Engines | Notes |
|---|---|---|
| **LP** — linear programs | dual simplex, primal simplex, PDLP, **cuPDLPx**, homogeneous self-dual interior point, GPU interior point, and a **concurrent** race of several of these | every engine returns the same `Optimal` / `Infeasible` / `Unbounded` verdicts in the model's original terms |
| **MILP** — mixed-integer | branch-and-bound on the dual simplex | presolve, cutting planes, conflict analysis, propagation, primal heuristics (details below) |
| **QP** — convex quadratic | GPU interior point | implemented and verified on small instances; not yet benchmarked at scale |

**Input formats:** MPS (fixed and free), CPLEX LP, and QPLIB. `.mps.gz` works
when zlib is installed.

**The LP engines, one line each**

| `--method=` | Algorithm | Runs on | Strength |
|---|---|---|---|
| `dual-simplex` | revised dual simplex: LU factorization, dual steepest edge, bound flipping, cost perturbation | CPU | exact vertex solutions; the most reliable engine on small and medium LPs |
| `primal-simplex` | revised primal simplex on the same machinery | CPU | wins where the dual stalls; finishes dual runs as a cleanup |
| `pdlp` | PDLP — primal-dual hybrid gradient with adaptive restarts (Applegate et al. 2021) | CPU or GPU | factorization-free: only matrix-vector products, so it scales to very large models |
| `pdlpx` | **cuPDLPx** — reflected restarted Halpern PDHG (2025) | CPU or GPU | the newest first-order method here; the one to use on **large LPs on a GPU** |
| `hsd` | homogeneous self-dual interior point (Andersen & Andersen) | CPU | proves infeasibility and unboundedness by construction |
| `ipm` | primal-dual interior point, Mehrotra predictor-corrector | GPU | also solves convex **QP** |
| `concurrent` | races dual simplex, cuPDLPx, primal simplex and HSD on separate cores; first verdict wins | CPU | you do not have to guess which engine suits the model |

**MILP features** (all on by default, each with a flag to switch it off):
presolve including column-removing reductions; Gomory mixed-integer and c-MIR
cuts plus root cover/GCD cuts; conflict analysis; domain propagation;
reliability branching; best-estimate node selection with plunging; and
primal heuristics — simple rounding, diving, the feasibility pump and RENS.

---

## Quick start

### 1. Build — CPU only (Windows or Linux)

Needs a C++20 compiler, CMake ≥ 3.24 and Ninja.

```sh
cmake --preset release
cmake --build build
ctest --test-dir build --output-on-failure      # 29 test suites
```

Optional, Linux/WSL only: install SuiteSparse first and the CPU interior point
(`--method=hsd`) factors its normal equations with CHOLMOD. Without it, the
in-house sparse LDL' does the same job, only slower on the largest models.

```sh
sudo apt-get install -y libsuitesparse-dev   # then re-run cmake --preset ...
```

### 2. Build — with the GPU engines (Linux or WSL2 + CUDA toolkit)

```sh
export PATH=/usr/local/cuda/bin:$PATH
cmake --preset cuda
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure # 32 test suites
```

A CPU-only build still has every engine except the GPU ones, and says so
plainly if a GPU method is requested.

### 3. Solve something

```sh
# let the solver race engines and keep the fastest verdict
./build/tools/solve/solve tests/data/netlib/afiro.mps --method=concurrent

# a mixed-integer model: branch-and-bound
./build/tools/solve/solve model.mps --method=dual-simplex

# a large LP on the GPU
./build-cuda/tools/solve/solve big.mps --method=pdlpx --gpu-resident=1

# every option, with its default and why
./build/tools/solve/solve
```

The output is one `key=value` line per quantity — status, objective, iterations,
residuals, and the time spent in each pipeline stage — so it is easy to read
by eye and trivial to parse. On Windows the binary is `solve.exe`.

---

## Which method should I use?

| Your model | Use | Why |
|---|---|---|
| Small or medium LP | `--method=concurrent` | Races several engines; measured **1.71×** faster than always using the best single engine (see [Results](#results)) |
| **Large LP, GPU available** | `--method=pdlpx --gpu-resident=1` | Matrix-free, whole iteration on the GPU, replayed as CUDA graphs |
| Large LP, CPU only | `--method=pdlpx` | Same algorithm on the CPU |
| Mixed-integer | `--method=dual-simplex` | Runs branch-and-bound |
| Convex QP | `--method=ipm` (CUDA build) | The interior-point path handles the quadratic term |
| Need a certificate that the model is infeasible or unbounded | `--method=hsd` | The homogeneous embedding produces these verdicts by construction |

Nothing chooses the GPU for you: GPU engines are used only when you ask for
them, so a run's hardware is always explicit.

---

## How it works

Every engine sits in the middle of the same pipeline. Everything before and
after it is shared, which is why all engines report answers in the same form.

```text
 model file ─► parse ─► canonicalize ─► presolve ─► scale ─► ENGINE ─► undo scaling,
 (MPS/LP/     (keeps     (one standard   (remove     (equilib-           presolve and
  QPLIB)      what the    form, every     redundant   rate A)             canonical form
              file said)  step recorded   rows and                        ─► answer in the
                          so it can be    columns)                          model's own
                          undone)                                           variables
```

- **Two models, not one.** The parser keeps exactly what the file said. A
  separate canonicalizer converts it to the solver's standard form and records
  every transformation on a stack, so the final answer is reconstructed
  exactly, including dual values and ranged rows.
- **One canonical form for every engine:** equality rows first, inequality rows
  with slacks, and variable bounds kept as bounds rather than turned into extra
  rows (which would have grown some models by over 700%). The mathematics is
  written down once, in [`docs/FORMULATION.md`](docs/FORMULATION.md).
- **The GPU path keeps the whole iteration on the device.** For `pdlpx` the
  iterate, the restart decision and the primal-weight update all live on the
  GPU; each block of 40 iterations is issued as a single captured CUDA graph,
  and the host reads results back only at termination checks.
- **Deterministic.** GPU reductions use a fixed summation order (no atomics),
  so a GPU run is reproducible bit for bit, and it takes the same number of
  iterations as the CPU run of the same method.

---

## Results

Measured on the development machine: a laptop with an NVIDIA **GeForce RTX
3050 Laptop GPU (4 GB)**, CUDA builds under WSL2. Every number below is
reproducible with the scripts in [Benchmarking it yourself](#benchmarking-it-yourself);
the full measurement notes, including what did *not* work, are in
[`module.txt`](module.txt).

### Correctness on the Netlib LP library (99 models)

| | |
|---|---|
| Pure-LP models in `tests/data/lp` | 94 |
| Reached a correct verdict with the dual simplex | **94 / 94** — 93 `Optimal`, 1 `Unbounded` (`gas11`, which genuinely is) |
| Limit | 240 s per model |

The remaining 5 of the 99 are mixed-integer models (`bell5`, `egout`,
`flugpl`, `gt2`, `rgn`), for which the engines above solve the LP relaxation
and branch-and-bound solves the integer problem.

### The concurrent race

Over 17 Netlib models, racing the engines was **1.71× faster** than always
running the single best engine (dual simplex: 3.015 s total → race: 1.760 s),
because no single engine wins everywhere — on one model the spread between the
fastest and slowest engine was over 3000×. *(Measured when PDLP held the
first-order slot in the race; it is now cuPDLPx.)*

Honest limit: on the very largest models the racing engines compete for memory
bandwidth, and the race was measured about 2× slower than the dual simplex
alone on `dfl001`, `fit2p` and `pilot87`.

### First-order methods on the GPU

On `datt256` (262,000 columns, 95 MB):

| Configuration | Engine time |
|---|---|
| PDLP on the CPU | 25.15 s |
| PDLP on the GPU | 6.70 s |
| cuPDLPx on the CPU | 5.84 s |
| **cuPDLPx on the GPU** | **2.2 s** |

What made the GPU path fast, each step measured separately:

- **cuPDLPx's algorithm** itself needed 4.8× fewer matrix passes than PDLP on
  the 13 Netlib models both solved, and solved 3 that PDLP could not.
- **Estimating ‖A‖₂ on the GPU** instead of through host copies removed about
  0.5 s of start-up on `datt256`.
- **CUDA graphs** — replaying each 40-iteration block as one launch — cut
  kernel launches 40× and made the GPU **3.3–4.8× faster** on medium models
  (e.g. `stair` 23.1 s → 5.4 s), with results **bit-identical** to launching
  the kernels one by one.

On small models (a few thousand variables) the CPU remains faster: a GPU run
pays about 1.5 s of fixed start-up. That is why the GPU is opt-in and why
`concurrent` is the recommendation for small and medium models.

*Note on these timings:* on 2026-09-27 the cuPDLPx defaults were changed to
published constants (see below); the timings above were taken with the
previous defaults. Re-run `scripts/benchmark.py --paper --gpu` for current
numbers.

### What a larger GPU should change

Not measured here — a prediction, stated as one. PDLP-type methods spend
their time streaming the matrix from memory, so they are limited by memory
bandwidth: roughly 190 GB/s on the development GPU against 1.5–2 TB/s on an
A100, which suggests a large speed-up per iteration on large models. An A100's
40–80 GB of memory also fits models that do not fit in 4 GB. Small models and
file loading, which run on the CPU, would not change.

---

## How the results are checked

- **Engines against each other.** PDLP and cuPDLPx are tested against the
  exact answer of the simplex on the same models, and the concurrent race
  against each engine run alone — racing may change how fast the answer
  comes, never what it is.
- **Against published answers.** `scripts/oracle_check.py` compares against
  Netlib's published optimal values and against an independent solver used
  purely as a reference (below).
- **Against brute force.** Mixed-integer tests compare branch-and-bound with
  exhaustive enumeration, for every branching rule.
- **GPU against CPU.** Every GPU operation is tested against its CPU
  counterpart, step by step, including across restarts; CUDA graph replay is
  tested for bit-identical results.
- **Mutation testing.** For key components, deliberately broken versions were
  run against the tests to confirm the tests actually catch the breakage.
- **Fuzzing.** The file readers survived 300,000 mutated inputs with no
  crashes.
- **Architecture.** A test fails the build if any module includes code from a
  layer it should not depend on.

Current state: **28/28** test suites pass on the CPU build and **31/31** on
the CUDA build.

### Where the constants come from

The first-order methods have tuning constants, and the policy is that each
default is **either taken from a published paper or measured here and
documented as such** — never quietly guessed. For cuPDLPx: the step size,
reflection and tolerances are from the cuPDLPx paper; the restart constants
and the primal-weight rule are from HPR-LP, which has been proven to generate
exactly the same iterates as cuPDLPx under the settings used here. Each
choice, and the measurement behind it, is written next to it in
[`include/sovsolve/model/Options.hpp`](include/sovsolve/model/Options.hpp).

---

## Benchmarking it yourself

```sh
python scripts/benchmark.py                   # Netlib, default settings, one CSV
python scripts/benchmark.py --compare         # every engine side by side
python scripts/benchmark.py --gpu             # CPU vs GPU
python scripts/benchmark.py --paper           # the cuPDLPx paper's own protocol
python scripts/benchmark.py --corpus mip --node-limit 2000
```

The script finds the solver binary itself, records the machine, GPU and git
commit in every row, and keeps every run in the CSV — including timeouts and
failures — so a result cannot be quietly filtered. `--paper` follows the
cuPDLPx paper's rules exactly: both of its tolerances (10⁻⁴ and 10⁻⁸),
convergence checked on the original rather than the rescaled model, a time
limit instead of an iteration cap, and its SGM10 summary metric. Details:
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

---

## On the "from scratch" constraint

PS 26119 requires that the solver *"shall not be built upon any existing open
source solver library."*

- **No solver library is linked, vendored or called at solve time.** Every
  engine — both simplex methods, PDLP, cuPDLPx, both interior-point methods,
  branch-and-bound and presolve — is implemented here from the published
  algorithm, with the paper cited in the code next to what it supports.
- **GPU libraries are used as primitives, not as solvers.** On every solve
  path, the GPU work is cuSPARSE sparse matrix-vector products plus kernels
  written here; the GPU interior point is matrix-free (conjugate gradient and
  MINRES), so it never factorizes a matrix.
  - cuSOLVER's *dense* LU and Cholesky are compiled into the GPU library, but
    only as **test oracles** — they check the matrix-free algebra in the test
    suite, and no solve path calls them.
  - cuDSS, a complete sparse direct solver, is **never called, and the build
    does not look for or link it.**
  - Why a complete sparse solver is out of bounds while cuSPARSE is in:
    [`docs/ARCHITECTURE-REVIEW.md`](docs/ARCHITECTURE-REVIEW.md).
- **A reference solver is used only to check answers.**
  `scripts/oracle_check.py` compares our results with HiGHS through SciPy, as
  a development-time test. It is never linked, not on any build path, and
  skipped when SciPy is absent. [`docs/HIGHS-COMPARISON.md`](docs/HIGHS-COMPARISON.md)
  explains how the designs differ. Reference copies of other solvers' sources
  kept locally for comparison are git-ignored, never built, and were not read
  when writing the engines.
- **No test framework dependency either.** The test harness is about 100
  hand-written lines ([`tests/TestMain.hpp`](tests/TestMain.hpp)).
- **zlib is optional** and only decompresses `.mps.gz` files.

---

## Repository map

```text
include/sovsolve/   public headers, one folder per module
  core/             storage: aligned vectors, sparse CSR/CSC matrices
  model/            the model, canonical form, transform stack, all options
  io/               MPS / LP / QPLIB readers, MPS writer
  solver/           presolve, scaling, every engine, branch-and-bound
    simplex/        dual and primal revised simplex
    pdlp/           PDLP and cuPDLPx, and their CPU/GPU iteration backends
    gpu/            GPU interior point and the GPU PDLP / cuPDLPx backend
src/                implementations, same layout
tools/solve/        the command-line solver
tests/              unit, property, corpus and fuzz tests; test data
scripts/            benchmark.py, oracle_check.py, fetch_netlib.py, layering check
docs/               design documents (below)
module.txt          module-by-module engineering log: what was built, how it
                    was verified, and what was measured, including dead ends
```

### Documentation

| Document | What it covers |
|---|---|
| [`docs/FORMULATION.md`](docs/FORMULATION.md) | The mathematics: canonical form, sign conventions, optimality conditions — the single source of truth |
| [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) | How to run and read the benchmarks |
| [`docs/ARCHITECTURE-REVIEW.md`](docs/ARCHITECTURE-REVIEW.md) | The design review done before the solver was written, including the compliance findings |
| [`docs/HIGHS-COMPARISON.md`](docs/HIGHS-COMPARISON.md) | How this solver differs from HiGHS |
| [`docs/SIH-VERSION-PROGRESS.md`](docs/SIH-VERSION-PROGRESS.md) | Version-by-version history: each problem found and how it was fixed |
| [`docs/MPS-FORMAT-NOTES.md`](docs/MPS-FORMAT-NOTES.md), [`docs/LP-FORMAT-NOTES.md`](docs/LP-FORMAT-NOTES.md) | File-format traps that silently produce wrong models, and how each is handled |
| [`docs/DATA-STRUCTURES.md`](docs/DATA-STRUCTURES.md), [`docs/CANONICAL-FORM-ADDENDUM.md`](docs/CANONICAL-FORM-ADDENDUM.md) | Storage contracts, and the measurements behind the canonical-form choice |
| [`docs/ENGINEERING-NOTES.md`](docs/ENGINEERING-NOTES.md) | The previous README: detailed build notes (zlib, sanitizers) and the ingestion layer's design record |

---

## References

The algorithms implemented here, as cited in the code:

- D. Applegate et al., *Practical Large-Scale Linear Programming using
  Primal-Dual Hybrid Gradient*, NeurIPS 2021 — PDLP.
- D. Applegate et al., *Faster first-order primal-dual methods for linear
  programming using restarts and sharpness*, arXiv 2105.12715 — the restart
  criterion and trust region.
- D. Applegate et al., *Infeasibility detection with primal-dual hybrid
  gradient for large-scale linear programming*, arXiv 2102.04592.
- H. Lu, Z. Peng, J. Yang, *cuPDLPx: A Further Enhanced GPU-Based First-Order
  Solver for Linear Programming*, arXiv 2507.14051.
- H. Lu, J. Yang, *Restarted Halpern PDHG for Linear Programming*,
  arXiv 2407.16144.
- K. Chen, D. Sun, Y. Yuan, G. Zhang, X. Zhao, *HPR-LP: An implementation of an
  HPR method for solving linear programming*, arXiv 2408.12179.
- K. Chen, D. Sun, Y. Yuan, G. Zhang, X. Zhao, *On the Relationships among
  GPU-Accelerated First-Order Methods for Solving Linear Programming*,
  arXiv 2509.23903.
- A. Koberstein, *The Dual Simplex Method, Techniques for a Fast and Stable
  Implementation*, PhD thesis, 2005.
- E. D. Andersen, K. D. Andersen, *The MOSEK interior point optimizer for
  linear programming: an implementation of the homogeneous algorithm*, 2000.
- T. Achterberg, *Constraint Integer Programming*, PhD thesis, 2007 —
  branch-and-bound, propagation, conflict analysis, presolve.
- T. Achterberg, R. Bixby, Z. Gu, E. Rothberg, D. Weninger, *Presolve
  Reductions in Mixed Integer Programming*, ZIB Report 16-44.
- S. Mehrotra, *On the implementation of a primal-dual interior point
  method*, SIAM J. Optimization, 1992.
