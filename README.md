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
