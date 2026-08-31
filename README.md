# sovsolve — model ingestion & storage layer

Foundation layer for the sovereign LP/MILP/QP solver core (SIH PS 26119, MRPL).

This subproject owns everything between a model file on disk and the numbers the
solver iterates on:

| Module | Responsibility |
|---|---|
| `core` | storage primitives — aligned vectors, CSR/CSC matrices, spans, workspace arena |
| `model` | the faithful `Problem`, the canonicalizer, the reversible transform stack |
| `io` | MPS / LP / QPLIB readers |
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

Other presets: `debug`, and `asan` (address + undefined sanitizers, which the
MPS reader's pointer arithmetic and the fuzz suite need).

**C++20, not C++23**, deliberately: `nvcc` on Windows requires MSVC, whose C++23
support lags. Core headers must compile under both MinGW g++ and MSVC/nvcc, so
GCC-only builtins stay behind `#ifdef`.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/FORMULATION.md`](docs/FORMULATION.md) | **The single source of mathematical truth.** Sign conventions, residuals, step lengths, convergence criteria. Do not restate these equations elsewhere — link here. |
| [`docs/ARCHITECTURE-REVIEW.md`](docs/ARCHITECTURE-REVIEW.md) | Review of the team's handoff docs against PS 26119. Fifteen findings, ranked, with owners. |
| [`docs/MPS-FORMAT-NOTES.md`](docs/MPS-FORMAT-NOTES.md) | Format quirks and traps — RANGES, BOUNDS, the objective-constant sign, the `QUADOBJ` ½ factor. Every rule is a silent-wrong-answer trap in a real file. |
| [`docs/DATA-STRUCTURES.md`](docs/DATA-STRUCTURES.md) | Storage contracts and invariants. |

---

## Design decisions worth knowing before you use this layer

**Two-layer model.** `Problem` stores what the file said — ranged rows, general
bounds, maximization, objective constants, integrality flags. A separate
`Canonicalizer` produces the solver's `Ax + s = b, x ≥ 0` form and records a
reversible transform stack. A loader restricted to `Ax ≤ b, x ≥ 0` cannot
usefully read a single Netlib or MIPLIB instance.

**Rows are `[row_lower, row_upper]`**, not sense + rhs. Collapses `L`/`G`/`E`/`N`
and `RANGES` into one representation and removes a class of sign bugs.

**`A` is stored in both CSR and CSC.** Every IPM iteration needs `Ax` *and*
`Aᵀy`; computing the latter from CSR alone needs GPU atomics or a cache-hostile
scatter.

**Storage orientation and memory space are types, not runtime fields.** A runtime
`(format × location)` pair puts a four-way branch inside every kernel.

**Dense matrices are column-major**, matching cuBLAS and LAPACK.

**Duals carry `w ≥ 0` explicitly** rather than eliminating it as `w = −y`. Costs
one length-`m` vector; saves inverting a sign in every step-length test and every
debug print. See `FORMULATION.md` §4.

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

- **`third_party/` contains zlib only** — a compression library, needed because
  MIPLIB ships `.mps.gz`. Not a solver.
- **The test harness is hand-written** (`tests/TestMain.hpp`, ~60 lines) rather
  than GoogleTest or Catch2. A test framework is plainly not a solver library,
  but keeping the dependency list at one entry means the question never has to be
  argued.
- **`tests/oracle/` uses `highspy` as a cross-check oracle** — it loads the same
  instance and dumps dimensions, bounds and coefficients to JSON so the C++
  reader can be verified against it. This is **development tooling only**. It is
  never linked into the solver, never shipped, and is not on any build path that
  produces a solver binary.
