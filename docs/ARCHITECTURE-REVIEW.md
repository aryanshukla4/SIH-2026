# Architecture Review — LP/QP/MILP Solver Core

**Scope:** review of `datatype_corrected`, `architecture_corrected`, `module_corrected`,
`prompts_corrected` against SIH PS 26119 (MRPL).
**Author:** ingestion & storage layer owner
**Status:** for team discussion — several items change what other modules should build, so
please read §2 and §3 before writing more code.

---

## 1. What is correct in the current design

Credit where it is due — the core mathematics in the handoff docs is **right**, and the
"corrected" revisions already caught the two bugs that most commonly kill a first IPM
(`rp = Ax − b` instead of `Ax + s − b`, and dropping `rsy` entirely).

Verified by hand. With the Lagrangian

```
L = ½xᵀQx + cᵀx − yᵀ(Ax + s − b) − zᵀx − wᵀs
```

we get

```
∂L/∂x = Qx + c − Aᵀy − z          ✓ matches the doc's rd
∂L/∂s = −y − w  = 0  ⟹  w = −y    ✓ matches the doc's convention
```

And the reduced system is genuinely SPD for LP. Eliminating gives

```
(A Θ Aᵀ − Y⁻¹S) Δy = rhs        with  Θ = Z⁻¹X ≻ 0
```

Since `Y` is negative diagonal under this convention and `S` is positive, `−Y⁻¹S ≻ 0`, so the
whole operator is SPD. The doc's claim holds.

The module decomposition is clean, and the reversibility requirements on presolve and scaling are
correctly identified.

**The issues below are about scope and depth, not correctness.**

---

## 2. Blocking issues

### 2.1 🔴 No simplex, so MILP is blocked

PS 26119 states LP, **MILP** and QP are "the initial focus", and names "revised simplex" among
the required algorithms. The current architecture has no simplex, no LU factorization, no
Markowitz pivoting, and no basis-update machinery; MILP is deferred to §17 "Future".

The technical reason this matters more than a scheduling slip: **branch-and-bound solves 10⁴–10⁶
LP relaxations, each differing from its parent by a single bound change.** Dual simplex
warm-starts that in a handful of pivots. **An IPM cannot warm-start** — it restarts from the
central path at every node. A MILP built on an IPM-only core runs roughly two orders of magnitude
slower than it needs to.

This also matters for the customer. MRPL's listed use cases — refinery scheduling, crude
blending, unit commitment, tank assignment, batch decisions — are **discrete**. A pure LP/QP
solver addresses none of them.

### 2.2 🔴 Normal equations are the wrong primary path for QP

`architecture_corrected.txt` §8 selects the reduced normal-equation form as the primary
implementation, and §16 presents QP as a substitution: "For LP use Q = 0. For QP include Q
correctly."

It is not a substitution. Eliminating `Δx` requires inverting `(Q + X⁻¹Z)`:

- **LP** (`Q = 0`): diagonal. Trivial, and `AΘAᵀ` stays sparse.
- **QP** (`Q ≠ 0`): a full matrix. `Θ = (Q + X⁻¹Z)⁻¹` is dense in general, so `AΘAᵀ` cannot be
  formed cheaply and is dense anyway.

The standard answer for QP is the **augmented (quasi-definite) KKT system with an `LDLᵀ`
factorization** — Vanderbei's LOQO formulation — which the docs currently list as a possible
*future* backend. For QP support that ordering is backwards.

### 2.3 🔴 Dense columns destroy the normal equations

A single dense column `aⱼ` of `A` contributes `θⱼ aⱼaⱼᵀ` to `AΘAᵀ` — a **full rank-1 dense m×m
update**. One dense column makes the entire normal-equations matrix dense, regardless of how
sparse `A` is.

Netlib and MIPLIB contain many such instances. The standard mitigation is dense-column detection
followed by Sherman–Morrison splitting (solve the sparse part, correct with a low-rank update).
Nothing in the current docs addresses this.

*The ingestion layer will detect and report dense columns as a first-class field in the matrix
analysis output, so whoever owns the KKT builder has the information available.*

### 2.4 🔴 Ordering and symbolic factorization are unowned

`architecture_corrected.txt` §8 and `module_corrected.txt` §8 delegate the linear solve to
"cuSOLVER / cuDSS". That skips the components that actually determine IPM performance:

- **Fill-reducing ordering** (AMD, nested dissection). Often the difference between a factor with
  10⁶ nonzeros and one with 10⁹. This single choice dominates everything else.
- **Symbolic factorization done once.** The sparsity pattern of `AΘAᵀ` is **constant across all
  IPM iterations** — only the values of `Θ` change. Computing the pattern, elimination tree, and
  supernode structure once and reusing it every iteration is the standard design. Missing it
  means paying the expensive analysis phase on every single iteration.

There is also a **compliance risk**: PS 26119 says the solver "shall not be built upon any
existing open source solver library". cuBLAS and cuSPARSE are BLAS-level primitives, defensible
in the same way LAPACK is. **cuDSS is a direct sparse solver** — leaning on it means the hardest
numerical kernel in the project is not ours. Recommend at minimum owning a CPU reference sparse
Cholesky / `LDLᵀ`.

### 2.5 🔴 Nothing consumes the transformation information

`module_corrected.txt` §3 requires presolve transformations to be reversible, and §4 requires the
scaler to retain inverse scaling information. Both are correct requirements.

But **no module in the list ever consumes that information.** There is no Solution Reconstructor.
As specified, the solver computes an answer in transformed space and has no defined path back to
the user's original variables and duals. This is a missing module, not a missing line of code —
the inverse of a transform stack has to be designed alongside the forward direction.

---

## 3. Design issues that get expensive to fix later

### 3.1 🟠 `SolverState` structurally blocks Mehrotra predictor-corrector

Two independent problems:

**One `alpha`, but IPM needs two.** Primal and dual variables reach their boundaries at different
distances. Standard practice is separate `α_p` and `α_d`; forcing a single α discards the larger
of the two steps. Typical cost is **20–30% more iterations**, for no benefit.

**One direction set, but PC needs two.** Mehrotra's method computes an affine-scaling direction
`(Δx_aff, Δs_aff, Δy_aff, Δz_aff)`, derives σ adaptively from how much progress that direction
would make, then computes a corrector. `SolverState` holds exactly one `(dx, ds, dy, dz)`.

This is not a minor omission. PC is the difference between roughly **15–25 iterations and ~50**,
and it is nearly free because the second solve reuses the same factorization. The docs defer it
("A later implementation can use predictor-corrector"), but the IPM loop diagram in
`module_corrected.txt` also shows **one linear solve per iteration** — so the loop structure
itself has to change, not just the μ controller. Much cheaper to accommodate now.

### 3.2 🟠 Convergence tolerances are absolute; they must be relative

`architecture_corrected.txt` §13 specifies `||rp|| < tolerance`. Absolute tests fail in both
directions:

- With `||b|| ~ 10⁶`, `||rp|| < 1e-8` is unreachable — the solver never terminates.
- With `||b|| ~ 10⁻⁶`, it passes immediately at a meaningless point — the solver returns garbage
  and reports optimality.

Standard criteria:

```
||rp||∞ / (1 + ||b||∞)      < tol_primal
||rd||∞ / (1 + ||c||∞)      < tol_dual
|cᵀx − bᵀy| / (1 + |cᵀx|)   < tol_gap
```

Separately: **which norm is never specified anywhere in the docs.** ∞-norm and 2-norm give
different answers and different tolerance calibrations. Convention is ∞-norm for feasibility.
This needs to be pinned down in one place and referenced everywhere.

### 3.3 🟠 Regularization is absent, and it is what the PS actually grades

`cond(AΘAᵀ) ~ 1/μ²` **by construction** — the normal-equations matrix becomes catastrophically
ill-conditioned as the method converges. This is not an edge case; it is the expected behaviour
of every IPM.

PS 26119 asks for "a clear demonstration of numerical robustness ... solving challenging
large-scale optimization problems involving degeneracy, weak LP relaxations or ill-conditioned
constraint matrices, where simpler implementations struggle". The docs currently say
"singular/ill-conditioned handling where practical".

The established answer is **primal-dual regularization** (Altman–Gondzio) plus iterative
refinement, and it has to be designed into the KKT builder rather than bolted on afterwards.
It should also be *instrumented* — the count of regularization and refinement events is the
evidence for the robustness claim in the submission.

### 3.4 🟠 No mechanism detects infeasibility

`module_corrected.txt` §13 lists `INFEASIBLE` as a return status, but nothing in the formulation
can produce it. An IPM on an infeasible problem diverges or stalls; distinguishing that from slow
convergence requires machinery that is not present.

The **homogeneous self-dual embedding** is the clean answer, and it simultaneously resolves the
initialization question that §5 currently answers with "a robust simple initialization strategy"
(the standard concrete answer being Mehrotra's starting-point heuristic).

### 3.5 🟡 FP64 on our actual GPU is 1/64 speed

Worth checking before more GPU design work: the target card is an **RTX 3050 Laptop (GA107,
compute 8.6)**. Consumer Ampere runs FP64 at **1/64 of FP32 throughput**. The docs specify
"Prefer FP64 for the first numerical implementation" and put factorization on the GPU.

The distinction that matters:

| Operation | Bound by | FP64 penalty on GA107 |
|---|---|---|
| Dense/sparse factorization | FLOPs | **~64×** — slower than our CPU |
| SpMV (`Ax`, `Aᵀy`) | Memory bandwidth | **~2×** (8 bytes vs 4) — perfectly acceptable |

> **Both rows above are spec sheet plus roofline, not measurements** — and the
> second is routinely misread. Its "~2×" is the **FP64-vs-FP32 cost on the GPU**
> (doubles move twice the bytes, so a bandwidth-bound kernel costs ~2× more in
> FP64). It is **not** a GPU-vs-CPU speedup; that is a different quantity
> (~192 GB/s against a laptop's ~51 GB/s of DDR4, so under 4× as a ceiling).
>
> **Measured 2026-09-15** (`module.txt` §24E, `--method=pdlp --gpu-spmv=1`),
> identical `matrix_products` on both paths:
>
> | instance | size | CPU | GPU | |
> |---|---|---|---|---|
> | `25fv47` | 360 KB | 0.14 s | 3.52 s | GPU **25× slower** |
> | `maros-r7` | 4.7 MB | 1.61 s | 3.33 s | GPU 2.1× slower |
> | `datt256` | 93 MB | 15.81 s | 11.94 s | GPU **1.32× faster** |
>
> **This whole section is about GA107 specifically, and reads as more general
> than it is.** "GPU factorization is slower than our CPU" is a statement about
> a **1:64** FP64 ratio, which is a consumer-Ampere product decision, not
> physics. On a datacentre card (A100: 1:2 FP64, ~1.5 TB/s HBM) both rows change
> sign — factorization is compute-bound, so it would go from ~95 GFLOPS to
> ~9.7 TFLOPS against a CPU's ~150, and the SpMV ceiling rises from ~3.8× to
> ~30×. Nothing here should be quoted as "GPUs are bad at factorization"; it
> means *this* GPU is, and the matrix-free design is what makes the code
> portable to one where it isn't.
>
> **Residency was built on 2026-09-19** (`module.txt` §24F): `datt256`
> 16.02 s → 4.47 s (3.6×), `bab2` 15.96 s → 3.32 s (4.8×). `bab2` beats the
> "~3.8× ceiling" above, which exposes an assumption in it: the ceiling is the
> bandwidth ratio against a CPU that *saturates* its memory bus, and our CPU
> PDLP is single-threaded. Quote these against "our CPU implementation", not
> "the CPU".
>
> *(Superseded:)* Device residency deferred on the same reasoning, and the reasoning is
> hardware-specific in the opposite direction: it is worth ~4× here, but PCIe
> does not scale with the GPU, so on a faster card the transfer fraction grows
> and residency becomes the gate rather than an optimization. Deferred because
> 4× loses to the ~20× this project has measured from single algorithmic changes
> — not because it is unimportant.
>
> GPU time is near-**constant** across a 260× size range, so the crossover is
> set by per-product launch latency, not by arithmetic. Attributing the
> `datt256` run: kernel 1.63 s, host↔device transfer 2.94 s (**1.8× the
> kernel**), and ~7.4 s of PDLP's own host-side vector arithmetic that is
> identical on both paths. Netting that out, the SpMV itself is ~1.8× faster on
> the GPU — consistent with the ceiling this section predicted. The prediction
> held; the *threshold* is what only measurement could supply.

So on this hardware the credible GPU story is the **SpMV-heavy, matrix-free path**, not GPU
factorization. Two consequences worth discussing:

1. If we want GPU factorization, **mixed precision + iterative refinement** (factor in FP32,
   refine to FP64 accuracy) is the established technique for exactly this situation.
2. Note that PS 26119's body is more measured than its title: GPU acceleration "considered where
   it provides **measurable benefits**". An honest benchmark showing where GPU helps and where it
   does not is a stronger submission than an unconditional GPU claim.

Also unaddressed: **4 GB VRAM**. A million-variable Cholesky factor can exceed it, and there is
no fallback strategy. And the transfer optimization should be explicit — upload the sparsity
**pattern once**, transfer only **values** per iteration.

### 3.6 🟡 Presolve is under-specified

`module_corrected.txt` §3 lists five rules (empty rows/columns, fixed variables, redundant
constraints, obvious infeasibility, simple bound tightening).

PS 26119 names presolve explicitly as a required component, and in practice it is one of the
highest-return parts of a solver — presolve routinely shrinks MIPLIB instances by **30–60%**
before the first iteration. Production presolve also includes singleton rows/columns, doubleton
equations, forcing constraints, dominated columns, duplicate row/column removal, implied free
variables, coefficient tightening, and for MILP: probing, clique merging, and knapsack
coefficient reduction.

### 3.7 🟡 The `w = −y` convention is correct but costly in practice

The convention is internally consistent (verified in §1). The practical cost is that a whole dual
vector must remain strictly **negative**:

- every step-length computation reads `max α such that −(y + αΔy) > 0`
- every diagnostic print shows negative duals, which look like bugs during debugging
- every cross-check against a textbook or another solver needs mental sign inversion

Carrying `w ≥ 0` explicitly costs **one vector of length m**. Recommend carrying it.

---

## 4. Missing modules

Ten components have no owner in `module_corrected.txt`:

| Module | Why it is needed |
|---|---|
| **Canonicalizer** | Nobody owns faithful model → `Ax + s = b, x ≥ 0`. See §5. |
| **Solution Reconstructor** | §2.5 — the answer never gets mapped back. |
| **Ordering / Symbolic Analysis** | §2.4 — the dominant IPM cost. |
| **Regularization** | §3.3 — the graded robustness deliverable. |
| **Simplex + basis/LU** | §2.1 — required by the PS, required for MILP. |
| **Crossover** | IPM returns an *interior* point, not a vertex. B&B needs a basis, and industrial users often require a basic solution. |
| **Model writer** | Round-trip testing and debugging. Cheap, high value. |
| **Benchmark harness** | The PS *requires* comparison against an established solver across MIPLIB/Netlib. That is a graded deliverable with no owner. |
| **Options / parameters** | Otherwise tolerances become magic constants scattered across modules. |
| **Logging** | Distinct from Diagnostics — the standard per-iteration solver log line. |

Also: **Matrix Analyzer (§2) specifies what to report but defines no output type**, so no module
can consume its results.

---

## 5. The model representation gap — and what the ingestion layer is doing about it

This one is in my scope, so it is being fixed rather than raised. Flagging it because it changes
the `Problem` type that every other module consumes.

The docs assume the model is always:

```
minimize cᵀx   subject to   Ax ≤ b,  x ≥ 0
```

Real benchmark instances are not. **A loader restricted to that form cannot usefully read a
single Netlib or MIPLIB file.** MPS files routinely contain ranged rows, free rows, general
variable bounds, maximization, objective constants, and integer markers.

`Problem` is therefore gaining the following fields:

| Field | Consequence of omitting it |
|---|---|
| **integrality flags** (`MARKER INTORG`/`INTEND`) | **MILP permanently blocked** |
| **objective sense** | maximization models silently solve backwards |
| **objective constant** | every reported objective is wrong by a constant |
| **row ranges** (`RANGES`) | `l ≤ aᵀx ≤ u` is inexpressible with `sense` + `b` |
| **row/column names** | no solution reporting, no debugging, no round-trip test |
| **free rows** | extra `N` rows must be dropped, not parsed as constraints |
| **objective row identity** | MPS designates it by name among the `N` rows |
| **SOS / semi-continuous** | appear in real refinery blending models |

Two structural changes come with it:

1. **`b` + `constraint_sense` → `row_lower` + `row_upper`.** This collapses `L`/`G`/`E`/`N` *and*
   `RANGES` into one uniform representation (what HiGHS and the commercial solvers do internally)
   and eliminates an entire class of sign bugs.

2. **Two layers: faithful `Problem` + reversible `Canonicalizer`.** `Problem` stores what the file
   said; a separate transform stage produces the solver's `Ax + s = b, x ≥ 0` form and records
   how to invert it — primal *and* dual. This composes with the presolve and scaling transforms
   into one ordered stack, which is also the fix for §2.5.

Note a contradiction in the current doc worth resolving: `datatype_corrected.txt` §4 lists
`lower_bounds` / `upper_bounds` on `Problem`, but the formulation section states `x ≥ 0`. Where
general bounds actually live is undefined. The two-layer split resolves it.

`Matrix` is also gaining **CSC alongside CSR**. Every IPM iteration needs both `Ax` and `Aᵀy`;
computing `Aᵀy` from CSR alone requires atomics on GPU or a cache-hostile scatter on CPU. The 2×
index/value memory is worth it and is what every serious solver does.

---

## 6. A process risk in `prompts_corrected.txt`

The prompts instruct an AI to implement each module **independently**. That only works if the
interfaces between them are fully specified — and several are not:

- `MatrixAnalysis` has no output type
- transform / presolve records have no type
- options and tolerances have no type
- `loadProblem` has no error channel (returns `Problem` by value, with no way to report *which
  line* of a 500 MB file failed to parse)

Four modules generated independently against under-specified interfaces **will not compose**, and
integration becomes a rewrite rather than a link step.

**Recommendation:** freeze `include/sovsolve/**/*.hpp` as a compiling, implementation-free header
contract, review it as a team, and *then* implement modules against fixed headers. The ingestion
layer is delivering its headers this way; suggest we do the same across the board.

§16 of `prompts_corrected.txt` (validation requirements — hand-checkable example, dimension
checks, CPU reference path, residual verification, no silent sign-convention changes) is good as
written. Keep it verbatim.

---

## 7. Summary

| Severity | Item | Owner |
|---|---|---|
| 🔴 | `Problem` missing integrality flags → MILP blocked | ingestion *(fixed)* |
| 🔴 | `Problem` missing ranges + general bounds → cannot read Netlib/MIPLIB | ingestion *(fixed)* |
| 🔴 | No simplex → MILP ~100× slower; PS names revised simplex | **unowned** |
| 🔴 | Normal equations wrong for QP → need augmented KKT + `LDLᵀ` | KKT builder |
| 🔴 | Ordering + symbolic factorization unowned → dominant IPM cost | **unowned** |
| 🔴 | Nothing consumes transform info → solution never mapped back | **unowned** |
| 🟠 | Single α + single direction set → blocks Mehrotra PC, ~2× iterations | SolverState / IPM loop |
| 🟠 | Absolute tolerances → false convergence *and* non-convergence | convergence checker |
| 🟠 | No dense-column handling → `AΘAᵀ` goes dense on real instances | KKT builder *(detection provided)* |
| 🟠 | No regularization → the exact robustness the PS grades | KKT builder |
| 🟠 | No infeasibility detection mechanism | IPM / formulation |
| 🟡 | `Matrix` missing CSC, sorted-index invariant, index type, dense layout | ingestion *(fixed)* |
| 🟡 | FP64 on GA107 is 1/64 → GPU design assumption unvalidated | CUDA backend |
| 🟡 | Presolve under-specified vs PS requirements | presolve |
| 🟡 | `w = −y` convention costs debuggability for one length-m vector | formulation |

**Suggested next steps for discussion:**

1. Decide whether MILP is in scope for the submission. If yes, someone needs to own simplex, and
   that decision should be made now rather than after the IPM is finished.
2. Decide the QP linear-algebra path — normal equations for LP plus augmented `LDLᵀ` for QP, or
   augmented for both.
3. Assign ordering + symbolic factorization; it is the single highest-impact unowned component.
4. Freeze the header contract across all modules before further implementation (§6).
