# Addendum — the canonical form decision

**Follows:** `ARCHITECTURE-REVIEW.md`
**Scope:** one open decision in the revised architecture, with measurements from our own corpus.
**Status:** needs a team decision. Everything else here is either resolved or informational.

---

## 0. What the revision resolved

The revised documents adopt most of the original memo. Recording it so nobody re-argues settled points:

| Finding | Status in the revision |
|---|---|
| §2.2 Normal equations wrong for QP | **Resolved** — augmented/quasi-definite `LDLᵀ` for general QP, with a `ReductionDescriptor` so the KKT builder picks the path. Better than what the memo proposed. |
| §2.4 Ordering / symbolic factorization unowned | **Resolved** — Module 10, including reuse across iterations. |
| §2.5 Nothing consumes transform info | **Resolved** — Module 17, Solution Reconstructor. |
| §3.1 Single α, single direction set | **Resolved** — `alpha_primal`/`alpha_dual`, affine and corrector direction sets, full Mehrotra controller (Module 8). |
| §3.2 Absolute tolerances | **Resolved** — relative, scale-aware, ∞-norm (Module 18). |
| §3.3 Regularization absent | **Resolved** — Module 11, with event counting in Diagnostics. |
| §4 Ten missing modules | **Six added** — Canonicalizer, Solution Reconstructor, Ordering, Regularization, Logging, Options. |
| Matrix Analyzer had no output type | **Resolved** — `MatrixAnalysis`, including `dense_columns`. |
| `Problem` missing sense / constant / integrality | **Resolved**. |
| `Scalar` as a struct with a runtime precision field | **Resolved** — `using Scalar = double`. |
| §6 Modules generated against under-specified interfaces | **Resolved** — the Generation Rule now starts with "use the frozen shared type definitions". |

§2.1 (no simplex, MILP blocked) is now an **explicit deferral** with a stated
extensibility contract — integrality flags kept at ingestion, canonicalizer
separate from the IPM, transform records preserved. That is a legitimate scope
call, clearly made, and the ingestion layer already honours all three
conditions. It remains a submission risk against a problem statement that puts
MILP in initial scope, but it is a decision rather than an oversight.

---

## 1. The open decision

The canonical form is unchanged:

```
    A x + s = b
    x >= 0
    s >= 0
```

**The algebra is correct.** Given the model the architecture assumes —
`Ax ≤ b, x ≥ 0` — adding a non-negative slack to reach `Ax + s = b` is exactly
right. There is no error in the formulation.

The issue is one level up: real instances are not `Ax ≤ b, x ≥ 0`. Two distinct
costs follow, and they are very different in size.

---

## 2. Cost A — equality rows (resolved, zero cost)

If a row is `aᵀx = b` and you write `aᵀx + s = b` with `s ≥ 0`, you have turned
it into `aᵀx ≤ b`. The constraint is **relaxed**. Keeping it an equality needs
`s = 0`, and the form offers only `s ≥ 0` — there is no upper bound to pin it
with.

Not a corner case. From our corpus:

| Instance | rows | equality | share |
|---|---:|---:|---:|
| gas11 | 459 | 459 | **100%** |
| shell | 536 | 534 | **99.6%** |
| greenbea | 2 392 | 2 199 | **92%** |
| etamacro | 400 | 272 | 68% |
| 25fv47 | 821 | 516 | 63% |
| stair | 356 | 209 | 59% |

On `gas11` the assumed form represents **none** of the constraints.

### Why the obvious fix is the wrong one

The tempting move is to split `aᵀx = b` into `aᵀx ≤ b` and `−aᵀx ≤ −b`. Two
reasons not to.

**It breaks the normal equations exactly when you converge.** Rows `a` and `−a`
make `A` rank-deficient. Their 2×2 contribution to `AΘAᵀ` is

```
    [  t  −t ]         t = aᵀΘa > 0        determinant = t² − t² = 0
    [ −t   t ]
```

Singular. It stays invertible only because of the `+D` term (`D = W⁻¹S`) the
slacks contribute — but at optimality a satisfied equality makes **both**
inequalities tight, so both slacks go to zero and `D → 0`. The term holding the
system together vanishes precisely as the method converges. That is
self-inflicted ill-conditioning, on the exact axis PS 26119 asks us to
demonstrate robustness along.

**It doubles `m` where it hurts most.** `gas11` goes 459 → 918 rows. `AΘAᵀ` is
`m × m`, so that is roughly 4–8× the factorization cost — on the instances that
are *most* equality-heavy.

### What the ingestion layer does instead

Equality rows get **no slack column**. Slacks are appended as columns, only to
inequality rows:

```
    A_canonical = [ A_structural | I_slack ]        x_canonical = [ x ; s ]
```

Read the trailing block as `s` and this **is** the architecture's form.
`CanonicalProblem::slack_begin()` marks the boundary. Cost: **zero extra rows**
— 25fv47, afiro, adlittle, e226 and israel all canonicalize with 0% row growth.

This is already implemented and property-tested. No decision needed.

> **For the KKT builder:** because the slack block is exactly `I`, the reduced
> system never needs it formed. `A_canon Θ A_canonᵀ = A_s Θ_x A_sᵀ + Θ_s` with
> `Θ_s` diagonal — that diagonal *is* the `+D` term. Use `slack_begin()` to
> split the two.

---

## 3. Cost B — the open one

`x ≥ 0` is the only variable bound the canonical form permits, so **every finite
upper bound has to become a constraint row**: `x ≤ u` becomes `x + t = u, t ≥ 0`.

Measured on our corpus, canonicalizing exactly as the architecture specifies:

| Instance | rows | boxed cols | canonical rows | row growth |
|---|---:|---:|---:|---:|
| rgn | 24 | 180 | 204 | **+750%** |
| gt2 | 29 | 188 | 217 | **+648%** |
| 80bau3b | 2 262 | 3 484 | 5 746 | **+154%** |
| egout | 98 | 86 | 184 | +88% |
| shell | 536 | 367 | 903 | +68% |
| bell5 | 91 | 58 | 149 | +64% |
| flugpl | 18 | 11 | 29 | +61% |
| etamacro | 400 | 217 | 617 | +54% |
| gas11 | 459 | 175 | 634 | +38% |

**`gt2` goes from 29 rows to 217.** `AΘAᵀ` grows from 29×29 to 217×217 — on the
order of 400× the factorization work, for a model that did not change.

Note *which* instances blow up: rgn, gt2, bell5, egout, flugpl. **Every MILP
instance in the corpus.** Binary variables are boxed `[0,1]` by definition, so
each one costs a row. The penalty lands hardest on precisely the problem class
MRPL's use cases are made of — even though MILP is deferred, the LP relaxations
are what a future branch-and-bound would solve millions of times.

### The alternative, and why it is cheap

Use a **bounded-variable IPM**:

```
    minimize    1/2 xᵀQx + cᵀx
    subject to  A x = b
                0 <= x <= u
```

Two complementarity pairs per variable instead of one — `(x, z)` for `x ≥ 0`
and `(u − x, w)` for `x ≤ u`. The reduced system **stays `m × m`**; finite upper
bounds only change the diagonal:

```
    Θ  =  ( X⁻¹Z + (U−X)⁻¹W )⁻¹          still diagonal, still cheap
```

Cost: one extra vector pair. Against up to eight times the rows.

This is what production solvers do, and it is the same shape of change as the
`w = −y` question below — cheap now, expensive after the KKT builder,
step-length calculator and residual code are written against the current
variable set.

**Nothing on the ingestion side changes either way.** `Problem` already carries
`col_lower` and `col_upper`; the canonicalizer emits whichever form the team
picks.

---

## 4. Still open after the revision

Short list, none of them new work for ingestion.

**4.1 `w = −y` is now an interoperability problem, not a style preference.**
The revision keeps the convention where the slack dual is eliminated as
`w = −y`, so `y` must stay strictly negative. The ingestion layer carries
`w ≥ 0` explicitly (documented in `FORMULATION.md` §4) because it makes every
step-length test and diagnostic print read naturally. The two are related by
exactly `w_here = −y_theirs`. **One of the two has to move**, or every dual
crossing the boundary needs a conversion nobody will remember to write. Cost of
carrying `w` explicitly: one vector of length `m`.

**4.2 `Matrix` still has no CSC.** Module 7 computes `Aᵀy` every iteration.
From CSR alone that needs atomics on GPU or a cache-hostile scatter on CPU.
Ingestion already produces both orientations and keeps them consistent by
construction — the CSR is a counting-sort transpose of the CSC, so they cannot
drift.

**4.3 "One Matrix type" vs. compile-time dispatch.** The Memory Rule forbids
separate `DenseMatrix`/`SparseMatrix` public types. Ingestion currently uses
typed storage, because a runtime `(format × location)` pair puts a four-way
branch inside every kernel. This is a genuine conflict with a stated rule and
should be settled by the team, not unilaterally. A façade that presents one
`Matrix` at module boundaries over typed storage underneath is possible; it
costs an indirection at the seam.

**4.4 FP64 on the target GPU.** Still "use FP64 for the first numerical
implementation" with factorization on GPU. On a GA107 (RTX 3050) FP64 runs at
**1/64** of FP32 — dense factorization is FLOP-bound and lands slower than the
CPU, while SpMV is bandwidth-bound and only costs ~2×. The revision added
"iterative refinement" to the GPU boundary, which is the right hook for mixed
precision; worth making that explicit rather than incidental.

**4.5 Nothing produces an infeasibility certificate.** Module 18 correctly says
not to infer infeasibility from stagnation — but no module can produce the
positive evidence either. A homogeneous self-dual embedding supplies both that
and the initialization Module 6 currently describes as "robust".

**4.6 New tension introduced by the revision: Module 10 vs. cuDSS.** Module 10
now owns fill-reducing ordering and symbolic factorization — correct, and the
memo asked for it. But `cuDSS` performs its own analysis phase, including its
own reordering. If both exist, one is wasted work and the two may disagree.
Worth deciding whether Module 10 feeds an ordering into the backend, or whether
the backend owns analysis and Module 10 covers only the CPU reference path.

---

## 5. Decisions needed

1. **Bounded-variable IPM, or bound rows?** (§3) The measurements say bounded
   variables; the cost of switching rises steeply once the IPM modules exist.
2. **`w ≥ 0` or `w = −y`?** (§4.1) Either is fine — but it must be one of them,
   written down in `FORMULATION.md`, and used everywhere.
3. **Does `Matrix` gain CSC?** (§4.2) Ingestion already produces it.
4. **One `Matrix` type, or typed storage with a façade?** (§4.3)
5. **Who owns ordering — Module 10 or the sparse backend?** (§4.6)

Items 1, 2 and 3 are the ones that get expensive to change later.
