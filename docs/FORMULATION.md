# FORMULATION — single source of mathematical truth

Every module references **this file**. Do not restate these equations elsewhere; link here.
If a sign or convention changes, it changes here first, and the change is announced to the team.

---

## 1. Faithful model (what a file actually contains)

This is what `Problem` stores. It is *not* the solver's working form.

```
    minimize / maximize     ½ xᵀQx + cᵀx + c₀

    subject to              rl ≤ A x ≤ ru
                            cl ≤   x ≤ cu
                            xⱼ ∈ ℤ  for j ∈ I
```

| Symbol | Meaning |
|---|---|
| `c₀` | objective constant |
| `rl`, `ru` | row bounds; `−∞`/`+∞` allowed, `rl = ru` means equality |
| `cl`, `cu` | column bounds; `−∞`/`+∞` allowed, `cl = cu` means fixed |
| `I` | index set of integer variables |
| `Q` | symmetric; `Q ⪰ 0` required for convex QP; `Q = 0` for LP |

**Infinity convention:** any bound with magnitude `≥ 1e20` **is** infinity. This is the MPS
convention and is used consistently throughout the codebase (`core::INF`).

Row senses `L`/`G`/`E`/`N` and the `RANGES` section are all *encodings* of `[rl, ru]`.
See `MPS-FORMAT-NOTES.md` for the exact translation tables.

---

## 2. Canonical model (the solver's working form)

Produced by the Canonicalizer, with a reversible transform stack.

```
    minimize      1/2 xQx + cx

    subject to    A_E x       = b_E
                  A_I x + s   = b_I,   s >= 0
                  l <= x <= u
```

This is the **bounded-variable** form. Two properties of it carry the design:

**Equality rows carry no slack.** A form that appends `s >= 0` to every row
cannot express an equality — there is nowhere to put a slack that must be zero
— and equalities are the majority of rows in real instances: 516 of 821 in
Netlib `25fv47`, all 459 in `gas11`.

**Finite variable bounds stay native.** Turning each `x <= u` into a constraint
row grows the reduced system by one row per boxed column. Measured on our
corpus under the previous `Ax = b, x >= 0` form: **+750% rows on `rgn`, +648%
on `gt2`, +154% on `80bau3b`** — worst on exactly the MILP instances, since a
binary variable is boxed `[0,1]` by definition. Two complementarity pairs per
variable costs one extra vector instead, and the reduced system stays `m x m`.

`A_E` and `A_I` are contiguous **row blocks of one matrix**, not two matrices:
rows `[0, num_equality)` are `A_E`, the rest are `A_I`, and `s[k]` belongs to
canonical row `num_equality + k`. One matrix keeps `Ax` and `Ay` single kernel
calls, which is what the IPM does with them every iteration. The fill-reducing
ordering permutes rows again downstream, so preserving file order buys nothing.

For LP, `Q = 0`.

### 2.0 Objective sense — negate both `c` and `Q`

The canonical form is a **minimization**. A maximization input is negated,
recorded as `NegateObjective`, and inverted when the objective is reported.

`Q` must be negated along with `c` and the constant. A legitimate concave
maximization arrives with `Q` **negative** semidefinite, and

```
    max  cᵀx + ½xᵀQx      ≡      min  (−c)ᵀx + ½xᵀ(−Q)x
```

so `−Q` is the PSD matrix §10 assumes. Negating `c` alone leaves the
factorization facing an indefinite matrix it was promised it would never see,
surfacing as a wrong-signed pivot with no obvious cause.

### 2.1 Column transforms — there are none

This is the point of the bounded-variable form. No column is shifted, reflected
or split; bounds pass through to `col_lower` / `col_upper` untouched. The only
column operation is **removal**, and only for a fixed column.

In particular, **splitting a free variable as `x = x+ - x-` is prohibited.** It
doubles the column count, makes the two columns structurally dependent, and is
well documented to converge poorly in interior-point methods. A free column
stays a single column with both bounds infinite; §10.1 covers the consequence.

### 2.2 The startability contract

The canonicalizer's output is either **startable by the IPM, or a definite
infeasibility verdict**. This is a contract, not an optimization, so it must not
depend on the presolver — callers can switch that off.

**No column has `l == u`.** The IPM requires `x-l > 0` and `u-x > 0` at once;
their sum is `u-l`, so a fixed column admits no strictly interior point and the
method cannot take a first step. Fixed columns are substituted out:

```
    b_E := b_E - A_E[:,j] l_j
    b_I := b_I - A_I[:,j] l_j
    c   := c   + Q[:,j] l_j                 (over the remaining columns)
    offset += c_j l_j + 1/2 Q_jj l_j^2
    column j deleted
```

recorded as `RemoveFixedVariable(j, l_j)`. **1067 fixed columns across 8 of our
19 instances** — 498 in `80bau3b`, 250 in `shell`, 103 in `greenbea`.

**No row of `A` is entirely zero.** An all-zero row makes that row and column of
`A T A` identically zero for every `T` — an exact zero pivot, not an
ill-conditioning, and nothing in the regularization policy can repair it. A
consistent empty row is dropped; an inconsistent one (`0 = 5`) returns
`PrimalInfeasible`, because regularizing it away would report
`NUMERICAL_FAILURE` for a model that is provably infeasible.

**Emptiness is tested after the substitution, not before.** Removing a column
can take away the last entry a row had. Measured on our own corpus, inside the
canonicalizer, one stage before the presolver runs:

| instance | empty rows before | rows dropped after | created |
|---|---:|---:|---:|
| gas11 | 0 | 2 | **2** |
| 80bau3b | 25 | 27 | **2** |
| standata | 0 | 1 | **1** |
| greenbea | 3 | 3 | 0 |

Three of those had none to begin with. `CanonicalProblem::is_ipm_startable()`
exposes the same `O(m + n + nnz)` check for the initializer to re-run, because
the canonicalizer is not the last stage to touch the model.

### 2.3 Row transforms

| Original | Canonical |
|---|---|
| `ax = b` | equality block, no slack |
| `ax <= u` | `ax + s = u`, `s >= 0` |
| `ax >= l` | negate the row, then `-ax + s = -l` |
| `l <= ax <= u` | equality `ax + t = u` with a bounded column `0 <= t <= u-l` |
| both bounds infinite | dropped (vacuous) |
| empty over kept columns | dropped, or `PrimalInfeasible` |

Row negation flips the sign of that row's dual; recovery inverts it (§4).

A **ranged** row does not produce a two-sided slack. It becomes an equality
carrying one extra bounded column, so `s` stays strictly one-sided everywhere
and `rsy` in §5 remains correct as written.

Two alternatives were rejected. Giving `s` a finite upper bound makes it
two-sided, which costs a second dual, an extra complementarity residual, an
extra step-length test and an extra term in `mu` — six modules of change against
one column. Splitting the row into `ax <= u` and `-ax >= -l` is worse: the two
rows are negatives, so their 2×2 contribution to `AΘAᵀ` is

```
    [  d  -d ]      d = aᵀΘa > 0        determinant = d² - d² = 0
    [ -d   d ]
```

singular, propped up only by the slack diagonal — which vanishes exactly at
convergence, when a satisfied range makes both sides tight.

No instance in our corpus has a ranged row, so this path is unit-tested but not
exercised by the benchmark set.

---

## 3. Primal-dual variables

| Variable | Length | Meaning | Sign requirement |
|---|---|---|---|
| `x` | `n` | primal variables | `x-l > 0` and `u-x > 0` where the bound is finite |
| `s` | `m_I` | inequality-row slacks | `s > 0` |
| `y` | `m` | row duals | free on `A_E`; `-y_I > 0` on `A_I` |
| `z` | `n` | duals for `x >= l` | `z > 0` |
| `v` | `n` | duals for `x <= u` | `v > 0` |

`s` has length `m_I = m - num_equality`, **not** `m`. Equality rows have no
slack. The reduced cost of column `j` is `z_j - v_j`.

A pair belonging to an **infinite** bound does not exist and is omitted
everywhere — from `rxz`/`ruv`, from the `mu` average, and from step lengths. A
free column contributes no pair at all.

---

## 4. Sign convention — **read this before touching any dual**

The slack dual is **eliminated**, not stored:

```
    w_s = -y_I          so  -y_I > 0  on inequality rows
```

`w_s` is not an independent quantity. Storing it duplicates state that can drift
from `-y_I` under rounding, leaving an invariant nobody maintains. Where a sign
test would otherwise read backwards, use a helper rather than a second vector:

```
    slack_dual(y) = -y          // step-length and interiority tests read positively
```

### This convention already matches what other solvers report

Verified against HiGHS. For a **minimization**:

```
    <= row                        ->  dual <= 0        matches -y_I > 0 here
    >= row                        ->  dual >= 0
    == row                        ->  free sign
    variable at its lower bound   ->  reduced cost >= 0    matches z >= 0 here
```

So **no global sign normalization is applied at the exit point.** The only flip
`recover_solution()` performs is local: `>=` rows were negated in §2.3, so their
duals are negated back — and *that* flip is what produces the non-negative dual
a `>=` row is reported with.

An earlier draft of this file claimed the opposite and specified a global
normalization at the reconstructor. It was wrong; applying it would invert every
inequality dual. Recorded here so nobody reintroduces it.

Derivation, for the record. With

```
L = 1/2 xQx + cx - y(Ax + s - b) - z(x - l) - v(u - x)
```

stationarity gives

```
dL/dx  =  Qx + c - Ay - z + v  =  0
dL/ds  =  -y - w_s             =  0     =>   w_s = -y
```

---

## 5. Residuals

```
    rp_E =  A_E x - b_E                 (length m_E)   primal, equality rows
    rp_I =  A_I x + s - b_I             (length m_I)   primal, inequality rows
    rd   =  Qx + c - Ay - z + v         (length n)     dual / stationarity
    rxz  =  (x-l).*z - mu 1             (length n)     lower-bound complementarity
    ruv  =  (u-x).*v - mu 1             (length n)     upper-bound complementarity
    rsy  =  -s.*y_I - mu 1              (length m_I)   slack complementarity
```

`X_L`, `U-X`, `Z`, `V`, `S`, `Y_I` denote the corresponding diagonals.
**They are never materialized** — always elementwise vector operations.

Omit the `rxz` entry where the lower bound is infinite and the `ruv` entry where
the upper bound is infinite.

---

## 6. Complementarity measure

```
    mu  =  [ sum (x-l).*z  +  sum (u-x).*v  +  sum (-s).*y_I ] / active_pair_count
```

`active_pair_count` counts only pairs that exist: one per finite lower bound,
one per finite upper bound, one per inequality row. Dividing by `2n + m_I`
instead understates `mu` badly on a model with many free or one-sided columns —
`gas11` is 44% free columns, which contribute no pair at all.

---

## 7. Newton system

```
    Q dx - A dy - dz + dv   =  -rd
    A_E dx                  =  -rp_E
    A_I dx + ds             =  -rp_I
    Z dx + X_L dz           =  -rxz
    -V dx + (U-X) dv        =  -ruv
    -Y_I ds - S dy_I        =  -rsy
```

Six blocks. Do not collapse to a smaller system before §10 selects a reduction.

## 8. Step lengths — **two, not one**

Primal and dual variables reach their boundaries at different distances. Using a single α
discards the larger of the two steps and costs 20–30% more iterations.

```
    α_p  =  η · max { α ∈ (0,1] :  l < x + αΔx < u   and   s + αΔs > 0 }
    α_d  =  η · max { α ∈ (0,1] :  z + αΔz > 0,  v + αΔv > 0,
                                   slack_dual(y_I + αΔy_I) > 0 }
```

with safety factor `0 < η < 1` (typically `η ≈ 0.995`).

A bound test is skipped where that bound is infinite. Equality-row `y` is
unrestricted and is never limited by the ratio test. Write the slack-dual test
through the `slack_dual` helper rather than as an inline `−y > 0`, which is easy
to write backwards.

Update:

```
    x ← x + α_p Δx        y ← y + α_d Δy
    s ← s + α_p Δs        z ← z + α_d Δz
                          v ← v + α_d Δv
```

---

## 9. Convergence — **relative, ∞-norm**

The norm is **`∞`** everywhere unless explicitly stated otherwise.

```
    ||rp||∞ / (1 + ||b||∞)        <  tol_primal
    ||rd||∞ / (1 + ||c||∞)        <  tol_dual
    |cᵀx − bᵀy| / (1 + |cᵀx|)     <  tol_gap
```

Absolute tests are **wrong** — they never terminate on well-scaled problems
(`||b|| ~ 10⁶`) and falsely report optimality on small ones (`||b|| ~ 10⁻⁶`).

Convergence is never declared on the basis of `x` changing slowly.

---

## 10. Reduced systems

Two backends, selected by problem class.

### 10.1 Normal equations — LP only

Eliminating `dz`, `dv`, `ds`, then `dx`:

```
    ( A T A + D_s + delta_d I ) dy  =  rhs
```

with `D_s` the diagonal slack contribution on inequality rows. SPD, so Cholesky
applies. The reduced system stays `m x m` because variable bounds are not rows.

**Build `T^-1`, never `T`.** The diagonal is

```
    T^-1 = X_L^-1 Z + (U-X)^-1 V         term omitted where that bound is infinite
    T^-1 <- max(T^-1, delta_p)           elementwise floor, unconditional, every iteration
    T    <- (T^-1)^-1                    now always finite
```

**The floor covers the whole diagonal, not only free columns.** A free variable
contributes neither term, so its entry is exactly zero at iteration 0 — that is
the visible case. But the entry of any variable that ends up strictly *between*
its bounds at the optimum also goes to zero, because complementarity drives
`z_j -> 0` and `v_j -> 0` there while `(x-l)` and `(u-x)` stay away from zero:

```
    at a bound at the optimum:        T^-1 -> large,  T -> ~mu
    strictly inside at the optimum:   T^-1 -> ~mu,    T -> large
```

At `mu = 1e-10` the two groups differ by roughly `1e20` — which is the
`cond(A T A) ~ 1/mu^2` growth below, made concrete. Double precision resolves
about `1e16`. So free variables are not a special case; they are the first
instance of a failure that reaches every basic variable near convergence. A
floor applied only to columns tagged free fixes iteration 0 and leaves
iteration 35 broken.

Nor does presolve remove the free columns for you. Free column singleton
substitution — the rule that eliminates a hand-written slack exactly — covers
only **89 of the corpus's 381 free columns (23%)**, and on `gas11` 89 of 375.
Roughly 286 genuine free columns still reach the KKT builder there, a third of
that instance's 862 columns. The floor is the primary mechanism, not a fallback
for whatever presolve leaves behind.

**`delta_d` is separate and equally mandatory.** `delta_p` bounds `T` from
above; it cannot help when `A` itself is rank deficient, since then `A T A` is
singular for every `T`. Empty rows are the clearest instance — the canonicalizer
removes those (§2.2), but presolve creates new ones and `delta_d` covers what
detection misses.

Both floors are applied **at construction, unconditionally, by the KKT builder**
— not by a reactive regularization module. A zero entry yields `inf` in IEEE754
with no exception, `0*inf` yields `NaN`, and `NaN` compares false against every
threshold: the solve then runs silently to the iteration limit and reports
`MAX_ITERATIONS` on a solvable problem. There is nothing for a breakdown
detector to detect.

Regularization perturbs the Newton direction, so every regularized solve is
followed by iterative refinement against the **unregularized** residual.

**Only valid for `Q = 0`.** For QP, eliminating `dx` requires `(Q + T^-1)^-1`,
which is a full matrix — the reduction is not available. See §10.2.

**Caveat:** a single dense column `a_j` of `A` contributes `t_j a_j a_j`, a full
dense rank-1 update that makes `A T A` dense. Dense columns must be detected
(the matrix analyzer reports them; Netlib `israel` has 19) and split out via
Sherman-Morrison.

### 10.2 Augmented / quasi-definite KKT — LP and QP

```
    | -(Q + T^-1 + delta_p I)          A          | | dx |     | . |
    |                                             | |    |  =  |   |
    |          A                (D_s + delta_d I) | | dy |     | . |
```

**Both deltas are mandatory here, and neither is optional decoration.** `T^-1`
is never inverted on this path, so the §10.1 floor does not apply — `delta_p`
replaces it.

- `delta_p` — a free column's zero entry of `T^-1` is exact and harmless as a
  *value*, which is why free columns select this path. But the (1,1) block still
  has to be definite, and `Q` is only positive *semi*definite: on a free column
  where `Q` is also zero the block has an entirely zero row.
- `delta_d` — `D_s` is zero on every **equality** row by construction, since an
  equality row has no slack. So the (2,2) block is singular across the whole
  equality block before rank deficiency in `A` is even considered. On an
  all-equality model it is entirely zero; `gas11` is 459 of 459 equality rows.

Quasi-definiteness is the objective, not a side effect. A quasi-definite matrix
— negative definite `(1,1)`, positive definite `(2,2)` — admits an `LDLᵀ`
factorization for **any** symmetric permutation. That is exactly what lets §10.3
fix an ordering from the sparsity pattern once and reuse it with no numerical
pivoting. Drop either delta and the matrix is merely symmetric indefinite, and a
fixed ordering is no longer safe.

This is the required path for QP and a valid path for LP.

**A high free-column share also selects this path.** Here a zero entry of
`T^-1` is exact and harmless — the zero only causes trouble when `dx` is
eliminated to form the normal equations. `gas11` is 375 free of 862 columns
(44%), so this is a real selection criterion, not a hypothetical one. Record the
reason in the `ReductionDescriptor`.

### 10.3 Invariants both backends rely on

- The **sparsity pattern is constant across all IPM iterations**; only values change.
  Ordering and symbolic factorization are computed **once** and reused.
- `cond(AΘAᵀ) ~ 1/μ²` by construction. Primal-dual regularization plus iterative refinement
  is required, not optional, and regularization events must be counted for diagnostics.
- Regularization **escalates on breakdown and decays on success**:
  `δ ← min(δ·escalate, δ_max)` after a failed factorization or refinement,
  `δ ← max(δ/decay, δ_floor)` after a clean solve. A pure ratchet is wrong — once
  escalated and never lowered, every later iteration solves a system perturbed
  more than it needs and refinement pays for it in passes. At `δ_max` with a
  still-failing factorization, return `NUMERICAL_ERROR` rather than escalating
  further.
- `MatrixAnalysis` **goes stale**. It is a pure function of the matrix handed to
  it, and presolve changes `free_columns` directly (free column singleton
  substitution exists to reduce it), along with `dense_columns` and `empty_rows`.
  A reduction chosen from the ingestion-time analysis is chosen on numbers that
  have since changed, so §10 selects from the analysis of the model it will
  actually factorize — after presolve and scaling.

---

## 11. Status codes

```
    OPTIMAL             converged per §9
    INFEASIBLE          primal infeasibility reliably detected
    UNBOUNDED           dual infeasibility reliably detected
    NONCONVEX           Q not PSD — detected via a wrong-signed pivot during factorization,
                        not by an up-front test (no cheap PSD test exists)
    MAX_ITERATIONS      iteration limit reached
    TIME_LIMIT          time limit reached
    NUMERICAL_ERROR     factorization or refinement failed irrecoverably
    NOT_CONVERGED       still running / stalled without a reliable classification
```

On stall or limit, return the **best iterate seen**, not the last one.
