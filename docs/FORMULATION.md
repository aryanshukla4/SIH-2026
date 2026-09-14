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

---

## 12. Simplex working form — **added 2026-09-13 (Module 23)**

§§3–11 describe the interior-point path. This section describes the second
engine, the revised simplex (`src/solver/simplex/`), which shares §1 and §2
unchanged and diverges from §3 onward. Both engines consume the same
`model::CanonicalProblem` and both produce a `model::Solution` that the same
`recover_solution()` postsolve consumes — postsolve reads values, never a
basis, so it needed no change.

### 12.1 Augmented working form

§2's canonical form is already bounded-variable:

```
    min c'x   s.t.   A_E x = b_E,   A_I x + s = b_I, s >= 0,   l <= x <= u
```

The simplex adds **one logical variable per row** so that a starting basis
exists by inspection:

```
    w = (x, xi) in R^(n+m)          the working vector
    Ahat = [ A | I_m ]              the working matrix
    Ahat w = b
    xi_i in [0, 0]                  for i <  num_equality   (fixed logical)
    xi_i in [0, +INF)               for i >= num_equality   (this IS §2's slack s)
    x_j  in [col_lower_j, col_upper_j]
```

`I_m` is **never materialized.** Working column `n+i` *is* `e_i`, and
`AugmentedMatrix` (`Basis.hpp`) answers `for_each_in_column(n+i)` with a
single implicit entry. This matches the canonicalizer's existing deliberate
choice not to store the slack block, and lets `A.csc`/`A.csr` be used as-is.

Range columns that the canonicalizer appends for ranged rows are ordinary
bounded structurals here. No special case.

### 12.2 Variable status

```
    enum class VarStatus : uint8_t { Basic, AtLower, AtUpper, Fixed, Free };
```

A `Basis` is `status` (length `n+m`, indexed by working index) plus `basic`
(length `m`, the working index occupying each basic *slot*). Both are
`std::vector`-backed and the struct is copyable, because branch-and-bound
needs to hand a parent's basis to a child.

`Basis::validate()` must check for a **duplicated** entry in `basic`, not
merely that the number of `Basic` statuses equals `m`. Counting alone cannot
catch it: `basic = [5, 5]` with both 5 and 7 marked `Basic` gives a matching
count and a rank-deficient basis. This was a real bug.

### 12.3 Index spaces — the primary bug class

Two different maps are in play and confusing them is the single most common
source of silent wrong answers here:

```
    ftran(v)  solves  B d = v         input indexed by ROW,  output by SLOT
    btran(v)  solves  B' rho = v      input indexed by SLOT, output by ROW
```

`B`'s column `k` is `Ahat` column `basic[k]`. So an FTRAN result's component
`k` belongs to basic *slot* `k` (hence to variable `basic[k]`), while a BTRAN
result's component `i` belongs to *row* `i` (hence to dual `y_i`). Every
routine in `detail/SimplexEngine` names its vectors accordingly.

### 12.4 Dual signs — falls out of §4, no global flip

With `y = B^-T c_B` and `d = c - Ahat' y`:

An inequality row's logical has cost `0` and bounds `[0, +INF)`. At its lower
bound, dual feasibility for a minimization requires `d >= 0`, and

```
    d_{xi_i}  =  0 - (e_i)' y  =  -y_i  >=  0     =>   y_i <= 0
```

which is exactly §4's `-y_I > 0` with **no negation applied anywhere.** Then

```
    z_j = max(d_j, 0),   v_j = max(-d_j, 0),   reduced_cost = z_j - v_j
```

§4's warning ("an earlier draft claimed the opposite... applying it would
invert every inequality dual") applies here verbatim. This is asserted in
`tests/unit/dual_simplex_test.cpp`, not assumed.

### 12.5 The dual pivot, sigma-folded

When basic variable `p` in slot `r` violates a bound, define

```
    sigma = +1  if  x_B[r] > u_p        (too high: must come down)
    sigma = -1  if  x_B[r] < l_p        (too low:  must go up)
    delta = sigma * (x_B[r] - (sigma > 0 ? u_p : l_p))     > 0
```

Folding `sigma` into the pivot row once, as `arow_j = sigma * alpha_j` where
`alpha = rho_r' Ahat`, removes every later sign case:

```
    eligible at lower bound  iff  arow_j > 0
    eligible at upper bound  iff  arow_j < 0
    ratio_j = d_j / arow_j                       (>= 0 for any eligible j)
    entering q = argmin ratio_j                  (bound-flipping may pass several)
    primal step t     = delta / arow_q
    dual   step theta = ratio_q
    y += theta * sigma * rho_r
```

### 12.6 Termination certificates

Each row is a proof obligation, not a heuristic:

| Condition | `SolverStatus` |
|---|---|
| no primal infeasibility, dual feasible | `Optimal` |
| infeasible row with no eligible entering column | `Infeasible` |
| a variable rests on an **artificial** bound, and a constructed ray proves the model itself unbounded | `Unbounded` |
| a variable rests on an artificial bound, no ray proves it, box escalation exhausted | `NotConverged` |
| iteration / time budget exhausted | `MaxIterations` / `TimeLimit` |
| LU breakdown that basis repair cannot fix | `NumericalError` |

The `Unbounded` row is deliberately not the inference "the artificial bound
kept growing, so it is probably unbounded." `ray_is_unbounded()` moves every
offending column off its artificial bound simultaneously, each in its own
objective-improving direction, FTRANs the aggregated column once, and checks
that nothing along the ray ever meets a **true** bound. Testing each column
alone is not enough: `min -x-y s.t. x-y<=1, -x+y<=1` is unbounded only along
a joint direction. When no ray can be constructed the engine reports
`NotConverged` rather than guessing, and the composite path hands the basis
to the primal simplex, which has no artificial box to be trapped by.

**An `Infeasible` reported while an artificial bound is active is not a
verdict about the model** — it is a verdict about the artificially boxed
problem. The engine escalates the box (`artificial_bound_growth`, up to
`max_artificial_rounds`) and only reports `Infeasible` when no artificial
bound was in play. Getting this wrong made `greenbea` report a confident,
completely false `Infeasible`.

### 12.7 Primal simplex blocking events

The primal engine (`PrimalSimplex.cpp`) chooses an entering column first and
a leaving one second, and the ratio test must consider **three** kinds of
blocking event, not one:

1. a basic variable reaches one of its own bounds;
2. the **entering** variable reaches its own opposite bound before any basic
   variable blocks (a bound flip — the basis does not change);
3. in phase 1 only, an *infeasible* basic variable reaches the bound it was
   violating and becomes feasible — this bounds the step even though the
   variable is not leaving a feasible region.

Phase 1 minimizes the sum of infeasibilities with `c1_j = -1` below lower,
`+1` above upper, `0` otherwise, and `d1 = -Ahat' y1`. Phase 2 begins with no
artificial variables to remove.

Bland's smallest-index rule is an **escape from cycling, not a pricing
rule.** It engages after a run of degenerate pivots and must be released the
moment any pivot moves a positive distance. Latching it permanently left
`greenbea` 1.7 away from feasible after 600,000 pivots; releasing it on the
first real step solved the same instance in 112,421.

### 12.8 Composite dual → primal

`--primal-cleanup` (default on) hands the dual engine's final basis to the
primal engine. That basis is already primal feasible for the true bounds,
because the artificial box is a subset of the true bounds, so phase 1 is
skipped entirely — `phase1_iterations == 0` is asserted in
`tests/unit/primal_simplex_test.cpp`.

---

## 13. Homogeneous self-dual embedding

> **Status (2026-09-14): PARTIALLY IMPLEMENTED.**
>
> | part | where | state |
> |---|---|---|
> | §13.1 embedding, §13.2 verdicts, §13.4 mu | `solver/Homogeneous.{hpp,cpp}` | **built**, `homogeneous_test` |
> | step size, centering, starting point, stopping criteria | `solver/HomogeneousStep.{hpp,cpp}` | **built**, `homogeneous_step_test` |
> | §13.3 the bordered Newton solve | `solver/HomogeneousNewton.{hpp,cpp}` | **built**, `homogeneous_newton_test` |
>
> `IpmOptions::homogeneous_self_dual` exists and defaults **off**; nothing reads
> it yet, so the interior-point path still behaves exactly as §§5–11 describe.
> What remains is wiring: an iteration loop calling §13.3 and §13.4 in order,
> and an A/B against the direct path.
>
> **Provenance.** §§13.3–13.5 are now transcribed from
>
> > E. D. Andersen and K. D. Andersen, *"The MOSEK interior point optimizer for
> > linear programming: an implementation of the homogeneous algorithm"*, in
> > High Performance Optimization, Kluwer, 2000, pp. 197–232 — cited as **[AA]**.
>
> They previously carried our own derivation. Where [AA] and that derivation
> disagreed, [AA] is right and the difference is called out in place.

§§5–11 describe the IPM applied directly to the problem of §2. That
formulation has a structural gap this section closes: **it cannot report
`Infeasible` or `Unbounded` at all.** If the model has no feasible point there
is no interior to follow, the iterates diverge, and the only honest outcome is
`MaxIterations`. `README.md` records `gas11` running away to `-7.5e10` under
exactly that gap.

The fix is not a detector bolted on afterwards. It is to solve a *different,
always-feasible* problem whose solution answers the question either way.

### 13.1 The embedding

Introduce two scalars `tau > 0` and `kappa > 0` and homogenize every constant
by `tau` — `b -> b*tau`, `c -> c*tau`, `l -> l*tau`, `u -> u*tau`:

```
    A_E x  - b_E tau                       = 0
    A_I x  + s - b_I tau                   = 0            s >= 0
    A' y   + z - v - c tau                 = 0            z, v >= 0
    c'x    - b'y - l'z + u'v + kappa       = 0
                 l tau <= x <= u tau
```

The fourth row is the **self-dual gap row**: `kappa` is exactly the duality gap
`b'y + l'z - u'v - c'x` carried as a variable. The system is homogeneous, so it
always has the trivial solution, and the complementarity pairs are those of §6
plus one more:

```
    (x - l tau, z)     (u tau - x, v)     (s, -y_I)     (tau, kappa)
```

`l'z` skips columns with an infinite lower bound and `u'v` those with an
infinite upper bound, exactly as in §5 — an infinite bound has no pair, so it
contributes no term here either.

### 13.2 Reading the answer

At convergence with `mu -> 0`, complementarity forces `tau * kappa -> 0`, so
exactly one of two things holds:

| Outcome | Meaning | Recover by |
|---|---|---|
| `tau > 0`, `kappa -> 0` | The model has an optimal solution | dividing `x, s, y, z, v` by `tau` |
| `tau -> 0`, `kappa > 0` | The model is primal OR dual infeasible | inspecting the ray, below |

In the second case `(x, s, y, z, v)` is itself a certificate, and which kind is
decided by the sign of the two objective terms:

```
    b'y + l'z - u'v  >  0     ->  PRIMAL INFEASIBLE   (a dual ray)
    c'x              <  0     ->  DUAL INFEASIBLE     (a primal ray, i.e.
                                                       the primal is UNBOUNDED)
```

These are the same Farkas certificates §12.6 and Module 24 produce; the
difference is that here they fall out of ordinary convergence rather than
needing a separate test.

**Source: [AA] Theorem 3** — *"Let `(x*, tau*, y*, s*, kappa*)` be a strictly
complementary solution to (HLF) such that `kappa* > 0`. If `c'x* < 0`, then the
dual problem is infeasible. Similarly if `b'y* > 0`, then the primal problem is
infeasible."* Our `b'y` becomes the full dual objective `b'y + l'z - u'v`, which
is [AA]'s verbatim at `l = 0, u = inf`.

Two things [AA] makes explicit that are easy to get wrong:

- **Strict complementarity is a precondition, not a technicality.** Both
  Theorem 2 (`tau* > 0` iff feasible) and Theorem 3 assume it. §13.4's
  centrality condition is what delivers it.
- **Both tests can fire at once.** [AA] after (1.2): *"at least one of `(-c'x*)`
  or `(b'y*)` must be positive"* — at least one, not exactly one. A model can be
  both primal and dual infeasible, so `classify_homogeneous` returns
  `Indeterminate` rather than picking one, and likewise when neither fires.

### 13.3 The Newton system is the old one with a border

**Source: [AA] (1.25)–(1.29) for the structure; §13.3a for the bound terms.**
Implemented in `solver/HomogeneousNewton.{hpp,cpp}`, tested by
`homogeneous_newton_test`.

> This subsection was previously written from our own derivation and was **wrong
> in one respect**, corrected below.

Homogenizing adds `tau` to every existing block and adds one scalar equation, so
after the §10 eliminations the system is the **same matrix** `K` as before,
bordered by one row and one column:

```
    [ -Theta^-1   A'    h_x         ] [ dx   ]   [ rd_hat         ]
    [   A         D_s  -b           ] [ dy   ] = [ rp_hat         ]
    [   g_x'     -b'   -(w+kappa/t) ] [ dtau ]   [ rg_hat - rtk/t ]
```

Writing `Theta_l = Z/(x - l tau)` and `Theta_u = V/(u tau - x)` for the two
halves of §10.1's diagonal:

```
    Theta^-1 = Theta_l + Theta_u                 diagonal, length n
    D_s      = S/Sigma_I                         inequality rows only
    h_x      = Theta_l l + Theta_u u - c         border COLUMN
    g_x      = Theta_l l + Theta_u u + c         border ROW
    w        = l' Theta_l l + u' Theta_u u       extra trailing term
```

**The correction: the border row is NOT the negated border column.** The earlier
text wrote the block as `[K h; -h' k/t]`, assuming the bordered system inherits
`K`'s symmetry. It does not. `g_x - h_x = 2c` whenever a finite bound is
present: eliminating `dz` and `dv` puts the *same* bound term on both the dual
row and the gap row, while `c` enters them with opposite signs. Writing the
solve from `-h'` puts a sign error on every `dtau`.

Even in standard form, where `g_x = -h_x` does hold, the earlier text was
misleading for a second reason: [AA] (1.26)'s column is `(-c; -b)` while its row
is `(-c', +b')`, the `b` block agreeing rather than flipping, because the second
block row carries `+A` against the first row's `+A'`.

A bordered system with a `1x1` trailing block is solved by **two solves with `K`
and a scalar Schur complement** — [AA] (1.28)/(1.29), in our coordinates:

```
    K (p; q) = (h_x; -b)               <- once per ITERATION
    K (u; v) = (rd_hat; rp_hat)        <- once per RIGHT-HAND SIDE

    dtau     = ( g_x'u - b'v - rho ) / ( g_x'p - b'q + w + kappa/tau )
    (dx; dy) = (u; v) - dtau (p; q)
```

with `rho = rg_hat - rtk/tau`; then `dz`, `dv`, `ds`, `dkappa` back out by
substitution.

So `KktBuilder`, `Ordering`, `Preconditioner` and `LinearSolver` are
**unchanged** — `K` is exactly §10.2's augmented KKT with `Q = 0`. No new
matrix, no new factorization, no new preconditioner.

**The cost is less than one extra solve per direction.** [AA] §1.5: *"even
though the system (1.25) has to be solved for different right-hand sides, the
system (1.28) is only solved once in each iteration. Therefore, the main
computational cost associated with the homogeneous algorithm compared to the
primal-dual algorithm is the additional solution of a linear equation system of
the form (1.28)."* The predictor and the corrector share `(p; q)`; only `(u; v)`
is recomputed. `refresh_border_solve` is a separate entry point from
`solve_homogeneous_newton` precisely so this cannot be silently lost — folding it
into the step would double the cost and every test would still pass.

**Inequality rows touch only `K`'s (2,2) block**, never the border: the gap row
does not involve `s` at all. Checked separately against a dense Jacobian.

### 13.3a Where the general-bounds border came from — **CLOSED**

[AA] §1.2 is explicit: *"For simplicity we will work with the LP problem in
standard form"*, `Ax = b, x >= 0`. Every equation in §13.3 is stated there for
`l = 0, u = inf`, which is why its border is the clean `-c`. Our canonical form
(§2) has genuine finite bounds, and homogenizing sends `l -> l tau`,
`u -> u tau`, so `tau` appears **inside the complementarity rows**:

```
    (x - l tau) .* z = mu           (u tau - x) .* v = mu
```

Eliminating `dz` and `dv` then leaves `dtau` terms in both the dual block row
and the gap row. Standard form has `l = 0` (multiplying the first by zero) and
no `u` pair at all, so both vanish.

**This was resolved by a change of variables, not by a new derivation.** The
shift

```
    x = l tau + s1,      s2 = (u - l) tau - s1,      s1, s2 >= 0
```

carries our bounded embedding onto [AA]'s `(HLF)` **exactly**, with

```
    s1 = x - l tau       s2 = u tau - x       y2 = -v       sigma = (z, v)
```

and, on inequality rows, `sigma_I = -y_I` already being standard form. Verified
against a dense Jacobian in both coordinate systems, agreeing to `1e-15` in
every block. Two details worth recording:

- The transformed gap row is **ours plus a multiple of the dual row**, not ours
  exactly: `T3 = -R3 - l'R2`. That is an identity, and the test asserts it
  rather than assuming it.
- The `c'l tau` terms introduced by shifting the objective **cancel** against
  those introduced by shifting the right-hand side, which is why §13.1's gap row
  `c'x - b'y - l'z + u'v + kappa` emerges from the transformation unchanged.
  This confirms §13.1 independently of the LP-duality argument it was written
  from.

**The correspondence is also the test oracle**, which is the better reason to
record it. `homogeneous_newton_test` carries two:

1. §13.1 transcribed as a dense Jacobian and factorized whole — catches an error
   in the **elimination**.
2. The shifted standard-form instance with [AA]'s system written verbatim, our
   direction mapped across and required to satisfy it — catches an error in the
   **border itself**, which oracle 1 cannot, because oracle 1 and the
   implementation are both derived from §13.1 and could share a mistake.

Confirmed complementary by mutation: with oracle 1 disabled, oracle 2 alone
still catches a border column that loses its bound term (failing at the dual
row), a border row written as `-h_x` (failing at the gap row) and a dropped `w`
(likewise) — each at exactly the row that term belongs to.

The shift needs finite bounds, since a column with an infinite bound cannot be
shifted by `l`. That is not a hole: such a column has no complementarity pair
and so contributes no bound term, leaving its border entry at [AA]'s `-c_j`.
Oracle 2 covers exactly what [AA] does not, and a separate test checks the
infinite-bound case collapses onto [AA] directly.

Measured on a random instance with finite bounds: `||g_x + h_x|| = 7.14` and
`w = 17.6`. The old `[K h; -h' k/t]` was not a small error.

**The alternative was rejected.** Converting to standard form *in production* —
rather than only inside a test — would take us to `2n + m_I` columns and `m + n`
rows. `Canonical.hpp` keeps bounds native on purpose, and `gas11` is 44% free
columns, which the shift cannot even express.

### 13.4 Iteration control — step length, centering, start, stopping

**Source: [AA] §1.4 and Table 1.1.** Implemented in
`solver/HomogeneousStep.{hpp,cpp}`, tested by `homogeneous_step_test`.

[AA] §1.4.2 writes `x := (x; tau)` and `s := (s; kappa)` and from there treats
the embedding as an ordinary primal-dual system with `n+1` complementarity
pairs. We do the same, with our pair list from §6 rather than standard form's
single `(x, s)`:

```
    (x - l tau, z)     (u tau - x, v)     (s, -y_I)     (tau, kappa)
```

so [AA]'s `n+1` becomes `active_pair_count + 1`. That substitution is the only
change made to the step size and the stopping criteria, and it is what `mu`
already does:

```
    mu = [ sum (x - l tau).*z + sum (u tau - x).*v + sum (-s).*y_I + tau*kappa ]
         / (active_pair_count + 1)
```

| piece | [AA] | value |
|---|---|---|
| centering `gamma := (1-a)^2 min(1-a, beta_1)`, then `eta := 1-gamma` | (1.12) | `beta_1 = 0.1` |
| ratio test `argmax { (x;tau;s;kappa) + a d >= 0 }` | (1.21) | — |
| step `a := min(beta_3 a_max, 1)`, reduced until (1.20) holds | §1.4.3 | `beta_2 = 1e-8`, `beta_3 = 0.9999` |
| starting point `(e, 1, 0, e, 1)` | (1.22) | — |
| optimal: `rho_P, rho_D, rho_A` within tolerance | §1.4.5 | `1e-8, 1e-8, 1e-10` |
| infeasible: those plus `rho_G`, and `tau <= rho_I max(1, kappa)` | §1.4.5 | `rho_I = 1e-10` |
| ill-posed: `mu <= rho_mu mu_0` and `tau <= rho_I min(1, kappa)` | §1.4.5 | `rho_mu = 1e-10` |
| significant digits `rho_A = \|c'x - b'y\| / (tau + \|b'y\|)` | (1.24) | — |

Three things in that table are **not** [AA] verbatim, each marked GENERALIZED in
the header with the specialization that recovers [AA]:

1. **The ratio test carries a `tau` term the standard form cannot see.** `tau`
   scales the bounds, so the quantity `x - l tau` has direction `dx - l dtau`.
   At `l = 0` that term vanishes, which is why [AA] never writes it. Omitting it
   lets a step violate a bound while every coordinate of `dx` looks safe;
   `homogeneous_step_test` builds a direction where it is the only thing that
   binds, so deleting it turns a finite step into an unbounded one.

2. **`rho_A` uses the full dual objective** `b'y + l'z - u'v`, the same
   substitution §13.2 already makes. At `l = 0, u = inf` it is (1.24) verbatim.

3. **[AA] (1.22)'s `y := 0` is not usable unchanged.** It is admissible only
   because standard form has no inequality rows and therefore no `(s, -y_I)`
   pair. We have them, and at `y_I = 0` that pair's product is exactly zero — on
   the boundary, not near it — so the centrality condition (1.20) fails at every
   positive step and the solve stalls on iteration one with nothing to show for
   it. Measured, not predicted: that is what the first run did. Inequality rows
   start at `y_I = -1`; equality rows, unrestricted and pairless, stay at 0, so a
   model with no inequality rows still gets `y = 0`.

**Why (1.20) is not optional.** [AA]: *"if all the iterates satisfy the
condition (1.20), then they converge towards a strictly complementary
solution."* [AA] Theorem 2 (`tau* > 0` iff the problem is feasible) and Theorem 3
(which kind of infeasibility) are both stated for a **strictly** complementary
solution. Skipping the centrality test does not make the solver slower; it makes
the verdict unsound.

**One parameter has no citation.** `rho_bar_G` is used in [AA] §1.4.5's
infeasibility criterion but is absent from Table 1.1. We set it to `1e-8`,
matching its two siblings in that same criterion, and say so at the field.

**Not implemented from [AA]:** the elaborate starting point (1.23), which is
defined by two Newton solves and so is blocked by §13.3a; Gondzio multiple
centrality corrections (§1.4.2), whose `beta_4` is likewise absent from Table
1.1 — the direct path already has its own correctors.

### 13.5 Why this is opt-in

`IpmOptions::homogeneous_self_dual` defaults **off**. The direct formulation of
§§5–11 is what every measured result in `README.md` was produced with, and a
reformulation that changes the iterates on every instance must prove itself
before it replaces them. Turning it on is how the A/B is run — once §13.3a is
resolved and there is a Newton solve for it to turn on.
