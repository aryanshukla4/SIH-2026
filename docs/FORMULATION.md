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
    minimize      ½ xᵀQx + cᵀx

    subject to    A x = b
                  x ≥ 0
```

Every row is an equality and every variable is non-negative. This is the form
`Ax + s = b, x ≥ 0, s ≥ 0` with the slacks appended as **columns**:

```
    A_canonical = [ A_structural | I_slack ]        x_canonical = [ x ; s ]
```

Writing it that way is not cosmetic. The `Ax + s = b, s ≥ 0` form cannot
express an equality row — there is nowhere to put a slack that must be zero —
and equality rows are the majority in real instances (516 of 821 rows in Netlib
`25fv47`). Appending slack columns only to *inequality* rows handles both kinds
uniformly.

The column layout is `[structural | slack | bound]`, where `bound` holds the
auxiliary variables introduced when a finite upper bound had to become a
constraint row. `CanonicalProblem::slack_begin()` is where `s` starts.

For LP, `Q = 0`.

### 2.1 Column transforms are one affine map

Every variable transform is an instance of `x = d·x' + t` with `d ∈ {+1, −1}`,
so the whole update is uniform:

```
    A_new  = A D                     (scale column j by d_j)
    bounds shift by  −A t
    Q_new  = D Q D                   (Q_new[i][j] = d_i d_j Q[i][j])
    c_new  = D (Q t + c)
    offset = ½ tᵀQ t + cᵀt
```

| Original bounds | Transform | `d` | `t` |
|---|---|---|---|
| `l ≤ x`, no upper | shift | +1 | `l` |
| `x ≤ u`, no lower | reflect | −1 | `u` |
| `l ≤ x ≤ u` | shift, then a bound row `x' + t = u−l` | +1 | `l` |
| free | split `x = x⁺ − x⁻` | — | — |

Free variables are the one case outside the map. Splitting doubles the column
and worsens conditioning (`x⁺` and `x⁻` are perfectly correlated near the
solution), so the record is kept distinguishable — a later IPM can handle free
variables natively without the loader changing.

### 2.2 Row transforms

| Original | Canonical |
|---|---|
| `aᵀx = b` | `aᵀx = b` — no slack |
| `aᵀx ≤ u` | `aᵀx + s = u`, `s ≥ 0` |
| `aᵀx ≥ l` | negate the row, then `−aᵀx + s = −l` |
| `l ≤ aᵀx ≤ u` | `aᵀx + s = u` with a bound row for `s` |

Row negation flips the sign of that row's dual; recovery inverts it.

---

## 3. Primal-dual variables

| Variable | Length | Meaning | Sign requirement |
|---|---|---|---|
| `x` | `n` | primal variables | `x > 0` (strictly, in the interior) |
| `s` | `m` | row slacks | `s > 0` |
| `y` | `m` | duals for `Ax + s = b` | see §4 |
| `z` | `n` | duals for `x ≥ 0` | `z > 0` |
| `w` | `m` | duals for `s ≥ 0` | `w > 0` |

---

## 4. Sign convention — **read this before touching any dual**

The handoff docs adopt a convention in which the slack dual is *eliminated* as `w = −y`,
which forces `y < 0` throughout. That is internally consistent, but it inverts every
step-length test and every diagnostic print relative to the standard literature.

**This codebase carries `w` explicitly, with `w ≥ 0`.**

Cost: one vector of length `m`. Benefit: every positive variable is genuinely positive, so
step-length code, initialization, and debug output all read naturally, and results can be
cross-checked against textbooks and other solvers without mental sign inversion.

The relationship to the docs' convention is exactly:

```
    w_here  =  −y_docs
    y_here  =   y_docs
```

Modules that consume duals must state which convention they use. **No silent conversions.**

Derivation, for the record. With

```
L = ½xᵀQx + cᵀx − yᵀ(Ax + s − b) − zᵀx − wᵀs
```

stationarity gives

```
∂L/∂x  =  Qx + c − Aᵀy − z  =  0
∂L/∂s  =  −y − w            =  0     ⟹   w = −y
```

Carrying `w` explicitly means we keep both `y` (free sign) and `w ≥ 0` and enforce
`w + y = 0` as part of the dual residual rather than by substitution.

---

## 5. Residuals

```
    rp   =  A x + s − b                (length m)   primal feasibility
    rd   =  Q x + c − Aᵀy − z          (length n)   dual / stationarity
    rxz  =  X z − μ 1                  (length n)   x–z complementarity
    rsw  =  S w − μ 1                  (length m)   s–w complementarity
```

`X`, `S`, `Z`, `W` denote `diag(x)`, `diag(s)`, `diag(z)`, `diag(w)`.
**They are never materialized** — always elementwise vector operations.

Under the docs' `w = −y` convention, `rsw` is written `rsy = −Sy − μ1`. Identical equation.

---

## 6. Complementarity measure

```
    μ  =  ( xᵀz + sᵀw ) / ( n + m )
```

Equivalently `μ = (xᵀz − sᵀy) / (n + m)` under the docs' convention.

Both complementarity pairs contribute. Using `μ = xᵀz / n` alone is wrong whenever slacks are
explicit.

---

## 7. Newton system

```
    Q Δx − AᵀΔy − Δz        =  −rd
    A Δx + Δs               =  −rp
    Z Δx + X Δz             =  −rxz
    W Δs + S Δw             =  −rsw
```

Four blocks. Do not use a three-variable system.

---

## 8. Step lengths — **two, not one**

Primal and dual variables reach their boundaries at different distances. Using a single α
discards the larger of the two steps and costs 20–30% more iterations.

```
    α_p  =  η · max { α ∈ (0,1] :  x + αΔx ≥ 0  and  s + αΔs ≥ 0 }
    α_d  =  η · max { α ∈ (0,1] :  z + αΔz ≥ 0  and  w + αΔw ≥ 0 }
```

with safety factor `0 < η < 1` (typically `η ≈ 0.995`).

Update:

```
    x ← x + α_p Δx        y ← y + α_d Δy
    s ← s + α_p Δs        z ← z + α_d Δz
                          w ← w + α_d Δw
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

Eliminating `Δz`, `Δs`, `Δw`, then `Δx`:

```
    ( A Θ Aᵀ + D ) Δy  =  rhs        Θ = Z⁻¹X ≻ 0,   D = W⁻¹S ≻ 0
```

SPD, so Cholesky applies.

**Only valid for `Q = 0`.** For QP, eliminating `Δx` requires `(Q + X⁻¹Z)⁻¹`, which is a full
matrix — the reduction is not available. See §10.2.

**Caveat:** a single dense column `aⱼ` of `A` contributes `θⱼ aⱼaⱼᵀ`, a full dense rank-1
update that makes `AΘAᵀ` dense. Dense columns must be detected (the matrix analyzer reports
them) and split out via Sherman–Morrison.

### 10.2 Augmented / quasi-definite KKT — LP and QP

```
    ⎡ −(Q + X⁻¹Z)      Aᵀ    ⎤ ⎡ Δx ⎤     ⎡ · ⎤
    ⎢                        ⎥ ⎢    ⎥  =  ⎢   ⎥
    ⎣      A         W⁻¹S    ⎦ ⎣ Δy ⎦     ⎣ · ⎦
```

Quasi-definite, so an `LDLᵀ` factorization with a fixed (symmetry-based) ordering exists.
This is the required path for QP and a valid path for LP.

### 10.3 Invariants both backends rely on

- The **sparsity pattern is constant across all IPM iterations**; only values change.
  Ordering and symbolic factorization are computed **once** and reused.
- `cond(AΘAᵀ) ~ 1/μ²` by construction. Primal-dual regularization plus iterative refinement
  is required, not optional, and regularization events must be counted for diagnostics.

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
