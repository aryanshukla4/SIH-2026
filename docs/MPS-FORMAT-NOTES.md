# MPS format notes — quirks, traps, and exact semantics

Reference for the reader implementation. Every rule here is a real behaviour of real files in
Netlib / MIPLIB, and most of them are silent-wrong-answer traps rather than parse errors.

---

## 1. Fixed vs free format

**Netlib** ships fixed-format. **MIPLIB 2017** ships free-format (`.mps.gz`).

Fixed format field positions (1-indexed, inclusive):

| Field | Columns |
|---|---|
| 1 | 2–3 |
| 2 | 5–12 |
| 3 | 15–22 |
| 4 | 25–36 |
| 5 | 40–47 |
| 6 | 50–61 |

Free format is whitespace-delimited with no column significance.

**Why it matters:** names containing spaces are legal in fixed format and impossible in free
format. A fixed-format file parsed as free format will split such names and silently create
spurious rows/columns.

**Our approach:** auto-detect, default to free, fall back to fixed on failure. Detection heuristic
— if every data line's tokens land inside the fixed field windows *and* any line has more than 6
whitespace-separated tokens, treat as fixed.

---

## 2. Sections

```
NAME          problem name
OBJSENSE      optional; MAX / MAXIMIZE / MIN / MINIMIZE
ROWS          row declarations
COLUMNS       matrix entries, column-ordered
RHS           right-hand sides
RANGES        optional; converts a row to a two-sided range
BOUNDS        optional; variable bounds
QUADOBJ       optional; quadratic objective (triangle)
QMATRIX       optional; quadratic objective (full)
SOS           optional; special ordered sets
ENDATA
```

Sections may be absent. `RHS`, `RANGES`, `BOUNDS` entries reference rows/columns by name and may
carry a leading set name that is conventionally ignored.

---

## 3. ROWS — the objective is an `N` row

```
N  <name>     free row
L  <name>     aᵀx ≤ rhs
G  <name>     aᵀx ≥ rhs
E  <name>     aᵀx  = rhs
```

**Trap:** the **first `N` row is the objective**. Any *subsequent* `N` rows are **free rows** and
must be dropped, not parsed as constraints. Files with multiple `N` rows are common — they hold
alternative objectives.

Row bounds as stored (`INF = 1e20`):

| Type | `[row_lower, row_upper]` |
|---|---|
| `L` | `(−INF, rhs]` |
| `G` | `[rhs, +INF)` |
| `E` | `[rhs, rhs]` |
| `N` | dropped (objective) or `(−INF, +INF)` (free row) |

---

## 4. COLUMNS — column-ordered, which we exploit

Entries are grouped by column. Each line carries one column name and one or two
`(row name, value)` pairs.

**This ordering is why the reader needs no COO staging.** Two passes give CSC directly with exact
allocation; a single counting-sort transpose then yields CSR with sorted column indices.

### 4.1 Integer markers

```
    MARKER                 'MARKER'                 'INTORG'
    ... integer columns ...
    MARKER                 'MARKER'                 'INTEND'
```

Columns between `INTORG` and `INTEND` are **integer**. Markers may nest or repeat.
**Omitting this makes MILP impossible** — the integrality is nowhere else in the file.

Marker lines are recognized by the literal `'MARKER'` field, quotes included. The first field on
the line is a marker *name* and is arbitrary — do not match on it.

### 4.2 Duplicate entries

The same `(row, column)` pair may appear more than once. **Values are summed.** Silently keeping
the last one is a common bug and changes the model.

### 4.3 Objective coefficients

An entry whose row name is the objective row is a `c` coefficient, not a matrix entry.

---

## 5. RHS — and the objective-constant sign trap

Ordinary entries set the right-hand side of a constraint row.

**An RHS entry on the objective row sets the objective constant — negated.**

```
    RHS  RHS_SET  COST  5.0      ⟹   c₀ = −5.0
```

This convention is near-universal (CPLEX, Gurobi, HiGHS) but is not stated in most format
descriptions. Getting the sign wrong shifts every reported objective value.

---

## 6. RANGES — four cases, all different

A `RANGES` entry `R` on row with rhs `b` converts a one-sided row to a two-sided one.
**The sign of `R` only matters for `E` rows:**

| Row type | Range `R` | `[row_lower, row_upper]` |
|---|---|---|
| `L` | any | `[b − \|R\|, b]` |
| `G` | any | `[b, b + \|R\|]` |
| `E` | `R ≥ 0` | `[b, b + R]` |
| `E` | `R < 0` | `[b + R, b]` |

`R = 0` on an `E` row leaves it an equality.

Applying `|R|` uniformly to `E` rows is wrong and is the single most common RANGES bug.

---

## 7. BOUNDS — and the negative-`UP` trap

| Key | Effect on `[col_lower, col_upper]` | Value? |
|---|---|---|
| `LO` | `lower = v` | yes |
| `UP` | `upper = v` (see trap) | yes |
| `FX` | `lower = upper = v` | yes |
| `FR` | `(−INF, +INF)` | no |
| `MI` | `lower = −INF` | no |
| `PL` | `upper = +INF` | no |
| `BV` | `[0, 1]`, integer | no |
| `LI` | `lower = v`, integer | yes |
| `UI` | `upper = v`, integer | yes |
| `SC` | semi-continuous, `upper = v` | yes |

### 7.1 The negative-`UP` trap

```
    UP  BND  x  -5.0        with no prior LO on x
```

Default lower bound is `0`, so this would give the empty interval `[0, −5]`.
Readers disagree on the resolution:

- **Most (CPLEX, Gurobi, HiGHS):** set `lower = −INF`
- **Some older readers:** set `lower = 0` and treat it as infeasible
- **A few:** set `lower = v` and `upper = 0`

**Our behaviour:** default to `lower = −INF`, expose it as an option, and **log every time it
fires** so a discrepancy against another solver is traceable.

### 7.2 `MI` and the upper bound

Some historical readers set `upper = 0` when `MI` is applied. Modern convention leaves `upper`
untouched. We follow the modern convention.

### 7.3 Integer variables with no explicit bound

An integer column declared via `MARKER` with no `BOUNDS` entry: convention is `[0, +INF)`, though
some older files intend `[0, 1]`. We use `[0, +INF)` and log when an integer column has no
explicit upper bound.

---

## 8. QUADOBJ / QMATRIX — the factor-of-½ trap

The objective is

```
    ½ xᵀQx + cᵀx
```

- **`QUADOBJ`** lists only the **lower triangle**. Off-diagonal entry `(i,j,v)` implies
  `Q[i][j] = Q[j][i] = v`. The `½` is part of the objective definition, already accounted for.
- **`QMATRIX`** lists **all** entries, so off-diagonals appear twice and must agree.

Getting this wrong scales every quadratic objective by exactly 2 — the solution point may still
look plausible, which makes it hard to spot. A hand-checkable test pins it down.

We store `Q` expanded to **full symmetric CSR**.

---

## 9. General parsing rules

- Lines beginning with `*` in column 1 are comments.
- Blank lines are skipped.
- Section headers begin in column 1; data lines are indented.
- Numbers may use `D` exponents (`1.5D+02`) in old Fortran-produced files — normalize to `E`
  before `std::from_chars`.
- Any bound with `|v| ≥ 1e20` is infinity.
- Names are case-sensitive.
- `ENDATA` terminates; trailing content is ignored.

---

## 10. Netlib and MIPLIB specifics

**Netlib LP** ships in a compressed form requiring the `emps` utility to expand to fixed MPS.
`scripts/fetch_netlib.py` handles expansion. A few instances have known quirks documented in the
Netlib README (e.g. missing bounds sections, `RANGES` on `E` rows).

**MIPLIB 2017** ships `.mps.gz`, free format. Decompression uses zlib — a compression library,
not a solver library, so it is clean under the PS "from scratch" constraint.

---

## 11. What we reject rather than mis-parse

`SOS` sets and semi-continuous variables are **recorded and reported**, and the loader returns a
clear `UNSUPPORTED_FEATURE` status rather than silently dropping them. Silently ignoring an SOS
set changes the model and produces a confidently wrong answer.
