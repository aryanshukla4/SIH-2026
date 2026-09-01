# LP format — dialect notes

Companion to `MPS-FORMAT-NOTES.md`. Everything here is a decision the reader had
to make because the format does not decide it for us.

**LP format has no standard.** CPLEX, Gurobi and GLPK each accept a different
language. This reader takes **CPLEX as the base and accepts Gurobi's superset**
— rejecting a construct later is a one-line change, while adding one after
tests are written against the narrower language is not.

It buys no benchmark coverage: Netlib, MIPLIB 2017 and the Mittelmann sets all
ship as MPS, and QPLIB has its own format. What it buys is a model a person can
read, which matters for demonstration and for hand-written tests.

---

## 1. Lexical

| | |
|---|---|
| Comments | `\` to end of line, anywhere |
| Newlines | **insignificant** — a constraint may wrap freely |
| Case | keywords are case-insensitive; identifiers are not |
| `<=` | also spelled `=<` and `<` |
| `>=` | also spelled `=>` and `>` |
| Identifier chars | alphanumeric plus ``! " # $ % & ( ) , . ; ? @ _ ' ` { } ~ |`` |

Newline insignificance is why this reader uses a token stream over the whole
buffer rather than the line-and-field scanner the MPS reader uses. MPS is
column-significant; LP is not.

### The exponent ambiguity

`3e5` is one number. `3 e5` is a coefficient of 3 on a variable named `e5`.
So is `3e` followed by a non-digit. The rule: an `e` starts an exponent **only**
when digits follow, optionally after a sign. Getting this wrong silently scales
a coefficient by 10⁵.

`3x`, `3 x` and `3 * x` are all the same term.

---

## 2. Sections

| Section | Accepted spellings |
|---|---|
| Objective | `Minimize` `Minimise` `Min` `Minimum` / `Maximize` `Maximise` `Max` `Maximum` |
| Constraints | `Subject To` `subject to` `such that` `st` `st.` `s.t.` `s.t` |
| Bounds | `Bounds` `Bound` |
| Integer | `General` `Generals` `Gen` `Integer` `Integers` |
| Binary | `Binary` `Binaries` `Bin` |
| Semi-continuous | `Semi-Continuous` `Semis` `Semi` |
| SOS | `SOS` |
| Terminator | `End` |

The objective section is mandatory and must come first. `Subject To` and
`such that` are two tokens, and `Semi-Continuous` lexes as three (the `-` is an
operator), which is why the section matcher peeks ahead.

---

## 3. Where a model can silently change

These are the cases that produce a *different model* rather than an error, so
each has a named test.

### 3.1 The right-hand side must be a constant

This is the CPLEX rule, and taking it literally is what makes unnamed
constraints parseable at all. Given

```
Subject To
 x + y <= 5
 x - y >= 1
```

nothing separates the `5` from the `x` that follows. A reader that accepts
variables after the operator folds the second constraint into the first one's
right-hand side and reports success. Every LP writer normalizes variables to
the left, so the restriction costs nothing real; a file that violates it gets
`UnsupportedFeature` with an explanation.

### 3.2 A label is not a variable

In `c2: x - y >= 1`, `c2` is an identifier like any other. Consuming it as a
variable creates a phantom column. Both a following `:` and a section keyword
rule an identifier out as a variable reference.

### 3.3 Constants inside the expression move to the right

`x + 3 <= 10` constrains `x` to `<= 7`. For a **ranged** row the constant shifts
**both** sides: `-5 <= x + y + 3 <= 10` constrains `x + y` to `[-8, 7]`.
Shifting only one side is the easy mistake.

### 3.4 The objective constant has no sign trap

LP writes the constant directly: `obj: 3 x + 7` means `+7`. **MPS negates it** —
an RHS entry on the objective row is the negation of the constant. The two
readers must not be made to agree by copying one into the other; they agree
because both are right, and there is a test asserting exactly that.

### 3.5 The quadratic `/2`

```
Minimize
 obj: [ 2 x^2 + 4 x * y ] / 2
```

means the objective term is `x² + 2xy`. The model stores `½xᵀQx`, so `xᵀQx`
must be `2x² + 4xy`, i.e. `Q_xx = 2` and `Q_xy = Q_yx = 2` with `Q` held full
symmetric. An off-diagonal term written once contributes **half** its
coefficient to each of `(i,j)` and `(j,i)`.

With **no** `/d`, the bracket is read as the objective contribution itself, so
`[ 2 x^2 ]` means `2x²` and `Q_xx = 4`. CPLEX mandates the divisor; this is the
natural reading of a file that omits it.

Same factor-of-two trap as `QUADOBJ`. Getting it wrong scales every QP objective
by two and nothing downstream notices, so it is pinned by a hand-computed test.

---

## 4. Bounds

| Written | Meaning |
|---|---|
| *(absent)* | `[0, +inf)` — same default as MPS |
| `x >= 5` | lower `= 5` |
| `x <= 7` | upper `= 7` |
| `3 <= x <= 9` | both |
| `x = 4` | fixed |
| `x free` | `(-inf, +inf)` |
| `-inf <= x <= 5` | lower infinite |
| `Binary` section | `[0, 1]`, integer |

A leading signed number needs **backtracking**, not lookahead: `-5 <= x` and
`-5 x + y <= 3` start identically, and `-5` is two tokens. The lexer therefore
exposes a save/restore point, used in exactly one place.

**Deliberate divergence from the MPS reader:** MPS treats `UP` with a negative
value and no prior `LO` as implying a free lower bound (see
`MPS-FORMAT-NOTES.md`). LP format states bounds explicitly, so that quirk does
not apply and `ReaderOptions::negative_upper_implies_free_lower` is **not**
consulted here. Noted so the two readers are not assumed to disagree by
accident.

---

## 5. Not supported

Reported as `UnsupportedFeature`, never skipped — silently dropping a section
changes the model, and a model that quietly differs from its file is worse than
one that fails to load.

- SOS sets
- Semi-continuous columns
- Indicator constraints
- General constraints
- Quadratic **constraints** (a quadratic **objective** is supported)

---

## 6. Assembly

MPS is column-major, so `MpsReader` builds CSC first and transposes to CSR.
LP is **row**-major — a constraint is one expression — so `LpReader` builds CSR
first and transposes to CSC.

`SparseBuilder::count(row, col)` is orientation-agnostic, so the same two-pass,
no-sort, `O(nnz)` pipeline serves both; only the direction of the counting-sort
transpose differs. Adding LP required no new data structure.

One non-obvious constraint: `SymbolTable` stores `string_view`s and does not own
the characters. Column names are views into the input text, which outlives the
parse. Row names are not always — an unnamed constraint gets a generated `R<k>`
— so those live in a `std::deque<std::string>`, whose references stay valid
across insertion. A `std::vector<std::string>` would not do: reallocation moves
the elements, and with the small-string optimization it moves the characters
themselves.
