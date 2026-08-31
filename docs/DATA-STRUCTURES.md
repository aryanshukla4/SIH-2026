# Data structures — contracts and invariants

Storage decisions for the ingestion layer, with the reasoning. Companion to
`FORMULATION.md` (the mathematics) and `MPS-FORMAT-NOTES.md` (the input format).

---

## 1. Scalars and indices

| Type | Definition | Why |
|---|---|---|
| `Real` | `double` | FP64 throughout. See §6 for the GPU caveat. |
| `Index` | `int32_t` | Halves index bandwidth in CSR/CSC traversal, which is memory-bound. Matches the path cuSPARSE prefers. |
| `WideIndex` | `int64_t` | Fallback beyond 2³¹ nonzeros. The reader checks for overflow rather than truncating. |

There is **no `Scalar` type.** The handoff datatype doc proposes a struct holding
a value plus a runtime `precision` field; that wrapper adds nothing over `double`
and the runtime field puts a branch inside every arithmetic operation. If mixed
precision is ever wanted, template on the scalar type and resolve at compile time.

### The infinity convention

```cpp
inline constexpr Real INF = 1e20;
```

Any bound with magnitude ≥ 1e20 **is** infinity — the MPS convention. This is
deliberately *not* `std::numeric_limits<double>::infinity()`:

- Files contain literal `1e20` and `1e30` values that must compare equal to infinity.
- Arithmetic on a true IEEE infinity produces `NaN` (`INF - INF`), where `1e20`
  produces a large finite number. Bound shifting during canonicalization does
  exactly this arithmetic.

Test with `is_infinite()` / `is_finite_bound()`. **Never with `==`.**

---

## 2. `Vector<T, Alloc>`

Owning, aligned, 1-D buffer.

| Property | Value | Why |
|---|---|---|
| Alignment | 64 bytes | One cache line; satisfies AVX2 (32 B) and AVX-512 (64 B); divides the 128 B GPU coalescing granule. |
| Precision | template parameter | Not a runtime field — this type exists to be traversed in tight loops. |
| Memory space | allocator policy | Compile-time, same reason. |
| Copy | **deleted** | An accidental deep copy of a million doubles is invisible in source and ruinous in profile. Duplication goes through `clone()`. |
| Move | yes, `noexcept` | |
| Construction | **uninitialized** by default | Zeroing a buffer about to be overwritten is a real cost at load time. Ask for `Vector(n, 0.0)` when you need zeros. |

Allocator policies: `HostAllocator` (default, aligned pageable), `PinnedAllocator`
(`cudaHostAlloc` — 2–3× faster host↔device transfer), `DeviceAllocator`
(`cudaMalloc`). The latter two compile in when CUDA is present; nothing else
changes.

---

## 3. `Span<T, Space>`

Non-owning view. **The currency of module boundaries** — modules take spans,
only owners hold `Vector`.

Without a view type, every function must take `Vector&`, which makes it
impossible to pass a sub-range, a slice of the workspace arena, or a raw device
pointer without copying.

`Span` is `std::span` plus a `MemorySpace` tag. The tag is enforced by
`static_assert`: dereferencing or iterating a device span from host code is a
compile error, not a runtime fault. Crossing memory spaces has to be written
down, because the compiler cannot otherwise see the difference.

`data()` + `size()` are exactly what `cudaMemcpy` and a kernel launch need, so
this is also the zero-copy GPU handoff point.

---

## 4. `SparseMatrix<Format, T, Idx, Alloc>`

### Invariants

All four are checked by `validate()` and must hold whenever a matrix crosses a
module boundary:

| # | Invariant |
|---|---|
| **I1** | `offsets.size() == major_dim + 1`, `offsets[0] == 0`, `offsets[major] == nnz`, non-decreasing |
| **I2** | within each major slice, `indices` is **strictly increasing** — sorted, therefore duplicate-free |
| **I3** | every index in `[0, minor_dim)` |
| **I4** | `values.size() == indices.size() == nnz` |

**I2 is the one that gets violated.** cuSPARSE *requires* sorted indices within
each row for many routines; unsorted input silently produces wrong results or
drops to a slow path. Duplicates are summed at construction (MPS permits repeated
entries), never kept — keeping the last value silently changes the model.

An invariant nobody wrote down is an invariant nobody maintains, which is why
these are stated here and checked in code rather than assumed.

### Both orientations

`SparseMatrixPair` holds CSR **and** CSC of the same matrix.

The handoff datatype doc lists CSR only. Every IPM iteration computes `A*x`
(row-oriented) *and* `Aᵀ*y` (column-oriented). Computing `Aᵀy` from CSR alone
requires atomics on GPU or a cache-hostile scatter on CPU. Holding both costs 2×
index+value memory and is what every serious solver does.

They cannot drift apart, because of how they are built: the MPS reader produces
CSC directly (the `COLUMNS` section is column-ordered), and CSR is a single
counting-sort transpose of it — which also establishes I2 for free.

### Memory footprint

| Layout | Bytes per nonzero |
|---|---|
| CSR, `int32` indices | 12 (8 value + 4 index) |
| CSR + CSC | 24 |
| CSR, `int64` indices | 16 |

Plus `(major_dim + 1) × sizeof(Idx)` for offsets. These are the targets the
benchmark suite measures against.

---

## 5. `DenseMatrix<T, Alloc>`

**Column-major.** Element `(i, j)` at `data[j * ld + i]`.

The handoff doc does not state a layout, and the choice is not free: cuBLAS,
LAPACK and every BLAS are column-major. Row-major means transposing on every call
into them — a per-iteration cost for the reduced Newton system.

The leading dimension `ld` is padded up from `rows` so each column starts on a
64-byte boundary, letting a vectorized column traversal use aligned loads.

---

## 6. `Workspace`

Single-allocation scratch arena. Two-phase: `reserve<T>(n)` during setup returns
a handle, `commit()` performs one allocation, `get<T>(handle, n)` resolves handles
to spans.

The handoff doc names 14 vectors (`x s y z dx ds dy dz b c rp rd rxz rsy`) with no
statement about where they live. Allocated individually, with residuals rebuilt
per iteration, that puts `malloc` inside the IPM hot loop and fragments the heap
across hundreds of iterations.

The IPM's memory requirement is fully known once the problem is parsed. Allocate
once; allocate nothing per iteration. Every carved span is independently 64-byte
aligned.

> **Note for the IPM owner:** Mehrotra predictor-corrector needs *two* direction
> sets (affine-scaling and corrector), not one. Size the workspace accordingly —
> see `ARCHITECTURE-REVIEW.md` §3.1.

---

## 7. `NameArena`

One character blob plus an offset table. Two allocations total for any number of
names.

A MIPLIB instance can carry millions of names. `std::vector<std::string>` means
one heap allocation per name plus ~32 bytes of `std::string` overhead each — for
1 M names, ~32 MB of overhead and 1 M allocations before a single character is
stored. Lookup returns a `string_view` into the blob, with no copy.

Names are not optional: solution reporting, error messages that identify the
offending row, and MPS round-trip testing all need them. The handoff `Problem`
omits them entirely.

---

## 8. `LinearOperator`

Abstract `apply` / `apply_transpose`. Not in the handoff docs; added now because
retrofitting it later means editing every module.

What it buys:

1. **`AΘAᵀ` need never be materialized.** In an IPM its *values* change every
   iteration but its *structure* does not, and an iterative solver only ever
   applies it. Materializing is a choice — and a bad one when a dense column is
   present, since one dense column makes `AΘAᵀ` fully dense.
2. Iterative linear solvers (CG on normal equations, MINRES on the augmented
   system) without changing callers.
3. Matrix-free first-order methods — the credible GPU path on consumer hardware,
   where FP64 factorization runs at 1/64 speed but bandwidth-bound SpMV does not.

Cost: one virtual call per apply, against millions of FLOPs inside it.

`apply_transpose` is declared separately rather than derived, because the
efficient implementation reads a *different storage orientation* — CSC for the
transpose, CSR for the forward direction. This is why `Problem` holds both.

---

## 9. Error handling

Failures are values, not exceptions across module boundaries: `Expected<T>`
(modelled on C++23 `std::expected`, so migration is mechanical) carrying an
`Error` with a code, message, line, column and section.

The handoff signature `Problem loadProblem(const std::string&)` has no error
channel at all. On a 500 MB MPS file, "parse failed" without a line number is not
a usable diagnostic.
