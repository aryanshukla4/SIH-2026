# Benchmarks — ingestion layer

Reproduce with:

```sh
cmake --preset release && cmake --build build
python scripts/fetch_netlib.py --all          # optional, for the real instances
./build/bench/parse_bench tests/data/netlib/*.mps
```

Measurements below: GCC 15.2 (MinGW-w64 UCRT), `-O3`, Windows 11, single
thread. Timings are the best of N repeats, since we are measuring the parser
rather than the OS scheduler.

---

## Parse throughput

| Instance | rows | cols | nnz | size | parse | MB/s | Mnnz/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| synthetic/small | 1 000 | 1 500 | 8 983 | 414 KB | 2.1 ms | 201 | 4.4 |
| synthetic/medium | 20 000 | 30 000 | 239 959 | 10.4 MB | 71.8 ms | 145 | 3.3 |
| synthetic/dense-cols | 20 000 | 30 000 | 299 934 | 12.5 MB | 81.8 ms | 153 | 3.7 |
| synthetic/large | 100 000 | 150 000 | 1 499 920 | 62.8 MB | 697 ms | 90 | 2.2 |
| netlib/afiro | 27 | 32 | 83 | 3 KB | 0.8 ms | — | — |
| netlib/israel | 174 | 142 | 2 269 | 79 KB | 1.6 ms | 47 | 1.4 |
| netlib/25fv47 | 821 | 1 571 | 10 400 | 358 KB | 6.7 ms | 53 | 1.6 |
| netlib/greenbea | 2 392 | 5 405 | 30 877 | 1.0 MB | 26.9 ms | 37 | 1.1 |
| netlib/80bau3b | 2 262 | 9 799 | 21 002 | 1.1 MB | 18.5 ms | 58 | 1.1 |

**1.5 M nonzeros in 0.7 s.** Small-instance MB/s is dominated by fixed cost
(file open, symbol-table setup) and is not meaningful — `afiro` is 3 KB.

Throughput falls from ~200 MB/s to ~90 MB/s as instances grow. That is the
symbol table outgrowing L2 and the mapped file outgrowing the TLB, and it is
the first thing to attack in the performance-hardening phase (parallel
`COLUMNS` parsing, chunked on line boundaries).

For reference, a reader built on `std::istringstream` and `std::stod` — the
obvious implementation — typically lands around 5–15 MB/s, and is also
locale-dependent.

---

## Memory

**24.5–24.9 bytes per nonzero**, holding *both* CSR and CSC.

That is the number the design predicts, and measuring it is how the storage
claim gets checked rather than asserted:

| Component | Bytes/nnz |
|---|---:|
| value (`double`) | 8 |
| index (`int32`) | 4 |
| **CSR subtotal** | **12** |
| CSC (same again) | 12 |
| **Total** | **24** |
| offsets `(major+1) × 4`, amortized | +0.5 – 2.9 |

`int64` indices would make this 32 B/nnz. On a 100 M nonzero instance that is
800 MB of pure index traffic — which is why `Index` is `int32` with an overflow
check at load time rather than a silent truncation.

Peak RSS across the whole benchmark run (every case, largest 62 MB of text) is
**187 MB**.

---

## Correctness against published data

All 19 Netlib instances in `tests/data/netlib/` parse and satisfy every
structural invariant. Nine have dimensions published with the Netlib
collection; the parsed counts match all nine **exactly**:

| Instance | rows × cols | nnz |
|---|---|---:|
| afiro | 27 × 32 | 83 |
| adlittle | 56 × 97 | 383 |
| 25fv47 | 821 × 1 571 | 10 400 |
| 80bau3b | 2 262 × 9 799 | 21 002 |
| e226 | 223 × 282 | 2 578 |
| etamacro | 400 × 688 | 2 409 |
| israel | 174 × 142 | 2 269 |
| shell | 536 × 1 775 | 3 556 |
| stair | 356 × 467 | 3 856 |

`rows` excludes the objective row and `nnz` counts constraint-matrix entries
only — objective coefficients live in `c`, not in `A`. The header inside
`afiro.mps` reads `N=32, M=28, NZ=115` because that `M` counts the objective
row and that `NZ` counts objective coefficients; the corpus test pins the
accounting down so the discrepancy cannot be mistaken for a parser bug.

---

## Findings the corpus surfaced

**Dense columns are real.** `israel` has **19 columns** above the
`2·√m` threshold. Each contributes a full rank-1 dense `m×m` update to
`AΘAᵀ`, so the normal-equations matrix goes dense on an instance whose `A` is
99% sparse. This is `ARCHITECTURE-REVIEW.md` §2.3 showing up in the first
corpus we looked at, not a hypothetical.

**MILP instances parse correctly.** Five carry `MARKER INTORG/INTEND`:

| Instance | cols | integer |
|---|---:|---:|
| bell5 | 104 | 58 |
| egout | 141 | 55 |
| flugpl | 18 | 11 |
| gt2 | 188 | 188 |
| rgn | 180 | 100 |

Had `Problem` omitted integrality flags, all five would have loaded as LPs and
solved to fractional answers with no error raised.

**Ill-conditioning is visible before any factorization.** Value dynamic range
`max|a| / min|a|`: afiro 23, adlittle 5.4 × 10⁴, 25fv47 1.2 × 10⁶. The last is
the scaler's problem, and the analyzer reports it for free during load.

**Presolve targets exist.** `25fv47` has 1 empty row, 40 singleton rows, 28
singleton columns, and 1 duplicate-column candidate — all in the analyzer
output, all cheap presolve wins.

---

## What is not measured yet

- **Parallel parsing.** Single-threaded today. Chunking `COLUMNS` on line
  boundaries is the planned next step.
- **Pinned-memory transfer.** The allocator slot exists; there is no CUDA
  toolkit installed to measure against.
- **gzip.** zlib is not present in the current toolchain, so `.mps.gz` returns
  `UnsupportedFeature` rather than being mis-parsed. Needed for MIPLIB, which
  ships compressed.
- **Instances beyond ~1.5 M nonzeros.** The `int32` overflow path is tested by
  construction but not on a real instance that large.

---

## Measurement reliability — read before quoting a number

Parse throughput on the development machine (MinGW, Windows) is **not stable
enough to support a small optimisation claim**. Repeated runs of an identical
binary over identical input were measured at 24, 41, 56, 82 and 85 MB/s on the
same 64 MB instance — a 3.5x spread with nothing changed between runs. One
`spread` reading reached 104%.

`bench/parse_bench.cpp` therefore reports the **spread** alongside the best
time, rather than a best-of-N alone. A best-of-N presented on its own is what
turns this noise into a confident false claim: any two builds can be made to
look 2x apart by rerunning.

Rules that follow from this:

- **Do not quote a throughput improvement smaller than the reported spread.**
  On this machine that currently rules out anything under roughly 2x.
- Prefer measurements noise cannot corrupt. For a parser those exist: cache hit
  rates, allocation counts, bytes per nonzero, symbol-table probe counts. The
  column-memo in `MpsReader` is justified by a **57.5% measured hit rate over
  the Netlib corpus**, not by a timing delta.
- Numbers destined for the submission need a quiet machine, several runs, and
  the spread quoted with them.

`B/nnz` and `peak RSS` are stable and can be quoted directly — they do not
depend on timing.
