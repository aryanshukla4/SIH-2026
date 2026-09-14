# SIH 2026 — Version Progress Script (SIH26119)

**Purpose of this document.** To show a teacher/evaluator, with real evidence (commit
history, real bugs found, real fixes made — no invented numbers), that this project is a
genuine iterative engineering process: each version found a real problem and fixed it.
**Do not add speed numbers that aren't in this file** — every claim below is traceable to
a commit, a doc file, or a code comment that exists in the repo right now.

**Golden rule for the room:** the story is *"we kept finding real problems in our own
work and fixing them"* — not *"version 2 is 5x faster than version 1"*. We do not have
clean timing data across versions, so we never claim one. The bug-fix trail is the
honest, strong story, and it is also harder for anyone to poke a hole in than a speed
number would be.

---

## 0. The one-line framing (say this first)

> "Sir/Ma'am, hum aapko sirf ek final product nahi dikha rahe — hum aapko dikhana chahte
> hain ki humne kaise kaam kiya. Har version mein humein ek real problem mila — kabhi
> apne hi design mein, kabhi apne code mein — aur humne usse fix kiya, verify kiya, aur
> aage badhe. Yeh saari cheez git history mein hai, hum bana nahi rahe."

---

## 1. Version 0 — The design was reviewed *before* a line of solver code was written

**When:** 2026-08-30 to 2026-08-31 (`e62ee11`, `34d22c2`)

Before writing the actual solver, the team wrote out the intended architecture and then
**deliberately audited it for flaws** — this is `docs/ARCHITECTURE-REVIEW.md`, and it is
still in the repo, unedited, showing the original mistakes.

**What the review caught in the original design:**

| # | Problem in the original design | Why it mattered |
|---|---|---|
| 1 | Design said "delegate the linear solve to cuSOLVER / cuDSS" | **Compliance risk** — PS 26119 requires the solver not be built on an existing solver library. cuDSS is a direct sparse solver, not a primitive like cuBLAS. |
| 2 | One shared step length `alpha` for both primal and dual variables | Forces the smaller of two step lengths onto both sides — standard estimate is 20–30% more iterations for no reason |
| 3 | One direction set `(dx, ds, dy, dz)`, no separate affine/corrector | Blocks Mehrotra predictor-corrector entirely — the difference between roughly 15–25 iterations and ~50 |
| 4 | Convergence check was `‖rp‖ < tolerance` (absolute) | Absolute tests break at both ends: unreachable when `‖b‖` is large, meaningless-pass when `‖b‖` is tiny |
| 5 | No regularization design | `cond(AΘAᵀ) ~ 1/μ²` by construction as an IPM converges — this is not an edge case, it happens on every run |
| 6 | No infeasibility detection mechanism | `INFEASIBLE` was listed as a return status with nothing that could ever produce it |

**How to say it simply:** *"Humne code likhne se pehle apna khud ka design plan padha aur
usme 6 real problems dhoond nikale — including ek compliance-breaking problem (dusre ki
solver library use karne wala tha). Humne yeh sab fix kiya code likhne se pehle."*

**Show this:** open `docs/ARCHITECTURE-REVIEW.md` on screen, scroll to section 2.4 (the
cuSOLVER/cuDSS compliance problem) and 3.1 (the predictor-corrector problem).

---

## 2. Version 1 — Foundation: ingestion, and a discipline of honest measurement

**When:** 2026-08-30 to 2026-09-02 (`34d22c2` → `4eb87a3`, ~9 commits)

This version has **no solver yet** — it's the parser, canonicalizer, and the first real
external-validation habit. The important thing to show here is not speed, it's that the
team measured itself honestly from day one, including admitting when a benchmark number
was unreliable.

**Real, documented facts from `docs/BENCHMARKS.md`:**
- Parser handles 1.5 million nonzeros in 0.7s, 24.5–24.9 bytes per nonzero (both CSR and
  CSC held together) — measured, not estimated.
- All 19 Netlib benchmark instances parse with **exactly correct dimensions**, matched
  against the 9 instances Netlib publishes officially.
- Five instances (`bell5`, `egout`, `flugpl`, `gt2`, `rgn`) carry `MARKER INTORG/INTEND` —
  the doc states plainly: *"Had `Problem` omitted integrality flags, all five would have
  loaded as LPs and solved to fractional answers with no error raised."* — i.e. the team
  found and closed a silent-wrong-answer risk before it ever became a bug.
- **Self-caught measurement mistake, documented in the same file:** repeated runs of the
  identical binary on identical input measured 24, 41, 56, 82, 85 MB/s — a 3.5x spread
  with *nothing changed*. The team's own rule, written into the doc: *"do not quote a
  throughput improvement smaller than the reported spread."*

**How to say it simply:** *"Iss version mein solver tha hi nahi abhi — sirf file padhna
tha. Lekin humne yahi se ek habit banayi: kabhi bhi number bolne se pehle usse dobara
measure karo, aur agar number stable nahi hai to usko mat bolo. Yeh discipline poore
project mein continue hui."*

**Show this:** `docs/BENCHMARKS.md`, the "Measurement reliability" section at the bottom.

---

## 3. Version 2 — The first real solver: replacing the compliance-risk design with our own math

**When:** 2026-09-04 to 2026-09-07 (`dafcc47`, `4f16c55`, `bccd393`)

This is where Version 0's problem #1 (cuSOLVER/cuDSS delegation) actually gets fixed in
code: the solver ships with its **own** matrix-free Conjugate Gradient / MINRES solve and
its **own** from-scratch incomplete-Cholesky (IC(0)) preconditioner — cuSPARSE/cuBLAS are
used only for primitive operations (SpMV, dot products, triangular solves), never as a
linear-solver library. This boundary is written down and testable:

```sh
grep -rn "highspy\|cplex\|gurobi" --include=*.cpp --include=*.cu --include=*.hpp src/ include/
```
Expect: no output. (This is already in `docs/SIH-DEMO-SCRIPT.md` §7 as a live-checkable
claim — same grep works here.)

This version also adds the **first independent correctness check**: HiGHS used purely
offline, as an answer-checking oracle (`docs/HIGHS-COMPARISON.md`), never linked into the
solver binary.

**How to say it simply:** *"Version 0 mein hum bol rahe the ki ek dusre ki library use
karenge linear solve ke liye. Version 2 mein humne woh khud likha — apna CG solver, apna
IC(0) preconditioner. Aur humne yeh bhi bola: 'hum apna answer khud nahi trust karte, ek
independent solver (HiGHS) se cross-check karte hain.'"*

---

## 4. Version 3 — Presolve done, MILP added, and three self-caught bugs (2026-09-08)

**When:** all four on 2026-09-08, `6fc7146` (01:13) → `070b956` (04:42) → `1713ebb`
(11:13) → `8962542` (11:57) — a single, intense day of iteration.

This is the richest version for the "we test ourselves and catch our own mistakes"
story. Three real, sequential bugs were found and fixed on the `markshare_4_0.mps`
MILP benchmark instance, each one more serious than the last:

**Bug 1 — Binary detection.** MPS format can mark a binary variable two ways: an explicit
`BV` marker, or `INTORG ... UP bnd x 1` (integer type with bounds [0,1]). The presolve
code only recognized the first form. Fixed in `src/solver/gpu/BranchAndBound.cu`
(`is_binary_like`).

**Bug 2 — A presolve rule left a row un-simplified.** `markshare_4_0`'s rows mix binary
columns with one continuous "filler" column. The fix (`MilpPresolve.cpp`,
"absorbing singleton" elimination) is explained in full in this session's earlier answer
about "point four" — the short version: fold the continuous column out of the equality
row algebraically, so a mixed row becomes a pure discrete row that the discrete-only
cutting-plane rules can actually use.

**Bug 3 — the serious one: a silent false-optimality report.** Before the fix above was
done completely, an earlier version of the same rule fixed the continuous column at
whatever value minimized its own cost, **without checking what the row actually
required**. Result: the solver reported `objective = 0.0` and claimed the problem was
solved — while the true optimum (confirmed against HiGHS) is `1.0`. This is not a
"didn't converge" bug, it's a "confidently wrong" bug — the most dangerous class,
because it looks like success. It's documented in `src/solver/MilpPresolve.cpp` lines
147–153 and covered by a dedicated regression test in
`tests/property/milp_presolve_test.cpp` (`make_absorbing_row_problem`).

**Also this version:** MILP branch-and-bound layer added (best-first search via a
`std::priority_queue`, warm-started child nodes, cover/GCD cutting planes computed once
at the root) — commit `1713ebb` — and cross-row cutting plus a binary-column eligibility
fix — commit `8962542`.

**Honest limit, not hidden:** on the *harder* `markshare` family instances, the LP bound
still doesn't tighten with search depth — measured across three runs at 2,000 / 4,014 /
10,316 nodes, `best_bound` stuck in `[-2.2e-6, -2.5e-8]`. This is a known, documented,
unsolved limitation (adversarial structure against generic cover/GCD cuts), not swept
under the rug.

**How to say it simply:** *"Ek hi din mein humein teen bugs mile — pehla chota, doosra
usse bada, aur teesra sabse khatarnaak: solver bol raha tha 'answer 0 hai, solved!' jab
ki asli answer 1 tha. Humne yeh khud pakda, apne hi test se, kisi outsider ne nahi
bataya. Aur humne yeh bhi khule taur pe likha hai ki abhi bhi kuch harder cases pe hum
stuck hain — hum chhupa nahi rahe."*

**Show this:** `git log --oneline` for the four same-day commits; if there's time,
`tests/property/milp_presolve_test.cpp`'s `make_absorbing_row_problem` test.

---

## 5. Version 4 — right now, in progress (uncommitted, but real and running)

This is the most honest thing you can show a teacher: **work in progress, mid-fix,
visible in `git diff` right now** — not polished for the demo, actually happening.

```sh
git status
git diff --stat
```
Expect to see: `PredictorCorrector.hpp/.cu`, `Solve.cu`, `Options.hpp`,
`solver_gpu_algorithms_test.cpp` modified, nothing else.

**What's being fixed, and why (both are real, both are in the code right now):**

**(a) A measured algorithmic weakness — degenerate instances throttle every variable's
step length.** The step-length rule takes one single global minimum over *every* dual
coordinate. Confirmed by the team on real instances (`bandm.mps`, `grow15.mps`,
`scfxm1.mps`): the dual step length collapsed to ~0 for many consecutive iterations
while the residual it was supposed to fix stayed completely flat. The fix is **Gondzio
(1996) multiple centrality correctors** — a published, standard IPM technique, not
invented — implemented in `src/solver/gpu/PredictorCorrector.cu`
(`apply_gondzio_correctors`): each pass re-centers exactly the complementarity pairs an
extended trial step would push out of a healthy range, reusing the same matrix
factorization (so it's nearly free), and is only kept if it doesn't shrink either step
length.

**(b) A found, wasted computation, removed.** The outer solve loop
(`Solve.cu::solve_problem`) already computes `update_mu` + `compute_residuals` once per
iteration to check convergence. `run_iteration` was **silently redoing the exact same
computation again internally** just to fill its own diagnostics — a full extra
SpMV-based pass, every single iteration, of every solve, for nothing. Fixed by passing
the caller's already-computed value in instead of recomputing it.

**How to say it simply:** *"Yeh abhi live chal raha hai — hum aapko committed, polished
kaam nahi dikha rahe, hum aapko dikha rahe hain ki abhi is second bhi hum apna kaam check
kar rahe hain. Humne dekha ki kuch tricky (degenerate) sawaalon mein solver bahut chhote
steps le raha tha bina wajah — hum ek standard technique (Gondzio corrector) laga rahe
hain isse theek karne ke liye. Aur humne ek jagah dhoondi jahan solver har iteration mein
ek hi calculation do baar kar raha tha — usko hataya."*

**Caveat — say this honestly if asked:** this change is not yet committed and not yet
re-run through the full `ctest` suite tonight. If a judge/teacher asks "is this proven
working," the honest answer is: *"It's implemented and the same tests that cover
predictor-corrector already had to be updated to call it — we're running the full suite
before it goes in as a commit."* Do not claim it's verified until `ctest --test-dir
build-cuda --output-on-failure` has actually been rerun after this change.

---

## 5b. Version 5 — a second, independent LP engine: the revised simplex (2026-09-13)

**This is the version with the single biggest measurable jump, and the number is easy to
say in one breath: 6–7 out of 19, to 19 out of 19.**

Until this version the project had exactly **one** LP algorithm — the GPU interior-point
method. Three separate problems in our own docs all traced back to that one fact, and we
had written all three down *before* fixing them:

1. **Convergence.** `README.md`'s corpus sweep: only 6–7 of 19 local Netlib instances
   reached `Optimal`. Several (`stair`, `bell5`, `etamacro`, `25fv47`, `standata`)
   plateaued *just* above tolerance — our own note called them "genuinely stuck, not just
   cut off early."
2. **Verdicts the codebase could not produce at all.** `SolverStatus::Unbounded` existed
   as an enum value that **nothing ever returned**. `gas11` ran away to `-7.5e10`;
   `README.md` called that "an architectural gap, not a quick fix."
3. **MILP warm starts.** `BranchAndBound.hpp` states the limitation in its own header:
   every node cold-starts, because warm-starting an interior-point method across a bound
   change is a hard, still-researched problem — which is exactly why production MILP
   solvers use simplex for node relaxations.

**What we built.** A **revised simplex** engine, host-only, in *both* its dual and primal
forms, sharing one factorized-basis core. Sparse LU with Markowitz ordering and threshold
partial pivoting, product-form basis updates, a bound-flipping dual ratio test, dual
phase 1 by artificial bounds, primal phase 1 by sum of infeasibilities, Bland's rule as a
cycling escape. It reuses the **existing** canonicalize → presolve → scale → reconstruct
pipeline completely unchanged — nothing in presolve or postsolve had to be edited, which
is itself evidence the module boundaries were drawn in the right place three versions ago.

**The measured result — verified, not claimed:**

| | IPM | dual simplex | primal simplex |
|---|---|---|---|
| `Optimal`, 19 local Netlib instances | 6–7 | **19** | **19** |
| `Unbounded` ever returned | never | `gas11` | `gas11` |

Every objective cross-checked against `scripts/oracle_check.py`'s published table,
including its two documented archive-drift entries — and on `greenbea` our
`-7.2555248130e+07` matches the modern LP DASA / HiGHS value rather than the 1988
archive's number, which we say out loud rather than hiding as a mismatch.

**The compliance answer, because this is the version most likely to be challenged.** The
vendored `HiGHS/` tree contains its own dual simplex at `HiGHS/highs/simplex/HEkkDual.*`.
**We did not read it**, deliberately, and `docs/HIGHS-COMPARISON.md` §1 now sets out
component by component which published mathematical definition each piece was built from.
The honest evidence that it is an independent derivation is **the shape of the bugs it
produced** — all six are written down in `module.txt` §23: a sign convention that had to
be re-derived from scratch, a basis check that counted statuses instead of detecting a
duplicated slot, an artificially-bounded subproblem's infeasibility mistaken for a verdict
about the real model, and a Bland's-rule latch that never released and left `greenbea`
1.7 away from feasible after 600,000 pivots. Ported code does not fail in those ways.

**A side effect worth mentioning:** the simplex engine has no CUDA dependency, so
`cmake --preset release` on plain Windows — no WSL2, no CUDA toolkit — now produces a
working `solve` binary for the first time. Before this, the whole solver was unbuildable
without a GPU.

**Verified as of 2026-09-13, both builds:**

```sh
cmake --preset release  && cmake --build build      && ctest --test-dir build      # 17/17
cmake --preset cuda     && cmake --build build-cuda && ctest --test-dir build-cuda # 19/19
```

**What we did NOT do, said plainly.** Dual steepest edge, the Harris two-pass ratio test,
Forrest–Tomlin updates and hypersparse FTRAN/BTRAN are all known improvements on what is
here (we currently use Dantzig pricing, a single-pass guarded ratio test, product-form
updates and dense working vectors). They are left for a *measured* pass, one technique per
commit with before/after corpus numbers. The B&B warm start — problem 3 in the list above
— is **not** done either: `BranchAndBound.cu` still re-canonicalizes per node, which can
change the canonical shape between parent and child, and a basis index space cannot
survive that. So of the three problems this version was meant to address, two are fixed
and the third is unblocked but not yet delivered.

**How to say it simply:** *"Pehle humare paas sirf ek hi tarika tha LP solve karne ka, aur
19 mein se sirf 6–7 problems theek se solve ho rahe the. Ab humne ek doosra, bilkul alag
algorithm khud se banaya — simplex — aur ab 19 ke 19 solve ho rahe hain. Aur ek problem
(`gas11`) ka jawab 'iska koi answer hai hi nahi' hai — pehle humara solver yeh bata hi
nahi sakta tha, ab woh proof ke saath batata hai."*

---

## 5c. Version 6 — a third engine, and answers the solver could not give before (2026-09-14)

**The one-line version: three different algorithm families now, and the solver
can finally tell you WHICH of your constraints is the problem.**

### (a) PDLP — a first-order method, built in measured stages

Version 5 gave us simplex alongside the interior-point method. This adds a
third, from a family neither of them belongs to: **PDLP**, primal-dual hybrid
gradient. It **never factors a matrix** — its whole inner loop is two sparse
matrix-vector products. That matters for us specifically, because our own
architecture review measured FP64 *factorization* on this laptop GPU at ~64x
penalised (slower than the CPU) while SpMV is only ~2x. PDLP is the one
algorithm shaped to fit that.

It was built one enhancement at a time, each landing as its own commit with
before/after numbers, because the paper's own ablation says why: baseline PDHG
solves 50 of 383 instances where full PDLP solves 283. Ours, on the 19-instance
Netlib set:

| stage | correct | best speedup |
|---|---|---|
| baseline PDHG | 12/19 | — |
| + adaptive step size | 12/19 | 3.6x |
| + adaptive restarts | 14/19 | 3.9x |
| + primal weights | **18/19** | 20.6x |

**Say this part honestly if asked, because it is the more impressive claim:**
we found **three errors in the published papers**, each by measurement rather
than by reading.

1. The adaptive step-size rule **diverges as literally printed**. We
   instrumented it and watched the step size ratchet 1.0 → 8.6 while the
   iterate blew up to 1.4e20 — with every individual step passing the paper's
   own acceptance test.
2. The dual objective's sign contradicts the paper's own notation section. One
   three-line counterexample settles it: `min -x` over `0 ≤ x ≤ 1`.
3. The trust-region subroutine's reduction **silently drops finite upper
   bounds**, which our models have. We proved the test catches it by
   deliberately injecting the paper's version: a variable walks to 5.0 instead
   of stopping at its bound of 1.0, with no crash and a better-looking answer.

That is the strongest available evidence the work is ours. **Code copied from a
working implementation does not reproduce a paper's errata and then fix them.**
(Google's own PDLP sits in `or-tools/` in this repo. It was not read — same
rule we applied to HiGHS's simplex in Version 5.)

### (b) The solver can now say WHICH constraints are wrong

Previously the best answer to a broken model was "infeasible". Now
(`module.txt` §26) it returns an **irreducible infeasible subsystem** — the
specific rows that contradict each other, where dropping any one of them makes
the rest satisfiable.

This is the feature an actual refinery planner wants. "Your schedule is
impossible" is not actionable; "these four constraints are mutually impossible"
is.

And it came almost free, which is the nice part of the story: a 1990 theorem
(Gleeson and Ryan) says the answer is exactly the **support of the Farkas
certificate** — and our dual simplex was already computing that certificate as
a byproduct of the pivot that fails, then throwing it away. We stopped throwing
it away.

### (c) Interior point — the gap is closed, by a fourth engine

Version 4's notes called the interior-point method's inability to report
`Infeasible` or `Unbounded` an architectural gap. It was, and this closes it:
Module 25 is the **homogeneous self-dual embedding**, the same construction
MOSEK and CPLEX barrier use, now a fourth engine at `--method=hsd`.

The numbers are the headline. The interior-point *family* goes from **6-7/19**
to **18/19** on the corpus — matching PDLP, one behind the simplex — and it can
now answer "this model is impossible" instead of running to the iteration limit.

Two things to say honestly and unprompted:

- **`gas11`'s `Unbounded` is presolve's verdict, not the engine's.** Every
  engine reports it at iteration 0. HSD's infeasibility detection is proven by
  unit tests, not by the corpus.
- **The direct interior-point path is unchanged and still 6-7/19.** HSD is a
  separate engine, because that path is GPU-resident and the embedding is
  host-only. Both ship.

The interesting engineering story here is the same shape as the PDLP one. The
MOSEK paper (Andersen & Andersen 2000) works in **standard form** throughout,
and real models have bounds `l <= x <= u`, which changes the Newton system's
border. Rather than hunt for a paper stating the bounded case, we mapped our
problem onto theirs with a change of variables — and then kept that map as a
**test oracle**, so the bound terms are checked against the published algorithm
rather than against our own reasoning. Verified to 1e-15, and proven able to
fail: with the obvious oracle switched off, the second one alone still catches
three different wrong borders, each failing at exactly the row that term
belongs to.

And one measured default worth repeating because it is counter-intuitive: the
inner linear solver's iteration budget started at 500 and gave 14/19. At 5000
the five failures became `Optimal` in **fewer outer iterations**. An inexact
Newton direction does not make this method slower — it makes it **stall**,
because the step it produces is one the centrality test rejects outright.

### Verified, 2026-09-14

```sh
cmake --preset release && ctest --test-dir build        # 25/25
cmake --preset cuda    && ctest --test-dir build-cuda   # 27/27
```

| engine | correct verdicts on 19 Netlib instances |
|---|---|
| dual simplex | 19/19 |
| primal simplex | 19/19 |
| PDLP | 18/19 (`greenbea` open) |
| HSD (homogeneous, new) | 18/19 (`greenbea` open) |
| interior point (direct) | 6-7/19 |

*(`gas11` is genuinely unbounded and has no published optimum — so a perfect
score is 18 `Optimal` plus one `Unbounded`, not 19 `Optimal`.)*

### What is still missing — say this unprompted

- **The GPU backend for PDLP is not written.** The interface is there for it,
  but the measurement that would prove this hardware earns its GPU has not been
  run. We can explain why PDLP *should* suit it; we cannot yet show the number.
- **`greenbea`** is solved by both simplex engines and by nothing else.
- **The GPU backend for PDLP is still the biggest open item**, and it is the one
  the problem statement's GPU language actually rests on.
- **`greenbea`** is solved by both simplex engines and by neither first-order
  nor interior-point engine.
- **Do not say "the IPM detects infeasibility."** The HSD engine does; the
  direct interior-point path still does not, and both ship.

**How to say it simply:** *"Ab humare paas teen alag-alag tarike hain LP solve
karne ke, aur teeno alag family se hain. Aur agar aapka model solve nahi ho
sakta, toh ab solver sirf 'nahi ho sakta' nahi bolta — woh batata hai KAUN SE
constraints aapas mein takra rahe hain. Aur haan — humne research papers mein
teen galtiyan dhoondhi aur theek kiin, measurement se, padh kar nahi."*

---

## 6. What's next (from our own tracked plan — not invented for this talk)

There's a full internal plan (`Phase 0` through `Phase 4`) targeting the next concrete
goal: **every LP instance reaching `Optimal`, including the large ones** (`datt256` —
262,144 columns, `ken-18`, `maros-r7`). It names specific, already-identified next
defects — no iterative refinement against the true unregularized system yet, a fixed CG
tolerance that stops demanding accuracy exactly when it's needed most, a path-selection
rule (normal-equations vs augmented system) that's specified but not wired up yet. This
is deliberately not claimed as done — it's the honest "here's what's next" answer if
asked "so is it perfect now?"

**How to say it simply:** *"Hum yahin nahi ruk rahe — humare paas already agla plan hai,
aur usme bhi humne apne aap se kuch cheezein likhi hain jo abhi missing hain. Hum khud
apna kaam track kar rahe hain, sirf demo ke liye nahi bana rahe."*

---

## 7. Suggested run order for the room (what to *show*, and exactly where)

| Step | Command / file to open | Say it goes with |
|---|---|---|
| 1 | `git log --format="%h %ad %s" --date=short` | §1–4, point at the dates/messages as you talk through each version |
| 2 | `docs/ARCHITECTURE-REVIEW.md` §2.4 and §3.1 | §1 (Version 0 design flaws) |
| 3 | `docs/BENCHMARKS.md`, bottom section | §2 (measurement honesty) |
| 4 | `grep -rn "highspy\|cplex\|gurobi" --include=*.cpp --include=*.cu --include=*.hpp src/ include/` | §3 (own math, no library) |
| 5 | `tests/property/milp_presolve_test.cpp` → `make_absorbing_row_problem` | §4 (the 3-bug MILP story) |
| 6 | `git status` / `git diff --stat` | §5 (live, in-progress work) |
| 7 | `./build/tools/solve/solve tests/data/netlib/gas11.mps --method=dual-simplex` | §5b — the `Unbounded` verdict the solver could not produce at all until Version 5 |
| 8 | `./build/tools/solve/solve tests/data/netlib/afiro.mps --method=pdlp` | §5c(a) — the third engine, and `kkt_passes` as its honest cost metric |
| 9 | `tests/unit/iis_test.cpp` → `test_three_row_contradiction` | §5c(b) — the solver names *which* constraints contradict, including a three-row cycle where no pair is infeasible |
| 10 | `ctest --test-dir build-cuda --output-on-failure` (**24/24** as of 2026-09-14) | Close — "and everything above still passes, together, right now" |

Close with: *"Yeh sab ek hi timeline hai — aap khud verify kar sakte hain, humne kuch bhi
hide nahi kiya, na hi koi number banaya hai jo verify na ho sake."*
