# Tunables register

**Scope, for now:** every tunable value added or changed in the September 2026 interior
point, presolve and PDLPx work. A full-repository scan that brings EVERY tunable here
(simplex, MILP, older IPM options) is still to do.

**Rule:** a paper is the starting point, measurement on our corpus decides -- above all on
the GPU, where the research is still moving. Where a paper's value lost, both numbers are
kept below.

Each entry, in one place: its value, where it lives, where the value comes from, and
the flag that changes it at run time (if any).

**Source** is either a paper and section, or **OURS** with the measurement behind it.
Paper keys:

| key | paper |
|---|---|
| AA95 | Andersen & Andersen, "Presolving in linear programming", Math. Prog. 71 (1995) |
| AA00 | Andersen & Andersen, "The MOSEK interior point optimizer for linear programming" (2000) |
| AG99 | Altman & Gondzio, "Regularized symmetric indefinite systems in IPMs", OMS 11 (1999) |
| AGH  | Achterberg, Bixby, Gu, Rothberg, Weninger, "Presolve reductions in MIP", ZIB 16-44 |
| M92  | Mehrotra, "On the implementation of a primal-dual interior point method", SIAM J. Optim. 2 (1992) |
| W99  | Wright, "Modified Cholesky factorizations in interior-point algorithms for LP", SIAM J. Optim. 9 (1999) |
| PDLP | Applegate et al., "Practical large-scale LP using PDHG", NeurIPS 2021 |
| CPX  | cuPDLPx, arXiv 2507.14051 |
| HPR  | HPR-LP, arXiv 2408.12179 |
| AY96 | Andersen & Ye, "Combining interior-point and pivoting algorithms for linear programming", Management Science 42(12) (1996) |
| BS94 | Bixby & Saltzman, "Recovering an optimal LP basis from an interior point solution", Operations Research Letters 15(4) (1994) |
| GMSW89 | Gill, Murray, Saunders & Wright, "A practical anti-cycling procedure for linearly constrained optimization", Math. Prog. 45 (1989) |

Values in `Options.hpp` are runtime options; the rest are compile-time constants in the
file named.

## LP presolve (`src/solver/LpPresolve.cpp`)

| name | value | source | flag |
|---|---|---|---|
| reductions on/off | on | AA95 section 3 | `--presolve=0/1/2` (2 = this presolve) |
| `kActTol` | 1e-9 | OURS: = `Tolerances::bound_violation`; AA95 gives no tolerances | - |
| `kVerdictTol` | 1e-6 | OURS: no verdict changed on 94 feasible + 29 infeasible Netlib | - |
| `kDominanceTol` | 1e-7 | OURS: AA95 (26) is a strict inequality with no margin | - |
| `kPivotRatio` | 0.01 | AGH section 4.5 (Markowitz-type guard) | - |
| `kParallelTol` | 1e-12 | OURS: AA95 section 3.6 assumes exact arithmetic | - |
| `kMaxRounds` | 200 | OURS cap; AA95 section 5 repeats "until no reductions" | - |

## CPU interior point, HSD (`Options.hpp` `HsdOptions`, `src/solver/HomogeneousSolve.cpp`)

| name | value | source | flag |
|---|---|---|---|
| `direct` | on | factor + CG refinement design is OURS; AA00 section 1.5 factors the normal equations | `--hsd-direct=0/1` |
| `delta_d` | 1.49e-8 | AG99 section 5: r_d = eps^(1/2) | `--hsd-delta-d` |
| `theta_inv_floor` | 1e-12 | pre-existing; AG99's r_p = eps^(3/4) = 1.8e-12 is the same order | - |
| escalation factor | x10 | AG99 section 5 ("multiplied by 10") | - |
| escalation trigger | CG steps per solve > 2 + dense columns | OURS: AG99's "more than one refinement step", CG being the refinement; fixed brandy | - |
| escalation size | default x 10, next iteration only | AG99 section 5; compounding (tried first) pinned fit1p/fit2p at the cap and lost both | - |
| ambiguous-zone check | HSD Optimal with a row violated beyond `primal_feasibility` (1e-8): dual simplex on the feasibility problem decides | OURS: the stop accepts 10x the tolerance, and infeasible CPLEX2 stopped "optimal" at 3.8e-8; now proved Infeasible | - |
| dual-certificate check | dual simplex on the feasibility problem | OURS: a dual certificate means "unbounded OR infeasible" (AA00 1.4.5); Netlib CPLEX1 was reported Unbounded, now Infeasible | - |
| `kMaxDeltaD` | 1e-2 | OURS: AG99 states no cap; matches `IpmOptions::delta_max` | - |
| `regularization_retries` | 6 | OURS count; the x10 per retry is AG99 section 5 | - |
| `kAbsoluteFactor` | 10 | OURS: AA00 section 1.4.5 relaxes by 100 on fast convergence; 10 measured (grow7/15/22 floor at ~3e-8) | - |
| verdict follows termination reason | rule | AA00 section 1.4.5, Theorems 2-3 | - |
| Schur complement, cancellation-free | formula | exact algebra on AA00 (1.28)-(1.29) | - |

## Normal-equations factor (`src/solver/NormalFactor.cpp`, `src/solver/SparseLdl.cpp`)

| name | value | source | flag |
|---|---|---|---|
| sparse part factored, dense columns apart | design | AA00 section 1.5.3 | - |
| dense threshold | > max(20, m/10) nonzeros | OURS: AA00 calls detection "a heuristic ... subject for further research" | - |
| `kMaxDense` | 50 | OURS | - |
| pivot skipping (huge pivot) | mechanism | W99 sections 3 and 6 | - |
| `kPivotTolerance` (and `kHostPivotTolerance`) | 1e-14 x the pivot's own diagonal | OURS: W99's 1e-13 x largest diagonal lost 25fv47, fffff800, 80bau3b as a CG preconditioner | - |
| `kHugePivot` | 1e128 | OURS size of W99's "huge element" | - |
| ordering | approximate minimum degree | from general knowledge of Amestoy-Davis-Duff 1996 (paper not in hand); ordering 10-57x faster than the old minimum degree, fill within 5-10% of CHOLMOD except dfl001 (1.58M vs 1.14M); HSD 93/94 Netlib, 0 wrong | - |
| numeric factorization | left-looking supernodal, dense update blocks | standard method (AA00 section 1.5.2 describes MOSEK's supernodal Cholesky); dfl001 factor 1.09 s -> 0.22 s, pilot87/maros-r7 on par with CHOLMOD | - |
| `kMaxSupernode` | 64 columns | OURS: keeps a block cache-sized | - |
| backend | CHOLMOD if found, else SparseLdl | build option `SOVSOLVE_USE_CHOLMOD` | - |

## GPU interior point (`Options.hpp` `IpmOptions`, `src/solver/gpu/`)

| name | value | source | flag |
|---|---|---|---|
| `direct` | 2 (our SparseLdl) | OURS: 81/94 Netlib and repeatable; cuDSS 79-80/94 and varied run to run | `--ipm-direct=0/1/2` |
| cuDSS deterministic mode | on | OURS: capri/sctap1 flipped between runs without it | - |
| `mehrotra_start` | off | M92 section 7 (bounds adaptation OURS); off on measurement: helped boeing1, lost bnl1 | `--ipm-mehrotra-start` |
| optimality certificate | Optimal needs primal, per-column dual and gap <= 1e-6 on the ORIGINAL model | weak duality with PDLP's reduced-cost projection (NeurIPS 2021, section 2); per-column measure and 1e-6 are OURS: the norm-relative test passed greenbea 1.3e-3 off (per-column residual 2e-4); rejects pilot.we (row violation 1.6e-5), keeps 81/94, 0 wrong | - |
| `primal_regularization_floor` | 1e-8 | kept on measurement: AG99's r_p = 1.8e-12 gave 78/94 against 81/94 and twice the time (it did turn greenbea's wrong optimum into NotConverged) | `--pfloor` |
| `dual_regularization_floor` | 1e-8 | pre-existing; AG99 r_d = 1.5e-8 | `--dfloor` |

## PDLPx (`Options.hpp` `PdlpOptions`, `src/solver/pdlp/Pdlp.cpp`)

| name | value | source | flag |
|---|---|---|---|
| `termination_tolerance` | 1e-8 | CPX / PDLP | `--pdlp-tol` |
| `check_interval` | 40 | PDLP section 3 (cuPDLP.jl uses 64, HPR 150) | `--pdlp-check-interval` |
| `resident_check` | on | CPX section 4 evaluates termination on the GPU | `--pdlp-resident-check` |
| `certificate_check_every` | 10 | OURS: delays a certificate by at most 400 iterations | `--pdlp-certificate-every` |

## Primal simplex anti-degeneracy, EXPAND (`src/solver/simplex/PrimalSimplex.cpp`)

Replaced Bland's rule after 100 zero-length steps. Measured 2026-09-30, i5-12450H,
`--method=primal-simplex`, 60 s limit, all 123 Netlib LPs: 110 -> 122 solved, SGM10
2.79 s -> 0.78 s, no false Infeasible; truss 491,300 iterations unfinished -> 42,874
(6.4 s), cycle, d6cube, fit2p, forplan, scsd6, scsd8, stocfor2, tuff, wood1p, woodw and
KLEIN3 newly solved. Still unsolved: dfl001 (time limit). pilot's optimum is 2.3e-6
off HiGHS's with and without EXPAND (the same vertex).

| name | value | source | flag |
|---|---|---|---|
| two-pass ratio test with minimum step `tau / pivot` | mechanism | GMSW89 section 4.1 | - |
| working tolerance grows by `tau` every iteration | mechanism | GMSW89 section 4.2 | - |
| master tolerance delta_f | `primal_feasibility_tolerance` (1e-7) | OURS: GMSW89 uses eps^(3/8) = 1e-6; ours keeps each verdict's meaning | `--simplex-tol-primal` |
| `kExpandStart`, `kExpandEnd` | 0.5, 0.99 of delta_f | GMSW89 section 4.2 | - |
| `kExpandIterations` (K) | 10,000 | GMSW89 section 4.2 (eps^(-1/4)) | - |
| `kTerminationResets` (R) | 2 | GMSW89 section 4.3 (1, or 2 when badly conditioned) | - |
| nonbasic values kept across refactorizations | on | GMSW89 section 3.3; a reset on every refactorization would be K = 100 | - |
| phase 1 stuck within delta_f: widen delta, go on in phase 2 | rule | OURS: delta restarts at 0.5 delta_f, and a violation the caller's tolerance accepts must never prove infeasibility | - |

## Concurrent race crossover (`Options.hpp` `ConcurrentOptions`, `src/solver/ConcurrentSolve.cpp`, `src/solver/Crossover.cpp`)

Measured 2026-09-30, i5-12450H, best of 3, all 123 Netlib LPs: 123/123 correct off and
on; SGM10 0.175 s off, 0.220 s on. The crossover runs on the 31 models cuPDLPx or HSD
wins and reaches a vertex on 29; median cost 1.24x the race there. The path there, on
those 31 models: crash straight into the simplex, SGM10 1.21 s; + primal push, 1.01 s;
+ dual push, batch LU repair and the agreement guard, 0.76 s (0.57 s off).

| name | value | source | flag |
|---|---|---|---|
| `crossover` | off | OURS: fixes the objective's last digits (-2.7999999914 -> -2.8 on the milp_test LP) at +25% SGM10 overall | `--concurrent-crossover` |
| crash-basis ranking | x/(x+z) indicator | AY96; plain distance from bounds measured worse (SGM10 0.41 s vs 0.22 s off) | - |
| primal and dual push | on | Megiddo 1991; BS94 (Bixby & Saltzman, OR Letters 15(4), 1994) | - |
| `crossover_method` | dual | OURS: measured before EXPAND, when truss needed 688 dual vs >491,000 primal cleanup iterations (the primal stalled in Bland's rule). With EXPAND, truss's primal cleanup takes 33 iterations vs the dual's 257; the 31-model A/B has not been re-run, so the default is unchanged | `--concurrent-crossover-method` |
| `kCrossoverAgreement` | 1e-7 rel. | OURS: 10x the engines' 1e-8 stop; rejects pilot (2.0e-6) and pilot87 (1.6e-6) vertices, which were wrong answers | - |
| `kSnapTolerance` | 1e-9 rel. | OURS: below the engines' own 1e-8 residual | - |
| `kDualSuperbasicTolerance` | 1e-9 | OURS: duals are accurate to ~1e-8, anything smaller is noise | - |
| `kPushPivotTolerance`, `kHarrisTolerance` | 1e-9, 1e-9 | Harris 1973 two-pass ratio test; values OURS | - |
| `crossover_time_factor` | 1.0 | OURS: caps the worst case (pilot87) at twice the race | `--concurrent-crossover-factor` |
| `crossover_min_seconds` | 1.0 | OURS: keeps microsecond races from starving the crossover | `--concurrent-crossover-min` |

Two fixes found on the way, both outside the crossover: the LU repair now replaces every
dependent column of a stall in one pass (dfl001's crash basis: 1418 refactorizations ->
1, crash 3.66 s -> 0.007 s), and `solve_simplex` gives its primal cleanup only the time
the dual left (it had run for up to twice `--time-limit`).
