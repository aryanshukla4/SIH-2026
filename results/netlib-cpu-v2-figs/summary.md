# Benchmark summary

Machine: i5-12450H laptop, WSL2, on AC — 12 cores — Linux 6.6.87.2-microsoft-standard-WSL2. CPU only. Time limit 300 s. Commit `d7c8419` (tag `bench-netlib-v2`). Wall clock, including reading the file.

## Netlib feasible

| Solver | Correct | SGM10 (s) | Relative |
|---|---:|---:|---:|
| **sovsolve (concurrent)** | 94/94 | 0.190 | 1.00× |
| HiGHS | 94/94 | 0.206 | 1.09× |
| Clp (COIN-OR) | 93/94 | 0.511 | 2.69× |
| SoPlex | 93/94 | 0.646 | 3.40× |

## Netlib infeasible

| Solver | Correct | SGM10 (s) | Relative |
|---|---:|---:|---:|
| **sovsolve (concurrent)** | 29/29 | 0.032 | 1.00× |
| HiGHS | 29/29 | 0.053 | 1.67× |
| SoPlex | 28/29 | 1.288 | 40.79× |
| Clp (COIN-OR) | 28/29 | 1.338 | 42.37× |

## Infeasibility detection

| Solver | Proved infeasible | Wrong verdict | No verdict |
|---|---:|---:|---:|
| sovsolve (concurrent) | 29/29 | 0 | 0 |
| HiGHS | 29/29 | 0 | 0 |
| Clp (COIN-OR) | 28/29 | 1 | 0 |
| SoPlex | 28/29 | 1 | 0 |

## Concurrent race winners

- dual simplex: 63
- HSD: 39
- primal simplex: 10
- pdlpx: 2
