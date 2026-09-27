# Benchmark summary

Machine: i5-12450H laptop, WSL2, on AC — 12 cores — Linux 6.6.87.2-microsoft-standard-WSL2. CPU only. Time limit 300 s. Commit `8674e19`. Wall clock, including reading the file.

## Netlib feasible

| Solver | Correct | SGM10 (s) | Relative |
|---|---:|---:|---:|
| HiGHS | 94/94 | 0.216 | 1.00× |
| Clp (COIN-OR) | 93/94 | 0.517 | 2.39× |
| SoPlex | 93/94 | 0.660 | 3.06× |
| **sovsolve (concurrent)** | 93/94 | 0.747 | 3.46× |

## Netlib infeasible

| Solver | Correct | SGM10 (s) | Relative |
|---|---:|---:|---:|
| HiGHS | 29/29 | 0.045 | 1.00× |
| **sovsolve (concurrent)** | 29/29 | 0.089 | 1.98× |
| SoPlex | 28/29 | 1.284 | 28.62× |
| Clp (COIN-OR) | 28/29 | 1.345 | 29.97× |

## Infeasibility detection

| Solver | Proved infeasible | Wrong verdict | No verdict |
|---|---:|---:|---:|
| sovsolve (concurrent) | 29/29 | 0 | 0 |
| HiGHS | 29/29 | 0 | 0 |
| Clp (COIN-OR) | 28/29 | 1 | 0 |
| SoPlex | 28/29 | 1 | 0 |

## Concurrent race winners

- dual simplex: 90
- primal simplex: 22
- pdlpx: 6
