# Benchmarking

`scripts/benchmark.py` runs the solver over a corpus and writes one CSV.
Pure standard library, Python 3.8+, same on a laptop and on an A100 box.

## Build first

```sh
cmake --preset release && cmake --build build      # CPU
cmake --preset cuda    && cmake --build build-cuda # + GPU engines
```

The script finds the binary itself, preferring a CUDA build **that runs on
the current platform**. Override with `--solver PATH`.

## Run

```sh
python scripts/benchmark.py                          # Netlib, best config
python scripts/benchmark.py --compare                # every engine, A/B
python scripts/benchmark.py --corpus mip --node-limit 2000
python scripts/benchmark.py --all --repeat 3 --out results/full.csv
```

Fetch the LP corpus once with `python scripts/fetch_netlib.py --all`.
MIPLIB instances go in `tests/data/MILP/` by hand (they are not committed).

| Option | Meaning |
|---|---|
| `--corpus lp\|mip\|large\|all` | which built-in corpus (default `lp`) |
| `--instances FILE/DIR ...` | explicit models instead of a corpus |
| `--no-auto-family` | disable automatic LP/MILP classification (enabled by default) |
| `--compare` | every engine, not just the default one |
| `--configs a,b,c` | named configurations to run |
| `--config "name=--flags"` | define your own, repeatable |
| `--repeat N` | N runs per cell, best kept, spread recorded |
| `--timeout S` | per-run wall limit (default 300) |
| `--node-limit N` | MILP node budget, for comparable runs |
| `--limit N` | first N models only — use this to smoke-test |

## What comes out

One row per (instance × configuration). Every run is recorded, **including
timeouts, errors and non-optimal verdicts** — there is no filtering step, so
the CSV is the record and the printed summary is only a convenience.

Each row carries the machine, GPU, core count and git commit, so CSVs from
two machines concatenate into something meaningful. Columns beyond the fixed
lead set are whatever the solver printed, so new statistics appear
automatically.

Key columns: `instance, config, status, objective, best_bound,
solve_time_seconds, wall_seconds, iterations, nodes_explored,
concurrent_winner, timed_out, exit_code`.

`solve_time_seconds` is the solver's own timing; `wall_seconds` includes
process start-up and parsing. Both are present so the difference is visible
rather than assumed.

## Automatic LP/MILP selection

When a model is supplied through `--instances`, the harness detects standard
integrality declarations before selecting a configuration: MPS `INTORG`/`BV`,
CPLEX LP `General`/`Integer`/`Binary` sections, and QPLIB's variable-type
code. A detected discrete model gets the `mip` configuration by default;
continuous models use the normal LP default. The CSV's `family` column shows
the choice. Pass `--no-auto-family` to retain the original `custom` family,
or use `--configs` to choose configurations explicitly.

## GPU

GPU configurations (`ipm`, `pdlp-gpu`) are detected by running the solver,
not by guessing from the build directory. On a CPU-only build they are
recorded as `SKIPPED` with the reason, never silently dropped.

On a datacentre card also try `--config "concurrent-gpu=--method=concurrent
--concurrent-gpu-ipm=1"`. That is off by default because consumer Ampere runs
FP64 at 1/64 rate, which does not hold on A100 and later — see
`docs/ARCHITECTURE-REVIEW.md` §3.5.

## Cross-solver comparison

This script measures **this** solver. To compare against HiGHS, CBC and GLOP,
use `scripts/compare_solvers.py`; correctness against an independent
implementation is `scripts/oracle_check.py`, which runs under `ctest`.
