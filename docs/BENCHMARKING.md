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
python scripts/benchmark.py --gpu                    # CPU vs GPU (see below)
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
| `--gpu` | CPU/GPU A/B on LP models — **nothing uses the GPU without this** |
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

## GPU: you have to ask

**The solver does not pick CPU or GPU for you, and neither does this script.
A default run is CPU-only.** The GPU engines are reached through their own
`--method` and their own flags, so here they are their own configurations and
run only when named — with `--gpu`, `--compare`, or `--configs`.

```sh
python scripts/benchmark.py --gpu                    # CPU vs GPU A/B on LP
python scripts/benchmark.py --gpu --corpus large     # the big instances
python scripts/benchmark.py --configs pdlp,pdlp-gpu  # one pair only
```

`--gpu` runs `concurrent, pdlp, pdlpx, pdlp-gpu, pdlpx-gpu, pdlpx-gpu-nograph,
pdlp-gpu-spmv, ipm` on every LP instance. The host `pdlp` row is deliberately included: "the GPU took 0.4s" is
not a result, "the GPU took 0.4s where the same engine on the host took 0.9s"
is.

| Config | Solver flags | What runs on the GPU |
|---|---|---|
| `ipm` | `--method=ipm` | the whole interior-point path (CUDA-only engine) |
| `pdlp-gpu` | `--method=pdlp --gpu-resident=1` | the entire PDLP iterate, device-resident |
| `pdlpx-gpu` | `--method=pdlpx --gpu-resident=1` | cuPDLPx's reflected-Halpern scheme, iterate and restart logic on the device, each 40-iteration chunk replayed as one CUDA graph |
| `pdlpx-gpu-nograph` | `... --gpu-graphs=0` | the same kernels launched one by one — pair it with `pdlpx-gpu` to isolate the graph gain |
| `pdlp-gpu-spmv` | `--method=pdlp --gpu-spmv=1 --gpu-spmv-timing=1` | only `K` and `Kᵀ`; kernel and transfer time reported separately |
| `concurrent-gpu` | `--method=concurrent --concurrent-gpu-ipm=1` | adds the GPU interior-point engine to the race |

`concurrent-gpu` is not in `--gpu`'s list and `--concurrent-gpu-ipm` is off by
default, because consumer Ampere runs FP64 at 1/64 rate — that penalty does
not hold on A100 and later, so on a datacentre card it is worth adding:
`--configs concurrent,concurrent-gpu`. See `docs/ARCHITECTURE-REVIEW.md` §3.5.

Rows run on the device print `gpu_graphs=1|0` — whether graphs actually
**engaged**, not whether they were requested. A capture that fails falls back
to direct launches and says why in `gpu_graph_note`, so a CSV row can never
credit graphs with a run that did not use one.

Two things *are* automatic, and both are the safe direction:

* The script prefers a CUDA build (`build-cuda/`) over a CPU build when one
  exists **and executes on this platform** — a repo built both on Windows and
  under WSL has two binaries and only one of them runs.
* Whether the build actually has CUDA is decided by running the solver, not by
  the directory name. Without it, GPU configurations are recorded as `SKIPPED`
  with the reason in the CSV — never silently dropped, never a crash. `--gpu`
  also says so on the console before it starts.

## Cross-solver comparison

This script measures **this** solver. To compare against HiGHS, CBC and GLOP,
use `scripts/compare_solvers.py`; correctness against an independent
implementation is `scripts/oracle_check.py`, which runs under `ctest`.
