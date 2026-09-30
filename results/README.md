# Results

Every published number traces back to a CSV in this folder. Each CSV has one
row per run, including time-outs and wrong answers — nothing is filtered.

| File | What it is | Machine | Solver commit |
|---|---|---|---|
| [`netlib-cpu-v2.csv`](netlib-cpu-v2.csv) | **Current headline run.** 123 Netlib LPs (94 feasible + 29 infeasible), SovSolve `--method=concurrent` against HiGHS 1.15.1, SoPlex 9.0.0 and Clp. CPU only, 300 s limit, best of 3. | Intel i5-12450H laptop, WSL2 (Linux 6.6), 12 threads | tag `bench-netlib-v2` |
| [`netlib-cpu-v2-figs/`](netlib-cpu-v2-figs/) | Figures and [`summary.md`](netlib-cpu-v2-figs/summary.md) drawn from `netlib-cpu-v2.csv` by `scripts/plot_bench.py` | — | — |
| [`netlib-cpu-v1.csv`](netlib-cpu-v1.csv) | The same protocol on the same machine three days earlier. Kept to show the progress between the two commits. | same laptop | tag `bench-netlib-v1` |
| [`mittelmann-a100-race.csv`](mittelmann-a100-race.csv) | 10 large LPs from Mittelmann's benchmark (0.5 M – 30 M nonzeros), SovSolve's concurrent race, 600 s limit. No other solver was run on this machine, so there are no time comparisons here. | Cloud server with an NVIDIA A100 | `bench-netlib-v2` era |

## How a run is scored

`scripts/plot_bench.py` counts an answer as correct only if:

- **feasible model** — the status is `Optimal` and the objective is within
  1e-6 (relative) of the reference (HiGHS if it solved the model, else SoPlex,
  else Clp);
- **infeasible model** — the status is `Infeasible`. `InfeasibleOrUnbounded`
  is not a certificate, so it counts as no verdict.

A wrong or missing answer is charged the full time limit, as in Mittelmann's
benchmarks and the cuPDLPx paper's SGM10. SGM10 is the shifted geometric mean of
wall time with a 10 s shift; wall time includes reading the file.

## v1 → v2 at a glance

| SovSolve | v1 (`bench-netlib-v1`) | v2 (`bench-netlib-v2`) |
|---|---:|---:|
| Correct verdicts | 122 / 123 | **123 / 123** |
| SGM10, feasible set | 0.747 s (4th of 4) | **0.190 s (1st of 4)** |
| SGM10, infeasible set | 0.089 s | **0.032 s** |
| Median speed-up vs HiGHS, feasible | 3.1× | **4.8×** |
| Faster than HiGHS on | 72 / 93 | **85 / 94** |

The competitors' own times match between the two runs (HiGHS 0.216 s vs
0.206 s), so the change is the solver, not the machine.

## Which commit a run measured

Each run is pinned to a tag: `bench-netlib-v2` and `bench-netlib-v1`. The CSVs'
`git_commit` column still reads `d7c8419` and `8674e19`, the hashes the
benchmark script recorded on the day. A later cleanup of the repository's
history gave those same commits new hashes; the solver source is unchanged, so
check out the tag:

```sh
git checkout bench-netlib-v2
```

## Reproduce

```sh
python3 scripts/bench_suite.py \
    --set feasible=$HOME/bench/netlib_feasible \
    --set infeasible=$HOME/bench/netlib_infeasible \
    --time-limit 300 --repeat 3 --out results/netlib-cpu-v2.csv
python3 scripts/plot_bench.py results/netlib-cpu-v2.csv --out results/netlib-cpu-v2-figs
```

`bench_suite.py` looks for `highs`, `soplex` and `clp` binaries on the machine;
a missing one is skipped with a note. See [`docs/BENCHMARKING.md`](../docs/BENCHMARKING.md).
