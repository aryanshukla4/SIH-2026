#!/usr/bin/env python3
"""Reproducible benchmark harness: run sovsolve over a corpus, write one CSV.

WHY THIS EXISTS. Every performance number in this repository was produced by
hand, on one machine, and pasted into a document. That is fine for a working
note and useless to anyone checking the claim. This script is the other
thing: point it at a corpus, get a CSV that records what ran, on what
hardware, at which commit, and what came back -- including the runs that
failed, which are the ones a hand-written table tends to lose.

It is a MEASUREMENT tool, not a test. `ctest` is the correctness gate;
nothing here decides whether the solver is right. It only records what it did.

DESIGN NOTES, since a benchmark that cannot be trusted is worse than none:

  * Every run appears in the CSV, including timeouts, crashes and
    non-optimal verdicts. There is no filtering step. A summary is printed
    at the end for convenience, but the CSV is the record.
  * The solver's own reported `solve_time_seconds` is preferred over wall
    time, because wall time includes process start-up and file parsing. Both
    are recorded so the difference is visible rather than assumed.
  * `--repeat N` takes the BEST of N, which is the right statistic for
    "how fast can this go" on a machine that is also doing other things. The
    spread is recorded too, so a noisy run is detectable.
  * Machine, GPU, compiler build and git commit go in every row. Two CSVs
    from two machines concatenate into something meaningful.

PORTABILITY. Pure standard library, Python 3.8+. Runs the same on a laptop
and on an A100 box; GPU configurations are detected and skipped with a
recorded reason rather than failing.

CPU OR GPU? The solver does NOT choose for you, and neither does this script.
Every default run is on the CPU. The GPU engines are separate `--method`s and
separate flags, so they are separate configurations here (`ipm`, `pdlp-gpu`,
`pdlp-gpu-spmv`, `concurrent-gpu`) and only run when you ask -- with `--gpu`,
`--compare` or `--configs`. What IS automatic is the two things you cannot get
wrong by accident: the script prefers a CUDA build when one exists and runs on
this platform, and it records GPU configurations as SKIPPED with a reason on a
build that has no CUDA instead of failing.

USAGE
    python scripts/benchmark.py                     # LP corpus, best config, CPU
    python scripts/benchmark.py --gpu               # CPU vs GPU A/B
    python scripts/benchmark.py --all --repeat 3    # everything, 3 repeats
    python scripts/benchmark.py --compare           # every engine, for A/B
    python scripts/benchmark.py --help              # all options
"""

import argparse
import csv
import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------
#
# A "configuration" is a label plus the flags that produce it. Judges can add
# their own with --config "name=--flag --flag", so this table is a starting
# point and not a closed set.

LP_CONFIGS = {
    "concurrent":     ["--method=concurrent"],
    "dual-simplex":   ["--method=dual-simplex"],
    "primal-simplex": ["--method=primal-simplex"],
    "pdlp":           ["--method=pdlp"],
    # Module 31: the same engine on cuPDLPx's reflected-Halpern scheme.
    "pdlpx":          ["--method=pdlpx"],
    "hsd":            ["--method=hsd"],
    # CUDA builds only; skipped with a reason elsewhere. NOTHING here is
    # automatic: the solver never moves a run to the GPU on its own, so a GPU
    # measurement only happens when one of these names is selected.
    "ipm":            ["--method=ipm"],
    "pdlp-gpu":       ["--method=pdlp", "--gpu-resident=1"],
    "pdlp-gpu-spmv":  ["--method=pdlp", "--gpu-spmv=1", "--gpu-spmv-timing=1"],
    # Module 31 on the device. CUDA graphs are on by default; the -nograph row
    # is the same kernels launched one by one, so the pair isolates the gain.
    "pdlpx-gpu":      ["--method=pdlpx", "--gpu-resident=1"],
    "pdlpx-gpu-nograph": ["--method=pdlpx", "--gpu-resident=1", "--gpu-graphs=0"],
    "concurrent-gpu": ["--method=concurrent", "--concurrent-gpu-ipm=1"],
}

MIP_CONFIGS = {
    "mip":               ["--method=dual-simplex"],
    "mip-no-presolve":   ["--method=dual-simplex", "--mip-presolve=0"],
    "mip-no-cuts":       ["--method=dual-simplex", "--mip-gomory=0", "--mip-cmir=0"],
    "mip-no-conflicts":  ["--method=dual-simplex", "--mip-conflicts=0"],
}

# Which configurations need a CUDA build.
GPU_CONFIGS = {"ipm", "pdlp-gpu", "pdlp-gpu-spmv", "concurrent-gpu", "pdlpx-gpu",
               "pdlpx-gpu-nograph"}

# The solver's best LP setting and best MIP setting. This is what runs when
# no --config/--compare is given: a benchmark should show a tool at its
# intended settings, the same way every published solver comparison does.
DEFAULT_LP = ["concurrent"]
DEFAULT_MIP = ["mip"]

# `--gpu`: the CPU/GPU A/B. Each GPU configuration is paired with the host
# configuration it should be compared against, because "the GPU took 0.4s" is
# not a result -- "the GPU took 0.4s where the same engine on the host took
# 0.9s" is. `concurrent` leads so the row is also comparable to a default run.
GPU_COMPARE_LP = ["concurrent", "pdlp", "pdlpx", "pdlp-gpu", "pdlpx-gpu",
                  "pdlpx-gpu-nograph", "pdlp-gpu-spmv", "ipm"]

# Columns that lead the CSV, in this order. Everything the solver printed is
# appended after them, so new statistics need no change here.
LEAD_COLUMNS = [
    "instance", "family", "config", "status", "objective", "best_bound",
    "solve_time_seconds", "wall_seconds", "timed_out", "exit_code",
    "iterations", "nodes_explored", "repeat_index", "best_of",
    "time_spread_seconds", "skipped_reason",
]

META_COLUMNS = [
    "run_id", "timestamp", "host", "os", "cpu", "cores", "gpu",
    "git_commit", "solver", "size_bytes",
]


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def runnable(path):
    """True when this binary actually executes on THIS platform.

    Not a formality. A repository built on Windows AND under WSL has both
    `build-cuda/tools/solve/solve` (an ELF file) and
    `build/tools/solve/solve.exe`; picking the first by name gives
    "[WinError 193] %1 is not a valid Win32 application" on every single run.
    Cheaper to ask the OS once than to produce a CSV full of that.
    """
    try:
        subprocess.run([path, "--version"], capture_output=True, timeout=60)
        return True
    except OSError:
        return False
    except subprocess.TimeoutExpired:
        return True  # it started, which is all this asks


def find_solver(explicit):
    """The solve binary. A CUDA build is preferred when both run here."""
    if explicit:
        if not os.path.exists(explicit):
            sys.exit("solver not found: %s" % explicit)
        return os.path.abspath(explicit)
    name = "solve.exe" if os.name == "nt" else "solve"
    tried = []
    for build in ["build-cuda", "build"]:
        path = os.path.join(REPO, build, "tools", "solve", name)
        if os.path.exists(path):
            tried.append(path)
            if runnable(path):
                return path
    if tried:
        sys.exit("found %s but it does not run on this platform (%s).\n"
                 "Build for this platform, or pass --solver PATH.\n"
                 "  tried: %s" % (name, platform.system(), ", ".join(tried)))
    sys.exit(
        "no solve binary found. Build first:\n"
        "    cmake --preset release && cmake --build build\n"
        "or pass --solver PATH")


def has_cuda(solver):
    """True when this build has the GPU engines.

    Asked by running the thing rather than by inspecting the path: a build
    directory called 'build-cuda' proves nothing, and the solver already
    reports the absence in plain words.
    """
    probe = os.path.join(REPO, "tests", "data", "netlib", "afiro.mps")
    if not os.path.exists(probe):
        return False
    try:
        out = subprocess.run([solver, probe, "--method=ipm"],
                             capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "no CUDA" not in (out.stdout + out.stderr)


def gpu_name():
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else ""
    except (subprocess.TimeoutExpired, OSError, IndexError):
        return ""


def git_commit():
    try:
        out = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return ""


def cpu_name():
    name = platform.processor() or ""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return name


def environment(solver):
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "os": "%s %s" % (platform.system(), platform.release()),
        "cpu": cpu_name(),
        "cores": os.cpu_count() or 0,
        "gpu": gpu_name(),
        "git_commit": git_commit(),
        "solver": solver,
    }


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

def collect(paths, pattern):
    """Expand files and directories into a sorted list of model files."""
    found = []
    for path in paths:
        if os.path.isfile(path):
            found.append(os.path.abspath(path))
        elif os.path.isdir(path):
            for entry in sorted(os.listdir(path)):
                if pattern.search(entry):
                    found.append(os.path.abspath(os.path.join(path, entry)))
    return found


MODEL_RE = re.compile(r"\.(mps|lp|qplib)$", re.IGNORECASE)
MPS_DISCRETE_RE = re.compile(r"(?:INTORG|(?:^|\s)BV(?:\s|$))", re.IGNORECASE)
LP_DISCRETE_SECTION_RE = re.compile(
    r"^\s*(?:general(?:s)?|integer(?:s)?|binary|binaries)\s*$",
    re.IGNORECASE)


def has_discrete_variables(path):
    """Best-effort, format-aware classification before choosing a configuration.

    The solver remains the authority on parsing the model.  This deliberately
    recognizes only the standard declarations of integrality, so a failed
    inspection is safe: it selects the LP configuration rather than claiming
    an LP is an MILP.  `--corpus mip` is still an explicit MIP override.
    """
    suffix = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            if suffix == ".mps":
                for line in fh:
                    # An asterisk in column one starts an MPS comment; Netlib
                    # comments often contain the word "integer".
                    if not line.lstrip().startswith("*") and MPS_DISCRETE_RE.search(line):
                        return True
                return False
            if suffix == ".lp":
                # CPLEX LP's discrete declarations are section headers.
                return any(LP_DISCRETE_SECTION_RE.match(line) for line in fh)
            if suffix == ".qplib":
                # QPLIB's first data record is its three-letter type. Its
                # second character is C (continuous), B/I (all binary/integer),
                # or M (per-variable mixed types).
                for line in fh:
                    text = line.strip()
                    if text and not text.startswith(("!", "#")):
                        code = text.split()[0].upper()
                        return len(code) >= 2 and code[1] != "C"
    except OSError:
        pass
    return False


def family_for_model(path, fallback, auto_family):
    """Return MIP for a detected discrete model, otherwise its fallback family."""
    if fallback == "mip" or not auto_family:
        return fallback
    return "mip" if has_discrete_variables(path) else fallback


def default_corpora(which):
    """(family, directory) pairs that exist in this checkout."""
    candidates = []
    if which in ("lp", "all"):
        candidates.append(("lp", os.path.join(REPO, "tests", "data", "netlib")))
    if which in ("mip", "all"):
        # The folder is MILP/ on disk; a case-sensitive checkout may have milp/.
        for name in ("MILP", "milp"):
            candidates.append(("mip", os.path.join(REPO, "tests", "data", name)))
    if which in ("large", "all"):
        candidates.append(("large", os.path.join(REPO, "tests", "data")))
    return [(fam, d) for fam, d in candidates if os.path.isdir(d)]


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

KEY_VALUE_RE = re.compile(r"^([a-z_][a-z0-9_]*)=(.*)$")


def parse_output(text):
    """Every `key=value` line the solver printed, plus the race summary."""
    fields = {}
    winner = None
    engines = []
    for line in text.splitlines():
        line = line.strip()
        match = KEY_VALUE_RE.match(line)
        if match:
            fields[match.group(1)] = match.group(2)
            continue
        if "<-- winner" in line:
            winner = line.split()[0]
        # "  dual-simplex     Optimal   0.336s  iters=4416"
        parts = line.split()
        if len(parts) >= 3 and parts[2].endswith("s") and "iters=" in line:
            engines.append(parts[0])
    if winner:
        fields["concurrent_winner"] = winner
    if engines:
        fields["concurrent_engines"] = "|".join(engines)
    return fields


def run_once(solver, model, flags, timeout, extra):
    """One invocation. Never raises; a failure is data."""
    cmd = [solver, model] + flags + extra
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        wall = time.time() - started
        fields = parse_output(proc.stdout + "\n" + proc.stderr)
        fields["exit_code"] = proc.returncode
        fields["wall_seconds"] = "%.6f" % wall
        fields["timed_out"] = 0
        if proc.returncode != 0 and "status" not in fields:
            fields["status"] = "ERROR"
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            fields["skipped_reason"] = tail[-1][:200] if tail else "nonzero exit"
        return fields
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "timed_out": 1, "exit_code": "",
                "wall_seconds": "%.6f" % (time.time() - started)}
    except OSError as err:
        return {"status": "ERROR", "timed_out": 0, "exit_code": "",
                "skipped_reason": str(err)[:200]}


def seconds_of(fields):
    """The solver's own timing, falling back to wall time."""
    for key in ("solve_time_seconds", "wall_seconds"):
        try:
            return float(fields.get(key, ""))
        except (TypeError, ValueError):
            continue
    return None


def benchmark(args):
    solver = find_solver(args.solver)
    cuda = has_cuda(solver)
    env = environment(solver)

    print("solver   : %s" % solver)
    print("commit   : %s" % (env["git_commit"] or "unknown"))
    print("machine  : %s, %s cores" % (env["cpu"] or "unknown cpu", env["cores"]))
    print("gpu      : %s" % (env["gpu"] or "none detected"))
    print("cuda     : %s" % ("yes" if cuda else "no (GPU configs skipped)"))
    if args.gpu and not cuda:
        print()
        print("  --gpu was asked for but this build has no CUDA engines. The GPU")
        print("  rows will say SKIPPED. Build with: cmake --preset cuda &&")
        print("  cmake --build build-cuda")
    print()

    # --- which models -----------------------------------------------------
    models = []
    if args.instances:
        models = [(family_for_model(m, "custom", args.auto_family), m)
                  for m in collect(args.instances, MODEL_RE)]
    else:
        for family, directory in default_corpora(args.corpus):
            for path in collect([directory], MODEL_RE):
                models.append((family_for_model(path, family, args.auto_family), path))
    if args.limit:
        models = models[: args.limit]
    if not models:
        sys.exit("no model files found. Fetch a corpus first:\n"
                 "    python scripts/fetch_netlib.py --all")

    # --- which configurations --------------------------------------------
    table = dict(LP_CONFIGS)
    table.update(MIP_CONFIGS)
    for spec in args.config or []:
        if "=" not in spec:
            sys.exit("--config needs name=flags, got: %s" % spec)
        name, flags = spec.split("=", 1)
        table[name.strip()] = flags.split()

    def configs_for(family):
        if args.configs:
            return [c.strip() for c in args.configs.split(",") if c.strip()]
        if args.gpu and family != "mip":
            return list(GPU_COMPARE_LP)
        if args.compare:
            return list(MIP_CONFIGS) if family == "mip" else list(LP_CONFIGS)
        return DEFAULT_MIP if family == "mip" else DEFAULT_LP

    extra = list(args.solver_flag or [])
    if args.node_limit:
        extra.append("--mip-node-limit=%d" % args.node_limit)
    if args.time_limit:
        extra.append("--time-limit=%d" % args.time_limit)

    total = sum(len(configs_for(f)) for f, _ in models)
    print("%d models x configurations = %d runs, %d repeat(s) each"
          % (len(models), total, args.repeat))
    print()

    rows = []
    run_id = 0
    done = 0
    for family, model in models:
        name = os.path.splitext(os.path.basename(model))[0]
        try:
            size = os.path.getsize(model)
        except OSError:
            size = 0
        for config in configs_for(family):
            done += 1
            if config not in table:
                sys.exit("unknown config '%s'. Known: %s"
                         % (config, ", ".join(sorted(table))))
            run_id += 1
            base = dict(env)
            base.update({"run_id": run_id, "instance": name, "family": family,
                         "config": config, "size_bytes": size,
                         "best_of": args.repeat})

            if config in GPU_CONFIGS and not cuda:
                base.update({"status": "SKIPPED", "timed_out": 0,
                             "skipped_reason": "build has no CUDA"})
                rows.append(base)
                print("[%4d/%4d] %-22s %-16s SKIPPED (no CUDA)"
                      % (done, total, name, config))
                continue

            attempts = []
            for index in range(args.repeat):
                fields = run_once(solver, model, table[config], args.timeout, extra)
                fields["repeat_index"] = index
                attempts.append(fields)
                if fields.get("status") in ("TIMEOUT", "ERROR"):
                    break  # repeating a failure teaches nothing

            timings = [t for t in (seconds_of(a) for a in attempts) if t is not None]
            best = attempts[0]
            if timings:
                best = min(attempts, key=lambda a: seconds_of(a) or float("inf"))
                if len(timings) > 1:
                    best["time_spread_seconds"] = "%.6f" % (max(timings) - min(timings))
            base.update(best)
            rows.append(base)

            shown = seconds_of(best)
            print("[%4d/%4d] %-22s %-16s %-14s %s"
                  % (done, total, name, config, base.get("status", "?"),
                     ("%.4fs" % shown) if shown is not None else ""))

    write_csv(rows, args.out)
    print("\nwrote %d rows -> %s" % (len(rows), args.out))
    summarize(rows)
    return rows


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_csv(rows, path):
    seen = set()
    for row in rows:
        seen.update(row)
    ordered = [c for c in META_COLUMNS if c in seen]
    ordered += [c for c in LEAD_COLUMNS if c in seen and c not in ordered]
    ordered += sorted(c for c in seen if c not in ordered)

    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows):
    """A short readable digest. The CSV remains the record."""
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)

    by_config = {}
    for row in rows:
        by_config.setdefault(row.get("config", "?"), []).append(row)

    print("\n%-18s %7s %8s %8s %9s %12s" %
          ("config", "runs", "optimal", "failed", "skipped", "total_s"))
    for config in sorted(by_config):
        group = by_config[config]
        optimal = sum(1 for r in group if r.get("status") == "Optimal")
        skipped = sum(1 for r in group if r.get("status") == "SKIPPED")
        failed = sum(1 for r in group
                     if r.get("status") in ("ERROR", "TIMEOUT", "NumericalError"))
        total = sum(seconds_of(r) or 0.0 for r in group
                    if r.get("status") not in ("SKIPPED",))
        print("%-18s %7d %8d %8d %9d %12.3f"
              % (config, len(group), optimal, failed, skipped, total))

    # Per-instance best config, and how the default compares to it. This is
    # the number that says whether picking one engine up front would have
    # been good enough -- which is the entire argument for --method=concurrent.
    per_instance = {}
    for row in rows:
        if row.get("status") != "Optimal":
            continue
        seconds = seconds_of(row)
        if seconds is None:
            continue
        key = row.get("instance")
        current = per_instance.get(key)
        if current is None or seconds < current[1]:
            per_instance[key] = (row.get("config"), seconds)

    if len(by_config) > 1 and per_instance:
        print("\nbest configuration per instance:")
        tally = {}
        for config, _ in per_instance.values():
            tally[config] = tally.get(config, 0) + 1
        for config in sorted(tally, key=lambda c: -tally[c]):
            print("  %-18s wins on %d instance(s)" % (config, tally[config]))
        print("\n  No single winner means a fixed engine choice loses on the rest,")
        print("  which is what --method=concurrent exists to avoid.")

    unresolved = [r for r in rows
                  if r.get("status") not in ("Optimal", "Infeasible", "Unbounded",
                                             "SKIPPED", None)]
    if unresolved:
        print("\n%d run(s) did not reach a verdict:" % len(unresolved))
        for row in unresolved[:15]:
            print("  %-22s %-16s %s %s"
                  % (row.get("instance"), row.get("config"), row.get("status"),
                     row.get("skipped_reason", "")))
        if len(unresolved) > 15:
            print("  ... %d more, see the CSV" % (len(unresolved) - 15))


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sovsolve over a corpus and write a CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/benchmark.py
      LP corpus (Netlib), best configuration, one CSV.

  python scripts/benchmark.py --compare
      Every engine on every LP instance -- the A/B that shows no single
      engine wins everywhere.

  python scripts/benchmark.py --gpu
      CPU vs GPU on every LP instance. Nothing runs on the GPU without this
      (or --compare / --configs): the default run is CPU-only.

  python scripts/benchmark.py --corpus mip --node-limit 2000
      MIPLIB at a fixed node budget, so runs are comparable.

  python scripts/benchmark.py --all --repeat 3 --out results/full.csv
      Everything, best of 3, for a reporting run.

  python scripts/benchmark.py --instances my-models
      Automatically uses MIP settings for MPS/LP/QPLIB files that declare
      integer or binary variables. Pass --no-auto-family to disable this.
""")
    parser.add_argument("--solver", help="path to the solve binary (auto-detected)")
    parser.add_argument("--corpus", choices=["lp", "mip", "large", "all"],
                        default="lp", help="which built-in corpus (default: lp)")
    parser.add_argument("--all", dest="corpus", action="store_const", const="all",
                        help="shorthand for --corpus all")
    parser.add_argument("--instances", nargs="+",
                        help="explicit model files or directories")
    parser.add_argument("--no-auto-family", dest="auto_family", action="store_false",
                        default=True,
                        help="do not detect discrete models; retain the corpus/custom family")
    parser.add_argument("--configs",
                        help="comma-separated configuration names to run")
    parser.add_argument("--compare", action="store_true",
                        help="run every engine, not just the default one")
    parser.add_argument("--gpu", action="store_true",
                        help="CPU/GPU A/B on LP models; needs a CUDA build")
    parser.add_argument("--config", action="append", metavar="NAME=FLAGS",
                        help="define a configuration, repeatable")
    parser.add_argument("--solver-flag", action="append", metavar="FLAG",
                        help="extra flag passed to every run, repeatable")
    parser.add_argument("--repeat", type=int, default=1,
                        help="runs per cell, best is kept (default: 1)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="per-run wall limit in seconds (default: 300)")
    parser.add_argument("--time-limit", type=int,
                        help="solver's own time limit, seconds")
    parser.add_argument("--node-limit", type=int,
                        help="MILP node limit, for comparable MIP runs")
    parser.add_argument("--limit", type=int,
                        help="use only the first N models (smoke test)")
    parser.add_argument("--out", default=None, help="CSV path")
    args = parser.parse_args()

    if args.repeat < 1:
        sys.exit("--repeat must be at least 1")
    if args.out is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out = "benchmark-%s-%s.csv" % (platform.node(), stamp)

    benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
