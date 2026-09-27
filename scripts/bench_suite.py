#!/usr/bin/env python3
"""Cross-solver benchmark: sovsolve against HiGHS, Clp, SoPlex and SCIP.

`benchmark.py` measures OUR engines against each other. This measures us
against the established open-source solvers, on the same machine, the same
files, the same time limit and the same clock -- and writes one long CSV that
`plot_bench.py` turns into the figures for the README and the deck.

FAIRNESS RULES, because a comparison anyone can pick apart is worth nothing:

  * CPU only. No configuration here touches a GPU.
  * One clock for everyone: wall time of the whole process, measured here,
    including reading the file. Each solver's own reported time is recorded
    too (`solver_seconds`), so parse-vs-solve cost is visible, not assumed.
  * Every solver runs at its DEFAULT settings -- the configuration its
    authors ship -- apart from the shared time limit.
  * Runs are sequential. Two solvers never share the machine.
  * Every run is a row, including crashes and time-outs. Nothing is filtered.
  * Each set carries the verdict it SHOULD get (Optimal / Infeasible), and
    `plot_bench.py` scores objectives against a reference, so a fast wrong
    answer is counted as wrong rather than fast.

Solver lines (a missing binary is skipped with a note, never a crash):

  sovsolve         --method=concurrent: dual simplex, primal simplex, pdlpx
                   (reflected-Halpern PDHG) and HSD raced on separate cores
  sovsolve-dual / -primal / -pdlpx / -hsd    each engine alone (--singles)
  highs            HiGHS default ("choose": dual simplex + presolve on LP)
  highs-ipm        HiGHS interior point + crossover            (--extra)
  highs-pdlp       HiGHS cuPDLP-C on the CPU: the PDLP peer to our pdlpx
  clp              COIN-OR Clp, automatic dual/primal choice
  soplex           ZIB SoPlex (the LP engine inside SCIP)
  scip             SCIP 10 on SoPlex, its default LP solver -- a MIP solver,
                   so it adds presolve and overhead on a pure LP; included
                   because judges ask

USAGE (inside WSL, see docs/BENCHMARKS.md for the full recipe)
    python3 scripts/bench_suite.py \\
        --set feasible=$HOME/bench/netlib_feasible \\
        --set infeasible=$HOME/bench/netlib_infeasible \\
        --time-limit 300 --repeat 3 --singles --out results/netlib.csv
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
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import benchmark  # noqa: E402  -- reuse its environment probe and parser

PARENT = os.path.dirname(REPO)

# What a set's models SHOULD come back as. A set named anything else is
# recorded with no expectation and scored on objective agreement alone.
EXPECTED = {"feasible": "Optimal", "infeasible": "Infeasible"}


def first(pattern, text, cast=str):
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# One adapter per solver: command line, and a parser that reduces its output
# to {status, objective, solver_seconds, iterations}. Status is normalized to
# Optimal / Infeasible / Unbounded / InfeasibleOrUnbounded / TimeLimit /
# Error, so the plots never need to know which solver said what.
# --------------------------------------------------------------------------

def sov_cmd(exe, model, limit, method):
    return [exe, model, "--method=" + method, "--time-limit=%d" % limit]


def sov_parse(text):
    f = benchmark.parse_output(text)
    status = f.get("status", "Error")
    if status in ("MaxIterations", "NotConverged", "NumericalError", "Nonconvex"):
        status = "Error:" + status
    obj = None
    try:
        obj = float(f.get("objective", ""))
        if obj != obj:  # nan on Infeasible/Unbounded
            obj = None
    except ValueError:
        pass
    return {"status": status, "objective": obj,
            "solver_seconds": f.get("solve_time_seconds"),
            "presolve_seconds": f.get("lp_presolve_seconds"),
            "iterations": f.get("iterations"),
            "primal_infeasibility": f.get("primal_infeasibility"),
            "dual_infeasibility": f.get("dual_infeasibility"),
            "complementarity": f.get("complementarity"),
            "relative_gap": f.get("relative_gap"),
            "concurrent_winner": f.get("concurrent_winner", "")}


def highs_cmd(exe, model, limit, solver):
    cmd = [exe, "--time_limit", str(limit)]
    if solver:
        cmd += ["--solver", solver]
    return cmd + [model]


def highs_parse(text):
    raw = first(r"^Model status\s*:\s*(.+?)\s*$", text) or ""
    status = {"Optimal": "Optimal", "Infeasible": "Infeasible",
              "Unbounded": "Unbounded",
              "Primal infeasible or unbounded": "InfeasibleOrUnbounded",
              "Time limit reached": "TimeLimit"}.get(raw, "Error:" + (raw or "none"))
    iters = first(r"^\s*(?:Simplex|IPM|PDLP)\s+iterations:\s*(\d+)", text)
    # "Presolve reductions: rows 7(-20); columns 10(-22); nonzeros 28(-55)"
    reduced = re.search(r"Presolve reductions: rows (\d+)\(-?\d+\); columns (\d+)", text)
    return {"status": status,
            "objective": first(r"^Objective value\s*:\s*(\S+)", text, float)
            if status == "Optimal" else None,
            "solver_seconds": first(r"^HiGHS run time\s*:\s*(\S+)", text),
            "iterations": iters,
            "presolved_rows": reduced.group(1) if reduced else None,
            "presolved_cols": reduced.group(2) if reduced else None,
            "relative_gap": first(r"^P-D objective error\s*:\s*(\S+)", text)}


# Clp's own summary line rounds the objective to 6 digits -- too coarse to
# score against a reference -- so it also writes its solution file, whose
# header carries 8. A file, not /dev/stdout: opening /dev/stdout truncates
# the log when stdout is redirected to a file.
CLP_SOLUTION = os.path.join("/tmp" if os.path.isdir("/tmp") else ".",
                            "bench_suite_clp_%d.sol" % os.getpid())


def clp_cmd(exe, model, limit, _):
    # `-either` is ClpSolve's automatic dual/primal choice with presolve,
    # the setting Clp's own documentation recommends for an unknown LP.
    if os.path.exists(CLP_SOLUTION):
        os.remove(CLP_SOLUTION)
    return [exe, model, "-sec", str(limit), "-either", "-solu", CLP_SOLUTION, "-quit"]


def clp_parse(text):
    header = ""
    try:
        with open(CLP_SOLUTION, encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
    except OSError:
        pass
    # Header: "Optimal - objective value      -464.75314", "Infeasible - ...",
    # "Stopped on iterations or time - objective value ..."; summary line:
    # "✔ Optimal — Obj: -464.753   Iters: 5   Time: 0.001s".
    verdict = (first(r"^(.+?) - objective value", header)
               or first(r"^\S*\s*(\w[\w ]*?) — Obj:", text) or "")
    v = verdict.lower()
    if v.startswith("optimal"):
        status = "Optimal"
    elif "dual infeasible" in v or "unbounded" in v:
        status = "Unbounded"
    elif "infeasible" in v:
        status = "Infeasible"
    elif v.startswith("stopped"):
        status = "TimeLimit"
    else:
        status = "Error:" + (verdict or "none")
    return {"status": status,
            "objective": first(r"objective value\s+(\S+)", header, float)
            if status == "Optimal" else None,
            "solver_seconds": first(r"— Obj:.*Time:\s*([\d.eE+-]+)s", text),
            "iterations": first(r"— Obj:.*Iters:\s*(\d+)", text)}


def soplex_cmd(exe, model, limit, _):
    return [exe, "-t%d" % limit, model]


def soplex_parse(text):
    raw = first(r"^SoPlex status\s*:\s*.*\[(.+?)\]", text) or ""
    status = {"optimal": "Optimal", "infeasible": "Infeasible",
              "unbounded": "Unbounded",
              "time limit reached": "TimeLimit"}.get(raw, "Error:" + (raw or "none"))
    return {"status": status,
            "objective": first(r"^Objective value\s*:\s*(\S+)", text, float)
            if status == "Optimal" else None,
            "solver_seconds": first(r"^Solving time \(sec\)\s*:\s*(\S+)", text),
            "iterations": first(r"^Iterations\s*:\s*(\d+)", text)}


def scip_cmd(exe, model, limit, _):
    return [exe, "-c", "set limits time %d read %s optimize quit" % (limit, model)]


def scip_parse(text):
    raw = first(r"^SCIP Status\s*:\s*.*\[(.+?)\]", text) or ""
    if raw.startswith("optimal"):
        status = "Optimal"
    elif raw == "infeasible":
        status = "Infeasible"
    elif raw == "unbounded":
        status = "Unbounded"
    elif raw.startswith("infeasible or unbounded"):
        status = "InfeasibleOrUnbounded"
    elif raw.startswith("time limit"):
        status = "TimeLimit"
    else:
        status = "Error:" + (raw or "none")
    return {"status": status,
            "objective": first(r"^Primal Bound\s*:\s*(\S+)", text, float)
            if status == "Optimal" else None,
            "solver_seconds": first(r"^Solving Time \(sec\)\s*:\s*(\S+)", text),
            "iterations": first(r"^\s*primal LP\s*:.*?\s(\d+)\s*$", text)}


# name -> (binary key, command builder, extra arg, parser)
SOLVERS = {
    "sovsolve":        ("sovsolve", sov_cmd, "concurrent", sov_parse),
    "sovsolve-dual":   ("sovsolve", sov_cmd, "dual-simplex", sov_parse),
    "sovsolve-primal": ("sovsolve", sov_cmd, "primal-simplex", sov_parse),
    "sovsolve-pdlpx":  ("sovsolve", sov_cmd, "pdlpx", sov_parse),
    "sovsolve-hsd":    ("sovsolve", sov_cmd, "hsd", sov_parse),
    "highs":           ("highs", highs_cmd, None, highs_parse),
    "highs-ipm":       ("highs", highs_cmd, "ipm", highs_parse),
    "highs-pdlp":      ("highs", highs_cmd, "pdlp", highs_parse),
    "clp":             ("clp", clp_cmd, None, clp_parse),
    "soplex":          ("soplex", soplex_cmd, None, soplex_parse),
    "scip":            ("scip", scip_cmd, None, scip_parse),
}
# The three strongest open-source LP solvers (Mittelmann's LP benchmarks).
# SCIP is a MIP solver that calls SoPlex on an LP, and HiGHS PDLP is a peer
# for pdlpx only -- both stay available through --solvers.
DEFAULT_LINEUP = ["sovsolve", "highs", "soplex", "clp"]
SINGLES = ["sovsolve-dual", "sovsolve-primal", "sovsolve-pdlpx", "sovsolve-hsd"]
EXTRA = ["highs-ipm", "highs-pdlp", "scip"]

# Where each binary is looked for when not given explicitly: PATH first, then
# the build trees the recipe in docs/BENCHMARKS.md produces.
HOME = os.path.expanduser("~")
SEARCH = {
    "sovsolve": [os.path.join(HOME, "sov-build", "tools", "solve", "solve"),
                 os.path.join(REPO, "build-wsl", "tools", "solve", "solve"),
                 os.path.join(REPO, "build", "tools", "solve", "solve")],
    # The copy on WSL's own disk first: loaded from D: over the 9P bridge,
    # the binary and libhighs.so cost ~0.1 s per launch, which no other
    # solver here pays and which dominates sub-second Netlib solves.
    "highs": [os.path.join(HOME, "highs-build", "bin", "highs"),
              os.path.join(PARENT, "HiGHS", "build", "bin", "highs")],
    "clp": [os.path.join(HOME, "coin", "dist", "bin", "clp")],
    "soplex": [os.path.join(HOME, "soplex-build", "bin", "soplex"),
               os.path.join(HOME, "zib", "soplex", "build", "bin", "soplex")],
    "scip": [os.path.join(HOME, "scip-build", "bin", "scip"),
             os.path.join(HOME, "zib", "scip", "build", "bin", "scip")],
    "mpsinfo": [os.path.join(HOME, "sov-build", "tools", "mpsinfo", "mpsinfo"),
                os.path.join(REPO, "build", "tools", "mpsinfo", "mpsinfo")],
}

# What each line actually runs, for the `algorithm` column. The race's row
# names the engine that won instead.
ALGORITHM = {
    "sovsolve": "concurrent race", "sovsolve-dual": "dual simplex",
    "sovsolve-primal": "primal simplex", "sovsolve-pdlpx": "PDLP (reflected Halpern)",
    "sovsolve-hsd": "homogeneous self-dual IPM", "highs": "HiGHS choose (dual simplex)",
    "highs-ipm": "interior point + crossover", "highs-pdlp": "PDLP (cuPDLP-C)",
    "clp": "Clp either (dual/primal)", "soplex": "SoPlex simplex",
    "scip": "SCIP (LP via SoPlex)",
}

# Solvers whose Optimal objective defines the reference an answer is scored
# against, in order of trust: the FIRST that solved wins. HiGHS leads because
# it is the repo's oracle (scripts/oracle_check.py) and, on pilot87, it alone
# matched Netlib's published optimum to 10 digits -- a median of three had
# picked Clp's value, 1.1e-6 off. Ours is excluded: a solver must not grade
# itself.
REFERENCE_SOLVERS = ["highs", "soplex", "clp", "highs-ipm", "scip"]
REL_TOL = 1e-6


def model_size(mpsinfo, model):
    """rows, cols, nnz and density (%) of the model as read, via mpsinfo."""
    if not mpsinfo:
        return {}
    try:
        text = subprocess.run([mpsinfo, model], capture_output=True, text=True,
                              timeout=600).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    m = re.search(r"rows (\d+)\s+cols (\d+)\s+nnz (\d+)", text)
    if not m:
        return {}
    rows, cols, nnz = (int(g) for g in m.groups())
    density = 100.0 * nnz / (rows * cols) if rows and cols else 0.0
    return {"rows": rows, "cols": cols, "nnz": nnz, "density_pct": "%.6g" % density}


def score(rows):
    """Reference objective, relative error and a `correct` flag, per model."""
    refs = [float(r["objective"]) for s in REFERENCE_SOLVERS for r in rows
            if r["solver"] == s and r.get("status") == "Optimal"
            and r.get("objective") not in (None, "")]
    ref = refs[0] if refs else None
    for r in rows:
        r["reference_solver"] = next(
            (s for s in REFERENCE_SOLVERS for x in rows if x["solver"] == s
             and x.get("status") == "Optimal" and x.get("objective") not in (None, "")), "")
    for r in rows:
        r["reference_objective"] = "" if ref is None else "%.10e" % ref
        err = None
        if ref is not None and r.get("status") == "Optimal" and r.get("objective") not in (None, ""):
            err = abs(float(r["objective"]) - ref) / max(1.0, abs(ref))
            r["objective_rel_error"] = "%.3e" % err
        expected = r.get("expected")
        if r.get("status") == "Skipped":
            r["correct"] = ""
        elif expected == "Infeasible":
            r["correct"] = int(r.get("status") == "Infeasible")
        elif expected == "Optimal" or ref is not None:
            r["correct"] = int(r.get("status") == "Optimal"
                               and (err is None or err <= REL_TOL))
        else:
            r["correct"] = ""


def locate(key, explicit):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    if key != "sovsolve" and shutil.which(key):
        return shutil.which(key)
    for path in SEARCH[key]:
        for candidate in (path, path + ".exe"):
            if os.path.exists(candidate):
                return candidate
    return None


def version_of(key, exe):
    flag = {"sovsolve": None, "highs": "--version", "clp": None,
            "soplex": "--version", "scip": "-v"}[key]
    if flag is None:
        return ""
    try:
        out = subprocess.run([exe, flag], capture_output=True, text=True, timeout=30)
        line = (out.stdout or out.stderr).strip().splitlines()
        return line[0][:120] if line else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def run(cmd, parse, limit):
    """One process, wall-clocked. Never raises: a failure is a row.

    On Linux the child is reaped with os.wait4, which returns THAT process's
    own resource usage -- so peak memory is the solver's, not the harness's
    and not a running maximum over every earlier child. One floor remains:
    Linux carries the forking process's RSS into the child's maximum, so a
    solver that stays under ~14 MB reads as the harness's ~14 MB. Readings
    above that floor -- every model where memory matters -- are the solver's.
    """
    started = time.perf_counter()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as err:
        return {"status": "Error", "wall_seconds": 0.0, "note": str(err)[:200]}
    chunks = []
    reader = threading.Thread(target=lambda: chunks.append(proc.stdout.read()))
    reader.start()
    killer = threading.Timer(limit + 60, proc.kill)
    killer.start()
    peak_mb = None
    if hasattr(os, "wait4"):
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        peak_mb = usage.ru_maxrss / 1024.0  # Linux reports KiB
    else:
        proc.wait()
    wall = time.perf_counter() - started
    killer.cancel()
    reader.join()
    text = b"".join(c for c in chunks if c).decode("utf-8", errors="replace")
    if wall >= limit + 60:
        return {"status": "TimeLimit", "wall_seconds": wall, "note": "killed at limit+60s",
                "peak_memory_mb": peak_mb and "%.1f" % peak_mb}
    out = parse(text)
    out["wall_seconds"] = wall
    out["exit_code"] = proc.returncode
    if peak_mb is not None:
        out["peak_memory_mb"] = "%.1f" % peak_mb
    if out["status"].startswith("Error"):
        tail = text.strip().splitlines()
        out["note"] = tail[-1][:200] if tail else ""
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", metavar="NAME=DIR", required=True,
                    help="a named corpus; 'feasible'/'infeasible' carry an expected verdict")
    ap.add_argument("--solvers", help="comma list (default: %s)" % ",".join(DEFAULT_LINEUP))
    ap.add_argument("--singles", action="store_true",
                    help="also run each sovsolve engine alone")
    ap.add_argument("--extra", action="store_true", help="also run %s" % ",".join(EXTRA))
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per cell; the fastest is kept (default 1)")
    ap.add_argument("--limit", type=int, help="first N models per set (smoke test)")
    ap.add_argument("--machine", default="", help="free-text label, e.g. 'laptop, on AC'")
    ap.add_argument("--out", help="CSV path (default results/suite-<host>-<time>.csv)")
    for key in SEARCH:
        ap.add_argument("--" + key, help="path to the %s binary" % key)
    args = ap.parse_args()
    # Progress must reach a redirected log (nohup ... > run.log) as it
    # happens, not in 8 KB bursts.
    sys.stdout.reconfigure(line_buffering=True)

    lineup =args.solvers.split(",") if args.solvers else list(DEFAULT_LINEUP)
    if args.singles:
        lineup += [s for s in SINGLES if s not in lineup]
    if args.extra:
        lineup += [s for s in EXTRA if s not in lineup]
    unknown = [s for s in lineup if s not in SOLVERS]
    if unknown:
        sys.exit("unknown solver(s): %s. Known: %s" % (unknown, ", ".join(SOLVERS)))

    binaries, versions = {}, {}
    for key in sorted({SOLVERS[s][0] for s in lineup}):
        binaries[key] = locate(key, getattr(args, key))
        versions[key] = version_of(key, binaries[key]) if binaries[key] else ""
    mpsinfo = locate("mpsinfo", args.mpsinfo)
    if not binaries.get("sovsolve"):
        sys.exit("sovsolve binary not found; build it (docs/BENCHMARKS.md) or pass --sovsolve")

    env = benchmark.environment(binaries["sovsolve"])
    env["git_commit"] = benchmark.git_commit()
    env["machine_label"] = args.machine
    env["time_limit"] = args.time_limit

    print("machine : %s | %s cores | %s" % (env["cpu"] or "?", env["cores"], env["os"]))
    print("commit  : %s" % env["git_commit"])
    for key in sorted(binaries):
        print("%-8s: %s %s" % (key, binaries[key] or "NOT FOUND -- skipped",
                               ("(" + versions[key] + ")") if versions[key] else ""))
    print("mpsinfo : %s" % (mpsinfo or "NOT FOUND -- size columns left empty"))
    print()

    models = []
    for spec in args.set:
        if "=" not in spec:
            sys.exit("--set needs NAME=DIR, got %s" % spec)
        name, directory = spec.split("=", 1)
        found = benchmark.collect([directory], benchmark.MODEL_RE)
        if not found:
            sys.exit("no models in %s" % directory)
        models += [(name, m) for m in found[: args.limit or None]]

    out = args.out or os.path.join(REPO, "results", "suite-%s-%s.csv" % (
        platform.node(), datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # Two runs writing one CSV interleave their rows AND share the CPU, which
    # corrupts both the file and every timing in it. Hold a lock for the run.
    lock = open(out + ".lock", "w")
    try:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        pass  # Windows: no flock; the benchmark is meant to run under Linux
    except OSError:
        sys.exit("another bench_suite run is already writing %s -- wait for it or "
                 "stop it (pkill -f bench_suite.py) first" % out)

    columns = ["set", "instance", "rows", "cols", "nnz", "density_pct", "solver",
               "algorithm", "concurrent_winner", "status", "expected", "correct",
               "objective", "reference_objective", "reference_solver", "objective_rel_error",
               "wall_seconds", "solver_seconds", "presolve_seconds",
               "presolved_rows", "presolved_cols", "iterations",
               "primal_infeasibility", "dual_infeasibility", "complementarity",
               "relative_gap", "peak_memory_mb", "best_of", "time_spread_seconds",
               "exit_code", "note", "solver_version", "size_bytes", "timestamp",
               "host", "os", "cpu", "cores", "git_commit", "machine_label",
               "time_limit"]
    total = len(models) * len(lineup)
    done = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for set_name, model in models:
            inst = os.path.basename(model)
            inst = re.sub(r"\.(mps|lp|qplib)$", "", inst, flags=re.I)
            size = model_size(mpsinfo, model)
            batch = []
            for solver in lineup:
                done += 1
                key, build, extra, parse = SOLVERS[solver]
                row = dict(env, set=set_name, instance=inst, solver=solver,
                           algorithm=ALGORITHM.get(solver, ""),
                           expected=EXPECTED.get(set_name, ""),
                           solver_version=versions.get(key, ""),
                           size_bytes=os.path.getsize(model), best_of=args.repeat,
                           **size)
                if not binaries.get(key):
                    row.update(status="Skipped", note="binary not found")
                else:
                    tries = []
                    for _ in range(args.repeat):
                        tries.append(run(build(binaries[key], model, args.time_limit, extra),
                                         parse, args.time_limit))
                        if tries[-1]["status"] in ("TimeLimit",) or \
                                tries[-1]["status"].startswith("Error"):
                            break  # repeating a failure teaches nothing
                    best = min(tries, key=lambda t: t["wall_seconds"])
                    if len(tries) > 1:
                        walls = [t["wall_seconds"] for t in tries]
                        row["time_spread_seconds"] = "%.6f" % (max(walls) - min(walls))
                    row.update(best)
                    row["wall_seconds"] = "%.6f" % best["wall_seconds"]
                    if row.get("concurrent_winner"):
                        row["algorithm"] = "concurrent race, won by " + row["concurrent_winner"]
                batch.append(row)
                print("[%5d/%5d] %-10s %-18s %-16s %-22s %s" % (
                    done, total, set_name, inst[:18], solver, row["status"][:22],
                    row.get("wall_seconds", "")))
            # One model's rows are written together, once all are in, so each
            # can be scored against the reference the others establish. A
            # killed run still leaves every finished MODEL on disk.
            score(batch)
            writer.writerows(batch)
            fh.flush()
    print("\nwrote %s\nnext: python scripts/plot_bench.py %s" % (out, out))


if __name__ == "__main__":
    main()
