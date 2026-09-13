#!/usr/bin/env python3
"""Side-by-side comparison: sovsolve (ours, GPU) vs several open-source solvers.

A clean terminal table for demo purposes -- one row per instance, one column
per solver, objective + wall time, plus a verdict column that flags any
disagreement beyond tolerance. This is a DEMO/reporting tool, not a test:
`scripts/oracle_check.py` is the actual correctness gate (run via ctest) and
this script reuses its proven MPS-parsing/HiGHS-solving code rather than
duplicating it.

Solvers compared, each a genuinely SEPARATE codebase/algorithm (not the same
engine counted twice):
  - sovsolve  -- this project's own GPU interior-point solver (tools/solve)
  - HiGHS     -- via scipy.optimize.linprog(method="highs"); dev-only oracle,
                 never linked into the solver (see docs/HIGHS-COMPARISON.md)
  - CBC       -- COIN-OR CBC, via the standalone cbc.exe binary PuLP installs
                 (simplex + branch-and-bound, its own codebase)
  - GLOP      -- Google OR-Tools' own LP simplex solver (a third, independent
                 implementation -- neither HiGHS nor CBC's code)

Each solver is a small, independent plug-in function (see the "SOLVERS" list
at the bottom) returning {"objective": float|None, "status": str, "time": float}.
A solver whose library isn't installed is skipped with a note instead of
crashing the whole comparison -- add a new one by writing one function and
appending it to SOLVERS.

Usage:
    python scripts/compare_solvers.py [--solve-exe PATH] [--modeldump-exe PATH]
                                       [--cbc-exe PATH] [instance.mps ...]

With no instance arguments, runs a curated default set of small Netlib LPs
that are known to solve quickly and cleanly (safe for a live demo).
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse oracle_check's proven MPS-dump parsing and HiGHS-solving code rather
# than re-deriving the RANGES/BOUNDS/row-split logic a second time.
import oracle_check  # noqa: E402

DEFAULT_INSTANCES = [
    "tests/data/netlib/afiro.mps",
    "tests/data/netlib/avgas.mps",
    "tests/data/netlib/chip.mps",
]

REL_TOL = 1e-6

_dump_cache = {}  # mps path -> parsed "original" block, one dump per instance


def get_original_block(modeldump_exe, mps_path):
    key = str(mps_path)
    if key not in _dump_cache:
        out = subprocess.run([str(modeldump_exe), str(mps_path)],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            _dump_cache[key] = None
        else:
            blocks = oracle_check.parse_dump(out.stdout)
            _dump_cache[key] = blocks.get("original")
    return _dump_cache[key]


def close(a, b, tol=REL_TOL):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def fmt_obj(v):
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def fmt_time(t):
    return f"{t:.3f}s" if isinstance(t, float) else str(t)


def timed(fn):
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 -- one solver's crash must never
        # take down the whole comparison table during a live demo; report it
        # as that solver's status instead of an uncaught traceback.
        result = {"objective": None, "status": f"ERROR: {exc}"}
    result["time"] = time.perf_counter() - t0
    return result


# --------------------------------------------------------------------------
# Each solver plug-in: (name, availability_check, solve_fn).
# solve_fn(ctx, mps_path) -> {"objective": float|None, "status": str}
# `ctx` carries auto-detected tool paths (solve_exe, modeldump_exe, cbc_exe).
# --------------------------------------------------------------------------

def solve_sovsolve(ctx, mps_path):
    try:
        out = subprocess.run(
            [str(ctx["solve_exe"]), str(mps_path), str(ctx["max_iterations"])],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"objective": None, "status": "TIMEOUT"}
    status, objective = "ERROR", None
    m = re.search(r"^status=(\S+)", out.stdout, re.MULTILINE)
    if m:
        status = m.group(1)
    m = re.search(r"^objective=([-\d.eE+]+)", out.stdout, re.MULTILINE)
    if m:
        objective = float(m.group(1))
    return {"objective": objective, "status": status}


def solve_highs(ctx, mps_path):
    orig = get_original_block(ctx["modeldump_exe"], mps_path)
    if orig is None:
        return {"objective": None, "status": "DUMP_FAILED"}
    if orig.get("discrete", 0):
        return {"objective": None, "status": "MILP (skipped)"}
    got, err = oracle_check.solve_original(orig)
    if got is None:
        return {"objective": None, "status": err or "FAILED"}
    return {"objective": got, "status": "Optimal"}


def solve_cbc(ctx, mps_path):
    try:
        out = subprocess.run(
            [str(ctx["cbc_exe"]), str(mps_path), "solve"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"objective": None, "status": "TIMEOUT"}
    text = out.stdout
    if "Optimal - objective value" in text:
        m = re.search(r"Optimal - objective value\s+([-\d.eE+]+)", text)
        return {"objective": float(m.group(1)) if m else None, "status": "Optimal"}
    if "Infeasible" in text:
        return {"objective": None, "status": "Infeasible"}
    if "Unbounded" in text:
        return {"objective": None, "status": "Unbounded"}
    return {"objective": None, "status": "ERROR"}


def _big(v, threshold=1e19):
    return None if abs(v) >= threshold else v


def solve_glop(ctx, mps_path):
    """Google OR-Tools' own LP simplex solver -- a third, independent
    implementation (neither HiGHS's nor CBC's code). Built on the same
    parsed "original" block oracle_check.py already produces for HiGHS,
    fed through OR-Tools' own MPModelProto/LinearSolver API instead of
    scipy's -- so this exercises a genuinely different solve path, not
    just a relabeled HiGHS call."""
    from ortools.linear_solver import pywraplp

    orig = get_original_block(ctx["modeldump_exe"], mps_path)
    if orig is None:
        return {"objective": None, "status": "DUMP_FAILED"}
    if orig.get("discrete", 0):
        return {"objective": None, "status": "MILP (skipped)"}

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        return {"objective": None, "status": "GLOP_UNAVAILABLE"}

    def _bound(v, inf_default):
        # NOT "`_big(v) or default`": _big(0.0) is a legitimate lower bound
        # (the overwhelming common case for LP variables) but 0.0 is falsy in
        # Python, so `or` would silently replace a real zero bound with
        # +/-infinity -- turning most variables effectively free and changing
        # the feasible region. Must check `is not None` explicitly instead.
        b = _big(v)
        return b if b is not None else inf_default

    n, m = orig["cols"], orig["rows"]
    x = [solver.NumVar(_bound(orig["col_lower"][j], -solver.infinity()),
                       _bound(orig["col_upper"][j], solver.infinity()), f"x{j}")
         for j in range(n)]

    offsets, indices, values = orig["A_offsets"], orig["A_indices"], orig["A_values"]
    row_lower, row_upper = orig["row_lower"], orig["row_upper"]
    for i in range(m):
        lo, hi = _big(row_lower[i]), _big(row_upper[i])
        if lo is None and hi is None:
            continue  # a free row: no constraint at all
        lo_c = lo if lo is not None else -solver.infinity()
        hi_c = hi if hi is not None else solver.infinity()
        ct = solver.Constraint(lo_c, hi_c)
        for k in range(int(offsets[i]), int(offsets[i + 1])):
            ct.SetCoefficient(x[int(indices[k])], values[k])

    objective = solver.Objective()
    for j, cj in enumerate(orig["c"]):
        if cj != 0.0:
            objective.SetCoefficient(x[j], cj)
    if orig["sense"] == "max":
        objective.SetMaximization()
    else:
        objective.SetMinimization()

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        return {"objective": objective.Value(), "status": "Optimal"}
    if status == pywraplp.Solver.INFEASIBLE:
        return {"objective": None, "status": "Infeasible"}
    if status == pywraplp.Solver.UNBOUNDED:
        return {"objective": None, "status": "Unbounded"}
    return {"objective": None, "status": "NOT_SOLVED"}


def _has_scipy():
    try:
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


def _has_ortools():
    try:
        from ortools.linear_solver import pywraplp  # noqa: F401
        return True
    except ImportError:
        return False


# name, column label, availability check, solve function, is-baseline
# ("is-baseline" solvers are what MISMATCH is checked against; sovsolve is
# always included in the comparison set regardless).
SOLVERS = [
    ("sovsolve", "sovsolve (GPU)", lambda ctx: ctx["solve_exe"] is not None, solve_sovsolve),
    ("highs", "HiGHS", lambda ctx: _has_scipy(), solve_highs),
    ("cbc", "CBC", lambda ctx: ctx["cbc_exe"] is not None, solve_cbc),
    ("glop", "GLOP (OR-Tools)", lambda ctx: _has_ortools(), solve_glop),
]


def find_cbc_exe():
    exe = shutil.which("cbc")
    if exe:
        return exe
    try:
        import pulp
        candidate = Path(pulp.PULP_CBC_CMD().path)
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    return None


def _autodetect(*rel_paths):
    for build_dir in ("build-cuda", "build"):
        for rel in rel_paths:
            candidate = REPO_ROOT / build_dir / rel
            if candidate.exists():
                return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="*", default=DEFAULT_INSTANCES,
                    help="MPS files to compare (default: a small curated Netlib set)")
    ap.add_argument("--solve-exe", default=None)
    ap.add_argument("--modeldump-exe", default=None)
    ap.add_argument("--cbc-exe", default=None)
    ap.add_argument("--max-iterations", type=int, default=300)
    args = ap.parse_args()

    ctx = {
        "solve_exe": Path(args.solve_exe) if args.solve_exe else _autodetect(
            "tools/solve/solve", "tools/solve/solve.exe"),
        "modeldump_exe": Path(args.modeldump_exe) if args.modeldump_exe else _autodetect(
            "tools/modeldump/modeldump", "tools/modeldump/modeldump.exe"),
        "cbc_exe": args.cbc_exe or find_cbc_exe(),
        "max_iterations": args.max_iterations,
    }
    if ctx["solve_exe"] and not ctx["solve_exe"].exists():
        ctx["solve_exe"] = None
    if ctx["modeldump_exe"] and not ctx["modeldump_exe"].exists():
        ctx["modeldump_exe"] = None

    if ctx["modeldump_exe"] is None:
        print("modeldump not found -- HiGHS and GLOP columns need it and will be skipped.",
              file=sys.stderr)

    active = [(key, label, fn) for key, label, avail, fn in SOLVERS if avail(ctx)]
    skipped = [label for key, label, avail, fn in SOLVERS if not avail(ctx)]
    if skipped:
        print(f"Skipping (not available): {', '.join(skipped)}", file=sys.stderr)
    if not active:
        print("No solvers available -- nothing to compare.", file=sys.stderr)
        return 2

    col_w = max(15, max(len(label) for _, label, _ in active) + 1)
    header = "instance".ljust(14)
    for _, label, _ in active:
        header += f"{label:>{col_w}} {'time':>8}"
    header += "   verdict"
    print(header)
    print("=" * len(header))

    for inst in args.instances:
        path = Path(inst)
        if not path.is_absolute():
            path = REPO_ROOT / path
        stem = path.stem

        results = {}
        for key, label, fn in active:
            results[key] = timed(lambda fn=fn, path=path: fn(ctx, path))

        objs = [r["objective"] for r in results.values() if r["objective"] is not None]
        if len(objs) >= 2:
            verdict = "MATCH" if all(close(o, objs[0]) for o in objs) else "MISMATCH -- investigate"
        else:
            verdict = "n/a (need 2+ answers)"

        row = f"{stem:<14}"
        for key, label, _ in active:
            r = results[key]
            row += f"{fmt_obj(r['objective']):>{col_w}} {fmt_time(r['time']):>8}"
        row += f"   {verdict}"
        print(row)

        for key, label, _ in active:
            r = results[key]
            if r["objective"] is None:
                print(f"    {label}: {r['status']}")

    print()
    legend = {
        "sovsolve": "this project's own from-scratch GPU interior-point solver.",
        "HiGHS": "independent open-source solver (via SciPy), oracle-only, never linked in.",
        "CBC": "COIN-OR CBC, an independent open-source solver (separate codebase from HiGHS).",
        "GLOP (OR-Tools)": "Google OR-Tools' own LP simplex solver (third independent codebase).",
    }
    for _, label, _ in active:
        if label in legend:
            print(f"{label:<16} = {legend[label]}")


if __name__ == "__main__":
    sys.exit(main())
