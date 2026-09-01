#!/usr/bin/env python3
"""Validate the ingestion layer against an EXTERNAL solver.

Every other test in this repository checks the ingestion layer against itself:
the LP reader against the MPS reader, the readers against the writer, the
canonicalizer against a forward map written from the same specification. Those
catch a great deal, but they share one blind spot -- when two of my own
components hold the same misunderstanding they agree, and the test passes.

This closes that, and it answers two questions no in-tree test can:

  1. PARSER vs GROUND TRUTH.  Solve the parsed model with HiGHS and compare the
     optimum to the value Netlib publishes.  One number that exercises the whole
     read path at once: RANGES, BOUNDS, the negated objective constant, row
     senses, free rows.  A sign error anywhere shows up here.

  2. CANONICALIZER vs A SOLVER.  Solve the canonical model too and require the
     SAME optimum.  The canonicalizer substitutes fixed columns out, drops empty
     rows, negates `>=` rows and permutes equalities to the front.  The property
     tests check that a POINT maps correctly; only a solver can check that the
     OPTIMUM is preserved.

HiGHS arrives through `scipy.optimize.linprog`, which uses it as its backend --
so this is a genuinely independent implementation, not another of my own.

Dev-only.  Nothing here is linked into the solver; SciPy is never a build or
runtime dependency.  The "shall not be built upon any existing solver library"
constraint is about what the product contains, and this is a test oracle that
sits outside it entirely.

Usage:  python scripts/oracle_check.py <modeldump-exe> <netlib-dir>
"""

import subprocess
import sys
from pathlib import Path

# Optimal objective values as published by Netlib (www.netlib.org/lp/data/readme).
# External ground truth: not derived from anything in this repository.
PUBLISHED = {
    "afiro":    -4.6475314286e02,
    "adlittle":  2.2549496316e05,
    "25fv47":    5.5018458883e03,
    "80bau3b":   9.8723216072e05,
    "e226":     -1.8751929066e01,
    "etamacro": -7.5571521774e02,
    "israel":   -8.9664482186e05,
    "shell":     1.2088253460e09,
    "stair":    -2.5126695119e02,
    "greenbea": -7.2462405908e07,
    "standata":  1.2576995000e03,
}

# Netlib prints ten significant figures, but those digits are not all reliable.
# The archived optima were produced by MINOS 5.3 on a VAX in 1988, and for the
# numerically harder instances they are documented as agreeing to far fewer
# digits than they display.
REL_TOL = 1e-6

# Instances where a modern solver legitimately disagrees with the 1988 archive.
# Listed with the measured deviation and the reason, rather than hidden by a
# loosened global tolerance -- a tolerance wide enough to absorb greenbea would
# be wide enough to absorb a real parse error on every other instance.
#
# Each entry is (relative tolerance, why).
KNOWN_ARCHIVE_DRIFT = {
    "greenbea": (
        2e-3,
        "Netlib archive -7.2462405908e7 is MINOS 5.3/VAX (1988) and is "
        "documented as agreeing to only ~3 significant digits. At a 1e-12 "
        "tolerance LP DASA reports -7.25552503916e7; HiGHS here agrees with "
        "that to 6 figures, not with the archive.",
    ),
    "80bau3b": (
        2e-5,
        "Same archive vintage. Deviation is 8.1e-6 relative, and is unchanged "
        "under either reading of the negative-UP bound quirk, so it is not a "
        "bound-convention difference.",
    ),
}

# Instances with no published optimum to check against.
NO_GROUND_TRUTH = {
    "gas11": "not present in the Netlib summary table; HiGHS reports the model "
             "as unbounded, which is plausible given 375 free columns against "
             "all-equality rows, but there is no reference value to confirm it",
}


# Field kinds are declared rather than inferred. Guessing from the text is what
# broke first: `obj_offset 0` looks like an integer while `obj_offset 1e-05`
# does not, so a heuristic silently turned one of them into an empty array.
STRING_FIELDS = {"name", "sense", "status"}
INT_FIELDS = {"rows", "cols", "discrete", "num_equality", "num_range",
              "objective_negated"}
FLOAT_FIELDS = {"obj_constant", "obj_offset"}


def parse_dump(text):
    """Read the two sections modeldump writes into plain dicts."""
    blocks, current, name = {}, None, None
    for line in text.splitlines():
        if line.startswith("BEGIN "):
            name = line.split()[1]
            current = {}
            continue
        if line.startswith("END "):
            blocks[name] = current
            current = None
            continue
        if current is None or not line.strip():
            continue
        parts = line.split()
        key = parts[0]
        if key in STRING_FIELDS:
            current[key] = " ".join(parts[1:])
        elif key in INT_FIELDS:
            current[key] = int(parts[1])
        elif key in FLOAT_FIELDS:
            current[key] = float(parts[1])
        else:
            # "<key> <count> <values...>"
            current[key] = [float(v) for v in parts[2:]]
    return blocks


def to_dense_rows(offsets, indices, values, nrows, ncols):
    import numpy as np
    from scipy.sparse import csr_matrix
    return csr_matrix(
        (np.array(values), np.array(indices, dtype=int),
         np.array(offsets, dtype=int)),
        shape=(nrows, ncols))


def big(v, threshold=1e19):
    """The MPS infinity convention: anything past ~1e20 IS infinity."""
    return None if abs(v) >= threshold else v


def solve_original(b):
    """Solve the parsed model: rl <= Ax <= ru, cl <= x <= cu.

    `linprog` takes separate `A_ub`/`A_eq` blocks rather than two-sided row
    bounds, so the rows are split here. Doing it in the ORACLE rather than
    asking the dumper for it is deliberate: this translation is written against
    the meaning of `row_lower`/`row_upper`, independently of how the
    canonicalizer performs the same split, so the two cannot share a mistake.
    """
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import vstack

    n, m = b["cols"], b["rows"]
    A = to_dense_rows(b["A_offsets"], b["A_indices"], b["A_values"], m, n)

    ub_rows, ub_rhs, eq_rows, eq_rhs = [], [], [], []
    for i in range(m):
        lo, hi = big(b["row_lower"][i]), big(b["row_upper"][i])
        row = A[i]
        if lo is not None and hi is not None and lo == hi:
            eq_rows.append(row)
            eq_rhs.append(hi)
            continue
        if hi is not None:
            ub_rows.append(row)
            ub_rhs.append(hi)
        if lo is not None:
            ub_rows.append(-row)      # a'x >= lo  <=>  -a'x <= -lo
            ub_rhs.append(-lo)
        # both infinite: a free row, no constraint at all

    bounds = [(big(lo), big(hi))
              for lo, hi in zip(b["col_lower"], b["col_upper"])]

    c = np.array(b["c"])
    sign = -1.0 if b["sense"] == "max" else 1.0
    res = linprog(
        sign * c,
        A_ub=vstack(ub_rows) if ub_rows else None,
        b_ub=np.array(ub_rhs) if ub_rows else None,
        A_eq=vstack(eq_rows) if eq_rows else None,
        b_eq=np.array(eq_rhs) if eq_rows else None,
        bounds=bounds, method="highs")
    if not res.success:
        return None, res.message
    # The RAW objective, WITHOUT the objective constant.
    #
    # Netlib's published optima exclude it. e226 is the only instance in this
    # corpus with a nonzero constant (+7.113, from an RHS of -7.113 on the
    # objective row, negated per the MPS convention) and it settles the
    # question: the published -18.751929066 is the raw LP value, and adding the
    # constant gives -11.639, which matches nothing. The constant is returned
    # separately so the caller can report it rather than silently fold it in.
    return sign * res.fun, None


def solve_canonical(b):
    """Solve the canonical model: A_E x = b_E, A_I x <= b_I, l <= x <= u.

    `A_I x + s = b_I` with `s >= 0` is exactly `A_I x <= b_I`, so the slack
    never has to be materialised for the oracle.
    """
    import numpy as np
    from scipy.optimize import linprog

    n, m = b["cols"], b["rows"]
    neq = b["num_equality"]
    A = to_dense_rows(b["A_offsets"], b["A_indices"], b["A_values"], m, n)
    bvec = np.array(b["b"])
    bounds = [(big(lo), big(hi))
              for lo, hi in zip(b["col_lower"], b["col_upper"])]

    A_eq, b_eq = (A[:neq], bvec[:neq]) if neq else (None, None)
    A_ub, b_ub = (A[neq:], bvec[neq:]) if m > neq else (None, None)

    res = linprog(np.array(b["c"]), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return None, res.message
    # Canonical space is always a minimisation; undo the sense flip and add
    # back the constant folded out when fixed columns were substituted.
    obj = res.fun + b["obj_offset"]
    if b["objective_negated"]:
        obj = -obj
    return obj, None


def close(a, b, tol=REL_TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    dumper, data_dir = sys.argv[1], Path(sys.argv[2])

    try:
        import scipy  # noqa: F401
    except ImportError:
        print("SKIP: SciPy not installed; the oracle check cannot run")
        return 0

    files = sorted(data_dir.glob("*.mps"))
    if not files:
        print(f"SKIP: no .mps files under {data_dir}")
        return 0

    print(f"{'instance':<12}{'published':>18}{'parsed':>18}"
          f"{'canonical':>18}   verdict")
    print("-" * 82)

    failures = checked_published = checked_canonical = 0

    for path in files:
        stem = path.stem
        out = subprocess.run([dumper, str(path)], capture_output=True,
                             text=True)
        if out.returncode != 0:
            print(f"{stem:<12}  dump failed: {out.stderr.strip()[:50]}")
            failures += 1
            continue
        blocks = parse_dump(out.stdout)
        orig, canon = blocks.get("original"), blocks.get("canonical")

        # Integer variables make this a MILP; linprog would solve the
        # relaxation, which has a different optimum. Not a failure -- just
        # outside what this oracle can check.
        if orig.get("discrete", 0):
            print(f"{stem:<12}{'(MILP, skipped)':>56}")
            continue

        got, err = solve_original(orig)
        if got is None:
            reason = NO_GROUND_TRUTH.get(stem)
            if reason:
                print(f"{stem:<12}  unsolved, no ground truth: {err}")
            else:
                print(f"{stem:<12}  solver: {err}")
                failures += 1
            continue

        pub = PUBLISHED.get(stem)
        notes = []

        if pub is not None:
            checked_published += 1
            tol, why = KNOWN_ARCHIVE_DRIFT.get(stem, (REL_TOL, None))
            if not close(got, pub, tol):
                notes.append("PARSED != PUBLISHED")
                failures += 1
            elif why is not None:
                notes.append(f"archive drift {abs(got - pub) / abs(pub):.1e}")

        can_str = "-"
        if canon and canon.get("status") == "ok":
            cgot, cerr = solve_canonical(canon)
            if cgot is None:
                notes.append(f"canonical unsolved: {cerr}")
                failures += 1
            else:
                can_str = f"{cgot:.10g}"
                checked_canonical += 1
                if not close(got, cgot):
                    notes.append("CANONICAL != PARSED")
                    failures += 1

        verdict = "ok" if not notes else "; ".join(notes)
        pub_str = f"{pub:.10g}" if pub is not None else "-"
        print(f"{stem:<12}{pub_str:>18}{got:>18.10g}{can_str:>18}   {verdict}")

    print("-" * 82)
    print(f"{checked_published} checked against published optima, "
          f"{checked_canonical} canonical forms checked against the parsed model")
    if failures:
        print(f"FAIL  oracle_check  ({failures} mismatches)")
        return 1
    print("PASS  oracle_check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
