#!/usr/bin/env python3
"""Fetch Netlib LP instances into tests/data/netlib/.

Two instances (afiro, adlittle) are committed so the corpus test runs with no
network. This script pulls the wider set for benchmarking and for the broader
corpus sweep.

Note on the source: the canonical Netlib distribution ships in a compressed
format that needs the `emps` utility to expand. The HiGHS check corpus mirrors
the same instances as plain MPS, which avoids carrying an expander here. The
data is identical -- the corpus test verifies dimensions against the published
counts either way.

Usage:
    python scripts/fetch_netlib.py            # the default set
    python scripts/fetch_netlib.py --all      # everything listed below
    python scripts/fetch_netlib.py 25fv47 pilot87
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/"

# Verified present at the mirror as of 2026-09. Small ones first: the tiny
# instances are what make a failure debuggable, the large ones are what make
# parse throughput measurable.
DEFAULT = ["afiro", "adlittle", "chip", "avgas", "flugpl", "bell5", "egout",
           "gt2", "rgn"]

# Everything the mirror carries, including several MILP instances -- those
# exercise the MARKER INTORG/INTEND path that pure-LP corpora never reach.
EXTRA = ["25fv47", "80bau3b", "adlittle", "afiro", "avgas", "bell5", "chip",
         "e226", "egout", "etamacro", "flugpl", "gas11", "greenbea", "gt2",
         "israel", "rgn", "shell", "stair", "standata"]


def fetch(name: str, dest: Path) -> tuple[bool, str]:
    target = dest / f"{name}.mps"
    if target.exists():
        return True, f"have  {name} ({target.stat().st_size:,} bytes)"
    try:
        with urllib.request.urlopen(BASE + f"{name}.mps", timeout=30) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        return False, f"MISS  {name} (HTTP {e.code})"
    except Exception as e:  # noqa: BLE001 - report and continue the sweep
        return False, f"FAIL  {name} ({type(e).__name__})"
    target.write_bytes(data)
    return True, f"got   {name} ({len(data):,} bytes)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="instance names (without .mps)")
    ap.add_argument("--all", action="store_true", help="fetch the full list")
    args = ap.parse_args()

    if args.names:
        wanted = args.names
    elif args.all:
        wanted = sorted(set(EXTRA))
    else:
        wanted = DEFAULT

    dest = Path(__file__).resolve().parent.parent / "tests" / "data" / "netlib"
    dest.mkdir(parents=True, exist_ok=True)

    ok = 0
    for name in wanted:
        success, message = fetch(name, dest)
        ok += int(success)
        print(f"  {message}")

    print(f"\n{ok}/{len(wanted)} available in {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
