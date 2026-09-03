#!/usr/bin/env python3
"""Enforce the module dependency graph.

    core     -> (nothing)
    model    -> core
    analysis -> core
    io       -> core, model

The handoff docs state that the modules are independent, but nothing checked
it, and an architecture rule that is not mechanically checked stops being true
within a few weeks.

CMake alone cannot enforce this here: every module target exposes the same
``include/`` root, so any header can reach any other. This script closes that
gap by scanning the actual ``#include`` edges. It runs as a ctest case, so a
layering violation fails the build the same way a compile error does.

Exit status: 0 clean, 1 violations found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# module -> modules it is allowed to include from (plus itself)
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "model": {"core"},
    "analysis": {"core"},
    "solver": {"core", "model", "analysis"},
    "io": {"core", "model"},
}

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]sovsolve/([a-z]+)/', re.MULTILINE)


def module_of(path: Path, roots: list[Path]) -> str | None:
    """Return the module a file belongs to, or None if it is outside them."""
    for root in roots:
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) >= 2 and rel.parts[0] in ALLOWED:
            return rel.parts[0]
    return None


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    roots = [repo / "include" / "sovsolve", repo / "src"]

    violations: list[str] = []
    checked = 0

    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in {".hpp", ".cpp", ".h", ".cc"}:
                continue
            owner = module_of(path, roots)
            if owner is None:
                continue
            checked += 1

            text = path.read_text(encoding="utf-8", errors="replace")
            permitted = ALLOWED[owner] | {owner}
            for target in INCLUDE_RE.findall(text):
                if target not in ALLOWED:
                    continue  # not a module directory; ignore
                if target not in permitted:
                    violations.append(
                        f"{path.relative_to(repo)}: "
                        f"'{owner}' must not include from '{target}' "
                        f"(allowed: {sorted(permitted)})"
                    )

    if violations:
        print(f"FAIL  layering  ({len(violations)} violations "
              f"in {checked} files)")
        for v in violations:
            print(f"  {v}")
        return 1

    print(f"PASS  layering  ({checked} files, "
          f"{len(ALLOWED)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
