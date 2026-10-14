"""
Improved check_requirements():
- ignores blank lines and comments
- supports name==version and name>=version style pins
- prints a clear summary line
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REQ_PATH = Path(__file__).resolve().parent / "requirements.txt"
REQ_PATTERN = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9_.-]+)\s*(?P<op>>=|>|==|<=|<|!=)?\s*(?P<ver>[^\s#]+)?\s*(?:#.*)?$"
)


def _version_tuple(v: str) -> tuple:
    parts = []
    for piece in re.split(r"[.+-]", v):
        if piece.isdigit():
            parts.append(int(piece))
        elif piece:
            parts.append(piece)
    return tuple(parts)


def _satisfies(installed: str, op: str | None, required: str | None) -> bool:
    if not op or not required:
        return True
    a, b = _version_tuple(installed), _version_tuple(required)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ops = {
        ">=": a >= b,
        ">": a > b,
        "==": a == b,
        "<=": a <= b,
        "<": a < b,
        "!=": a != b,
    }
    return ops.get(op, False)


def check_requirements(path: str | Path = REQ_PATH) -> None:
    path = Path(path)
    total = 0
    satisfied = 0

    print(f"Checking requirements in: {path}\n")

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        match = REQ_PATTERN.match(line)
        if not match:
            print(f"  SKIP  (unrecognised line): {raw}")
            continue

        name = match.group("name")
        op = match.group("op")
        ver = match.group("ver")
        total += 1
        req_label = f"{name}{op or ''}{ver or ''}"

        try:
            installed = version(name)
        except PackageNotFoundError:
            print(f"  FAIL  {req_label}  (not installed)")
            continue

        if _satisfies(installed, op, ver):
            satisfied += 1
            print(f"  OK    {req_label}  (installed {installed})")
        else:
            print(f"  FAIL  {req_label}  (installed {installed})")

    print()
    print(f"{satisfied} of {total} requirements satisfied")


if __name__ == "__main__":
    check_requirements()
