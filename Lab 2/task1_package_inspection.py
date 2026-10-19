"""
Lab 2 Task 1:
Use python -m pip show on numpy and scikit-learn.
Report the version, location, and packages each one requires.
"""

import subprocess
import sys


def pip_show(package: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def parse_show(output: str) -> dict:
    info = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def main():
    for package in ("numpy", "scikit-learn"):
        print(f"=== pip show {package} ===")
        raw = pip_show(package)
        print(raw)
        info = parse_show(raw)
        print()
        print(f"Summary for {package}:")
        print(f"  Version:  {info.get('Version', 'N/A')}")
        print(f"  Location: {info.get('Location', 'N/A')}")
        print(f"  Requires: {info.get('Requires', '(none)')}")
        print()


if __name__ == "__main__":
    main()
