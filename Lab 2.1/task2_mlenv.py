"""
Lab 2.1 Task 2:
Create a virtual environment named mlenv, activate it (via its python),
install pandas and matplotlib only, and run pip list.
Explain why more than two packages appear.
"""

import subprocess
import sys
import venv
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
VENV_DIR = LAB_DIR / "mlenv"


def run(cmd, **kwargs):
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def main():
    if VENV_DIR.exists():
        print(f"Using existing venv: {VENV_DIR}")
    else:
        print(f"Creating virtual environment: {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)

    if sys.platform == "win32":
        python = VENV_DIR / "Scripts" / "python.exe"
    else:
        python = VENV_DIR / "bin" / "python"

    print("\n=== Installing pandas and matplotlib only ===")
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python), "-m", "pip", "install", "pandas", "matplotlib"])

    print("\n=== pip list ===")
    result = run([str(python), "-m", "pip", "list"], capture_output=True)
    print(result.stdout)

    # Count packages (skip header lines: Package / Version / ----)
    lines = [
        line
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("Package") and not line.startswith("-")
    ]
    print(f"Number of packages listed: {len(lines)}")
    print()
    print("=== Why more than the two installed? ===")
    print(
        "pip list shows pandas and matplotlib PLUS their dependencies"
    )
    print(
        "(e.g. numpy, python-dateutil, pillow, contourpy, cycler, fonttools,"
    )
    print(
        "kiwisolver, packaging, pyparsing, six) and also pip/setuptools that"
    )
    print("ship with / are needed by the virtual environment itself.")


if __name__ == "__main__":
    main()
