"""
Lab 2 Task 3:
In the mlenv virtual environment, install a specific older numpy version,
confirm with importlib.metadata, then upgrade it back.
"""

import subprocess
import sys
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
VENV_DIR = LAB_DIR / "mlenv"
PINNED = "1.26.4"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd):
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, text=True)


def numpy_version(python: Path) -> str:
    code = (
        "from importlib.metadata import version; "
        "print(version('numpy'))"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main():
    python = venv_python()
    if not python.exists():
        raise SystemExit(
            f"mlenv not found at {VENV_DIR}. Run task2_mlenv.py first."
        )

    print(f"=== Installing numpy=={PINNED} into mlenv ===")
    run([str(python), "-m", "pip", "install", f"numpy=={PINNED}"])
    pinned_ver = numpy_version(python)
    print(f"importlib.metadata version after pin: {pinned_ver}")
    assert pinned_ver == PINNED, f"Expected {PINNED}, got {pinned_ver}"

    print("\n=== Upgrading numpy back ===")
    run([str(python), "-m", "pip", "install", "--upgrade", "numpy"])
    upgraded = numpy_version(python)
    print(f"importlib.metadata version after upgrade: {upgraded}")
    print("Done.")


if __name__ == "__main__":
    main()
