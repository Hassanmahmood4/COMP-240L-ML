"""
Lab 2 Task 5:
State the import name for scikit-learn, opencv-python, Pillow and PyYAML,
then verify with importlib.util.find_spec (install packages if needed).
"""

import importlib.util
import subprocess
import sys


# Pip / distribution name  ->  import name
MAPPING = {
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "PyYAML": "yaml",
}


def ensure_installed(pip_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", pip_name],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Installing {pip_name}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name],
            check=True,
        )


def main():
    print("=== Stated import names (without running imports yet) ===")
    for pip_name, import_name in MAPPING.items():
        print(f"  {pip_name:15s} -> import {import_name}")

    print("\n=== Verifying with importlib.util.find_spec ===")
    for pip_name, import_name in MAPPING.items():
        ensure_installed(pip_name)
        spec = importlib.util.find_spec(import_name)
        found = spec is not None
        origin = getattr(spec, "origin", None) if spec else None
        print(f"  {pip_name:15s} import '{import_name}': found={found}")
        if origin:
            print(f"    origin: {origin}")


if __name__ == "__main__":
    main()
