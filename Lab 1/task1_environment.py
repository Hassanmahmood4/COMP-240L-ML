"""
Lab 1 Task 1:
Print the Python version and the version of every library used on this machine.
State whether Anaconda, pip, or Colab was used.
"""

import sys

import matplotlib
import numpy as np
import pandas as pd
import sklearn


def main():
    print("=== Environment ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()

    print("=== Library versions ===")
    print(f"numpy:       {np.__version__}")
    print(f"pandas:      {pd.__version__}")
    print(f"matplotlib:  {matplotlib.__version__}")
    print(f"scikit-learn:{sklearn.__version__}")
    print()

    print("=== Package manager / environment ===")
    print("Used: pip (local virtual environment .venv)")
    print("Not used: Anaconda")
    print("Not used: Google Colab")


if __name__ == "__main__":
    main()
