"""
Lab 3 Task 2 — Reshaping:
Create np.arange(24), reshape to (2, 3, 4), sum along each axis,
and state the shape of each result.
"""

import numpy as np


def main() -> None:
    a = np.arange(24)
    print("Original arange(24):")
    print(a)
    print(f"Shape: {a.shape}\n")

    b = a.reshape(2, 3, 4)
    print("Reshaped to (2, 3, 4):")
    print(b)
    print(f"Shape: {b.shape}\n")

    for axis in (0, 1, 2):
        s = b.sum(axis=axis)
        print(f"Sum along axis={axis}:")
        print(s)
        print(f"  Result shape: {s.shape}\n")

    print("Explanation:")
    print("  axis=0 collapses the first dim  -> shape (3, 4)")
    print("  axis=1 collapses the middle dim -> shape (2, 4)")
    print("  axis=2 collapses the last dim   -> shape (2, 3)")


if __name__ == "__main__":
    main()
