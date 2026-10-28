"""
Lab 3 Task 1 — NumPy practice:
Create a 5x4 array of random integers 10–99 (seed 0).
Find column maxima and the row with the largest sum;
replace values below 30 with 0 using a mask;
compute column-wise z-scores without loops.
"""

import numpy as np


def main() -> None:
    np.random.seed(0)
    arr = np.random.randint(10, 100, size=(5, 4))
    print("5x4 array (seed=0, ints 10–99):")
    print(arr)
    print(f"Shape: {arr.shape}\n")

    col_max = arr.max(axis=0)
    print("Maximum of each column:", col_max)

    row_sums = arr.sum(axis=1)
    best_row = int(np.argmax(row_sums))
    print("Row sums:", row_sums)
    print(f"Row with largest sum: index {best_row} -> {arr[best_row]} (sum={row_sums[best_row]})\n")

    masked = arr.copy()
    masked[masked < 30] = 0
    print("After replacing values < 30 with 0:")
    print(masked)
    print()

    # Column-wise z-scores: (x - mean) / std, no Python loops
    means = arr.mean(axis=0)
    stds = arr.std(axis=0, ddof=0)
    z = (arr - means) / stds
    print("Column means:", means.round(3))
    print("Column stds: ", stds.round(3))
    print("Column-wise z-scores:")
    print(np.round(z, 3))


if __name__ == "__main__":
    main()
