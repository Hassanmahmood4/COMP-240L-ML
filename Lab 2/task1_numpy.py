"""
Lab Task 1:
Use PIP to install numpy. Verify installation, import it, and create/manipulate arrays
using np.array() and np.sum().
"""

import numpy as np


def main():
    print("NumPy version:", np.__version__)
    print("NumPy imported successfully.\n")

    # Create arrays with np.array()
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[10, 20, 30], [40, 50, 60]])

    print("1D Array:", arr1)
    print("2D Array:\n", arr2)

    # Manipulate arrays
    print("\nArray shape (1D):", arr1.shape)
    print("Array shape (2D):", arr2.shape)
    print("Array dtype:", arr1.dtype)

    # Sum using np.sum()
    total_1d = np.sum(arr1)
    total_2d = np.sum(arr2)
    row_sums = np.sum(arr2, axis=1)
    col_sums = np.sum(arr2, axis=0)

    print("\nSum of 1D array:", total_1d)
    print("Sum of 2D array:", total_2d)
    print("Row-wise sums:", row_sums)
    print("Column-wise sums:", col_sums)

    # Extra manipulation: element-wise operations
    doubled = arr1 * 2
    added = arr1 + 10
    print("\nDoubled array:", doubled)
    print("Array + 10:", added)
    print("Mean of 1D array:", np.mean(arr1))
    print("Max of 2D array:", np.max(arr2))


if __name__ == "__main__":
    main()
