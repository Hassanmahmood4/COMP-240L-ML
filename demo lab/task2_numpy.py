"""
demo lab Task 2:
Create a 3x4 array of random integers from 0 to 9 (np.random.seed(0)).
Compute row means, column maximums, transpose, and reshape to 4x3.
"""

import numpy as np


def main():
    np.random.seed(0)

    arr = np.random.randint(0, 10, size=(3, 4))
    print("3x4 array of random integers (0-9):")
    print(arr)
    print(f"Shape: {arr.shape}\n")

    row_means = arr.mean(axis=1)
    print("Row means:", row_means)

    col_maxes = arr.max(axis=0)
    print("Column maximums:", col_maxes)

    transposed = arr.T
    print("\nTranspose (4x3):")
    print(transposed)
    print(f"Transpose shape: {transposed.shape}")

    reshaped = arr.reshape(4, 3)
    print("\nReshaped to 4x3:")
    print(reshaped)
    print(f"Reshape shape: {reshaped.shape}")


if __name__ == "__main__":
    main()
