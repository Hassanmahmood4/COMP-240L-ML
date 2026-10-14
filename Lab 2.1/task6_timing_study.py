"""
Lab 2.1 Task 6:
Compare NumPy vs pure-Python loop for array sizes 10^3 ... 10^7.
Tabulate times and speed-up; discuss how speed-up behaves.
"""

import time

import numpy as np


SIZES = [10**3, 10**4, 10**5, 10**6, 10**7]


def loop_sum(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += float(i) * float(i)
    return total


def numpy_sum(n: int) -> float:
    arr = np.arange(n, dtype=np.float64)
    return float(np.sum(arr * arr))


def time_once(fn, n: int) -> float:
    start = time.perf_counter()
    fn(n)
    return time.perf_counter() - start


def main():
    print("=== NumPy vs pure-Python loop (sum of squares) ===\n")
    print(f"{'n':>10}  {'Loop (s)':>12}  {'NumPy (s)':>12}  {'Speed-up':>10}")
    print("-" * 52)

    speedups = []
    for n in SIZES:
        # Warm-up small call for fairer NumPy timing
        if n >= 10**5:
            numpy_sum(1000)

        t_loop = time_once(loop_sum, n)
        t_numpy = time_once(numpy_sum, n)
        # Avoid division by zero
        speedup = t_loop / t_numpy if t_numpy > 0 else float("inf")
        speedups.append(speedup)
        print(f"{n:>10}  {t_loop:>12.6f}  {t_numpy:>12.6f}  {speedup:>10.2f}x")

    print("\n=== Discussion ===")
    print(
        "Speed-up generally grows as n increases: the Python loop pays a per-"
    )
    print(
        "element interpreter cost, while NumPy does vectorised C work. At very"
    )
    print(
        "small n the fixed overhead of creating arrays can shrink the apparent"
    )
    print(
        "gain, but for large n (10^5–10^7) NumPy is typically much faster and"
    )
    print("the speed-up tends to increase or stay high rather than shrink.")


if __name__ == "__main__":
    main()
