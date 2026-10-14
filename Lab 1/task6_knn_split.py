"""
Lab 1 Task 6:
Repeat the k-NN experiment with test sizes 10%, 20%, 30%, and 50%.
Observe how test accuracy and its stability change.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    test_sizes = [0.10, 0.20, 0.30, 0.50]
    k_values = [1, 3, 5, 7, 9, 15, 25]
    n_repeats = 10  # multiple seeds to gauge stability

    print("=== Test accuracy mean ± std over seeds (k=5 fixed for overview) ===\n")
    print(f"{'Test size':>10}  {'Mean Acc':>10}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 52)

    summary = []
    for test_size in test_sizes:
        scores = []
        for seed in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))
        scores = np.array(scores)
        summary.append((test_size, scores.mean(), scores.std(), scores.min(), scores.max()))
        print(
            f"{test_size:>10.0%}  {scores.mean():>10.4f}  {scores.std():>8.4f}  "
            f"{scores.min():>8.4f}  {scores.max():>8.4f}"
        )

    print("\n=== Full table: test accuracy by k and test size (random_state=42) ===\n")
    header = f"{'k':>4}" + "".join(f"  {ts:>8.0%}" for ts in test_sizes)
    print(header)
    print("-" * len(header))

    for k in k_values:
        line = f"{k:>4}"
        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            line += f"  {model.score(X_test, y_test):>8.4f}"
        print(line)

    print("\n=== Discussion ===")
    print(
        "Smaller test sets (10–20%) often give high but less stable test accuracy,"
    )
    print(
        "because a few misclassified points swing the score more (higher std)."
    )
    print(
        "Larger test sets (30–50%) usually give a more reliable estimate of"
    )
    print(
        "generalisation, though training data shrinks so accuracy may drop slightly."
    )
    print(
        "Overall: as the test fraction grows, mean accuracy can change a little,"
    )
    print("while stability (lower variance across seeds) often improves.")


if __name__ == "__main__":
    main()
