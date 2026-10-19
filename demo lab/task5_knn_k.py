"""
demo lab Task 5:
Train k-NN with k = 1, 3, 5, 7, 9, 15, 25.
Record training and test accuracy for each. Discuss best k and k = 1.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    k_values = [1, 3, 5, 7, 9, 15, 25]
    rows = []

    print("=== k-NN accuracy for different k (test_size=0.3) ===\n")
    print(f"{'k':>4}  {'Train Acc':>10}  {'Test Acc':>10}")
    print("-" * 30)

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        rows.append((k, train_acc, test_acc))
        print(f"{k:>4}  {train_acc:>10.4f}  {test_acc:>10.4f}")

    # Best k by test accuracy (prefer larger gap stability: pick max test, then mid k)
    best = max(rows, key=lambda r: (r[2], -abs(r[0] - 5)))
    print("\n=== Discussion ===")
    print(
        f"Best-looking k values are around {best[0]} "
        f"(test accuracy ≈ {best[2]:.4f})."
    )
    print(
        "Moderate k (e.g. 3–9) usually balances bias and variance well on Iris."
    )
    print(
        "At k = 1 the model memorises the training set, so training accuracy is"
    )
    print(
        "perfect (or nearly perfect), but it can overfit and be less stable on"
    )
    print("noisy points. Larger k smooths decisions but may underfit.")


if __name__ == "__main__":
    main()
