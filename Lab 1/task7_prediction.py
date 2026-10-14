"""
Lab 1 Task 7:
Predict the species for three new flowers (plausible measurements)
and report the class probabilities.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Invented but plausible Iris measurements:
    # [sepal length, sepal width, petal length, petal width] in cm
    new_flowers = np.array(
        [
            [5.0, 3.5, 1.4, 0.2],  # likely setosa
            [6.0, 2.8, 4.5, 1.5],  # likely versicolor
            [6.8, 3.0, 5.8, 2.2],  # likely virginica
        ]
    )
    descriptions = [
        "Small petals, wide sepal — setosa-like",
        "Medium petals — versicolor-like",
        "Long/wide petals — virginica-like",
    ]

    preds = model.predict(new_flowers)
    probs = model.predict_proba(new_flowers)

    print("=== Predictions for 3 new flowers (k=5) ===\n")
    print(
        "Features order: sepal length, sepal width, petal length, petal width (cm)\n"
    )

    for i, (flower, desc, pred, prob) in enumerate(
        zip(new_flowers, descriptions, preds, probs), start=1
    ):
        print(f"Flower {i}: {flower.tolist()}")
        print(f"  Note: {desc}")
        print(f"  Predicted species: {target_names[pred]}")
        print("  Class probabilities:")
        for name, p in zip(target_names, prob):
            print(f"    {name:12s}: {p:.4f}")
        print()


if __name__ == "__main__":
    main()
