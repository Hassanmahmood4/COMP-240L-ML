"""
Lab 1 Task 3:
From the Iris DataFrame:
(a) select only the virginica rows
(b) compute the mean of every feature for each species using groupby
(c) find the flower with the largest petal length
"""

import pandas as pd
from sklearn.datasets import load_iris


def main():
    iris = load_iris(as_frame=True)
    df = iris.frame
    # sklearn uses target names; map numeric target to species for readability
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

    print("=== Iris DataFrame (first 5 rows) ===")
    print(df.head())
    print(f"\nShape: {df.shape}\n")

    # (a) virginica rows only
    virginica = df[df["species"] == "virginica"]
    print("=== (a) Virginica rows only ===")
    print(f"Count: {len(virginica)}")
    print(virginica.head())
    print()

    # (b) mean of every feature for each species
    feature_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    means = df.groupby("species")[feature_cols].mean()
    print("=== (b) Mean of every feature by species ===")
    print(means)
    print()

    # (c) flower with largest petal length
    idx = df["petal length (cm)"].idxmax()
    largest = df.loc[idx]
    print("=== (c) Flower with largest petal length ===")
    print(largest)
    print(f"\nLargest petal length: {largest['petal length (cm)']} cm")
    print(f"Species: {largest['species']}")


if __name__ == "__main__":
    main()
