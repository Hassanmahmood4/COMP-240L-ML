"""
Lab 4 Task 8:
Clip income at its 99th percentile (training data only) before scaling.
Does it affect the MinMax result for income 500,000?
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from common import load_split


def main():
    X_train, X_test, y_train, y_test = load_split()

    imp = SimpleImputer(strategy="median")
    train_income = imp.fit_transform(X_train[["income"]])[:, 0]
    p99 = np.quantile(train_income, 0.99)
    train_clipped = np.clip(train_income, None, p99)

    print(f"Train income 99th percentile: {p99:.1f}")

    mm_raw = MinMaxScaler().fit(train_income.reshape(-1, 1))
    mm_clip = MinMaxScaler().fit(train_clipped.reshape(-1, 1))
    new = np.array([[500000.0]])

    print(
        "MinMax(500000) WITHOUT clipping:",
        round(float(mm_raw.transform(new)[0, 0]), 3),
    )
    print(
        "MinMax(500000) WITH 99th-pct clip on train:",
        round(float(mm_clip.transform(new)[0, 0]), 3),
    )
    print(
        "\nYes — clipping changes the training max used by MinMax, "
        "so 500,000 maps to a different scaled value."
    )


if __name__ == "__main__":
    main()
