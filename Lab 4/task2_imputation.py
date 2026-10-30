"""
Lab 4 Task 2:
Change the numeric imputer from median to mean, then to KNNImputer.
Compare the income histogram and model F1 for each.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from common import EDU_LEVELS, LOG_COLS, OTHER_NUM, OUT_DIR, load_split


def build_prep(imputer):
    return ColumnTransformer(
        [
            (
                "log_num",
                Pipeline(
                    [
                        ("imp", imputer),
                        (
                            "log",
                            FunctionTransformer(
                                np.log1p, feature_names_out="one-to-one"
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                LOG_COLS,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imp", type(imputer)(**imputer.get_params())),
                        ("scale", StandardScaler()),
                    ]
                ),
                OTHER_NUM,
            ),
            (
                "edu",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                categories=EDU_LEVELS,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ["education"],
            ),
            (
                "city",
                Pipeline(
                    [
                        (
                            "imp",
                            SimpleImputer(strategy="constant", fill_value="Missing"),
                        ),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["city"],
            ),
        ],
        sparse_threshold=0,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()

    strategies = {
        "median": SimpleImputer(strategy="median"),
        "mean": SimpleImputer(strategy="mean"),
        "KNNImputer": KNNImputer(n_neighbors=5),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (name, imputer) in zip(axes, strategies.items()):
        filled = type(imputer)(**imputer.get_params()).fit_transform(
            X_train[["income"]]
        )[:, 0]
        ax.hist(filled, bins=40, edgecolor="black")
        ax.set_title(f"Income after {name}")

        pipe = Pipeline(
            [
                ("prep", build_prep(type(imputer)(**imputer.get_params()))),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ).fit(X_train, y_train)
        print(f"{name:12s}  F1 (approved) = {f1_score(y_test, pipe.predict(X_test)):.3f}")

    fig.tight_layout()
    out = OUT_DIR / "task2_imputation_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
