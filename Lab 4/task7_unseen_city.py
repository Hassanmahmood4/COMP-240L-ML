"""
Lab 4 Task 7:
Remove handle_unknown="ignore" from OneHotEncoder and predict a new city.
Record the error message and explain it.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from common import EDU_LEVELS, LOG_COLS, OTHER_NUM, load_split


def main():
    X_train, X_test, y_train, y_test = load_split()

    preprocess = ColumnTransformer(
        [
            (
                "log_num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
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
                        ("imp", SimpleImputer(strategy="median")),
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
                        ("ohe", OneHotEncoder()),  # no handle_unknown="ignore"
                    ]
                ),
                ["city"],
            ),
        ],
        sparse_threshold=0,
    )

    pipe = Pipeline(
        [("prep", preprocess), ("model", LogisticRegression(max_iter=2000))]
    ).fit(X_train, y_train)

    row = X_test.iloc[[0]].copy()
    row["city"] = "Quetta"
    print("Test row city set to unseen value: Quetta")
    try:
        pipe.predict(row)
        print("Unexpected: prediction succeeded.")
    except Exception as e:
        print("Error message:")
        print(f"{type(e).__name__}: {e}")
        print(
            "\nWithout handle_unknown='ignore', OneHotEncoder only accepts "
            "categories seen during fit. Quetta was not in training, so transform fails."
        )


if __name__ == "__main__":
    main()
