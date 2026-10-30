"""
Lab 4 Task 5:
Train a DecisionTreeClassifier (max_depth=4) with and without scaling.
Does scaling change the result? Explain.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier

from common import EDU_LEVELS, LOG_COLS, OTHER_NUM, load_split


def make_tree_prep(scale: bool):
    log_steps = [("imp", SimpleImputer(strategy="median"))]
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        log_steps += [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
        num_steps.append(("scale", StandardScaler()))
    else:
        log_steps.append(
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
        )

    return ColumnTransformer(
        [
            ("log_num", Pipeline(log_steps), LOG_COLS),
            ("num", Pipeline(num_steps), OTHER_NUM),
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
    X_train, X_test, y_train, y_test = load_split()

    for scale in (False, True):
        pipe = Pipeline(
            [
                ("prep", make_tree_prep(scale)),
                ("model", DecisionTreeClassifier(max_depth=4, random_state=42)),
            ]
        ).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print(f"scaling={scale}")
        print(f"  accuracy: {accuracy_score(y_test, pred):.3f}")
        print(f"  F1      : {f1_score(y_test, pred):.3f}")
        print()

    print(
        "Trees split on thresholds, so they are mostly scale-invariant. "
        "Scaling usually does not change the result."
    )


if __name__ == "__main__":
    main()
