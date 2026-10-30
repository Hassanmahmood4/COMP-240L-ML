"""
Lab 4 shared helpers used by the exercise task scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
RAW_PATH = DATA_DIR / "loan_data_raw.csv"

EDU_LEVELS = [["High School", "Bachelors", "Masters", "PhD"]]
LOG_COLS = ["income", "loan_amount"]
OTHER_NUM = ["age", "credit_score", "emp_years"]


def ensure_raw_data():
    if RAW_PATH.exists():
        return
    raise SystemExit(
        f"{RAW_PATH} not found. Run preprocessing.py first."
    )


def load_split():
    ensure_raw_data()
    df = pd.read_csv(RAW_PATH)
    df.loc[df["age"] < 18, "age"] = np.nan
    X = df.drop(columns="approved")
    y = df["approved"]
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def make_preprocess(
    scaler=None, numeric_imputer="median", add_indicator=False, education="ordinal"
):
    if scaler is None:
        scaler = StandardScaler()

    log_imp = SimpleImputer(strategy=numeric_imputer, add_indicator=add_indicator)
    num_imp = SimpleImputer(strategy=numeric_imputer, add_indicator=add_indicator)

    log_pipe = Pipeline(
        [
            ("imp", log_imp),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", scaler),
        ]
    )
    num_pipe = Pipeline([("imp", num_imp), ("scale", scaler.__class__())])

    if education == "ordinal":
        edu_pipe = Pipeline(
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
        )
    else:
        edu_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

    city_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("log_num", log_pipe, LOG_COLS),
            ("num", num_pipe, OTHER_NUM),
            ("edu", edu_pipe, ["education"]),
            ("city", city_pipe, ["city"]),
        ],
        sparse_threshold=0,
    )
