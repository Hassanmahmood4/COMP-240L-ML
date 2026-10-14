"""
Train a StandardScaler + LogisticRegression pipeline on the Breast Cancer dataset.
Saves the fitted pipeline with joblib.
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path(__file__).resolve().parent / "breast_cancer_pipeline.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.30


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    accuracy = pipeline.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
