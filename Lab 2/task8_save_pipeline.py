"""
Lab 2 Task 8 (part A):
Train a Wine classification pipeline and save it with joblib.dump.
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_PATH = OUTPUT_DIR / "wine_pipeline.joblib"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.3, random_state=42, stratify=wine.target
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ]
    )
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved pipeline to: {MODEL_PATH}")
    print("Next: run task8_load_predict.py to reload and predict.")


if __name__ == "__main__":
    main()
