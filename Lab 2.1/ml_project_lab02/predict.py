"""
Load the saved pipeline and predict the first 10 samples of the Breast Cancer dataset.
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer


MODEL_PATH = Path(__file__).resolve().parent / "breast_cancer_pipeline.joblib"


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found: {MODEL_PATH}\nRun train.py first."
        )

    pipeline = joblib.load(MODEL_PATH)
    data = load_breast_cancer()
    X10 = data.data[:10]
    y10 = data.target[:10]
    preds = pipeline.predict(X10)

    print("Predictions for the first 10 samples\n")
    print(f"{'Index':>5}  {'True':>6}  {'Predicted':>10}  {'Label':>12}")
    print("-" * 40)
    for i, (true, pred) in enumerate(zip(y10, preds)):
        label = data.target_names[pred]
        print(f"{i:>5}  {true:>6}  {pred:>10}  {label:>12}")


if __name__ == "__main__":
    main()
