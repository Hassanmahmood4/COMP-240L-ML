"""
Lab 2 Task 8 (part B):
Reload the saved pipeline in a separate script and predict the cultivar
of the first five rows of the Wine dataset.
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_wine


MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "wine_pipeline.joblib"


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}. Run task8_save_pipeline.py first."
        )

    pipeline = joblib.load(MODEL_PATH)
    wine = load_wine()
    X5 = wine.data[:5]
    y5 = wine.target[:5]

    preds = pipeline.predict(X5)
    print("=== Predictions for first 5 Wine rows ===\n")
    print(f"{'Row':>4}  {'True':>6}  {'Predicted':>10}")
    print("-" * 26)
    for i, (true, pred) in enumerate(zip(y5, preds)):
        print(f"{i:>4}  {true:>6}  {pred:>10}")

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X5)
        print("\nClass probabilities:")
        for i, row in enumerate(probs):
            formatted = ", ".join(f"{p:.3f}" for p in row)
            print(f"  Row {i}: [{formatted}]")


if __name__ == "__main__":
    main()
