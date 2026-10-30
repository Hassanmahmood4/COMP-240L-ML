"""
Lab 4 Task 6:
Use predict_proba; compute precision and recall for thresholds 0.2, 0.3, 0.4, 0.5.
Choose a threshold for a bank that wants high recall.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline

from common import load_split, make_preprocess


def main():
    X_train, X_test, y_train, y_test = load_split()
    pipe = Pipeline(
        [
            ("prep", make_preprocess()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    ).fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    rows = []
    for thr in (0.2, 0.3, 0.4, 0.5):
        pred = (proba >= thr).astype(int)
        rows.append(
            {
                "threshold": thr,
                "precision": round(precision_score(y_test, pred, zero_division=0), 3),
                "recall": round(recall_score(y_test, pred), 3),
            }
        )
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    best = tab.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]
    print(
        f"\nFor high recall, choose threshold ≈ {best['threshold']} "
        f"(recall={best['recall']}, precision={best['precision']})."
    )


if __name__ == "__main__":
    main()
