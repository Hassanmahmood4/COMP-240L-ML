"""
Lab 4 Task 1:
Replace StandardScaler with RobustScaler for the numeric columns.
Compare logistic regression accuracy, precision and recall with the reference run.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from common import load_split, make_preprocess


def main():
    X_train, X_test, y_train, y_test = load_split()

    print("Reference (manual): LogReg full preprocessing → acc 0.876, prec 0.842, rec 0.364\n")

    for name, scaler in [
        ("StandardScaler (reference)", StandardScaler()),
        ("RobustScaler", RobustScaler()),
    ]:
        pipe = Pipeline(
            [
                ("prep", make_preprocess(scaler=scaler)),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print(f"=== {name} ===")
        print(f"  accuracy : {accuracy_score(y_test, pred):.3f}")
        print(f"  precision: {precision_score(y_test, pred, zero_division=0):.3f}")
        print(f"  recall   : {recall_score(y_test, pred):.3f}")
        print()


if __name__ == "__main__":
    main()
