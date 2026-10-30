"""
Lab 4 Task 3:
Add add_indicator=True to the numeric SimpleImputer objects.
How many output columns are there now, and does the F1 score change?
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from common import load_split, make_preprocess


def main():
    X_train, X_test, y_train, y_test = load_split()

    for flag in (False, True):
        prep = make_preprocess(add_indicator=flag)
        Xt = prep.fit_transform(X_train)
        pipe = Pipeline(
            [("prep", prep), ("model", LogisticRegression(max_iter=2000))]
        ).fit(X_train, y_train)
        print(f"add_indicator={flag}")
        print(f"  output columns: {Xt.shape[1]}")
        print(f"  F1 (approved) : {f1_score(y_test, pipe.predict(X_test)):.3f}")
        print()


if __name__ == "__main__":
    main()
