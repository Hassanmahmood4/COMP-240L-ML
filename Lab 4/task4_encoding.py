"""
Lab 4 Task 4:
Encode education with OneHotEncoder instead of OrdinalEncoder.
What happens to the number of columns and to model performance?
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from common import load_split, make_preprocess


def main():
    X_train, X_test, y_train, y_test = load_split()

    for how in ("ordinal", "onehot"):
        prep = make_preprocess(education=how)
        Xt = prep.fit_transform(X_train)
        pipe = Pipeline(
            [("prep", prep), ("model", LogisticRegression(max_iter=2000))]
        ).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        print(f"education encoding = {how}")
        print(f"  n_columns : {Xt.shape[1]}")
        print(f"  accuracy  : {accuracy_score(y_test, pred):.3f}")
        print(f"  F1        : {f1_score(y_test, pred):.3f}")
        print()

    print(
        "One-hot is better when education levels are not equally spaced "
        "or when you want separate effects per level."
    )


if __name__ == "__main__":
    main()
