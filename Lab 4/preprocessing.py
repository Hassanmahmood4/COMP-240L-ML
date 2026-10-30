"""
Lab 4: Data Preprocessing and Preparation for Machine Learning
Parts A–H from the lab manual (run as one file).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

print("=== PART A: The raw dataset ===")
rng = np.random.default_rng(7)
n = 1000
age = rng.integers(21, 66, n).astype(float)
income = rng.lognormal(mean=11.0, sigma=0.6, size=n).round(0)
loan_amount = (income * rng.uniform(1.5, 8.0, n)).round(0)
credit_score = rng.normal(650, 80, n).clip(300, 850).round(0)
emp_years = np.clip((age - 21) * rng.uniform(0.2, 0.9, n), 0, None).round(1)
city = rng.choice(
    ["Islamabad", "Lahore", "Karachi", "Peshawar"], n, p=[0.3, 0.3, 0.3, 0.1]
)
education = rng.choice(
    ["High School", "Bachelors", "Masters", "PhD"], n, p=[0.3, 0.45, 0.2, 0.05]
)
edu_rank = (
    pd.Series(education)
    .map({"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3})
    .to_numpy()
)
z = (
    -2.6
    + 1.0 * (np.log(income) - 11.0) / 0.6
    + 1.2 * (credit_score - 650) / 80
    - 0.6 * (np.log(loan_amount / income) - 1.5)
    + 0.35 * edu_rank
    + rng.normal(0, 0.8, n)
)
approved = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
data = pd.DataFrame(
    {
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "emp_years": emp_years,
        "city": city,
        "education": education,
        "approved": approved,
    }
)
for col, k in [
    ("income", 60),
    ("credit_score", 80),
    ("emp_years", 40),
    ("education", 50),
    ("city", 30),
]:
    data.loc[rng.choice(n, k, replace=False), col] = np.nan
data.loc[5, "age"] = -1.0
raw_path = DATA_DIR / "loan_data_raw.csv"
data.to_csv(raw_path, index=False)

df = pd.read_csv(raw_path)
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isna().sum())
print("\nClass balance (approved):\n", df["approved"].value_counts())
print("Approval rate:", round(df["approved"].mean(), 3))
print("\nSummary statistics:\n", df.describe().round(1))
print(
    "\nSkewness of numeric columns:\n",
    df[["age", "income", "loan_amount", "credit_score", "emp_years"]].skew().round(2),
)
print("\nRows with age < 18:", (df["age"] < 18).sum())
df.loc[df["age"] < 18, "age"] = np.nan

print("\n=== PART B: Train/test split ===")
X = df.drop(columns="approved")
y = df["approved"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("Train:", X_train.shape, "| Test:", X_test.shape)
print(
    "Approval rate  train:",
    round(y_train.mean(), 3),
    "| test:",
    round(y_test.mean(), 3),
)
num_cols = ["age", "income", "loan_amount", "credit_score", "emp_years"]

print("\n=== PART C: Imputation ===")
imp = SimpleImputer(strategy="median")
imp.fit(X_train[num_cols])
print("Medians learned from training data:")
print(pd.Series(imp.statistics_, index=num_cols).round(1))
Xtr_num = pd.DataFrame(
    imp.transform(X_train[num_cols]), columns=num_cols, index=X_train.index
)
Xte_num = pd.DataFrame(
    imp.transform(X_test[num_cols]), columns=num_cols, index=X_test.index
)
print(
    "Missing in train numeric after imputing:",
    int(Xtr_num.isna().sum().sum()),
    "| test:",
    int(Xte_num.isna().sum().sum()),
)

print("\n=== PART D: Scaling methods compared (training set) ===")
scalers = {
    "Standard": StandardScaler(),
    "MinMax": MinMaxScaler(),
    "Robust": RobustScaler(),
}
rows = []
for name, sc in scalers.items():
    tr = sc.fit_transform(Xtr_num[["income"]])[:, 0]
    te = sc.transform(Xte_num[["income"]])[:, 0]
    rows.append([name, tr.min(), tr.max(), tr.mean(), tr.std(), te.min(), te.max()])
tab = pd.DataFrame(
    rows,
    columns=[
        "scaler",
        "train_min",
        "train_max",
        "train_mean",
        "train_std",
        "test_min",
        "test_max",
    ],
).round(2)
print("Income after each scaler:\n", tab.to_string(index=False))
mm = MinMaxScaler().fit(Xtr_num[["income"]])
print(
    "\nMinMax value for a NEW applicant earning 500,000:",
    round(float(mm.transform(pd.DataFrame({"income": [500000]}))[0, 0]), 2),
    "(outside 0-1 because it exceeds the training maximum)",
)
print(
    "\nSkewness of income  raw:",
    round(Xtr_num["income"].skew(), 2),
    "| after log1p:",
    round(np.log1p(Xtr_num["income"]).skew(), 2),
)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].hist(Xtr_num["income"], bins=40, edgecolor="black")
axes[0].set_title("Income (raw, skewed)")
axes[1].hist(np.log1p(Xtr_num["income"]), bins=40, edgecolor="black", color="#4f81bd")
axes[1].set_title("log(1 + income)")
axes[2].hist(
    StandardScaler().fit_transform(np.log1p(Xtr_num[["income"]])),
    bins=40,
    edgecolor="black",
    color="#9bbb59",
)
axes[2].set_title("log, then standardised")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_scaling.png", dpi=150)
plt.close()

print("\n=== PART E: Encoding ===")
levels = [["High School", "Bachelors", "Masters", "PhD"]]
ord_enc = OrdinalEncoder(
    categories=levels, handle_unknown="use_encoded_value", unknown_value=np.nan
)
edu_filled = X_train[["education"]].fillna("Bachelors")
print("Education -> ordinal codes:")
print(
    pd.DataFrame(
        {
            "education": edu_filled["education"].head(5).to_numpy(),
            "code": ord_enc.fit_transform(edu_filled)[:5, 0],
        }
    )
)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
city_filled = X_train[["city"]].fillna("Missing")
ohe.fit(city_filled)
print("\nOne-hot columns:", list(ohe.get_feature_names_out()))
new = pd.DataFrame({"city": ["Lahore", "Quetta"]})
print("Encoding a known and an unseen city:\n", ohe.transform(new))

print("\n=== PART F: ColumnTransformer + Pipeline ===")
log_cols = ["income", "loan_amount"]
other_num = ["age", "credit_score", "emp_years"]
preprocess = ColumnTransformer(
    [
        (
            "log_num",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "log",
                        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    ),
                    ("scale", StandardScaler()),
                ]
            ),
            log_cols,
        ),
        (
            "num",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            other_num,
        ),
        (
            "edu",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    (
                        "ord",
                        OrdinalEncoder(
                            categories=levels,
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
Xt = preprocess.fit_transform(X_train)
print("Transformed training matrix shape:", Xt.shape)
print("Output feature names:", list(preprocess.get_feature_names_out()))
print("NaNs in transformed data:", int(np.isnan(Xt).sum()))

baseline_prep = ColumnTransformer(
    [("num", SimpleImputer(strategy="median"), num_cols)]
)


def evaluate(name, prep, model):
    pipe = Pipeline([("prep", prep), ("model", model)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return [
        name,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred),
        f1_score(y_test, pred),
    ]


results = [
    evaluate("KNN, impute only (raw scales)", baseline_prep, KNeighborsClassifier()),
    evaluate("KNN, full preprocessing", preprocess, KNeighborsClassifier()),
    evaluate("LogReg, impute only", baseline_prep, LogisticRegression(max_iter=2000)),
    evaluate(
        "LogReg, full preprocessing", preprocess, LogisticRegression(max_iter=2000)
    ),
    evaluate(
        "LogReg, full + class_weight=balanced",
        preprocess,
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ),
]
res = pd.DataFrame(
    results, columns=["model", "accuracy", "precision", "recall", "F1 (approved)"]
).round(3)
print("\nTest-set results:\n", res.to_string(index=False))
print("Majority-class baseline accuracy:", round(1 - y_test.mean(), 3))

fig, ax = plt.subplots(figsize=(8, 4))
pos = np.arange(len(res))
ax.barh(pos - 0.2, res["accuracy"], height=0.4, label="Accuracy")
ax.barh(pos + 0.2, res["F1 (approved)"], height=0.4, label="F1 (approved)")
ax.set_yticks(pos)
ax.set_yticklabels(res["model"], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.legend(loc="lower right")
ax.set_title("Effect of preprocessing")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_models.png", dpi=150)
plt.close()

print("\n=== PART G: Data leakage ===")
sc_wrong = StandardScaler().fit(
    pd.concat([Xtr_num[["income"]], Xte_num[["income"]]])
)
sc_right = StandardScaler().fit(Xtr_num[["income"]])
print(
    "Mean income learned by scaler, fit on train + test:",
    round(float(sc_wrong.mean_[0]), 1),
)
print(
    "Mean income learned by scaler, fit on train only   :",
    round(float(sc_right.mean_[0]), 1),
)
print("-> Small difference here, but the 'wrong' scaler has seen test data.")

rs = np.random.default_rng(0)
Xn = rs.normal(size=(100, 2000))
yn = rs.integers(0, 2, 100)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
Xsel = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)
leaky = cross_val_score(LogisticRegression(max_iter=1000), Xsel, yn, cv=cv).mean()
proper_pipe = Pipeline(
    [("sel", SelectKBest(f_classif, k=20)), ("clf", LogisticRegression(max_iter=1000))]
)
proper = cross_val_score(proper_pipe, Xn, yn, cv=cv).mean()
print(f"\nRandom labels, select-then-validate (leaky): CV accuracy = {leaky:.3f}")
print(f"Random labels, selection inside Pipeline   : CV accuracy = {proper:.3f}")
print("True accuracy on random labels is about 0.50.")

Xte_t = preprocess.transform(X_test)
cols = list(preprocess.get_feature_names_out())
pd.DataFrame(Xt, columns=cols).assign(approved=y_train.to_numpy()).to_csv(
    DATA_DIR / "loan_train_prepared.csv", index=False
)
pd.DataFrame(Xte_t, columns=cols).assign(approved=y_test.to_numpy()).to_csv(
    DATA_DIR / "loan_test_prepared.csv", index=False
)
print(
    "\n=== PART H ===\nSaved loan_train_prepared.csv",
    Xt.shape,
    "and loan_test_prepared.csv",
    Xte_t.shape,
)
