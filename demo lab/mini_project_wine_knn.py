"""
demo lab Mini Project: Wine dataset exploration and k-NN classification

1. Load Wine into a DataFrame; show shape, class distribution, summary stats
2. At least two meaningful plots
3. k-NN (k=5), 70/30 stratified split; accuracy, confusion matrix, report
4. StandardScaler (fit on train only); compare accuracy before/after
5. Written explanation (printed for the report)
Bonus: Test accuracy vs k (1..30) for unscaled and scaled on one graph
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "mini_project"
RANDOM_STATE = 42
TEST_SIZE = 0.30
K_DEFAULT = 5


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    section("1. Load Wine dataset")
    wine = load_wine(as_frame=True)
    df = wine.frame.copy()
    df["cultivar"] = df["target"].map(lambda t: f"class_{t}")

    print(f"Shape: {df.shape}  (samples={df.shape[0]}, columns={df.shape[1]})")
    print(f"Features: {len(wine.feature_names)}")
    print(f"Classes: {list(wine.target_names)} -> encoded as 0, 1, 2\n")

    print("Class distribution:")
    dist = df["target"].value_counts().sort_index()
    for cls, count in dist.items():
        print(f"  class_{cls}: {count} ({100 * count / len(df):.1f}%)")

    print("\nSummary statistics:")
    print(df[wine.feature_names].describe().T.round(3).to_string())

    # ------------------------------------------------------------------
    # 2. Plots
    # ------------------------------------------------------------------
    section("2. Exploratory plots")

    # Plot A: histogram of alcohol per class
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(df["target"].unique()):
        subset = df.loc[df["target"] == cls, "alcohol"]
        ax.hist(subset, bins=12, alpha=0.55, label=f"class_{cls}")
    ax.set_xlabel("Alcohol")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of alcohol by cultivar class")
    ax.legend()
    fig.tight_layout()
    hist_path = OUTPUT_DIR / "plot1_alcohol_hist.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {hist_path}")

    # Plot B: scatter of two useful features
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in sorted(df["target"].unique()):
        subset = df[df["target"] == cls]
        ax.scatter(
            subset["flavanoids"],
            subset["color_intensity"],
            alpha=0.8,
            label=f"class_{cls}",
            edgecolors="k",
            linewidths=0.25,
        )
    ax.set_xlabel("Flavanoids")
    ax.set_ylabel("Color intensity")
    ax.set_title("Scatter: flavanoids vs color intensity")
    ax.legend()
    fig.tight_layout()
    scatter_path = OUTPUT_DIR / "plot2_scatter.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {scatter_path}")

    # ------------------------------------------------------------------
    # 3. Baseline k-NN (unscaled)
    # ------------------------------------------------------------------
    section("3. k-NN (k=5), 70/30 stratified split — unscaled")
    X = df[wine.feature_names].to_numpy()
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    knn_raw = KNeighborsClassifier(n_neighbors=K_DEFAULT)
    knn_raw.fit(X_train, y_train)
    y_pred_raw = knn_raw.predict(X_test)
    acc_raw = accuracy_score(y_test, y_pred_raw)

    print(f"Test accuracy (unscaled): {acc_raw:.4f}")
    print("\nConfusion matrix (unscaled):")
    cm_raw = confusion_matrix(y_test, y_pred_raw)
    print(cm_raw)
    print("\nClassification report (unscaled):")
    report_raw = classification_report(y_test, y_pred_raw, digits=4)
    print(report_raw)

    # ------------------------------------------------------------------
    # 4. Standardised features
    # ------------------------------------------------------------------
    section("4. StandardScaler (fit on train only) + retrain")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn_scaled = KNeighborsClassifier(n_neighbors=K_DEFAULT)
    knn_scaled.fit(X_train_s, y_train)
    y_pred_scaled = knn_scaled.predict(X_test_s)
    acc_scaled = accuracy_score(y_test, y_pred_scaled)

    print(f"Test accuracy (scaled):   {acc_scaled:.4f}")
    print(f"Test accuracy (unscaled): {acc_raw:.4f}")
    print(f"Change: {acc_scaled - acc_raw:+.4f}")
    print("\nConfusion matrix (scaled):")
    cm_scaled = confusion_matrix(y_test, y_pred_scaled)
    print(cm_scaled)
    print("\nClassification report (scaled):")
    report_scaled = classification_report(y_test, y_pred_scaled, digits=4)
    print(report_scaled)

    # ------------------------------------------------------------------
    # 5. Explanation
    # ------------------------------------------------------------------
    section("5. Why scaling matters for k-NN on Wine")
    explanation = (
        "Wine features are measured in very different units and ranges "
        "(for example proline is often in the hundreds while phenols are near 0–5). "
        "k-NN classifies a point by nearest neighbours in Euclidean distance, so "
        "features with large numeric ranges dominate the distance and drown out "
        "smaller-scale but informative features. Fitting StandardScaler on the "
        "training set only (then transforming the test set) puts every feature on "
        "a comparable scale without leaking test information. On this dataset, "
        f"accuracy moved from {acc_raw:.4f} (unscaled) to {acc_scaled:.4f} (scaled). "
        "That change shows k-NN is sensitive to feature units: scaling usually "
        "improves fairness of neighbour search when features are heterogeneous."
    )
    print(explanation)

    # ------------------------------------------------------------------
    # Bonus: accuracy vs k for unscaled and scaled
    # ------------------------------------------------------------------
    section("Bonus: test accuracy vs k (1..30), unscaled vs scaled")
    ks = list(range(1, 31))
    acc_u, acc_s = [], []
    for k in ks:
        m_u = KNeighborsClassifier(n_neighbors=k)
        m_u.fit(X_train, y_train)
        acc_u.append(m_u.score(X_test, y_test))

        m_s = KNeighborsClassifier(n_neighbors=k)
        m_s.fit(X_train_s, y_train)
        acc_s.append(m_s.score(X_test_s, y_test))

    best_u_k = ks[int(np.argmax(acc_u))]
    best_s_k = ks[int(np.argmax(acc_s))]
    print(f"Best unscaled: k={best_u_k}, accuracy={max(acc_u):.4f}")
    print(f"Best scaled:   k={best_s_k}, accuracy={max(acc_s):.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, acc_u, marker="o", markersize=3, label="Unscaled")
    ax.plot(ks, acc_s, marker="s", markersize=3, label="Scaled")
    ax.axvline(best_u_k, color="C0", linestyle="--", alpha=0.4)
    ax.axvline(best_s_k, color="C1", linestyle="--", alpha=0.4)
    ax.set_xlabel("k")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Wine k-NN: test accuracy vs k (unscaled vs scaled)")
    ax.set_xticks(range(1, 31, 2))
    ax.legend()
    fig.tight_layout()
    bonus_path = OUTPUT_DIR / "plot3_accuracy_vs_k.png"
    fig.savefig(bonus_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {bonus_path}")

    # Persist text results for the report generator
    results_path = OUTPUT_DIR / "results.txt"
    results_path.write_text(
        "\n".join(
            [
                f"shape={df.shape}",
                f"class_distribution={dist.to_dict()}",
                f"acc_unscaled={acc_raw:.6f}",
                f"acc_scaled={acc_scaled:.6f}",
                f"best_unscaled_k={best_u_k}",
                f"best_unscaled_acc={max(acc_u):.6f}",
                f"best_scaled_k={best_s_k}",
                f"best_scaled_acc={max(acc_s):.6f}",
                "",
                "confusion_unscaled=",
                str(cm_raw),
                "",
                "confusion_scaled=",
                str(cm_scaled),
                "",
                "report_unscaled=",
                report_raw,
                "",
                "report_scaled=",
                report_scaled,
                "",
                "explanation=",
                explanation,
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nSaved results: {results_path}")
    print("\nMini project complete.")


if __name__ == "__main__":
    main()
