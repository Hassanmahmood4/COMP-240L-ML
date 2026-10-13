"""
Lab Task 5:
Set up a virtual environment, install scikit-learn, matplotlib, and pandas.
Load the Z-Alizadeh Sani sample dataset, preprocess with pandas, inspect
rows/columns, and visualize selected columns with matplotlib.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "data" / "z_alizadeh_sani_sample.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading Z-Alizadeh Sani sample dataset...")
    df = pd.read_csv(DATA_PATH)

    # Inspect rows and columns
    print("\n=== Dataset Shape ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n=== Column Names ===")
    for i, col in enumerate(df.columns, start=1):
        print(f"  {i:2d}. {col}")

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== First 5 Rows ===")
    print(df.head())

    # Preprocessing with pandas
    work = df.copy()

    # Encode categorical columns
    label_cols = ["Sex", "Cath"]
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        work[f"{col}_encoded"] = le.fit_transform(work[col])
        encoders[col] = le
        print(f"\nEncoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Select numeric features for scaling / visualization
    feature_cols = ["Age", "Weight", "BMI", "BP", "FBS", "LDL", "HDL", "EF-TTE"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(work[feature_cols])
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in feature_cols])

    print("\n=== Scaled Features (first 5 rows) ===")
    print(scaled_df.head().round(3))

    # ---- Visualizations with matplotlib ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Z-Alizadeh Sani Dataset — Column Visualizations", fontsize=14)

    # 1. Age distribution
    axes[0, 0].hist(work["Age"], bins=10, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Age Distribution")
    axes[0, 0].set_xlabel("Age")
    axes[0, 0].set_ylabel("Frequency")

    # 2. BMI by diagnosis
    cad_bmi = work.loc[work["Cath"] == "Cad", "BMI"]
    normal_bmi = work.loc[work["Cath"] == "Normal", "BMI"]
    axes[0, 1].boxplot([cad_bmi, normal_bmi], tick_labels=["Cad", "Normal"])
    axes[0, 1].set_title("BMI by Diagnosis (Cath)")
    axes[0, 1].set_ylabel("BMI")

    # 3. Blood Pressure vs Age (scatter, colored by Cath)
    colors = work["Cath"].map({"Cad": "crimson", "Normal": "seagreen"})
    axes[1, 0].scatter(work["Age"], work["BP"], c=colors, alpha=0.8, edgecolors="black")
    axes[1, 0].set_title("Age vs Blood Pressure")
    axes[1, 0].set_xlabel("Age")
    axes[1, 0].set_ylabel("BP")
    axes[1, 0].legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=8, label="Cad"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="seagreen", markersize=8, label="Normal"),
        ]
    )

    # 4. Mean of selected columns
    means = work[["Age", "BMI", "BP", "FBS", "LDL", "HDL"]].mean()
    axes[1, 1].bar(means.index, means.values, color="darkorange", edgecolor="black")
    axes[1, 1].set_title("Mean Values of Key Columns")
    axes[1, 1].set_ylabel("Mean")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "task5_column_visualizations.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    # Show interactive window only when a GUI backend is available
    backend = plt.get_backend().lower()
    if backend not in {"agg", "template"}:
        plt.show()
    else:
        plt.close(fig)

    print("\nTask 5 completed successfully.")


if __name__ == "__main__":
    main()
