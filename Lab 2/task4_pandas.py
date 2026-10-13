"""
Lab Task 4:
Install pandas with PIP. Load the Z-Alizadeh Sani sample CSV into a DataFrame,
perform filtering/sorting, and generate summary statistics (mean, median).
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "z_alizadeh_sani_sample.csv"


def main():
    print("Pandas version:", pd.__version__)
    print(f"Loading dataset from: {DATA_PATH}\n")

    # Load CSV into a DataFrame
    df = pd.read_csv(DATA_PATH)

    print("=== Original DataFrame (first 5 rows) ===")
    print(df.head())
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nColumn names:")
    print(list(df.columns))

    # Filtering: patients older than 55 with CAD
    older_cad = df[(df["Age"] > 55) & (df["Cath"] == "Cad")]
    print("\n=== Filtered: Age > 55 AND Cath == 'Cad' ===")
    print(older_cad[["Age", "Sex", "BMI", "BP", "FBS", "Cath"]])

    # Filtering: female patients with BMI >= 27
    female_high_bmi = df[(df["Sex"] == "Female") & (df["BMI"] >= 27)]
    print("\n=== Filtered: Female AND BMI >= 27 ===")
    print(female_high_bmi[["Age", "Sex", "BMI", "Weight", "Cath"]])

    # Sorting by Age (descending) then BMI (ascending)
    sorted_df = df.sort_values(by=["Age", "BMI"], ascending=[False, True])
    print("\n=== Sorted by Age (desc), BMI (asc) — top 10 ===")
    print(sorted_df[["Age", "Sex", "BMI", "BP", "Cath"]].head(10))

    # Summary statistics
    numeric_cols = ["Age", "Weight", "BMI", "BP", "FBS", "LDL", "HDL", "EF-TTE"]
    print("\n=== Summary Statistics (describe) ===")
    print(df[numeric_cols].describe())

    print("\n=== Mean ===")
    print(df[numeric_cols].mean().round(2))

    print("\n=== Median ===")
    print(df[numeric_cols].median().round(2))

    # Value counts for categorical target
    print("\n=== Cath (diagnosis) value counts ===")
    print(df["Cath"].value_counts())

    print("\n=== Sex value counts ===")
    print(df["Sex"].value_counts())


if __name__ == "__main__":
    main()
