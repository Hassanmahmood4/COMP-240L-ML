"""
Lab 3 Task 4 — Alternative outlier rule:
Detect outliers in price_per_sqft using |z| > 3.
Compare with rows flagged by the IQR rule on price.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
CLEAN_PATH = DATA_DIR / "housing_cleaned.csv"


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def main() -> None:
    df = pd.read_csv(CLEAN_PATH)

    # Z-score outliers on price_per_sqft
    pps = df["price_per_sqft"]
    z = (pps - pps.mean()) / pps.std(ddof=0)
    z_mask = z.abs() > 3
    z_outliers = df.loc[z_mask].copy()
    z_outliers["z_score"] = z[z_mask].round(3)

    # IQR outliers on price
    iqr_mask = iqr_outlier_mask(df["price"])
    iqr_outliers = df.loc[iqr_mask].copy()

    print("=== Z-score outliers on price_per_sqft (|z| > 3) ===")
    print(f"Count: {z_mask.sum()}")
    print(
        z_outliers[
            ["city", "area_sqft", "price", "price_per_sqft", "z_score", "condition"]
        ]
        .sort_values("z_score", key=np.abs, ascending=False)
        .head(15)
        .to_string(index=True)
    )

    print("\n=== IQR outliers on price (1.5 × IQR) ===")
    print(f"Count: {iqr_mask.sum()}")
    print(
        iqr_outliers[["city", "area_sqft", "price", "price_per_sqft", "condition"]]
        .sort_values("price", ascending=False)
        .head(15)
        .to_string(index=True)
    )

    both = z_mask & iqr_mask
    only_z = z_mask & ~iqr_mask
    only_iqr = iqr_mask & ~z_mask
    print("\n=== Comparison ===")
    print(f"Flagged by both rules:     {both.sum()}")
    print(f"Only z-score (pps):        {only_z.sum()}")
    print(f"Only IQR (price):          {only_iqr.sum()}")
    print(
        "\nComment: z-score on price_per_sqft finds extreme unit prices; "
        "IQR on total price finds extremely expensive/cheap listings overall. "
        "They overlap partly but are not identical."
    )


if __name__ == "__main__":
    main()
