"""
Lab 3 Task 5 — Filtering questions:
(a) Karachi houses older than 30 years with Poor condition
(b) five most expensive houses sold in 2024
(c) share of Excellent houses in each city
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
CLEAN_PATH = DATA_DIR / "housing_cleaned.csv"


def main() -> None:
    df = pd.read_csv(CLEAN_PATH, parse_dates=["sale_date"])

    # (a)
    a = df.loc[
        (df["city"] == "Karachi")
        & (df["age_years"] > 30)
        & (df["condition"] == "Poor")
    ]
    print("=== (a) Karachi, age > 30, Poor condition ===")
    print(f"Count: {len(a)}")
    print(
        a[["city", "age_years", "condition", "area_sqft", "price", "sale_date"]]
        .head(20)
        .to_string(index=False)
    )

    # (b)
    sold_2024 = df.loc[df["sale_date"].dt.year == 2024]
    top5 = sold_2024.nlargest(5, "price")
    print("\n=== (b) Five most expensive houses sold in 2024 ===")
    print(
        top5[
            ["city", "price", "area_sqft", "price_per_sqft", "condition", "sale_date"]
        ].to_string(index=False)
    )

    # (c)
    share = (
        df.assign(is_excellent=df["condition"].eq("Excellent"))
        .groupby("city")["is_excellent"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )
    print("\n=== (c) Share of Excellent houses by city (%) ===")
    print(share.to_string())


if __name__ == "__main__":
    main()
