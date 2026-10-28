"""
Lab 3 Task 8 — Merge:
Create a city → average monthly rent table, merge it, and compute
price-to-annual-rent ratio for each city.
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
CLEAN_PATH = DATA_DIR / "housing_cleaned.csv"
RENT_PATH = DATA_DIR / "city_monthly_rent.csv"


def main() -> None:
    df = pd.read_csv(CLEAN_PATH)

    # Invented but plausible average monthly rents (PKR)
    rent = pd.DataFrame(
        {
            "city": ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar"],
            "avg_monthly_rent_pkr": [95000, 85000, 140000, 70000, 55000],
        }
    )
    rent.to_csv(RENT_PATH, index=False)
    print("=== City average monthly rent (invented, PKR) ===")
    print(rent.to_string(index=False))
    print(f"Saved: {RENT_PATH}\n")

    merged = df.merge(rent, on="city", how="left")
    merged["annual_rent_pkr"] = merged["avg_monthly_rent_pkr"] * 12
    merged["price_to_annual_rent"] = (
        merged["price"] / merged["annual_rent_pkr"]
    ).round(2)

    print("=== Sample of merged rows ===")
    print(
        merged[
            [
                "city",
                "price",
                "avg_monthly_rent_pkr",
                "annual_rent_pkr",
                "price_to_annual_rent",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )

    city_summary = (
        merged.groupby("city")
        .agg(
            mean_price=("price", "mean"),
            avg_monthly_rent_pkr=("avg_monthly_rent_pkr", "first"),
            mean_price_to_annual_rent=("price_to_annual_rent", "mean"),
        )
        .round(2)
        .sort_values("mean_price_to_annual_rent", ascending=False)
    )
    print("\n=== City-level price-to-annual-rent ratio ===")
    print(city_summary.to_string())
    print(
        "\nInterpretation: ratio ≈ how many years of rent equal the mean sale "
        "price in that city (gross, ignoring costs)."
    )


if __name__ == "__main__":
    main()
