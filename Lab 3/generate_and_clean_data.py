"""
Lab 3 helper: generate a raw Pakistan housing CSV (with missing values /
outliers) and a cleaned CSV used by later tasks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PATH = DATA_DIR / "housing_raw.csv"
CLEAN_PATH = DATA_DIR / "housing_cleaned.csv"
RANDOM_STATE = 42


def generate_raw(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    cities = np.array(["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar"])
    conditions = np.array(["Poor", "Fair", "Good", "Excellent"])

    city = rng.choice(cities, size=n, p=[0.30, 0.28, 0.18, 0.14, 0.10])
    base_pps = {
        "Karachi": 18000,
        "Lahore": 16000,
        "Islamabad": 25000,
        "Rawalpindi": 14000,
        "Peshawar": 12000,
    }
    area = rng.integers(800, 4500, size=n).astype(float)
    age = rng.integers(0, 55, size=n)
    condition = rng.choice(conditions, size=n, p=[0.12, 0.25, 0.38, 0.25])

    # Sale dates spanning 2022–2024
    start = np.datetime64("2022-01-01")
    offsets = rng.integers(0, 365 * 3, size=n)
    sale_date = start + offsets.astype("timedelta64[D]")

    pps = np.array([base_pps[c] for c in city], dtype=float)
    pps *= rng.normal(1.0, 0.18, size=n)
    # Older / poorer homes a bit cheaper
    pps *= np.where(condition == "Poor", 0.75, 1.0)
    pps *= np.where(condition == "Excellent", 1.20, 1.0)
    pps *= np.clip(1.15 - 0.006 * age, 0.55, 1.20)

    # Inject extreme outliers in price_per_sqft
    outlier_idx = rng.choice(n, size=8, replace=False)
    pps[outlier_idx[:4]] *= rng.uniform(3.5, 5.0, size=4)
    pps[outlier_idx[4:]] *= rng.uniform(0.05, 0.15, size=4)

    price = area * pps

    df = pd.DataFrame(
        {
            "city": city,
            "area_sqft": area,
            "price": np.round(price, 0),
            "price_per_sqft": np.round(pps, 2),
            "age_years": age,
            "condition": condition,
            "sale_date": sale_date,
        }
    )

    # Missing area values (~8%)
    miss_idx = rng.choice(n, size=int(0.08 * n), replace=False)
    df.loc[miss_idx, "area_sqft"] = np.nan

    # Age groups for task 7
    df["age_group"] = pd.cut(
        df["age_years"],
        bins=[-0.1, 5, 15, 30, 100],
        labels=["0-5", "6-15", "16-30", "31+"],
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline clean: median-fill area_sqft; keep outliers for later comparison."""
    out = df.copy()
    out["area_sqft"] = out["area_sqft"].fillna(out["area_sqft"].median())
    out["sale_date"] = pd.to_datetime(out["sale_date"])
    # Recompute price_per_sqft where area was filled (optional consistency)
    out["price_per_sqft"] = (out["price"] / out["area_sqft"]).round(2)
    return out


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = generate_raw()
    cleaned = clean(raw)

    raw.to_csv(RAW_PATH, index=False)
    cleaned.to_csv(CLEAN_PATH, index=False)

    print(f"Wrote raw:     {RAW_PATH}  shape={raw.shape}")
    print(f"Wrote cleaned: {CLEAN_PATH}  shape={cleaned.shape}")
    print(f"Missing area in raw: {raw['area_sqft'].isna().sum()}")
    print(f"Missing area in cleaned: {cleaned['area_sqft'].isna().sum()}")
    print("\nCity counts:")
    print(cleaned["city"].value_counts())


if __name__ == "__main__":
    main()
