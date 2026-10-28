"""
Lab 3 Task 3 — Missing data:
Fill area_sqft with the mean area of each city (groupby + transform)
instead of a global median. Compare with median fill and show the
vertical-stripe effect in a scatter plot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_PATH = DATA_DIR / "housing_raw.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)

    print(f"Missing area_sqft in raw: {raw['area_sqft'].isna().sum()}")
    missing_mask = raw["area_sqft"].isna()

    # (A) Global median fill
    median_fill = raw.copy()
    global_median = median_fill["area_sqft"].median()
    median_fill["area_sqft"] = median_fill["area_sqft"].fillna(global_median)
    print(f"Global median area: {global_median:.2f}")

    # (B) City-wise mean fill via groupby + transform
    city_mean_fill = raw.copy()
    city_means = city_mean_fill.groupby("city")["area_sqft"].transform("mean")
    city_mean_fill["area_sqft"] = city_mean_fill["area_sqft"].fillna(city_means)
    print("\nCity mean areas used for fill:")
    print(raw.groupby("city")["area_sqft"].mean().round(2))

    print("\nSample of originally-missing rows after each fill:")
    compare = pd.DataFrame(
        {
            "city": raw.loc[missing_mask, "city"].values,
            "median_fill": median_fill.loc[missing_mask, "area_sqft"].values,
            "city_mean_fill": city_mean_fill.loc[missing_mask, "area_sqft"].values,
        }
    )
    print(compare.head(10).to_string(index=False))

    # Scatter: price vs area — median fill creates a vertical stripe at one x
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].scatter(
        median_fill.loc[~missing_mask, "area_sqft"],
        median_fill.loc[~missing_mask, "price"],
        alpha=0.45,
        s=18,
        label="observed",
    )
    axes[0].scatter(
        median_fill.loc[missing_mask, "area_sqft"],
        median_fill.loc[missing_mask, "price"],
        alpha=0.9,
        s=40,
        c="crimson",
        label="median-filled",
        edgecolors="k",
        linewidths=0.3,
    )
    axes[0].set_title("Median fill (vertical stripe)")
    axes[0].set_xlabel("area_sqft")
    axes[0].set_ylabel("price")
    axes[0].legend()

    axes[1].scatter(
        city_mean_fill.loc[~missing_mask, "area_sqft"],
        city_mean_fill.loc[~missing_mask, "price"],
        alpha=0.45,
        s=18,
        label="observed",
    )
    axes[1].scatter(
        city_mean_fill.loc[missing_mask, "area_sqft"],
        city_mean_fill.loc[missing_mask, "price"],
        alpha=0.9,
        s=40,
        c="darkorange",
        label="city-mean-filled",
        edgecolors="k",
        linewidths=0.3,
    )
    axes[1].set_title("City-mean fill (groupby + transform)")
    axes[1].set_xlabel("area_sqft")
    axes[1].legend()

    fig.suptitle("Missing area_sqft: median vs city-mean imputation")
    fig.tight_layout()
    out = OUT_DIR / "task3_missing_area_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {out}")
    print(
        "Comment: median fill stacks all imputed points on one x-value "
        "(vertical stripe). City-mean fill spreads them by city, which is "
        "usually more realistic."
    )


if __name__ == "__main__":
    main()
