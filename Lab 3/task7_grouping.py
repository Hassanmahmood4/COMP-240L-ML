"""
Lab 3 Task 7 — Grouping:
For each age_group, mean price_per_sqft and house count; bar chart + comment.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
CLEAN_PATH = DATA_DIR / "housing_cleaned.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH)

    grouped = (
        df.groupby("age_group", observed=True)
        .agg(
            mean_price_per_sqft=("price_per_sqft", "mean"),
            n_houses=("price_per_sqft", "size"),
        )
        .round(2)
    )
    # Ensure sensible age order
    order = ["0-5", "6-15", "16-30", "31+"]
    grouped = grouped.reindex([g for g in order if g in grouped.index])

    print("=== By age_group ===")
    print(grouped.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(grouped.index.astype(str), grouped["mean_price_per_sqft"], color="teal")
    axes[0].set_title("Mean price_per_sqft by age_group")
    axes[0].set_xlabel("age_group")
    axes[0].set_ylabel("Mean price per sqft (PKR)")

    axes[1].bar(grouped.index.astype(str), grouped["n_houses"], color="slateblue")
    axes[1].set_title("Number of houses by age_group")
    axes[1].set_xlabel("age_group")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    out = OUT_DIR / "task7_age_group_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {out}")

    print("\n=== Comment ===")
    print(
        "Newer homes (younger age_group) tend to show a higher mean price per "
        "sqft, while older groups are usually cheaper per unit area. Counts show "
        "how many listings fall in each age band."
    )


if __name__ == "__main__":
    main()
