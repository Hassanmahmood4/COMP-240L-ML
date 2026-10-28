"""
Lab 3 Task 6 — Date analysis:
Number of sales and mean price per month of sale; line chart; comment on trend.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
CLEAN_PATH = DATA_DIR / "housing_cleaned.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_PATH, parse_dates=["sale_date"])
    df["sale_month"] = df["sale_date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("sale_month")
        .agg(n_sales=("price", "size"), mean_price=("price", "mean"))
        .sort_index()
    )
    monthly["mean_price"] = monthly["mean_price"].round(0)

    print("=== Sales and mean price per month ===")
    print(monthly.to_string())

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(
        monthly.index,
        monthly["n_sales"],
        marker="o",
        color="steelblue",
        label="Number of sales",
    )
    ax1.set_xlabel("Month of sale")
    ax1.set_ylabel("Number of sales", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(
        monthly.index,
        monthly["mean_price"],
        marker="s",
        color="darkorange",
        label="Mean price",
    )
    ax2.set_ylabel("Mean price (PKR)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Monthly sales volume and mean price")
    fig.autofmt_xdate()
    fig.tight_layout()

    out = OUT_DIR / "task6_monthly_sales.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot: {out}")

    # Simple trend comment from correlation of month index vs mean price
    x = range(len(monthly))
    corr = pd.Series(list(x)).corr(monthly["mean_price"].reset_index(drop=True))
    print("\n=== Trend comment ===")
    if abs(corr) < 0.15:
        print(
            "No strong long-term trend: mean monthly price wanders without a "
            f"clear rise/fall (corr≈{corr:.2f}). Short-run ups and downs are visible."
        )
    elif corr > 0:
        print(
            f"Mild upward tendency in mean price over time (corr≈{corr:.2f}), "
            "though month-to-month noise remains."
        )
    else:
        print(
            f"Mild downward tendency in mean price over time (corr≈{corr:.2f}), "
            "with noticeable month-to-month variation."
        )


if __name__ == "__main__":
    main()
