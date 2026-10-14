"""
Lab 2.1 Task 7:
Create a seaborn pair plot of four Wine features coloured by cultivar
and write three observations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    wine = load_wine(as_frame=True)
    df = wine.frame
    df["cultivar"] = df["target"].map(lambda t: f"cultivar_{t}")

    # Four informative chemical features
    features = [
        "alcohol",
        "malic_acid",
        "flavanoids",
        "color_intensity",
    ]
    plot_df = df[features + ["cultivar"]]

    sns.set_theme(style="ticks")
    g = sns.pairplot(
        plot_df,
        hue="cultivar",
        diag_kind="hist",
        corner=False,
        plot_kws={"alpha": 0.7, "s": 30, "edgecolor": "none"},
    )
    g.fig.suptitle("Wine features coloured by cultivar", y=1.02)

    out_path = OUTPUT_DIR / "task7_wine_pairplot.png"
    g.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved pair plot: {out_path}")

    print("\n=== Three observations ===")
    print(
        "1. Cultivars form visible clusters in several feature pairs"
        " (especially involving flavanoids / color intensity)."
    )
    print(
        "2. Flavanoids separate classes particularly well: one cultivar tends"
        " to have clearly higher flavanoid values than another."
    )
    print(
        "3. Some pairs still overlap (e.g. alcohol alone is less clean), so no"
        " single feature is perfect, but combinations improve separation."
    )

    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
