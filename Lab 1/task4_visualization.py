"""
Lab 1 Task 4:
Plot sepal length against sepal width coloured by species.
Compare whether this pair separates classes as well as the petal pair.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

    colors = {"setosa": "tab:blue", "versicolor": "tab:orange", "virginica": "tab:green"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sepal pair
    ax = axes[0]
    for species, group in df.groupby("species"):
        ax.scatter(
            group["sepal length (cm)"],
            group["sepal width (cm)"],
            c=colors[species],
            label=species,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.3,
        )
    ax.set_xlabel("Sepal length (cm)")
    ax.set_ylabel("Sepal width (cm)")
    ax.set_title("Sepal length vs sepal width")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Petal pair for comparison
    ax = axes[1]
    for species, group in df.groupby("species"):
        ax.scatter(
            group["petal length (cm)"],
            group["petal width (cm)"],
            c=colors[species],
            label=species,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.3,
        )
    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_title("Petal length vs petal width")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "task4_sepal_vs_petal.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")

    print("\n=== Comparison ===")
    print(
        "The sepal pair is NOT as good at separating classes as the petal pair."
    )
    print(
        "Justification: In the sepal plot, versicolor and virginica overlap a lot,"
    )
    print(
        "while setosa is only partly separated. In the petal plot, the three species"
    )
    print(
        "form clearer, more distinct clusters with much less overlap, so petal"
    )
    print("features separate classes better.")

    # Show window only when a display is available
    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
