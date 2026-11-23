"""Step 7 - figures for the README (reads results/table_A_default_all_questions.csv)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .run_experiments import RES

STYLE = {"tfidf": ("tab:gray", "--", "o"), "bm25": ("tab:blue", "-", "s"),
         "bm25+stem": ("tab:green", "-", "^"), "bm25+stem+ov25": ("tab:red", "-", "D")}


def main():
    df = pd.read_csv(RES / "table_A_default_all_questions.csv")
    df = df[df.method != "random"]
    panels = [("hit@1", "Hit@1 (top-1 chunk is right)"),
              ("hit@3", "Hit@3 (right chunk in top 3)"),
              ("hit@1000w", "Hit within a 1000-word context budget"),
              ("mrr@10", "MRR@10")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, (metric, title) in zip(axes, panels):
        for name, (c, ls, m) in STYLE.items():
            d = df[df.method == name].sort_values("chunk_size")
            ax.plot(d.chunk_size, d[metric], color=c, ls=ls, marker=m, label=name)
        ax.set_xscale("log")
        ax.set_xticks([100, 200, 300, 500, 1000])
        ax.set_xticklabels(["100", "200", "300", "500", "1000"])
        for x in (200, 500, 1000):
            ax.axvline(x, color="k", alpha=.08, lw=6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("chunk size (words)")
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("score (103 questions)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Effect of chunk size on retrieval quality (default parameters, all questions)")
    fig.tight_layout()
    fig.savefig(RES / "fig_chunk_size_effect.png", dpi=140)

    # per-category heat-table figure for the three required sizes
    cat = pd.read_csv(RES / "by_category.csv")
    cat = cat[cat.method == "bm25+stem+ov25"]
    piv = cat.pivot(index="category", columns="chunk_size", values="hit@1")
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.2))
    im = ax2.imshow(piv.values, vmin=0, vmax=1, cmap="YlGn")
    ax2.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax2.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax2.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center")
    ax2.set_xlabel("chunk size (words)")
    ax2.set_title("Hit@1 by question category (bm25+stem+ov25)", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(RES / "fig_by_category.png", dpi=140)
    print("figures saved")


if __name__ == "__main__":
    main()
