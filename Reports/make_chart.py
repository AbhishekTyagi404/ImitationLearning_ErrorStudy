"""Figure 3 of the final report: collision and success rate for clean BC and the six errors at 20%.

Run from the repository root, after stats_brick9.py:
    python Reports/make_chart.py

Reads results/results.csv (one row per trained policy) and results/brick9_changes.csv (the statistical
tests), and writes Reports/figures/results_chart.pdf. Every number is read from those files, so the figure
cannot drift from the tables. Bars are dark for an error that clears both statistical bars of the report
(both 95% intervals exclude zero, and the permutation p survives the Holm correction for six errors) and
pale otherwise. Clean BC and the six errors at 20% have ten seeds each.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["expert_clean", "delayed_turn", "missing_recovery", "wall_hugging", "speed_fast", "wrong_right",
         "wrong_left"]
NAME = {"expert_clean": "clean", "delayed_turn": "delayed turn", "missing_recovery": "missing recovery",
        "wall_hugging": "wall hugging", "speed_fast": "speed too fast", "wrong_right": "wrong right turn",
        "wrong_left": "wrong left turn"}
INK, INK2, GRID, DARK, PALE, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#9cc1ec", "#a8a7a2"


def main():
    r = pd.read_csv("results/results.csv")
    r = r[r.dataset.isin(ORDER)].copy()
    r[["success_rate", "collision_rate"]] *= 100
    s = r.groupby("dataset")[["success_rate", "collision_rate"]].agg(["mean", "std"]).reindex(ORDER)
    seeds = r.groupby("dataset").size().reindex(ORDER)

    c = pd.read_csv("results/brick9_changes.csv")
    c = c[c.metric == "collision"].set_index("dataset")
    solid = {d: bool(c.loc[d, "solid"]) and c.loc[d, "holm_p_six_errors"] < 0.05 for d in ORDER[1:]}

    rows = ORDER[::-1]
    y = np.arange(len(rows))
    colors = [GREY if d == "expert_clean" else (DARK if solid[d] else PALE) for d in rows]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, metric, title, lim in ((axes[0], "collision_rate", "Collision rate (% of test trips)", (0, 72)),
                                   (axes[1], "success_rate", "Success rate (% of test trips)", (78, 100))):
        m = s[(metric, "mean")].reindex(rows).values
        sd = s[(metric, "std")].reindex(rows).values
        ax.axvline(s.loc["expert_clean", (metric, "mean")], color=INK2, lw=1, ls=(0, (3, 2)), zorder=1)
        if metric == "collision_rate":      # magnitude from zero: bars
            ax.barh(y, m, height=0.55, color=colors, zorder=2)
        else:                                # small differences near 90%: dots, no truncated bars
            ax.scatter(m, y, s=36, color=colors, zorder=3, edgecolor="white", linewidth=1)
        ax.errorbar(m, y, xerr=sd, fmt="none", ecolor=INK, elinewidth=1, capsize=2.5, zorder=4)
        for yi, mi, si in zip(y, m, sd):
            ax.text(mi + si + (1.0 if metric == "collision_rate" else 0.4), yi, f"{mi:.2f}", va="center",
                    ha="left", fontsize=8, color=INK2, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.set_xlim(*lim)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([NAME[d] for d in rows])
    out = Path("Reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "results_chart.pdf", bbox_inches="tight")
    print(f"Saved {out / 'results_chart.pdf'}  (seeds per dataset: {dict(seeds)})")
    print("Dark bars:", [NAME[d] for d in ORDER[1:] if solid[d]])


if __name__ == "__main__":
    main()
