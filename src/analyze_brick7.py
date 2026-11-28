"""Brick 7 analysis: dose response, matched error steps and pairs of errors.

Run after run_brick7.py:   python src/analyze_brick7.py
Reads results/results.csv and data/exposure.csv, writes results/brick7_summary.csv and three figures:
  figures/brick7_dose.png      collision and success rate against the share of corrupted trips
  figures/brick7_matched.png   collision change when every error has about 2,200 error steps
  figures/brick7_pairs.png     each pair of errors against the sum of its two single error damages
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SINGLE = ["wrong_left", "wrong_right", "speed_fast", "delayed_turn", "wall_hugging", "missing_recovery"]
LABEL = {"wrong_left": "Wrong left turn", "wrong_right": "Wrong right turn", "speed_fast": "Speed too fast",
         "delayed_turn": "Delayed turn", "wall_hugging": "Wall hugging", "missing_recovery": "Missing recovery"}
COLOR = dict(zip(SINGLE, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
PAIRS = [("delayed_turn", "missing_recovery"), ("delayed_turn", "wall_hugging"), ("missing_recovery", "speed_fast")]
MATCHED = {"wrong_left": "wrong_left", "wrong_right": "wrong_right", "missing_recovery": "missing_recovery",
           "delayed_turn": "delayed_turn_n20", "speed_fast": "speed_fast_n17", "wall_hugging": "wall_hugging_n11"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = ["success_rate", "collision_rate", "path_efficiency", "time_to_goal"]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def summary():
    r = pd.read_csv("results/results.csv")
    r = r[r.dataset != "expert_driver"].copy()
    # Every dataset this analysis adds (50% and 100% doses, matched cuts, pairs, mixed) has five seeds.
    # Clean BC and the six errors at 20% later gained seeds 5 to 9 for the main comparison; here they are
    # held to the same five seeds so each comparison meets seed s with seed s.
    r = r[r.seed < 5]
    r[["success_rate", "collision_rate"]] *= 100
    s = r.groupby("dataset")[METRICS].agg(["mean", "std"])
    s.columns = [f"{m}_{a}" for m, a in s.columns]
    s.insert(0, "seeds", r.groupby("dataset").size())
    exp = pd.read_csv("data/exposure.csv").set_index("error")
    s["error_steps"] = exp.active_steps.reindex(s.index)
    s["flawed_trips"] = exp.flawed_trips.reindex(s.index)
    clean = s.loc["expert_clean"]
    for m, worse in (("collision_rate", 1), ("success_rate", -1)):
        s[f"{m}_change"] = s[f"{m}_mean"] - clean[f"{m}_mean"]
        bar = 2 * np.maximum(s[f"{m}_std"], clean[f"{m}_std"])
        s[f"{m}_passes_rule"] = s[f"{m}_change"].abs() > bar
    return s


def interaction(s, a, b, metric="collision_rate"):
    """I = m(A+B) - m(A) - m(B) + m(clean); its standard error from the four seed spreads."""
    names = [f"{a}_and_{b}", a, b, "expert_clean"]
    mean = [s.loc[n, f"{metric}_mean"] for n in names]
    value = mean[0] - mean[1] - mean[2] + mean[3]
    se = np.sqrt(sum(s.loc[n, f"{metric}_std"] ** 2 / s.loc[n, "seeds"] for n in names))
    return value, se


def fig_dose(s, out):
    share = [0, 20, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, title in ((axes[0], "collision_rate", "Trips with a wall contact (%)"),
                              (axes[1], "success_rate", "Trips that reach the goal (%)")):
        for e in SINGLE:
            names = ["expert_clean", e, f"{e}_p50", f"{e}_p100"]
            y = [s.loc[n, f"{metric}_mean"] for n in names]
            sd = [s.loc[n, f"{metric}_std"] for n in names]
            ax.errorbar(share, y, yerr=sd, color=COLOR[e], lw=2, marker="o", ms=6, capsize=3,
                        mec="white", mew=1.5, label=LABEL[e])
        ax.set_xticks(share)
        ax.set_xticklabels([f"{x}%" for x in share])
        ax.set_xlabel("Share of training trips driven by the flawed driver", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        style(ax)
    # direct labels at the right end of the collision panel, nudged apart
    ends = sorted(((s.loc[f"{e}_p100", "collision_rate_mean"], e) for e in SINGLE))
    placed = []
    for y, e in ends:
        y = max(y, placed[-1] + 3.6) if placed else y
        placed.append(y)
        axes[0].annotate(LABEL[e], (100, s.loc[f"{e}_p100", "collision_rate_mean"]), xytext=(106, y),
                         textcoords="data", fontsize=8.5, color=INK, va="center")
    axes[0].set_xlim(-5, 140)
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("Dose response: mean and seed spread over five seeds, 200 test trips each",
                 x=0.01, ha="left", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def fig_matched(s, out):
    rows = []
    for e in SINGLE:
        n = MATCHED[e]
        rows.append((e, s.loc[n, "collision_rate_change"], s.loc[n, "collision_rate_std"], int(s.loc[n, "error_steps"]),
                     bool(s.loc[n, "collision_rate_passes_rule"])))
    rows.sort(key=lambda r: r[1])
    clean_sd = s.loc["expert_clean", "collision_rate_std"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (e, d, sd, steps, ok) in enumerate(rows):
        ax.errorbar(d, i, xerr=sd, fmt="o", ms=8, color="#2a78d6" if ok else "#9dbfe9", capsize=3,
                    mec="white", mew=1.5, lw=2)
        ax.annotate(f"{d:+.1f}", (d, i), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvspan(-2 * clean_sd, 2 * clean_sd, color=GRID, alpha=0.6, lw=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{LABEL[e]}  ({steps:,} steps)" for e, _, _, steps, _ in rows], fontsize=9, color=INK)
    ax.set_xlabel("Change in trips with a wall contact vs clean BC (percentage points)", color=MUTED, fontsize=9)
    ax.set_title("Same amount of wrong data: about 2,200 error steps each", loc="left", color=INK, fontsize=11)
    ax.text(0.99, 0.02, "grey band: within twice the clean seed spread\ndark dot: passes our rule",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def fig_pairs(s, out):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels, x = [], np.arange(len(PAIRS))
    w = 0.36
    for i, (a, b) in enumerate(PAIRS):
        add = s.loc[a, "collision_rate_change"] + s.loc[b, "collision_rate_change"]
        obs = s.loc[f"{a}_and_{b}", "collision_rate_change"]
        sd = s.loc[f"{a}_and_{b}", "collision_rate_std"]
        ax.bar(i - w / 2, add, w, color="#c9c8c2", edgecolor="white", linewidth=2,
               label="Sum of the two single damages" if i == 0 else None)
        ax.bar(i + w / 2, obs, w, color="#2a78d6", edgecolor="white", linewidth=2, yerr=sd, capsize=3,
               error_kw=dict(ecolor=MUTED, lw=1.2), label="Both errors on the same trips" if i == 0 else None)
        ax.annotate(f"{add:+.1f}", (i - w / 2, add), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
        ax.annotate(f"{obs:+.1f}", (i + w / 2, obs + sd), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
        labels.append(f"{LABEL[a]}\n+ {LABEL[b].lower()}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel("Change in wall contact vs clean\n(percentage points)", color=MUTED, fontsize=9)
    ax.set_title("Pairs of errors: damage of the pair against the sum of its parts", loc="left", color=INK,
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    s = summary()
    Path("results").mkdir(exist_ok=True)
    s.round(4).to_csv("results/brick7_summary.csv")
    print("Interaction on collision rate, I = m(A+B) - m(A) - m(B) + m(clean), in points (standard error):")
    for a, b in PAIRS:
        v, se = interaction(s, a, b)
        print(f"  {a} + {b}: {v:+.1f} ({se:.1f})")
    mixed = s.loc["mixed", "collision_rate_mean"]
    avg = np.mean([s.loc[e, "collision_rate_mean"] for e in SINGLE])
    se = np.sqrt(s.loc["mixed", "collision_rate_std"] ** 2 / 5
                 + sum(s.loc[e, "collision_rate_std"] ** 2 / 5 for e in SINGLE) / len(SINGLE) ** 2)
    print(f"Mixed dataset: {mixed:.1f}% vs {avg:.1f}% for the average single error dataset: "
          f"{mixed - avg:+.1f} ({se:.1f})")
    Path("figures").mkdir(exist_ok=True)
    fig_dose(s, "figures/brick7_dose.png")
    fig_matched(s, "figures/brick7_matched.png")
    fig_pairs(s, "figures/brick7_pairs.png")
    print("Saved results/brick7_summary.csv and figures/brick7_dose.png, brick7_matched.png, brick7_pairs.png")


if __name__ == "__main__":
    main()
