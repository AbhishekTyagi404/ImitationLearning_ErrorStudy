"""Brick 9 statistics: permutation tests, paired bootstrap intervals and per seed interactions.

Run:  python src/stats_brick9.py                    (reads results/trips)
      python src/stats_brick9.py --trips PATH --draws 10000

Three ways of measuring the same uncertainty, each with its own weakness:

  permutation   The honest test. Five clean policies and five flawed ones are pooled and split again
                in all 252 ways, with the test trips held fixed. The p value is the share of splits
                whose gap is at least as large as the one we saw. With five against five the smallest
                p value possible is 2/252 = 0.0079, so after Holm correction for six errors nothing
                can fall below 0.048. These p values feed the Holm correction.
  bootstrap     The interval. Each draw picks 200 trip positions once and applies them to every
                dataset, so the trips stay paired, and picks five seeds with replacement. Its p value
                comes from where the interval sits, which is not a null test, so it is reported but
                not corrected.
  seed t        The matched check. Seed s of a dataset against seed s of clean BC, five differences,
                a t interval. It uses the shared seed numbers that the bootstrap throws away.

A result counts as solid when the bootstrap interval and the seed interval both exclude zero.

Metrics are the collision rate, the success rate and an efficiency over all 200 trips (the expert's
distance over the policy's distance on a trip that reached the goal, zero on a trip that did not),
which the success conditioned efficiency in evaluate.py cannot see because failed trips drop out of it.

Writes:
  results/brick9_seed_results.csv     success and collision rate of every policy
  results/brick9_changes.csv          change against clean BC with 95% interval and p values
  results/brick9_interactions.csv     interaction of each pair, per seed and bootstrap, and mixed vs singles
  figures/brick9_intervals.png        change against clean BC with 95% intervals for the six errors
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SINGLE = ["wrong_left", "wrong_right", "speed_fast", "delayed_turn", "wall_hugging", "missing_recovery"]
PAIRS = [("delayed_turn", "missing_recovery"), ("delayed_turn", "wall_hugging"), ("missing_recovery", "speed_fast")]
SEEDS = 5
KEY = ["start_row", "start_col", "goal_row", "goal_col"]
METRICS = ["collision", "success", "efficiency_all"]
# every way of splitting ten policies into two groups of five, for the permutation test
LEFT = np.array(list(combinations(range(2 * SEEDS), SEEDS)))
RIGHT = np.array([[i for i in range(2 * SEEDS) if i not in set(c)] for c in LEFT])


def dataset_list():
    names = ["expert_clean"] + SINGLE
    names += [f"{e}_p{p}" for p in (50, 100) for e in SINGLE]
    names += ["delayed_turn_n20", "speed_fast_n17", "wall_hugging_n11"]
    names += [f"{a}_and_{b}" for a, b in PAIRS] + ["mixed"]
    return names


def trip_values(t, metric):
    """One number per test trip, between 0 and 1. Everything downstream turns it into points."""
    if metric == "collision":
        return t.collided.astype(float).values
    if metric == "success":
        return t.success.astype(float).values
    ok = t.success.astype(bool).values                      # efficiency over all trips, failures count as zero
    out = np.zeros(len(t))
    out[ok] = t.expert_travelled.values[ok] / t.travelled.values[ok]
    return out


def load(trips_dir):
    """Returns {metric: {dataset: array (seeds, trips)}} after checking the trips line up."""
    out = {m: {} for m in METRICS}
    ref = None
    for name in dataset_list():
        rows = {m: [] for m in METRICS}
        for seed in range(SEEDS):
            t = pd.read_csv(Path(trips_dir) / f"{name}_seed{seed}.csv").sort_values("trip").reset_index(drop=True)
            if ref is None:
                ref = t[KEY].copy()
            assert len(t) == len(ref) and (t[KEY].values == ref.values).all(), f"trips differ in {name} seed {seed}"
            for m in METRICS:
                rows[m].append(trip_values(t, m))
        for m in METRICS:
            out[m][name] = np.array(rows[m])
    return out


def permutation_p(a, b):
    """Exact two sided permutation test over the ten policies, with the test trips held fixed.

    Under the null the condition label carries no information, so any five of the ten policies could
    have been the flawed group. The p value is the share of the 252 splits whose gap is at least as
    large as the one we saw.
    """
    pool = np.concatenate([a.mean(1), b.mean(1)])
    observed = a.mean() - b.mean()
    gaps = pool[LEFT].mean(1) - pool[RIGHT].mean(1)
    return float((np.abs(gaps) >= abs(observed) - 1e-12).mean())


def draw_indices(rng, draws, n_trips):
    trips = rng.integers(0, n_trips, size=(draws, n_trips))
    return trips


def boot_mean(arr, trip_idx, seed_idx):
    """Mean over the drawn seeds and trips for every draw: shape (draws,)."""
    return arr[seed_idx[:, :, None], trip_idx[:, None, :]].mean(axis=(1, 2))


def interval(x):
    return np.percentile(x, [2.5, 97.5])


def p_value(x):
    """Where the bootstrap interval sits relative to zero. This reads like a p value and is reported as
    one, but it comes from a distribution centred on the observed change, so it is not a null test and
    the Holm correction uses permutation_p instead. The smallest value it can give is 2 / draws."""
    n = len(x)
    return min(1.0, 2 * (min((x <= 0).sum(), (x >= 0).sum()) + 1) / (n + 1))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def changes(data, rng, draws):
    n_trips = data["collision"]["expert_clean"].shape[1]
    trip_idx = draw_indices(rng, draws, n_trips)
    rows = []
    for metric in METRICS:
        clean = data[metric]["expert_clean"]
        clean_seed = rng.integers(0, SEEDS, size=(draws, SEEDS))
        clean_boot = boot_mean(clean, trip_idx, clean_seed)
        for name, arr in data[metric].items():
            if name == "expert_clean":
                continue
            seed_idx = rng.integers(0, SEEDS, size=(draws, SEEDS))
            d = (boot_mean(arr, trip_idx, seed_idx) - clean_boot) * 100
            lo, hi = interval(d)
            per_seed = (arr.mean(1) - clean.mean(1)) * 100          # seed s against seed s of clean BC
            half = stats.t.ppf(0.975, SEEDS - 1) * per_seed.std(ddof=1) / np.sqrt(SEEDS)
            rows.append({"metric": metric, "dataset": name, "clean_mean": clean.mean() * 100,
                         "dataset_mean": arr.mean() * 100, "change": (arr.mean() - clean.mean()) * 100,
                         "ci_low": lo, "ci_high": hi, "p_permutation": permutation_p(arr, clean),
                         "p_bootstrap": p_value(d),
                         "seed_ci_low": per_seed.mean() - half, "seed_ci_high": per_seed.mean() + half})
    df = pd.DataFrame(rows)
    df["holm_p_six_errors"] = np.nan
    for metric in METRICS:
        m = (df.metric == metric) & df.dataset.isin(SINGLE)
        df.loc[m, "holm_p_six_errors"] = holm(df.loc[m, "p_permutation"].values)
    df["interval_excludes_zero"] = (df.ci_low > 0) | (df.ci_high < 0)
    df["seed_interval_excludes_zero"] = (df.seed_ci_low > 0) | (df.seed_ci_high < 0)
    df["solid"] = df.interval_excludes_zero & df.seed_interval_excludes_zero      # both intervals exclude zero
    return df


LABEL = {"wrong_left": "Wrong left turn", "wrong_right": "Wrong right turn", "speed_fast": "Speed too fast",
         "delayed_turn": "Delayed turn", "wall_hugging": "Wall hugging", "missing_recovery": "Missing recovery"}
INK, MUTED, GRID, BLUE, PALE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#9dbfe9"


def draw_intervals(ch, path):
    """Change against clean BC with 95% bootstrap intervals: collisions at 20%, success at 100% corrupted trips."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ch[(ch.metric == "collision") & ch.dataset.isin(SINGLE)].sort_values("change").dataset.tolist()
    panels = (("collision", "", "Change in trips with a wall contact (points)", "Collisions, 20% of trips corrupted"),
              ("success", "_p100", "Change in trips that reach the goal (points)", "Success, 100% of trips corrupted"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (metric, suffix, xlabel, title) in zip(axes, panels):
        for i, e in enumerate(order):
            r = ch[(ch.metric == metric) & (ch.dataset == e + suffix)].iloc[0]
            # dark: both intervals exclude zero (and, for collisions at 20%, Holm p below 0.05)
            solid = bool(r.solid) and (r.holm_p_six_errors < 0.05 if metric == "collision" else True)
            ax.errorbar(r.change, i, xerr=[[r.change - r.ci_low], [r.ci_high - r.change]], fmt="o", ms=8,
                        color=BLUE if solid else PALE, capsize=3, mec="white", mew=1.5, lw=2)
            ax.annotate(f"{r.change:+.1f}", (r.change, i), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([LABEL[e] for e in order], fontsize=9.5, color=INK)
    fig.suptitle("Change against clean BC with 95% bootstrap intervals (five seeds, same 200 test trips)",
                 x=0.01, ha="left", color=MUTED, fontsize=9)
    fig.text(0.99, 0.01, "dark dot: bootstrap and seed intervals both exclude zero (collisions also need Holm p < 0.05)",
             ha="right", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def per_seed_interaction(a, b, ab, clean):
    """I_s for each seed, in percentage points. Arrays are (seeds, trips)."""
    return (ab.mean(1) - a.mean(1) - b.mean(1) + clean.mean(1)) * 100


def summarize_seedwise(label, per_seed, boot):
    t = stats.t.ppf(0.975, len(per_seed) - 1)
    se = per_seed.std(ddof=1) / np.sqrt(len(per_seed))
    lo, hi = interval(boot)
    return {"comparison": label, "per_seed_values": " ".join(f"{v:+.1f}" for v in per_seed),
            "mean": per_seed.mean(), "seed_se": se, "seed_ci_low": per_seed.mean() - t * se,
            "seed_ci_high": per_seed.mean() + t * se, "boot_ci_low": lo, "boot_ci_high": hi,
            "boot_p": p_value(boot), "seed_p": stats.ttest_1samp(per_seed, 0).pvalue}


def interactions(data, rng, draws):
    c = data["collision"]
    n_trips = c["expert_clean"].shape[1]
    trip_idx = draw_indices(rng, draws, n_trips)
    seed_idx = rng.integers(0, SEEDS, size=(draws, SEEDS))
    rows = []
    for a, b in PAIRS:
        ab = c[f"{a}_and_{b}"]
        ps = per_seed_interaction(c[a], c[b], ab, c["expert_clean"])
        boot = (boot_mean(ab, trip_idx, seed_idx) - boot_mean(c[a], trip_idx, seed_idx)
                - boot_mean(c[b], trip_idx, seed_idx) + boot_mean(c["expert_clean"], trip_idx, seed_idx)) * 100
        rows.append(summarize_seedwise(f"{a} + {b}", ps, boot))
    mixed = c["mixed"]
    singles = np.mean([c[e] for e in SINGLE], axis=0)           # (seeds, trips): average of the six single datasets
    ps = (mixed.mean(1) - singles.mean(1)) * 100
    boot = (boot_mean(mixed, trip_idx, seed_idx) - boot_mean(singles, trip_idx, seed_idx)) * 100
    rows.append(summarize_seedwise("mixed minus average of six singles", ps, boot))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", default="results/trips")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    Path("results").mkdir(exist_ok=True)
    data = load(args.trips)

    seed_rows = []
    for name in dataset_list():
        for s in range(SEEDS):
            seed_rows.append({"dataset": name, "seed": s,
                              "success_rate": data["success"][name][s].mean() * 100,
                              "collision_rate": data["collision"][name][s].mean() * 100,
                              "efficiency_all_trips": data["efficiency_all"][name][s].mean() * 100})
    pd.DataFrame(seed_rows).round(2).to_csv("results/brick9_seed_results.csv", index=False)

    ch = changes(data, rng, args.draws)
    ch.round(4).to_csv("results/brick9_changes.csv", index=False)
    ix = interactions(data, rng, args.draws)
    ix.round(3).to_csv("results/brick9_interactions.csv", index=False)
    Path("figures").mkdir(exist_ok=True)
    draw_intervals(ch, "figures/brick9_intervals.png")

    pd.set_option("display.width", 230)
    print(f"Clean BC: success {data['success']['expert_clean'].mean() * 100:.1f}%, "
          f"collisions {data['collision']['expert_clean'].mean() * 100:.1f}%, "
          f"efficiency over all trips {data['efficiency_all']['expert_clean'].mean() * 100:.1f}"
          f"  ({args.draws} bootstrap draws, 252 permutations)\n")
    cols = ["dataset", "dataset_mean", "change", "ci_low", "ci_high", "seed_ci_low", "seed_ci_high",
            "p_permutation", "holm_p_six_errors", "solid"]
    titles = {"collision": "collision rate", "success": "success rate",
              "efficiency_all": "efficiency over all trips (failures count as zero)"}
    for metric in METRICS:
        print(f"Change in {titles[metric]} against clean BC (points), 95% intervals:")
        print(ch[ch.metric == metric][cols].round(3).to_string(index=False), "\n")
    print(f"Smallest p value five against five policies can give: {2 / len(LEFT):.4f}; "
          f"after Holm for six errors: {6 * 2 / len(LEFT):.3f}\n")
    print("Interaction on collision rate (points):")
    print(ix.round(2).to_string(index=False))
    print("\nSaved results/brick9_changes.csv, brick9_interactions.csv, brick9_seed_results.csv "
          "and figures/brick9_intervals.png")


if __name__ == "__main__":
    main()
