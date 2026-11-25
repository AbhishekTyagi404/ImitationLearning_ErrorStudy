"""Brick 9 statistics: paired bootstrap intervals, Holm correction and per seed interactions.

Run:  python src/stats_brick9.py                    (reads results/trips)
      python src/stats_brick9.py --trips PATH --draws 10000

Every policy drives the same 200 test trips, so a bootstrap draw picks 200 trip positions once and
applies them to every dataset (the trips stay paired). Training seeds are the second source of
variation: each draw also picks five seeds with replacement. For a comparison with clean BC the two
datasets draw their seeds independently, which assumes nothing about how seeds line up. For an
interaction the four datasets share the seed draw, since I is computed seed by seed.

Writes:
  results/brick9_seed_results.csv     success and collision rate of every policy
  results/brick9_changes.csv          change against clean BC with 95% interval and p value
  results/brick9_interactions.csv     interaction of each pair, per seed and bootstrap, and mixed vs singles
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SINGLE = ["wrong_left", "wrong_right", "speed_fast", "delayed_turn", "wall_hugging", "missing_recovery"]
PAIRS = [("delayed_turn", "missing_recovery"), ("delayed_turn", "wall_hugging"), ("missing_recovery", "speed_fast")]
SEEDS = 5
KEY = ["start_row", "start_col", "goal_row", "goal_col"]
METRICS = {"collision": "collided", "success": "success"}


def dataset_list():
    names = ["expert_clean"] + SINGLE
    names += [f"{e}_p{p}" for p in (50, 100) for e in SINGLE]
    names += ["delayed_turn_n20", "speed_fast_n17", "wall_hugging_n11"]
    names += [f"{a}_and_{b}" for a, b in PAIRS] + ["mixed"]
    return names


def load(trips_dir):
    """Returns {metric: {dataset: array (seeds, trips) of 0/1}} after checking the trips line up."""
    out = {m: {} for m in METRICS}
    ref = None
    for name in dataset_list():
        rows = {m: [] for m in METRICS}
        for seed in range(SEEDS):
            t = pd.read_csv(Path(trips_dir) / f"{name}_seed{seed}.csv").sort_values("trip").reset_index(drop=True)
            if ref is None:
                ref = t[KEY].copy()
            assert len(t) == len(ref) and (t[KEY].values == ref.values).all(), f"trips differ in {name} seed {seed}"
            for m, col in METRICS.items():
                rows[m].append(t[col].astype(float).values)
        for m in METRICS:
            out[m][name] = np.array(rows[m])
    return out


def draw_indices(rng, draws, n_trips):
    trips = rng.integers(0, n_trips, size=(draws, n_trips))
    return trips


def boot_mean(arr, trip_idx, seed_idx):
    """Mean over the drawn seeds and trips for every draw: shape (draws,)."""
    return arr[seed_idx[:, :, None], trip_idx[:, None, :]].mean(axis=(1, 2))


def interval(x):
    return np.percentile(x, [2.5, 97.5])


def p_value(x):
    """Two sided bootstrap p value for 'the change is zero'; the smallest resolvable value is 2 / draws."""
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
            rows.append({"metric": metric, "dataset": name, "clean_mean": clean.mean() * 100,
                         "dataset_mean": arr.mean() * 100, "change": (arr.mean() - clean.mean()) * 100,
                         "ci_low": lo, "ci_high": hi, "p_value": p_value(d)})
    df = pd.DataFrame(rows)
    df["holm_p_six_errors"] = np.nan
    for metric in METRICS:
        m = (df.metric == metric) & df.dataset.isin(SINGLE)
        df.loc[m, "holm_p_six_errors"] = holm(df.loc[m, "p_value"].values)
    df["interval_excludes_zero"] = (df.ci_low > 0) | (df.ci_high < 0)
    return df


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
            "boot_p": p_value(boot)}


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
                              "collision_rate": data["collision"][name][s].mean() * 100})
    pd.DataFrame(seed_rows).round(2).to_csv("results/brick9_seed_results.csv", index=False)

    ch = changes(data, rng, args.draws)
    ch.round(4).to_csv("results/brick9_changes.csv", index=False)
    ix = interactions(data, rng, args.draws)
    ix.round(3).to_csv("results/brick9_interactions.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"Clean BC: success {data['success']['expert_clean'].mean() * 100:.1f}%, "
          f"collisions {data['collision']['expert_clean'].mean() * 100:.1f}%  ({args.draws} bootstrap draws)\n")
    cols = ["dataset", "dataset_mean", "change", "ci_low", "ci_high", "p_value", "holm_p_six_errors",
            "interval_excludes_zero"]
    for metric in METRICS:
        print(f"Change in {metric} rate against clean BC (percentage points), 95% interval:")
        print(ch[ch.metric == metric][cols].round(3).to_string(index=False), "\n")
    print("Interaction on collision rate (percentage points):")
    print(ix.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
