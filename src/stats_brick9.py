"""Brick 9 statistics: permutation tests, paired bootstrap intervals and per seed interactions.

Run:  python src/stats_brick9.py                    (reads results/trips)
      python src/stats_brick9.py --trips PATH --draws 10000

Three ways of measuring the same uncertainty, each with its own weakness:

Seed counts. Clean BC and the six errors at 20% have ten seeds (0 to 9); every other dataset has five
(0 to 4). Every comparison uses the seeds both sides share, so a five seed dataset meets clean BC's
seeds 0 to 4 and seed s always meets seed s.

  permutation   The unpaired test. The n clean policies and n flawed ones are pooled and split again
                in every way, with the test trips held fixed: C(10,5) = 252 splits at five seeds,
                C(20,10) = 184,756 at ten. The p value is the share of splits whose gap is at least as
                large as the one we saw. The smallest p possible is 2/252 = 0.0079 at five seeds
                (0.048 after Holm for six errors) and 2/184,756 at ten. These p values feed the main
                Holm correction. The test treats the 2n runs as exchangeable, which the shared seed
                numbers argue against, so the paired test below is reported beside it.
  paired perm.  The matched version. Seed s of a dataset and seed s of clean BC are a pair, and the
                sign of each pair's difference is flipped in every way: 2^5 = 32 assignments at five
                seeds, 2^10 = 1,024 at ten. Its floor is 2/32 = 0.0625 at five seeds (0.375 after Holm,
                so it cannot show significance there however large the effect) and 2/1,024 = 0.00195
                at ten (0.0117 after Holm). It is exact under the assumption that, with no effect, each
                pair's difference is as likely to come out positive as negative (sign symmetry).
  bootstrap     The interval. Each draw picks 200 trip positions and n seed numbers with replacement
                and applies the same draw to the dataset and to clean BC, so both the trips and the
                seeds stay paired. Its p value comes from where the interval sits, which is not a null
                test, so it is reported but not corrected.
  seed t        The matched check. Seed s of a dataset against seed s of clean BC, n differences,
                a t interval.

The solid column of brick9_changes.csv marks a change whose bootstrap interval and seed interval both
exclude zero. The report asks one more thing of a solid finding: an unpaired Holm p below 0.05.

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
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SINGLE = ["wrong_left", "wrong_right", "speed_fast", "delayed_turn", "wall_hugging", "missing_recovery"]
PAIRS = [("delayed_turn", "missing_recovery"), ("delayed_turn", "wall_hugging"), ("missing_recovery", "speed_fast")]
MAIN = ["expert_clean"] + SINGLE
KEY = ["start_row", "start_col", "goal_row", "goal_col"]
METRICS = ["collision", "success", "efficiency_all"]


@lru_cache(maxsize=None)
def splits(n):
    """Every way of splitting 2n policies into two groups of n, for the unpaired permutation test."""
    left = np.array(list(combinations(range(2 * n), n)))
    right = np.array([[i for i in range(2 * n) if i not in set(c)] for c in left])
    return left, right


@lru_cache(maxsize=None)
def sign_flips(n):
    """Every way of flipping the label inside each of the n seed pairs, for the paired permutation test."""
    return np.array(list(product([1, -1], repeat=n)))


def available_seeds(trips_dir, name):
    """Seed numbers this dataset has a scored trip file for, in order. They must run 0, 1, 2 and on with no
    gap; a missing seed in the middle stops the script rather than being skipped without notice."""
    found = sorted(int(p.stem.rsplit("_seed", 1)[1]) for p in Path(trips_dir).glob(f"{name}_seed*.csv")
                   if p.stem.rsplit("_seed", 1)[0] == name and p.stem.rsplit("_seed", 1)[1].isdigit())
    if found != list(range(len(found))):
        raise FileNotFoundError(f"{name}: seed files {found} have a gap; expected 0 to {max(found)}")
    return found


def matched(a, b):
    """Cut two policy arrays to the seeds they share, so seed s always meets seed s."""
    n = min(len(a), len(b))
    return a[:n], b[:n]


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
        for seed in available_seeds(trips_dir, name):
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
    a, b = matched(a, b)
    left, right = splits(len(a))
    pool = np.concatenate([a.mean(1), b.mean(1)])
    observed = a.mean() - b.mean()
    gaps = pool[left].mean(1) - pool[right].mean(1)
    return float((np.abs(gaps) >= abs(observed) - 1e-12).mean())


def paired_permutation_p(a, b):
    """Exact two sided sign flip test over the n matched seed pairs.

    The same seed numbers set the starting weights and the batch order in every dataset, so seed s of
    a flawed dataset and seed s of clean BC are a matched pair rather than two unrelated draws. Under
    the null each pair's difference is as likely to be positive as negative (sign symmetry), so each
    of the n differences could have come out with either sign, which gives 2^n assignments: 32 at
    five seeds, 1,024 at ten. The p value is the share of those whose mean difference is at least as
    large in size as the one we saw. The test is exact under that sign symmetry assumption.

    The smallest p this can return is 2/2^n: 0.0625 at five seeds (0.375 after Holm for six errors,
    so it cannot reach significance there however large the effect) and 0.00195 at ten (0.0117).
    """
    a, b = matched(a, b)
    d = a.mean(1) - b.mean(1)
    observed = d.mean()
    means = (sign_flips(len(d)) * d).mean(1)
    return float((np.abs(means) >= abs(observed) - 1e-12).mean())


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
        clean_all = data[metric]["expert_clean"]
        for name, arr in data[metric].items():
            if name == "expert_clean":
                continue
            arr, clean = matched(arr, clean_all)                    # seed s of each, as far as both go
            n = len(arr)
            seed_idx = rng.integers(0, n, size=(draws, n))          # one seed draw for both: seed s meets seed s
            d = (boot_mean(arr, trip_idx, seed_idx) - boot_mean(clean, trip_idx, seed_idx)) * 100
            lo, hi = interval(d)
            per_seed = (arr.mean(1) - clean.mean(1)) * 100          # seed s against seed s of clean BC
            half = stats.t.ppf(0.975, n - 1) * per_seed.std(ddof=1) / np.sqrt(n)
            rows.append({"metric": metric, "dataset": name, "seeds": n, "clean_mean": clean.mean() * 100,
                         "dataset_mean": arr.mean() * 100, "change": (arr.mean() - clean.mean()) * 100,
                         "ci_low": lo, "ci_high": hi, "p_permutation": permutation_p(arr, clean),
                         "p_paired_permutation": paired_permutation_p(arr, clean),
                         "p_bootstrap": p_value(d),
                         "seed_ci_low": per_seed.mean() - half, "seed_ci_high": per_seed.mean() + half})
    df = pd.DataFrame(rows)
    df["holm_p_six_errors"] = np.nan
    df["holm_p_paired"] = np.nan
    for metric in METRICS:
        m = (df.metric == metric) & df.dataset.isin(SINGLE)
        df.loc[m, "holm_p_six_errors"] = holm(df.loc[m, "p_permutation"].values)
        df.loc[m, "holm_p_paired"] = holm(df.loc[m, "p_paired_permutation"].values)
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
        n = int(ch[(ch.metric == metric) & (ch.dataset == order[0] + suffix)].seeds.iloc[0])
        title = f"{title} ({n} seeds)"
        for i, e in enumerate(order):
            r = ch[(ch.metric == metric) & (ch.dataset == e + suffix)].iloc[0]
            # dark: both intervals exclude zero (and, for collisions at 20%, Holm p below 0.05)
            solid = bool(r.solid) and (r.holm_p_six_errors < 0.05 if metric == "collision" else True)
            ax.errorbar(r.change, i, xerr=[[r.change - r.ci_low], [r.ci_high - r.change]], fmt="o", ms=8,
                        color=BLUE if solid else PALE, capsize=3, mec="white", mew=1.5, lw=2)
            ax.annotate(f"{r.change:+.2f}", (r.change, i), xytext=(0, 10), textcoords="offset points",
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
    fig.suptitle("Change against clean BC with 95% bootstrap intervals (same 200 test trips; seeds per panel)",
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
    rows = []
    for a, b in PAIRS:
        parts = [c[f"{a}_and_{b}"], c[a], c[b], c["expert_clean"]]
        n = min(len(x) for x in parts)                          # a pair dataset may have fewer seeds than a single
        ab, sa, sb, clean = (x[:n] for x in parts)
        seed_idx = rng.integers(0, n, size=(draws, n))
        ps = per_seed_interaction(sa, sb, ab, clean)
        boot = (boot_mean(ab, trip_idx, seed_idx) - boot_mean(sa, trip_idx, seed_idx)
                - boot_mean(sb, trip_idx, seed_idx) + boot_mean(clean, trip_idx, seed_idx)) * 100
        rows.append(summarize_seedwise(f"{a} + {b}", ps, boot))
    n = min([len(c["mixed"])] + [len(c[e]) for e in SINGLE])
    mixed = c["mixed"][:n]
    singles = np.mean([c[e][:n] for e in SINGLE], axis=0)       # (seeds, trips): average of the six single datasets
    seed_idx = rng.integers(0, n, size=(draws, n))
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
        for s in range(len(data["success"][name])):
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
          f"  ({args.draws} bootstrap draws; exact permutation tests)\n")
    cols = ["dataset", "seeds", "dataset_mean", "change", "ci_low", "ci_high", "seed_ci_low", "seed_ci_high",
            "p_permutation", "holm_p_six_errors", "p_paired_permutation", "holm_p_paired", "solid"]
    titles = {"collision": "collision rate", "success": "success rate",
              "efficiency_all": "efficiency over all trips (failures count as zero)"}
    for metric in METRICS:
        print(f"Change in {titles[metric]} against clean BC (points), 95% intervals:")
        print(ch[ch.metric == metric][cols].round(3).to_string(index=False), "\n")
    for n in sorted({int(v) for v in ch.seeds.unique()}):
        unpaired = 2 / len(splits(n)[0])
        paired = 2 / len(sign_flips(n))
        print(f"With {n} seeds against {n}: unpaired test floor {unpaired:.5f} raw, {min(1, 6 * unpaired):.4f} "
              f"after Holm for six errors; paired test floor {paired:.5f} raw, "
              f"{min(1, 6 * paired):.4f} after Holm.")
    print()
    print("Interaction on collision rate (points):")
    print(ix.round(2).to_string(index=False))
    print("\nSaved results/brick9_changes.csv, brick9_interactions.csv, brick9_seed_results.csv "
          "and figures/brick9_intervals.png")


if __name__ == "__main__":
    main()
