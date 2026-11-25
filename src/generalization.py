"""Do the results hold on start and goal pairs the training data never contained?

Run:  python src/generalization.py            (reads data/expert_clean.csv and results/trips)

The 200 test trips are drawn from the same maze cells as the 500 training trips, so a fair question is
whether a policy is only repeating routes it was taught. The maze has 26 open cells, which gives 650
ordered start and goal pairs. The 500 training trips cover 359 of them. That leaves part of the test set
on pairs the training data never contained, and this script splits the test trips on that line and scores
both halves. Nothing is retrained: the split uses the start and goal cells already stored in each trip log.

A trip counts as seen when its (start cell, goal cell) pair appears in at least one training trip. Every
single cell appears in training as both a start and a goal, so an unseen trip is a new route through a
known maze, not a new maze.

Writes results/generalization.csv and prints the table.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import free_cells, make_env, xy_to_cell
from stats_brick9 import SEEDS, SINGLE, dataset_list

SHOW = ["expert_clean"] + SINGLE + [f"{e}_p100" for e in SINGLE]


def training_pairs(env, path):
    """The (start cell, goal cell) pair of every training trip, read from the clean dataset."""
    d = pd.read_csv(path, usecols=["episode", "step", "x", "y", "goal_x", "goal_y"])
    first = d[d.step == 0]
    return {(xy_to_cell(env, (r.x, r.y)), xy_to_cell(env, (r.goal_x, r.goal_y))) for r in first.itertuples()}


def seen_mask(env, trips_dir, train):
    """True for each test trip whose start and goal pair appears in the training trips."""
    t = pd.read_csv(Path(trips_dir) / "expert_clean_seed0.csv").sort_values("trip")
    pairs = [((int(r.start_row), int(r.start_col)), (int(r.goal_row), int(r.goal_col))) for r in t.itertuples()]
    return np.array([p in train for p in pairs]), pairs


def scores(trips_dir, mask):
    rows = []
    for name in dataset_list():
        for seed in range(SEEDS):
            t = pd.read_csv(Path(trips_dir) / f"{name}_seed{seed}.csv").sort_values("trip")
            for group, m in (("seen", mask), ("unseen", ~mask)):
                rows.append({"dataset": name, "seed": seed, "group": group, "trips": int(m.sum()),
                             "success_rate": t.success.values[m].mean() * 100,
                             "collision_rate": t.collided.values[m].mean() * 100})
    return pd.DataFrame(rows)


def summary(per_policy):
    g = per_policy.groupby(["dataset", "group"])[["success_rate", "collision_rate"]].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    out = g.reset_index()
    clean = out[out.dataset == "expert_clean"].set_index("group")
    for metric in ("success_rate", "collision_rate"):
        base = out.group.map(clean[f"{metric}_mean"])
        out[f"{metric}_change"] = out[f"{metric}_mean"] - base
    return out


def wide(out, metric):
    """One row per dataset, the seen and unseen columns side by side."""
    p = out.pivot(index="dataset", columns="group", values=[f"{metric}_mean", f"{metric}_change"])
    p.columns = [f"{'value' if a.endswith('mean') else 'change'}_{b}" for a, b in p.columns]
    return p.reindex(SHOW).round(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", default="results/trips")
    ap.add_argument("--clean-data", default="data/expert_clean.csv")
    args = ap.parse_args()
    env = make_env()
    cells = free_cells(env)
    train = training_pairs(env, args.clean_data)
    mask, pairs = seen_mask(env, args.trips, train)

    print(f"Maze: {len(cells)} open cells, {len(cells) * (len(cells) - 1)} ordered start and goal pairs.")
    print(f"Training: 500 trips covering {len(train)} distinct pairs.")
    print(f"Test: {len(pairs)} trips on {len(set(pairs))} distinct pairs, "
          f"{int(mask.sum())} seen in training and {int((~mask).sum())} never seen.")
    starts = {a for a, _ in train}
    goals = {b for _, b in train}
    print(f"Test trips whose start cell is new: {sum(1 for a, _ in pairs if a not in starts)}; "
          f"whose goal cell is new: {sum(1 for _, b in pairs if b not in goals)}.\n")

    per_policy = scores(args.trips, mask)
    Path("results").mkdir(exist_ok=True)
    per_policy.round(2).to_csv("results/generalization.csv", index=False)
    out = summary(per_policy)
    for metric, title in (("collision_rate", "Trips with a wall contact (%)"),
                          ("success_rate", "Trips that reach the goal (%)")):
        print(f"{title}, and the change against clean BC inside the same group:")
        print(wide(out, metric).to_string(), "\n")
    print("Saved results/generalization.csv")


if __name__ == "__main__":
    main()
