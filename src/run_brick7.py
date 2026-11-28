"""Brick 7 runner: trains and scores five seeds for every dataset, skipping models already scored.

Run:  python src/run_brick7.py              (all groups, seeds 0 to 4, in order)
      python src/run_brick7.py --group pairs
      python src/run_brick7.py --group base --seeds 5 9   (five more seeds for clean and the six errors)

Groups, in the order they run:
  base     clean data and the six single errors at 20% (the reference every comparison needs)
  pairs    three pairs of errors on the same 100 trips, and the mixed dataset
  matched  delayed turn, speed too fast and wall hugging cut to about 2,200 error steps
  dose     the six single errors at 50% and 100% corrupted trips

Build the datasets first with corrupt.py (see README). Each model takes about a minute to train and
about ten seconds to score; results go to results/results.csv as usual.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

SINGLE = ["wrong_left", "wrong_right", "speed_fast", "delayed_turn", "wall_hugging", "missing_recovery"]
GROUPS = {
    "base": ["expert_clean"] + SINGLE,
    "pairs": ["delayed_turn_and_missing_recovery", "delayed_turn_and_wall_hugging",
              "missing_recovery_and_speed_fast", "mixed"],
    "matched": ["delayed_turn_n20", "speed_fast_n17", "wall_hugging_n11"],
    "dose": [f"{e}_p{p}" for p in (50, 100) for e in SINGLE],
}
SEEDS = range(5)


def scored():
    path = Path("results/results.csv")
    return set(pd.read_csv(path).model) if path.exists() else set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS) + ["all"], default="all")
    parser.add_argument("--seeds", nargs=2, type=int, default=(0, 4), metavar=("FIRST", "LAST"),
                        help="inclusive range of training seeds to run, default 0 4")
    args = parser.parse_args()
    seeds = range(args.seeds[0], args.seeds[1] + 1)
    groups = list(GROUPS) if args.group == "all" else [args.group]
    if "expert_driver" not in scored():
        subprocess.run([sys.executable, "src/evaluate.py", "--expert", "--models", "models/none*.pt"], check=False)
    for group in groups:
        for seed in seeds:
            for name in GROUPS[group]:
                model = f"{name}_seed{seed}"
                if model in scored():
                    continue
                data = Path("data") / f"{name}.npz"
                if not data.exists():
                    sys.exit(f"missing {data}: build it with corrupt.py first")
                print(f"=== {group}: {model}", flush=True)
                subprocess.run([sys.executable, "src/train_bc.py", "--data", str(data), "--seed", str(seed),
                                "--test-trips", "1"], check=True)
                subprocess.run([sys.executable, "src/evaluate.py", "--models", f"models/{model}.pt"], check=True)


if __name__ == "__main__":
    main()
