"""Evaluation: each trained policy drives the same fixed test trips and gets four scores.

Run:  python src/evaluate.py                         (every model in models/, 200 test trips)
      python src/evaluate.py --models "models/expert_clean_seed*.pt"
      python src/evaluate.py --expert                (adds a row for the expert driver as the upper bound)
      python src/evaluate.py --trips 300

Metrics:
  success_rate     share of test trips that reach the goal within the step limit (600 steps)
  collision_rate   share of test trips with at least one step where the ball touches a wall: a MuJoCo
                   contact between 'particle_geom' and a geom named 'block_...' with distance <= 0.
                   Near contacts inside MuJoCo's 2 mm safety margin are ignored.
  path_efficiency  distance the expert travels on the same trip / distance the policy travelled,
                   on successful trips only. The expert scores exactly 1.0; lower means a longer route.
  time_to_goal     steps until success, on successful trips only

The expert driver runs the same trips first (about 5 s) to provide the reference distances.

Writes:
  results/results.csv            one row per (model, number of trips). Running a model again replaces
                                 its row, so the file never holds duplicates.
  results/trips/<model>.csv      one row per test trip: start, goal, success, collisions, end point and
                                 first wall contact. These are the features for the error type classifier.

Running the same command twice gives identical numbers.
"""
import argparse
import glob
import re
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import torch

from common import MAX_STEPS, ExpertDriver, make_env
from train_bc import BCPolicy, test_pairs

RESULTS = Path("results/results.csv")
TRIPS_DIR = Path("results/trips")


class WallContact:
    """Checks after every step whether the ball is touching a wall."""

    def __init__(self, env):
        pe = env.unwrapped.point_env
        self.model, self.data = pe.model, pe.data
        name = lambda i: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        self.ball = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "particle_geom")
        self.walls = {i for i in range(self.model.ngeom) if name(i).startswith("block_")}
        assert self.ball >= 0 and self.walls, "could not find the ball or wall geoms in the MuJoCo model"

    def touching(self):
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            if c.dist > 0:          # inside the safety margin but not actually touching
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == self.ball and g2 in self.walls) or (g2 == self.ball and g1 in self.walls):
                return True
        return False


def evaluate(drive, n_trips, make_driver=None):
    """drive(obs_dict) returns an action. make_driver is an optional callback run after each reset, used for the expert."""
    env = make_env()
    walls = WallContact(env)
    trips = []
    for i, (start, goal, reset_seed) in enumerate(test_pairs(env, n_trips)):
        obs, _ = env.reset(seed=reset_seed, options={"reset_cell": np.array(start), "goal_cell": np.array(goal)})
        if make_driver:
            make_driver(env, obs)
        goal_xy = obs["desired_goal"].copy()
        travelled, contact_steps, first_contact_xy = 0.0, 0, None
        info, steps = {"success": False}, 0
        for steps in range(1, MAX_STEPS + 1):
            pos_before = obs["observation"][:2].copy()
            obs, _, terminated, truncated, info = env.step(drive(obs))
            travelled += float(np.linalg.norm(obs["observation"][:2] - pos_before))
            if walls.touching():
                contact_steps += 1
                if first_contact_xy is None:
                    first_contact_xy = obs["observation"][:2].copy()
            if terminated or truncated:
                break
        success = bool(info["success"])
        end_xy = obs["observation"][:2]
        trips.append(dict(
            trip=i, start_row=start[0], start_col=start[1], goal_row=goal[0], goal_col=goal[1],
            success=success, steps=steps, collided=contact_steps > 0, contact_steps=contact_steps,
            travelled=round(travelled, 4),
            end_x=round(float(end_xy[0]), 4), end_y=round(float(end_xy[1]), 4),
            end_dist_to_goal=round(float(np.linalg.norm(end_xy - goal_xy)), 4),
            first_contact_x=None if first_contact_xy is None else round(float(first_contact_xy[0]), 4),
            first_contact_y=None if first_contact_xy is None else round(float(first_contact_xy[1]), 4),
        ))
    env.close()
    return pd.DataFrame(trips)


def run_expert(n_trips):
    """Drive every test trip with the expert: a fresh BFS plan after each reset, then the PD controller."""
    holder = {}

    def make_driver(env, obs):
        holder["d"] = ExpertDriver(env)
        holder["d"].plan(obs["observation"][:2], obs["desired_goal"])

    return evaluate(lambda obs: holder["d"].act(obs["observation"]), n_trips, make_driver)


def summarize(trips):
    ok = trips[trips.success]
    return dict(
        success_rate=round(trips.success.mean(), 4),
        collision_rate=round(trips.collided.mean(), 4),
        path_efficiency=round((ok.expert_travelled / ok.travelled).mean(), 4) if len(ok) else np.nan,
        time_to_goal=round(ok.steps.mean(), 1) if len(ok) else np.nan,
    )


def parse_name(stem):
    """Turns 'expert_clean_seed1' into ('expert_clean', 1)."""
    m = re.match(r"(.+)_seed(\d+)$", stem)
    return (m.group(1), int(m.group(2))) if m else (stem, -1)


def load_policy(path):
    policy = BCPolicy()
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy


def save_row(row):
    """Append to results.csv, replacing any earlier row for the same model and number of trips."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        old = old[~((old.model == row["model"]) & (old.n_trips == row["n_trips"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(RESULTS, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models/*.pt", help="file pattern of the models to score")
    parser.add_argument("--trips", type=int, default=200, help="number of fixed test trips (use 200 or more)")
    parser.add_argument("--expert", action="store_true", help="score the expert driver as well (upper bound)")
    args = parser.parse_args()

    torch.set_num_threads(2)
    TRIPS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.expert:
        jobs.append(("expert_driver", None))
    jobs += [(Path(p).stem, p) for p in sorted(glob.glob(args.models))]
    if not jobs:
        print(f"No models match {args.models!r}. Train one first with src/train_bc.py.")
        return

    print(f"Scoring {len(jobs)} policies on {args.trips} fixed test trips each")
    expert = run_expert(args.trips)
    if not expert.success.all():
        print(f"WARNING: the expert failed {int((~expert.success).sum())} test trips; efficiency on those is skipped")
    print()
    for name, path in jobs:
        t0 = time.time()
        if path is None:
            trips, (dataset, seed) = expert.copy(), ("expert_driver", -1)
        else:
            policy = load_policy(path)
            trips = evaluate(lambda obs: policy.act(obs["observation"], obs["desired_goal"]), args.trips)
            dataset, seed = parse_name(name)
        trips["expert_travelled"] = expert.travelled.where(expert.success)
        trips["expert_steps"] = expert.steps.where(expert.success)

        trips.to_csv(TRIPS_DIR / f"{name}.csv", index=False)
        s = summarize(trips)
        secs = time.time() - t0
        save_row(dict(model=name, dataset=dataset, seed=seed, n_trips=args.trips, **s,
                      eval_seconds=round(secs), evaluated_at=datetime.now().isoformat(timespec="seconds")))
        print(f"{name:28s} success {s['success_rate']:6.1%}   collisions {s['collision_rate']:6.1%}   "
              f"efficiency {s['path_efficiency']:.3f}   time {s['time_to_goal']:5.1f} steps   ({secs:.0f} s)")

    # one line per dataset: mean and spread across seeds
    df = pd.read_csv(RESULTS)
    df = df[df.n_trips == args.trips]
    cols = ["success_rate", "collision_rate", "path_efficiency", "time_to_goal"]
    table = df.groupby("dataset")[cols].agg(["mean", "std"]).round(3)
    table.insert(0, "seeds", df.groupby("dataset").size())
    print(f"\nSummary by dataset ({args.trips} trips, mean and std across seeds):")
    print(table.to_string())
    print(f"\nSaved {RESULTS} and the trip details in {TRIPS_DIR}/")


if __name__ == "__main__":
    main()
