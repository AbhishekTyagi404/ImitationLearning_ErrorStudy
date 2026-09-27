"""BRICK 1-2: create the clean 'expert demonstrations' (the bronze table of the project).

Run:  python src/make_expert_data.py --episodes 500

Writes two files you can use:
  data/expert_clean.npz   fast format used by the training code
  data/expert_clean.csv   one row per step, so you can explore it with SQL / pandas / DuckDB

Each row is one step of one trip:
  episode, step, x, y, vx, vy, goal_x, goal_y, fx, fy
  (state = x, y, vx, vy and the goal; action = fx, fy = the push the expert applied)
"""
import argparse
from pathlib import Path

import numpy as np

from common import ENV_ID, MAX_STEPS, ExpertDriver, free_cells, make_env


def collect(n_episodes: int, seed: int):
    env = make_env()
    cells = free_cells(env)
    driver = ExpertDriver(env)
    rng = np.random.default_rng(seed)

    rows = dict(episode=[], step=[], obs=[], goal=[], action=[])
    kept = attempted = 0
    while kept < n_episodes:
        attempted += 1
        # random start cell and random, different goal cell -> diverse trips
        s, g = rng.choice(len(cells), 2, replace=False)
        obs, _ = env.reset(seed=int(rng.integers(1_000_000_000)),
                           options={"reset_cell": np.array(cells[s]), "goal_cell": np.array(cells[g])})
        driver.plan(obs["observation"][:2], obs["desired_goal"])

        ep = dict(obs=[], goal=[], action=[])
        for _ in range(MAX_STEPS):
            a = driver.act(obs["observation"])
            ep["obs"].append(obs["observation"].copy())
            ep["goal"].append(obs["desired_goal"].copy())
            ep["action"].append(a)
            obs, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        if not info["success"]:          # keep only perfect trips: this is the "expert" data
            continue
        for t in range(len(ep["obs"])):
            rows["episode"].append(kept)
            rows["step"].append(t)
            rows["obs"].append(ep["obs"][t])
            rows["goal"].append(ep["goal"][t])
            rows["action"].append(ep["action"][t])
        kept += 1
    return env, {k: np.array(v) for k, v in rows.items()}, attempted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/expert_clean")
    args = parser.parse_args()

    env, d, attempted = collect(args.episodes, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out.with_suffix(".npz"),
        episode=d["episode"], step=d["step"], obs=d["obs"], goal=d["goal"], action=d["action"],
        maze_map=np.array(env.unwrapped.maze.maze_map), env_id=ENV_ID,
    )

    import pandas as pd
    df = pd.DataFrame({
        "episode": d["episode"], "step": d["step"],
        "x": d["obs"][:, 0], "y": d["obs"][:, 1], "vx": d["obs"][:, 2], "vy": d["obs"][:, 3],
        "goal_x": d["goal"][:, 0], "goal_y": d["goal"][:, 1],
        "fx": d["action"][:, 0], "fy": d["action"][:, 1],
    })
    df.to_csv(out.with_suffix(".csv"), index=False)

    lengths = df.groupby("episode").size()
    print(f"Kept {args.episodes} successful trips out of {attempted} attempts")
    print(f"Total rows (steps): {len(df)}   trip length: mean {lengths.mean():.0f}, min {lengths.min()}, max {lengths.max()}")
    print(f"Saved {out.with_suffix('.npz')} and {out.with_suffix('.csv')}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
