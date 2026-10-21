"""BRICK 5: the mistake makers. Re-record the expert trips, but let a flawed driver make some of them.

Run:  python src/corrupt.py --error wrong_left
      python src/corrupt.py --error all          (every error type this file knows)

How a corrupted dataset is built (the same recipe for every error type, so they compare fairly):
  1. Replay exactly the same 500 trips as data/expert_clean.npz (same start, goal and reset seed).
  2. A fixed 20% of those trips (100 trips, the SAME 100 for every error type) are driven by a flawed driver.
  3. The other 80% are driven by the normal expert, so they are identical to the clean data.
Because the flawed driver really drives the ball, every state matches the action that led to it.
(Editing action numbers inside the finished clean file would break that link.)

Error types in this file (Brick 5). Brick 6 adds delayed turn, obstacle mistake, missing recovery.
  wrong_left   once per trip, at a junction where a side corridor opens to the LEFT, the driver drives into it
               for `detour` steps, then realizes and re-plans from wherever it ended up.
  wrong_right  the same, to the RIGHT.
  speed_fast   the driver's speed cap is 1.5x the expert's (3.0 instead of 2.0): overshoots corners.
  speed_slow   the speed cap is 0.5x (1.0): slow, overly careful driving.

Writes, for each error type:
  data/<error>.npz / .csv       same columns as the clean data, plus 'corrupted' (1 for the flawed trips)
  figures/corrupt_<error>.png   4 corrupted trips drawn over their clean versions (the Brick 5 check:
                                if you cannot see the difference, the mistake is too weak to matter)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ENV_ID, MAX_STEPS, ExpertDriver, cell_to_xy, free_cells, make_env

CORRUPT_FRACTION = 0.2
CORRUPT_SEED = 7          # picks WHICH trips get corrupted; keep fixed so every error type uses the same trips


# ---------------------------------------------------------------- the flawed drivers
class SteeringMixin:
    """The expert's two-layer controller, with the speed cap as a parameter."""

    def steer(self, target, pos, vel, vmax):
        v_des = self.gain * (target - pos)
        speed = np.linalg.norm(v_des)
        if speed > vmax:
            v_des = v_des * (vmax / speed)
        return np.clip(self.k_vel * (v_des - vel), -1.0, 1.0).astype(np.float32)


class SpeedErrorDriver(SteeringMixin, ExpertDriver):
    """Perfect route, wrong speed: the speed cap is multiplied by `factor`."""

    def __init__(self, env, rng, factor):
        super().__init__(env)
        self.factor = factor
        self.events = 0

    def plan(self, pos, goal_xy):
        super().plan(pos, goal_xy)
        self.events = 1                      # the whole trip is driven at the wrong speed

    def act(self, obs):
        pos, vel = obs[:2], obs[2:4]
        while self.idx < len(self.path) - 1 and np.linalg.norm(cell_to_xy(self.env, self.path[self.idx]) - pos) < self.radius:
            self.idx += 1
        target = self.goal_xy if self.idx >= len(self.path) - 1 else cell_to_xy(self.env, self.path[self.idx])
        return self.steer(target, pos, vel, self.vmax * self.factor)


class WrongTurnDriver(SteeringMixin, ExpertDriver):
    """Takes ONE wrong turn per trip into a side corridor on the given side, then re-plans.

    At plan time we list every junction on the route where the cell to the left (or right) of the travel
    direction is open floor, and pick one of them at random. When the driver reaches that junction it heads
    into the side cell for `detour` steps instead of continuing, then plans a new shortest path from where
    it is. If the route has no such junction, the trip stays clean (counted in the printout).
    """

    def __init__(self, env, rng, side, detour=25):
        super().__init__(env)
        self.rng, self.side, self.detour = rng, side, detour
        self.grid = np.array(env.unwrapped.maze.maze_map)

    def side_cell(self, cur, nxt):
        dr, dc = nxt[0] - cur[0], nxt[1] - cur[1]
        # rows grow downward, so "left of the travel direction" is (-dc, dr) and "right" is (dc, -dr)
        sr, sc = (-dc, dr) if self.side == "left" else (dc, -dr)
        cell = (cur[0] + sr, cur[1] + sc)
        return cell if self.grid[cell] != 1 else None

    def plan(self, pos, goal_xy):
        super().plan(pos, goal_xy)
        options = [k for k in range(1, len(self.path)) if self.side_cell(self.path[k - 1], self.path[k])]
        self.wrong_at = int(self.rng.choice(options)) if options else None   # the junction where it goes wrong
        self.steps_left = 0
        self.events = 0

    def act(self, obs):
        pos, vel = obs[:2], obs[2:4]
        if self.steps_left > 0:                       # currently driving down the wrong corridor
            self.steps_left -= 1
            action = self.steer(self.detour_xy, pos, vel, self.vmax)
            if self.steps_left == 0:                  # notice the mistake: new shortest path from here
                ExpertDriver.plan(self, pos, self.goal_xy)
                self.wrong_at = None
            return action

        old = self.idx
        while self.idx < len(self.path) - 1 and np.linalg.norm(cell_to_xy(self.env, self.path[self.idx]) - pos) < self.radius:
            self.idx += 1
        if self.wrong_at is not None and old < self.wrong_at <= self.idx:
            wrong = self.side_cell(self.path[self.wrong_at - 1], self.path[self.wrong_at])
            self.detour_xy = cell_to_xy(self.env, wrong)
            self.steps_left = self.detour - 1
            self.events += 1
            return self.steer(self.detour_xy, pos, vel, self.vmax)

        target = self.goal_xy if self.idx >= len(self.path) - 1 else cell_to_xy(self.env, self.path[self.idx])
        return self.steer(target, pos, vel, self.vmax)


ERRORS = {
    "wrong_left": lambda env, rng: WrongTurnDriver(env, rng, "left"),
    "wrong_right": lambda env, rng: WrongTurnDriver(env, rng, "right"),
    "speed_fast": lambda env, rng: SpeedErrorDriver(env, rng, 1.5),
    "speed_slow": lambda env, rng: SpeedErrorDriver(env, rng, 0.5),
}


# ---------------------------------------------------------------- re-recording the trips
def corrupted_trips(n_episodes):
    rng = np.random.default_rng(CORRUPT_SEED)
    return set(rng.choice(n_episodes, int(round(CORRUPT_FRACTION * n_episodes)), replace=False).tolist())


def collect(error, n_episodes, seed):
    """Same draws as make_expert_data.collect(), so trip i here is trip i of the clean data."""
    env = make_env()
    cells = free_cells(env)
    rng = np.random.default_rng(seed)               # start/goal draws: identical to the clean data
    flaw_rng = np.random.default_rng(seed + 1000)   # the flawed driver's own random choices
    bad = corrupted_trips(n_episodes)
    expert, flawed = ExpertDriver(env), ERRORS[error](env, flaw_rng)

    rows = dict(episode=[], step=[], obs=[], goal=[], action=[], corrupted=[])
    stats = dict(flawed=0, with_error=0, failed=0, flawed_steps=0)
    for trip in range(n_episodes):
        s, g = rng.choice(len(cells), 2, replace=False)
        obs, _ = env.reset(seed=int(rng.integers(1_000_000_000)),
                           options={"reset_cell": np.array(cells[s]), "goal_cell": np.array(cells[g])})
        driver = flawed if trip in bad else expert
        driver.plan(obs["observation"][:2], obs["desired_goal"])

        n0 = len(rows["obs"])
        for t in range(MAX_STEPS):
            a = driver.act(obs["observation"])
            rows["episode"].append(trip)
            rows["step"].append(t)
            rows["obs"].append(obs["observation"].copy())
            rows["goal"].append(obs["desired_goal"].copy())
            rows["action"].append(a)
            rows["corrupted"].append(int(trip in bad))
            obs, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        if trip in bad:
            stats["flawed"] += 1
            stats["with_error"] += int(driver.events > 0)
            stats["flawed_steps"] += len(rows["obs"]) - n0
        # A flawed trip that never reaches the goal is still kept: a bad demonstration is the point.
        stats["failed"] += int(not info["success"])
    return env, {k: np.array(v) for k, v in rows.items()}, stats


def save(env, d, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), episode=d["episode"], step=d["step"], obs=d["obs"],
                        goal=d["goal"], action=d["action"], corrupted=d["corrupted"],
                        maze_map=np.array(env.unwrapped.maze.maze_map), env_id=ENV_ID)
    pd.DataFrame({
        "episode": d["episode"], "step": d["step"], "corrupted": d["corrupted"],
        "x": d["obs"][:, 0], "y": d["obs"][:, 1], "vx": d["obs"][:, 2], "vy": d["obs"][:, 3],
        "goal_x": d["goal"][:, 0], "goal_y": d["goal"][:, 1],
        "fx": d["action"][:, 0], "fy": d["action"][:, 1],
    }).to_csv(out.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------- the Brick 5 check: clean vs corrupted
def plot_compare(error, data, clean_path, out, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_trajectories import draw_maze

    clean = np.load(clean_path, allow_pickle=True) if Path(clean_path).exists() else None
    bad = sorted(set(data["episode"][data["corrupted"] == 1].tolist()))
    # show the corrupted trips that differ most from clean (longest), so the mistake is visible
    lengths = {e: int((data["episode"] == e).sum()) for e in bad}
    if clean is not None:
        lengths = {e: lengths[e] - int((clean["episode"] == e).sum()) for e in bad}
    shown = sorted(bad, key=lambda e: -abs(lengths[e]))[:n]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    draw_maze(axes[0], data["maze_map"])
    for i, e in enumerate(shown):
        c = plt.cm.tab10(i % 10)
        m = data["episode"] == e
        if clean is not None:
            mc = clean["episode"] == e
            axes[0].plot(*clean["obs"][mc][:, :2].T, color=c, lw=1, alpha=0.5, ls="--",
                         label="clean (dashed)" if i == 0 else None)
            axes[1].plot(np.linalg.norm(clean["obs"][mc][:, 2:4], axis=1), color=c, lw=1, alpha=0.5, ls="--")
        axes[0].plot(*data["obs"][m][:, :2].T, color=c, lw=2, label=f"{error} (solid)" if i == 0 else None)
        axes[0].scatter(*data["obs"][m][0, :2], marker="o", s=40, color=c, zorder=3)
        axes[0].scatter(*data["goal"][m][0], marker="*", s=140, color=c, zorder=3)
        axes[1].plot(np.linalg.norm(data["obs"][m][:, 2:4], axis=1), color=c, lw=2)
    axes[0].set_title(f"{error}: {len(shown)} corrupted trips vs the same trips clean")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].set_title("Speed over time (solid = corrupted, dashed = clean)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("speed")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error", required=True, choices=list(ERRORS) + ["all"])
    parser.add_argument("--episodes", type=int, default=500, help="must match the clean data (500)")
    parser.add_argument("--seed", type=int, default=0, help="must match the clean data (0)")
    parser.add_argument("--clean", default="data/expert_clean.npz", help="used only for the comparison plot")
    args = parser.parse_args()

    for error in (list(ERRORS) if args.error == "all" else [args.error]):
        env, d, st = collect(error, args.episodes, args.seed)
        out = Path("data") / error
        save(env, d, out)
        d["maze_map"] = np.array(env.unwrapped.maze.maze_map)
        fig = Path("figures") / f"corrupt_{error}.png"
        plot_compare(error, d, args.clean, fig)
        print(f"{error:12s} {st['flawed']} flawed trips ({st['flawed'] / args.episodes:.0%}), "
              f"{st['with_error']} actually contain the error, {st['failed']} never reached the goal; "
              f"{len(d['obs'])} rows ({st['flawed_steps']} from flawed trips)")
        print(f"             saved {out.with_suffix('.npz')}, {out.with_suffix('.csv')} and {fig}")


if __name__ == "__main__":
    main()
