"""Corrupted demonstrations: replay the expert trips and let a flawed driver make some of them.

Run:  python src/corrupt.py --error wrong_left
      python src/corrupt.py --error all          (every error type in this file)

Every corrupted dataset follows the same recipe, so the error types compare fairly:
  1. Replay exactly the same 500 trips as data/expert_clean.npz (same start, goal and reset seed).
  2. A fixed 20% of those trips (100 trips, the SAME 100 for every error type) are driven by a flawed driver.
  3. The other 80% are driven by the normal expert and match the clean data exactly.
The flawed driver really drives the ball, so every recorded state is the result of the action before it.
Editing action numbers inside the finished clean file would break that link.

Error types:
  wrong_left   once per trip, at a junction where a side corridor opens to the LEFT, the driver turns into it
               for `detour` steps, then plans a new path from wherever it ended up.
  wrong_right  the same, to the RIGHT.
  speed_fast   the speed cap is 1.5x the expert's (3.0 instead of 2.0), so the ball overshoots corners.
  speed_slow   the speed cap is 0.5x (1.0): slow, overly careful driving.
  delayed_turn at every corner the driver keeps going straight for 45 steps, then turns.
  wall_hugging obstacle mistake: the driver aims 0.38 toward a side wall instead of the cell center.
  missing_recovery
               once per trip the ball is pushed off its route for 25 steps. The slip is recorded,
               but the steps that steer back onto the route are left out of the data.

Exposure: every flawed driver sets `active` on the steps where it is doing the wrong thing. This is only
read and counted; it never changes an action, a plan or a random draw, so the datasets stay the same.
  wrong_left, wrong_right   the detour steps
  delayed_turn              the straight steps after each corner
  speed_fast, speed_slow    steps where the wrong speed cap changes the command (the desired speed is above
                            the lower of the two caps)
  wall_hugging              steps aimed at a shifted point instead of the cell center
  missing_recovery          the push steps (the recovery steps left out of the data are counted on their own)

Writes, for each error type:
  data/<error>.npz / .csv       same columns as the clean data, plus 'corrupted' (1 for the flawed trips)
                                and 'error_active' (1 on steps where the error was active)
  data/exposure.csv             one row per error type: affected trips, error events, active steps, lengths
  figures/corrupt_<error>.png   4 corrupted trips drawn over their clean versions
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ENV_ID, MAX_STEPS, ExpertDriver, cell_to_xy, free_cells, make_env

CORRUPT_FRACTION = 0.2
CORRUPT_SEED = 7          # picks WHICH trips get corrupted; keep fixed so every error type uses the same trips


class SteeringMixin:
    """The expert's two layer controller, with the speed cap as a parameter."""

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
        wanted = np.linalg.norm(self.gain * (target - pos))
        self.active = wanted > min(self.vmax, self.vmax * self.factor)   # the two caps give different commands
        return self.steer(target, pos, vel, self.vmax * self.factor)


class WrongTurnDriver(SteeringMixin, ExpertDriver):
    """Takes ONE wrong turn per trip into a side corridor on the given side, then plans again.

    At plan time the driver lists every junction on the route where the cell to the left (or right) of the
    travel direction is open floor, and picks one of them at random. On reaching that junction it heads into
    the side cell for `detour` steps instead of continuing, then plans a new shortest path from where it is.
    A route with no such junction stays clean; the printout counts those trips.
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
        self.active = False
        if self.steps_left > 0:                       # currently driving down the wrong corridor
            self.active = True
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
            self.active = True
            return self.steer(self.detour_xy, pos, vel, self.vmax)

        target = self.goal_xy if self.idx >= len(self.path) - 1 else cell_to_xy(self.env, self.path[self.idx])
        return self.steer(target, pos, vel, self.vmax)


class DelayedTurnDriver(SteeringMixin, ExpertDriver):
    """Turns late at every corner: keeps driving straight for `delay` steps after reaching a turn.

    When the driver reaches a waypoint where the route changes direction, it keeps pushing in the old
    direction for `delay` steps, then steers to the next waypoint as usual. A corner is detected 0.5
    before its center and the ball covers about 0.02 per step, so 45 steps carry it past the center and
    into the wall ahead. Straight routes stay clean.
    """

    def __init__(self, env, rng, delay=45):
        super().__init__(env)
        self.delay = delay

    def plan(self, pos, goal_xy):
        super().plan(pos, goal_xy)
        self.steps_left = 0
        self.events = 0

    def heading(self, k):
        """Unit vector of travel from path[k - 1] to path[k] in x, y."""
        d = cell_to_xy(self.env, self.path[k]) - cell_to_xy(self.env, self.path[k - 1])
        return d / np.linalg.norm(d)

    def act(self, obs):
        pos, vel = obs[:2], obs[2:4]
        self.active = False
        if self.steps_left > 0:                       # still going straight past the corner
            self.active = True
            self.steps_left -= 1
            return self.steer(pos + self.straight, pos, vel, self.vmax)

        old = self.idx
        while self.idx < len(self.path) - 1 and np.linalg.norm(cell_to_xy(self.env, self.path[self.idx]) - pos) < self.radius:
            self.idx += 1
        k = self.idx - 1                              # the waypoint just reached
        if self.idx > old and k >= 1 and self.idx < len(self.path):
            if not np.allclose(self.heading(k), self.heading(self.idx)):
                self.straight = self.heading(k)
                self.steps_left = self.delay - 1
                self.events += 1
                self.active = True
                return self.steer(pos + self.straight, pos, vel, self.vmax)

        target = self.goal_xy if self.idx >= len(self.path) - 1 else cell_to_xy(self.env, self.path[self.idx])
        return self.steer(target, pos, vel, self.vmax)


class WallHuggingDriver(SteeringMixin, ExpertDriver):
    """Obstacle mistake: aims at a point shifted `offset` toward a side wall instead of the cell center.

    For each waypoint the driver looks at the two cells beside the travel direction. If one of them is a
    wall it aims `offset` toward that wall; if both are walls it uses the side picked for this trip. With
    a ball of radius 0.1 and walls 0.5 from the center, an offset of 0.38 leaves a gap of 0.02.
    """

    def __init__(self, env, rng, offset=0.38):
        super().__init__(env)
        self.rng, self.offset = rng, offset
        self.grid = np.array(env.unwrapped.maze.maze_map)

    def plan(self, pos, goal_xy):
        super().plan(pos, goal_xy)
        self.side = self.rng.choice(["left", "right"])
        self.targets = [self.shifted(k) for k in range(len(self.path))]
        self.events = int(any(not np.allclose(t, cell_to_xy(self.env, c)) for t, c in zip(self.targets, self.path)))

    def shifted(self, k):
        center = cell_to_xy(self.env, self.path[k])
        if k == 0:
            return center
        dr, dc = self.path[k][0] - self.path[k - 1][0], self.path[k][1] - self.path[k - 1][1]
        # rows grow downward, so "left of the travel direction" is (-dc, dr) and "right" is (dc, -dr)
        sides = {"left": (-dc, dr), "right": (dc, -dr)}
        walls = [s for s, (sr, sc) in sides.items() if self.grid[self.path[k][0] + sr, self.path[k][1] + sc] == 1]
        if not walls:
            return center
        sr, sc = sides[self.side if self.side in walls else walls[0]]
        return center + self.offset * np.array([sc, -sr], dtype=float)   # a row step down is a step in -y

    def act(self, obs):
        pos, vel = obs[:2], obs[2:4]
        while self.idx < len(self.path) - 1 and np.linalg.norm(cell_to_xy(self.env, self.path[self.idx]) - pos) < self.radius:
            self.idx += 1
        target = self.goal_xy if self.idx >= len(self.path) - 1 else self.targets[self.idx]
        self.active = self.idx < len(self.path) - 1 and not np.allclose(target, cell_to_xy(self.env, self.path[self.idx]))
        return self.steer(target, pos, vel, self.vmax)


class MissingRecoveryDriver(SteeringMixin, ExpertDriver):
    """Slips off the path once per trip, and the steps that steer back are left out of the data.

    At a random step between `first` and `last` the ball is pushed with `force` in a random direction
    for `push` steps; those steps are recorded with the push as their action. The driver then plans a new
    route from where it is. While it steers back, `recording` is False and collect() drops those rows.
    Recording starts again once the ball is within `back` of the route it had before the slip, or close
    to the goal.
    """

    def __init__(self, env, rng, push=25, force=0.5, first=10, last=80, back=0.2):
        super().__init__(env)
        self.rng, self.push, self.force = rng, push, force
        self.first, self.last, self.back = first, last, back

    def plan(self, pos, goal_xy):
        super().plan(pos, goal_xy)
        self.t = 0
        self.slip_at = int(self.rng.integers(self.first, self.last))
        angle = self.rng.uniform(0, 2 * np.pi)
        self.push_dir = self.force * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        self.route = [cell_to_xy(self.env, c) for c in self.path] + [np.asarray(goal_xy, dtype=float)]
        self.recording, self.state, self.events = True, "before", 0

    def off_route(self, pos):
        """Distance from pos to the nearest point of the route planned before the slip."""
        best = np.inf
        for a, b in zip(self.route[:-1], self.route[1:]):
            ab = b - a
            u = 0.0 if not ab.any() else np.clip(np.dot(pos - a, ab) / np.dot(ab, ab), 0.0, 1.0)
            best = min(best, np.linalg.norm(pos - (a + u * ab)))
        return best

    def act(self, obs):
        pos, vel = obs[:2], obs[2:4]
        self.t += 1
        self.active = False
        if self.state == "before" and self.t >= self.slip_at:
            self.state, self.push_left, self.events = "slip", self.push, 1
        if self.state == "slip":
            self.active = True
            self.push_left -= 1
            if self.push_left == 0:                   # slip over: plan from here, stop recording
                ExpertDriver.plan(self, pos, self.goal_xy)
                self.state, self.recording = "recover", False
            return self.push_dir.copy()
        near_goal = np.linalg.norm(pos - self.goal_xy) < 2 * self.radius
        if self.state == "recover" and (self.off_route(pos) < self.back or near_goal):
            self.state, self.recording = "after", True

        while self.idx < len(self.path) - 1 and np.linalg.norm(cell_to_xy(self.env, self.path[self.idx]) - pos) < self.radius:
            self.idx += 1
        target = self.goal_xy if self.idx >= len(self.path) - 1 else cell_to_xy(self.env, self.path[self.idx])
        return self.steer(target, pos, vel, self.vmax)


class ComboDriver(SteeringMixin, ExpertDriver):
    """Two or more mistakes on the same trip (Brick 7).

    `parts` is any set of speed_fast, delayed_turn, wall_hugging and missing_recovery. Each part behaves
    exactly like its single error driver above, with the same settings and the same random draws, so a
    ComboDriver with one part reproduces that single error dataset step for step (checked in
    check_combo_matches_single). When parts meet on the same step:
      the slip of missing_recovery overrides everything else while it lasts, then the route is planned
      again (and the wall hugging aim points are shifted again for the new route);
      a delayed turn overrides the normal target while the ball keeps going straight;
      speed_fast changes the speed cap of every steering command, straight runs included.
    A step counts as error active when any part is active on it.
    """

    PARTS = ("speed_fast", "delayed_turn", "wall_hugging", "missing_recovery")

    def __init__(self, env, rng, parts, factor=1.5, delay=45, offset=0.38,
                 push=25, force=0.5, first=10, last=80, back=0.2):
        super().__init__(env)
        unknown = set(parts) - set(self.PARTS)
        if unknown:
            raise ValueError(f"cannot combine {sorted(unknown)}; choose from {self.PARTS}")
        self.rng, self.parts = rng, set(parts)
        self.factor = factor if "speed_fast" in self.parts else 1.0
        self.delay, self.offset = delay, offset
        self.push, self.force, self.first, self.last, self.back = push, force, first, last, back
        self.grid = np.array(env.unwrapped.maze.maze_map)

    # the single error drivers' helpers, reused as they are
    heading = DelayedTurnDriver.heading
    shifted = WallHuggingDriver.shifted
    off_route = MissingRecoveryDriver.off_route

    def plan(self, pos, goal_xy):
        ExpertDriver.plan(self, pos, goal_xy)
        self.steps_left, self.events, self.recording = 0, 0, True
        if "speed_fast" in self.parts:
            self.events += 1                         # the whole trip is driven at the wrong speed
        if "wall_hugging" in self.parts:
            self.side = self.rng.choice(["left", "right"])
            self.targets = [self.shifted(k) for k in range(len(self.path))]
            self.events += int(any(not np.allclose(t, cell_to_xy(self.env, c))
                                   for t, c in zip(self.targets, self.path)))
        if "missing_recovery" in self.parts:
            self.t = 0
            self.slip_at = int(self.rng.integers(self.first, self.last))
            angle = self.rng.uniform(0, 2 * np.pi)
            self.push_dir = self.force * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            self.route = [cell_to_xy(self.env, c) for c in self.path] + [np.asarray(goal_xy, dtype=float)]
            self.state = "before"

    def replan(self, pos):
        """New shortest route from here, used after a slip."""
        ExpertDriver.plan(self, pos, self.goal_xy)
        self.steps_left = 0
        if "wall_hugging" in self.parts:
            self.targets = [self.shifted(k) for k in range(len(self.path))]

    def act(self, obs):
        pos, vel = obs[:2], obs[2:4]
        vmax = self.vmax * self.factor
        self.active = False

        if "missing_recovery" in self.parts:
            self.t += 1
            if self.state == "before" and self.t >= self.slip_at:
                self.state, self.push_left = "slip", self.push
                self.events += 1
            if self.state == "slip":
                self.active = True
                self.push_left -= 1
                if self.push_left == 0:               # slip over: plan from here, stop recording
                    self.replan(pos)
                    self.state, self.recording = "recover", False
                return self.push_dir.copy()
            near_goal = np.linalg.norm(pos - self.goal_xy) < 2 * self.radius
            if self.state == "recover" and (self.off_route(pos) < self.back or near_goal):
                self.state, self.recording = "after", True

        if "delayed_turn" in self.parts and self.steps_left > 0:   # still going straight past the corner
            self.active = True
            self.steps_left -= 1
            return self.steer(pos + self.straight, pos, vel, vmax)

        old = self.idx
        while self.idx < len(self.path) - 1 and np.linalg.norm(cell_to_xy(self.env, self.path[self.idx]) - pos) < self.radius:
            self.idx += 1
        if "delayed_turn" in self.parts:
            k = self.idx - 1                          # the waypoint just reached
            if self.idx > old and k >= 1 and self.idx < len(self.path):
                if not np.allclose(self.heading(k), self.heading(self.idx)):
                    self.straight = self.heading(k)
                    self.steps_left = self.delay - 1
                    self.events += 1
                    self.active = True
                    return self.steer(pos + self.straight, pos, vel, vmax)

        if self.idx >= len(self.path) - 1:
            target = self.goal_xy
        elif "wall_hugging" in self.parts:
            target = self.targets[self.idx]
            self.active = not np.allclose(target, cell_to_xy(self.env, self.path[self.idx]))
        else:
            target = cell_to_xy(self.env, self.path[self.idx])
        if "speed_fast" in self.parts:
            wanted = np.linalg.norm(self.gain * (target - pos))
            self.active = self.active or wanted > min(self.vmax, vmax)
        return self.steer(target, pos, vel, vmax)


ERRORS = {
    "wrong_left": lambda env, rng: WrongTurnDriver(env, rng, "left"),
    "wrong_right": lambda env, rng: WrongTurnDriver(env, rng, "right"),
    "speed_fast": lambda env, rng: SpeedErrorDriver(env, rng, 1.5),
    "speed_slow": lambda env, rng: SpeedErrorDriver(env, rng, 0.5),
    "delayed_turn": lambda env, rng: DelayedTurnDriver(env, rng),
    "wall_hugging": lambda env, rng: WallHuggingDriver(env, rng),
    "missing_recovery": lambda env, rng: MissingRecoveryDriver(env, rng),
}

# the six error types of the study (speed_slow is held back: 10 of its trips time out)
STUDY_ERRORS = ["wrong_left", "wrong_right", "speed_fast", "delayed_turn", "wall_hugging", "missing_recovery"]


class MixedDriver:
    """The mixed dataset (Brick 7): each flawed trip gets ONE of the six study errors, taken in turn.

    The k th flawed trip in trip order (see trip_order) uses STUDY_ERRORS[k % 6], so 100 flawed trips
    give 17 each of the first four types and 16 each of the last two.
    """

    def __init__(self, env, rng, order):
        self.drivers = {e: ERRORS[e](env, rng) for e in STUDY_ERRORS}
        self.kind = {trip: STUDY_ERRORS[k % len(STUDY_ERRORS)] for k, trip in enumerate(order)}
        self.current = None

    def for_trip(self, trip):
        self.current = self.drivers[self.kind[trip]]
        return self.current


def make_flawed(error, env, rng, order):
    """Driver for an error name: a single error, a pair joined with '+', or 'mixed'."""
    if error == "mixed":
        return MixedDriver(env, rng, order)
    if "+" in error:
        return ComboDriver(env, rng, [p for p in error.split("+") if p])
    return ERRORS[error](env, rng)


def trip_order(n_episodes):
    """The order in which trips get corrupted as the corruption level rises.

    The first 100 are exactly the 20% set used since Brick 5 (same seed, same draw), so every earlier
    dataset is unchanged. The rest follow in a fixed random order. Taking the first k trips gives nested
    sets: the 20% trips are inside the 50% trips, which are inside the 100% trips, and a matched set with
    fewer trips is a subset of the 20% trips.
    """
    rng = np.random.default_rng(CORRUPT_SEED)
    first = rng.choice(n_episodes, int(round(CORRUPT_FRACTION * n_episodes)), replace=False).tolist()
    rest = sorted(set(range(n_episodes)) - set(first))
    rest = np.random.default_rng(CORRUPT_SEED + 1).permutation(rest).tolist()
    return first + rest


def corrupted_trips(n_episodes, n_bad=None):
    """The set of flawed trips: the first n_bad trips of trip_order (default: 20% of the trips)."""
    n_bad = int(round(CORRUPT_FRACTION * n_episodes)) if n_bad is None else n_bad
    return set(trip_order(n_episodes)[:n_bad])


def collect(error, n_episodes, seed, n_bad=None):
    """Same draws as make_expert_data.collect(), so trip i here is trip i of the clean data."""
    env = make_env()
    cells = free_cells(env)
    rng = np.random.default_rng(seed)               # start and goal draws, identical to the clean data
    flaw_rng = np.random.default_rng(seed + 1000)   # the flawed driver's own random choices
    bad = corrupted_trips(n_episodes, n_bad)
    expert = ExpertDriver(env)
    flawed = make_flawed(error, env, flaw_rng, [t for t in trip_order(n_episodes) if t in bad])

    rows = dict(episode=[], step=[], obs=[], goal=[], action=[], corrupted=[], error_active=[])
    stats = dict(flawed=0, with_error=0, failed=0, flawed_steps=0, dropped=0, events=0, active_steps=0,
                 affected_steps=0)
    for trip in range(n_episodes):
        s, g = rng.choice(len(cells), 2, replace=False)
        obs, _ = env.reset(seed=int(rng.integers(1_000_000_000)),
                           options={"reset_cell": np.array(cells[s]), "goal_cell": np.array(cells[g])})
        driver = flawed if trip in bad else expert
        if isinstance(driver, MixedDriver):
            driver = driver.for_trip(trip)
        driver.plan(obs["observation"][:2], obs["desired_goal"])

        n0 = len(rows["obs"])
        for t in range(MAX_STEPS):
            keep = getattr(driver, "recording", True)    # only MissingRecoveryDriver ever hides rows
            a = driver.act(obs["observation"])
            active = bool(getattr(driver, "active", False))   # read only: the expert never sets it
            if keep:
                rows["episode"].append(trip)
                rows["step"].append(t)
                rows["obs"].append(obs["observation"].copy())
                rows["goal"].append(obs["desired_goal"].copy())
                rows["action"].append(a)
                rows["corrupted"].append(int(trip in bad))
                rows["error_active"].append(int(active))
                stats["active_steps"] += int(active)
            else:
                stats["dropped"] += 1
            obs, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        if trip in bad:
            stats["flawed"] += 1
            stats["with_error"] += int(driver.events > 0)
            stats["events"] += driver.events
            stats["flawed_steps"] += len(rows["obs"]) - n0
            stats["affected_steps"] += (len(rows["obs"]) - n0) if driver.events > 0 else 0
        # A flawed trip that never reaches the goal is still kept: a bad demonstration is the point.
        stats["failed"] += int(not info["success"])
    return env, {k: np.array(v) for k, v in rows.items()}, stats


def save(env, d, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), episode=d["episode"], step=d["step"], obs=d["obs"],
                        goal=d["goal"], action=d["action"], corrupted=d["corrupted"], error_active=d["error_active"],
                        maze_map=np.array(env.unwrapped.maze.maze_map), env_id=ENV_ID)
    pd.DataFrame({
        "episode": d["episode"], "step": d["step"], "corrupted": d["corrupted"], "error_active": d["error_active"],
        "x": d["obs"][:, 0], "y": d["obs"][:, 1], "vx": d["obs"][:, 2], "vy": d["obs"][:, 3],
        "goal_x": d["goal"][:, 0], "goal_y": d["goal"][:, 1],
        "fx": d["action"][:, 0], "fy": d["action"][:, 1],
    }).to_csv(out.with_suffix(".csv"), index=False)


def save_exposure(error, n_rows, st, path=Path("data") / "exposure.csv"):
    """Adds or replaces this error type's row in the exposure table."""
    row = dict(error=error, flawed_trips=st["flawed"], affected_trips=st["with_error"], error_events=st["events"],
               active_steps=st["active_steps"], active_share=round(st["active_steps"] / n_rows, 4),
               affected_trip_steps=st["affected_steps"], flawed_trip_steps=st["flawed_steps"],
               dataset_steps=n_rows, dropped_steps=st["dropped"], failed_trips=st["failed"])
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(table):
        table = table[table.error != error]
    order = list(ERRORS)
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    # the seven single errors first in their usual order, then the Brick 7 datasets by name
    rank = lambda name: (order.index(name), "") if name in order else (len(order), name)
    table = table.iloc[sorted(range(len(table)), key=lambda i: rank(table.error.iloc[i]))].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def with_gaps(steps, values):
    """Inserts NaN where steps were left out of a trip, so plots show a gap instead of a straight jump."""
    breaks = np.where(np.diff(steps) > 1)[0] + 1
    return np.insert(steps.astype(float), breaks, np.nan), np.insert(values.astype(float), breaks, np.nan, axis=0)


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
        steps, obs = with_gaps(data["step"][m], data["obs"][m])
        axes[0].plot(*obs[:, :2].T, color=c, lw=2, label=f"{error} (solid)" if i == 0 else None)
        axes[0].scatter(*data["obs"][m][0, :2], marker="o", s=40, color=c, zorder=3)
        axes[0].scatter(*data["goal"][m][0], marker="*", s=140, color=c, zorder=3)
        axes[1].plot(steps, np.linalg.norm(obs[:, 2:4], axis=1), color=c, lw=2)
    axes[0].set_title(f"{error}: {len(shown)} corrupted trips vs the same trips clean")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].set_title("Speed over time (solid = corrupted, dashed = clean)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("speed")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def check_combo_matches_single(n_episodes=500, seed=0):
    """A ComboDriver with one part must give exactly the same data as that single error driver."""
    for part in ComboDriver.PARTS:
        _, single, _ = collect(part, n_episodes, seed)
        _, combo, _ = collect(part + "+", n_episodes, seed)
        same = all(np.array_equal(single[k], combo[k]) for k in single)
        print(f"{part:16s} one part combo equals the single error data: {same}")
        if not same:
            raise SystemExit("ComboDriver does not reproduce the single error driver")


def dataset_name(error, n_bad, n_episodes):
    """data/<name>.npz: the plain error name at 20%, then _p50, _p100 for other levels, _n<k> for k trips."""
    name = error.replace("+", "_and_")
    default = int(round(CORRUPT_FRACTION * n_episodes))
    if n_bad is None or n_bad == default:
        return name
    pct = 100 * n_bad / n_episodes
    return f"{name}_p{pct:.0f}" if pct == int(pct) and n_bad > default else f"{name}_n{n_bad}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error", required=True,
                        help="one of " + ", ".join(ERRORS) + "; 'all' for those seven; a pair such as "
                             "delayed_turn+missing_recovery; or 'mixed' (one of the six study errors per flawed trip)")
    parser.add_argument("--fraction", type=float, default=None,
                        help="share of trips to corrupt (default 0.2); the sets are nested, see trip_order")
    parser.add_argument("--trips", type=int, default=None,
                        help="number of trips to corrupt, instead of --fraction (used for matched error steps)")
    parser.add_argument("--episodes", type=int, default=500, help="must match the clean data (500)")
    parser.add_argument("--seed", type=int, default=0, help="must match the clean data (0)")
    parser.add_argument("--clean", default="data/expert_clean.npz", help="used only for the comparison plot")
    parser.add_argument("--no-plot", action="store_true", help="skip the comparison figure")
    parser.add_argument("--check", action="store_true",
                        help="only check that a one part ComboDriver reproduces each single error dataset")
    args = parser.parse_args()

    if args.check:
        check_combo_matches_single(args.episodes, args.seed)
        return
    n_bad = args.trips if args.trips is not None else (
        None if args.fraction is None else int(round(args.fraction * args.episodes)))
    errors = list(ERRORS) if args.error == "all" else [args.error]
    for error in errors:
        if error not in ERRORS and error != "mixed" and "+" not in error:
            parser.error(f"unknown error type {error}")
        env, d, st = collect(error, args.episodes, args.seed, n_bad)
        name = dataset_name(error, n_bad, args.episodes)
        out = Path("data") / name
        save(env, d, out)
        save_exposure(name, len(d["obs"]), st)
        d["maze_map"] = np.array(env.unwrapped.maze.maze_map)
        fig = Path("figures") / f"corrupt_{name}.png"
        if not args.no_plot:
            plot_compare(name, d, args.clean, fig)
        print(f"{name:12s} {st['flawed']} flawed trips ({st['flawed'] / args.episodes:.0%}), "
              f"{st['with_error']} actually contain the error, {st['failed']} never reached the goal; "
              f"{len(d['obs'])} rows ({st['flawed_steps']} from flawed trips, {st['dropped']} recovery steps left out)")
        print(f"             {st['events']} error events, {st['active_steps']} steps with the error active "
              f"({st['active_steps'] / len(d['obs']):.1%} of all rows)")
        print(f"             saved {out.with_suffix('.npz')}, {out.with_suffix('.csv')} and {fig}")


if __name__ == "__main__":
    main()
