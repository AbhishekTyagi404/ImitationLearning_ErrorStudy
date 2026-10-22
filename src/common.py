"""Shared helpers: environment creation, maze geometry and the expert driver.

Every other script imports from this file, so the environment and the expert are
defined in one place.

PointMaze conventions used in the project:
  observation = [x, y, vx, vy]
  action      = [fx, fy] in [-1, 1], a force, so the ball keeps its momentum
  x = column - center_col,  y = center_row - row   (y points up while rows count down)
"""
from __future__ import annotations

import os
from collections import deque

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

ENV_ID = os.environ.get("IL_ENV", "PointMaze_Medium-v3")   # UMaze is too easy: clean BC already scores 100%
MAX_STEPS = int(os.environ.get("IL_MAX_STEPS", 600))   # Medium trips take up to ~460 steps


def make_env(env_id: str = ENV_ID, max_steps: int = MAX_STEPS):
    """Create the maze. continuing_task=False means the episode ends when the goal is reached."""
    return gym.make(env_id, max_episode_steps=max_steps, continuing_task=False)


def free_cells(env) -> list[tuple[int, int]]:
    """All (row, col) cells that are not walls."""
    grid = np.array(env.unwrapped.maze.maze_map)
    rows, cols = np.where(grid != 1)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def xy_to_cell(env, xy) -> tuple[int, int]:
    rc = env.unwrapped.maze.cell_xy_to_rowcol(np.asarray(xy, dtype=float))
    return int(rc[0]), int(rc[1])


def cell_to_xy(env, cell) -> np.ndarray:
    return np.asarray(env.unwrapped.maze.cell_rowcol_to_xy(np.array(cell)), dtype=float)


def bfs_path(env, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Shortest path between two cells as a list of cells, start first, found by breadth first search."""
    grid = np.array(env.unwrapped.maze.maze_map)
    prev = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1]:
                if grid[nxt] != 1 and nxt not in prev:
                    prev[nxt] = cur
                    queue.append(nxt)
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


class ExpertDriver:
    """Expert driver: plans a path over maze cells, then steers toward the center of each cell.

    Two layers of control:
      1. Desired velocity = gain * (waypoint - position), capped at `vmax`.
      2. Force = k_vel * (desired velocity - current velocity), clipped to [-1, 1].
    The default settings came from a grid search: 100% success over 150 random start and goal pairs
    in PointMaze Medium, with room left to drive slower or faster for the speed error datasets.
    """

    def __init__(self, env, gain: float = 3.0, vmax: float = 2.0, radius: float = 0.5, k_vel: float = 8.0):
        self.env = env
        self.gain, self.vmax, self.radius, self.k_vel = gain, vmax, radius, k_vel

    def plan(self, pos, goal_xy):
        """Call once after env.reset(): computes the cell path from the current position to the goal."""
        self.goal_xy = np.asarray(goal_xy, dtype=float)
        self.path = bfs_path(self.env, xy_to_cell(self.env, pos), xy_to_cell(self.env, goal_xy))
        self.idx = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Maps obs = [x, y, vx, vy] to action = [fx, fy]."""
        pos, vel = obs[:2], obs[2:4]
        # switch to the next waypoint once the ball is close enough to the current one
        while self.idx < len(self.path) - 1 and np.linalg.norm(cell_to_xy(self.env, self.path[self.idx]) - pos) < self.radius:
            self.idx += 1
        target = self.goal_xy if self.idx >= len(self.path) - 1 else cell_to_xy(self.env, self.path[self.idx])
        v_des = self.gain * (target - pos)
        speed = np.linalg.norm(v_des)
        if speed > self.vmax:
            v_des = v_des * (self.vmax / speed)
        return np.clip(self.k_vel * (v_des - vel), -1.0, 1.0).astype(np.float32)
