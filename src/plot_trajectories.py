"""BRICK 2: look at the data.  Run:  python src/plot_trajectories.py

Draws a few expert trips on top of the maze so you can see what 'good driving' looks like.
Later you will use the same function to compare clean vs corrupted trips.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw_maze(ax, maze_map):
    """Walls are grey squares. x = col - center_col, y = center_row - row (same as PointMaze)."""
    n_rows, n_cols = maze_map.shape
    for r in range(n_rows):
        for c in range(n_cols):
            if maze_map[r, c] == 1:
                x, y = c - n_cols / 2 + 0.5, n_rows / 2 - r - 0.5
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="0.75"))
    ax.set_xlim(-n_cols / 2, n_cols / 2)
    ax.set_ylim(-n_rows / 2, n_rows / 2)
    ax.set_aspect("equal")


def plot_episodes(ax, data, episodes, color=None, label=None):
    for i, e in enumerate(episodes):
        c = color or plt.cm.tab10(i % 10)          # one color per trip, shared by line, start and goal
        m = data["episode"] == e
        xy = data["obs"][m][:, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=c, alpha=0.8, label=label if i == 0 else None)
        ax.scatter(*xy[0], marker="o", s=40, color=c, zorder=3)                # start
        ax.scatter(*data["goal"][m][0], marker="*", s=140, color=c, zorder=3)  # goal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/expert_clean.npz")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--out", default="figures/expert_trajectories.png")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    episodes = np.unique(data["episode"])[: args.n]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    draw_maze(axes[0], data["maze_map"])
    plot_episodes(axes[0], data, episodes)
    axes[0].set_title(f"{len(episodes)} expert trips (circle = start, star = goal)")

    # Speed over time for the same trips: this is where 'speed errors' will show up later
    for i, e in enumerate(episodes):
        m = data["episode"] == e
        speed = np.linalg.norm(data["obs"][m][:, 2:4], axis=1)
        axes[1].plot(speed, color=plt.cm.tab10(i % 10))
    axes[1].set_title("Expert speed over time")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("speed")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
