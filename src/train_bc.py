"""BRICK 3: train a behavior-cloning (BC) policy and give it a quick driving test.

Run:  python src/train_bc.py                                   (clean data, seed 0)
      python src/train_bc.py --data data/expert_clean.npz --seed 1

Behavior cloning = plain supervised learning: input the state, predict the expert's push.
  input  (6 numbers): x, y, vx, vy, goal_x, goal_y
  output (2 numbers): fx, fy in [-1, 1]

Writes:
  models/<data name>_seed<N>.pt   the trained network plus its input scaling
Prints the success rate on a FIXED set of test trips (same trips for every policy, so results compare).
The full evaluation with all four metrics comes in Brick 4 (src/evaluate.py).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import MAX_STEPS, free_cells, make_env

TEST_SEED = 12345  # fixes the test start/goal pairs; never change it between experiments


# ---------------------------------------------------------------- the network
class BCPolicy(nn.Module):
    """MLP 6 -> 128 -> 128 -> 2. Tanh at the end keeps the push inside [-1, 1]."""

    def __init__(self, in_dim=6, hidden=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.Tanh(),
        )
        # input scaling (mean/std of the training data), saved with the model
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)

    @torch.no_grad()
    def act(self, obs, goal):
        x = torch.as_tensor(np.concatenate([obs, goal]), dtype=torch.float32)
        return self(x).numpy()


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    X = np.concatenate([d["obs"], d["goal"]], axis=1).astype(np.float32)
    Y = d["action"].astype(np.float32)
    return X, Y, len(np.unique(d["episode"]))


# ---------------------------------------------------------------- training
def train(X, Y, seed, epochs=120, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = BCPolicy()
    policy.mu.copy_(torch.from_numpy(X.mean(0)))
    policy.sd.copy_(torch.from_numpy(X.std(0) + 1e-6))

    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(Xt), generator=g)
        total = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            loss = loss_fn(policy(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep == 1 or ep % 20 == 0:
            print(f"  epoch {ep:3d}/{epochs}   loss {total / len(Xt):.4f}")
    return policy


# ---------------------------------------------------------------- quick driving test
def test_pairs(env, n):
    """The fixed list of (start cell, goal cell) test trips."""
    cells = free_cells(env)
    rng = np.random.default_rng(TEST_SEED)
    pairs = []
    for _ in range(n):
        s, g = rng.choice(len(cells), 2, replace=False)
        pairs.append((cells[s], cells[g], int(rng.integers(1_000_000_000))))
    return pairs


def quick_test(policy, n_trips):
    env = make_env()
    wins = 0
    for start, goal, reset_seed in test_pairs(env, n_trips):
        obs, _ = env.reset(seed=reset_seed, options={"reset_cell": np.array(start), "goal_cell": np.array(goal)})
        info = {"success": False}
        for _ in range(MAX_STEPS):
            a = policy.act(obs["observation"], obs["desired_goal"])
            obs, _, terminated, truncated, info = env.step(a)
            if terminated or truncated:
                break
        wins += bool(info["success"])
    return wins / n_trips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/expert_clean.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--test-trips", type=int, default=100)
    args = parser.parse_args()

    torch.set_num_threads(2)  # small network: more threads only adds overhead
    X, Y, n_trips = load_dataset(args.data)
    print(f"Data: {args.data}   {n_trips} trips, {len(X)} steps   seed {args.seed}")

    t0 = time.time()
    policy = train(X, Y, args.seed, epochs=args.epochs)
    print(f"Training took {time.time() - t0:.0f} s")

    out = Path("models") / f"{Path(args.data).stem}_seed{args.seed}.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save(policy.state_dict(), out)
    print(f"Saved {out}")

    t0 = time.time()
    rate = quick_test(policy, args.test_trips)
    print(f"Quick test: success {rate:.0%} on {args.test_trips} fixed test trips ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
