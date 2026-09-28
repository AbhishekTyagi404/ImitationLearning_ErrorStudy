# Impact of Demonstration Error Types on Imitation Learning (PointMaze)

CS59300-ILR course project. Question: which kinds of mistakes in demonstration data hurt a
behavior-cloning navigation policy the most, and do combinations of mistakes hurt more than the sum of the parts?

## Status: Bricks 1-3 are done, Brick 4 is next

| Brick | What | File | Done |
| --- | --- | --- | --- |
| 1 | Environment installs and runs | `src/check_env.py` | yes |
| 2 | Clean expert demonstrations (500 trips) | `src/make_expert_data.py` | yes |
| 2 | Look at the data | `src/plot_trajectories.py` | yes |
| 3 | Train clean BC baseline | `src/train_bc.py` | yes |
| 4 | Evaluation harness (4 metrics, seeds) | `src/evaluate.py` | next |
| 5-6 | Six mistake makers | `src/corrupt.py` | |
| 7 | Combinations | | |
| 8 | Diagnostic classifier (bonus) | | |

## Results so far

| Policy | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | --- | --- | --- | --- |
| Expert (the controller itself) | 100% | | | 100% |
| Clean BC, 120 epochs | 91% | 89% | 93% | 91% |

Success rate on 100 fixed test trips (`TEST_SEED = 12345`). Brick 4 will re-test on more trips and add
collision rate, path efficiency and time to goal.

## Setup (Windows, macOS or Linux)

Run everything from the repository root. Python 3.11 was used (3.10 to 3.12 should work).

With Anaconda (used for this project, Windows):

```bash
conda create -n ilproject python=3.11 -y
conda activate ilproject
pip install -r requirements.txt
```

Or with plain Python:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows.  On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then:

```bash
python src/check_env.py                     # last line must say ALL GOOD
python src/make_expert_data.py --episodes 500
python src/plot_trajectories.py             # writes figures/expert_trajectories.png
python src/train_bc.py --seed 0             # about 1 minute on CPU; repeat with --seed 1 and --seed 2
```

A warning about `AdroitHand...` environments is printed on every run. It is about other robots in the same
package and can be ignored.

## Repository layout

```
src/        all code; src/common.py defines the environment and the expert
data/       generated datasets (.npz, .csv), not committed
models/     trained policies (.pt), not committed
figures/    plots
results/    evaluation CSVs, not committed
```

Data and models are not committed because every script re-creates them exactly from fixed seeds.

## The world

![Expert trajectories](figures/expert_trajectories.png)

- Environment: `PointMaze_Medium-v3` (Gymnasium-Robotics). A ball with momentum in an 8x8 maze
  (outer ring is wall). A trip ends when the ball reaches the goal or after 600 steps.
- State: `[x, y, vx, vy]` plus the goal `[goal_x, goal_y]` (6 numbers for the policy).
- Action: `[fx, fy]`, a push in [-1, 1].
- Expert: BFS path over maze cells + a speed-capped PD controller (`src/common.py`). 100% success
  (500 of 500 trips, 106,543 steps, trips 15 to 463 steps long).
- Why Medium and not UMaze: in UMaze a clean BC policy already scores 100%, so there is no room to
  measure damage. In Medium, clean BC scores 89-93% (mean 91%), which leaves room for mistakes to show.

## Experiment plan

- **Mistake types:** wrong left turn, wrong right turn, delayed turn, speed error, obstacle-avoidance
  mistake, missing recovery. Each is a flawed version of the expert that re-records trips.
- **Datasets:** clean, 6 single-mistake sets (20% of trips corrupted), 3 pairs, 1 mix of all = 11 datasets.
- **Runs:** 11 datasets x 5 seeds = 55 BC policies.
- **Metrics:** success rate, collision rate, path efficiency, time to goal. Every policy is tested on the
  same fixed set of start/goal pairs (`TEST_SEED = 12345`).
- **Interaction test (Brick 7):** a pair of mistakes "hurts more than the sum of the parts" if its drop in
  success from the clean baseline is larger than the two single drops added together.

## Design decisions worth knowing

- **BC is written in plain PyTorch, not with the `imitation` library.** `imitation` 1.0.1 pins
  `gymnasium==0.29.1`, which conflicts with the current PointMaze (`gymnasium>=1.2`). Behavior cloning is
  supervised regression (state in, action out), so a short PyTorch script does the same job.
  Network: MLP 6-128-128-2 with ReLU and a final tanh (keeps the push in [-1, 1]); inputs are scaled by the
  training data's mean and standard deviation, saved inside the model file. MSE loss, Adam (lr 1e-3),
  batch 256, 120 epochs.
- **120 epochs, not 40.** In early tests, 40 epochs gave 81-93% across seeds; 120 epochs gives a stronger,
  steadier baseline. A weak baseline would blur the damage caused by each mistake type.
- **Results can differ by a few points between computers** (floating-point differences between CPUs and
  library builds), even with the same seed and data. All reported numbers come from one machine.
- **Expert data is generated locally, not downloaded from Minari.** This makes the project self-contained and
  lets the mistake makers see the maze walls. The Minari PointMaze datasets are a fine alternative.
- **Collisions** (Brick 4) are counted as MuJoCo contacts between the ball (`particle_geom`) and any wall
  geom (`block_*`).
- The CSV (`data/expert_clean.csv`) has one row per step so you can explore it with SQL, pandas or DuckDB.
