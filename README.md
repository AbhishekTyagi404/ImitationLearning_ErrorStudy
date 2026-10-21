# Impact of Demonstration Error Types on Imitation Learning (PointMaze)

CS59300-ILR course project. Question: which kinds of mistakes in demonstration data hurt a
behavior-cloning navigation policy the most, and do combinations of mistakes hurt more than the sum of the parts?

## Status: Bricks 1-5 are done, Brick 6 is next

| Brick | What | File | Done |
| --- | --- | --- | --- |
| 1 | Environment installs and runs | `src/check_env.py` | yes |
| 2 | Clean expert demonstrations (500 trips) | `src/make_expert_data.py` | yes |
| 2 | Look at the data | `src/plot_trajectories.py` | yes |
| 3 | Train clean BC baseline | `src/train_bc.py` | yes |
| 4 | Evaluation harness (4 metrics, 200 fixed test trips) | `src/evaluate.py` | yes |
| 5 | First mistake makers: wrong left/right turn, speed error | `src/corrupt.py` | yes |
| 6 | Remaining mistakes: delayed turn, obstacle mistake, missing recovery | `src/corrupt.py` | next |
| 7 | Combinations | | |
| 8 | Diagnostic classifier (bonus) | | |

## Results so far (midterm, 2026-10-21)

3 seeds per dataset, 200 fixed test trips (`TEST_SEED = 12345`), mean ± standard deviation across seeds.
20% of the 500 training trips are corrupted in each dataset (the same 100 trips for every error type).

| Policy | Success | Collisions | Path efficiency | Time to goal (steps) |
| --- | --- | --- | --- | --- |
| Expert driver (upper bound) | 100% | 0% | 1.000 | 211 |
| BC, clean data | 89.7 ± 2.8% | 32.3 ± 2.8% | 0.871 ± 0.007 | 257 ± 1 |
| BC, speed_fast | 90.3 ± 1.6% | **41.8 ± 4.3%** | 0.886 ± 0.013 | **234 ± 13** |
| BC, wrong_left | 90.0 ± 3.1% | 27.7 ± 2.8% | 0.859 ± 0.022 | 260 ± 12 |
| BC, wrong_right | 88.2 ± 1.5% | 32.3 ± 8.6% | 0.868 ± 0.018 | 261 ± 8 |

**First finding:** at a 20% dose no error type lowers the success rate, but the speed error clearly
transfers into the policy (about 9% faster, wall contacts on 42% of trips instead of 32%), while wrong
turns are averaged away. Behavior cloning with an MSE loss learns the average action in each state, so a
*systematic* error (every action of a flawed trip pushed the same way) survives, while a *sporadic* error
(one detour at a random junction) does not. Even clean BC falls short of the expert (10% failures, 32%
of trips touch a wall): that gap is compounding error.

Limits: 3 seeds and one corruption level so far. Next: a dose-response test (20%, 50%, 100%), 5 seeds,
the remaining error types, and the combinations.

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

## Reproduce everything

```bash
python src/check_env.py                       # last line must say ALL GOOD
python src/make_expert_data.py --episodes 500 # clean data: 500 trips, 106,543 steps
python src/plot_trajectories.py               # figures/expert_trajectories.png
python src/corrupt.py --error all             # wrong_left, wrong_right, speed_fast, speed_slow datasets + plots

# train 3 seeds per dataset (about 1 minute each on CPU)
python src/train_bc.py --data data/expert_clean.npz --seed 0
python src/train_bc.py --data data/wrong_left.npz --seed 0
python src/train_bc.py --data data/wrong_right.npz --seed 0
python src/train_bc.py --data data/speed_fast.npz --seed 0
# ... and the same with --seed 1 and --seed 2

python src/evaluate.py --expert               # scores every model in models/, writes results/results.csv
```

`evaluate.py` replaces a model's row when it is re-run, so `results/results.csv` never has duplicates.
Running it twice gives identical numbers. Per-trip details (end point, first wall contact, ...) are saved in
`results/trips/` for the diagnostic classifier (Brick 8).

A warning about `AdroitHand...` environments is printed on every run. It is about other robots in the same
package and can be ignored.

## Repository layout

```
src/        all code; src/common.py defines the environment and the expert
data/       generated datasets (.npz, .csv), not committed
models/     trained policies (.pt), not committed
figures/    plots (expert trajectories, clean vs corrupted trips)
results/    evaluation CSVs, not committed
```

Data, models and results are not committed because every script re-creates them exactly from fixed seeds.

## The world

![Expert trajectories](figures/expert_trajectories.png)

- Environment: `PointMaze_Medium-v3` (Gymnasium-Robotics). A ball with momentum in an 8x8 maze
  (outer ring is wall). A trip ends when the ball reaches the goal or after 600 steps.
- State: `[x, y, vx, vy]` plus the goal `[goal_x, goal_y]` (6 numbers for the policy).
- Action: `[fx, fy]`, a push in [-1, 1].
- Expert: BFS path over maze cells + a speed-capped PD controller (`src/common.py`). 100% success
  (500 of 500 trips, 106,543 steps, trips 15 to 463 steps long).
- Why Medium and not UMaze: in UMaze a clean BC policy already scores 100%, so there is no room to
  measure damage. In Medium, clean BC scores about 90%, which leaves room for mistakes to show.

## The mistake makers (`src/corrupt.py`)

Each corrupted dataset replays exactly the same 500 trips as the clean data (same start, goal and seed).
A fixed 20% of trips (the same 100 for every error type) are driven by a flawed driver; the other 400
are identical to the clean data. The flawed driver really drives the ball, so every recorded state is
the true result of the action before it (editing actions in the finished file would break that link).

| Error | What the flawed driver does | Trips that contain it |
| --- | --- | --- |
| `wrong_left` | At one random junction with an open corridor on its left, drives into it for 25 steps, then re-plans | 84 of 100 |
| `wrong_right` | The same, to the right | 89 of 100 |
| `speed_fast` | Speed cap 3.0 instead of 2.0 for the whole trip | 100 of 100 |
| `speed_slow` | Speed cap 1.0 (10 trips time out; not used in the midterm) | 100 of 100 |

Some wrong-turn trips have no side corridor on that side, so they stay clean.

| Wrong left turn | Wrong right turn |
| --- | --- |
| ![wrong_left](figures/corrupt_wrong_left.png) | ![wrong_right](figures/corrupt_wrong_right.png) |
| **Speed too fast** | **Speed too slow** |
| ![speed_fast](figures/corrupt_speed_fast.png) | ![speed_slow](figures/corrupt_speed_slow.png) |

Solid lines are corrupted trips, dashed lines the same trips driven by the expert.

## Metrics (`src/evaluate.py`)

| Metric | Definition | Measured on |
| --- | --- | --- |
| Success rate | Share of test trips that reach the goal within 600 steps | All trips |
| Collision rate | Share of trips with at least one real wall contact: MuJoCo contact between `particle_geom` and a `block_*` geom with distance <= 0 | All trips |
| Path efficiency | Expert's travelled distance on the same trip / policy's travelled distance (expert = 1.0) | Successful trips |
| Time to goal | Steps until the goal is reached | Successful trips |

Two definitions were changed after testing:
- MuJoCo also reports "near" contacts inside a 2 mm safety margin. Counting those raised the clean
  collision rate from about 31% to 42%, so only real touches (distance <= 0) count.
- Path efficiency was first defined against the shortest path through cell centres, but the expert itself
  scored 1.41 on that (a real ball cuts corners). The expert's own route on the same trip is the reference.

## Experiment plan

- **Mistake types:** wrong left turn, wrong right turn, delayed turn, speed error, obstacle-avoidance
  mistake, missing recovery. Each is a flawed version of the expert that re-records trips.
- **Datasets:** clean, 6 single-mistake sets (20% of trips corrupted), 3 pairs, 1 mix of all = 11 datasets.
- **Runs:** 11 datasets x 5 seeds = 55 BC policies.
- **Fairness:** every policy uses the same network and training settings and is tested on the same 200
  start/goal pairs. A difference is called real only if it is larger than about twice the seed-to-seed spread.
- **Interaction test (Brick 7):** for errors A and B,
  `interaction = (S_clean - S_A+B) - [(S_clean - S_A) + (S_clean - S_B)]`;
  positive means the pair hurts more than the two errors added together.

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
  lets the mistake makers see the maze walls.
- The CSVs (`data/*.csv`) have one row per step (with a `corrupted` column in the corrupted sets), so you can
  explore them with SQL, pandas or DuckDB.
