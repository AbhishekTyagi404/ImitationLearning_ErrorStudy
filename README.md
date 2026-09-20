# Impact of Demonstration Error Types on Imitation Learning (PointMaze)

CS59300-ILR course project. Question: which kinds of mistakes in demonstration data hurt a
behavior-cloning navigation policy the most, and do combinations of mistakes hurt more than the sum of the parts?

## Status: Bricks 1 and 2 are done

| Brick | What | File | Done |
| --- | --- | --- | --- |
| 1 | Environment installs and runs | `src/check_env.py` | yes |
| 1-2 | Clean expert demonstrations (500 trips) | `src/make_expert_data.py` | yes |
| 2 | Look at the data | `src/plot_trajectories.py` | yes |
| 3 | Train clean BC baseline | `src/train_bc.py` | next |
| 4 | Evaluation harness (4 metrics, seeds) | `src/evaluate.py` | |
| 5-6 | Six mistake makers | `src/corrupt.py` | |
| 7 | Combinations | | |
| 8 | Diagnostic classifier (bonus) | | |

## Setup (Windows, macOS or Linux)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows.  On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/check_env.py         # last line must say ALL GOOD
python src/make_expert_data.py --episodes 500
python src/plot_trajectories.py # look at figures/expert_trajectories.png
```

## The world

- Environment: `PointMaze_Medium-v3` (Gymnasium-Robotics). A ball with momentum in an 8x8 maze.
- State: `[x, y, vx, vy]` plus the goal `[goal_x, goal_y]` (6 numbers for the policy).
- Action: `[fx, fy]`, a push in [-1, 1].
- Expert: BFS path over maze cells + a speed-capped PD controller (`src/common.py`). 100% success.
- Why Medium and not UMaze: in UMaze a clean BC policy already scores 100%, so there is no room to
  measure damage. In Medium clean BC scored 93% in a quick test, which leaves room for mistakes to show.

## Design decisions worth knowing

- **BC is written in plain PyTorch, not with the `imitation` library.** `imitation` 1.0.1 pins
  `gymnasium==0.29.1`, which conflicts with the current PointMaze (`gymnasium>=1.2`). Behavior cloning is
  supervised regression (state in, action out), about 30 lines, so nothing is lost. If your abstract
  requires the library, say in the report that you re-implemented BC for compatibility, or use a separate venv.
- **Expert data is generated locally, not downloaded from Minari.** This makes the project self-contained and
  lets the mistake makers see the maze walls. The Minari PointMaze datasets are a fine alternative.
- The CSV (`data/expert_clean.csv`) has one row per step so you can explore it with SQL, pandas or DuckDB.
