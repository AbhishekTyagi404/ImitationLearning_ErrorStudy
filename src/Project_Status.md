# Project status (as of 2026-09-27)

Owner does the work solo (teammates: Nishad Gupta, Richard Li are not contributing). Background: 4 yrs Azure data engineering, strong SQL/Spark, moderate Python, learning RL. Deadline: 4-6 weeks. Hardware: Ryzen 9 + RTX 4060 (GPU optional). Windows PC, runs code in Anaconda Prompt, conda env `ilproject` (Python 3.11), repo cloned via GitHub Desktop at C:\Users\mecha\Documents\GitHub\ImitationLearning_ErrorStudy. GitHub: user AbhishekTyagi404, repo ImitationLearning_ErrorStudy (public), project board https://github.com/users/AbhishekTyagi404/projects/1 (private/blocked for Claude; cards Brick 1-8 exist as issues #1-#7+, WIP limits Backlog 5, In progress 3, In review 5). The owner said "brick by brick" as an idiom (step by step); "Brick N" labels are just step names.

## Repo layout (pushed 2026-09-27, commit a633e33)
src/ (check_env.py, common.py, make_expert_data.py, plot_trajectories.py), figures/, .gitignore, README.md, requirements.txt. Run everything from repo root. A final README (status table, layout, experiment plan, design decisions) was given to the owner; train_bc.py given 2026-09-27, not yet committed.

## Decisions made (with evidence from testing)
- Environment: PointMaze_Medium-v3 (Gymnasium-Robotics 1.4.2, gymnasium 1.3). UMaze rejected: clean BC scored 100% (no room to measure damage). Medium: expert 100%.
- Do NOT use the `imitation` library: v1.0.1 pins gymnasium==0.29.1, conflicts with gymnasium-robotics>=1.2. BC is plain PyTorch (MLP 6-128-128-2 + tanh, input normalization stored as buffers, MSE, Adam 1e-3, batch 256). Mention the swap in the report.
- Train 120 epochs, not 40. Clean BC success over seeds 0,1,2: 40 epochs = 86/81/93%; 120 epochs = 98/95/94%. Re-test 2026-09-27 of the repo's train_bc.py: seed 0 = 94%, ~65 s training on 2 threads. About +-4 points of sampling noise with 100 test trips; consider 200+ test trips in Brick 4.
- Expert data generated locally (BFS over cells + speed-capped PD controller, gain 3, vmax 2, radius 0.5, k_vel 8). Owner's PC run reproduced the sandbox exactly: 500/500 trips, 106543 rows.
- Policy state = [x, y, vx, vy] + goal [gx, gy]; action = [fx, fy] in [-1, 1].
- Fixed test set: TEST_SEED=12345 in train_bc.py (test_pairs function), same 100 start/goal pairs for every policy.
- Collision detection: MuJoCo contact between `particle_geom` and a geom named `block_*` (verified).
- Experiment grid: 11 datasets (clean, 6 single at 20% corrupted trips, 3 pairs, 1 mixed) x 5 seeds = 55 runs, about 2 min each.

## Bricks
- Done on owner's PC: Brick 1 (check_env ALL GOOD), Brick 2 (data + plot).
- Brick 3 train_bc.py: written and tested in sandbox (seed 0 = 94%), handed to owner to run and commit. Saves models/<data>_seed<N>.pt.
- Next: Brick 4 `evaluate.py` (4 metrics, results CSV, more test trips; reuse BCPolicy and test_pairs from train_bc.py), Bricks 5-6 `corrupt.py` (flawed ExpertDriver subclasses, re-record trips).

## Roadmap doc
Living doc "Imitation Learning Project Roadmap: Brick by Brick": https://claude.ai/code/artifact/077547c3-612f-4773-aa14-112ca6b7281e
