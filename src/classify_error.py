"""Brick 8: can the behavior of a trained policy tell us which mistake was in its training data?

Run:  python src/classify_error.py                      (reads results/trips)
      python src/classify_error.py --trips PATH

Each trained policy drives 200 test trips. We boil those trips down to eight numbers and train a small
classifier to name the error type. Evaluation is leave one seed out: train on four seeds, test on the
fifth, rotate five times, so every policy is predicted by a model that never saw its seed.

Runs:
  main    six error types at 20% (30 policies), chance 16.7%
  detect  the same plus clean policies as a seventh class (35 policies), chance 14.3%
  pooled  six error types at 20%, 50% and 100% (90 policies), chance 16.7%
  dose    hold out a whole dose: train on two doses, test on the third (only seeds the model has not seen)
  stages  stage 1 of a two stage diagnosis: clean or flawed? (stage 2 is the main run)

Writes results/brick8_features.csv, results/brick8_scores.csv, results/brick8_confusion_*.csv,
results/brick8_heldout.csv, results/brick8_stage1.csv and figures/brick8_confusion.png.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SINGLE = ["wrong_left", "wrong_right", "speed_fast", "delayed_turn", "wall_hugging", "missing_recovery"]
LABEL = {"wrong_left": "Wrong left", "wrong_right": "Wrong right", "speed_fast": "Speed fast",
         "delayed_turn": "Delayed turn", "wall_hugging": "Wall hugging", "missing_recovery": "Missing recovery",
         "expert_clean": "Clean"}
SEEDS = range(5)
STEP_CAP = 600
FEATURES = ["success_rate", "collision_rate", "timeout_rate", "step_ratio", "speed", "path_efficiency",
            "end_dist_failed", "contact_steps_per_hit"]


def policy_features(path):
    t = pd.read_csv(path)
    ok = t.success.astype(bool)
    hit = t.collided.astype(bool)
    return {
        "success_rate": ok.mean(),
        "collision_rate": hit.mean(),
        "timeout_rate": (t.steps >= STEP_CAP).mean(),
        "step_ratio": t.steps.sum() / t.expert_steps.sum(),              # slower than the expert above 1
        "speed": t.travelled.sum() / t.steps.sum(),                     # distance per step
        "path_efficiency": (t.expert_travelled[ok] / t.travelled[ok]).mean() if ok.any() else 0.0,
        "end_dist_failed": t.end_dist_to_goal[~ok].mean() if (~ok).any() else 0.0,
        "contact_steps_per_hit": t.contact_steps[hit].mean() if hit.any() else 0.0,
    }


def load(trips_dir):
    rows = []
    for error in SINGLE + ["expert_clean"]:
        for dose, suffix in ((20, ""), (50, "_p50"), (100, "_p100")):
            if error == "expert_clean" and dose != 20:
                continue
            for seed in SEEDS:
                f = Path(trips_dir) / f"{error}{suffix}_seed{seed}.csv"
                if f.exists():
                    rows.append({"error": error, "dose": dose, "seed": seed, **policy_features(f)})
    return pd.DataFrame(rows)


def models():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=1, random_state=0),
    }


def leave_one_seed_out(df, name):
    pred = pd.Series(index=df.index, dtype=object)
    for seed in SEEDS:
        train, test = df[df.seed != seed], df[df.seed == seed]
        m = models()[name].fit(train[FEATURES], train.error)
        pred[test.index] = m.predict(test[FEATURES])
    return pred


def confusion(truth, pred, classes):
    c = pd.crosstab(truth, pred).reindex(index=classes, columns=classes, fill_value=0)
    c.index.name, c.columns.name = "true", "predicted"
    return c


def merged(s):
    return s.replace({"wrong_right": "wrong_turn", "wrong_left": "wrong_turn"})


def run(df, tag, classes):
    out, mats = [], {}
    for name in models():
        pred = leave_one_seed_out(df, name)
        acc = (pred == df.error).mean() * 100
        acc_merged = (merged(pred) == merged(df.error)).mean() * 100
        per_class = {e: (pred[df.error == e] == e).mean() * 100 for e in classes}
        out.append({"run": tag, "model": name, "policies": len(df), "chance": 100 / len(classes),
                    "accuracy": acc, "accuracy_wrong_turns_merged": acc_merged,
                    **{f"recall_{e}": v for e, v in per_class.items()}})
        mats[name] = confusion(df.error, pred, classes)
        mats[name].to_csv(f"results/brick8_confusion_{tag}_{name}.csv")
    return out, mats


def heldout_dose(df, name="logistic"):
    """Train on two doses, test on the third. The test policy's seed is also left out of training, so the
    model has seen neither that dose nor that seed. Fifteen fits (3 doses x 5 seeds)."""
    pred = pd.Series(index=df.index, dtype=object)
    for dose in sorted(df.dose.unique()):
        for seed in SEEDS:
            train = df[(df.dose != dose) & (df.seed != seed)]
            test = df[(df.dose == dose) & (df.seed == seed)]
            m = models()[name].fit(train[FEATURES], train.error)
            pred[test.index] = m.predict(test[FEATURES])
    return pred


def stage_one(df, name="logistic"):
    """Clean or flawed, leave one seed out, classes weighted equally (5 clean against 30 flawed policies)."""
    y = (df.error != "expert_clean").map({True: "flawed", False: "clean"})
    pred = pd.Series(index=df.index, dtype=object)
    for seed in SEEDS:
        train, test = df[df.seed != seed], df[df.seed == seed]
        model = models()[name]
        if name == "logistic":
            model.set_params(logisticregression__class_weight="balanced")
        else:
            model.set_params(class_weight="balanced")
        model.fit(train[FEATURES], y[train.index])
        pred[test.index] = model.predict(test[FEATURES])
    return y, pred


def draw(mats, path):
    fig, axes = plt.subplots(1, len(mats), figsize=(5.2 * len(mats), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (title, c) in zip(axes, mats.items()):
        share = c.div(c.sum(axis=1), axis=0).values * 100
        ax.imshow(share, cmap="Blues", vmin=0, vmax=100)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                ax.text(j, i, f"{c.values[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 55 else "#0b0b0b")
        names = [LABEL[x] for x in c.index]
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel("Predicted error", fontsize=9, color="#52514e")
        ax.set_ylabel("True error", fontsize=9, color="#52514e")
        ax.set_title(title, loc="left", fontsize=10.5, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", default="results/trips")
    args = ap.parse_args()
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    df = load(args.trips)
    df.round(4).to_csv("results/brick8_features.csv", index=False)

    main_df = df[(df.dose == 20) & (df.error != "expert_clean")].reset_index(drop=True)
    detect_df = df[df.dose == 20].reset_index(drop=True)
    pooled_df = df[df.error != "expert_clean"].reset_index(drop=True)

    scores, shown = [], {}
    for tag, d, classes in (("main", main_df, SINGLE), ("detect", detect_df, SINGLE + ["expert_clean"]),
                            ("pooled", pooled_df, SINGLE)):
        s, mats = run(d, tag, classes)
        scores += s
        if tag in ("main", "pooled"):
            shown[f"{'Six errors at 20%' if tag == 'main' else 'Six errors, all doses'}: logistic regression"] = mats["logistic"]
    pd.DataFrame(scores).round(1).to_csv("results/brick8_scores.csv", index=False)

    held = []
    for name in models():
        pred = heldout_dose(pooled_df, name)
        for dose in sorted(pooled_df.dose.unique()):
            m = pooled_df.dose == dose
            held.append({"model": name, "held_out_dose": f"{dose}%", "policies": int(m.sum()),
                         "accuracy": (pred[m] == pooled_df.error[m]).mean() * 100,
                         "accuracy_wrong_turns_merged": (merged(pred[m]) == merged(pooled_df.error[m])).mean() * 100})
        held.append({"model": name, "held_out_dose": "all three", "policies": len(pooled_df),
                     "accuracy": (pred == pooled_df.error).mean() * 100,
                     "accuracy_wrong_turns_merged": (merged(pred) == merged(pooled_df.error)).mean() * 100})
        if name == "logistic":
            confusion(pooled_df.error, pred, SINGLE).to_csv("results/brick8_confusion_heldout_logistic.csv")
    held = pd.DataFrame(held)
    held.round(1).to_csv("results/brick8_heldout.csv", index=False)

    stage = []
    for name in models():
        y, pred = stage_one(detect_df, name)
        stage.append({"model": name, "policies": len(y), "accuracy": (pred == y).mean() * 100,
                      "clean_found": (pred[y == "clean"] == "clean").mean() * 100,
                      "flawed_found": (pred[y == "flawed"] == "flawed").mean() * 100,
                      "balanced_accuracy": ((pred[y == "clean"] == "clean").mean()
                                            + (pred[y == "flawed"] == "flawed").mean()) * 50})
    stage = pd.DataFrame(stage)
    stage.round(1).to_csv("results/brick8_stage1.csv", index=False)
    draw(shown, "figures/brick8_confusion.png")

    print(pd.DataFrame(scores)[["run", "model", "policies", "chance", "accuracy",
                                "accuracy_wrong_turns_merged"]].round(1).to_string(index=False))
    print("\nPer class recall (%):")
    print(pd.DataFrame(scores).filter(regex="^(run|model|recall_)").round(0).to_string(index=False))
    for tag in ("main", "pooled"):
        print(f"\nConfusion, {tag}, logistic regression (rows true, columns predicted):")
        print(pd.read_csv(f"results/brick8_confusion_{tag}_logistic.csv", index_col=0).to_string())
    print("\nHold out a whole dose (train on two doses and four seeds, test on the third dose and fifth seed):")
    print(held.round(1).to_string(index=False))
    print("\nStage 1, clean or flawed (35 policies at 20%, leave one seed out, classes weighted equally):")
    print(stage.round(1).to_string(index=False))
    print("\nFeature means at 20% by error:")
    print(main_df.groupby("error")[FEATURES].mean().round(3).to_string())


if __name__ == "__main__":
    main()
