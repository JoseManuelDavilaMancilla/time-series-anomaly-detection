"""
Experiment: Scorer Stacking / Learned Ensemble Weighting

Instead of averaging scorer outputs, train a small meta-model per window
(RandomForest or LogisticRegression) to combine scorer outputs. The meta-
model is trained on the training data and applied to validation/test.

Also tests: mean ensemble, weighted average by train F1, top-3 ensemble.
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_all_window_dirs,
    load_window,
    time_based_split,
    f1_at_topk,
    summarize_results,
    save_results,
)
from scorers import ALL_SCORERS


def run_experiment(sample_size: int = 100, random_seed: int = 42):
    np.random.seed(random_seed)
    window_dirs = get_all_window_dirs()
    sample = np.random.choice(window_dirs, size=min(sample_size, len(window_dirs)), replace=False)

    scorer_names = [k for k in ALL_SCORERS.keys() if k not in ("supervised_rf", "autoencoder")]

    results = {
        "mean_ensemble": [],
        "weighted_softmax": [],
        "top3_ensemble": [],
        "stack_lr": [],
        "stack_rf": [],
    }
    per_window = []

    for wdir in sample:
        window = load_window(wdir)
        split = time_based_split(window["train_x"], window["train_y"])
        if split is None:
            continue
        tr_x, tr_y, val_x, val_y = split
        if np.sum(tr_y) == 0:
            continue

        rate = np.mean(tr_y)
        S_train = np.zeros((len(tr_x), len(scorer_names)))
        S_val = np.zeros((len(val_x), len(scorer_names)))
        train_f1s = []

        valid_scorers = []
        for j, name in enumerate(scorer_names):
            try:
                scorer = ALL_SCORERS[name]
                s_train = scorer(tr_x, tr_x, tr_y)
                s_val = scorer(tr_x, val_x, tr_y)
                S_train[:, j] = s_train
                S_val[:, j] = s_val
                train_f1s.append(f1_at_topk(s_train, tr_y, rate))
                valid_scorers.append(j)
            except Exception:
                train_f1s.append(0.0)

        if not valid_scorers:
            continue

        # Mean ensemble
        mean_scores = np.mean(S_val, axis=1)
        results["mean_ensemble"].append(f1_at_topk(mean_scores, val_y, rate))

        # Weighted by softmax of train F1
        weights = np.exp(np.array(train_f1s) * 3)
        weights = weights / np.sum(weights)
        weighted_scores = S_val @ weights
        results["weighted_softmax"].append(f1_at_topk(weighted_scores, val_y, rate))

        # Top-3 ensemble
        top3_idx = np.argsort(train_f1s)[-3:]
        top3_scores = np.mean(S_val[:, top3_idx], axis=1)
        results["top3_ensemble"].append(f1_at_topk(top3_scores, val_y, rate))

        # Stack with LogisticRegression
        try:
            lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5)
            lr.fit(S_train, tr_y)
            proba = lr.predict_proba(S_val)[:, 1]
            results["stack_lr"].append(f1_at_topk(proba, val_y, rate))
        except Exception:
            pass

        # Stack with small RF
        try:
            rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42, class_weight="balanced")
            rf.fit(S_train, tr_y)
            proba = rf.predict_proba(S_val)[:, 1]
            results["stack_rf"].append(f1_at_topk(proba, val_y, rate))
        except Exception:
            pass

        per_window.append({
            "window_id": window["window_id"],
            "window_name": window["window_name"],
            "train_rate": float(rate),
        })

    summary = {}
    for name, vals in results.items():
        if vals:
            summary[name] = summarize_results(vals)
            print(f"{name}: mean_f1={summary[name]['mean_f1']:.4f}, median={summary[name]['median_f1']:.4f}, n={len(vals)}")

    save_results("stacking", {
        "description": "Stacking and ensemble weighting strategies for unsupervised scorers",
        "sample_size": sample_size,
        "summary": summary,
        "per_window": per_window,
    })


if __name__ == "__main__":
    run_experiment(sample_size=200)
