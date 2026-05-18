"""
Experiment: Online / Adaptive Anomaly Detection

For windows with distribution shift between train and test, global train-
based statistics become unreliable. This experiment tests scorers that adapt
to the test data distribution using only past test points:

- online_rolling_zscore: z-score relative to recent test history
- online_diff_zscore: z-score of first differences within test window
- online_jerk: z-score of second differences within test window

These are combined with global train-based scorers in a hybrid ensemble.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_all_window_dirs,
    load_window,
    time_based_split,
    f1_at_topk,
    summarize_results,
    save_results,
)
from scorers import (
    score_online_rolling_zscore,
    score_online_diff_zscore,
    score_online_jerk,
    score_zscore,
    score_mad,
    score_diff_zscore,
    normalize,
)


def run_experiment(sample_size: int = 100, random_seed: int = 42):
    np.random.seed(random_seed)
    window_dirs = get_all_window_dirs()
    sample = np.random.choice(window_dirs, size=min(sample_size, len(window_dirs)), replace=False)

    conditions = {
        "online_only": [],
        "global_only": [],
        "hybrid_equal": [],
        "hybrid_global_heavy": [],
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

        # Online scorers (computed on val data only, using val history)
        s_online_roll = score_online_rolling_zscore(tr_x, val_x, window=15)
        s_online_diff = score_online_diff_zscore(tr_x, val_x, window=15)
        s_online_jerk = score_online_jerk(tr_x, val_x, window=15)
        online_ensemble = normalize((s_online_roll + s_online_diff + s_online_jerk) / 3.0)

        # Global scorers (computed relative to train data)
        s_zscore = score_zscore(tr_x, val_x)
        s_mad = score_mad(tr_x, val_x)
        s_diff = score_diff_zscore(tr_x, val_x)
        global_ensemble = normalize((s_zscore + s_mad + s_diff) / 3.0)

        conditions["online_only"].append(f1_at_topk(online_ensemble, val_y, rate))
        conditions["global_only"].append(f1_at_topk(global_ensemble, val_y, rate))
        conditions["hybrid_equal"].append(f1_at_topk(
            normalize((online_ensemble + global_ensemble) / 2.0), val_y, rate
        ))
        conditions["hybrid_global_heavy"].append(f1_at_topk(
            normalize(0.25 * online_ensemble + 0.75 * global_ensemble), val_y, rate
        ))

        per_window.append({
            "window_id": window["window_id"],
            "window_name": window["window_name"],
            "train_rate": float(rate),
        })

    summary = {}
    for name, vals in conditions.items():
        if vals:
            summary[name] = summarize_results(vals)
            print(f"{name}: mean_f1={summary[name]['mean_f1']:.4f}, median={summary[name]['median_f1']:.4f}, n={len(vals)}")

    save_results("online_adaptive", {
        "description": "Online adaptive scorers vs global train-based scorers",
        "sample_size": sample_size,
        "summary": summary,
        "per_window": per_window,
    })


if __name__ == "__main__":
    run_experiment(sample_size=200)
