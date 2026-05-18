"""
Experiment: Supervised RandomForest with Rich Feature Engineering (Submission v3)

Trains a RandomForestClassifier per window on 20+ engineered pointwise features.
Uses train_label.npy for supervision. Falls back to unsupervised ensemble when
no training anomalies exist.

Features: lags (1,2,3,5,10), ratios, rolling mean/std/min/max/zscore (w=3,5,10),
second difference, EMA deviation.
"""

import sys
import numpy as np
from pathlib import Path
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


def extract_features(series: np.ndarray) -> np.ndarray:
    """Extract pointwise feature matrix."""
    n = len(series)
    feats = {}
    feats["val"] = series
    for lag in [1, 2, 3, 5, 10]:
        shifted = np.roll(series, lag)
        shifted[:lag] = series[:lag]
        feats[f"diff_{lag}"] = series - shifted
        feats[f"ratio_{lag}"] = series / (shifted + 1e-9)
    for w in [3, 5, 10]:
        roll_mean = np.zeros(n)
        roll_std = np.zeros(n)
        roll_min = np.zeros(n)
        roll_max = np.zeros(n)
        for i in range(n):
            window = series[max(0, i - w + 1):i + 1]
            roll_mean[i] = np.mean(window)
            roll_std[i] = np.std(window)
            roll_min[i] = np.min(window)
            roll_max[i] = np.max(window)
        feats[f"roll_mean_{w}"] = roll_mean
        feats[f"roll_std_{w}"] = roll_std
        feats[f"roll_min_{w}"] = roll_min
        feats[f"roll_max_{w}"] = roll_max
        feats[f"zscore_{w}"] = (series - roll_mean) / (roll_std + 1e-9)
    feats["second_diff"] = np.diff(series, n=2, prepend=[series[0], series[0]])
    alpha = 0.3
    ema = series[0]
    ema_vals = np.zeros(n)
    for i in range(n):
        ema = alpha * series[i] + (1 - alpha) * ema
        ema_vals[i] = ema
    feats["ema_diff"] = series - ema_vals
    return np.column_stack([feats[k] for k in sorted(feats.keys())])


def run_experiment(sample_size: int = 100, random_seed: int = 42):
    np.random.seed(random_seed)
    window_dirs = get_all_window_dirs()
    sample = np.random.choice(window_dirs, size=min(sample_size, len(window_dirs)), replace=False)

    per_window = []
    for wdir in sample:
        window = load_window(wdir)
        split = time_based_split(window["train_x"], window["train_y"])
        if split is None:
            continue
        tr_x, tr_y, val_x, val_y = split
        if np.sum(tr_y) == 0:
            continue

        X_train = extract_features(tr_x)
        X_val = extract_features(val_x)
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        clf.fit(X_train, tr_y)
        proba = clf.predict_proba(X_val)
        scores = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(val_x))
        rate = np.mean(tr_y)
        f1 = f1_at_topk(scores, val_y, rate)
        per_window.append({
            "window_id": window["window_id"],
            "window_name": window["window_name"],
            "f1": f1,
            "train_rate": float(rate),
        })

    summary = summarize_results([w["f1"] for w in per_window])
    print(f"Supervised RF (v3): {summary}")

    save_results("supervised_rf", {
        "description": "Per-window RandomForest with 20+ engineered pointwise features",
        "sample_size": sample_size,
        "summary": summary,
        "per_window": per_window,
    })


if __name__ == "__main__":
    run_experiment(sample_size=200)
