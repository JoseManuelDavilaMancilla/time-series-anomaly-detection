"""
Experiment: Cross-Window Training

Instead of training a separate model per window, train a single model across
all windows. This tests whether anomaly patterns generalize across different
metrics and systems.

For each window, we extract the same pointwise features and hold out that
window for validation (leave-one-out style, but using a random sample for speed).
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_all_window_dirs,
    load_window,
    f1_at_topk,
    summarize_results,
    save_results,
)


def extract_features(series: np.ndarray) -> np.ndarray:
    """Extract pointwise feature matrix (same as supervised RF)."""
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


def run_experiment(sample_size: int = 50, random_seed: int = 42):
    """
    Build a cross-window dataset from a sample of windows, train a single RF,
    and evaluate on held-out windows.
    """
    np.random.seed(random_seed)
    window_dirs = get_all_window_dirs()
    sample = np.random.choice(window_dirs, size=min(sample_size, len(window_dirs)), replace=False)

    # Build cross-window dataset
    X_all, y_all = [], []
    window_meta = []
    for wdir in sample:
        window = load_window(wdir)
        train_x = window["train_x"]
        train_y = window["train_y"]
        if np.sum(train_y) == 0:
            continue
        X = extract_features(train_x)
        X_all.append(X)
        y_all.append(train_y)
        window_meta.append({
            "window_id": window["window_id"],
            "window_name": window["window_name"],
            "n_points": len(train_x),
            "anomaly_rate": float(np.mean(train_y)),
        })

    if len(X_all) < 5:
        print("Not enough windows with anomalies for cross-window experiment")
        return

    # Leave-one-out evaluation: for each window, train on all others
    per_window = []
    for i in range(len(X_all)):
        X_train = np.vstack([X_all[j] for j in range(len(X_all)) if j != i])
        y_train = np.hstack([y_all[j] for j in range(len(y_all)) if j != i])
        X_val = X_all[i]
        y_val = y_all[i]

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)
        scores = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(y_val))

        rate = np.mean(y_val)
        f1 = f1_at_topk(scores, y_val, rate)
        per_window.append({
            "window_id": window_meta[i]["window_id"],
            "window_name": window_meta[i]["window_name"],
            "f1": f1,
            "train_rate": float(rate),
        })

    summary = summarize_results([w["f1"] for w in per_window])
    print(f"Cross-Window RF (leave-one-out): {summary}")

    save_results("cross_window", {
        "description": "Single RF trained across all windows, leave-one-out validation",
        "sample_size": sample_size,
        "n_windows_used": len(X_all),
        "summary": summary,
        "per_window": per_window,
    })


if __name__ == "__main__":
    run_experiment(sample_size=100)
