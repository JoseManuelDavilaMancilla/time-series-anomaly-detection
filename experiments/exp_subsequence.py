"""
Experiment: Subsequence / Discord-Based Anomaly Detection

Anomalies often appear as anomalous subsequences (shapelets) rather than
individual extreme points. This experiment tests:

- Nearest-neighbor subsequence distance: for each test subsequence of length m,
  compute its z-normalized Euclidean distance to the nearest training subsequence.
  High distance = anomalous subsequence.

Window sizes tested: m=3, 5, 7, 10
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_all_window_dirs,
    load_window,
    time_based_split,
    f1_at_topk,
    summarize_results,
    save_results,
)


def znorm(seq: np.ndarray) -> np.ndarray:
    """Z-normalize a sequence."""
    std = np.std(seq)
    if std < 1e-9:
        return seq - np.mean(seq)
    return (seq - np.mean(seq)) / std


def extract_subsequences(series: np.ndarray, m: int) -> np.ndarray:
    """Extract all z-normalized subsequences of length m."""
    n = len(series)
    subs = []
    for i in range(n - m + 1):
        subs.append(znorm(series[i:i + m]))
    return np.array(subs)


def subsequence_anomaly_score(train_x: np.ndarray, test_x: np.ndarray, m: int = 5) -> np.ndarray:
    """
    For each test subsequence of length m, find distance to nearest train subsequence.
    Score is assigned to the center point of the subsequence.
    """
    train_subs = extract_subsequences(train_x, m)
    test_subs = extract_subsequences(test_x, m)

    if len(train_subs) == 0 or len(test_subs) == 0:
        return np.zeros(len(test_x))

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(train_subs)
    dists, _ = nn.kneighbors(test_subs)

    scores = np.zeros(len(test_x))
    counts = np.zeros(len(test_x))
    for i in range(len(test_subs)):
        center = i + m // 2
        scores[center] += dists[i, 0]
        counts[center] += 1

    mask = counts > 0
    scores[mask] /= counts[mask]
    if np.any(~mask):
        scores[~mask] = np.mean(scores[mask]) if np.any(mask) else 0.0
    return scores


def run_experiment(sample_size: int = 100, random_seed: int = 42):
    np.random.seed(random_seed)
    window_dirs = get_all_window_dirs()
    sample = np.random.choice(window_dirs, size=min(sample_size, len(window_dirs)), replace=False)

    conditions = {f"m{m}": [] for m in [3, 5, 7, 10]}
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

        for m in [3, 5, 7, 10]:
            try:
                scores = subsequence_anomaly_score(tr_x, val_x, m)
                f1 = f1_at_topk(scores, val_y, rate)
                conditions[f"m{m}"].append(f1)
            except Exception:
                pass

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

    save_results("subsequence_discord", {
        "description": "Subsequence-based anomaly detection using nearest-neighbor distance",
        "sample_size": sample_size,
        "summary": summary,
        "per_window": per_window,
    })


if __name__ == "__main__":
    run_experiment(sample_size=200)
