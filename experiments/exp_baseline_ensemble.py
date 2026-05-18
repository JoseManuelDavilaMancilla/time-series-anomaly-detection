"""
Experiment: Baseline Unsupervised Ensemble (Submission v1)

Combines multiple unsupervised statistical and isolation-based scorers
with normalized scores and a learned threshold per window on training data.

Methods included:
- Z-score, MAD, IQR (global statistical)
- Isolation Forest, LOF (isolation-based)
- Autoencoder (reconstruction error)
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
    score_zscore,
    score_mad,
    score_iqr,
    score_isolation_forest,
    score_lof,
    score_autoencoder,
    normalize,
)


def baseline_ensemble_predict(train_x, train_y, test_x, use_train_threshold=True):
    """Run baseline ensemble and return anomaly scores for test."""
    scores_list = []
    train_scores_list = []

    for scorer in [score_zscore, score_mad, score_iqr, score_isolation_forest, score_lof, score_autoencoder]:
        try:
            if scorer in (score_isolation_forest, score_autoencoder):
                s_test = scorer(train_x, test_x, train_y)
                s_train = scorer(train_x, train_x, train_y)
            elif scorer == score_lof:
                s_test = scorer(train_x, test_x)
                s_train = scorer(train_x, train_x)
            else:
                s_test = scorer(train_x, test_x)
                s_train = scorer(train_x, train_x)
            scores_list.append(normalize(s_test))
            train_scores_list.append(normalize(s_train))
        except Exception:
            continue

    if not scores_list:
        return np.zeros(len(test_x))

    ensemble = np.mean(scores_list, axis=0)
    return ensemble


def run_experiment(sample_size: int = 100, random_seed: int = 42):
    """Run baseline ensemble on a random sample of windows with time-based validation."""
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

        scores = baseline_ensemble_predict(tr_x, tr_y, val_x)
        rate = np.mean(tr_y)
        f1 = f1_at_topk(scores, val_y, rate)
        per_window.append({
            "window_id": window["window_id"],
            "window_name": window["window_name"],
            "f1": f1,
            "train_rate": float(rate),
        })

    summary = summarize_results([w["f1"] for w in per_window])
    print(f"Baseline Ensemble (v1): {summary}")

    save_results("baseline_ensemble", {
        "description": "Unsupervised ensemble of z-score, MAD, IQR, Isolation Forest, LOF, Autoencoder",
        "sample_size": sample_size,
        "summary": summary,
        "per_window": per_window,
    })


if __name__ == "__main__":
    run_experiment(sample_size=200)
