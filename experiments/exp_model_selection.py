"""
Experiment: Per-Window Model Selection

For each window, evaluate all unsupervised scorers on the training data
and select the one with the highest train F1. Apply the selected scorer
to the validation/test data.

Hypothesis: different windows have different anomaly signatures, so a
one-size-fits-all ensemble may be suboptimal.
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
from scorers import ALL_SCORERS


def run_experiment(sample_size: int = 100, random_seed: int = 42):
    np.random.seed(random_seed)
    window_dirs = get_all_window_dirs()
    sample = np.random.choice(window_dirs, size=min(sample_size, len(window_dirs)), replace=False)

    per_window = []
    scorer_names = [k for k in ALL_SCORERS.keys() if k != "supervised_rf" and k != "autoencoder"]

    for wdir in sample:
        window = load_window(wdir)
        split = time_based_split(window["train_x"], window["train_y"])
        if split is None:
            continue
        tr_x, tr_y, val_x, val_y = split
        if np.sum(tr_y) == 0:
            continue

        rate = np.mean(tr_y)
        best_name = None
        best_f1 = -1.0

        for name in scorer_names:
            try:
                scorer = ALL_SCORERS[name]
                train_scores = scorer(tr_x, tr_x, tr_y)
                f1 = f1_at_topk(train_scores, tr_y, rate)
                if f1 > best_f1:
                    best_f1 = f1
                    best_name = name
            except Exception:
                continue

        if best_name is None:
            continue

        try:
            val_scores = ALL_SCORERS[best_name](tr_x, val_x, tr_y)
            val_f1 = f1_at_topk(val_scores, val_y, rate)
        except Exception:
            continue

        per_window.append({
            "window_id": window["window_id"],
            "window_name": window["window_name"],
            "f1": val_f1,
            "train_rate": float(rate),
            "selected_scorer": best_name,
            "train_f1_of_selected": best_f1,
        })

    summary = summarize_results([w["f1"] for w in per_window])
    print(f"Model Selection: {summary}")

    # Count how often each scorer was selected
    selection_counts = {}
    for w in per_window:
        selection_counts[w["selected_scorer"]] = selection_counts.get(w["selected_scorer"], 0) + 1

    save_results("model_selection", {
        "description": "Per-window model selection based on training F1",
        "sample_size": sample_size,
        "summary": summary,
        "selection_counts": selection_counts,
        "per_window": per_window,
    })


if __name__ == "__main__":
    run_experiment(sample_size=200)
