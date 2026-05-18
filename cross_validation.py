"""
Cross-window leave-N-out validation.

Our existing `validation.py` uses 70/30 time-split WITHIN each holdout
window. This has been giving us misleading signals: v22/v32/v34 all "won" on
this validation but LOST on the leaderboard.

The likely reason: the 70/30 time-split's "test" portion (last 30% of train_x)
is drawn from the same distribution as the "train" portion (first 70%). The
CW model has effectively seen this window's distribution at training time. So
the val measures "interpolation accuracy" rather than "cross-window
generalization", which is closer to what the LB measures.

This module replaces that with a **cross-window leave-N-out** harness:

  1. Hold out N whole windows from training.
  2. Train the model on the OTHER (1000 − N) windows' train portions.
  3. For each held-out window, predict on its `train_x` (which the model has
     truly never seen as labels).
  4. F1 = per-point F1 averaged over held-out windows.

This is closer to LB because the model has never seen the held-out windows'
distribution at training time.

NOTE: holdout windows have known labels (train_y), so this is honest evaluation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from validation import (
    Window, all_window_dirs, load_window, point_f1, stratified_holdout,
)


def cross_window_evaluate(
    predictor: Callable[[Window], np.ndarray],
    holdout: Sequence[Path],
) -> dict:
    """Evaluate a predictor on held-out whole windows.

    `predictor(window)` is called with the full Window object and must return
    binary 0/1 predictions of length len(window.train_x).

    F1 is computed per-window against `window.train_y`, then averaged.
    """
    per_window = []
    per_category = defaultdict(list)

    for wdir in holdout:
        w = load_window(wdir)
        try:
            pred = predictor(w)
            pred = np.asarray(pred).astype(int).ravel()
            if len(pred) != len(w.train_x):
                raise ValueError(f"len(pred)={len(pred)} but len(train_x)={len(w.train_x)}")
        except Exception as e:
            print(f"  ! {w.wid}: predictor failed ({e})")
            pred = np.zeros(len(w.train_x), dtype=int)

        f1 = point_f1(w.train_y, pred)
        per_window.append({
            "wid": w.wid, "metric_type": w.metric_type, "f1": f1,
            "n": int(len(w.train_x)),
            "true_pos": int(w.train_y.sum()),
        })
        per_category[w.metric_type].append(f1)

    overall = float(np.mean([r["f1"] for r in per_window]))
    by_type = {mt: {"n": len(v), "mean_f1": float(np.mean(v))}
               for mt, v in per_category.items()}
    return {"overall_f1": overall, "by_metric_type": by_type, "per_window": per_window}


def print_summary_v2(report: dict, name: str = "") -> None:
    header = f" [{name}] " if name else " "
    print(f"\n──────{header}cross-window LOO summary ──────")
    print(f"  overall mean F1 : {report['overall_f1']:.4f}  (n={len(report['per_window'])})")
    print("  by metric_type:")
    for mt, stat in sorted(report["by_metric_type"].items()):
        print(f"    {mt:<28} n={stat['n']:>3}  F1={stat['mean_f1']:.4f}")
    print()


if __name__ == "__main__":
    # Sanity check: zero-prediction baseline
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=42)
    print(f"Train pool: {len(train_pool)} windows, Holdout: {len(holdout)} windows")

    def zero_predictor(w: Window) -> np.ndarray:
        return np.zeros(len(w.train_x), dtype=int)

    rep = cross_window_evaluate(zero_predictor, holdout)
    print_summary_v2(rep, "zero-baseline (CW-LOO)")
