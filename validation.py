"""
Validation harness for the anomaly-detection pipeline.

We have no labels for the real test set — only `train.npy` + `train_label.npy`
per window. To get an honest estimate of leaderboard F1 we do a two-level split:

  1. Window-level holdout: stratified by `metric_type`, hold out ~10% of the
     1000 windows. The cross-window model NEVER sees these windows during fit.
  2. Within each held-out window, time-split `train.npy` into the first 70%
     (used as "training" for per-window logic) and the last 30% (used as the
     "test" we score against).

This mirrors the real setup: predict the *future* of a series given its past,
on series the global model has never seen.

The leaderboard reports mean point-wise F1 across windows, so that is what
this harness reports.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

DATASET_ROOT = Path("student_dataset")
RESULTS_DIR = Path("results")


@dataclass
class Window:
    wid: str
    wdir: Path
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    info: dict

    @property
    def metric_type(self) -> str:
        return self.info.get("metric_type", "Unknown")


def load_window(wdir: Path) -> Window:
    return Window(
        wid=wdir.name.split("_", 1)[0],
        wdir=wdir,
        train_x=np.load(wdir / "train.npy"),
        train_y=np.load(wdir / "train_label.npy"),
        test_x=np.load(wdir / "test.npy"),
        info=json.loads((wdir / "info.json").read_text()),
    )


def all_window_dirs(root: Path = DATASET_ROOT) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "test.npy").is_file()])


def stratified_holdout(
    window_dirs: Sequence[Path], frac: float = 0.10, seed: int = 42
) -> tuple[list[Path], list[Path]]:
    """Hold out `frac` of windows stratified by metric_type. Returns (train_pool, holdout)."""
    rng = np.random.default_rng(seed)
    by_type: dict[str, list[Path]] = defaultdict(list)
    for w in window_dirs:
        info = json.loads((w / "info.json").read_text())
        by_type[info.get("metric_type", "Unknown")].append(w)

    holdout, train_pool = [], []
    for mt, group in by_type.items():
        idx = rng.permutation(len(group))
        n_hold = max(1, int(round(len(group) * frac)))
        for i, j in enumerate(idx):
            (holdout if i < n_hold else train_pool).append(group[j])
    return sorted(train_pool), sorted(holdout)


def time_split(
    train_x: np.ndarray, train_y: np.ndarray, frac: float = 0.70
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a series chronologically. Returns (sub_train_x, sub_train_y, sub_test_x, sub_test_y)."""
    n = len(train_x)
    cut = max(8, int(round(n * frac)))
    cut = min(cut, n - 4)
    return train_x[:cut], train_y[:cut], train_x[cut:], train_y[cut:]


def point_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        # Conventional choice: both arrays all-zero → F1 = 1.0 (perfect agreement)
        if fp == 0 and fn == 0:
            return 1.0
        return 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate(
    predictor: Callable[[np.ndarray, np.ndarray, np.ndarray, dict, float], np.ndarray],
    holdout: Sequence[Path],
    frac: float = 0.70,
) -> dict:
    """
    `predictor(sub_train_x, sub_train_y, sub_test_x, info, expected_ratio) -> binary array`

    `expected_ratio` is the anomaly ratio in the sub_test portion (we know it because
    we have the held-out labels). This mirrors how `info.json` test ratio is provided
    in the real task.
    """
    per_window = []
    per_category: dict[str, list[float]] = defaultdict(list)

    for wdir in holdout:
        w = load_window(wdir)
        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = time_split(w.train_x, w.train_y, frac=frac)
        ratio = float(sub_te_y.mean()) if len(sub_te_y) else 0.0

        try:
            pred = predictor(sub_tr_x, sub_tr_y, sub_te_x, w.info, ratio)
            pred = np.asarray(pred).astype(int).ravel()
            if len(pred) != len(sub_te_x):
                raise ValueError(f"len(pred)={len(pred)} but len(sub_test_x)={len(sub_te_x)}")
        except Exception as e:
            print(f"  ! {w.wid}: predictor failed ({e}); falling back to all-zeros")
            pred = np.zeros(len(sub_te_x), dtype=int)

        f1 = point_f1(sub_te_y, pred)
        per_window.append({"wid": w.wid, "metric_type": w.metric_type, "f1": f1,
                           "n_test": int(len(sub_te_x)), "ratio": ratio})
        per_category[w.metric_type].append(f1)

    overall = float(np.mean([r["f1"] for r in per_window]))
    by_type = {mt: {"n": len(v), "mean_f1": float(np.mean(v))} for mt, v in per_category.items()}
    return {"overall_f1": overall, "by_metric_type": by_type, "per_window": per_window}


def save_report(report: dict, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}_eval.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def print_summary(report: dict, name: str = "") -> None:
    header = f" [{name}] " if name else " "
    print(f"\n──────{header}validation summary ──────")
    print(f"  overall mean F1 : {report['overall_f1']:.4f}  (n={len(report['per_window'])})")
    print("  by metric_type:")
    for mt, stat in sorted(report["by_metric_type"].items()):
        print(f"    {mt:<28} n={stat['n']:>3}  F1={stat['mean_f1']:.4f}")
    print()


if __name__ == "__main__":
    # Sanity check: a zero-prediction baseline
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=42)
    print(f"Train pool: {len(train_pool)} windows, Holdout: {len(holdout)} windows")

    def zero_predictor(tr_x, tr_y, te_x, info, ratio):
        return np.zeros(len(te_x), dtype=int)

    report = evaluate(zero_predictor, holdout)
    print_summary(report, name="zero-baseline")
