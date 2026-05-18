"""
Shared utilities for time-series anomaly detection experiments.

This module provides common functions used across all experiment scripts:
- Window loading and inspection
- Train/validation splitting
- F1-score computation (point-wise)
- Score normalization
- Result persistence
"""

from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Any
import json
import numpy as np


def load_window(window_dir: Path) -> Dict[str, Any]:
    """Load all files for a single window directory."""
    return {
        "train_x": np.load(window_dir / "train.npy"),
        "train_y": np.load(window_dir / "train_label.npy"),
        "test_x": np.load(window_dir / "test.npy"),
        "test_ts": np.load(window_dir / "test_timestamp.npy"),
        "info": json.loads((window_dir / "info.json").read_text(encoding="utf-8")),
        "window_id": window_dir.name.split("_", 1)[0],
        "window_name": window_dir.name,
    }


def get_all_window_dirs(dataset_root: Path = None) -> List[Path]:
    """Return sorted list of all window directories."""
    if dataset_root is None:
        # Resolve relative to this file's parent (project root)
        dataset_root = Path(__file__).parent.parent / "student_dataset"
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def time_based_split(
    train_x: np.ndarray,
    train_y: np.ndarray,
    split_ratio: float = 0.7,
    min_train: int = 15,
    min_val: int = 5,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split training data chronologically into train and validation.
    
    Returns (tr_x, tr_y, val_x, val_y) or None if window is too short.
    """
    split_idx = int(split_ratio * len(train_x))
    if split_idx < min_train or len(train_x) - split_idx < min_val:
        return None
    return train_x[:split_idx], train_y[:split_idx], train_x[split_idx:], train_y[split_idx:]


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute point-wise binary F1 score."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        if fp == 0 and fn == 0:
            return 1.0  # Perfect prediction on window with no anomalies
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def f1_at_topk(
    scores: np.ndarray,
    labels: np.ndarray,
    rate: float,
) -> float:
    """
    Compute F1 when selecting top-k points by score, where k = round(n * rate).
    
    This mimics how our submission pipeline selects anomalies using the
    known test anomaly ratio from info.json.
    """
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).astype(int).ravel()
    n = len(labels)
    k = max(1, int(round(n * rate)))
    pred = np.zeros(n, dtype=int)
    if k > 0:
        pred[np.argpartition(scores, -k)[-k:]] = 1
    return compute_f1(labels, pred)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    scores = np.asarray(scores).ravel()
    mn, mx = np.min(scores), np.max(scores)
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x)
    x = x / temperature
    x_max = np.max(x)
    e = np.exp(x - x_max)
    return e / np.sum(e)


def save_results(
    experiment_name: str,
    results: Dict[str, Any],
    output_dir: Path = None,
) -> Path:
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    """Save experiment results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment_name}.json"
    
    # Convert numpy types to Python native types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    output_path.write_text(json.dumps(convert(results), indent=2), encoding="utf-8")
    return output_path


def summarize_results(per_window_f1s: List[float]) -> Dict[str, float]:
    """Compute summary statistics from a list of per-window F1 scores."""
    arr = np.array(per_window_f1s)
    return {
        "mean_f1": float(np.mean(arr)),
        "median_f1": float(np.median(arr)),
        "std_f1": float(np.std(arr)),
        "min_f1": float(np.min(arr)),
        "max_f1": float(np.max(arr)),
        "count": int(len(arr)),
    }


def find_contiguous_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous anomalous segments in a label array."""
    segments = []
    start = None
    for i, y in enumerate(labels):
        if y == 1 and start is None:
            start = i
        elif y == 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments
