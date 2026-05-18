"""
author v37 — clean-room replication of friend's 0.644 approach.

Friend's reported pipeline:
  - Single Random Forest
  - Per-window smoothing of anomaly probabilities ("anomalies are segments,
    not single points")
  - Top-k selection (no threshold) sized by info.json's test_ratio

Our existing 0.6238 LB submission has CW RF + per-window RF + CNN ensemble +
IF-on-test + segment-growth post-processing. Possibly too many moving parts.

Friend's 0.644 was achieved with a much simpler stack — and our recent
"improvements" (v22/v32/v34) all REGRESSED on LB despite winning validation,
suggesting our extra channels and segment-growth post-processing may be adding
noise that doesn't generalize.

This script: strip back to bare minimum and sweep smoothing width.

Variants:
  - smooth ∈ {1, 3, 5, 7, 9, 11, 13, 15, 21}
  - top-k via argpartition (no segment-growth, no merge, no shifts)
  - Cross-window RF on scale-invariant features (same as our existing CW
    `extract_features(include_value=False)`)

The variant with the highest holdout F1 produces `submission_simple_rf_smooth.json`.

Run:  uv run python v37_simple_rf_smooth.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from shared_lib import extract_features, predict_topk
from validation import (
    all_window_dirs, evaluate, load_window, point_f1, print_summary, save_report,
    stratified_holdout, time_split,
)


SMOOTH_WIDTHS = (1, 3, 5, 7, 9, 11, 13, 15, 21)


class SimpleCrossWindowRF:
    """One RF, scale-invariant features, returns per-point P(anomaly)."""

    def __init__(self, n_estimators: int = 500, max_depth: int = 15,
                 min_samples_leaf: int = 3, seed: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.clf: RandomForestClassifier = None

    def fit(self, window_dirs):
        X_all, y_all = [], []
        for wdir in window_dirs:
            try:
                train_y = np.load(wdir / "train_label.npy")
            except FileNotFoundError:
                continue
            if train_y.sum() == 0:
                continue
            train_x = np.load(wdir / "train.npy")
            X_all.append(extract_features(train_x, include_value=False))
            y_all.append(train_y)
        X = np.vstack(X_all); y = np.hstack(y_all)
        self.clf = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, class_weight="balanced",
            random_state=self.seed, n_jobs=4,
        )
        self.clf.fit(X, y)
        return self

    def predict_proba(self, test_x: np.ndarray) -> np.ndarray:
        X = extract_features(test_x, include_value=False)
        proba = self.clf.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(test_x))


def smooth_proba(p: np.ndarray, width: int) -> np.ndarray:
    """Per-window centered moving average. Reflective padding for edges."""
    if width <= 1:
        return p.copy()
    half = width // 2
    padded = np.concatenate([
        p[half - 1 :: -1] if half > 0 else p[:0],
        p,
        p[-1 : -half - 1 : -1] if half > 0 else p[:0],
    ])
    kernel = np.ones(width) / width
    out = np.convolve(padded, kernel, mode="valid")
    if len(out) > len(p):
        out = out[: len(p)]
    elif len(out) < len(p):
        # Fallback
        out = np.convolve(p, kernel, mode="same")
    return out


def predict_for_window(proba: np.ndarray, smooth_width: int, ratio: float) -> np.ndarray:
    smoothed = smooth_proba(proba, smooth_width)
    k = int(round(len(proba) * ratio))
    return predict_topk(smoothed, k)


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training single cross-window RF (scale-invariant features)…")
    t0 = time.time()
    rf = SimpleCrossWindowRF().fit(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    # Precompute holdout probabilities once
    print(">>> Pre-computing holdout probabilities…")
    cached = []
    for wdir in holdout:
        w = load_window(wdir)
        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = time_split(w.train_x, w.train_y, frac=0.70)
        ratio = float(sub_te_y.mean()) if len(sub_te_y) else 0.0
        proba = rf.predict_proba(sub_te_x)
        cached.append({
            "wid": w.wid, "metric_type": w.metric_type, "proba": proba,
            "y_true": sub_te_y, "ratio": ratio, "n": len(sub_te_x),
        })

    print(f">>> Sweeping smoothing widths {SMOOTH_WIDTHS}…")
    results = {}
    for smooth in SMOOTH_WIDTHS:
        f1s = []
        f1_by_metric: Dict[str, list] = {}
        for c in cached:
            pred = predict_for_window(c["proba"], smooth, c["ratio"])
            f1 = point_f1(c["y_true"], pred)
            f1s.append(f1)
            f1_by_metric.setdefault(c["metric_type"], []).append(f1)
        mean_f1 = float(np.mean(f1s))
        by_metric = {mt: float(np.mean(v)) for mt, v in f1_by_metric.items()}
        results[smooth] = {"overall_f1": mean_f1, "by_metric": by_metric}
        print(f"  smooth={smooth:>3}  F1={mean_f1:.4f}  "
              + "  ".join(f"{mt[:4]}={f1:.3f}" for mt, f1 in sorted(by_metric.items())))

    print("\n──────  summary ──────")
    rows = sorted(results.items(), key=lambda kv: -kv[1]["overall_f1"])
    for smooth, r in rows:
        print(f"  smooth={smooth:>3}  F1={r['overall_f1']:.4f}")
    winner_smooth = rows[0][0]
    winner_f1 = rows[0][1]["overall_f1"]
    print(f"\n  Winner: smooth={winner_smooth}  F1={winner_f1:.4f}")

    report = {
        "results": {str(s): r["overall_f1"] for s, r in results.items()},
        "winner_smooth": winner_smooth,
        "winner_f1": winner_f1,
        "per_metric": {str(s): r["by_metric"] for s, r in results.items()},
        "seed": seed, "n_holdout": len(holdout),
    }
    save_report(report, "v37_simple_rf_smooth")
    return report, rf


def generate_submission(smooth_width: int, rf,
                        output: Path = Path("submission_simple_rf_smooth.json")) -> Path:
    print(f"\n>>> Re-training RF on ALL 1000 windows…")
    t0 = time.time()
    rf_full = SimpleCrossWindowRF().fit(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    print(f">>> Generating predictions with smooth={smooth_width}…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        ratio = float(w.info.get("test set anomaly ratio", 0.0))
        proba = rf_full.predict_proba(w.test_x)
        pred = predict_for_window(proba, smooth_width, ratio)
        preds[w.wid] = pred.astype(int).tolist()
        if i % 100 == 0:
            print(f"    {i}/1000 ({time.time() - t0:.0f}s)")

    assert len(preds) == 1000
    output.write_text(
        json.dumps({"predictions": preds}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f">>> Wrote {output}")
    return output


if __name__ == "__main__":
    rep, _ = run_validation()
    # Generate for the best smoothing width; also save a couple of alternates so
    # the user can pick if validation isn't trustworthy
    rf_full_done = None
    target_widths = [rep["winner_smooth"]]
    # Also save smooth=5 and smooth=11 as alternates
    for w in (5, 11):
        if w != rep["winner_smooth"]:
            target_widths.append(w)

    print(f"\n>>> Re-training RF on ALL 1000 windows (once)…")
    t0 = time.time()
    rf_full = SimpleCrossWindowRF().fit(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    for w in target_widths:
        print(f">>> Generating submission for smooth={w}…")
        preds = {}
        for wdir in all_window_dirs():
            wnd = load_window(wdir)
            ratio = float(wnd.info.get("test set anomaly ratio", 0.0))
            proba = rf_full.predict_proba(wnd.test_x)
            pred = predict_for_window(proba, w, ratio)
            preds[wnd.wid] = pred.astype(int).tolist()
        assert len(preds) == 1000
        out = Path(f"submission_simple_rf_smooth_{w}.json")
        out.write_text(json.dumps({"predictions": preds}, ensure_ascii=False, separators=(",", ":")),
                       encoding="utf-8")
        print(f"  wrote {out}")
