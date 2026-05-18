"""
author v22 — Add metadata features (intervals, log-seq-len) to the CW model.

So far the CW model has used only point-level features (lags, rolling stats,
EMA). It has NO access to window-level metadata like the sampling rate. But:
  - intervals ∈ {60, 300, 345, 600, 864, 1200, 3600} seconds — 7 buckets,
    each ~140–150 windows. A 10-point segment is 10 min at intervals=60 but
    10 hours at intervals=3600. The model has been treating these identically.
  - log(seq_len) varies — longer training/test sequences mean more context.

Hypothesis: conditioning the CW model on these scalars lets it learn rate-
specific patterns. Concretely, each per-point feature row gets the window's
intervals and log(seq_len) appended; both become "always-the-same" columns
for the points within a window, so trees can split on them and create
intervals-specific subtrees.

Sweep three configurations:
  A. baseline: existing CW (no metadata)
  B. + intervals only
  C. + intervals + log(train_seq_len) + log(test_seq_len)

Run:  uv run python v22_metadata_features.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from shared_lib import (
    HybridCrossWindowModel,
    categorize_window,
    extract_features,
    global_distance_score,
    isolation_forest_test,
    normalize_scores,
    online_ensemble,
    per_window_rf_score,
    predict_segments,
)
from v6_cnn import SPECIALIZED, build_training_pool as v6_pool
from v7_cnn_ensemble import CNN_SEEDS, ensemble_cnn_score, fit_cnn_with_seed
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)
CNN_WEIGHT = 0.35

# Known intervals values (verified by metadata scan)
INTERVALS = (60, 300, 345, 600, 864, 1200, 3600)
INTERVAL_TO_IDX = {v: i for i, v in enumerate(INTERVALS)}


def metadata_vector(info: dict, mode: str) -> np.ndarray:
    """Returns the metadata features to broadcast across all points in a window.

    mode ∈ {'none', 'intervals', 'intervals+seqlen'}"""
    if mode == "none":
        return np.zeros(0, dtype=np.float32)
    interval = info.get("intervals", 0)
    interval_log = np.log10(max(1, interval))
    if mode == "intervals":
        return np.array([interval_log], dtype=np.float32)
    # intervals + seqlen
    tr_log = np.log10(max(1, info.get("train_seq_len", 1)))
    te_log = np.log10(max(1, info.get("test_seq_len", 1)))
    return np.array([interval_log, tr_log, te_log], dtype=np.float32)


# ─────────────────────────────────────────────
# Metadata-aware cross-window model
# ─────────────────────────────────────────────


class MetadataCrossWindowModel:
    """RF cross-window model that appends per-window metadata to every point feature."""

    def __init__(self, mode: str, n_estimators: int = 500, max_depth: int = 15,
                 min_samples_leaf: int = 3, seed: int = 42, n_jobs: int = 4,
                 per_metric: bool = False):
        from sklearn.ensemble import RandomForestClassifier
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.n_jobs = n_jobs
        self.per_metric = per_metric
        self._RF = RandomForestClassifier
        self._models: dict = {}

    def _features_for_window(self, train_x: np.ndarray, info: dict) -> np.ndarray:
        base = extract_features(train_x, include_value=False)
        meta = metadata_vector(info, self.mode)
        if meta.size == 0:
            return base
        # Broadcast metadata to (n_points, n_meta) and concatenate horizontally
        broadcast = np.tile(meta, (base.shape[0], 1))
        return np.hstack([base, broadcast])

    def fit(self, window_dirs):
        X_by_key, y_by_key = {}, {}
        for wdir in window_dirs:
            try:
                train_y = np.load(wdir / "train_label.npy")
            except FileNotFoundError:
                continue
            if train_y.sum() == 0:
                continue
            train_x = np.load(wdir / "train.npy")
            info = json.loads((wdir / "info.json").read_text())
            key = info.get("metric_type", "Unknown") if self.per_metric else "ALL"
            X_by_key.setdefault(key, []).append(self._features_for_window(train_x, info))
            y_by_key.setdefault(key, []).append(train_y)

        for key in X_by_key:
            X = np.vstack(X_by_key[key])
            y = np.hstack(y_by_key[key])
            clf = self._RF(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, class_weight="balanced",
                random_state=self.seed, n_jobs=self.n_jobs,
            )
            clf.fit(X, y)
            self._models[key] = clf
        return self

    def predict_proba(self, test_x: np.ndarray, metric_type: str = "ALL",
                      info: dict = None) -> np.ndarray:
        if info is None:
            raise ValueError("MetadataCrossWindowModel needs info dict")
        X = self._features_for_window(test_x, info)
        key = metric_type if self.per_metric else "ALL"
        if key not in self._models:
            key = next(iter(self._models))
        proba = self._models[key].predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(test_x))


class MetadataHybridCW:
    """Hybrid wrapper that routes per-metric metadata models for SPECIALIZED types."""

    def __init__(self, global_model: MetadataCrossWindowModel,
                 per_metric_model: MetadataCrossWindowModel,
                 specialized: frozenset):
        self.global_model = global_model
        self.per_metric_model = per_metric_model
        self.specialized = specialized

    def predict_proba(self, test_x: np.ndarray, metric_type: str = "ALL",
                      info: dict = None) -> np.ndarray:
        if metric_type in self.specialized:
            return self.per_metric_model.predict_proba(test_x, metric_type=metric_type,
                                                       info=info)
        return self.global_model.predict_proba(test_x, metric_type="ALL", info=info)


def build_metadata_hybrid(window_dirs, mode: str):
    g = MetadataCrossWindowModel(mode=mode, per_metric=False,
                                 n_estimators=500, max_depth=15).fit(window_dirs)
    p = MetadataCrossWindowModel(mode=mode, per_metric=True,
                                 n_estimators=500, max_depth=15).fit(window_dirs)
    return MetadataHybridCW(g, p, SPECIALIZED)


def scores_v14_with_meta(train_x, train_y, test_x, cw, cnn_models, info,
                         metric_type) -> np.ndarray:
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type, info=info))
    cnn_s = normalize_scores(ensemble_cnn_score(cnn_models, test_x))

    if category == "constant_train":
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * if_s + CNN_WEIGHT * cnn_s
    if category == "disjoint":
        g_s = normalize_scores(global_distance_score(train_x, test_x))
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return 0.35 * cw_s + 0.30 * g_s + (0.35 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
    if category == "partial_overlap":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        return 0.35 * cw_s + 0.35 * pw + (0.30 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn_s
    if category == "test_within_train":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * pw + CNN_WEIGHT * cnn_s
    return np.zeros(len(test_x))


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training 3-seed CNN ensemble…")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Training metadata-CW models (3 modes)…")
    cw_by_mode = {}
    for mode in ("none", "intervals", "intervals+seqlen"):
        t0 = time.time()
        cw_by_mode[mode] = build_metadata_hybrid(train_pool, mode)
        print(f"    mode={mode}  fit {time.time() - t0:.1f}s")

    def predictor(mode: str):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_v14_with_meta(sub_tr_x, sub_tr_y, sub_te_x, cw_by_mode[mode],
                                          cnn_models, info, info.get("metric_type", "ALL"))
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    results = {}
    for mode in ("none", "intervals", "intervals+seqlen"):
        print(f"\n>>> Eval mode={mode}…")
        rep = evaluate(predictor(mode), holdout)
        print_summary(rep, name=f"meta={mode}")
        results[mode] = rep["overall_f1"]

    print("\n──────  summary ──────")
    base = results["none"]
    for mode, f1 in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  meta={mode:<20}  F1={f1:.4f}  Δ_vs_none={f1 - base:+.4f}")

    winner_mode = max(results, key=lambda m: results[m])
    winner_f1 = results[winner_mode]
    report = {
        "results": results,
        "winner": winner_mode,
        "delta_vs_baseline": winner_f1 - base,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v22_metadata_features")
    return report, cw_by_mode, cnn_models


def generate_submission(mode: str, cnn_models,
                        output: Path = Path("submission_metadata_cw.json")) -> Path:
    print(f"\n>>> Re-training on ALL 1000 windows (mode={mode})…")
    t0 = time.time()
    cw_full = build_metadata_hybrid(all_window_dirs(), mode)
    print(f"    cw fit {time.time() - t0:.1f}s")

    # Retrain CNNs on full data
    Xc, yc = v6_pool(all_window_dirs())
    cnn_full = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_full.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_v14_with_meta(w.train_x, w.train_y, w.test_x, cw_full, cnn_full,
                                      w.info, w.metric_type)
        k = int(round(len(w.test_x) * ratio))
        preds[w.wid] = predict_segments(scores, k, **SEG_KWARGS).astype(int).tolist()
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
    rep, cw_by_mode, cnn_models = run_validation()
    if rep["winner"] != "none" and rep["delta_vs_baseline"] > 0.001:
        print(f"\n{rep['winner']} beats baseline by {rep['delta_vs_baseline']:+.4f}; "
              "generating submission.")
        generate_submission(rep["winner"], cnn_models)
    else:
        print(f"\nMetadata features did not meaningfully help "
              f"(best Δ = {rep['delta_vs_baseline']:+.4f}); submission NOT generated.")
