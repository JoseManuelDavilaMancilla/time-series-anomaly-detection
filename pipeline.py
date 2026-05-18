"""
author v23 — per-intervals hybrid CW routing.

v22 showed intervals matters as a feature (+0.0011). The natural extension:
train ONE CW model per intervals bucket (7 total, ~140 windows each) and
route at inference based on the window's intervals. Same logic as the
metric_type hybrid that won earlier, but on the new metadata axis.

Sweep:
  A. baseline: existing hybrid CW (metric_type routing only)
  B. interval routing: 7 CW models, one per intervals value
  C. interval routing + intervals as feature (combines v22 with v23)
  D. metric_type + interval doubly-routed: 6 × 7 = 42 sub-models (probably
     too sparse, ~24 windows per cell)

We expect B or C to win. D is included just to confirm it overfits.

Run:  uv run python v23_per_interval_cw.py
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

from shared_lib import (
    CrossWindowModel,
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
from v22_metadata_features import (
    INTERVALS, MetadataCrossWindowModel, MetadataHybridCW, build_metadata_hybrid,
    scores_v14_with_meta,
)
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


class PerIntervalCW:
    """7 RF models, one per intervals bucket. Fallback to a global model
    if a bucket has too few windows."""

    def __init__(self, n_estimators: int = 500, max_depth: int = 15,
                 min_samples_leaf: int = 3, include_intervals_feat: bool = False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.include_intervals_feat = include_intervals_feat
        self._models: Dict[int, RandomForestClassifier] = {}
        self._global_fallback: RandomForestClassifier = None

    def _features_for_window(self, train_x: np.ndarray, intervals: int) -> np.ndarray:
        base = extract_features(train_x, include_value=False)
        if not self.include_intervals_feat:
            return base
        interval_log = float(np.log10(max(1, intervals)))
        broadcast = np.full((base.shape[0], 1), interval_log, dtype=np.float32)
        return np.hstack([base, broadcast])

    def fit(self, window_dirs):
        X_by_interval: Dict[int, list] = {iv: [] for iv in INTERVALS}
        y_by_interval: Dict[int, list] = {iv: [] for iv in INTERVALS}
        X_all, y_all = [], []
        for wdir in window_dirs:
            try:
                train_y = np.load(wdir / "train_label.npy")
            except FileNotFoundError:
                continue
            if train_y.sum() == 0:
                continue
            train_x = np.load(wdir / "train.npy")
            info = json.loads((wdir / "info.json").read_text())
            interval = info.get("intervals", 0)
            if interval not in INTERVALS:
                continue
            feats = self._features_for_window(train_x, interval)
            X_by_interval[interval].append(feats)
            y_by_interval[interval].append(train_y)
            X_all.append(feats)
            y_all.append(train_y)

        # Train one RF per interval
        for iv in INTERVALS:
            if not X_by_interval[iv]:
                continue
            X = np.vstack(X_by_interval[iv])
            y = np.hstack(y_by_interval[iv])
            if y.sum() < 5 or len(X) < 100:
                continue
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, class_weight="balanced",
                random_state=42, n_jobs=4,
            )
            clf.fit(X, y)
            self._models[iv] = clf

        # Global fallback
        X_g = np.vstack(X_all); y_g = np.hstack(y_all)
        self._global_fallback = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, class_weight="balanced",
            random_state=42, n_jobs=4,
        )
        self._global_fallback.fit(X_g, y_g)
        return self

    def predict_proba(self, test_x: np.ndarray, info: dict = None,
                      metric_type: str = "ALL") -> np.ndarray:
        # metric_type is ignored — routing is purely by intervals
        if info is None:
            raise ValueError("PerIntervalCW needs info dict")
        interval = info.get("intervals", 0)
        feats = self._features_for_window(test_x, interval)
        clf = self._models.get(interval, self._global_fallback)
        proba = clf.predict_proba(feats)
        return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(test_x))


class CombinedCW:
    """Routes per-intervals via PerIntervalCW for >=N windows; falls back to
    metadata-hybrid (metric_type routing + intervals feature) otherwise."""

    def __init__(self, per_iv: PerIntervalCW, meta_hybrid: MetadataHybridCW,
                 use_per_interval: bool = True):
        self.per_iv = per_iv
        self.meta_hybrid = meta_hybrid
        self.use_per_interval = use_per_interval

    def predict_proba(self, test_x: np.ndarray, metric_type: str = "ALL",
                      info: dict = None) -> np.ndarray:
        if self.use_per_interval:
            return self.per_iv.predict_proba(test_x, info=info)
        return self.meta_hybrid.predict_proba(test_x, metric_type=metric_type, info=info)


def scores_v14_route_iv(train_x, train_y, test_x, cw, cnn_models, info,
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

    print(">>> Training v22 baseline (metric-type hybrid + intervals feature)…")
    t0 = time.time()
    cw_v22 = build_metadata_hybrid(train_pool, mode="intervals")
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training PerIntervalCW (no intervals feature, 7 sub-models)…")
    t0 = time.time()
    cw_per_iv = PerIntervalCW(include_intervals_feat=False).fit(train_pool)
    print(f"    fit {time.time() - t0:.1f}s  models for intervals: {sorted(cw_per_iv._models)}")

    print(">>> Training PerIntervalCW (WITH intervals feature, 7 sub-models)…")
    t0 = time.time()
    cw_per_iv_feat = PerIntervalCW(include_intervals_feat=True).fit(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    # Build CombinedCW variants
    cw_v23_a = cw_per_iv
    cw_v23_b = cw_per_iv_feat

    def predictor(cw_obj, route_fn):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = route_fn(sub_tr_x, sub_tr_y, sub_te_x, cw_obj, cnn_models, info,
                              info.get("metric_type", "ALL"))
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    print("\n>>> Eval [v22 baseline: metric-type hybrid + intervals feature]…")
    rep_v22 = evaluate(predictor(cw_v22, scores_v14_with_meta), holdout)
    print_summary(rep_v22, name="v22 baseline")

    print(">>> Eval [v23a: per-interval routing, no intervals feature]…")
    rep_v23a = evaluate(predictor(cw_v23_a, scores_v14_route_iv), holdout)
    print_summary(rep_v23a, name="v23a per-interval")

    print(">>> Eval [v23b: per-interval routing + intervals feature]…")
    rep_v23b = evaluate(predictor(cw_v23_b, scores_v14_route_iv), holdout)
    print_summary(rep_v23b, name="v23b per-interval+feat")

    base = rep_v22["overall_f1"]
    print("\n──────  summary ──────")
    print(f"  v22 baseline (metric_type hybrid + intervals feat)   F1={base:.4f}")
    print(f"  v23a per-interval routing (no feat)                  F1={rep_v23a['overall_f1']:.4f}  Δ={rep_v23a['overall_f1'] - base:+.4f}")
    print(f"  v23b per-interval routing + intervals feat           F1={rep_v23b['overall_f1']:.4f}  Δ={rep_v23b['overall_f1'] - base:+.4f}")

    report = {
        "v22_baseline_f1": base,
        "v23a_f1": rep_v23a["overall_f1"],
        "v23b_f1": rep_v23b["overall_f1"],
        "delta_a": rep_v23a["overall_f1"] - base,
        "delta_b": rep_v23b["overall_f1"] - base,
        "winner": "v23a" if rep_v23a["overall_f1"] > rep_v23b["overall_f1"] else "v23b",
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v23_per_interval_cw")
    return report, cw_v23_a, cw_v23_b, cnn_models


def generate_submission(cw, cnn_models, route_fn,
                        output: Path = Path("submission_per_interval_cw.json")) -> Path:
    print(f"\n>>> Re-training all models on ALL 1000 windows…")
    # Re-train the chosen CW on full data
    if isinstance(cw, PerIntervalCW):
        include = cw.include_intervals_feat
        t0 = time.time()
        cw_full = PerIntervalCW(include_intervals_feat=include).fit(all_window_dirs())
        print(f"    cw fit {time.time() - t0:.1f}s")
    else:
        raise NotImplementedError

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
        scores = route_fn(w.train_x, w.train_y, w.test_x, cw_full, cnn_full, w.info,
                          w.metric_type)
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
    rep, cw_a, cw_b, cnn_models = run_validation()
    best_delta = max(rep["delta_a"], rep["delta_b"])
    if best_delta > 0.001:
        winner_cw = cw_a if rep["delta_a"] >= rep["delta_b"] else cw_b
        print(f"\nWinner: {rep['winner']} beats v22 by {best_delta:+.4f}; generating submission.")
        generate_submission(winner_cw, cnn_models, scores_v14_route_iv)
    else:
        print(f"\nNeither per-interval variant beat v22 baseline meaningfully "
              f"(best Δ = {best_delta:+.4f}); submission NOT generated.")
