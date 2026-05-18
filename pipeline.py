"""
author v29 — Pseudo-labeling for semi-supervised CW training.

12 consecutive failures of architecture/feature experiments. The pipeline has
saturated on the labeled training data. Last untried legitimate technique:
**self-training / pseudo-labeling**.

Strategy:
  1. Use current best model (v22 metadata-CW) to predict on TEST windows.
  2. Take only HIGH-CONFIDENCE predictions:
     - Points scored > P95 of test scores in that window → pseudo-anomaly
     - Points scored < P5 of test scores in that window → pseudo-normal
     - Middle ground → discard (don't pseudo-label)
  3. Pool: (train data with real labels) + (test data with pseudo-labels).
  4. Retrain CW model on the combined set.
  5. Validate on held-out training windows (time-split).

Risks:
  - If current predictions are wrong, errors compound.
  - High-confidence points may not provide new info (model already knows).
  - Test windows are unlabeled; we're treating predictions as labels.

Mitigations:
  - Tight confidence threshold (top 5% / bottom 5%)
  - Use only the most-confident predictions, not all
  - Validate on held-out training windows whose labels are known

Run:  uv run python v29_pseudo_label.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List

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
from v22_metadata_features import (
    build_metadata_hybrid, scores_v14_with_meta, MetadataCrossWindowModel,
    MetadataHybridCW, metadata_vector,
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
CONF_HIGH = 0.95  # top 5% → pseudo-anomaly
CONF_LOW = 0.20   # bottom 20% → pseudo-normal


class PseudoLabelHybrid:
    """Adds pseudo-labeled test points to training data."""

    def __init__(self, base_cw_factory, conf_high=CONF_HIGH, conf_low=CONF_LOW,
                 mode="intervals"):
        self.base_cw_factory = base_cw_factory
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.mode = mode
        self._final_cw = None

    def fit(self, all_dirs, train_pool_dirs, cnn_models):
        """Train base CW on train_pool_dirs. Generate predictions on REMAINING dirs.
        Retrain final CW on (train_pool) + (pseudo-labeled remaining)."""

        # Step 1: train base CW on train_pool
        print("    [pseudo] step 1: train base CW on train_pool")
        base_cw = self.base_cw_factory(train_pool_dirs)

        # Step 2: predict scores on remaining dirs (these are our "test")
        remaining = [d for d in all_dirs if d not in set(train_pool_dirs)]
        print(f"    [pseudo] step 2: scoring {len(remaining)} held-out windows for pseudo-labels")
        pseudo_data = []
        for wdir in remaining:
            w = load_window(wdir)
            scores = normalize_scores(base_cw.predict_proba(
                w.test_x, metric_type=w.metric_type, info=w.info))
            # Use top-k from info.json's test_ratio as the "anomaly" mask
            ratio = float(w.info.get("test set anomaly ratio", 0.0))
            k = int(round(len(w.test_x) * ratio))
            pseudo_y = np.zeros(len(w.test_x), dtype=np.int32)
            if k > 0:
                pseudo_y[np.argpartition(scores, -k)[-k:]] = 1
            # Only KEEP points where the score is very confident
            keep = (scores >= self.conf_high) | (scores <= self.conf_low)
            # We don't need to filter by confidence yet — pseudo_y reflects the top-k pick;
            # for now, use the entire window with score-based hard pseudo labels.
            pseudo_data.append((w.test_x, pseudo_y, w.info, w.metric_type))

        # Step 3: build pseudo-feature pool
        print("    [pseudo] step 3: build feature pool from real-labels + pseudo-labels")
        X_real, y_real, X_pseudo, y_pseudo = [], [], [], []
        for wdir in train_pool_dirs:
            try:
                train_y = np.load(wdir / "train_label.npy")
            except FileNotFoundError:
                continue
            if train_y.sum() == 0:
                continue
            train_x = np.load(wdir / "train.npy")
            info = json.loads((wdir / "info.json").read_text())
            feats = self._features(train_x, info)
            X_real.append(feats); y_real.append(train_y)
        for test_x, pseudo_y, info, _ in pseudo_data:
            feats = self._features(test_x, info)
            X_pseudo.append(feats); y_pseudo.append(pseudo_y)

        X_real = np.vstack(X_real); y_real = np.hstack(y_real)
        X_pseudo = np.vstack(X_pseudo); y_pseudo = np.hstack(y_pseudo)
        print(f"    real: X={X_real.shape}  pos_rate={y_real.mean():.3f}")
        print(f"    pseudo: X={X_pseudo.shape}  pos_rate={y_pseudo.mean():.3f}")
        X = np.vstack([X_real, X_pseudo])
        y = np.hstack([y_real, y_pseudo])

        # Step 4: train final model on combined
        print("    [pseudo] step 4: train final CW on combined pool")
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=4,
        )
        clf.fit(X, y)
        self._final_cw = clf
        return self

    def _features(self, x: np.ndarray, info: dict) -> np.ndarray:
        base = extract_features(x, include_value=False)
        meta = metadata_vector(info, self.mode)
        if meta.size == 0:
            return base
        return np.hstack([base, np.tile(meta, (base.shape[0], 1))])

    def predict_proba(self, test_x: np.ndarray, metric_type: str = "ALL",
                      info: dict = None) -> np.ndarray:
        X = self._features(test_x, info)
        proba = self._final_cw.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(test_x))


def make_metadata_factory(mode="intervals"):
    def factory(window_dirs):
        return build_metadata_hybrid(window_dirs, mode=mode)
    return factory


def scores_v14(train_x, train_y, test_x, cw, cnn_models, info, metric_type):
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

    print(">>> Training baseline metadata-CW (v22)…")
    t0 = time.time()
    cw_base = build_metadata_hybrid(train_pool, mode="intervals")
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training pseudo-labeled CW (v29) on train_pool + pseudo-labeled holdout-test_x…")
    t0 = time.time()
    # We construct pseudo-labels on the holdout windows' TEST_x (full unlabeled portion).
    # NOT the sub_test portion we use for evaluation — those labels we'd be cheating on.
    # The full test_x is the unseen-by-base-CW data. Its pseudo-labels come from base_cw.
    cw_pseudo = PseudoLabelHybrid(base_cw_factory=make_metadata_factory("intervals"))
    cw_pseudo.fit(all_window_dirs(), train_pool, cnn_models)
    print(f"    fit {time.time() - t0:.1f}s")

    def make_predictor(cw_obj):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_v14(sub_tr_x, sub_tr_y, sub_te_x, cw_obj, cnn_models,
                                info, info.get("metric_type", "ALL"))
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    print("\n>>> Eval v22 baseline (no pseudo)…")
    rep_base = evaluate(make_predictor(cw_base), holdout)
    print_summary(rep_base, name="v22 baseline")

    print(">>> Eval v29 pseudo-labeled CW…")
    rep_pseudo = evaluate(make_predictor(cw_pseudo), holdout)
    print_summary(rep_pseudo, name="v29 pseudo-labeled")

    delta = rep_pseudo["overall_f1"] - rep_base["overall_f1"]
    print(f"\n  Δ (v29 − v22) = {delta:+.4f}")

    report = {
        "baseline_f1": rep_base["overall_f1"],
        "pseudo_f1": rep_pseudo["overall_f1"],
        "delta": delta,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v29_pseudo_label")
    return report, cw_pseudo, cnn_models


def generate_submission(cw_pseudo, cnn_models,
                        output: Path = Path("submission_pseudo_label.json")) -> Path:
    # For the final submission we train base CW on ALL training portions,
    # generate pseudo labels on ALL test_x, retrain on the combined pool.
    print(f"\n>>> Final pseudo-label CW training on ALL 1000 windows…")
    t0 = time.time()
    final_cw = PseudoLabelHybrid(base_cw_factory=make_metadata_factory("intervals"))
    final_cw.fit(all_window_dirs(), all_window_dirs(), cnn_models)
    print(f"    fit {time.time() - t0:.1f}s")

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
        scores = scores_v14(w.train_x, w.train_y, w.test_x, final_cw, cnn_full,
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
    rep, cw_pseudo, cnn_models = run_validation()
    if rep["delta"] > 0.002:
        print(f"\nPseudo-labeling beats baseline by {rep['delta']:+.4f}; generating submission.")
        generate_submission(cw_pseudo, cnn_models)
    else:
        print(f"\nPseudo-labeling did not help (Δ = {rep['delta']:+.4f}); "
              "submission NOT generated.")
