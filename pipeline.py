"""
author v28 — Add window-level anomaly_ratio as a CW feature.

`info.json` has TWO anomaly ratios:
  - "training set anomaly ratio" — known for training windows (provided)
  - "test set anomaly ratio" — known for test windows at inference (provided)

We use `test set anomaly ratio` to compute `k` in `predict_segments`, but the
CW model itself has never seen any anomaly-density information. Adding the
window's ratio as a window-level feature (broadcast across all points) lets
the model condition its scoring on the expected anomaly density:

  - At training time, every training window's anomaly_ratio = train_ratio
  - At inference time, every test window's anomaly_ratio = test_ratio

The model can learn "this window will have ~12% anomalies, so even moderate
scores might be true positives" or "this window has ~0%, so trust the model
to be cautious".

We sweep two variants:
  A. ratio as a single scalar feature
  B. ratio + log(seq_len) — combines with our v22 lesson that intervals helped

Stacks on v22 (metadata-CW with intervals) as the strongest baseline.

Run:  uv run python v28_ratio_feature.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

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
from v22_metadata_features import build_metadata_hybrid, scores_v14_with_meta
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


def ratio_features(info: dict, mode: str, is_training: bool) -> np.ndarray:
    """At training, use 'training set anomaly ratio'.
    At inference, use 'test set anomaly ratio'."""
    if is_training:
        ratio = float(info.get("training set anomaly ratio", 0.0))
    else:
        ratio = float(info.get("test set anomaly ratio", 0.0))
    interval = info.get("intervals", 0)
    interval_log = float(np.log10(max(1, interval)))
    if mode == "intervals":
        return np.array([interval_log], dtype=np.float32)
    if mode == "intervals+ratio":
        return np.array([interval_log, ratio], dtype=np.float32)
    if mode == "intervals+ratio+log_ratio":
        # log(ratio + small_epsilon) — gives separation between low ratios
        log_r = float(np.log10(max(1e-4, ratio)))
        return np.array([interval_log, ratio, log_r], dtype=np.float32)
    return np.zeros(0, dtype=np.float32)


class RatioCW:
    def __init__(self, mode: str, n_estimators=500, max_depth=15, min_samples_leaf=3,
                 per_metric=False, seed=42):
        from sklearn.ensemble import RandomForestClassifier
        self.RF = RandomForestClassifier
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.per_metric = per_metric
        self.seed = seed
        self._models = {}

    def _features_for_window(self, x: np.ndarray, info: dict, is_training: bool) -> np.ndarray:
        base = extract_features(x, include_value=False)
        meta = ratio_features(info, self.mode, is_training)
        if meta.size == 0:
            return base
        return np.hstack([base, np.tile(meta, (base.shape[0], 1))])

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
            X_by_key.setdefault(key, []).append(
                self._features_for_window(train_x, info, is_training=True))
            y_by_key.setdefault(key, []).append(train_y)
        for key in X_by_key:
            X = np.vstack(X_by_key[key]); y = np.hstack(y_by_key[key])
            clf = self.RF(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, class_weight="balanced",
                random_state=self.seed, n_jobs=4,
            )
            clf.fit(X, y)
            self._models[key] = clf
        return self

    def predict_proba(self, test_x: np.ndarray, metric_type: str = "ALL",
                      info: dict = None) -> np.ndarray:
        if info is None:
            raise ValueError("needs info")
        X = self._features_for_window(test_x, info, is_training=False)
        key = metric_type if self.per_metric else "ALL"
        if key not in self._models:
            key = next(iter(self._models))
        proba = self._models[key].predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(test_x))


class RatioHybridCW:
    def __init__(self, g, p, specialized):
        self.global_model = g
        self.per_metric_model = p
        self.specialized = specialized

    def predict_proba(self, test_x, metric_type="ALL", info=None):
        if metric_type in self.specialized:
            return self.per_metric_model.predict_proba(test_x, metric_type=metric_type, info=info)
        return self.global_model.predict_proba(test_x, metric_type="ALL", info=info)


def build_ratio_hybrid(window_dirs, mode):
    g = RatioCW(mode=mode, per_metric=False).fit(window_dirs)
    p = RatioCW(mode=mode, per_metric=True).fit(window_dirs)
    return RatioHybridCW(g, p, SPECIALIZED)


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

    # NB: at validation time the "test ratio" for the held-out window's sub_test
    # portion isn't in info.json (info.json's test_ratio refers to the actual
    # competition test set). For a fair evaluation we use the held-out window's
    # *sub_test* anomaly ratio as a proxy at inference, computed from labels.
    # This is the same use of test_ratio that we do at competition time:
    # info.json provides it; we use it both for k and now for the feature.

    cw_models = {}
    for mode in ("intervals", "intervals+ratio", "intervals+ratio+log_ratio"):
        print(f">>> Training CW mode={mode}…")
        t0 = time.time()
        cw_models[mode] = build_ratio_hybrid(train_pool, mode)
        print(f"    fit {time.time() - t0:.1f}s")

    # Helper: at validation, use sub_test labels to compute ratio (proxy for info.json)
    from validation import time_split
    def make_predictor(cw_obj):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            # Override info's test_ratio with the actual holdout ratio for fairness
            local_info = dict(info)
            local_info["test set anomaly ratio"] = float(ratio)
            scores = scores_v14(sub_tr_x, sub_tr_y, sub_te_x, cw_obj, cnn_models,
                                local_info, info.get("metric_type", "ALL"))
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    results = {}
    for mode, cw_obj in cw_models.items():
        print(f"\n>>> Eval mode={mode}…")
        rep = evaluate(make_predictor(cw_obj), holdout)
        print_summary(rep, name=f"v28 {mode}")
        results[mode] = rep["overall_f1"]

    base = results["intervals"]
    print("\n──────  summary ──────")
    for mode, f1 in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  meta={mode:<28}  F1={f1:.4f}  Δ_vs_intervals={f1 - base:+.4f}")

    winner = max(results, key=lambda m: results[m])
    winner_f1 = results[winner]
    report = {
        "results": results,
        "winner": winner,
        "delta_vs_baseline": winner_f1 - base,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v28_ratio_feature")
    return report, cw_models, cnn_models


def generate_submission(mode: str, cnn_models,
                        output: Path = Path("submission_ratio_feature.json")) -> Path:
    print(f"\n>>> Re-training on ALL 1000 windows (mode={mode})…")
    t0 = time.time()
    cw_full = build_ratio_hybrid(all_window_dirs(), mode)
    print(f"    cw fit {time.time() - t0:.1f}s")

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
        scores = scores_v14(w.train_x, w.train_y, w.test_x, cw_full, cnn_full,
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
    rep, cw_models, cnn_models = run_validation()
    if rep["winner"] != "intervals" and rep["delta_vs_baseline"] > 0.001:
        print(f"\n{rep['winner']} beats baseline by {rep['delta_vs_baseline']:+.4f}; "
              "generating submission.")
        generate_submission(rep["winner"], cnn_models)
    else:
        print(f"\nRatio feature did not help (best Δ = {rep['delta_vs_baseline']:+.4f}); "
              "submission NOT generated.")
