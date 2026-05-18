"""
author v27 — Spectral / wavelet features as a new CW channel.

Hypothesis: existing features (lags, rolling stats, EMA) are all *temporal-
domain* statistics. Anomalies often manifest as frequency-domain changes:
loss of periodicity, new high-frequency content, energy redistribution.
We've never given the CW model access to spectral features.

For each point, extract:
  - FFT magnitudes at low/mid/high frequency bins (4 features)
  - Spectral entropy of the local 32-pt window
  - Wavelet detail coefficients at 3 scales (3 features)
  - Local zero-crossing rate (1 feature)

These become per-point features alongside the existing time-domain ones,
boosting the CW feature dimension from 27 → 36. The model can then learn
"high-frequency content drops in this anomaly" or "this is the only point
with non-zero wavelet detail at scale 2".

Run:  uv run python v27_spectral_features.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import torch

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
    MetadataCrossWindowModel, MetadataHybridCW, INTERVALS, metadata_vector,
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
WIN = 32  # spectral window size


def spectral_features(series: np.ndarray, win: int = WIN) -> np.ndarray:
    """Per-point spectral features computed over a centered window of size `win`."""
    n = len(series)
    half = win // 2
    s = series.astype(np.float64)
    padded = np.concatenate([np.full(half, s[0]), s, np.full(half, s[-1])])

    out = np.zeros((n, 8), dtype=np.float32)
    for i in range(n):
        w = padded[i : i + win]
        w = w - w.mean()
        # FFT magnitudes
        fft_mag = np.abs(np.fft.rfft(w))  # (win//2 + 1,)
        # Normalize so they're scale-invariant
        total = fft_mag.sum() + 1e-9
        fft_norm = fft_mag / total
        # Low, mid, high band energy fractions
        n_bins = len(fft_mag)
        b1 = n_bins // 3
        b2 = 2 * n_bins // 3
        out[i, 0] = float(fft_norm[:b1].sum())
        out[i, 1] = float(fft_norm[b1:b2].sum())
        out[i, 2] = float(fft_norm[b2:].sum())
        # Dominant frequency index (normalized)
        out[i, 3] = float(np.argmax(fft_mag) / max(1, n_bins - 1))
        # Spectral entropy
        p = fft_norm[fft_norm > 0]
        out[i, 4] = float(-(p * np.log(p)).sum()) if len(p) else 0.0
        # Zero-crossing rate
        out[i, 5] = float(((w[:-1] * w[1:]) < 0).sum() / max(1, len(w) - 1))
        # Total energy (relative to window size)
        out[i, 6] = float((w * w).sum() / win)
        # Variance ratio: first half vs second half
        h1, h2 = w[:half], w[half:]
        out[i, 7] = float((h1.var() + 1e-9) / (h2.var() + 1e-9))
    return out


class SpectralMetadataCW:
    """CW model with metadata features (intervals) + spectral features."""

    def __init__(self, n_estimators=500, max_depth=15, min_samples_leaf=3,
                 per_metric=False, seed=42):
        from sklearn.ensemble import RandomForestClassifier
        self.RF = RandomForestClassifier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.per_metric = per_metric
        self.seed = seed
        self._models = {}

    def _features_for_window(self, train_x, info):
        base = extract_features(train_x, include_value=False)
        meta = metadata_vector(info, mode="intervals")
        spec = spectral_features(train_x)
        broadcast_meta = np.tile(meta, (base.shape[0], 1))
        return np.hstack([base, broadcast_meta, spec])

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
            clf = self.RF(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, class_weight="balanced",
                random_state=self.seed, n_jobs=4,
            )
            clf.fit(X, y)
            self._models[key] = clf
        return self

    def predict_proba(self, test_x, metric_type="ALL", info=None):
        if info is None:
            raise ValueError("needs info")
        X = self._features_for_window(test_x, info)
        key = metric_type if self.per_metric else "ALL"
        if key not in self._models:
            key = next(iter(self._models))
        proba = self._models[key].predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(test_x))


class SpectralHybridCW:
    def __init__(self, g, p, specialized):
        self.global_model = g
        self.per_metric_model = p
        self.specialized = specialized

    def predict_proba(self, test_x, metric_type="ALL", info=None):
        if metric_type in self.specialized:
            return self.per_metric_model.predict_proba(test_x, metric_type=metric_type,
                                                       info=info)
        return self.global_model.predict_proba(test_x, metric_type="ALL", info=info)


def build_spectral_hybrid(window_dirs):
    g = SpectralMetadataCW(per_metric=False).fit(window_dirs)
    p = SpectralMetadataCW(per_metric=True).fit(window_dirs)
    return SpectralHybridCW(g, p, SPECIALIZED)


def scores_with_meta(train_x, train_y, test_x, cw, cnn_models, info, metric_type):
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

    print(">>> Training metadata-CW baseline (v22)…")
    t0 = time.time()
    from v22_metadata_features import build_metadata_hybrid
    cw_v22 = build_metadata_hybrid(train_pool, mode="intervals")
    print(f"    cw v22 fit {time.time() - t0:.1f}s")

    print(">>> Training spectral+metadata CW (v27)…")
    t0 = time.time()
    cw_v27 = build_spectral_hybrid(train_pool)
    print(f"    cw v27 fit {time.time() - t0:.1f}s")

    print(">>> Training 3-seed CNN ensemble…")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    def pred(cw_obj):
        def fn(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_with_meta(sub_tr_x, sub_tr_y, sub_te_x, cw_obj, cnn_models,
                                      info, info.get("metric_type", "ALL"))
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return fn

    print("\n>>> Eval v22 baseline…")
    rep_v22 = evaluate(pred(cw_v22), holdout)
    print_summary(rep_v22, name="v22 baseline")

    print(">>> Eval v27 (+ spectral features)…")
    rep_v27 = evaluate(pred(cw_v27), holdout)
    print_summary(rep_v27, name="v27 spectral+meta")

    delta = rep_v27["overall_f1"] - rep_v22["overall_f1"]
    print(f"\n  Δ (v27 − v22) = {delta:+.4f}")

    report = {
        "v22_f1": rep_v22["overall_f1"],
        "v27_f1": rep_v27["overall_f1"],
        "delta": delta,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v27_spectral_features")
    return report, cw_v27, cnn_models


def generate_submission(cw, cnn_models, output: Path = Path("submission_spectral.json")) -> Path:
    print(f"\n>>> Re-training on ALL 1000 windows…")
    t0 = time.time()
    cw_full = build_spectral_hybrid(all_window_dirs())
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
        scores = scores_with_meta(w.train_x, w.train_y, w.test_x, cw_full, cnn_full,
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
    rep, cw, cnn_models = run_validation()
    if rep["delta"] > 0.001:
        print(f"\nSpectral features beat v22 by {rep['delta']:+.4f}; generating submission.")
        generate_submission(cw, cnn_models)
    else:
        print(f"\nSpectral features did not help (Δ = {rep['delta']:+.4f}); "
              "submission NOT generated.")
