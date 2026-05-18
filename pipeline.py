"""
author v12 — Per-metric CNN routing.

The hybrid per-metric CW model is +0.0089 over global CW on validation. The CNN
ensemble is also +0.011 over single-seed CNN. The natural extension: train
6 CNN ensembles, one per metric_type, and route at inference time.

Hypothesis: per-metric specialization should help the CNN for the same 3 metric
types where it helped the CW model (ErrorCount, ResourceUtilizationRate,
SuccessRate). On those types, train a dedicated 3-seed CNN ensemble on that
type's data only. For the other 3 (Count, LatencySecond, QPS), use the global
CNN ensemble.

Caveat: per-metric training pool is ~1/6 the size (~15k samples instead of
90k). Subsampling negatives to 30% leaves ~5k per type, which is borderline
for a CNN. Will check loss curves and skip per-metric routing for any type
with < 3k samples.

Stacks on top of `submission_kimi_combo_all_v12.json` (v11 winning config):
hybrid CW + segments + 3-seed CNN + all_v12 ensemble weights.

Run:  uv run python v12_per_metric_cnn.py
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
    CrossWindowModel,
    HybridCrossWindowModel,
    categorize_window,
    global_distance_score,
    isolation_forest_test,
    normalize_scores,
    online_ensemble,
    per_window_rf_score,
    predict_segments,
)
from v6_cnn import (
    SPECIALIZED,
    SEED,
    SUBSAMPLE_NEG,
    build_contexts as v6_build_contexts,
    build_training_pool as v6_build_training_pool,
)
from v7_cnn_ensemble import (
    CNN_SEEDS,
    ensemble_cnn_score,
    fit_cnn_with_seed,
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
MIN_PER_METRIC_SAMPLES = 3000  # below this, fall back to global CNN
USE_V12_WEIGHTS = True
IF_DISJOINT = True
IF_CONSTANT = True


def build_per_metric_training_pool(window_dirs, metric_type: str
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """Same as v6 pool but only windows of a single metric_type."""
    Xs, ys = [], []
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        if info.get("metric_type") != metric_type:
            continue
        train_y = np.load(wdir / "train_label.npy")
        if train_y.sum() == 0:
            continue
        train_x = np.load(wdir / "train.npy")
        if len(train_x) < 20:
            continue
        Xs.append(v6_build_contexts(train_x))
        ys.append(train_y.astype(np.float32))
    if not Xs:
        return np.zeros((0, 32), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    if 0 < SUBSAMPLE_NEG < 1.0:
        rng = np.random.default_rng(SEED)
        neg_idx = np.where(y == 0)[0]
        keep_n = int(len(neg_idx) * SUBSAMPLE_NEG)
        keep_idx = rng.choice(neg_idx, size=keep_n, replace=False)
        all_idx = np.concatenate([np.where(y == 1)[0], keep_idx])
        rng.shuffle(all_idx)
        X = X[all_idx]
        y = y[all_idx]
    return X, y


def train_cnn_routes(window_dirs):
    """Train global CNN ensemble + per-metric CNN ensembles for specialized types."""
    print(">>> Training GLOBAL 3-seed CNN ensemble…")
    X, y = v6_build_training_pool(window_dirs)
    print(f"    global pool: X={X.shape}  y.mean={y.mean():.3f}")
    global_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        global_models.append(fit_cnn_with_seed(X, y, s))
        print(f"    global seed={s}  fit {time.time() - t0:.1f}s")

    per_metric_models: dict[str, list] = {}
    for mt in SPECIALIZED:
        print(f">>> Training per-metric CNN ensemble for {mt}…")
        Xm, ym = build_per_metric_training_pool(window_dirs, mt)
        print(f"    {mt} pool: X={Xm.shape}  y.mean={ym.mean():.3f}")
        if len(Xm) < MIN_PER_METRIC_SAMPLES:
            print(f"    SKIP {mt}: only {len(Xm)} samples (< {MIN_PER_METRIC_SAMPLES})")
            continue
        models = []
        for s in CNN_SEEDS:
            t0 = time.time()
            models.append(fit_cnn_with_seed(Xm, ym, s))
            print(f"    {mt} seed={s}  fit {time.time() - t0:.1f}s")
        per_metric_models[mt] = models

    print(f">>> Per-metric CNN routes built for: {sorted(per_metric_models.keys())}")
    return global_models, per_metric_models


def routed_cnn_score(test_x: np.ndarray, metric_type: str,
                     global_models, per_metric_models) -> np.ndarray:
    """Use the per-metric ensemble if available, otherwise global."""
    if metric_type in per_metric_models:
        return ensemble_cnn_score(per_metric_models[metric_type], test_x)
    return ensemble_cnn_score(global_models, test_x)


def scores_for_routed(train_x, train_y, test_x, cw, global_models,
                      per_metric_models, metric_type) -> np.ndarray:
    """v11 all_v12 weights with the per-metric-routed CNN ensemble."""
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
    cnn_s = normalize_scores(routed_cnn_score(test_x, metric_type,
                                              global_models, per_metric_models))

    if category == "constant_train":
        if IF_CONSTANT:
            if_s = normalize_scores(isolation_forest_test(test_x, train_y))
            if USE_V12_WEIGHTS:
                return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * if_s + CNN_WEIGHT * cnn_s
            return 0.40 * cw_s + (0.60 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
        local = normalize_scores(online_ensemble(test_x, window=15))
        return 0.40 * cw_s + (0.60 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn_s
    if category == "disjoint":
        g_s = normalize_scores(global_distance_score(train_x, test_x))
        if IF_DISJOINT:
            if_s = normalize_scores(isolation_forest_test(test_x, train_y))
            if USE_V12_WEIGHTS:
                return 0.35 * cw_s + 0.30 * g_s + (0.35 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
            return 0.30 * cw_s + 0.30 * g_s + (0.40 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
        local = normalize_scores(online_ensemble(test_x, window=15))
        return 0.30 * cw_s + 0.30 * g_s + (0.40 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn_s
    if category == "partial_overlap":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        return 0.35 * cw_s + 0.35 * pw + (0.30 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn_s
    if category == "test_within_train":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * pw + CNN_WEIGHT * cnn_s
    return np.zeros(len(test_x))


def build_hybrid_strong(window_dirs):
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


def scores_for_global(train_x, train_y, test_x, cw, global_models, metric_type):
    """v11 all_v12 baseline (global CNN ensemble, no routing)."""
    return scores_for_routed(train_x, train_y, test_x, cw, global_models, {},
                             metric_type)


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training stronger hybrid CW (500/15/3)…")
    t0 = time.time()
    cw = build_hybrid_strong(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    global_models, per_metric_models = train_cnn_routes(train_pool)

    def pred_baseline(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores = scores_for_global(sub_tr_x, sub_tr_y, sub_te_x, cw, global_models,
                                   info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    def pred_routed(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores = scores_for_routed(sub_tr_x, sub_tr_y, sub_te_x, cw,
                                   global_models, per_metric_models,
                                   info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    print("\n>>> Eval: v11 all_v12 baseline (global CNN)…")
    rep_base = evaluate(pred_baseline, holdout)
    print_summary(rep_base, name="v11 all_v12 baseline")

    print(">>> Eval: v12 per-metric CNN routing…")
    rep_routed = evaluate(pred_routed, holdout)
    print_summary(rep_routed, name="v12 per-metric CNN routed")

    delta = rep_routed["overall_f1"] - rep_base["overall_f1"]
    print(f"\n  Δ (routed − baseline) = {delta:+.4f}")

    report = {
        "baseline": rep_base,
        "routed": rep_routed,
        "delta_routed_vs_baseline": delta,
        "specialized_metrics_actually_routed": sorted(per_metric_models.keys()),
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v12_per_metric_cnn")
    return report


def generate_submission(output: Path = Path("submission_per_metric_cnn.json")) -> Path:
    print("\n>>> Training stronger hybrid CW on ALL 1000 windows…")
    t0 = time.time()
    cw = build_hybrid_strong(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    global_models, per_metric_models = train_cnn_routes(all_window_dirs())

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_for_routed(w.train_x, w.train_y, w.test_x, cw,
                                   global_models, per_metric_models, w.metric_type)
        k = int(round(len(w.test_x) * test_ratio))
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
    rep = run_validation()
    if rep["delta_routed_vs_baseline"] > 0.002:
        print(f"\nRouted CNN beats baseline by {rep['delta_routed_vs_baseline']:+.4f}; "
              "generating submission.")
        generate_submission()
    else:
        print(f"\nPer-metric CNN routing did not meaningfully help "
              f"(Δ = {rep['delta_routed_vs_baseline']:+.4f}); submission NOT generated.")
