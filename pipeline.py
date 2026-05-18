"""
author v7 — Multi-seed CNN ensemble + score-space averaging variant.

Two questions this answers:

1. Is the v6 CNN gain (+0.006) seed-stable? Train CNN with 3 seeds and average
   their per-point probabilities. If single-seed gain is real, the 3-seed
   average should be ≥ single-seed (variance reduction). If single-seed was
   lucky, the 3-seed average will regress.

2. Does the weighted-add blend (v6) beat or lose to a clean score-space average
   of [hybrid_score, cnn_score]? Different way to combine the two signals.

Run:  uv run python v7_cnn_ensemble.py
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
    normalize_scores,
    online_ensemble,
    per_window_rf_score,
    predict_segments,
    v8_style_scores,
)
from v6_cnn import (
    SEG_KWARGS,
    SPECIALIZED,
    TinyCNN,
    build_contexts,
    build_training_pool,
    cnn_score,
    fit_cnn,
)
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

N_SEEDS = 3
CNN_SEEDS = (42, 123, 7)


def build_hybrid(window_dirs):
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=200, max_depth=12).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=200, max_depth=12).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


def fit_cnn_with_seed(X, y, seed: int):
    return fit_cnn(X, y, seed=seed)


def ensemble_cnn_score(models, test_x: np.ndarray) -> np.ndarray:
    """Average per-point probability across CNN models."""
    scores = np.stack([cnn_score(m, test_x) for m in models], axis=0)
    return scores.mean(axis=0)


def scores_weighted(train_x, train_y, test_x, cw, cnn_models,
                    metric_type, cnn_weight=0.35):
    """v6-style weighted ensemble using the multi-seed CNN average."""
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
    cnn_s = normalize_scores(ensemble_cnn_score(cnn_models, test_x))

    if category == "constant_train":
        local = normalize_scores(online_ensemble(test_x, window=15))
        scores = 0.40 * cw_s + (0.60 - cnn_weight) * local + cnn_weight * cnn_s
    elif category == "disjoint":
        g = normalize_scores(global_distance_score(train_x, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        scores = 0.30 * cw_s + 0.30 * g + (0.40 - cnn_weight) * local + cnn_weight * cnn_s
    elif category == "partial_overlap":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        scores = 0.35 * cw_s + 0.35 * pw + (0.30 - cnn_weight) * local + cnn_weight * cnn_s
    elif category == "test_within_train":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        scores = (0.50 - cnn_weight) * cw_s + 0.50 * pw + cnn_weight * cnn_s
    else:
        scores = np.zeros(len(test_x))
    return scores


def scores_score_avg(train_x, train_y, test_x, cw, cnn_models, metric_type):
    """Alternative: compute the hybrid+segments score, the CNN score, and 50/50 them."""
    hybrid_scores, _ = v8_style_scores(train_x, train_y, test_x, cw, metric_type=metric_type)
    hybrid_n = normalize_scores(hybrid_scores)
    cnn_n = normalize_scores(ensemble_cnn_score(cnn_models, test_x))
    return 0.5 * hybrid_n + 0.5 * cnn_n


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training hybrid cross-window…")
    t0 = time.time()
    cw = build_hybrid(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    print(f">>> Training CNN ensemble ({N_SEEDS} seeds: {CNN_SEEDS})…")
    X, y = build_training_pool(train_pool)
    print(f"    pool X.shape={X.shape}  y.mean={y.mean():.3f}")
    cnn_models = []
    for s in CNN_SEEDS:
        print(f">>> Training CNN seed={s}")
        t0 = time.time()
        m = fit_cnn_with_seed(X, y, s)
        cnn_models.append(m)
        print(f"    fit {time.time() - t0:.1f}s")

    # Variant A: v6 single-seed CNN (using first seed)
    def pred_single_cnn(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores = scores_weighted(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models[:1],
                                 info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    # Variant B: v7 multi-seed CNN ensemble
    def pred_multi_cnn(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores = scores_weighted(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                 info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    # Variant C: score-space averaging (50/50 hybrid + multi-seed CNN)
    def pred_score_avg(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores = scores_score_avg(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                  info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    # Hybrid + segments only (v3 baseline)
    def pred_baseline(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores, _ = v8_style_scores(sub_tr_x, sub_tr_y, sub_te_x, cw,
                                    metric_type=info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    print("\n>>> Eval: v3 hybrid + segments (baseline)…")
    rep_v3 = evaluate(pred_baseline, holdout)
    print_summary(rep_v3, name="v3 baseline")

    print(">>> Eval: v6 single-seed CNN…")
    rep_v6 = evaluate(pred_single_cnn, holdout)
    print_summary(rep_v6, name="v6 single CNN")

    print(">>> Eval: v7 multi-seed CNN ensemble…")
    rep_v7 = evaluate(pred_multi_cnn, holdout)
    print_summary(rep_v7, name="v7 multi-seed CNN")

    print(">>> Eval: score-space avg (hybrid 0.5 + multi-CNN 0.5)…")
    rep_avg = evaluate(pred_score_avg, holdout)
    print_summary(rep_avg, name="score-space avg")

    print(f"\n    Δ (v6 single − v3) = {rep_v6['overall_f1'] - rep_v3['overall_f1']:+.4f}")
    print(f"    Δ (v7 multi − v3)  = {rep_v7['overall_f1'] - rep_v3['overall_f1']:+.4f}")
    print(f"    Δ (v7 multi − v6 single) = {rep_v7['overall_f1'] - rep_v6['overall_f1']:+.4f}")
    print(f"    Δ (score-avg − v3) = {rep_avg['overall_f1'] - rep_v3['overall_f1']:+.4f}")
    print(f"    Δ (score-avg − v7) = {rep_avg['overall_f1'] - rep_v7['overall_f1']:+.4f}")

    report = {
        "v3_baseline": rep_v3,
        "v6_single_cnn": rep_v6,
        "v7_multi_cnn": rep_v7,
        "score_space_avg": rep_avg,
        "deltas": {
            "v6_minus_v3": rep_v6["overall_f1"] - rep_v3["overall_f1"],
            "v7_minus_v3": rep_v7["overall_f1"] - rep_v3["overall_f1"],
            "v7_minus_v6": rep_v7["overall_f1"] - rep_v6["overall_f1"],
            "avg_minus_v3": rep_avg["overall_f1"] - rep_v3["overall_f1"],
            "avg_minus_v7": rep_avg["overall_f1"] - rep_v7["overall_f1"],
        },
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v7_cnn_ensemble")
    return report, cw, cnn_models


def generate_submission(strategy: str,
                        output: Path = Path("submission_cnn_ensemble.json")) -> Path:
    print(f"\n>>> Training hybrid on ALL 1000 windows…")
    t0 = time.time()
    cw = build_hybrid(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Building full CNN pool…")
    X, y = build_training_pool(all_window_dirs())
    print(f"    X.shape={X.shape}  y.mean={y.mean():.3f}")

    cnn_models = []
    for s in CNN_SEEDS:
        print(f">>> Training CNN seed={s} (full data)")
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(X, y, s))
        print(f"    fit {time.time() - t0:.1f}s")

    print(f">>> Generating predictions (strategy={strategy})…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        if strategy == "weighted":
            scores = scores_weighted(w.train_x, w.train_y, w.test_x, cw, cnn_models,
                                     w.metric_type)
        else:
            scores = scores_score_avg(w.train_x, w.train_y, w.test_x, cw, cnn_models,
                                      w.metric_type)
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
    rep, _, _ = run_validation()
    v6_v3 = rep["deltas"]["v6_minus_v3"]
    v7_v3 = rep["deltas"]["v7_minus_v3"]
    avg_v3 = rep["deltas"]["avg_minus_v3"]

    # Pick the best strategy that beats v3 by ≥ 0.003
    best_label, best_delta = "v3", 0.0
    for label, delta in (("v7_weighted", v7_v3), ("score_avg", avg_v3), ("v6_weighted", v6_v3)):
        if delta > best_delta:
            best_label, best_delta = label, delta

    if best_label in ("v7_weighted", "score_avg") and best_delta > 0.003:
        print(f"\n{best_label} beats v3 by {best_delta:+.4f}; generating submission.")
        strategy = "weighted" if best_label == "v7_weighted" else "score_avg"
        generate_submission(strategy)
    else:
        print(f"\nNo variant improved meaningfully over the single-seed v6 CNN; "
              "keeping submission_cnn.json as the best CNN submission.")
