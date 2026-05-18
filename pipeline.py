"""
author v21 — Per-metric TTA routing.

v20 showed TTA-batch helps SuccessRate (+0.018) and Count (+0.008) but hurts
ErrorCount/LatencySecond/QPS/ResourceUtil. Same bimodal pattern that worked
for hybrid CW: route per metric_type.

Sweep three routing configurations:
  A. TTA on SuccessRate only
  B. TTA on SuccessRate + Count
  C. TTA on SuccessRate + Count + ResourceUtil (the three weakest baselines)

Run:  uv run python v21_tta_routed.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Set

import numpy as np

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
from v6_cnn import SPECIALIZED, build_training_pool as v6_pool
from v7_cnn_ensemble import CNN_SEEDS, fit_cnn_with_seed
from v20_tta import ensemble_score_with_bn
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


def scores_with_routed_tta(train_x, train_y, test_x, cw, cnn_models, metric_type,
                           tta_metrics: Set[str]) -> np.ndarray:
    """Use batch-mode BN inference for metrics in tta_metrics, else eval-mode."""
    mode = "batch" if metric_type in tta_metrics else "eval"

    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
    cnn_s = normalize_scores(ensemble_score_with_bn(cnn_models, test_x, mode=mode))

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


def build_rf_hybrid(window_dirs):
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


VARIANTS = {
    "baseline_eval": frozenset(),
    "tta_sr_only": frozenset({"SuccessRate"}),
    "tta_sr_count": frozenset({"SuccessRate", "Count"}),
    "tta_sr_count_resource": frozenset({"SuccessRate", "Count", "ResourceUtilizationRate"}),
}


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training RF hybrid (500/15)…")
    t0 = time.time()
    cw = build_rf_hybrid(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training 3-seed CNN ensemble…")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    def predictor(tta_metrics: Set[str]):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_with_routed_tta(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                            info.get("metric_type", "ALL"), tta_metrics)
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    results = {}
    for name, metrics in VARIANTS.items():
        print(f"\n>>> Eval [{name}] TTA on metrics: {sorted(metrics) or 'none'}…")
        rep = evaluate(predictor(metrics), holdout)
        print_summary(rep, name=name)
        results[name] = rep["overall_f1"]

    print("\n──────  summary ──────")
    base = results["baseline_eval"]
    rows = sorted(results.items(), key=lambda kv: -kv[1])
    for name, f1 in rows:
        print(f"  {name:<25}  F1={f1:.4f}  Δ_vs_baseline={f1 - base:+.4f}")

    winner = rows[0][0]
    winner_f1 = rows[0][1]
    report = {
        "results": results,
        "winner": winner,
        "delta_vs_baseline": winner_f1 - base,
        "tta_metrics": sorted(VARIANTS[winner]),
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v21_tta_routed")
    return report, cw, cnn_models


def generate_submission(tta_metrics: Set[str], cw, cnn_models,
                        output: Path = Path("submission_tta_routed.json")) -> Path:
    print(f"\n>>> Re-training on ALL 1000 windows (TTA metrics: {sorted(tta_metrics)})…")
    t0 = time.time()
    cw_full = build_rf_hybrid(all_window_dirs())
    print(f"    rf fit {time.time() - t0:.1f}s")

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
        scores = scores_with_routed_tta(w.train_x, w.train_y, w.test_x, cw_full, cnn_full,
                                        w.metric_type, tta_metrics)
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
    if rep["winner"] != "baseline_eval" and rep["delta_vs_baseline"] > 0.001:
        tta_metrics = VARIANTS[rep["winner"]]
        print(f"\n{rep['winner']} beats baseline by {rep['delta_vs_baseline']:+.4f}; "
              "generating submission.")
        generate_submission(tta_metrics, cw, cnn_models)
    else:
        print(f"\nNo TTA routing helps (best Δ = {rep['delta_vs_baseline']:+.4f}); "
              "submission NOT generated.")
