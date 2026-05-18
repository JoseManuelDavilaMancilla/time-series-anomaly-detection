"""
author v14 — RF/LGBM hybrid CW with LGBM routed for QPS only.

v13 showed that 5-seed LGBM cross-window loses to RF overall (−0.006) but BEATS
RF specifically on QPS windows (LGBM 0.9957 vs RF 0.9888 = +0.0069 on QPS).

Hypothesis: route QPS windows to a 5-seed LGBM ensemble, all others to the
existing hybrid RF. Net expected gain ≈ (+0.0069 × 18/100) = +0.0012 overall on
validation, with a 1.4× transfer ratio that's ~+0.002 on the leaderboard.

Small but free, since we already know LGBM is right for QPS.

Stacks on top of v11 all_v12 (which is currently at 0.6238 LB).

Run:  uv run python v14_hybrid_qps_lgbm.py
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
warnings.filterwarnings("ignore", category=UserWarning)

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
from v6_cnn import build_training_pool as v6_pool, SPECIALIZED
from v7_cnn_ensemble import CNN_SEEDS, ensemble_cnn_score, fit_cnn_with_seed
from v13_gbm_seedhack import fit_lgbm, _build_pool as gbm_pool, gbm_ensemble_predict, GBM_SEEDS
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
QPS_METRIC = "QPS"


class RouterCW:
    """Hybrid CW that delegates QPS to an LGBM ensemble, others to the RF hybrid."""

    def __init__(self, rf_hybrid: HybridCrossWindowModel, lgbm_models: List):
        self.rf_hybrid = rf_hybrid
        self.lgbm_models = lgbm_models

    def predict_proba(self, test_x: np.ndarray, metric_type: str = "ALL") -> np.ndarray:
        if metric_type == QPS_METRIC and self.lgbm_models:
            return gbm_ensemble_predict(self.lgbm_models, test_x)
        return self.rf_hybrid.predict_proba(test_x, metric_type=metric_type)


def build_rf_hybrid(window_dirs):
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


def scores_v11(train_x, train_y, test_x, cw, cnn_models, metric_type):
    """v11 all_v12 ensemble (stronger CW + IF-on-test + v12 weights + CNN)."""
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
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

    print(">>> Training RF hybrid (500/15)…")
    t0 = time.time()
    rf_hybrid = build_rf_hybrid(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training 3-seed CNN ensemble…")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Training 5-seed LGBM (homogeneous) — for QPS routing…")
    Xg, yg = gbm_pool(train_pool)
    print(f"    GBM pool X={Xg.shape}")
    lgbm_models = []
    for s in GBM_SEEDS:
        t0 = time.time()
        lgbm_models.append(fit_lgbm(Xg, yg, seed=s))
        print(f"    lgbm seed={s} fit {time.time() - t0:.1f}s")

    cw_baseline = rf_hybrid
    cw_routed = RouterCW(rf_hybrid, lgbm_models)

    def predictor_for(cw):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_v11(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                info.get("metric_type", "ALL"))
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    print("\n>>> Eval: v11 baseline (RF hybrid, no LGBM routing)…")
    rep_base = evaluate(predictor_for(cw_baseline), holdout)
    print_summary(rep_base, name="v11 baseline (RF hybrid)")

    print(">>> Eval: v14 (LGBM routed for QPS only)…")
    rep_routed = evaluate(predictor_for(cw_routed), holdout)
    print_summary(rep_routed, name="v14 LGBM-for-QPS routed")

    delta = rep_routed["overall_f1"] - rep_base["overall_f1"]
    qps_delta = (rep_routed["by_metric_type"].get("QPS", {"mean_f1": 0})["mean_f1"]
                 - rep_base["by_metric_type"].get("QPS", {"mean_f1": 0})["mean_f1"])
    print(f"\n  Δ overall = {delta:+.4f}")
    print(f"  Δ on QPS (should be the only change) = {qps_delta:+.4f}")

    report = {
        "baseline": rep_base,
        "routed": rep_routed,
        "delta_overall": delta,
        "delta_qps": qps_delta,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v14_hybrid_qps_lgbm")
    return report


def generate_submission(output: Path = Path("submission_qps_lgbm_routed.json")) -> Path:
    print("\n>>> Training RF hybrid on ALL 1000 windows…")
    t0 = time.time()
    rf_hybrid = build_rf_hybrid(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training 3-seed CNN ensemble on full data…")
    Xc, yc = v6_pool(all_window_dirs())
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Training 5-seed LGBM on full data…")
    Xg, yg = gbm_pool(all_window_dirs())
    lgbm_models = []
    for s in GBM_SEEDS:
        t0 = time.time()
        lgbm_models.append(fit_lgbm(Xg, yg, seed=s))
        print(f"    lgbm seed={s} fit {time.time() - t0:.1f}s")

    cw = RouterCW(rf_hybrid, lgbm_models)

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_v11(w.train_x, w.train_y, w.test_x, cw, cnn_models, w.metric_type)
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
    if rep["delta_overall"] > 0.0005:
        print(f"\nLGBM-for-QPS routing wins by {rep['delta_overall']:+.4f}; generating submission.")
        generate_submission()
    else:
        print(f"\nLGBM-for-QPS routing did not help meaningfully "
              f"(Δ = {rep['delta_overall']:+.4f}); submission NOT generated.")
