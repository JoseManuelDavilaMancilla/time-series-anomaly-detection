"""
author v2 — LightGBM cross-window model (replaces the RF) + segments.

Hypothesis: gradient-boosted trees handle the 9% class imbalance and noisy point
features better than RF. We keep the v1 segment-selection prediction on top.
Includes a 3-way comparison so we can attribute any gain cleanly to the model
swap (versus segments alone).

Run:  uv run python v2_lgbm.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from shared_lib import (
    CrossWindowModel,
    predict_segments,
    predict_topk,
    v8_style_scores,
)
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

SEG_KWARGS = dict(smooth=5, thr_frac=0.6, small_k_cutoff=4, max_seg=80)


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training RF cross-window (control)…")
    t0 = time.time()
    cw_rf = CrossWindowModel(backend="rf", per_metric=False,
                             n_estimators=200, max_depth=12).fit(train_pool)
    print(f"    rf fit {time.time() - t0:.1f}s")

    print(">>> Training LightGBM cross-window…")
    t0 = time.time()
    cw_lgbm = CrossWindowModel(backend="lgbm", per_metric=False,
                               n_estimators=400, max_depth=8,
                               min_samples_leaf=10).fit(train_pool)
    print(f"    lgbm fit {time.time() - t0:.1f}s")

    def make_predictor(cw, mode):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores, _ = v8_style_scores(sub_tr_x, sub_tr_y, sub_te_x, cw,
                                        metric_type=info.get("metric_type", "ALL"))
            k = int(round(len(sub_te_x) * ratio))
            return predict_topk(scores, k) if mode == "topk" else predict_segments(scores, k, **SEG_KWARGS)
        return pred

    print(">>> Eval: RF + segments  (v1 baseline)")
    rep_rf_seg = evaluate(make_predictor(cw_rf, "segments"), holdout)
    print_summary(rep_rf_seg, name="RF + segments")

    print(">>> Eval: LGBM + segments  (v2)")
    rep_lgbm_seg = evaluate(make_predictor(cw_lgbm, "segments"), holdout)
    print_summary(rep_lgbm_seg, name="LGBM + segments")

    print(">>> Eval: LGBM + top-k  (ablation)")
    rep_lgbm_topk = evaluate(make_predictor(cw_lgbm, "topk"), holdout)
    print_summary(rep_lgbm_topk, name="LGBM + top-k")

    delta_model = rep_lgbm_seg["overall_f1"] - rep_rf_seg["overall_f1"]
    delta_seg = rep_lgbm_seg["overall_f1"] - rep_lgbm_topk["overall_f1"]
    print(f"    Δ (LGBM − RF, both w/ segments) = {delta_model:+.4f}")
    print(f"    Δ (segments − top-k, both LGBM) = {delta_seg:+.4f}")

    report = {
        "rf_segments": rep_rf_seg,
        "lgbm_segments": rep_lgbm_seg,
        "lgbm_topk": rep_lgbm_topk,
        "delta_model_swap": delta_model,
        "delta_segments_vs_topk": delta_seg,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v2_lgbm")
    return report


def generate_submission(output: Path = Path("submission_lgbm.json")) -> Path:
    print("\n>>> Training LightGBM cross-window on ALL 1000 windows…")
    t0 = time.time()
    cw = CrossWindowModel(backend="lgbm", per_metric=False,
                          n_estimators=400, max_depth=8,
                          min_samples_leaf=10).fit(all_window_dirs())
    print(f"    fit time {time.time() - t0:.1f}s")

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    cats: dict[str, int] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores, cat = v8_style_scores(w.train_x, w.train_y, w.test_x, cw,
                                      metric_type=w.metric_type)
        cats[cat] = cats.get(cat, 0) + 1
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
    print(f">>> Categories: {cats}")
    return output


if __name__ == "__main__":
    rep = run_validation()
    if rep["lgbm_segments"]["overall_f1"] >= rep["rf_segments"]["overall_f1"] - 0.005:
        print("\nLGBM ≥ RF (within tolerance). Generating submission.")
        generate_submission()
    else:
        print(f"\n!! LGBM underperformed RF by {rep['rf_segments']['overall_f1'] - rep['lgbm_segments']['overall_f1']:.4f}; "
              "submission NOT generated. Inspect results/v2_lgbm_eval.json")
