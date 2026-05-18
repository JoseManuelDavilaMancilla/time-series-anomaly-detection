"""
author v15 — Re-sweep segment params on the v14 score distribution.

The current segment params (smooth=3, thr_frac=0.7, small_k=4, max_seg=60) were
optimized in v9 against the older score distribution (3-seed CNN + hybrid CW
only). v14's pipeline added stronger CW + IF-on-test + LGBM-for-QPS, which
produces sharper, more confident scores. The optimum segment parameters likely
shifted toward less smoothing and tighter growth thresholds.

Wider sweep this time:
  smooth     ∈ {1, 3, 5, 7}        — 1 = no smoothing
  thr_frac   ∈ {0.5, 0.6, 0.7, 0.8} — segment-growth threshold
  small_k    ∈ {3, 4, 5}
  max_seg    ∈ {40, 60, 80, 120}

4*4*3*4 = 192 configurations on cached scores (~5 min after training).

Run:  uv run python v15_seg_sweep_on_v14.py
"""

from __future__ import annotations

import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

from shared_lib import predict_segments
from v6_cnn import build_training_pool as v6_pool, SPECIALIZED
from v7_cnn_ensemble import CNN_SEEDS, fit_cnn_with_seed
from v13_gbm_seedhack import fit_lgbm, _build_pool as gbm_pool, GBM_SEEDS
from v14_hybrid_qps_lgbm import (
    RouterCW,
    build_rf_hybrid,
    scores_v11,
)
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    point_f1,
    print_summary,
    save_report,
    stratified_holdout,
    time_split,
)

SMOOTH = (1, 3, 5, 7)
THR_FRAC = (0.5, 0.6, 0.7, 0.8)
SMALL_K = (3, 4, 5)
MAX_SEG = (40, 60, 80, 120)


def precompute_scores(holdout, cw, cnn_models):
    cache = []
    for wdir in holdout:
        w = load_window(wdir)
        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = time_split(w.train_x, w.train_y, frac=0.70)
        ratio = float(sub_te_y.mean()) if len(sub_te_y) else 0.0
        scores = scores_v11(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models, w.metric_type)
        cache.append({
            "wid": w.wid, "metric_type": w.metric_type,
            "scores": scores, "ratio": ratio,
            "y_true": sub_te_y, "n": len(sub_te_x),
        })
    return cache


def eval_cache(cache, **seg_kwargs) -> float:
    f1s = []
    for c in cache:
        k = int(round(c["n"] * c["ratio"]))
        pred = predict_segments(c["scores"], k, **seg_kwargs)
        f1s.append(point_f1(c["y_true"], pred))
    return float(np.mean(f1s))


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

    print(">>> Training 5-seed LGBM (for QPS)…")
    Xg, yg = gbm_pool(train_pool)
    lgbm_models = []
    for s in GBM_SEEDS:
        t0 = time.time()
        lgbm_models.append(fit_lgbm(Xg, yg, seed=s))
        print(f"    lgbm seed={s} fit {time.time() - t0:.1f}s")

    cw = RouterCW(rf_hybrid, lgbm_models)

    print("\n>>> Pre-computing v14 scores for 100 holdout windows…")
    t0 = time.time()
    cache = precompute_scores(holdout, cw, cnn_models)
    print(f"    done in {time.time() - t0:.1f}s")

    # Baseline = current v14 params
    baseline_kwargs = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)
    baseline_f1 = eval_cache(cache, **baseline_kwargs)
    print(f"    baseline (smooth=3, thr=0.7, small_k=4, max_seg=60): F1={baseline_f1:.4f}")

    print("\n>>> Sweeping 192 segment configurations…")
    results = []
    t0 = time.time()
    total = len(SMOOTH) * len(THR_FRAC) * len(SMALL_K) * len(MAX_SEG)
    for i, (sm, th, sk, ms) in enumerate(itertools.product(SMOOTH, THR_FRAC, SMALL_K, MAX_SEG)):
        kwargs = dict(smooth=sm, thr_frac=th, small_k_cutoff=sk, max_seg=ms)
        f1 = eval_cache(cache, **kwargs)
        results.append({
            "smooth": sm, "thr_frac": th, "small_k_cutoff": sk, "max_seg": ms,
            "f1": f1, "delta_vs_baseline": f1 - baseline_f1,
        })
        if (i + 1) % 32 == 0:
            print(f"    {i + 1}/{total}  ({time.time() - t0:.0f}s)")

    results.sort(key=lambda r: -r["f1"])
    print("\n>>> Top 15 configurations:")
    for r in results[:15]:
        print(f"  smooth={r['smooth']} thr={r['thr_frac']} small_k={r['small_k_cutoff']} "
              f"max_seg={r['max_seg']:>3}  F1={r['f1']:.4f}  Δ={r['delta_vs_baseline']:+.4f}")

    best = results[0]
    print(f"\n  Best: {best}")

    report = {
        "baseline_kwargs": baseline_kwargs,
        "baseline_f1": baseline_f1,
        "best_kwargs": {k: best[k] for k in ("smooth", "thr_frac", "small_k_cutoff", "max_seg")},
        "best_f1": best["f1"],
        "delta_vs_baseline": best["delta_vs_baseline"],
        "all_results": results,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v15_seg_sweep_on_v14")
    return report


def generate_submission(seg_kwargs: dict,
                        output: Path = Path("submission_qps_lgbm_segtuned.json")) -> Path:
    print(f"\n>>> Training all models on ALL 1000 windows…")
    t0 = time.time()
    rf_hybrid = build_rf_hybrid(all_window_dirs())
    print(f"    rf fit {time.time() - t0:.1f}s")

    Xc, yc = v6_pool(all_window_dirs())
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    Xg, yg = gbm_pool(all_window_dirs())
    lgbm_models = []
    for s in GBM_SEEDS:
        t0 = time.time()
        lgbm_models.append(fit_lgbm(Xg, yg, seed=s))
        print(f"    lgbm seed={s} fit {time.time() - t0:.1f}s")

    cw = RouterCW(rf_hybrid, lgbm_models)

    print(f">>> Generating predictions with {seg_kwargs}…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_v11(w.train_x, w.train_y, w.test_x, cw, cnn_models, w.metric_type)
        k = int(round(len(w.test_x) * test_ratio))
        preds[w.wid] = predict_segments(scores, k, **seg_kwargs).astype(int).tolist()
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
    if rep["delta_vs_baseline"] > 0.0005:
        print(f"\nBest config beats baseline by {rep['delta_vs_baseline']:+.4f}; "
              "generating submission.")
        generate_submission(rep["best_kwargs"])
    else:
        print(f"\nBaseline segment params already optimal "
              f"(best Δ = {rep['delta_vs_baseline']:+.4f}); submission NOT generated.")
