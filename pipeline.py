"""
author v9 — Segment-selection parameter sweep.

Now that the ensemble side is saturated, see whether tighter or looser segment
selection squeezes more out of the existing scores. No model retraining — we
reuse the v7 3-seed CNN ensemble pipeline and only vary the segment params.

Swept axes:
  smooth     ∈ {3, 5, 7, 9}        — denoise width of the score smoother
  thr_frac   ∈ {0.4, 0.5, 0.6, 0.7} — segment-growth threshold
  small_k    ∈ {3, 4, 5}            — k below which we fall back to top-k
  max_seg    ∈ {60, 80, 120}        — cap a single segment's length

That's 4*4*3*3 = 144 configurations. To keep this tractable we evaluate each
on the same 100-window holdout. Each eval is ~3s (no model fitting), so ~7 min
total.

Run:  uv run python v9_seg_sweep.py
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

from shared_lib import (
    CrossWindowModel,
    HybridCrossWindowModel,
    predict_segments,
)
from v6_cnn import (
    SPECIALIZED,
    build_training_pool,
)
from v7_cnn_ensemble import (
    CNN_SEEDS,
    build_hybrid,
    fit_cnn_with_seed,
    scores_weighted,
)
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

SMOOTH = (3, 5, 7, 9)
THR_FRAC = (0.4, 0.5, 0.6, 0.7)
SMALL_K = (3, 4, 5)
MAX_SEG = (60, 80, 120)


def precompute_scores(holdout, cw, cnn_models):
    """Compute the (scores, ratio, true_labels) per window so we don't pay
    score-computation cost every sweep config. ~2 min for 100 windows."""
    cache = []
    from validation import time_split, load_window
    for wdir in holdout:
        w = load_window(wdir)
        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = time_split(w.train_x, w.train_y, frac=0.70)
        ratio = float(sub_te_y.mean()) if len(sub_te_y) else 0.0
        scores = scores_weighted(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                 w.metric_type)
        cache.append({
            "wid": w.wid,
            "metric_type": w.metric_type,
            "scores": scores,
            "ratio": ratio,
            "y_true": sub_te_y,
            "n": len(sub_te_x),
        })
    return cache


def eval_cache(cache, **seg_kwargs) -> float:
    from validation import point_f1
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

    print(">>> Training hybrid…")
    t0 = time.time()
    cw = build_hybrid(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Building CNN pool + training 3 CNNs…")
    X, y = build_training_pool(train_pool)
    models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        models.append(fit_cnn_with_seed(X, y, s))
        print(f"    seed={s}  fit {time.time() - t0:.1f}s")

    print("\n>>> Pre-computing scores for all 100 holdout windows…")
    t0 = time.time()
    cache = precompute_scores(holdout, cw, models)
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n>>> Sweeping segment params over the cached scores…")
    results = []
    baseline_kwargs = dict(smooth=5, thr_frac=0.6, small_k_cutoff=4, max_seg=80)
    baseline_f1 = eval_cache(cache, **baseline_kwargs)
    print(f"    baseline (v7 params): F1 = {baseline_f1:.4f}")

    t0 = time.time()
    total = len(SMOOTH) * len(THR_FRAC) * len(SMALL_K) * len(MAX_SEG)
    for i, (sm, th, sk, ms) in enumerate(itertools.product(SMOOTH, THR_FRAC, SMALL_K, MAX_SEG)):
        kwargs = dict(smooth=sm, thr_frac=th, small_k_cutoff=sk, max_seg=ms)
        f1 = eval_cache(cache, **kwargs)
        results.append({"smooth": sm, "thr_frac": th, "small_k": sk, "max_seg": ms,
                        "f1": f1, "delta_vs_baseline": f1 - baseline_f1})
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{total}  ({time.time() - t0:.0f}s)")

    results.sort(key=lambda r: -r["f1"])
    print(f"\n>>> Top 10 configurations:")
    for r in results[:10]:
        print(f"  smooth={r['smooth']} thr={r['thr_frac']} small_k={r['small_k']} "
              f"max_seg={r['max_seg']}  F1={r['f1']:.4f}  Δ={r['delta_vs_baseline']:+.4f}")

    print(f"\n>>> Bottom 5 configurations (sanity check):")
    for r in results[-5:]:
        print(f"  smooth={r['smooth']} thr={r['thr_frac']} small_k={r['small_k']} "
              f"max_seg={r['max_seg']}  F1={r['f1']:.4f}  Δ={r['delta_vs_baseline']:+.4f}")

    best = results[0]
    print(f"\n  Best: {best}")
    report = {
        "baseline_f1": baseline_f1,
        "baseline_kwargs": baseline_kwargs,
        "best_kwargs": {k: best[k] for k in ("smooth", "thr_frac", "small_k", "max_seg")},
        "best_f1": best["f1"],
        "delta_vs_baseline": best["delta_vs_baseline"],
        "all_results": results,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v9_seg_sweep")
    return report


def generate_submission(seg_kwargs: dict,
                        output: Path = Path("submission_segtuned.json")) -> Path:
    # Map small_k → small_k_cutoff for the actual call
    kwargs = {
        "smooth": seg_kwargs["smooth"],
        "thr_frac": seg_kwargs["thr_frac"],
        "small_k_cutoff": seg_kwargs["small_k"],
        "max_seg": seg_kwargs["max_seg"],
    }
    print(f"\n>>> Training hybrid on ALL 1000 windows…")
    t0 = time.time()
    cw = build_hybrid(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Building full CNN pool + training 3 CNNs…")
    X, y = build_training_pool(all_window_dirs())
    models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        models.append(fit_cnn_with_seed(X, y, s))
        print(f"    seed={s}  fit {time.time() - t0:.1f}s")

    print(f">>> Generating predictions with {kwargs}…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_weighted(w.train_x, w.train_y, w.test_x, cw, models,
                                 w.metric_type)
        k = int(round(len(w.test_x) * test_ratio))
        preds[w.wid] = predict_segments(scores, k, **kwargs).astype(int).tolist()
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
    if rep["delta_vs_baseline"] > 0.001:
        print(f"\nBest config beats baseline by {rep['delta_vs_baseline']:+.4f}; generating submission.")
        generate_submission(rep["best_kwargs"])
    else:
        print(f"\nNo meaningful gain from segment-param tuning "
              f"(best Δ = {rep['delta_vs_baseline']:+.4f}); submission NOT generated.")
