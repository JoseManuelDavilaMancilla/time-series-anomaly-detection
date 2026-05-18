"""
author v1 — Segment selection on v8-style scores.

Hypothesis: teammate's v8 produces good per-point scores but its prediction step
(point-wise top-k via argpartition) fragments predictions. Anomalies in the
truth are contiguous (mean segment length ~17), so picking K isolated peaks
misses neighboring true-positive points.

This script:
  1. Validates v8-top-k vs v8+segments on a 100-window stratified holdout.
  2. Generates `submission_segments.json` on the full 1000 windows using
     segment selection.

Run:  uv run python v1_segments.py
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
    DATASET_ROOT,
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
    time_split,
)


# ─────────────────────────────────────────────
# Validation: compare top-k vs segments on a holdout
# ─────────────────────────────────────────────


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10% of 1000 windows)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training cross-window RF on train_pool…")
    t0 = time.time()
    cw = CrossWindowModel(backend="rf", per_metric=False, n_estimators=200, max_depth=12).fit(train_pool)
    print(f"    fit time {time.time() - t0:.1f}s")

    def make_predictor(prediction_mode: str):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores, _ = v8_style_scores(
                sub_tr_x, sub_tr_y, sub_te_x, cw, metric_type=info.get("metric_type", "ALL")
            )
            k = int(round(len(sub_te_x) * ratio))
            if prediction_mode == "topk":
                return predict_topk(scores, k)
            return predict_segments(scores, k, smooth=5, thr_frac=0.6, small_k_cutoff=4, max_seg=80)
        return pred

    print(">>> Evaluating v8-style + TOP-K (teammate's original prediction)…")
    rep_topk = evaluate(make_predictor("topk"), holdout)
    print_summary(rep_topk, name="v8-style + top-k")

    print(">>> Evaluating v8-style + SEGMENT selection (author v1)…")
    rep_seg = evaluate(make_predictor("segments"), holdout)
    print_summary(rep_seg, name="v8-style + segments")

    delta = rep_seg["overall_f1"] - rep_topk["overall_f1"]
    print(f"    Δ (segments − top-k) = {delta:+.4f}")

    combined = {
        "topk": rep_topk,
        "segments": rep_seg,
        "delta_segments_vs_topk": delta,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(combined, "v1_segments")
    return combined


# ─────────────────────────────────────────────
# Final submission generation on the full 1000 windows
# ─────────────────────────────────────────────


def generate_submission(output: Path = Path("submission_segments.json")) -> Path:
    print("\n>>> Training cross-window RF on ALL 1000 windows for submission…")
    t0 = time.time()
    cw = CrossWindowModel(backend="rf", per_metric=False, n_estimators=200, max_depth=12).fit(all_window_dirs())
    print(f"    fit time {time.time() - t0:.1f}s")

    print(">>> Generating predictions for all 1000 windows…")
    predictions: dict[str, list[int]] = {}
    category_counts: dict[str, int] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores, category = v8_style_scores(
            w.train_x, w.train_y, w.test_x, cw, metric_type=w.metric_type
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        k = int(round(len(w.test_x) * test_ratio))
        pred = predict_segments(scores, k, smooth=5, thr_frac=0.6, small_k_cutoff=4, max_seg=80)
        predictions[w.wid] = pred.astype(int).tolist()
        if i % 100 == 0:
            print(f"    {i}/1000 ({time.time() - t0:.0f}s elapsed)")

    assert len(predictions) == 1000, f"Expected 1000 windows, got {len(predictions)}"
    output.write_text(
        json.dumps({"predictions": predictions}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"\n>>> Wrote {output} with {len(predictions)} windows")
    print(f">>> Category distribution: {category_counts}")
    return output


if __name__ == "__main__":
    rep = run_validation()
    if rep["delta_segments_vs_topk"] > -0.01:
        print("\nSegment selection ≥ top-k (within noise). Generating submission.")
        generate_submission()
    else:
        print(f"\n!! Segment selection underperformed top-k by {-rep['delta_segments_vs_topk']:.4f}; "
              "submission NOT generated. Inspect results/v1_segments_eval.json")
