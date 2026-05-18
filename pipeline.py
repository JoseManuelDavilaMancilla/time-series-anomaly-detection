"""
author v36 — bidirectional segment shift + end-of-window suppression.

v35 revealed two new patterns:
  1. **Over-correction**: v34's left-only shift over-corrected; we now have
     2 windows shifted RIGHT and 2 shifted LEFT (was 5 right, 0 left).
     Fix: try BOTH directions and pick the one with the highest in-segment
     score sum.
  2. **End-of-window spurious segments**: wid=944 and wid=510 have phantom
     short segments at the end of the window. Fix: when a prediction has 2+
     segments, drop the smallest if it sits in the last 10% of the window
     and is much smaller than the main segment.

Three variants tested:
  A. v34 baseline (left-only, merge=3)
  B. bidirectional shift (max=2), merge=3
  C. bidirectional shift (max=2) + EOW suppress, merge=3

Run:  uv run python v36_bidi_shift.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from shared_lib import (
    categorize_window,
    global_distance_score,
    isolation_forest_test,
    normalize_scores,
    online_ensemble,
    per_window_rf_score,
)
from v6_cnn import SPECIALIZED, build_training_pool as v6_pool
from v7_cnn_ensemble import CNN_SEEDS, ensemble_cnn_score, fit_cnn_with_seed
from v22_metadata_features import build_metadata_hybrid, scores_v14_with_meta
from v32_edge_fix import predict_segments_v32
from v34_postproc_fixes import extract_segments, fix_merge_close
from validation import (
    all_window_dirs, evaluate, load_window, print_summary, save_report, stratified_holdout,
)

SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60,
                  edge_pad_mode="reflect", boundary_penalty=0.0, edge_width=5)
CNN_WEIGHT = 0.35


def fix_bidi_shift(pred: np.ndarray, scores: np.ndarray, max_shift: int = 2) -> np.ndarray:
    """For each segment, try shifts in {-max_shift..+max_shift}. Pick the one
    with the highest score-sum across the shifted positions. Length is preserved."""
    n = len(pred)
    segs = extract_segments(pred)
    if not segs:
        return pred
    new_pred = np.zeros(n, dtype=int)
    for s, e in segs:
        seg_len = e - s + 1
        best_start = s
        best_score = float(scores[s : e + 1].sum())
        for shift in range(-max_shift, max_shift + 1):
            if shift == 0:
                continue
            new_s = s + shift
            new_e = new_s + seg_len - 1
            if new_s < 0 or new_e >= n:
                continue
            cand = float(scores[new_s : new_e + 1].sum())
            if cand > best_score:
                best_score = cand
                best_start = new_s
        new_pred[best_start : best_start + seg_len] = 1
    return new_pred


def fix_eow_suppress(pred: np.ndarray, eow_frac: float = 0.10,
                     min_main_ratio: float = 2.0) -> np.ndarray:
    """If prediction has 2+ segments, drop the smallest one if:
      - It sits in the last `eow_frac` of the window AND
      - It's at most 1/`min_main_ratio` the size of the largest segment

    Returns the modified prediction. The dropped positions become 0."""
    n = len(pred)
    segs = extract_segments(pred)
    if len(segs) < 2:
        return pred
    eow_start = int(n * (1 - eow_frac))
    seg_lengths = [e - s + 1 for s, e in segs]
    max_len = max(seg_lengths)
    new_pred = pred.copy()
    for (s, e), seg_len in zip(segs, seg_lengths):
        # In last eow_frac of window AND much smaller than the biggest segment
        if s >= eow_start and seg_len * min_main_ratio <= max_len:
            new_pred[s : e + 1] = 0
    return new_pred


def predict_segments_v36(scores, k, *, do_bidi_shift=True, do_merge=True,
                         do_eow=True, max_shift=2, merge_gap=3,
                         eow_frac=0.10, min_main_ratio=2.0, **seg_kwargs):
    pred = predict_segments_v32(scores, k, **seg_kwargs)
    if do_bidi_shift:
        pred = fix_bidi_shift(pred, scores, max_shift=max_shift)
    if do_merge:
        pred = fix_merge_close(pred, merge_gap=merge_gap)
    if do_eow:
        pred = fix_eow_suppress(pred, eow_frac=eow_frac, min_main_ratio=min_main_ratio)
    return pred


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training metadata CW + 3-seed CNN…")
    t0 = time.time()
    cw = build_metadata_hybrid(train_pool, mode="intervals")
    print(f"    cw fit {time.time() - t0:.1f}s")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    # Bring v34 (left-shift only) in for direct comparison
    from v34_postproc_fixes import predict_segments_v34

    def make_pred_v34():
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_v14_with_meta(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                          info, info.get("metric_type", "ALL"))
            return predict_segments_v34(scores, int(round(len(sub_te_x) * ratio)),
                                        do_left_shift=True, do_merge=True,
                                        max_shift=2, merge_gap=3, relative_thresh=0.95,
                                        **SEG_KWARGS)
        return pred

    def make_pred_v36(do_bidi, do_eow, eow_frac=0.10, min_main_ratio=2.0):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_v14_with_meta(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                          info, info.get("metric_type", "ALL"))
            return predict_segments_v36(scores, int(round(len(sub_te_x) * ratio)),
                                        do_bidi_shift=do_bidi, do_merge=True, do_eow=do_eow,
                                        max_shift=2, merge_gap=3, eow_frac=eow_frac,
                                        min_main_ratio=min_main_ratio, **SEG_KWARGS)
        return pred

    variants = {
        "v34 baseline (left-only, merge)":    make_pred_v34(),
        "v36 bidi shift, no EOW":             make_pred_v36(True, False),
        "v36 bidi + EOW (10%, ratio=2.0)":    make_pred_v36(True, True, 0.10, 2.0),
        "v36 bidi + EOW (15%, ratio=2.0)":    make_pred_v36(True, True, 0.15, 2.0),
        "v36 bidi + EOW (10%, ratio=3.0)":    make_pred_v36(True, True, 0.10, 3.0),
        "v36 EOW only (10%, ratio=2.0)":      make_pred_v36(False, True, 0.10, 2.0),
    }

    results = {}
    for name, pred_fn in variants.items():
        print(f"\n>>> Eval [{name}]")
        rep = evaluate(pred_fn, holdout)
        print_summary(rep, name=name)
        results[name] = rep

    base = results["v34 baseline (left-only, merge)"]["overall_f1"]
    print("\n──────  summary ──────")
    rows = sorted(results.items(), key=lambda kv: -kv[1]["overall_f1"])
    for name, rep in rows:
        d = rep["overall_f1"] - base
        print(f"  {name:<40}  F1={rep['overall_f1']:.4f}  Δ_vs_v34={d:+.4f}")

    winner = rows[0][0]
    report = {
        "results": {name: r["overall_f1"] for name, r in results.items()},
        "winner": winner,
        "delta_vs_v34": rows[0][1]["overall_f1"] - base,
        "seed": seed, "n_holdout": len(holdout),
    }
    save_report(report, "v36_bidi_shift")
    return report, cw, cnn_models


def generate_submission(do_bidi, do_eow, eow_frac, min_main_ratio, cw, cnn_models,
                       output: Path = Path("submission_bidi_eow.json")) -> Path:
    print(f"\n>>> Re-training on ALL 1000 windows…")
    t0 = time.time()
    cw_full = build_metadata_hybrid(all_window_dirs(), mode="intervals")
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
        scores = scores_v14_with_meta(w.train_x, w.train_y, w.test_x, cw_full, cnn_full,
                                      w.info, w.metric_type)
        k = int(round(len(w.test_x) * ratio))
        preds[w.wid] = predict_segments_v36(
            scores, k, do_bidi_shift=do_bidi, do_merge=True, do_eow=do_eow,
            max_shift=2, merge_gap=3, eow_frac=eow_frac, min_main_ratio=min_main_ratio,
            **SEG_KWARGS
        ).astype(int).tolist()
        if i % 100 == 0:
            print(f"    {i}/1000 ({time.time() - t0:.0f}s)")

    assert len(preds) == 1000
    output.write_text(
        json.dumps({"predictions": preds}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f">>> Wrote {output}")
    return output


VARIANT_PARAMS = {
    "v36 bidi shift, no EOW":             (True, False, 0.10, 2.0),
    "v36 bidi + EOW (10%, ratio=2.0)":    (True, True, 0.10, 2.0),
    "v36 bidi + EOW (15%, ratio=2.0)":    (True, True, 0.15, 2.0),
    "v36 bidi + EOW (10%, ratio=3.0)":    (True, True, 0.10, 3.0),
    "v36 EOW only (10%, ratio=2.0)":      (False, True, 0.10, 2.0),
}

if __name__ == "__main__":
    rep, cw, cnn_models = run_validation()
    if rep["winner"] != "v34 baseline (left-only, merge)" and rep["delta_vs_v34"] > 0.0005:
        do_bidi, do_eow, eow_frac, min_ratio = VARIANT_PARAMS[rep["winner"]]
        print(f"\n{rep['winner']} beats v34 by {rep['delta_vs_v34']:+.4f}; generating submission.")
        generate_submission(do_bidi, do_eow, eow_frac, min_ratio, cw, cnn_models)
    else:
        print(f"\nv36 did not meaningfully help (best Δ_vs_v34 = {rep['delta_vs_v34']:+.4f}); "
              "submission NOT generated.")
