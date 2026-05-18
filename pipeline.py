"""
author v34 — Targeted post-processing fixes from v33 findings.

v33 classified 11 holdout failures (F1 < 0.9 after v32 reflect fix) into:
  - 5 "off_by_few" cases — ALL shifted RIGHT by 1-4 positions
  - 2 "split_segment" cases — one truth segment broken into 2 predictions
    by a temporary score dip
  - 2 "extra_segments" — predicted more segments than truth
  - 2 "other"

This script implements two surgical post-processing fixes:

**FIX A — left-shift correction**:
  For each predicted segment, compute the score sum across positions covered.
  Try shifting the segment 1 or 2 positions left (clamped to window). If the
  total score within the shifted segment is at least 0.95× the original score
  (i.e., comparable), prefer the leftmost equivalent position. This corrects
  the CNN's asymmetric context bias without aggressively reshaping segments.

**FIX B — merge close segments**:
  After greedy segment selection, walk through the predicted segments. If
  two adjacent segments are within `merge_gap` positions of each other, merge
  them into one segment spanning the union. The intermediate "gap" positions
  become 1's. This costs no extra budget (gaps are short, marginal increase
  is small relative to budget k).

Both fixes are applied AFTER `predict_segments_v32`. Stack independently so
we can isolate their contributions.

Run:  uv run python v34_postproc_fixes.py
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
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60,
                  edge_pad_mode="reflect", boundary_penalty=0.0, edge_width=5)
CNN_WEIGHT = 0.35


def extract_segments(pred: np.ndarray) -> List[Tuple[int, int]]:
    if pred.sum() == 0:
        return []
    diffs = np.diff(np.concatenate([[0], pred.astype(int), [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def fix_left_shift(pred: np.ndarray, scores: np.ndarray, max_shift: int = 2,
                   relative_thresh: float = 0.95) -> np.ndarray:
    """For each segment, try shifting left by 1..max_shift positions and keep
    the leftmost shift whose score-sum is at least relative_thresh × original.
    Length is preserved."""
    n = len(pred)
    segs = extract_segments(pred)
    if not segs:
        return pred
    # Build new pred from scratch to avoid double-counting
    new_pred = np.zeros(n, dtype=int)
    for s, e in segs:
        seg_len = e - s + 1
        original = float(scores[s : e + 1].sum())
        best_start = s
        for shift in range(1, max_shift + 1):
            new_s = s - shift
            if new_s < 0:
                break
            new_e = new_s + seg_len - 1
            cand = float(scores[new_s : new_e + 1].sum())
            if cand >= relative_thresh * original:
                best_start = new_s
        new_pred[best_start : best_start + seg_len] = 1
    return new_pred


def fix_merge_close(pred: np.ndarray, merge_gap: int = 3) -> np.ndarray:
    """If two adjacent predicted segments are within `merge_gap` of each other,
    fill the gap to merge them. Cheap fill — small positions get marked."""
    n = len(pred)
    segs = extract_segments(pred)
    if len(segs) < 2:
        return pred
    new_pred = pred.copy()
    for i in range(len(segs) - 1):
        s1, e1 = segs[i]
        s2, e2 = segs[i + 1]
        gap = s2 - e1 - 1
        if 0 < gap <= merge_gap:
            new_pred[e1 + 1 : s2] = 1
    return new_pred


def predict_segments_v34(scores, k, *, do_left_shift=True, do_merge=True,
                         max_shift=2, merge_gap=3, relative_thresh=0.95,
                         **seg_kwargs):
    """v32 predict_segments + optional left-shift correction + merge close segments."""
    pred = predict_segments_v32(scores, k, **seg_kwargs)
    if do_left_shift:
        pred = fix_left_shift(pred, scores, max_shift=max_shift,
                              relative_thresh=relative_thresh)
    if do_merge:
        pred = fix_merge_close(pred, merge_gap=merge_gap)
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

    def make_predictor(do_ls, do_merge, max_shift=2, merge_gap=3, relative_thresh=0.95):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_v14_with_meta(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                          info, info.get("metric_type", "ALL"))
            return predict_segments_v34(scores, int(round(len(sub_te_x) * ratio)),
                                        do_left_shift=do_ls, do_merge=do_merge,
                                        max_shift=max_shift, merge_gap=merge_gap,
                                        relative_thresh=relative_thresh, **SEG_KWARGS)
        return pred

    variants = {
        "v32 baseline (no postproc)":        (False, False, 0, 0, 1.0),
        "+ left_shift(max=1, t=0.95)":       (True,  False, 1, 0, 0.95),
        "+ left_shift(max=2, t=0.95)":       (True,  False, 2, 0, 0.95),
        "+ left_shift(max=2, t=0.90)":       (True,  False, 2, 0, 0.90),
        "+ merge(gap=3)":                    (False, True,  0, 3, 1.0),
        "+ merge(gap=5)":                    (False, True,  0, 5, 1.0),
        "+ both (shift=2, merge=3)":         (True,  True,  2, 3, 0.95),
        "+ both (shift=2, merge=5)":         (True,  True,  2, 5, 0.95),
    }

    results = {}
    for name, (do_ls, do_merge, max_shift, merge_gap, rt) in variants.items():
        print(f"\n>>> Eval [{name}]")
        rep = evaluate(make_predictor(do_ls, do_merge, max_shift, merge_gap, rt), holdout)
        print_summary(rep, name=name)
        results[name] = rep

    base = results["v32 baseline (no postproc)"]["overall_f1"]
    print("\n──────  summary ──────")
    rows = sorted(results.items(), key=lambda kv: -kv[1]["overall_f1"])
    for name, rep in rows:
        d = rep["overall_f1"] - base
        print(f"  {name:<40}  F1={rep['overall_f1']:.4f}  Δ_vs_base={d:+.4f}")

    winner = rows[0][0]
    report = {
        "results": {name: r["overall_f1"] for name, r in results.items()},
        "winner": winner,
        "delta_vs_baseline": rows[0][1]["overall_f1"] - base,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v34_postproc_fixes")
    return report, cw, cnn_models, variants


def generate_submission(do_ls, do_merge, max_shift, merge_gap, rt, cw, cnn_models,
                        output: Path = Path("submission_postproc.json")) -> Path:
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
        preds[w.wid] = predict_segments_v34(
            scores, k, do_left_shift=do_ls, do_merge=do_merge, max_shift=max_shift,
            merge_gap=merge_gap, relative_thresh=rt, **SEG_KWARGS
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


if __name__ == "__main__":
    rep, cw, cnn_models, variants = run_validation()
    if rep["winner"] != "v32 baseline (no postproc)" and rep["delta_vs_baseline"] > 0.0005:
        do_ls, do_merge, max_shift, merge_gap, rt = variants[rep["winner"]]
        print(f"\n{rep['winner']} beats baseline by {rep['delta_vs_baseline']:+.4f}; generating submission.")
        generate_submission(do_ls, do_merge, max_shift, merge_gap, rt, cw, cnn_models)
    else:
        print(f"\nNo postproc fix meaningfully helped (best Δ = {rep['delta_vs_baseline']:+.4f}); "
              "submission NOT generated.")
