"""
author v33 — investigate ALL failure cases (across all metric types) to find
common patterns for further post-processing fixes.

v31 showed that ResourceUtil's 0.95 ceiling came from 2/17 windows with
specific edge-effect bugs. v32 fixed one of them (SuccessRate) via reflective
padding. This script extends the investigation to every holdout window where
v32's pipeline still scores below 0.9 F1, looking for common patterns we can
post-process around.

Outputs a structured table: (wid, metric_type, category, ratio, F1, failure
mode classification). Failure mode is one of:
  - "off_by_few" — predicted segment exists but is shifted N positions
  - "extra_segment" — model predicted a real segment + a phantom one
  - "missing_segment" — model missed a true segment entirely
  - "wrong_length" — segment overlaps but is too short/long
  - "split_segment" — one truth seg split into multiple predicted segs
  - "other"

Then for each common pattern, propose a specific post-processing fix.

Run:  uv run python v33_investigate_all.py
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter, defaultdict
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
from v22_metadata_features import build_metadata_hybrid
from v32_edge_fix import predict_segments_v32
from validation import (
    all_window_dirs,
    load_window,
    point_f1,
    save_report,
    stratified_holdout,
    time_split,
)

SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60,
                  edge_pad_mode="reflect", boundary_penalty=0.0, edge_width=5)
CNN_WEIGHT = 0.35


def true_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.sum() == 0:
        return []
    diffs = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def classify_failure(truth: List[Tuple[int, int]], pred: List[Tuple[int, int]]) -> str:
    """Determine the failure mode by comparing truth vs prediction segments."""
    if not truth and not pred:
        return "no_anomaly_both"
    if not truth and pred:
        return "phantom_anomaly"  # truth has none, we predicted some
    if truth and not pred:
        return "missing_all"
    if len(truth) == 1 and len(pred) == 1:
        ts, te = truth[0]; ps, pe = pred[0]
        i_start, i_end = max(ts, ps), min(te, pe)
        if i_end < i_start:
            return "no_overlap"
        intersection = i_end - i_start + 1
        truth_len = te - ts + 1
        pred_len = pe - ps + 1
        if abs(ps - ts) <= 5 and abs(pe - te) <= 5 and pred_len == truth_len:
            return "off_by_few"
        if pred_len < truth_len * 0.7:
            return "too_short"
        if pred_len > truth_len * 1.3:
            return "too_long"
        return "shifted"
    if len(truth) == 1 and len(pred) > 1:
        return "split_segment"
    if len(truth) > 1 and len(pred) == 1:
        return "merged_segments"
    if len(pred) > len(truth):
        return "extra_segments"
    return "other"


def ensemble_score(channels, category):
    cw_s, cnn_s, if_s, g_s, local_s, pw_s = (channels[k] for k in ["cw", "cnn", "if", "g", "local", "pw"])
    if category == "constant_train":
        return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * if_s + CNN_WEIGHT * cnn_s
    if category == "disjoint":
        return 0.35 * cw_s + 0.30 * g_s + (0.35 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
    if category == "partial_overlap":
        return 0.35 * cw_s + 0.35 * pw_s + (0.30 - CNN_WEIGHT) * local_s + CNN_WEIGHT * cnn_s
    if category == "test_within_train":
        return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * pw_s + CNN_WEIGHT * cnn_s
    return np.zeros(len(cw_s))


def compute_channels(train_x, train_y, test_x, cw, cnn_models, info, metric_type):
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type, info=info))
    cnn_s = normalize_scores(ensemble_cnn_score(cnn_models, test_x))
    if_s = normalize_scores(isolation_forest_test(test_x, train_y))
    g_s = normalize_scores(global_distance_score(train_x, test_x))
    local_s = normalize_scores(online_ensemble(test_x, window=15))
    pw_s = normalize_scores(per_window_rf_score(train_x, train_y, test_x)) if train_y.sum() > 0 \
        else np.zeros(len(test_x))
    return {"cw": cw_s, "cnn": cnn_s, "if": if_s, "g": g_s, "local": local_s, "pw": pw_s}


def ascii_sparkline(values: np.ndarray, width: int = 80) -> str:
    if len(values) == 0:
        return ""
    chars = " ▁▂▃▄▅▆▇█"
    v = np.asarray(values, dtype=float)
    if v.max() == v.min():
        return chars[4] * min(width, len(v))
    norm = (v - v.min()) / (v.max() - v.min())
    if len(v) <= width:
        idx = list(range(len(v)))
    else:
        idx = np.linspace(0, len(v) - 1, width).astype(int)
    return "".join(chars[min(8, int(round(norm[i] * 8)))] for i in idx)


def mark_segments(n: int, seglist: List[Tuple[int, int]], width: int = 80) -> str:
    mask = np.zeros(n, dtype=int)
    for s, e in seglist:
        mask[s : e + 1] = 1
    chars = ".X"
    if n <= width:
        idx = list(range(n))
    else:
        idx = np.linspace(0, n - 1, width).astype(int)
    return "".join(chars[mask[i]] for i in idx)


def investigate():
    print(">>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=42)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training metadata CW + 3-seed CNN ensemble…")
    cw = build_metadata_hybrid(train_pool, mode="intervals")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))

    print(">>> Evaluating all 100 holdout windows with v32 edge fix…")
    failures = []
    f1_by_metric = defaultdict(list)
    failure_modes = Counter()
    failures_by_metric = defaultdict(list)

    for wdir in holdout:
        w = load_window(wdir)
        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = time_split(w.train_x, w.train_y, frac=0.70)
        ratio = float(sub_te_y.mean()) if len(sub_te_y) else 0.0
        category = categorize_window(sub_tr_x, sub_te_x)
        chans = compute_channels(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models, w.info,
                                 w.metric_type)
        score = ensemble_score(chans, category)
        k = int(round(len(sub_te_x) * ratio))
        pred = predict_segments_v32(score, k, **SEG_KWARGS)
        f1 = point_f1(sub_te_y, pred)

        truth_segs = true_segments(sub_te_y)
        pred_segs = true_segments(pred)
        mode = classify_failure(truth_segs, pred_segs)

        f1_by_metric[w.metric_type].append(f1)
        if f1 < 0.9:
            failure_modes[mode] += 1
            failures_by_metric[w.metric_type].append({
                "wid": w.wid, "f1": f1, "mode": mode, "category": category,
                "n": len(sub_te_x), "ratio": ratio, "k": k,
                "truth_segs": truth_segs, "pred_segs": pred_segs,
                "sub_te_x": sub_te_x, "sub_te_y": sub_te_y, "pred": pred,
                "score": score,
            })

    # Summary by metric type
    print(f"\n>>> Per-metric F1 summary (after v32 reflect fix):")
    for mt in sorted(f1_by_metric):
        vals = f1_by_metric[mt]
        n_lt9 = sum(1 for f in vals if f < 0.9)
        print(f"  {mt:<28}  n={len(vals)}  mean_F1={np.mean(vals):.4f}  "
              f"n(F1<0.9)={n_lt9}")

    print(f"\n>>> Failure mode counts (F1 < 0.9):")
    for mode, cnt in failure_modes.most_common():
        print(f"  {mode:<20}  {cnt}")

    # Detailed look at the worst failures
    print(f"\n>>> Detailed analysis of failure windows by metric type:")
    for mt, fails in sorted(failures_by_metric.items()):
        if not fails:
            continue
        print(f"\n  === {mt} ({len(fails)} failures) ===")
        for fail in sorted(fails, key=lambda x: x["f1"]):
            r = fail
            print(f"\n  wid={r['wid']}  F1={r['f1']:.3f}  mode={r['mode']}  "
                  f"cat={r['category']}  ratio={r['ratio']:.3f}  k={r['k']}")
            print(f"    truth: {r['truth_segs']}    pred: {r['pred_segs']}")
            print(f"    series: {ascii_sparkline(r['sub_te_x'])}")
            print(f"    score : {ascii_sparkline(r['score'])}")
            print(f"    truth : {mark_segments(r['n'], r['truth_segs'])}")
            print(f"    pred  : {mark_segments(r['n'], r['pred_segs'])}")

    # Clean before save
    for mt, fails in failures_by_metric.items():
        for r in fails:
            for key in ("sub_te_x", "sub_te_y", "pred", "score"):
                r.pop(key, None)
    save_report({
        "f1_by_metric": {k: list(v) for k, v in f1_by_metric.items()},
        "failure_modes": dict(failure_modes),
        "failures_by_metric": dict(failures_by_metric),
    }, "v33_investigate_all")


if __name__ == "__main__":
    investigate()
