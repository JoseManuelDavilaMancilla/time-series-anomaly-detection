"""
author v35 — Re-investigate failure modes AFTER v34's post-processing fixes.

v34 closed several failure modes but the remaining failures (~6 windows still
F1<0.9) may reveal new patterns now that the dominant right-shift bias is
corrected. Same methodology: classify failures, find the most common new mode,
propose a targeted fix.

Run:  uv run python v35_investigate_after_v34.py
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
from v22_metadata_features import build_metadata_hybrid, scores_v14_with_meta
from v34_postproc_fixes import predict_segments_v34, SEG_KWARGS
from v33_investigate_all import (
    classify_failure, true_segments, compute_channels, ensemble_score,
    ascii_sparkline, mark_segments,
)
from validation import (
    all_window_dirs, load_window, point_f1, save_report, stratified_holdout, time_split,
)


def investigate():
    print(">>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=42)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training metadata CW + 3-seed CNN…")
    cw = build_metadata_hybrid(train_pool, mode="intervals")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))

    print(">>> Evaluating with v34 (left_shift max=2 + merge=3)…")
    failures_by_metric = defaultdict(list)
    failure_modes = Counter()
    f1_by_metric = defaultdict(list)

    for wdir in holdout:
        w = load_window(wdir)
        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = time_split(w.train_x, w.train_y, frac=0.70)
        ratio = float(sub_te_y.mean()) if len(sub_te_y) else 0.0
        category = categorize_window(sub_tr_x, sub_te_x)
        chans = compute_channels(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models, w.info,
                                 w.metric_type)
        score = ensemble_score(chans, category)
        k = int(round(len(sub_te_x) * ratio))
        pred = predict_segments_v34(score, k, do_left_shift=True, do_merge=True,
                                    max_shift=2, merge_gap=3, relative_thresh=0.95,
                                    **SEG_KWARGS)
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
                "sub_te_x": sub_te_x, "score": score,
            })

    print(f"\n>>> Per-metric F1 after v34:")
    for mt in sorted(f1_by_metric):
        vals = f1_by_metric[mt]
        n_lt9 = sum(1 for f in vals if f < 0.9)
        print(f"  {mt:<28}  n={len(vals)}  mean_F1={np.mean(vals):.4f}  n(F1<0.9)={n_lt9}")

    print(f"\n>>> Failure mode counts (F1<0.9):")
    for mode, cnt in failure_modes.most_common():
        print(f"  {mode:<20}  {cnt}")

    print(f"\n>>> Detailed remaining failures:")
    for mt, fails in sorted(failures_by_metric.items()):
        if not fails:
            continue
        print(f"\n  === {mt} ({len(fails)} failures) ===")
        for r in sorted(fails, key=lambda x: x["f1"]):
            print(f"\n  wid={r['wid']}  F1={r['f1']:.3f}  mode={r['mode']}  "
                  f"cat={r['category']}  ratio={r['ratio']:.3f}  k={r['k']}")
            print(f"    truth: {r['truth_segs']}    pred: {r['pred_segs']}")
            print(f"    series: {ascii_sparkline(r['sub_te_x'])}")
            print(f"    score : {ascii_sparkline(r['score'])}")
            print(f"    truth : {mark_segments(r['n'], r['truth_segs'])}")
            print(f"    pred  : {mark_segments(r['n'], r['pred_segs'])}")

    for mt, fails in failures_by_metric.items():
        for r in fails:
            for key in ("sub_te_x", "score"):
                r.pop(key, None)
    save_report({"f1_by_metric": {k: list(v) for k, v in f1_by_metric.items()},
                 "failure_modes": dict(failure_modes),
                 "failures_by_metric": dict(failures_by_metric)},
                "v35_investigate_after_v34")


if __name__ == "__main__":
    investigate()
