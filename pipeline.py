"""
author v31 — Investigation: why does ResourceUtilizationRate keep failing?

Every architectural experiment shows the same per-metric pattern:
ResourceUtilizationRate sits at ~0.95 F1 and is the *first* metric to drop
when we change anything. It has 17 windows in our holdout.

Hypothesis: there's a specific anomaly pattern in ResourceUtil that none of
our channels (CW point-wise scores + CNN local context + IF on test) can
detect reliably. If we can find it, we can build a targeted scorer just
for ResourceUtil and route it.

This script:
  1. Builds the v22 best pipeline
  2. Runs it on holdout ResourceUtil windows
  3. Identifies the bottom-K windows by per-window F1
  4. For each, prints the time series, the true segments, our predictions,
     and the channel-by-channel scores at the truth points
  5. Looks for a structural pattern (e.g., "we miss long segments",
     "we miss low-magnitude dips", "we miss boundary anomalies")

This is investigation, not experimentation — output is descriptive analysis
that will inform what to build next.

Run:  uv run python v31_investigate.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

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
    predict_segments,
)
from v6_cnn import SPECIALIZED, build_training_pool as v6_pool
from v7_cnn_ensemble import CNN_SEEDS, ensemble_cnn_score, fit_cnn_with_seed
from v22_metadata_features import build_metadata_hybrid
from validation import (
    all_window_dirs,
    load_window,
    point_f1,
    save_report,
    stratified_holdout,
    time_split,
)

SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)
CNN_WEIGHT = 0.35
TARGET_METRIC = "ResourceUtilizationRate"


def true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.sum() == 0:
        return []
    diffs = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def compute_channels(train_x, train_y, test_x, cw, cnn_models, info, metric_type):
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type, info=info))
    cnn_s = normalize_scores(ensemble_cnn_score(cnn_models, test_x))
    if_s = normalize_scores(isolation_forest_test(test_x, train_y))
    g_s = normalize_scores(global_distance_score(train_x, test_x))
    local_s = normalize_scores(online_ensemble(test_x, window=15))
    if train_y.sum() > 0:
        pw_s = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
    else:
        pw_s = np.zeros(len(test_x))
    return {"cw": cw_s, "cnn": cnn_s, "if": if_s, "g": g_s, "local": local_s, "pw": pw_s}


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


def ascii_sparkline(values: np.ndarray, width: int = 80) -> str:
    if len(values) == 0:
        return ""
    chars = " ▁▂▃▄▅▆▇█"
    v = np.asarray(values, dtype=float)
    if v.max() == v.min():
        return chars[4] * min(width, len(v))
    norm = (v - v.min()) / (v.max() - v.min())
    # Resample to width
    if len(v) <= width:
        idx = list(range(len(v)))
    else:
        idx = np.linspace(0, len(v) - 1, width).astype(int)
    return "".join(chars[min(8, int(round(norm[i] * 8)))] for i in idx)


def mark_segments(n: int, seglist: list[tuple[int, int]], width: int = 80) -> str:
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

    print(f"\n>>> Filtering holdout to {TARGET_METRIC} windows…")
    target_dirs = []
    for wdir in holdout:
        info = json.loads((wdir / "info.json").read_text())
        if info.get("metric_type") == TARGET_METRIC:
            target_dirs.append(wdir)
    print(f"    found {len(target_dirs)} {TARGET_METRIC} windows in holdout")

    print(f"\n>>> Evaluating each window and gathering channel statistics…")
    per_window = []
    for wdir in target_dirs:
        w = load_window(wdir)
        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = time_split(w.train_x, w.train_y, frac=0.70)
        ratio = float(sub_te_y.mean()) if len(sub_te_y) else 0.0
        category = categorize_window(sub_tr_x, sub_te_x)
        chans = compute_channels(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models, w.info,
                                 w.metric_type)
        score = ensemble_score(chans, category)
        k = int(round(len(sub_te_x) * ratio))
        pred = predict_segments(score, k, **SEG_KWARGS)
        f1 = point_f1(sub_te_y, pred)

        truth_segs = true_segments(sub_te_y)
        pred_segs = true_segments(pred)
        # Channel-mean over truth positions
        truth_mask = sub_te_y.astype(bool)
        chan_means_at_truth = {k: float(chans[k][truth_mask].mean()) if truth_mask.any() else 0.0
                               for k in chans}
        chan_means_at_normal = {k: float(chans[k][~truth_mask].mean()) for k in chans}

        per_window.append({
            "wid": w.wid, "wdir": wdir, "f1": f1, "category": category,
            "n": len(sub_te_x), "ratio": ratio, "k": k,
            "n_truth_segs": len(truth_segs), "n_pred_segs": len(pred_segs),
            "truth_seg_lens": [e - s + 1 for s, e in truth_segs],
            "pred_seg_lens": [e - s + 1 for s, e in pred_segs],
            "truth_n_anomalies": int(sub_te_y.sum()),
            "pred_n_anomalies": int(pred.sum()),
            "chan_means_at_truth": chan_means_at_truth,
            "chan_means_at_normal": chan_means_at_normal,
            "sub_te_x": sub_te_x,
            "sub_te_y": sub_te_y,
            "pred": pred,
            "score": score,
            "truth_segs": truth_segs,
            "pred_segs": pred_segs,
        })

    per_window.sort(key=lambda r: r["f1"])
    print(f"\n>>> {TARGET_METRIC} window F1 distribution (sorted ascending):")
    for r in per_window:
        print(f"  wid={r['wid']}  cat={r['category']:<18}  "
              f"F1={r['f1']:.3f}  ratio={r['ratio']:.3f}  k={r['k']:>3}  "
              f"truth_segs={r['n_truth_segs']}({r['truth_seg_lens']})  "
              f"pred_segs={r['n_pred_segs']}({r['pred_seg_lens']})")

    print(f"\n>>> Bottom 5 windows (worst F1):")
    for r in per_window[:5]:
        print(f"\n  --- wid={r['wid']}  F1={r['f1']:.3f}  category={r['category']} ---")
        print(f"      ratio={r['ratio']:.3f}  k={r['k']}  truth_n={r['truth_n_anomalies']}  "
              f"pred_n={r['pred_n_anomalies']}")
        print(f"      truth_segs: {r['truth_segs']}")
        print(f"      pred_segs:  {r['pred_segs']}")
        print(f"      chan means at TRUTH points:  "
              + ", ".join(f"{k}={v:.3f}" for k, v in r['chan_means_at_truth'].items()))
        print(f"      chan means at NORMAL points: "
              + ", ".join(f"{k}={v:.3f}" for k, v in r['chan_means_at_normal'].items()))
        # ASCII visualizations (80-col)
        print(f"      time series : {ascii_sparkline(r['sub_te_x'])}")
        print(f"      score       : {ascii_sparkline(r['score'])}")
        print(f"      truth       : {mark_segments(r['n'], r['truth_segs'])}")
        print(f"      prediction  : {mark_segments(r['n'], r['pred_segs'])}")

    # Aggregate analysis
    print(f"\n>>> Aggregate over {len(per_window)} {TARGET_METRIC} windows:")
    truth_lens = [l for r in per_window for l in r["truth_seg_lens"]]
    pred_lens = [l for r in per_window for l in r["pred_seg_lens"]]
    print(f"  truth seg lengths: n={len(truth_lens)}, mean={np.mean(truth_lens) if truth_lens else 0:.1f}, "
          f"median={np.median(truth_lens) if truth_lens else 0:.1f}, "
          f"max={max(truth_lens) if truth_lens else 0}, min={min(truth_lens) if truth_lens else 0}")
    print(f"  pred seg lengths:  n={len(pred_lens)}, mean={np.mean(pred_lens) if pred_lens else 0:.1f}, "
          f"median={np.median(pred_lens) if pred_lens else 0:.1f}, "
          f"max={max(pred_lens) if pred_lens else 0}, min={min(pred_lens) if pred_lens else 0}")

    # Channel separation: which channel best separates truth from normal?
    print(f"\n  Channel separation (mean at truth − mean at normal):")
    for k in ["cw", "cnn", "if", "g", "local", "pw"]:
        diff = np.mean([r["chan_means_at_truth"][k] - r["chan_means_at_normal"][k]
                        for r in per_window if r["truth_n_anomalies"] > 0])
        print(f"    {k:<6}  Δ={diff:+.3f}")

    # Strip out heavy numpy arrays before saving JSON
    for r in per_window:
        for key in ("sub_te_x", "sub_te_y", "pred", "score"):
            r.pop(key, None)
        # Also drop wdir Path (not JSON-serializable)
        r.pop("wdir", None)
    save_report({"target": TARGET_METRIC, "per_window": per_window}, "v31_investigate")


if __name__ == "__main__":
    investigate()
