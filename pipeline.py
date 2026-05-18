"""
author v18 — segment-level classifier (radical change).

Hypothesis: every channel we have scores INDIVIDUAL POINTS. The truth is
CONTIGUOUS SEGMENTS (mean length ~17, ~2 segments per window). We've been
working around this with post-processing (smooth + grow). A segment-level
classifier directly learns "what does an anomalous segment look like" by
seeing whole segments as training examples, with segment-level features
that aren't visible point-by-point: length, position, boundary discontinuity,
context delta, channel-score statistics across the segment.

Pipeline:
  1. Train the existing v14 score channels (CW, CNN, IF, online, global).
  2. For each training window, score points with the channels, generate
     ~60–150 candidate (start, end) segments at multiple thr_frac×smooth
     combos, label them by IoU>=0.5 with true anomaly segments.
  3. Extract ~25 features per candidate (geometry + value stats + context
     deltas + channel score stats).
  4. Train RandomForest segment classifier on pooled features.
  5. At inference, generate candidates on test, classify each, greedily
     pick non-overlapping highest-scoring segments until budget=k reached.

Caveats:
- CW score on training portion is biased (CW saw those labels). We accept
  this for the first version; if it works we add OOF predictions in v18b.
- Per-window RF is excluded from the channel set — its train-portion scores
  are pathologically overfit.

Run:  uv run python v18_segment_classifier.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

from shared_lib import (
    CrossWindowModel,
    HybridCrossWindowModel,
    categorize_window,
    global_distance_score,
    isolation_forest_test,
    normalize_scores,
    online_ensemble,
    per_window_rf_score,
    predict_segments,
)
from v6_cnn import build_training_pool as v6_pool, SPECIALIZED
from v7_cnn_ensemble import CNN_SEEDS, ensemble_cnn_score, fit_cnn_with_seed
from validation import (
    all_window_dirs,
    load_window,
    point_f1,
    save_report,
    stratified_holdout,
)

# Candidate generation knobs
THR_FRACS = (0.4, 0.5, 0.6, 0.7, 0.8)
SMOOTHS = (3, 5)
PEAK_QUANTILE = 0.7
MIN_SEG_LEN = 2
MAX_SEG_LEN = 80

# Labeling
IOU_POSITIVE = 0.5

# v14 fixed channel weights (used both to build per-point ensemble score
# and as features into the segment classifier)
CNN_WEIGHT = 0.35


# ─────────────────────────────────────────────
# Candidate generation
# ─────────────────────────────────────────────


def _find_peaks(s: np.ndarray, min_height: float) -> np.ndarray:
    """Local maxima ≥ min_height. Simple comparison, no scipy."""
    n = len(s)
    if n < 3:
        return np.array([], dtype=int)
    higher_than_prev = s[1:-1] > s[:-2]
    higher_than_next = s[1:-1] >= s[2:]
    above = s[1:-1] >= min_height
    return np.where(higher_than_prev & higher_than_next & above)[0] + 1


def generate_candidates(scores: np.ndarray) -> List[Tuple[int, int]]:
    """From a per-point score array, return unique (start, end) candidate segments."""
    candidates: set[Tuple[int, int]] = set()
    n = len(scores)
    for smooth in SMOOTHS:
        s = np.convolve(scores, np.ones(smooth) / smooth, mode="same") if smooth > 1 else scores
        threshold = float(np.quantile(s, PEAK_QUANTILE))
        peaks = _find_peaks(s, threshold)
        for peak in peaks:
            peak_v = s[peak]
            for tf in THR_FRACS:
                thr = tf * peak_v
                L, R = int(peak), int(peak)
                while L > 0 and s[L - 1] >= thr:
                    L -= 1
                while R < n - 1 and s[R + 1] >= thr:
                    R += 1
                seg_len = R - L + 1
                if MIN_SEG_LEN <= seg_len <= MAX_SEG_LEN:
                    candidates.add((L, R))
    return sorted(candidates)


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────


CHANNEL_KEYS = ("cw", "cnn", "if", "g", "local")

FEATURE_NAMES = [
    "length", "length_ratio", "rel_pos_center",
    "val_mean", "val_std", "val_min", "val_max", "val_range",
    "val_z_train", "val_slope_normalized",
    "before_mean", "before_std", "boundary_jump_left",
    "after_mean", "after_std", "boundary_jump_right",
    # Channel score stats
] + [f"{c}_mean" for c in CHANNEL_KEYS] + [f"{c}_max" for c in CHANNEL_KEYS] + [
    "ensemble_mean", "ensemble_max",
]


def _safe_div(num, den):
    return num / (den + 1e-9)


def segment_features(
    seg: Tuple[int, int], series: np.ndarray, channels: Dict[str, np.ndarray],
    train_mean: float, train_std: float,
) -> np.ndarray:
    start, end = seg
    n = len(series)
    seg_x = series[start : end + 1].astype(np.float64)
    seg_len = end - start + 1

    # Geometry
    f = [
        seg_len,
        seg_len / 16.85,
        (start + end) / (2.0 * n),
    ]
    # Value stats
    f += [
        float(seg_x.mean()),
        float(seg_x.std()),
        float(seg_x.min()),
        float(seg_x.max()),
        float(seg_x.max() - seg_x.min()),
        _safe_div(seg_x.mean() - train_mean, train_std),
        _safe_div(seg_x[-1] - seg_x[0], seg_len),
    ]
    # Context
    before = series[max(0, start - 16) : start].astype(np.float64)
    after = series[end + 1 : min(n, end + 17)].astype(np.float64)
    f += [
        float(before.mean()) if len(before) else 0.0,
        float(before.std()) if len(before) > 1 else 0.0,
        float(abs(seg_x[0] - before.mean())) if len(before) else 0.0,
        float(after.mean()) if len(after) else 0.0,
        float(after.std()) if len(after) > 1 else 0.0,
        float(abs(seg_x[-1] - after.mean())) if len(after) else 0.0,
    ]
    # Channel-score stats
    for ck in CHANNEL_KEYS:
        chs = channels[ck][start : end + 1]
        f.append(float(chs.mean()))
    for ck in CHANNEL_KEYS:
        chs = channels[ck][start : end + 1]
        f.append(float(chs.max()))
    # Ensemble stats (using v14-style weights would require knowing category;
    # use simple normalized average across channels as a stable summary)
    ens = np.mean([channels[ck] for ck in CHANNEL_KEYS], axis=0)
    es = ens[start : end + 1]
    f += [float(es.mean()), float(es.max())]

    return np.array(f, dtype=np.float32)


# ─────────────────────────────────────────────
# IoU labeling
# ─────────────────────────────────────────────


def true_segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.sum() == 0:
        return []
    diffs = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def label_by_iou(candidates: List[Tuple[int, int]],
                 mask: np.ndarray, threshold: float = IOU_POSITIVE) -> np.ndarray:
    truth = true_segments_from_mask(mask)
    if not truth:
        return np.zeros(len(candidates), dtype=np.int32)
    labels = np.zeros(len(candidates), dtype=np.int32)
    for i, (cs, ce) in enumerate(candidates):
        best_iou = 0.0
        for (ts, te) in truth:
            i_start = max(cs, ts)
            i_end = min(ce, te)
            if i_end < i_start:
                continue
            inter = i_end - i_start + 1
            union = (ce - cs + 1) + (te - ts + 1) - inter
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
        if best_iou >= threshold:
            labels[i] = 1
    return labels


# ─────────────────────────────────────────────
# Channel score computation (no per-window RF, no v14 ensemble)
# ─────────────────────────────────────────────


def compute_channels(train_x, train_y, test_x, cw, cnn_models, metric_type
                     ) -> Dict[str, np.ndarray]:
    n = len(test_x)
    return {
        "cw":   normalize_scores(cw.predict_proba(test_x, metric_type=metric_type)),
        "cnn":  normalize_scores(ensemble_cnn_score(cnn_models, test_x)),
        "if":   normalize_scores(isolation_forest_test(test_x, train_y)),
        "g":    normalize_scores(global_distance_score(train_x, test_x)),
        "local": normalize_scores(online_ensemble(test_x, window=15)),
    }


def ensemble_score(channels: Dict[str, np.ndarray], category: str) -> np.ndarray:
    """v14-style blend, used to drive candidate generation."""
    cw, cnn, if_, g, local = (channels[k] for k in CHANNEL_KEYS)
    if category == "constant_train":
        return (0.50 - CNN_WEIGHT) * cw + 0.50 * if_ + CNN_WEIGHT * cnn
    if category == "disjoint":
        return 0.35 * cw + 0.30 * g + (0.35 - CNN_WEIGHT) * if_ + CNN_WEIGHT * cnn
    if category == "partial_overlap":
        return 0.35 * cw + (0.30 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn + 0.35 * if_  # use if_ in place of pw
    if category == "test_within_train":
        return (0.50 - CNN_WEIGHT) * cw + 0.50 * if_ + CNN_WEIGHT * cnn  # if_ in place of pw
    return np.zeros(len(cw))


# ─────────────────────────────────────────────
# Training data construction
# ─────────────────────────────────────────────


def build_segment_training_data(
    window_dirs, cw, cnn_models,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each window with anomalies, generate candidates from train portion,
    label via IoU, extract features. Pool. Returns X, y, window_idx."""
    Xs, ys, idxs = [], [], []
    n_pos_total = 0
    n_neg_total = 0
    for wi, wdir in enumerate(window_dirs):
        train_y = np.load(wdir / "train_label.npy")
        if train_y.sum() == 0:
            continue
        train_x = np.load(wdir / "train.npy")
        if len(train_x) < 20:
            continue
        info = json.loads((wdir / "info.json").read_text())
        metric_type = info.get("metric_type", "Unknown")

        chans = compute_channels(train_x, train_y, train_x, cw, cnn_models, metric_type)
        category = categorize_window(train_x, train_x)
        scores = ensemble_score(chans, category)

        candidates = generate_candidates(scores)
        if not candidates:
            continue

        labels = label_by_iou(candidates, train_y)
        train_mean = float(np.mean(train_x))
        train_std = float(np.std(train_x))

        for cand, lab in zip(candidates, labels):
            feat = segment_features(cand, train_x, chans, train_mean, train_std)
            Xs.append(feat); ys.append(int(lab)); idxs.append(wi)
            if lab == 1:
                n_pos_total += 1
            else:
                n_neg_total += 1

    X = np.vstack(Xs)
    y = np.array(ys, dtype=np.int32)
    idx = np.array(idxs, dtype=np.int32)
    print(f"  segment training data: X={X.shape}  pos={n_pos_total}  neg={n_neg_total}  pos_rate={y.mean():.3f}")
    return X, y, idx


def fit_segment_classifier(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
    )
    clf.fit(X, y)
    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    print("  top 10 features:")
    for n, i in importances[:10]:
        print(f"    {n:<25}  {i:.3f}")
    return clf


# ─────────────────────────────────────────────
# Inference: candidate scoring + greedy non-overlapping selection
# ─────────────────────────────────────────────


def greedy_pick(candidates: List[Tuple[int, int]], scores: np.ndarray,
                budget: int, n: int) -> np.ndarray:
    pred = np.zeros(n, dtype=int)
    if budget <= 0:
        return pred
    order = np.argsort(-scores)
    used_total = 0
    used_intervals: List[Tuple[int, int]] = []
    for i in order:
        cs, ce = candidates[i]
        # check overlap with existing intervals
        overlap = False
        for (us, ue) in used_intervals:
            if not (ce < us or cs > ue):
                overlap = True
                break
        if overlap:
            continue
        # Truncate to remaining budget
        seg_len = ce - cs + 1
        if used_total + seg_len > budget:
            ce = cs + (budget - used_total) - 1
            seg_len = ce - cs + 1
            if seg_len <= 0:
                continue
        pred[cs : ce + 1] = 1
        used_intervals.append((cs, ce))
        used_total += seg_len
        if used_total >= budget:
            break
    return pred


def predict_one_window(
    train_x, train_y, test_x, cw, cnn_models, metric_type, seg_classifier,
    test_ratio: float, fallback_to_heuristic: bool = True,
) -> np.ndarray:
    chans = compute_channels(train_x, train_y, test_x, cw, cnn_models, metric_type)
    category = categorize_window(train_x, test_x)
    scores = ensemble_score(chans, category)
    candidates = generate_candidates(scores)

    n = len(test_x)
    k = int(round(n * test_ratio))

    if not candidates:
        if fallback_to_heuristic:
            return predict_segments(scores, k,
                                    smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)
        return np.zeros(n, dtype=int)

    train_mean = float(np.mean(train_x))
    train_std = float(np.std(train_x))
    X = np.vstack([segment_features(c, test_x, chans, train_mean, train_std)
                   for c in candidates])
    proba = seg_classifier.predict_proba(X)
    cand_scores = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(candidates))
    return greedy_pick(candidates, cand_scores, k, n)


# ─────────────────────────────────────────────
# Build CW + CNN (shared with v14)
# ─────────────────────────────────────────────


def build_rf_hybrid(window_dirs):
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────


def run_validation(seed: int = 42) -> dict:
    from validation import time_split

    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training RF hybrid (500/15)…")
    t0 = time.time()
    cw = build_rf_hybrid(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training 3-seed CNN ensemble…")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    print("\n>>> Building segment training data from train_pool…")
    t0 = time.time()
    Xseg, yseg, _ = build_segment_training_data(train_pool, cw, cnn_models)
    print(f"    built in {time.time() - t0:.1f}s")

    print(">>> Training segment classifier (RF)…")
    t0 = time.time()
    seg_clf = fit_segment_classifier(Xseg, yseg)
    print(f"    fit {time.time() - t0:.1f}s")

    print("\n>>> Evaluating on 100-window holdout (70/30 time split)…")
    f1_v18, f1_baseline = [], []
    for wdir in holdout:
        w = load_window(wdir)
        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = time_split(w.train_x, w.train_y, frac=0.70)
        ratio = float(sub_te_y.mean()) if len(sub_te_y) else 0.0

        # v18 prediction
        pred18 = predict_one_window(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                    w.metric_type, seg_clf, ratio)
        f1_v18.append(point_f1(sub_te_y, pred18))

        # v14 baseline prediction (heuristic post-processing)
        chans = compute_channels(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models, w.metric_type)
        category = categorize_window(sub_tr_x, sub_te_x)
        score = ensemble_score(chans, category)
        k = int(round(len(sub_te_x) * ratio))
        pred_base = predict_segments(score, k, smooth=3, thr_frac=0.7,
                                     small_k_cutoff=4, max_seg=60)
        f1_baseline.append(point_f1(sub_te_y, pred_base))

    v18_f1 = float(np.mean(f1_v18))
    base_f1 = float(np.mean(f1_baseline))
    delta = v18_f1 - base_f1
    print(f"\n  v14 baseline (heuristic post-proc) F1 = {base_f1:.4f}")
    print(f"  v18 segment classifier             F1 = {v18_f1:.4f}")
    print(f"  Δ (v18 − v14) = {delta:+.4f}")

    report = {
        "baseline_f1": base_f1,
        "v18_f1": v18_f1,
        "delta": delta,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v18_segment_classifier")
    return report, cw, cnn_models, seg_clf, Xseg, yseg


def generate_submission(output: Path = Path("submission_segment_classifier.json")) -> Path:
    print("\n>>> Training RF hybrid on ALL 1000 windows…")
    t0 = time.time()
    cw = build_rf_hybrid(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training 3-seed CNN ensemble on full data…")
    Xc, yc = v6_pool(all_window_dirs())
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Building segment training data from ALL 1000 windows…")
    Xseg, yseg, _ = build_segment_training_data(all_window_dirs(), cw, cnn_models)
    print(">>> Training segment classifier (RF)…")
    seg_clf = fit_segment_classifier(Xseg, yseg)

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        ratio = float(w.info.get("test set anomaly ratio", 0.0))
        pred = predict_one_window(w.train_x, w.train_y, w.test_x, cw, cnn_models,
                                  w.metric_type, seg_clf, ratio)
        preds[w.wid] = pred.astype(int).tolist()
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
    rep, *_ = run_validation()
    if rep["delta"] > 0.003:
        print(f"\nv18 beats v14 baseline by {rep['delta']:+.4f}; generating submission.")
        generate_submission()
    else:
        print(f"\nv18 did not meaningfully beat v14 baseline "
              f"(Δ = {rep['delta']:+.4f}); submission NOT generated.")
