"""
author v17 — Stacking meta-classifier on channel scores.

The current pipeline blends channels (cw, cnn, pw, if, online, global_distance)
with hardcoded weights per category: e.g. partial_overlap = 0.35 cw + 0.35 pw +
(0.30 - CNN_WEIGHT) local + CNN_WEIGHT cnn. These weights were chosen by ablation,
not learned.

Hypothesis: a logistic regression trained on holdout-window scores (with the
true labels as target) will find better blending weights than our hand-tuned
constants. Per-category models can capture different optimal weights for each
window category.

Procedure:
  1. Train the full model stack (RF hybrid + 3 CNNs + 5 LGBM for QPS).
  2. On the holdout (100 windows), compute all channel scores per point.
  3. Train 4 LogisticRegression meta-classifiers, one per category, using true
     labels as target. Use L2 regularization to avoid overfitting.
  4. At inference, use the appropriate meta-classifier per category.

Risk: meta-classifiers overfit to the 100-window holdout. Mitigation: very
strong L2 regularization, plus train each meta-classifier on cross-validation
splits of the holdout to ensure it sees both label patterns.

Run:  uv run python v17_stacking.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

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
    evaluate,
    load_window,
    point_f1,
    print_summary,
    save_report,
    stratified_holdout,
    time_split,
)

SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)
CATEGORIES = ("constant_train", "disjoint", "partial_overlap", "test_within_train")


def build_rf_hybrid(window_dirs):
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


def compute_channels(train_x, train_y, test_x, cw, cnn_models,
                     metric_type) -> Dict[str, np.ndarray]:
    """Return a dict of all normalized channel scores for the test window."""
    n = len(test_x)
    chans = {
        "cw":   normalize_scores(cw.predict_proba(test_x, metric_type=metric_type)),
        "cnn":  normalize_scores(ensemble_cnn_score(cnn_models, test_x)),
        "if":   normalize_scores(isolation_forest_test(test_x, train_y)),
        "g":    normalize_scores(global_distance_score(train_x, test_x)),
        "local": normalize_scores(online_ensemble(test_x, window=15)),
    }
    if train_y.sum() > 0:
        chans["pw"] = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
    else:
        chans["pw"] = np.zeros(n)
    return chans


def channels_to_features(chans: Dict[str, np.ndarray], category: str) -> np.ndarray:
    """Build a (n_points, n_features) array for the meta-classifier."""
    keys = ["cw", "cnn", "if", "g", "local", "pw"]
    X = np.column_stack([chans[k] for k in keys])
    return X


def baseline_blend(chans: Dict[str, np.ndarray], category: str) -> np.ndarray:
    """The current hand-tuned v11 all_v12 blend, for comparison."""
    cw, cnn, if_, g, local, pw = (chans[k] for k in ["cw", "cnn", "if", "g", "local", "pw"])
    CNN_WEIGHT = 0.35
    if category == "constant_train":
        return (0.50 - CNN_WEIGHT) * cw + 0.50 * if_ + CNN_WEIGHT * cnn
    if category == "disjoint":
        return 0.35 * cw + 0.30 * g + (0.35 - CNN_WEIGHT) * if_ + CNN_WEIGHT * cnn
    if category == "partial_overlap":
        return 0.35 * cw + 0.35 * pw + (0.30 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn
    if category == "test_within_train":
        return (0.50 - CNN_WEIGHT) * cw + 0.50 * pw + CNN_WEIGHT * cnn
    return np.zeros_like(cw)


def fit_stackers(channel_data: List[dict]):
    """Train one LogisticRegression per category on cached holdout channels.

    `channel_data` is a list of dicts with keys: category, chans, y_true."""
    from sklearn.linear_model import LogisticRegression

    stackers = {}
    for cat in CATEGORIES:
        rows = [r for r in channel_data if r["category"] == cat]
        if not rows:
            print(f"  {cat}: no holdout windows; using None (will fall back to baseline blend)")
            stackers[cat] = None
            continue
        Xs = np.vstack([channels_to_features(r["chans"], cat) for r in rows])
        ys = np.concatenate([r["y_true"].astype(int) for r in rows])
        if ys.sum() < 5 or (ys == 0).sum() < 5:
            print(f"  {cat}: too few labels (pos={ys.sum()}, neg={(ys == 0).sum()}); fallback")
            stackers[cat] = None
            continue
        clf = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced")
        clf.fit(Xs, ys)
        stackers[cat] = clf
        coefs = dict(zip(["cw", "cnn", "if", "g", "local", "pw"], clf.coef_[0]))
        print(f"  {cat}: trained on {len(rows)} windows, pos_rate={ys.mean():.3f}, coefs={coefs}")
    return stackers


def stacker_blend(chans, category, stackers):
    clf = stackers.get(category)
    if clf is None:
        return baseline_blend(chans, category)
    X = channels_to_features(chans, category)
    proba = clf.predict_proba(X)
    return proba[:, 1] if proba.shape[1] > 1 else np.zeros(X.shape[0])


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

    # Two-fold cross-validation on the holdout:
    # - fold A: train stackers on first 50 windows, eval on last 50
    # - fold B: vice versa
    # Reports the mean of both folds as the unbiased validation F1.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(holdout))
    fold_a = [holdout[i] for i in idx[:50]]
    fold_b = [holdout[i] for i in idx[50:]]

    print("\n>>> Caching channels for both folds…")
    t0 = time.time()
    cache = {}
    for w_dirs, fold_name in [(fold_a, "A"), (fold_b, "B")]:
        cache[fold_name] = []
        for wdir in w_dirs:
            w = load_window(wdir)
            sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = time_split(w.train_x, w.train_y, frac=0.70)
            ratio = float(sub_te_y.mean()) if len(sub_te_y) else 0.0
            cat = categorize_window(sub_tr_x, sub_te_x)
            chans = compute_channels(sub_tr_x, sub_tr_y, sub_te_x, rf_hybrid, cnn_models,
                                     w.metric_type)
            cache[fold_name].append({
                "wid": w.wid, "category": cat,
                "chans": chans, "y_true": sub_te_y, "ratio": ratio, "n": len(sub_te_x),
            })
    print(f"    cache built in {time.time() - t0:.1f}s")

    print("\n>>> Eval BASELINE blend (hand-tuned weights)…")
    baseline_f1s = []
    for fold_name in ["A", "B"]:
        for r in cache[fold_name]:
            scores = baseline_blend(r["chans"], r["category"])
            k = int(round(r["n"] * r["ratio"]))
            pred = predict_segments(scores, k, **SEG_KWARGS)
            baseline_f1s.append(point_f1(r["y_true"], pred))
    baseline_f1 = float(np.mean(baseline_f1s))
    print(f"    baseline F1 = {baseline_f1:.4f}")

    print("\n>>> Train stackers on fold A, eval on fold B…")
    stackers_A = fit_stackers(cache["A"])
    f1s_b = []
    for r in cache["B"]:
        scores = stacker_blend(r["chans"], r["category"], stackers_A)
        k = int(round(r["n"] * r["ratio"]))
        pred = predict_segments(scores, k, **SEG_KWARGS)
        f1s_b.append(point_f1(r["y_true"], pred))
    f1_b = float(np.mean(f1s_b))
    print(f"    fold B F1 = {f1_b:.4f}")

    print("\n>>> Train stackers on fold B, eval on fold A…")
    stackers_B = fit_stackers(cache["B"])
    f1s_a = []
    for r in cache["A"]:
        scores = stacker_blend(r["chans"], r["category"], stackers_B)
        k = int(round(r["n"] * r["ratio"]))
        pred = predict_segments(scores, k, **SEG_KWARGS)
        f1s_a.append(point_f1(r["y_true"], pred))
    f1_a = float(np.mean(f1s_a))
    print(f"    fold A F1 = {f1_a:.4f}")

    stacker_f1 = (f1_a + f1_b) / 2
    delta = stacker_f1 - baseline_f1
    print(f"\n  Stacker mean F1 (held-out per fold) = {stacker_f1:.4f}")
    print(f"  Baseline blend F1                   = {baseline_f1:.4f}")
    print(f"  Δ (stacker − baseline) = {delta:+.4f}")

    report = {
        "baseline_f1": baseline_f1,
        "stacker_fold_a_f1": f1_a,
        "stacker_fold_b_f1": f1_b,
        "stacker_mean_f1": stacker_f1,
        "delta_stacker_vs_baseline": delta,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v17_stacking")
    return report, rf_hybrid, cnn_models, cache


def generate_submission(rf_hybrid, cnn_models, all_holdout_cache,
                        output: Path = Path("submission_stacking.json")) -> Path:
    """Train stackers on ALL holdout windows, generate test predictions."""
    print("\n>>> Training stackers on FULL holdout (no cv splits)…")
    full_data = all_holdout_cache["A"] + all_holdout_cache["B"]
    stackers = fit_stackers(full_data)

    print(">>> Re-training stack on ALL 1000 windows for inference models…")
    t0 = time.time()
    rf_full = build_rf_hybrid(all_window_dirs())
    print(f"    rf full fit {time.time() - t0:.1f}s")

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
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        cat = categorize_window(w.train_x, w.test_x)
        chans = compute_channels(w.train_x, w.train_y, w.test_x, rf_full, cnn_full,
                                 w.metric_type)
        scores = stacker_blend(chans, cat, stackers)
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
    return output


if __name__ == "__main__":
    rep, rf, cnn, cache = run_validation()
    if rep["delta_stacker_vs_baseline"] > 0.003:
        print(f"\nStacker beats hand-tuned blend by {rep['delta_stacker_vs_baseline']:+.4f}; "
              "generating submission.")
        generate_submission(rf, cnn, cache)
    else:
        print(f"\nStacker did not meaningfully beat baseline "
              f"(Δ = {rep['delta_stacker_vs_baseline']:+.4f}); submission NOT generated.")
