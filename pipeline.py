"""
author v20 — test-time adaptation via train-mode BatchNorm at inference (radical).

After 4 consecutive failures of new channels / model classes, the validation
ceiling at exactly 0.9639 says: the architecture isn't the bottleneck.
The remaining gap to the LB is distribution shift between training and test.

This experiment tests the simplest TTA approach: switch the CNN ensemble's
BatchNorm layers from eval-mode (training-time running statistics) to
train-mode (batch statistics computed on the current test window) at
inference. If the test distribution differs from the training distribution,
batch stats from the test window will normalize the activations to a more
useful range, and the rest of the network's weights — which were trained
to operate on well-normalized activations — will see input closer to their
training distribution.

Trade-offs:
  + Almost free: one boolean flip per BN layer, no retraining, no new params.
  + Targets distribution shift directly (the suspected bottleneck).
  − Test batch is ~300 points per window, which may give noisy BN stats.
  − Could hurt if the training population stats were already well-calibrated.

Sweep: compare three CNN inference modes
  A. eval-mode (current baseline, running statistics)
  B. train-mode BN (batch statistics from the test window)
  C. blended: 50/50 average of A and B per-point scores

Run:  uv run python v20_tta.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message="X does not have valid feature names")

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
from v6_cnn import (
    DEVICE, SPECIALIZED, TinyCNN,
    build_contexts, build_training_pool as v6_pool,
)
from v7_cnn_ensemble import CNN_SEEDS, fit_cnn_with_seed
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)
CNN_WEIGHT = 0.35


def cnn_score_with_bn_mode(model: TinyCNN, test_x: np.ndarray, *,
                           use_batch_stats: bool) -> np.ndarray:
    """Score with the model. If use_batch_stats=True, BN uses the test
    batch's statistics (train mode for BN only)."""
    # We always want gradients off
    model.eval()  # default: gradients off, dropout off, BN uses running stats
    if use_batch_stats:
        # Flip only BN layers into train mode so they compute batch stats.
        # We do NOT call backward, so running_mean/running_var won't be updated
        # in a way that contaminates other windows — but to be safe, after
        # this call we set them back to eval.
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.train()
    X = build_contexts(test_x)
    with torch.no_grad():
        xb = torch.from_numpy(X).to(DEVICE)
        logits = model(xb)
        proba = torch.sigmoid(logits).cpu().numpy()
    # Always restore eval mode at the end
    model.eval()
    return proba


def ensemble_score_with_bn(models: List, test_x: np.ndarray, *,
                           mode: str) -> np.ndarray:
    """mode ∈ {'eval', 'batch', 'blend'}"""
    if mode == "eval":
        eval_scores = np.stack([cnn_score_with_bn_mode(m, test_x, use_batch_stats=False)
                                for m in models])
        return eval_scores.mean(axis=0)
    if mode == "batch":
        batch_scores = np.stack([cnn_score_with_bn_mode(m, test_x, use_batch_stats=True)
                                 for m in models])
        return batch_scores.mean(axis=0)
    # blend
    e = np.stack([cnn_score_with_bn_mode(m, test_x, use_batch_stats=False)
                  for m in models]).mean(axis=0)
    b = np.stack([cnn_score_with_bn_mode(m, test_x, use_batch_stats=True)
                  for m in models]).mean(axis=0)
    return 0.5 * e + 0.5 * b


def scores_v14_with_cnn(train_x, train_y, test_x, cw, cnn_models, metric_type,
                        cnn_mode: str) -> np.ndarray:
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
    cnn_s = normalize_scores(ensemble_score_with_bn(cnn_models, test_x, mode=cnn_mode))

    if category == "constant_train":
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * if_s + CNN_WEIGHT * cnn_s
    if category == "disjoint":
        g_s = normalize_scores(global_distance_score(train_x, test_x))
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return 0.35 * cw_s + 0.30 * g_s + (0.35 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
    if category == "partial_overlap":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        return 0.35 * cw_s + 0.35 * pw + (0.30 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn_s
    if category == "test_within_train":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * pw + CNN_WEIGHT * cnn_s
    return np.zeros(len(test_x))


def build_rf_hybrid(window_dirs):
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


def run_validation(seed: int = 42) -> dict:
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

    def predictor(cnn_mode: str):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_v14_with_cnn(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                         info.get("metric_type", "ALL"), cnn_mode)
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    results = {}
    for mode in ("eval", "batch", "blend"):
        print(f"\n>>> Eval: CNN inference mode = {mode}…")
        rep = evaluate(predictor(mode), holdout)
        print_summary(rep, name=f"CNN-{mode}")
        results[mode] = rep

    print("\n──────  summary ──────")
    base = results["eval"]["overall_f1"]
    for mode, rep in results.items():
        print(f"  cnn_mode={mode:<6}  F1={rep['overall_f1']:.4f}  Δ_vs_eval={rep['overall_f1'] - base:+.4f}")

    # Pick winner
    winner_mode = max(results, key=lambda m: results[m]["overall_f1"])
    winner_f1 = results[winner_mode]["overall_f1"]
    print(f"\n  Winner: cnn_mode={winner_mode}  F1={winner_f1:.4f}")

    report = {
        "by_mode": {m: r["overall_f1"] for m, r in results.items()},
        "winner_mode": winner_mode,
        "delta_winner_vs_eval": winner_f1 - base,
        "per_metric_by_mode": {m: r["by_metric_type"] for m, r in results.items()},
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v20_tta")
    return report, cw, cnn_models


def generate_submission(cnn_mode: str, cw, cnn_models,
                        output: Path = Path("submission_tta.json")) -> Path:
    print(f"\n>>> Re-training all models on ALL 1000 windows (cnn_mode={cnn_mode})…")
    t0 = time.time()
    cw_full = build_rf_hybrid(all_window_dirs())
    print(f"    rf fit {time.time() - t0:.1f}s")

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
        scores = scores_v14_with_cnn(w.train_x, w.train_y, w.test_x, cw_full, cnn_full,
                                     w.metric_type, cnn_mode)
        k = int(round(len(w.test_x) * ratio))
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
    rep, cw, cnn_models = run_validation()
    if rep["winner_mode"] != "eval" and rep["delta_winner_vs_eval"] > 0.002:
        print(f"\nCNN mode {rep['winner_mode']} beats eval by {rep['delta_winner_vs_eval']:+.4f}; "
              "generating submission.")
        generate_submission(rep["winner_mode"], cw, cnn_models)
    else:
        print(f"\nTTA did not meaningfully help "
              f"(best Δ = {rep['delta_winner_vs_eval']:+.4f}); submission NOT generated.")
