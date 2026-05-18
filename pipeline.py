"""
author v16 — 3-seed long-context CNN ensemble (context=64 instead of 32).

v7's 3-seed CNN ensemble used 32-point context and was +0.011 over single-seed.
v10 tested CNN_Long (context=64) as a *single-seed* model and lost as part of a
mixed-architecture ensemble. We never tested 3-seed of just CNN_Long.

Hypothesis: longer context (64 vs 32) lets the CNN see longer temporal patterns
(seasonality, ramps over 30-60 samples). The CNN_Long architecture from
`v10_cnn_diverse.py` has kernel sizes (5, 9, 15) tuned for that range.
Three seeds of CNN_Long, predictions averaged, should match or beat the short-
context 3-seed ensemble. If it does, it's a broad architectural change (every
window gets a different score) — the kind that transfers well on the LB per
today's calibration lessons.

Stacks on top of v11 all_v12 (which is at 0.6238 LB).

Run:  uv run python v16_long_context_cnn.py
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from shared_lib import (
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
    DEVICE, EPOCHS, BATCH, LR, WEIGHT_POS, SUBSAMPLE_NEG, SEED, SPECIALIZED,
    _zscore_series,
)
from v7_cnn_ensemble import build_hybrid, CNN_SEEDS
from v10_cnn_diverse import CNN_Long, build_contexts as long_contexts
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
CONTEXT = 64  # long context


def build_long_training_pool(window_dirs) -> tuple[np.ndarray, np.ndarray]:
    """v6_pool but at CONTEXT=64."""
    Xs, ys = [], []
    for wdir in window_dirs:
        train_y = np.load(wdir / "train_label.npy")
        if train_y.sum() == 0:
            continue
        train_x = np.load(wdir / "train.npy")
        if len(train_x) < CONTEXT // 2 + 4:
            continue
        Xs.append(long_contexts(train_x, CONTEXT))
        ys.append(train_y.astype(np.float32))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    if 0 < SUBSAMPLE_NEG < 1.0:
        rng = np.random.default_rng(SEED)
        neg_idx = np.where(y == 0)[0]
        keep_n = int(len(neg_idx) * SUBSAMPLE_NEG)
        keep_idx = rng.choice(neg_idx, size=keep_n, replace=False)
        all_idx = np.concatenate([np.where(y == 1)[0], keep_idx])
        rng.shuffle(all_idx)
        X = X[all_idx]
        y = y[all_idx]
    return X, y


def fit_long_cnn(X: np.ndarray, y: np.ndarray, seed: int = 42) -> nn.Module:
    """Train CNN_Long (context=64) with the given seed."""
    import sys
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNN_Long().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor([WEIGHT_POS], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    for epoch in range(EPOCHS):
        model.train()
        running, n_batches = 0.0, 0
        t0 = time.time()
        for xb, yb in dl:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            running += loss.item(); n_batches += 1
        print(f"      epoch {epoch + 1}/{EPOCHS}  loss={running / n_batches:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        sys.stdout.flush()
    return model


def long_cnn_score(model: nn.Module, test_x: np.ndarray) -> np.ndarray:
    model.eval()
    X = long_contexts(test_x, CONTEXT)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
        proba = torch.sigmoid(logits).cpu().numpy()
    return proba


def long_ensemble_score(models: List, test_x: np.ndarray) -> np.ndarray:
    return np.mean([long_cnn_score(m, test_x) for m in models], axis=0)


def scores_v11(train_x, train_y, test_x, cw, cnn_models, metric_type,
               cnn_score_fn) -> np.ndarray:
    """v11 all_v12 ensemble template parameterized by the CNN score function."""
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
    cnn_s = normalize_scores(cnn_score_fn(cnn_models, test_x))

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
    from shared_lib import CrossWindowModel
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


def run_validation(seed: int = 42) -> dict:
    from v6_cnn import build_training_pool as short_pool
    from v7_cnn_ensemble import fit_cnn_with_seed, ensemble_cnn_score

    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training RF hybrid (500/15)…")
    t0 = time.time()
    rf_hybrid = build_rf_hybrid(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    # Short-context (32) ensemble — baseline (v11 current best)
    print(">>> Training short-context 3-seed CNN ensemble (context=32) — baseline…")
    Xs, ys = short_pool(train_pool)
    short_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        short_models.append(fit_cnn_with_seed(Xs, ys, s))
        print(f"    short cnn seed={s} fit {time.time() - t0:.1f}s")

    # Long-context (64) ensemble — new variant
    print(">>> Training long-context 3-seed CNN ensemble (context=64)…")
    Xl, yl = build_long_training_pool(train_pool)
    print(f"    long pool X={Xl.shape}")
    long_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        long_models.append(fit_long_cnn(Xl, yl, seed=s))
        print(f"    long cnn seed={s} fit {time.time() - t0:.1f}s")

    def predictor(cnn_models, cnn_fn):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_v11(sub_tr_x, sub_tr_y, sub_te_x, rf_hybrid, cnn_models,
                                info.get("metric_type", "ALL"), cnn_fn)
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    print("\n>>> Eval: short-context 3-seed CNN (v11 baseline)…")
    rep_short = evaluate(predictor(short_models, ensemble_cnn_score), holdout)
    print_summary(rep_short, name="short-context CNN")

    print(">>> Eval: long-context 3-seed CNN (v16)…")
    rep_long = evaluate(predictor(long_models, long_ensemble_score), holdout)
    print_summary(rep_long, name="long-context CNN")

    delta = rep_long["overall_f1"] - rep_short["overall_f1"]
    print(f"\n  Δ (long − short) = {delta:+.4f}")

    report = {
        "short_context": rep_short,
        "long_context": rep_long,
        "delta_long_vs_short": delta,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v16_long_context_cnn")
    return report


def generate_submission(output: Path = Path("submission_long_context_cnn.json")) -> Path:
    print("\n>>> Training RF hybrid on ALL 1000 windows…")
    t0 = time.time()
    rf_hybrid = build_rf_hybrid(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training long-context 3-seed CNN ensemble on full data…")
    Xl, yl = build_long_training_pool(all_window_dirs())
    print(f"    long pool X={Xl.shape}")
    long_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        long_models.append(fit_long_cnn(Xl, yl, seed=s))
        print(f"    long cnn seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_v11(w.train_x, w.train_y, w.test_x, rf_hybrid, long_models,
                            w.metric_type, long_ensemble_score)
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
    rep = run_validation()
    if rep["delta_long_vs_short"] > 0.003:
        print(f"\nLong-context CNN beats short-context by {rep['delta_long_vs_short']:+.4f}; "
              "generating submission.")
        generate_submission()
    else:
        print(f"\nLong-context CNN did not meaningfully beat short-context "
              f"(Δ = {rep['delta_long_vs_short']:+.4f}); submission NOT generated.")
