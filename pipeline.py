"""
author v10 — Architecturally diverse CNN ensemble.

v7's 3-seed ensemble saturated at 3 because all 3 models share architecture —
only initialization differs. Real ensembling benefit usually requires *bias*
diversity, not just variance diversity.

This experiment trains 3 CNNs with deliberately different inductive biases:
  A: current — 32 channels, kernels (3, 5, 7), context 32   → short, narrow
  B: wide    — 64 channels, kernels (3, 5, 7), context 32   → short, wide
  C: long    — 32 channels, kernels (5, 9, 15), context 64  → long context

All three trained with the same seed (42) to isolate architecture-vs-seed
effects. We compare:
  v7  : 3-seed identical-arch ensemble  (current best CNN: 0.9497)
  v10 : 3-arch single-seed ensemble
  v10+: 3-arch ensemble × 3-seed each (9 models) — only if 3-arch wins

Run:  uv run python v10_cnn_diverse.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

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
    normalize_scores,
    online_ensemble,
    per_window_rf_score,
    predict_segments,
    v8_style_scores,
)
from v6_cnn import (
    DEVICE, EPOCHS, BATCH, LR, WEIGHT_POS, SUBSAMPLE_NEG, SEED,
    SPECIALIZED,
    _zscore_series,
    build_training_pool as v6_build_training_pool,
)
from v7_cnn_ensemble import build_hybrid
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

# Best segment params from v9
SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)


# ─────────────────────────────────────────────
# Three architectures
# ─────────────────────────────────────────────


class CNN_Narrow(nn.Module):
    """Architecture A: 32-ch, kernels 3/5/7, context 32. Identical to v6 TinyCNN."""

    def __init__(self):
        super().__init__()
        self.context = 32
        self.c1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.c3 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(32), nn.BatchNorm1d(32), nn.BatchNorm1d(32)
        self.head = nn.Linear(32 * self.context, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        return self.head(x.flatten(1)).squeeze(-1)


class CNN_Wide(nn.Module):
    """Architecture B: 64-ch, kernels 3/5/7, context 32."""

    def __init__(self):
        super().__init__()
        self.context = 32
        self.c1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.c3 = nn.Conv1d(64, 64, kernel_size=7, padding=3)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(64), nn.BatchNorm1d(64)
        self.head = nn.Linear(64 * self.context, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        return self.head(x.flatten(1)).squeeze(-1)


class CNN_Long(nn.Module):
    """Architecture C: 32-ch, kernels 5/9/15, context 64. Captures longer patterns."""

    def __init__(self):
        super().__init__()
        self.context = 64
        self.c1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.c2 = nn.Conv1d(32, 32, kernel_size=9, padding=4)
        self.c3 = nn.Conv1d(32, 32, kernel_size=15, padding=7)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(32), nn.BatchNorm1d(32), nn.BatchNorm1d(32)
        self.head = nn.Linear(32 * self.context, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        return self.head(x.flatten(1)).squeeze(-1)


# ─────────────────────────────────────────────
# Data prep — context-aware
# ─────────────────────────────────────────────


def build_contexts(series: np.ndarray, context: int) -> np.ndarray:
    s = _zscore_series(series)
    half = context // 2
    padded = np.pad(s, (half, half), mode="constant", constant_values=0.0)
    n = len(s)
    out = np.empty((n, context), dtype=np.float32)
    for i in range(n):
        out[i] = padded[i : i + context]
    return out


def build_training_pool(window_dirs, context: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for wdir in window_dirs:
        train_y = np.load(wdir / "train_label.npy")
        if train_y.sum() == 0:
            continue
        train_x = np.load(wdir / "train.npy")
        if len(train_x) < context // 2 + 4:
            continue
        Xs.append(build_contexts(train_x, context))
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


def fit_cnn(model: nn.Module, X: np.ndarray, y: np.ndarray, seed: int = 42) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
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
        print(f"      epoch {epoch + 1}/{EPOCHS}  loss={running / n_batches:.4f}  ({time.time() - t0:.1f}s)",
              flush=True)
    return model


def cnn_score(model: nn.Module, test_x: np.ndarray, context: int) -> np.ndarray:
    model.eval()
    X = build_contexts(test_x, context)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
        proba = torch.sigmoid(logits).cpu().numpy()
    return proba


def ensemble_diverse_score(models, contexts, test_x: np.ndarray) -> np.ndarray:
    scores = np.stack([cnn_score(m, test_x, c) for m, c in zip(models, contexts)], axis=0)
    return scores.mean(axis=0)


# ─────────────────────────────────────────────
# Ensemble integration
# ─────────────────────────────────────────────


def scores_with_cnn_ensemble(train_x, train_y, test_x, cw, cnn_models, contexts,
                             metric_type, cnn_weight=0.35):
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
    cnn_s = normalize_scores(ensemble_diverse_score(cnn_models, contexts, test_x))

    if category == "constant_train":
        local = normalize_scores(online_ensemble(test_x, window=15))
        return 0.40 * cw_s + (0.60 - cnn_weight) * local + cnn_weight * cnn_s
    if category == "disjoint":
        g = normalize_scores(global_distance_score(train_x, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        return 0.30 * cw_s + 0.30 * g + (0.40 - cnn_weight) * local + cnn_weight * cnn_s
    if category == "partial_overlap":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        return 0.35 * cw_s + 0.35 * pw + (0.30 - cnn_weight) * local + cnn_weight * cnn_s
    if category == "test_within_train":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        return (0.50 - cnn_weight) * cw_s + 0.50 * pw + cnn_weight * cnn_s
    return np.zeros(len(test_x))


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training hybrid CW…")
    t0 = time.time()
    cw = build_hybrid(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    # Train the 3 architectures
    archs = [
        ("narrow", CNN_Narrow, 32),
        ("wide",   CNN_Wide,   32),
        ("long",   CNN_Long,   64),
    ]
    pools = {}  # context -> (X, y)
    pools[32] = build_training_pool(train_pool, 32)
    pools[64] = build_training_pool(train_pool, 64)
    print(f"    pool@32 X={pools[32][0].shape}  pool@64 X={pools[64][0].shape}")

    cnn_models = []
    contexts = []
    for name, cls, ctx in archs:
        print(f">>> Training CNN_{name}  (context={ctx})")
        t0 = time.time()
        X, y = pools[ctx]
        m = fit_cnn(cls(), X, y, seed=42)
        cnn_models.append(m); contexts.append(ctx)
        print(f"    fit {time.time() - t0:.1f}s")

    # Reference baselines: also import v7 ensemble for direct comparison
    from v6_cnn import TinyCNN, build_training_pool as v6_pool
    from v7_cnn_ensemble import fit_cnn_with_seed
    print(">>> Training v7 reference (3 seeds, identical arch)…")
    v7_X, v7_y = v6_pool(train_pool)
    v7_models = [fit_cnn_with_seed(v7_X, v7_y, s) for s in (42, 123, 7)]

    def pred_baseline(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores, _ = v8_style_scores(sub_tr_x, sub_tr_y, sub_te_x, cw,
                                    metric_type=info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    def pred_v7(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        from v7_cnn_ensemble import scores_weighted as _v7s
        scores = _v7s(sub_tr_x, sub_tr_y, sub_te_x, cw, v7_models,
                      info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    def pred_v10(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores = scores_with_cnn_ensemble(sub_tr_x, sub_tr_y, sub_te_x, cw,
                                          cnn_models, contexts,
                                          info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    print("\n>>> Eval: v3 hybrid (baseline, segtuned segments)…")
    rep_v3 = evaluate(pred_baseline, holdout)
    print_summary(rep_v3, name="v3 baseline (segtuned)")

    print(">>> Eval: v7 3-seed identical-arch ensemble (segtuned)…")
    rep_v7 = evaluate(pred_v7, holdout)
    print_summary(rep_v7, name="v7 3-seed identical")

    print(">>> Eval: v10 3-arch diverse ensemble (segtuned)…")
    rep_v10 = evaluate(pred_v10, holdout)
    print_summary(rep_v10, name="v10 3-arch diverse")

    print(f"\n  Δ (v7  − v3 ) = {rep_v7['overall_f1']  - rep_v3['overall_f1']:+.4f}")
    print(f"  Δ (v10 − v3 ) = {rep_v10['overall_f1'] - rep_v3['overall_f1']:+.4f}")
    print(f"  Δ (v10 − v7 ) = {rep_v10['overall_f1'] - rep_v7['overall_f1']:+.4f}")

    report = {
        "v3_baseline": rep_v3,
        "v7_identical_arch": rep_v7,
        "v10_diverse_arch": rep_v10,
        "deltas": {
            "v7_minus_v3": rep_v7["overall_f1"] - rep_v3["overall_f1"],
            "v10_minus_v3": rep_v10["overall_f1"] - rep_v3["overall_f1"],
            "v10_minus_v7": rep_v10["overall_f1"] - rep_v7["overall_f1"],
        },
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v10_cnn_diverse")
    return report


def generate_submission(output: Path = Path("submission_cnn_diverse.json")) -> Path:
    print("\n>>> Training hybrid on ALL 1000…")
    t0 = time.time()
    cw = build_hybrid(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    archs = [("narrow", CNN_Narrow, 32), ("wide", CNN_Wide, 32), ("long", CNN_Long, 64)]
    pools = {32: build_training_pool(all_window_dirs(), 32),
             64: build_training_pool(all_window_dirs(), 64)}
    print(f"    pool@32 X={pools[32][0].shape}  pool@64 X={pools[64][0].shape}")

    cnn_models, contexts = [], []
    for name, cls, ctx in archs:
        print(f">>> CNN_{name} (full data, context={ctx})")
        t0 = time.time()
        X, y = pools[ctx]
        cnn_models.append(fit_cnn(cls(), X, y, seed=42)); contexts.append(ctx)
        print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_with_cnn_ensemble(w.train_x, w.train_y, w.test_x, cw,
                                          cnn_models, contexts, w.metric_type)
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
    if rep["deltas"]["v10_minus_v7"] > 0.002:
        print(f"\nDiverse architectures beat identical by {rep['deltas']['v10_minus_v7']:+.4f}; "
              "generating submission.")
        generate_submission()
    else:
        print(f"\nDiverse architectures did not beat identical "
              f"(Δ = {rep['deltas']['v10_minus_v7']:+.4f}); submission NOT generated.")
