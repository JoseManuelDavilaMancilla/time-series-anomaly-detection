"""
author v26 — Transformer encoder anomaly scorer.

We need a fundamentally different model class than CNN. Attention over the
entire context lets the model learn position-aware long-range dependencies
that 1D conv (limited to ~32-pt receptive field at our settings) cannot.

Architecture:
  - Input: 64-pt z-scored context per target point
  - Linear projection to d_model=64
  - Sinusoidal position embedding
  - 3 TransformerEncoder layers (n_heads=4, dim_feedforward=128, dropout=0.1)
  - Mean-pool across time
  - Linear head → sigmoid → anomaly probability

3-seed ensemble (variance reduction is critical for transformers on small data).
Replaces the CNN channel in the v22 pipeline (metadata CW + IF + segments).

If this works, we may try Transformer + CNN dual-channel for further gain.

Run:  uv run python v26_transformer.py
"""

from __future__ import annotations

import json
import math
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
    categorize_window,
    global_distance_score,
    isolation_forest_test,
    normalize_scores,
    online_ensemble,
    per_window_rf_score,
    predict_segments,
)
from v6_cnn import (
    DEVICE, BATCH, LR, WEIGHT_POS, SUBSAMPLE_NEG, SEED, SPECIALIZED,
    _zscore_series,
)
from v7_cnn_ensemble import CNN_SEEDS
from v22_metadata_features import build_metadata_hybrid
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

CONTEXT = 64
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
FF_DIM = 128
DROPOUT = 0.1
TF_EPOCHS = 6
SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)
TF_WEIGHT = 0.35


def build_contexts(series: np.ndarray, context: int = CONTEXT) -> np.ndarray:
    s = _zscore_series(series)
    half = context // 2
    padded = np.pad(s, (half, half), mode="constant", constant_values=0.0)
    n = len(s)
    out = np.empty((n, context), dtype=np.float32)
    for i in range(n):
        out[i] = padded[i : i + context]
    return out


def build_training_pool(window_dirs, context: int = CONTEXT
                        ) -> tuple[np.ndarray, np.ndarray]:
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
    X = np.vstack(Xs); y = np.concatenate(ys)
    if 0 < SUBSAMPLE_NEG < 1.0:
        rng = np.random.default_rng(SEED)
        neg_idx = np.where(y == 0)[0]
        keep_n = int(len(neg_idx) * SUBSAMPLE_NEG)
        keep_idx = rng.choice(neg_idx, size=keep_n, replace=False)
        all_idx = np.concatenate([np.where(y == 1)[0], keep_idx])
        rng.shuffle(all_idx)
        X = X[all_idx]; y = y[all_idx]
    return X, y


class TinyTransformer(nn.Module):
    def __init__(self, context: int = CONTEXT, d_model: int = D_MODEL):
        super().__init__()
        self.context = context
        self.proj = nn.Linear(1, d_model)
        # Sinusoidal position embedding (no learnable params, more stable on small data)
        pe = torch.zeros(context, d_model)
        position = torch.arange(0, context, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, context, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=N_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        # Predict the center position by using its encoded representation
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, context) → (B, context, d_model)
        x = self.proj(x.unsqueeze(-1)) + self.pe
        z = self.encoder(x)
        # Center position
        center = z[:, self.context // 2, :]
        return self.head(center).squeeze(-1)


def fit_tf(X: np.ndarray, y: np.ndarray, seed: int) -> TinyTransformer:
    import sys
    torch.manual_seed(seed); np.random.seed(seed)
    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor([WEIGHT_POS], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    for epoch in range(TF_EPOCHS):
        model.train()
        running, n_batches = 0.0, 0
        t0 = time.time()
        for xb, yb in dl:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            running += loss.item(); n_batches += 1
        print(f"      TF epoch {epoch + 1}/{TF_EPOCHS}  loss={running / n_batches:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        sys.stdout.flush()
    return model


def tf_score(model: TinyTransformer, test_x: np.ndarray) -> np.ndarray:
    model.eval()
    X = build_contexts(test_x)
    with torch.no_grad():
        xb = torch.from_numpy(X).to(DEVICE)
        logits = model(xb)
        return torch.sigmoid(logits).cpu().numpy()


def tf_ensemble_score(models: List, test_x: np.ndarray) -> np.ndarray:
    return np.mean([tf_score(m, test_x) for m in models], axis=0)


def scores_with_tf(train_x, train_y, test_x, cw, tf_models, info, metric_type):
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type, info=info))
    tf_s = normalize_scores(tf_ensemble_score(tf_models, test_x))

    if category == "constant_train":
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return (0.50 - TF_WEIGHT) * cw_s + 0.50 * if_s + TF_WEIGHT * tf_s
    if category == "disjoint":
        g_s = normalize_scores(global_distance_score(train_x, test_x))
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return 0.35 * cw_s + 0.30 * g_s + (0.35 - TF_WEIGHT) * if_s + TF_WEIGHT * tf_s
    if category == "partial_overlap":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        return 0.35 * cw_s + 0.35 * pw + (0.30 - TF_WEIGHT) * local + TF_WEIGHT * tf_s
    if category == "test_within_train":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        return (0.50 - TF_WEIGHT) * cw_s + 0.50 * pw + TF_WEIGHT * tf_s
    return np.zeros(len(test_x))


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training metadata-CW (v22 baseline)…")
    t0 = time.time()
    cw = build_metadata_hybrid(train_pool, mode="intervals")
    print(f"    cw fit {time.time() - t0:.1f}s")

    print(">>> Building TF training pool (context=64)…")
    X, y = build_training_pool(train_pool)
    print(f"    TF pool X={X.shape}  y.mean={y.mean():.3f}")

    print(">>> Training 3-seed Transformer ensemble…")
    tf_models = []
    for s in CNN_SEEDS:
        print(f">>> TF seed={s}")
        t0 = time.time()
        tf_models.append(fit_tf(X, y, s))
        print(f"    fit {time.time() - t0:.1f}s  params={sum(p.numel() for p in tf_models[-1].parameters()):,}")

    # Baseline: load CNN ensemble for the same v22 pipeline
    from v22_metadata_features import scores_v14_with_meta as v22_scores_cnn
    from v6_cnn import build_training_pool as v6_pool
    from v7_cnn_ensemble import fit_cnn_with_seed
    print(">>> Training 3-seed CNN ensemble for baseline comparison…")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    def pred_tf(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores = scores_with_tf(sub_tr_x, sub_tr_y, sub_te_x, cw, tf_models, info,
                                info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    def pred_cnn(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores = v22_scores_cnn(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models, info,
                                info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    print("\n>>> Eval CNN baseline (v22 architecture)…")
    rep_cnn = evaluate(pred_cnn, holdout)
    print_summary(rep_cnn, name="CNN baseline (v22)")

    print(">>> Eval Transformer ensemble (v26)…")
    rep_tf = evaluate(pred_tf, holdout)
    print_summary(rep_tf, name="Transformer ensemble")

    delta = rep_tf["overall_f1"] - rep_cnn["overall_f1"]
    print(f"\n  Δ (TF − CNN) = {delta:+.4f}")

    report = {
        "cnn_f1": rep_cnn["overall_f1"],
        "tf_f1": rep_tf["overall_f1"],
        "delta": delta,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v26_transformer")
    return report, cw, tf_models


def generate_submission(cw, tf_models, output: Path = Path("submission_transformer.json")) -> Path:
    print(f"\n>>> Re-training on ALL 1000 windows…")
    t0 = time.time()
    cw_full = build_metadata_hybrid(all_window_dirs(), mode="intervals")
    print(f"    cw fit {time.time() - t0:.1f}s")

    X, y = build_training_pool(all_window_dirs())
    print(f"    TF pool X={X.shape}")
    tf_full = []
    for s in CNN_SEEDS:
        t0 = time.time()
        tf_full.append(fit_tf(X, y, s))
        print(f"    tf seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_with_tf(w.train_x, w.train_y, w.test_x, cw_full, tf_full,
                                w.info, w.metric_type)
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
    rep, cw, tf_models = run_validation()
    if rep["delta"] > 0.003:
        print(f"\nTransformer beats CNN by {rep['delta']:+.4f}; generating submission.")
        generate_submission(cw, tf_models)
    else:
        print(f"\nTransformer did not beat CNN (Δ = {rep['delta']:+.4f}); "
              "submission NOT generated.")
