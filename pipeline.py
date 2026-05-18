"""
author v30 — Anomaly Transformer (Xu et al., ICLR 2022).

The key insight of the Anomaly Transformer is the **Association Discrepancy**:
normal points' attention concentrates locally (well-approximated by a
Gaussian centered on the point), while anomalous points attend to distant
positions (their series-attention diverges from any local-Gaussian prior).

This is a fundamentally different inductive bias from CNN or plain
Transformer encoders:

  - **Series-Association**: standard multi-head softmax attention
  - **Prior-Association**: a learnable Gaussian for each head:
        prior[h, i, j] = exp(-(j-i)² / (2·σ[h]²)) / Z
  - **Discrepancy**: KL(prior ‖ series) per position
  - **Loss**: BCE + λ · directed_discrepancy
        where directed_discrepancy is +K(prior‖series) for normal points
        (encourages local attention) and −K(prior‖series) for anomalies
        (encourages distant attention)
  - **Inference**: anomaly_score = sigmoid(logits) blended with normalized
    discrepancy

Phase 1: validate that this architecture beats the plain TF encoder (v26).
If yes, proceed to multi-seed ensemble and submission. If no, the model
class is wrong for this dataset.

Run:  uv run python v30_anomaly_transformer.py
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from v6_cnn import (
    DEVICE, BATCH, LR, WEIGHT_POS, SUBSAMPLE_NEG, SEED, SPECIALIZED,
    _zscore_series,
)
from v26_transformer import build_contexts, build_training_pool

CONTEXT = 64
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 3
FF_DIM = 256
DROPOUT = 0.1
AT_EPOCHS = 6
LAMBDA_DISCREPANCY = 0.1  # weight on association discrepancy term — tuned because raw disc magnitude (~10) is much larger than BCE (~1)


class AnomalyAttention(nn.Module):
    """One Anomaly Transformer encoder layer: series + prior + discrepancy."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, context: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.context = context

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        # Per-head sigma for prior: parameterized via softplus to keep positive
        # Start with sigma=1.0 per head (i.e., prior with std=1 timestep)
        self.sigma_raw = nn.Parameter(torch.zeros(n_heads))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(DROPOUT)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(ff_dim, d_model),
        )

        # Precompute relative-position tensor (constant for fixed context)
        pos = torch.arange(context, dtype=torch.float32)
        diff_sq = (pos[None, :] - pos[:, None]).pow(2)  # (L, L)
        self.register_buffer("diff_sq", diff_sq)

    def gaussian_prior(self) -> torch.Tensor:
        """Returns prior of shape (n_heads, L, L), softmax-normalized along last dim."""
        sigma = F.softplus(self.sigma_raw) + 1e-3  # (n_heads,)
        sigma_sq = sigma.pow(2)[:, None, None]  # (n_heads, 1, 1)
        log_prior = -self.diff_sq / (2.0 * sigma_sq)  # (n_heads, L, L)
        return F.softmax(log_prior, dim=-1)

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        series = F.softmax(scores, dim=-1)  # (B, n_heads, L, L)
        series = self.dropout(series)
        z = torch.matmul(series, v)  # (B, n_heads, L, head_dim)
        z = z.transpose(1, 2).reshape(B, L, D)
        z = self.proj(z)
        x = self.norm1(x + self.dropout(z))
        x = self.norm2(x + self.dropout(self.ffn(x)))

        prior = self.gaussian_prior()  # (n_heads, L, L), no batch dim
        return x, series, prior


def association_discrepancy(series: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between series (B,h,L,L) and prior (h,L,L), summed over key dim.
    Returns (B, L): discrepancy per position averaged over heads."""
    # Broadcast prior to batch
    prior_b = prior.unsqueeze(0)  # (1, h, L, L)
    eps = 1e-9
    kl_pq = (prior_b * (prior_b + eps).log() - prior_b * (series + eps).log()).sum(dim=-1)
    kl_qp = (series * (series + eps).log() - series * (prior_b + eps).log()).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp).mean(dim=1)  # (B, L)


class AnomalyTransformer(nn.Module):
    def __init__(self, context: int = CONTEXT, d_model: int = D_MODEL):
        super().__init__()
        self.context = context
        self.d_model = d_model
        self.proj = nn.Linear(1, d_model)
        # Sinusoidal PE
        pe = torch.zeros(context, d_model)
        position = torch.arange(0, context, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.layers = nn.ModuleList(
            [AnomalyAttention(d_model, N_HEADS, FF_DIM, context) for _ in range(N_LAYERS)]
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, context) — we return logits and per-layer discrepancies at center
        z = self.proj(x.unsqueeze(-1)) + self.pe
        discrepancies = []
        for layer in self.layers:
            z, series, prior = layer(z)
            discrepancies.append(association_discrepancy(series, prior))  # (B, L)
        center = self.context // 2
        logits = self.head(z[:, center, :]).squeeze(-1)  # (B,)
        # Average discrepancy across layers, at center
        disc_center = torch.stack([d[:, center] for d in discrepancies]).mean(dim=0)  # (B,)
        return logits, disc_center


def fit_at(X: np.ndarray, y: np.ndarray, seed: int) -> AnomalyTransformer:
    """Train AT with BCE + λ * directed discrepancy.

    Directed discrepancy:
      - For normal points (y=0): minimize discrepancy (small KL)
      - For anomaly points (y=1): maximize discrepancy (large KL)
    """
    import sys
    torch.manual_seed(seed); np.random.seed(seed)
    model = AnomalyTransformer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor([WEIGHT_POS], dtype=torch.float32, device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    for epoch in range(AT_EPOCHS):
        model.train()
        running_bce, running_disc, n_batches = 0.0, 0.0, 0
        t0 = time.time()
        for xb, yb in dl:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            logits, disc = model(xb)
            bce_loss = bce(logits, yb)
            # Directed discrepancy: anomalies → large disc (push up), normal → small disc (push down)
            # disc_loss = mean( (1 - 2y) * disc ) so:
            #   y=0 → +disc (penalty when disc is large)
            #   y=1 → -disc (reward when disc is large)
            disc_loss = ((1.0 - 2.0 * yb) * disc).mean()
            loss = bce_loss + LAMBDA_DISCREPANCY * disc_loss
            loss.backward(); opt.step()
            running_bce += bce_loss.item()
            running_disc += disc_loss.item()
            n_batches += 1
        print(f"      AT epoch {epoch + 1}/{AT_EPOCHS}  "
              f"bce={running_bce / n_batches:.4f}  disc={running_disc / n_batches:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        sys.stdout.flush()
    return model


def at_score(model: AnomalyTransformer, test_x: np.ndarray, disc_weight: float = 0.5
             ) -> np.ndarray:
    """Per-point score = sigmoid(logits) + disc_weight * normalized discrepancy."""
    model.eval()
    X = build_contexts(test_x)
    with torch.no_grad():
        xb = torch.from_numpy(X).to(DEVICE)
        logits, disc = model(xb)
        proba = torch.sigmoid(logits).cpu().numpy()
        disc_np = disc.cpu().numpy()
    # Normalize discrepancy to [0,1]
    if disc_np.max() > disc_np.min():
        disc_norm = (disc_np - disc_np.min()) / (disc_np.max() - disc_np.min())
    else:
        disc_norm = np.zeros_like(disc_np)
    return (1.0 - disc_weight) * proba + disc_weight * disc_norm


# Phase 1 validation only — run a single seed, evaluate against v26's TF for comparison.
def run_phase1(seed: int = 42) -> dict:
    from validation import all_window_dirs, evaluate, load_window, print_summary, stratified_holdout, save_report
    from v22_metadata_features import build_metadata_hybrid
    from v7_cnn_ensemble import CNN_SEEDS, fit_cnn_with_seed, ensemble_cnn_score
    from v6_cnn import build_training_pool as v6_pool
    from shared_lib import (predict_segments, normalize_scores, categorize_window,
                            online_ensemble, global_distance_score, per_window_rf_score,
                            isolation_forest_test)

    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training metadata CW (v22)…")
    t0 = time.time()
    cw = build_metadata_hybrid(train_pool, mode="intervals")
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training 3-seed CNN ensemble (for v22 baseline comparison)…")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Building AT training pool (context=64)…")
    X, y = build_training_pool(train_pool)
    print(f"    AT pool X={X.shape}  y.mean={y.mean():.3f}")

    print(f">>> Training 1-seed Anomaly Transformer (phase 1)…")
    t0 = time.time()
    at_model = fit_at(X, y, seed=42)
    print(f"    fit {time.time() - t0:.1f}s  params={sum(p.numel() for p in at_model.parameters()):,}")

    SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)
    CNN_WEIGHT = 0.35

    def ensemble_with_x(x_score, train_x, train_y, test_x, info, metric_type):
        category = categorize_window(train_x, test_x)
        cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type, info=info))
        x_s = normalize_scores(x_score)
        if category == "constant_train":
            if_s = normalize_scores(isolation_forest_test(test_x, train_y))
            return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * if_s + CNN_WEIGHT * x_s
        if category == "disjoint":
            g_s = normalize_scores(global_distance_score(train_x, test_x))
            if_s = normalize_scores(isolation_forest_test(test_x, train_y))
            return 0.35 * cw_s + 0.30 * g_s + (0.35 - CNN_WEIGHT) * if_s + CNN_WEIGHT * x_s
        if category == "partial_overlap":
            pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
            local = normalize_scores(online_ensemble(test_x, window=15))
            return 0.35 * cw_s + 0.35 * pw + (0.30 - CNN_WEIGHT) * local + CNN_WEIGHT * x_s
        if category == "test_within_train":
            pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
            return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * pw + CNN_WEIGHT * x_s
        return np.zeros(len(test_x))

    def pred_cnn(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        x_score = ensemble_cnn_score(cnn_models, sub_te_x)
        scores = ensemble_with_x(x_score, sub_tr_x, sub_tr_y, sub_te_x, info,
                                 info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    def make_pred_at(disc_w):
        def fn(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            x_score = at_score(at_model, sub_te_x, disc_weight=disc_w)
            scores = ensemble_with_x(x_score, sub_tr_x, sub_tr_y, sub_te_x, info,
                                     info.get("metric_type", "ALL"))
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return fn

    print("\n>>> Eval v22 CNN baseline (3-seed CNN ensemble)…")
    rep_cnn = evaluate(pred_cnn, holdout)
    print_summary(rep_cnn, name="v22 CNN baseline")

    results = {"cnn_v22_baseline": rep_cnn["overall_f1"]}
    for disc_w in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"\n>>> Eval AT (disc_w={disc_w:.2f})…")
        rep = evaluate(make_pred_at(disc_w), holdout)
        print_summary(rep, name=f"AT disc_w={disc_w:.2f}")
        results[f"at_disc_{disc_w:.2f}"] = rep["overall_f1"]

    print("\n──────  phase 1 summary ──────")
    base = results["cnn_v22_baseline"]
    rows = sorted(results.items(), key=lambda kv: -kv[1])
    for name, f1 in rows:
        print(f"  {name:<22}  F1={f1:.4f}  Δ_vs_cnn_baseline={f1 - base:+.4f}")

    winner = max(results, key=lambda k: results[k])
    report = {
        "results": results,
        "winner": winner,
        "winner_f1": results[winner],
        "delta_vs_cnn": results[winner] - base,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v30_anomaly_transformer_phase1")
    return report


if __name__ == "__main__":
    rep = run_phase1()
    if rep["winner"].startswith("at_") and rep["delta_vs_cnn"] > 0:
        print(f"\nPHASE 1 PASS: {rep['winner']} beats CNN by {rep['delta_vs_cnn']:+.4f}.")
        print("→ Proceed to phase 2 (3-seed ensemble + tuning).")
    else:
        print(f"\nPHASE 1 FAIL: AT did not beat CNN (best Δ = {rep['delta_vs_cnn']:+.4f}).")
        print("→ Model class wrong for this dataset. Do not proceed to phase 2.")
