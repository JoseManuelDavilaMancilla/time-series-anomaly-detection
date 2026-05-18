"""
author v19 — 1D convolutional autoencoder reconstruction-error channel (radical).

v18's segment-level classifier showed that every existing channel measures
essentially the same thing: "is this point unusual relative to a reference".
The feature importances were dominated by CW score statistics. Adding shape
features did not help because CW already captures shape implicitly.

A genuinely different signal type is **reconstruction error** under a model
that learned only the NORMAL distribution. Train an autoencoder on training
windows (with high weight on normal points, zero/low weight on anomalies);
at inference, the points the AE cannot reconstruct are anomalous.

Implementation:
  - 1D conv autoencoder: input is a 32-point z-scored context window.
    Encoder: Conv(1→32, k=3) → Conv(32→64, k=5, stride=2) → Conv(64→64, k=7).
    Decoder: mirror. Bottleneck of 16 channels × 16 timesteps.
  - Trained on training-window contexts where train_y == 0 (normal points
    only — this is what makes it an "unsupervised normality model").
  - Anomaly score per point = squared reconstruction error at the centre.
  - Added to v14's ensemble at a tunable weight (sweep 0.05/0.10/0.15).

Stacks on top of v14 (our current 0.6238 LB best).

Run:  uv run python v19_autoencoder.py
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
    BATCH, DEVICE, LR, SEED, SPECIALIZED, SUBSAMPLE_NEG,
    _zscore_series,
    build_training_pool as v6_pool,
)
from v7_cnn_ensemble import CNN_SEEDS, ensemble_cnn_score, fit_cnn_with_seed
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

CONTEXT = 32
AE_EPOCHS = 6
AE_LR = 1e-3
SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)
CNN_WEIGHT = 0.35


# ─────────────────────────────────────────────
# Autoencoder model
# ─────────────────────────────────────────────


class ConvAE(nn.Module):
    """Small 1D conv autoencoder for 32-pt context. ~30k params."""

    def __init__(self, context: int = CONTEXT):
        super().__init__()
        self.context = context
        # Encoder
        self.e1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)            # 32, 32
        self.e2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)  # 64, 16
        self.e3 = nn.Conv1d(64, 64, kernel_size=7, padding=3)            # 64, 16
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(32), nn.BatchNorm1d(64), nn.BatchNorm1d(64)
        # Decoder
        self.d1 = nn.Conv1d(64, 64, kernel_size=7, padding=3)
        self.d2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.d3 = nn.Conv1d(32, 1, kernel_size=3, padding=1)
        self.bn4, self.bn5 = nn.BatchNorm1d(64), nn.BatchNorm1d(32)

    def forward(self, x):
        # x: (B, context)
        x = x.unsqueeze(1)  # (B, 1, context)
        h = F.relu(self.bn1(self.e1(x)))
        h = F.relu(self.bn2(self.e2(h)))
        h = F.relu(self.bn3(self.e3(h)))
        h = F.relu(self.bn4(self.d1(h)))
        h = F.relu(self.bn5(self.d2(h)))
        out = self.d3(h)
        return out.squeeze(1)  # (B, context)


# ─────────────────────────────────────────────
# Data: build NORMAL-only contexts pool
# ─────────────────────────────────────────────


def build_contexts(series: np.ndarray, context: int = CONTEXT) -> np.ndarray:
    s = _zscore_series(series)
    half = context // 2
    padded = np.pad(s, (half, half), mode="constant", constant_values=0.0)
    n = len(s)
    out = np.empty((n, context), dtype=np.float32)
    for i in range(n):
        out[i] = padded[i : i + context]
    return out


def build_normal_pool(window_dirs) -> np.ndarray:
    """Per-point contexts where train_label == 0 only. AE learns to reconstruct normal points."""
    Xs = []
    for wdir in window_dirs:
        train_y = np.load(wdir / "train_label.npy")
        train_x = np.load(wdir / "train.npy")
        if len(train_x) < CONTEXT // 2 + 4:
            continue
        ctxs = build_contexts(train_x)
        mask = train_y == 0
        Xs.append(ctxs[mask])
    X = np.vstack(Xs)
    # Subsample to keep training quick (~80k samples)
    if len(X) > 80000:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X), 80000, replace=False)
        X = X[idx]
    return X


def fit_autoencoder(X: np.ndarray, seed: int = 42) -> ConvAE:
    import sys
    torch.manual_seed(seed); np.random.seed(seed)
    model = ConvAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    for epoch in range(AE_EPOCHS):
        model.train()
        running, n_batches = 0.0, 0
        t0 = time.time()
        for (xb,) in dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward(); opt.step()
            running += loss.item(); n_batches += 1
        print(f"      AE epoch {epoch + 1}/{AE_EPOCHS}  loss={running / n_batches:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        sys.stdout.flush()
    return model


# ─────────────────────────────────────────────
# Reconstruction-error scoring
# ─────────────────────────────────────────────


def ae_score(model: ConvAE, test_x: np.ndarray) -> np.ndarray:
    """Per-point reconstruction error at the centre of each context window."""
    model.eval()
    X = build_contexts(test_x)
    with torch.no_grad():
        xb = torch.from_numpy(X).to(DEVICE)
        recon = model(xb).cpu().numpy()
    # Squared error at centre position
    centre = CONTEXT // 2
    return (recon[:, centre] - X[:, centre]) ** 2


# ─────────────────────────────────────────────
# Ensemble integration
# ─────────────────────────────────────────────


def scores_with_ae(train_x, train_y, test_x, cw, cnn_models, ae_model,
                   metric_type, ae_weight: float = 0.10) -> np.ndarray:
    """v14 ensemble + autoencoder channel at low weight, trimming CW share."""
    category = categorize_window(train_x, test_x)
    cw_s   = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
    cnn_s  = normalize_scores(ensemble_cnn_score(cnn_models, test_x))
    ae_s   = normalize_scores(ae_score(ae_model, test_x))

    if category == "constant_train":
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return ((0.50 - ae_weight) * cw_s + 0.50 * if_s
                + CNN_WEIGHT * cnn_s + ae_weight * ae_s
                - CNN_WEIGHT * cw_s)  # cancel CNN's slice off cw, keep formula consistent
    if category == "disjoint":
        g_s = normalize_scores(global_distance_score(train_x, test_x))
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return (0.35 * cw_s + 0.30 * g_s + (0.35 - CNN_WEIGHT - ae_weight) * if_s
                + CNN_WEIGHT * cnn_s + ae_weight * ae_s)
    if category == "partial_overlap":
        pw    = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        return (0.35 * cw_s + 0.35 * pw + (0.30 - CNN_WEIGHT - ae_weight) * local
                + CNN_WEIGHT * cnn_s + ae_weight * ae_s)
    if category == "test_within_train":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        return ((0.50 - CNN_WEIGHT - ae_weight) * cw_s + 0.50 * pw
                + CNN_WEIGHT * cnn_s + ae_weight * ae_s)
    return np.zeros(len(test_x))


def scores_v14(train_x, train_y, test_x, cw, cnn_models, metric_type) -> np.ndarray:
    """Baseline v14 with no AE channel."""
    return scores_with_ae(train_x, train_y, test_x, cw, cnn_models,
                          ae_model=None, metric_type=metric_type, ae_weight=0.0) \
        if False else _scores_v14_noae(train_x, train_y, test_x, cw, cnn_models, metric_type)


def _scores_v14_noae(train_x, train_y, test_x, cw, cnn_models, metric_type) -> np.ndarray:
    category = categorize_window(train_x, test_x)
    cw_s   = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
    cnn_s  = normalize_scores(ensemble_cnn_score(cnn_models, test_x))
    if category == "constant_train":
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * if_s + CNN_WEIGHT * cnn_s
    if category == "disjoint":
        g_s = normalize_scores(global_distance_score(train_x, test_x))
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return 0.35 * cw_s + 0.30 * g_s + (0.35 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
    if category == "partial_overlap":
        pw    = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
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

    print(">>> Building NORMAL-only pool for AE…")
    Xae = build_normal_pool(train_pool)
    print(f"    AE pool X={Xae.shape}")
    print(">>> Training autoencoder…")
    t0 = time.time()
    ae_model = fit_autoencoder(Xae, seed=42)
    print(f"    fit {time.time() - t0:.1f}s")

    def predictor_factory(ae_w):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            if ae_w == 0:
                scores = _scores_v14_noae(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                          info.get("metric_type", "ALL"))
            else:
                scores = scores_with_ae(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models,
                                        ae_model, info.get("metric_type", "ALL"),
                                        ae_weight=ae_w)
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    print("\n>>> Eval: v14 baseline (no AE)…")
    rep_base = evaluate(predictor_factory(0.0), holdout)
    print_summary(rep_base, name="v14 baseline")

    results = {"baseline": rep_base["overall_f1"]}
    for w in (0.05, 0.10, 0.15, 0.20):
        print(f"\n>>> Eval: v19 with AE_WEIGHT={w:.2f}…")
        rep = evaluate(predictor_factory(w), holdout)
        print_summary(rep, name=f"v19 ae_w={w:.2f}")
        results[f"ae_w_{w:.2f}"] = rep["overall_f1"]

    print("\n──────  summary ──────")
    base = results["baseline"]
    rows = sorted([(k, v) for k, v in results.items() if k != "baseline"], key=lambda kv: -kv[1])
    print(f"  baseline   F1={base:.4f}")
    for name, f1 in rows:
        print(f"  {name}  F1={f1:.4f}  Δ_vs_baseline={f1 - base:+.4f}")

    winner_name, winner_f1 = rows[0]
    report = {
        "results": results,
        "winner": winner_name,
        "delta_vs_baseline": winner_f1 - base,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v19_autoencoder")
    return report, cw, cnn_models, ae_model


def generate_submission(ae_weight: float, cw, cnn_models, ae_model,
                        output: Path = Path("submission_autoencoder.json")) -> Path:
    print(f"\n>>> Re-training all models on ALL 1000 windows (ae_weight={ae_weight:.2f})…")
    t0 = time.time()
    cw_full = build_rf_hybrid(all_window_dirs())
    print(f"    rf fit {time.time() - t0:.1f}s")

    Xc, yc = v6_pool(all_window_dirs())
    cnn_full = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_full.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    Xae = build_normal_pool(all_window_dirs())
    print(f"    AE pool X={Xae.shape}")
    t0 = time.time()
    ae_full = fit_autoencoder(Xae, seed=42)
    print(f"    ae fit {time.time() - t0:.1f}s")

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_with_ae(w.train_x, w.train_y, w.test_x, cw_full, cnn_full,
                                ae_full, w.metric_type, ae_weight=ae_weight)
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
    rep, cw, cnn_models, ae_model = run_validation()
    if rep["delta_vs_baseline"] > 0.003:
        ae_w = float(rep["winner"].replace("ae_w_", ""))
        print(f"\nAE channel at weight {ae_w} beats baseline by {rep['delta_vs_baseline']:+.4f}; "
              "generating submission.")
        generate_submission(ae_w, cw, cnn_models, ae_model)
    else:
        print(f"\nAE channel did not meaningfully help "
              f"(best Δ = {rep['delta_vs_baseline']:+.4f}); submission NOT generated.")
