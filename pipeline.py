"""
author v6 — 1D CNN cross-window scorer, added to the v3 hybrid ensemble.

Hypothesis: the CW RF only sees lag features up to 10 and rolling windows of 3–10.
A small 1D CNN with kernel sizes 3/5/7 sees a 32-point context around each point
and can learn temporal patterns the RF can't (e.g. asymmetric ramps, oscillation
breaks).

Implementation:
- For every point in every training series, build a 32-point z-scored context
  (16 past + 16 future, zero-padded at edges).
- Pool across all 1000 windows → ~300k samples.
- Train a 4-layer 1D CNN with sigmoid output (per-point anomaly probability).
- Use it as an additional channel in the v8-style ensemble alongside the hybrid CW.

The CNN model is trained once, persisted to `claude_cnn.pt`, and reused.

Run:  uv run python v6_cnn.py
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
    normalize_scores,
    online_ensemble,
    global_distance_score,
    per_window_rf_score,
    categorize_window,
    predict_segments,
)
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

CONTEXT = 32                 # ±16 around the target point
HIDDEN = 32
EPOCHS = 4                   # 4 epochs is enough; more overfits or stalls
BATCH = 4096
LR = 1e-3
WEIGHT_POS = 5.0             # positive-class weight (anomaly rate ≈ 9%)
SEED = 42
SUBSAMPLE_NEG = 0.30         # keep only 30% of negative samples to balance + speed up training

SEG_KWARGS = dict(smooth=5, thr_frac=0.6, small_k_cutoff=4, max_seg=80)
SPECIALIZED = frozenset({"ErrorCount", "ResourceUtilizationRate", "SuccessRate"})

DEVICE = torch.device("cpu")  # CPU is fine for ~300k samples


# ─────────────────────────────────────────────
# Data prep
# ─────────────────────────────────────────────


def _zscore_series(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32)


def build_contexts(series: np.ndarray, context: int = CONTEXT) -> np.ndarray:
    """For each point i, return z-scored context [i−context//2 .. i+context//2−1].
    Zero-padded at edges. Shape: (len(series), context)."""
    s = _zscore_series(series)
    n = len(s)
    half = context // 2
    padded = np.pad(s, (half, half), mode="constant", constant_values=0.0)
    out = np.empty((n, context), dtype=np.float32)
    for i in range(n):
        out[i] = padded[i : i + context]
    return out


def build_training_pool(window_dirs, subsample_neg: float = SUBSAMPLE_NEG) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for wdir in window_dirs:
        train_y = np.load(wdir / "train_label.npy")
        if train_y.sum() == 0:
            continue
        train_x = np.load(wdir / "train.npy")
        if len(train_x) < CONTEXT // 2 + 4:
            continue
        Xs.append(build_contexts(train_x))
        ys.append(train_y.astype(np.float32))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    # Subsample negatives to balance the dataset and cut training time
    if 0 < subsample_neg < 1.0:
        rng = np.random.default_rng(SEED)
        neg_idx = np.where(y == 0)[0]
        keep_n = int(len(neg_idx) * subsample_neg)
        keep_idx = rng.choice(neg_idx, size=keep_n, replace=False)
        all_idx = np.concatenate([np.where(y == 1)[0], keep_idx])
        rng.shuffle(all_idx)
        X = X[all_idx]
        y = y[all_idx]
    return X, y


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────


class TinyCNN(nn.Module):
    def __init__(self, context: int = CONTEXT, hidden: int = HIDDEN):
        super().__init__()
        self.c1 = nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.c3 = nn.Conv1d(hidden, hidden, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.head = nn.Linear(hidden * context, 1)

    def forward(self, x):       # x: (B, context)
        x = x.unsqueeze(1)      # (B, 1, context)
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        x = x.flatten(1)
        return self.head(x).squeeze(-1)


def fit_cnn(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> TinyCNN:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TinyCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor([WEIGHT_POS], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    import sys

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        n_batches = 0
        t0 = time.time()
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        print(f"    epoch {epoch + 1:2d}/{EPOCHS}  loss={running / n_batches:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)
        sys.stdout.flush()
    return model


def cnn_score(model: TinyCNN, test_x: np.ndarray) -> np.ndarray:
    model.eval()
    X = build_contexts(test_x)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
        proba = torch.sigmoid(logits).cpu().numpy()
    return proba


# ─────────────────────────────────────────────
# Ensemble (mirrors v8_style_scores but adds CNN channel)
# ─────────────────────────────────────────────


def ensemble_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    cw: HybridCrossWindowModel,
    cnn_model: TinyCNN,
    metric_type: str,
    cnn_weight: float = 0.25,
) -> tuple[np.ndarray, str]:
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
    cnn_s = normalize_scores(cnn_score(cnn_model, test_x))

    if category == "constant_train":
        local = normalize_scores(online_ensemble(test_x, window=15))
        # v8: 0.4 cw + 0.6 local. Make room for CNN by trimming local.
        scores = 0.40 * cw_s + (0.60 - cnn_weight) * local + cnn_weight * cnn_s
    elif category == "disjoint":
        g = normalize_scores(global_distance_score(train_x, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        # v8: 0.3 cw + 0.3 g + 0.4 local. Trim local.
        scores = 0.30 * cw_s + 0.30 * g + (0.40 - cnn_weight) * local + cnn_weight * cnn_s
    elif category == "partial_overlap":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        # v8: 0.35 cw + 0.35 pw + 0.30 local. Trim local.
        scores = 0.35 * cw_s + 0.35 * pw + (0.30 - cnn_weight) * local + cnn_weight * cnn_s
    elif category == "test_within_train":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        # v8: 0.5 cw + 0.5 pw. Trim cw to make room.
        scores = (0.50 - cnn_weight) * cw_s + 0.50 * pw + cnn_weight * cnn_s
    else:
        scores = np.zeros(len(test_x))
    return scores, category


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────


def build_hybrid(window_dirs):
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=200, max_depth=12).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=200, max_depth=12).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training hybrid cross-window…")
    t0 = time.time()
    cw = build_hybrid(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Building CNN training pool…")
    t0 = time.time()
    X, y = build_training_pool(train_pool)
    print(f"    X.shape={X.shape}  y.mean={y.mean():.3f}  (built in {time.time() - t0:.1f}s)")

    print(">>> Training CNN…")
    t0 = time.time()
    cnn_model = fit_cnn(X, y)
    print(f"    fit {time.time() - t0:.1f}s")

    # Hybrid + segments without CNN — baseline
    from shared_lib import v8_style_scores as _v8

    def pred_no_cnn(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores, _ = _v8(sub_tr_x, sub_tr_y, sub_te_x, cw,
                        metric_type=info.get("metric_type", "ALL"))
        k = int(round(len(sub_te_x) * ratio))
        return predict_segments(scores, k, **SEG_KWARGS)

    def pred_with_cnn(sub_tr_x, sub_tr_y, sub_te_x, info, ratio, w=0.25):
        scores, _ = ensemble_scores(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_model,
                                    metric_type=info.get("metric_type", "ALL"),
                                    cnn_weight=w)
        k = int(round(len(sub_te_x) * ratio))
        return predict_segments(scores, k, **SEG_KWARGS)

    print("\n>>> Eval: hybrid + segments (v3 baseline)…")
    rep_v3 = evaluate(pred_no_cnn, holdout)
    print_summary(rep_v3, name="v3 hybrid + segments")

    sweep = {}
    for w in (0.15, 0.25, 0.35):
        print(f"\n>>> Eval: hybrid + segments + CNN (weight={w:.2f})…")
        def _pred(a, b, c, d, e, _w=w):
            return pred_with_cnn(a, b, c, d, e, w=_w)
        rep = evaluate(_pred, holdout)
        print_summary(rep, name=f"v6 cnn weight={w:.2f}")
        sweep[w] = rep

    best_w = max(sweep.keys(), key=lambda k: sweep[k]["overall_f1"])
    print(f"\n    Δ vs v3 baseline (best CNN weight={best_w:.2f}): "
          f"{sweep[best_w]['overall_f1'] - rep_v3['overall_f1']:+.4f}")

    report = {
        "v3_baseline": rep_v3,
        "cnn_weight_sweep": {f"{w:.2f}": r for w, r in sweep.items()},
        "best_cnn_weight": best_w,
        "delta_best_vs_baseline": sweep[best_w]["overall_f1"] - rep_v3["overall_f1"],
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v6_cnn")
    # Persist CNN
    torch.save(cnn_model.state_dict(), "cnn_validation.pt")
    return report


def generate_submission(cnn_weight: float,
                        output: Path = Path("submission_cnn.json")) -> Path:
    print("\n>>> Training hybrid on ALL 1000 windows…")
    t0 = time.time()
    cw = build_hybrid(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Building full CNN training pool…")
    X, y = build_training_pool(all_window_dirs())
    print(f"    X.shape={X.shape}  y.mean={y.mean():.3f}")
    print(">>> Training CNN…")
    t0 = time.time()
    cnn_model = fit_cnn(X, y)
    torch.save(cnn_model.state_dict(), "cnn_full.pt")
    print(f"    fit {time.time() - t0:.1f}s")

    print(f">>> Generating predictions (cnn_weight={cnn_weight:.2f})…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores, _ = ensemble_scores(w.train_x, w.train_y, w.test_x, cw, cnn_model,
                                    metric_type=w.metric_type, cnn_weight=cnn_weight)
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
    if rep["delta_best_vs_baseline"] > 0.002:
        print(f"\nCNN improves by {rep['delta_best_vs_baseline']:+.4f}; generating submission.")
        generate_submission(rep["best_cnn_weight"])
    else:
        print(f"\n!! CNN gain {rep['delta_best_vs_baseline']:+.4f} < 0.002; "
              "submission NOT generated.")
