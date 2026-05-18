"""
author v38 — faithful replica of friend's 0.6453 LB pipeline.

Key design points (from friend's notes):

  1. 68 engineered features per point:
     - 22 point-level (value, robust/standard z, diffs, rolling mean/std at
       w=5/11/21, rolling median+MAD at w=11, EWMA + residual + z, percentile
       rank vs train ECDF, position-in-window, time-of-day, day-of-week)
     - 7 long-range (rolling stats at w=41 + vol_ratio + max/min deltas)
     - 39 static window-level (metric_type one-hot×6, top-30 service one-hot,
       intervals in hours, train/test anomaly ratios)

  2. Per-metric-type training (NOT fully pooled, NOT per-window):
     - Group labeled windows by metric_type (6 categories).
     - Pool all labeled training points across windows of that type.
     - Train 3 RF (seeds 0/1/2) + 5 HGBT (seeds 0..4) + 1 LR per type.

  3. Blend probabilities:
     prob_mean = 0.80 * HGBT_avg + 0.10 * RF_avg + 0.10 * LR_avg

  4. Smooth (centered):
     rm = rolling_mean(prob_mean, 5)
     prob_final = 0.2 * prob_mean + 0.8 * rm

  5. Top-k via plain argpartition. Zero-ratio windows → predict all zeros.

  NO segment growth, NO IF, NO per-window RF, NO CNN. Our complex extras
  have been hurting LB despite winning validation — this is the lesson.

Run:  uv run python v38_friend_repro.py
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

from validation import all_window_dirs, load_window
from cross_validation import cross_window_evaluate, print_summary_v2
from validation import stratified_holdout, point_f1


METRIC_TYPES = ("Count", "ErrorCount", "LatencySecond", "QPS",
                "ResourceUtilizationRate", "SuccessRate")
TOP_K_SERVICES = 30
SMOOTH_W = 5
SMOOTH_ALPHA = 0.8


# ─────────────────────────────────────────────
# Feature engineering — 68 features per point
# ─────────────────────────────────────────────


def _rolling_mean_std(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Causal rolling mean+std via cumsum trick. Edge windows use whatever is available."""
    n = len(x)
    s = x.astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(s)])
    csum2 = np.concatenate([[0.0], np.cumsum(s * s)])
    idx = np.arange(n)
    start = np.maximum(0, idx - w + 1)
    counts = (idx - start + 1).astype(np.float64)
    sums = csum[idx + 1] - csum[start]
    sums2 = csum2[idx + 1] - csum2[start]
    mean = sums / counts
    var = np.maximum(sums2 / counts - mean * mean, 0.0)
    std = np.sqrt(var)
    return mean, std


def _rolling_minmax(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    rmin = np.empty(n)
    rmax = np.empty(n)
    for i in range(n):
        seg = x[max(0, i - w + 1) : i + 1]
        rmin[i] = seg.min()
        rmax[i] = seg.max()
    return rmin, rmax


def _rolling_median_mad(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    med = np.empty(n)
    mad = np.empty(n)
    for i in range(n):
        seg = x[max(0, i - w + 1) : i + 1]
        m = np.median(seg)
        med[i] = m
        mad[i] = np.median(np.abs(seg - m))
    return med, mad


def _ewma(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    e = float(x[0])
    for i, v in enumerate(x):
        e = alpha * float(v) + (1 - alpha) * e
        out[i] = e
    return out


def _percentile_rank_vs(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Each value's percentile rank in `reference`, in [0, 1]."""
    ref_sorted = np.sort(reference)
    n_ref = len(ref_sorted)
    if n_ref == 0:
        return np.full(len(values), 0.5)
    idx = np.searchsorted(ref_sorted, values, side="right")
    return idx.astype(np.float64) / n_ref


def _time_features(timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (time_of_day_fraction, day_of_week_fraction)."""
    n = len(timestamps)
    tod = np.empty(n, dtype=np.float64)
    dow = np.empty(n, dtype=np.float64)
    for i, t in enumerate(timestamps):
        d = datetime.fromtimestamp(int(t), tz=timezone.utc)
        tod[i] = (d.hour + d.minute / 60.0 + d.second / 3600.0) / 24.0
        dow[i] = d.weekday() / 7.0
    return tod, dow


def parse_service(case_name: str) -> str:
    if "##" in case_name:
        prefix = case_name.split("##", 1)[0]
    else:
        prefix = case_name
    if "_" in prefix:
        # case_name format: "000_dns-resolver"
        parts = prefix.split("_", 1)
        return parts[1] if len(parts) > 1 else prefix
    return prefix


def make_features(
    x: np.ndarray, timestamps: np.ndarray, train_x_ref: np.ndarray,
    info: dict, service: str, top_services: List[str],
) -> np.ndarray:
    """Build the 68-feature matrix for a window of values `x`."""
    n = len(x)
    feats = []

    # 1. raw value
    feats.append(x.astype(np.float64))
    # 2. robust z (global MAD)
    median = float(np.median(train_x_ref))
    mad = float(np.median(np.abs(train_x_ref - median))) + 1e-9
    feats.append((x - median) / (1.4826 * mad))
    # 3. standard z
    mu = float(np.mean(train_x_ref))
    sd = float(np.std(train_x_ref)) + 1e-9
    feats.append((x - mu) / sd)
    # 4. first diff
    d1 = np.diff(x, prepend=x[0])
    feats.append(d1)
    # 5. second diff
    d2 = np.diff(x, n=2, prepend=[x[0], x[0]])
    feats.append(d2)

    # 6-11. rolling mean/std at w=5,11,21 (6 features)
    for w in (5, 11, 21):
        m, s = _rolling_mean_std(x, w)
        feats.append(m)
        feats.append(s)

    # 12-13. rolling median + MAD at w=11
    rmed11, rmad11 = _rolling_median_mad(x, 11)
    feats.append(rmed11)
    feats.append(rmad11)
    # 14. deviation from rolling median
    feats.append(x - rmed11)
    # 15. z vs rolling MAD
    feats.append((x - rmed11) / (1.4826 * rmad11 + 1e-9))

    # 16. EWMA
    ewma = _ewma(x, alpha=0.3)
    feats.append(ewma)
    # 17. EWMA residual
    res = x - ewma
    feats.append(res)
    # 18. EWMA z
    feats.append(res / (np.std(res) + 1e-9))

    # 19. percentile rank vs train ECDF
    feats.append(_percentile_rank_vs(x, train_x_ref))

    # 20. position in window
    feats.append(np.arange(n, dtype=np.float64) / max(1, n - 1))

    # 21. time-of-day, 22. day-of-week
    tod, dow = _time_features(timestamps)
    feats.append(tod)
    feats.append(dow)

    # ─── Long-range (7) ───
    rmean41, rstd41 = _rolling_mean_std(x, 41)
    feats.append(rmean41)
    feats.append(rstd41)
    rmed41, rmad41 = _rolling_median_mad(x, 41)
    feats.append(rmed41)
    feats.append(rmad41)
    _, rs5 = _rolling_mean_std(x, 5)
    feats.append(rs5 / (rstd41 + 1e-9))  # vol_ratio
    rmin11, rmax11 = _rolling_minmax(x, 11)
    feats.append(rmax11 - x)
    feats.append(x - rmin11)

    # ─── Static window-level (39) — broadcast across all points ───
    static = []
    mt = info.get("metric_type", "Unknown")
    for m in METRIC_TYPES:
        static.append(1.0 if mt == m else 0.0)
    # Service one-hot (top-30)
    for ts in top_services:
        static.append(1.0 if service == ts else 0.0)
    # Pad to 30 if fewer top services
    static += [0.0] * (TOP_K_SERVICES - len(top_services))
    # intervals in hours
    interval_h = float(info.get("intervals", 0)) / 3600.0
    static.append(interval_h)
    # train and test anomaly ratios
    static.append(float(info.get("training set anomaly ratio", 0.0)))
    static.append(float(info.get("test set anomaly ratio", 0.0)))

    static_arr = np.array(static, dtype=np.float64)
    feats_static = np.tile(static_arr, (n, 1))

    point_feats = np.column_stack(feats)
    return np.hstack([point_feats, feats_static]).astype(np.float32)


# ─────────────────────────────────────────────
# Top-30 services from corpus
# ─────────────────────────────────────────────


def compute_top_services(window_dirs, k: int = TOP_K_SERVICES) -> List[str]:
    counts = Counter()
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        counts[parse_service(info.get("case_name", ""))] += 1
    return [s for s, _ in counts.most_common(k)]


# ─────────────────────────────────────────────
# Per-metric-type ensemble training
# ─────────────────────────────────────────────


def _build_pool_for_metric(window_dirs, top_services: List[str], target_mt: str
                           ) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        if info.get("metric_type") != target_mt:
            continue
        try:
            train_y = np.load(wdir / "train_label.npy")
        except FileNotFoundError:
            continue
        if train_y.sum() == 0:
            continue
        train_x = np.load(wdir / "train.npy")
        try:
            train_ts = np.load(wdir / "train_timestamp.npy")
        except FileNotFoundError:
            train_ts = np.arange(len(train_x), dtype=np.int64) * info.get("intervals", 60)
        service = parse_service(info.get("case_name", ""))
        feats = make_features(train_x, train_ts, train_x, info, service, top_services)
        Xs.append(feats)
        ys.append(train_y)
    if not Xs:
        return np.zeros((0, 68), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.vstack(Xs), np.hstack(ys)


def fit_per_metric_ensemble(window_dirs, top_services: List[str]
                            ) -> Dict[str, dict]:
    """For each metric_type, train 3 RF + 5 HGBT + 1 LR."""
    models_by_mt: Dict[str, dict] = {}
    for mt in METRIC_TYPES:
        print(f"  [{mt}] building pool…", flush=True)
        t0 = time.time()
        X, y = _build_pool_for_metric(window_dirs, top_services, mt)
        if len(X) < 100:
            print(f"    SKIP — too few samples ({len(X)})")
            continue
        print(f"    pool X={X.shape}  pos_rate={y.mean():.3f}  build={time.time() - t0:.1f}s")

        # Standardize features for LR
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        # 3 RFs (seeds 0,1,2)
        t0 = time.time()
        rfs = []
        for s in (0, 1, 2):
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_leaf=10,
                class_weight="balanced", random_state=s, n_jobs=4,
            )
            rf.fit(X, y)
            rfs.append(rf)
        print(f"    3-seed RF fit {time.time() - t0:.1f}s")

        # 5 HGBTs (seeds 0..4)
        t0 = time.time()
        hgbts = []
        for s in range(5):
            hgbt = HistGradientBoostingClassifier(
                max_iter=200, max_depth=8, learning_rate=0.05,
                min_samples_leaf=20, random_state=s, class_weight="balanced",
            )
            hgbt.fit(X, y)
            hgbts.append(hgbt)
        print(f"    5-seed HGBT fit {time.time() - t0:.1f}s")

        # 1 LR
        t0 = time.time()
        lr = LogisticRegression(
            C=0.5, max_iter=500, class_weight="balanced", solver="lbfgs", n_jobs=4,
        )
        lr.fit(X_scaled, y)
        print(f"    1 LR fit {time.time() - t0:.1f}s")

        models_by_mt[mt] = {"rfs": rfs, "hgbts": hgbts, "lr": lr, "scaler": scaler}
    return models_by_mt


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────


def predict_proba_window(
    test_x: np.ndarray, test_ts: np.ndarray, train_x_ref: np.ndarray,
    info: dict, service: str, top_services: List[str], models_by_mt: Dict[str, dict],
) -> np.ndarray:
    mt = info.get("metric_type", "Unknown")
    if mt not in models_by_mt:
        # Fall back to first available metric type's models
        mt = next(iter(models_by_mt))
    bundle = models_by_mt[mt]

    X = make_features(test_x, test_ts, train_x_ref, info, service, top_services)
    X_scaled = bundle["scaler"].transform(X)

    rf_avg = np.mean([clf.predict_proba(X)[:, 1] for clf in bundle["rfs"]], axis=0)
    hgbt_avg = np.mean([clf.predict_proba(X)[:, 1] for clf in bundle["hgbts"]], axis=0)
    lr_p = bundle["lr"].predict_proba(X_scaled)[:, 1]

    prob_mean = 0.80 * hgbt_avg + 0.10 * rf_avg + 0.10 * lr_p
    return prob_mean


def smooth_centered(p: np.ndarray, w: int = SMOOTH_W) -> np.ndarray:
    if w <= 1:
        return p.copy()
    kernel = np.ones(w) / w
    # Reflect padding for edges
    half = w // 2
    padded = np.concatenate([
        p[half - 1 :: -1] if half > 0 else p[:0],
        p,
        p[-1 : -half - 1 : -1] if half > 0 else p[:0],
    ])
    out = np.convolve(padded, kernel, mode="valid")
    if len(out) > len(p):
        out = out[: len(p)]
    elif len(out) < len(p):
        out = np.convolve(p, kernel, mode="same")
    return out


def predict_window(test_x, test_ts, train_x_ref, info, service, top_services,
                   models_by_mt) -> np.ndarray:
    n = len(test_x)
    ratio = float(info.get("test set anomaly ratio", 0.0))
    k = max(0, min(int(round(n * ratio)), n))
    if k == 0:
        return np.zeros(n, dtype=int)

    prob_mean = predict_proba_window(test_x, test_ts, train_x_ref, info, service,
                                     top_services, models_by_mt)
    rm = smooth_centered(prob_mean, SMOOTH_W)
    prob_final = (1.0 - SMOOTH_ALPHA) * prob_mean + SMOOTH_ALPHA * rm

    # Plain top-k via argpartition (tie-break: lower index first)
    order = np.lexsort((np.arange(n), -prob_final))
    top_idx = order[:k]
    pred = np.zeros(n, dtype=int)
    pred[top_idx] = 1
    return pred


# ─────────────────────────────────────────────
# Validation + submission
# ─────────────────────────────────────────────


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Computing top-30 services from train_pool…")
    top_services = compute_top_services(train_pool, k=TOP_K_SERVICES)
    print(f"    top services: {top_services[:5]} … ({len(top_services)} total)")

    print(">>> Fitting per-metric-type ensemble on train_pool…")
    t0 = time.time()
    models_by_mt = fit_per_metric_ensemble(train_pool, top_services)
    print(f"    total fit time {time.time() - t0:.1f}s")

    # CROSS-WINDOW LOO evaluation: predict on holdout windows' TRAIN_X (which the
    # models have NOT seen as training data). This is closer to LB than time-split.
    def predictor(window):
        # Use the window's own train_x as both the data and as the reference for
        # percentile-rank/z (we have it at inference time too)
        try:
            train_ts = np.load(window.wdir / "train_timestamp.npy")
        except FileNotFoundError:
            train_ts = np.arange(len(window.train_x), dtype=np.int64) * window.info.get("intervals", 60)
        service = parse_service(window.info.get("case_name", ""))
        return predict_window(window.train_x, train_ts, window.train_x, window.info,
                              service, top_services, models_by_mt)

    print(">>> Cross-window LOO evaluation on holdout train_x…")
    rep = cross_window_evaluate(predictor, holdout)
    print_summary_v2(rep, "v38 friend-repro (CW-LOO)")

    from validation import save_report
    save_report(rep, "v38_friend_repro_loo")
    return rep, models_by_mt, top_services


def generate_submission(models_by_mt: Dict[str, dict], top_services: List[str],
                        output: Path = Path("submission_friend_repro.json")) -> Path:
    print(f"\n>>> Generating predictions on all 1000 test windows…")
    preds: Dict[str, list] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        try:
            test_ts = np.load(wdir / "test_timestamp.npy")
        except FileNotFoundError:
            test_ts = np.arange(len(w.test_x), dtype=np.int64) * w.info.get("intervals", 60)
        service = parse_service(w.info.get("case_name", ""))
        pred = predict_window(w.test_x, test_ts, w.train_x, w.info, service,
                              top_services, models_by_mt)
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
    rep, models_by_mt, top_services = run_validation()
    # Re-train on ALL 1000 windows for the actual submission
    print("\n>>> Re-training on ALL 1000 windows for final submission…")
    t0 = time.time()
    top_services_full = compute_top_services(all_window_dirs(), k=TOP_K_SERVICES)
    models_full = fit_per_metric_ensemble(all_window_dirs(), top_services_full)
    print(f"    full fit {time.time() - t0:.1f}s")
    generate_submission(models_full, top_services_full)
