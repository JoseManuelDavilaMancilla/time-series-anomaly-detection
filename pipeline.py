"""
author v59 — LightGBM + multi-round pseudo-labeling.

Two big bets combined:
  1. Replace sklearn HGBT with LightGBM (typically stronger on tabular imbalanced data)
  2. Compress 8 rounds of pseudo-label iteration into ONE submission slot.
     Each round: fit → predict all test windows → new pseudo-labels → refit.
     Previously: +0.001–0.003 per slot. Now 8 rounds in 1 slot.

Changes vs v58:
  - 5x LGBMClassifier (n_estimators=400, num_leaves=63) replaces 5x HGBT
  - PSEUDO_WEIGHT = 0.60 (was 0.50) — labels are high-quality after 6 iterations
  - N_ROUNDS = 8 internal pseudo-label iterations
  - Seed pseudo-labels: submission_v58_global_ctx.json (LB 0.6847)
  - No LOO validation: both changes are architecturally sound per NOTES_FOR_KIMI.md rule

Run:  uv run python v59_lgbm_multirounds.py
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
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

from validation import all_window_dirs, load_window
from validation import stratified_holdout, point_f1


METRIC_TYPES = ("Count", "ErrorCount", "LatencySecond", "QPS",
                "ResourceUtilizationRate", "SuccessRate")
TOP_K_SERVICES = 30
SMOOTH_W = 5
SMOOTH_ALPHA = 0.8
W_SHIFT = 0.30
SPLIT_FRAC = 0.70
N_FEATS_P1 = 77
N_FEATS_P2 = 84
PSEUDO_WEIGHT = 0.60          # up from 0.50 — labels are high-quality after v58
PSEUDO_SOURCE = Path("submission_v58_global_ctx.json")
N_ROUNDS = 8                  # internal pseudo-label rounds


# ─────────────────────────────────────────────
# Global context: cross-window population stats
# ─────────────────────────────────────────────

def compute_window_global_stats(window_dirs, data_file: str = "train.npy") -> Dict[Path, Tuple[float, float, float]]:
    raw: Dict[Path, Tuple[str, float, float, float]] = {}
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        mt = info.get("metric_type", "Unknown")
        try:
            x = np.load(wdir / data_file).astype(np.float64)
        except FileNotFoundError:
            x = np.load(wdir / "train.npy").astype(np.float64)
        raw[wdir] = (mt, float(np.mean(x)), float(np.std(x)), float(np.max(x)))

    by_mt: Dict[str, List] = defaultdict(list)
    for wdir, (mt, m, s, mx) in raw.items():
        by_mt[mt].append((wdir, m, s, mx))

    result: Dict[Path, Tuple[float, float, float]] = {}
    for mt, entries in by_mt.items():
        wdirs = [e[0] for e in entries]
        means = np.array([e[1] for e in entries])
        stds  = np.array([e[2] for e in entries])
        maxs  = np.array([e[3] for e in entries])

        def _zscore(arr: np.ndarray) -> np.ndarray:
            mu, sigma = arr.mean(), arr.std()
            return np.clip((arr - mu) / (sigma + 1e-9), -5.0, 5.0)

        mz = _zscore(means)
        sz = _zscore(stds)
        xz = _zscore(maxs)
        for i, wdir in enumerate(wdirs):
            result[wdir] = (float(mz[i]), float(sz[i]), float(xz[i]))

    return result


# ─────────────────────────────────────────────
# Rolling helpers — all centered
# ─────────────────────────────────────────────

def _rolling_mean_std(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    s = pd.Series(x.astype(np.float64))
    r = s.rolling(w, center=True, min_periods=1)
    return r.mean().to_numpy(), r.std(ddof=0).fillna(0.0).to_numpy()


def _rolling_minmax(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    s = pd.Series(x.astype(np.float64))
    r = s.rolling(w, center=True, min_periods=1)
    return r.min().to_numpy(), r.max().to_numpy()


def _rolling_median_mad(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    half = w // 2
    n = len(x)
    med = np.empty(n)
    mad = np.empty(n)
    for i in range(n):
        start, end = max(0, i - half), min(n, i + half + 1)
        seg = x[start:end]
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
    ref_sorted = np.sort(reference)
    n_ref = len(ref_sorted)
    if n_ref == 0:
        return np.full(len(values), 0.5)
    idx = np.searchsorted(ref_sorted, values, side="right")
    return idx.astype(np.float64) / n_ref


def _time_features(timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        parts = prefix.split("_", 1)
        return parts[1] if len(parts) > 1 else prefix
    return prefix


# ─────────────────────────────────────────────
# Feature builders (identical to v58)
# ─────────────────────────────────────────────

def make_features(
    x: np.ndarray, timestamps: np.ndarray, train_x_ref: np.ndarray,
    info: dict, service: str, top_services: List[str],
    win_global: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """77-feature P1 matrix (identical to v58)."""
    n = len(x)
    feats = []

    median = float(np.median(train_x_ref))
    mad = float(np.median(np.abs(train_x_ref - median))) + 1e-9
    mu = float(np.mean(train_x_ref))
    sd = float(np.std(train_x_ref)) + 1e-9

    feats.append(x.astype(np.float64))
    feats.append((x - median) / (1.4826 * mad))
    feats.append((x - mu) / sd)
    feats.append(np.diff(x, prepend=x[0]))
    feats.append(np.diff(x, n=2, prepend=[x[0], x[0]]))

    for w in (5, 11, 21):
        m, s = _rolling_mean_std(x, w)
        feats.append(m)
        feats.append(s)

    rmed11, rmad11 = _rolling_median_mad(x, 11)
    feats.append(rmed11)
    feats.append(rmad11)
    feats.append(x - rmed11)
    feats.append((x - rmed11) / (1.4826 * rmad11 + 1e-9))

    ewma = _ewma(x, alpha=0.3)
    res = x - ewma
    feats.append(ewma)
    feats.append(res)
    feats.append(res / (np.std(res) + 1e-9))

    feats.append(_percentile_rank_vs(x, train_x_ref))
    feats.append(np.arange(n, dtype=np.float64) / max(1, n - 1))

    tod, dow = _time_features(timestamps)
    feats.append(tod)
    feats.append(dow)

    rmean41, rstd41 = _rolling_mean_std(x, 41)
    feats.append(rmean41)
    feats.append(rstd41)
    rmed41, rmad41 = _rolling_median_mad(x, 41)
    feats.append(rmed41)
    feats.append(rmad41)
    _, rs5 = _rolling_mean_std(x, 5)
    feats.append(rs5 / (rstd41 + 1e-9))
    rmin11, rmax11 = _rolling_minmax(x, 11)
    feats.append(rmax11 - x)
    feats.append(x - rmin11)

    for w_mm in (5, 21, 41):
        rmin_w, rmax_w = _rolling_minmax(x, w_mm)
        feats.append(rmax_w - x)
        feats.append(x - rmin_w)

    mean_z, std_z, max_z = win_global
    feats.append(np.full(n, mean_z, dtype=np.float64))
    feats.append(np.full(n, std_z,  dtype=np.float64))
    feats.append(np.full(n, max_z,  dtype=np.float64))

    static = []
    mt = info.get("metric_type", "Unknown")
    for m in METRIC_TYPES:
        static.append(1.0 if mt == m else 0.0)
    for ts in top_services:
        static.append(1.0 if service == ts else 0.0)
    static += [0.0] * (TOP_K_SERVICES - len(top_services))
    static.append(float(info.get("intervals", 0)) / 3600.0)
    static.append(float(info.get("training set anomaly ratio", 0.0)))
    static.append(float(info.get("test set anomaly ratio", 0.0)))

    static_arr = np.array(static, dtype=np.float64)
    feats_static = np.tile(static_arr, (n, 1))
    point_feats = np.column_stack(feats)
    return np.hstack([point_feats, feats_static]).astype(np.float32)


def make_features_shift(
    x: np.ndarray, timestamps: np.ndarray, ref_x: np.ndarray,
    info: dict, service: str, top_services: List[str],
    win_global: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """84-feature P2 matrix (identical to v58)."""
    base = make_features(x, timestamps, ref_x, info, service, top_services, win_global)
    n = len(x)

    x_med = float(np.median(x))
    x_mad = float(np.median(np.abs(x - x_med))) + 1e-9
    ref_std = float(np.std(ref_x)) + 1e-9

    rank_in_self    = _percentile_rank_vs(x, x)
    self_robust_z   = (x - x_med) / (1.4826 * x_mad)
    above_ref_max   = np.maximum(0.0, x - float(np.max(ref_x)))
    below_ref_min   = np.maximum(0.0, float(np.min(ref_x)) - x)
    mean_shift_bc   = np.full(n, float(np.mean(x)) - float(np.mean(ref_x)))
    std_ratio_bc    = np.full(n, float(np.std(x)) / ref_std)
    median_shift_bc = np.full(n, float(np.median(x)) - float(np.median(ref_x)))

    shift_feats = np.column_stack([
        rank_in_self, self_robust_z, above_ref_max, below_ref_min,
        mean_shift_bc, std_ratio_bc, median_shift_bc,
    ]).astype(np.float32)

    return np.hstack([base, shift_feats])


# ─────────────────────────────────────────────
# Top-30 services
# ─────────────────────────────────────────────

def compute_top_services(window_dirs, k: int = TOP_K_SERVICES) -> List[str]:
    counts = Counter()
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        counts[parse_service(info.get("case_name", ""))] += 1
    return [s for s, _ in counts.most_common(k)]


# ─────────────────────────────────────────────
# Pseudo-label helpers
# ─────────────────────────────────────────────

def load_pseudo_labels(path: Path) -> Dict[str, np.ndarray]:
    data = json.loads(path.read_text())
    return {wid: np.array(v, dtype=np.int64)
            for wid, v in data["predictions"].items()}


def build_wid_map(window_dirs) -> Dict[str, Path]:
    return {wdir.name.split("_", 1)[0]: wdir for wdir in window_dirs}


def _load_test_arrays(wdir: Path, info: dict):
    test_x = np.load(wdir / "test.npy")
    try:
        test_ts = np.load(wdir / "test_timestamp.npy")
    except FileNotFoundError:
        test_ts = np.arange(len(test_x), dtype=np.int64) * info.get("intervals", 60)
    return test_x, test_ts


# ─────────────────────────────────────────────
# Training pool builders (identical to v58)
# ─────────────────────────────────────────────

def _build_pool_p1(window_dirs, top_services, target_mt,
                   train_global_stats, test_global_stats,
                   pseudo_labels=None, wid_map=None):
    Xs, ys, sample_ws = [], [], []

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
        wg = train_global_stats.get(wdir, (0.0, 0.0, 0.0))
        feats = make_features(train_x, train_ts, train_x, info, service, top_services, wg)
        Xs.append(feats)
        ys.append(train_y)
        sample_ws.append(np.ones(len(train_y), dtype=np.float32))

    if pseudo_labels and wid_map:
        for wid, pseudo_y in pseudo_labels.items():
            if pseudo_y.sum() == 0:
                continue
            wdir = wid_map.get(wid)
            if wdir is None:
                continue
            info = json.loads((wdir / "info.json").read_text())
            if info.get("metric_type") != target_mt:
                continue
            train_x = np.load(wdir / "train.npy")
            test_x, test_ts = _load_test_arrays(wdir, info)
            if len(test_x) != len(pseudo_y):
                continue
            service = parse_service(info.get("case_name", ""))
            wg = test_global_stats.get(wdir, (0.0, 0.0, 0.0))
            feats = make_features(test_x, test_ts, train_x, info, service, top_services, wg)
            Xs.append(feats)
            ys.append(pseudo_y)
            sample_ws.append(np.full(len(pseudo_y), PSEUDO_WEIGHT, dtype=np.float32))

    if not Xs:
        return (np.zeros((0, N_FEATS_P1), np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.float32))
    return np.vstack(Xs), np.hstack(ys), np.hstack(sample_ws)


def _build_pool_p2(window_dirs, top_services, target_mt,
                   train_global_stats, test_global_stats,
                   split_frac=SPLIT_FRAC, pseudo_labels=None, wid_map=None):
    Xs, ys, sample_ws = [], [], []

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
        n = len(train_x)
        cut = max(10, int(n * split_frac))
        if n - cut < 5:
            continue
        pseudo_y_split = train_y[cut:]
        if pseudo_y_split.sum() == 0:
            continue
        ref_x = train_x[:cut]
        pseudo_x = train_x[cut:]
        try:
            train_ts = np.load(wdir / "train_timestamp.npy")
        except FileNotFoundError:
            train_ts = np.arange(n, dtype=np.int64) * info.get("intervals", 60)
        pseudo_ts = train_ts[cut:]
        service = parse_service(info.get("case_name", ""))
        wg = train_global_stats.get(wdir, (0.0, 0.0, 0.0))
        feats = make_features_shift(pseudo_x, pseudo_ts, ref_x, info, service, top_services, wg)
        Xs.append(feats)
        ys.append(pseudo_y_split)
        sample_ws.append(np.ones(len(pseudo_y_split), dtype=np.float32))

    if pseudo_labels and wid_map:
        for wid, pseudo_y in pseudo_labels.items():
            if pseudo_y.sum() == 0:
                continue
            wdir = wid_map.get(wid)
            if wdir is None:
                continue
            info = json.loads((wdir / "info.json").read_text())
            if info.get("metric_type") != target_mt:
                continue
            train_x = np.load(wdir / "train.npy")
            cut = max(1, int(len(train_x) * split_frac))
            ref_x = train_x[:cut]
            test_x, test_ts = _load_test_arrays(wdir, info)
            if len(test_x) != len(pseudo_y) or len(test_x) < 5:
                continue
            service = parse_service(info.get("case_name", ""))
            wg = test_global_stats.get(wdir, (0.0, 0.0, 0.0))
            feats = make_features_shift(test_x, test_ts, ref_x, info, service, top_services, wg)
            Xs.append(feats)
            ys.append(pseudo_y)
            sample_ws.append(np.full(len(pseudo_y), PSEUDO_WEIGHT, dtype=np.float32))

    if not Xs:
        return (np.zeros((0, N_FEATS_P2), np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.float32))
    return np.vstack(Xs), np.hstack(ys), np.hstack(sample_ws)


# ─────────────────────────────────────────────
# Ensemble fitting — LightGBM replaces HGBT
# ─────────────────────────────────────────────

def _fit_models(X: np.ndarray, y: np.ndarray, label: str,
                sample_weight: np.ndarray | None = None) -> dict:
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    t0 = time.time()
    rfs = []
    for s in (0, 1, 2):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            class_weight="balanced", random_state=s, n_jobs=4,
        )
        rf.fit(X, y, sample_weight=sample_weight)
        rfs.append(rf)
    print(f"    {label} 3-seed RF fit {time.time() - t0:.1f}s")

    # LightGBM: replaces sklearn HGBT
    t0 = time.time()
    lgbms = []
    for s in range(5):
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=4,
            verbose=-1,
            random_state=s,
        )
        model.fit(X, y, sample_weight=sample_weight)
        lgbms.append(model)
    print(f"    {label} 5-seed LGB fit {time.time() - t0:.1f}s")

    t0 = time.time()
    lr = LogisticRegression(C=0.5, max_iter=500, class_weight="balanced", solver="lbfgs")
    lr.fit(X_scaled, y, sample_weight=sample_weight)
    print(f"    {label} 1 LR fit {time.time() - t0:.1f}s")

    return {"rfs": rfs, "lgbms": lgbms, "lr": lr, "scaler": scaler}


def fit_both_ensembles(window_dirs, top_services,
                       train_global_stats, test_global_stats,
                       pseudo_labels=None, wid_map=None) -> Dict[str, dict]:
    ensembles: Dict[str, dict] = {}
    for mt in METRIC_TYPES:
        print(f"  [{mt}] building pools…", flush=True)
        t0 = time.time()
        X1, y1, w1 = _build_pool_p1(window_dirs, top_services, mt,
                                     train_global_stats, test_global_stats,
                                     pseudo_labels, wid_map)
        X2, y2, w2 = _build_pool_p2(window_dirs, top_services, mt,
                                     train_global_stats, test_global_stats,
                                     pseudo_labels=pseudo_labels, wid_map=wid_map)

        print(f"    P1 X={X1.shape} pos={y1.mean():.3f}  "
              f"P2 X={X2.shape} pos={y2.mean() if len(y2) else 0:.3f}  "
              f"build={time.time()-t0:.1f}s")

        if len(X1) < 100:
            print(f"    SKIP P1 — too few samples")
            continue

        bundle_p1 = _fit_models(X1, y1, "P1", sample_weight=w1)
        bundle_p2 = None
        if len(X2) >= 50:
            bundle_p2 = _fit_models(X2, y2, "P2", sample_weight=w2)
        else:
            print(f"    SKIP P2 — too few samples ({len(X2)})")

        ensembles[mt] = {"p1": bundle_p1, "p2": bundle_p2}
    return ensembles


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def _score_bundle(bundle: dict, X: np.ndarray) -> np.ndarray:
    X_scaled = bundle["scaler"].transform(X)
    rf_avg   = np.mean([clf.predict_proba(X)[:, 1] for clf in bundle["rfs"]], axis=0)
    lgb_avg  = np.mean([clf.predict_proba(X)[:, 1] for clf in bundle["lgbms"]], axis=0)
    lr_p     = bundle["lr"].predict_proba(X_scaled)[:, 1]
    return 0.80 * lgb_avg + 0.10 * rf_avg + 0.10 * lr_p


def predict_proba_window(
    test_x, test_ts, train_x_ref, info, service, top_services, ensembles,
    win_global=(0.0, 0.0, 0.0),
) -> np.ndarray:
    mt = info.get("metric_type", "Unknown")
    if mt not in ensembles:
        mt = next(iter(ensembles))
    bundle = ensembles[mt]

    X1 = make_features(test_x, test_ts, train_x_ref, info, service, top_services, win_global)
    prob_p1 = _score_bundle(bundle["p1"], X1)

    if bundle["p2"] is not None:
        X2 = make_features_shift(test_x, test_ts, train_x_ref, info, service, top_services, win_global)
        prob_p2 = _score_bundle(bundle["p2"], X2)
        return (1.0 - W_SHIFT) * prob_p1 + W_SHIFT * prob_p2
    return prob_p1


def smooth_centered(p: np.ndarray, w: int = SMOOTH_W) -> np.ndarray:
    if w <= 1:
        return p.copy()
    kernel = np.ones(w) / w
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
                   ensembles, win_global=(0.0, 0.0, 0.0)) -> np.ndarray:
    n = len(test_x)
    ratio = float(info.get("test set anomaly ratio", 0.0))
    k = max(0, min(int(round(n * ratio)), n))
    if k == 0:
        return np.zeros(n, dtype=int)

    prob_mean = predict_proba_window(test_x, test_ts, train_x_ref, info, service,
                                     top_services, ensembles, win_global)
    rm = smooth_centered(prob_mean, SMOOTH_W)
    prob_final = (1.0 - SMOOTH_ALPHA) * prob_mean + SMOOTH_ALPHA * rm

    order = np.lexsort((np.arange(n), -prob_final))
    pred = np.zeros(n, dtype=int)
    pred[order[:k]] = 1
    return pred


# ─────────────────────────────────────────────
# Multi-round pseudo-label loop
# ─────────────────────────────────────────────

def predict_all_test_windows(ensembles, top_services, test_global_stats,
                              window_dirs) -> Dict[str, np.ndarray]:
    """Score all test windows and return binary predictions."""
    preds = {}
    for wdir in window_dirs:
        w = load_window(wdir)
        try:
            test_ts = np.load(wdir / "test_timestamp.npy")
        except FileNotFoundError:
            test_ts = np.arange(len(w.test_x), dtype=np.int64) * w.info.get("intervals", 60)
        service = parse_service(w.info.get("case_name", ""))
        wg = test_global_stats.get(wdir, (0.0, 0.0, 0.0))
        pred = predict_window(w.test_x, test_ts, w.train_x, w.info,
                              service, top_services, ensembles, wg)
        preds[w.wid] = pred
    return preds


def run_multirounds():
    all_wdirs = list(all_window_dirs())
    wid_map = build_wid_map(all_wdirs)

    print(">>> Pre-computing shared stats (done once)…")
    top_services = compute_top_services(all_wdirs, k=TOP_K_SERVICES)
    train_gs = compute_window_global_stats(all_wdirs, data_file="train.npy")
    test_gs  = compute_window_global_stats(all_wdirs, data_file="test.npy")
    print(f"    top_services={len(top_services)}  "
          f"train_gs={len(train_gs)}  test_gs={len(test_gs)}")

    print(f"\n>>> Loading seed pseudo-labels from {PSEUDO_SOURCE}…")
    pseudo_labels = load_pseudo_labels(PSEUDO_SOURCE)
    n_with = sum(1 for v in pseudo_labels.values() if v.sum() > 0)
    print(f"    {len(pseudo_labels)} windows, {n_with} with predicted anomalies (seed)")

    final_preds: Dict[str, np.ndarray] = {}
    t_total = time.time()

    for round_i in range(N_ROUNDS):
        t_round = time.time()
        n_pseudo = sum(1 for v in pseudo_labels.values() if v.sum() > 0)
        print(f"\n{'='*60}")
        print(f"  ROUND {round_i + 1}/{N_ROUNDS}  —  pseudo-labeled windows: {n_pseudo}")
        print(f"{'='*60}")

        ensembles = fit_both_ensembles(
            all_wdirs, top_services, train_gs, test_gs,
            pseudo_labels, wid_map,
        )

        print(f"  Predicting all 1000 test windows…")
        preds = predict_all_test_windows(ensembles, top_services, test_gs, all_wdirs)
        final_preds = preds

        n_new = sum(1 for v in preds.values() if v.sum() > 0)
        n_changed = sum(
            1 for wid, v in preds.items()
            if wid in pseudo_labels and not np.array_equal(v, pseudo_labels[wid])
        )
        print(f"  Round {round_i + 1} done in {time.time() - t_round:.0f}s  "
              f"|  windows-with-anomalies: {n_new}  |  changed: {n_changed}")

        # Update pseudo-labels for next round
        pseudo_labels = {wid: arr.astype(np.int64) for wid, arr in preds.items()}

    print(f"\n>>> All {N_ROUNDS} rounds done in {time.time() - t_total:.0f}s total")
    return final_preds


# ─────────────────────────────────────────────
# Write submission
# ─────────────────────────────────────────────

def write_submission(preds: Dict[str, np.ndarray],
                     output: Path = Path("submission_v59_lgbm_multirounds.json")) -> Path:
    out = {wid: arr.astype(int).tolist() for wid, arr in preds.items()}
    assert len(out) == 1000
    output.write_text(
        json.dumps({"predictions": out}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f">>> Wrote {output}  ({len(out)} windows)")
    return output


if __name__ == "__main__":
    print(f"LightGBM {lgb.__version__}")
    print(f"N_ROUNDS={N_ROUNDS}  PSEUDO_WEIGHT={PSEUDO_WEIGHT}  W_SHIFT={W_SHIFT}")
    print(f"Seed: {PSEUDO_SOURCE}\n")

    final_preds = run_multirounds()
    write_submission(final_preds)
    print("\nDone. Submit submission_v59_lgbm_multirounds.json")
