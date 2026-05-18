"""
author v61 — Persistent homology (TDA) features.

Approach: Takens delay embedding (dim=2, lag=1) converts each 1D time series
into a 2D point cloud, then ripser computes H0 and H1 persistence diagrams.

10 new window-level features (broadcast to all points):
  H0 (connected components / gaps):
    test_h0_max        — largest gap in the test point cloud (outlier signal)
    test_h0_sum        — total H0 persistence (overall spread)
    test_h0_entropy    — persistence entropy (how concentrated the gaps are)
  H1 (loops / cycles):
    test_h1_max        — most persistent cycle (periodicity signal)
    test_h1_sum        — total loop structure
    test_h1_n_sig      — number of significant cycles (> max/4)
  Train reference (context):
    train_h0_max       — train's gap structure (scale reference)
    train_h1_max       — train's periodicity (context)
  Topological shift (train→test):
    h0_bottleneck_dist — bottleneck distance between train and test H0 diagrams
    h1_bottleneck_dist — same for H1 (topological distribution shift)

Why this matters:
  - H1 max captures periodicity destruction: a normal QPS window has strong
    cycles; an anomalous window that breaks the pattern has weaker H1.
  - H0 max captures large outliers: a spike creates an isolated point far
    from the main cluster → large H0 lifetime.
  - Bottleneck distance is a direct "topological shift" signal that no
    rolling or FFT feature can represent.

Stacked on v60 (FFT features):
  N_FEATS_P1: 82 → 92   (+10 TDA)
  N_FEATS_P2: 89 → 99   (+10 TDA)

Pseudo-labels from submission_v60_fft_features.json (LB 0.6859, current best).
PSEUDO_WEIGHT = 0.50 (unchanged).

Run:  uv run python v61_tda.py
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
from ripser import ripser
from persim import bottleneck as persim_bottleneck
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
SMOOTH_W       = 5
SMOOTH_ALPHA   = 0.8
W_SHIFT        = 0.30
SPLIT_FRAC     = 0.70
N_TDA_FEATS    = 10
N_FFT_FEATS    = 5
N_FEATS_P1     = 77 + N_FFT_FEATS + N_TDA_FEATS   # 92
N_FEATS_P2     = 84 + N_FFT_FEATS + N_TDA_FEATS   # 99
PSEUDO_WEIGHT  = 0.50
PSEUDO_SOURCE  = Path("submission_v60_fft_features.json")   # LB 0.6859
TDA_MAX_PTS    = 300    # subsample point cloud for speed
TDA_DIM        = 2      # Takens embedding dimension
TDA_LAG        = 1      # Takens lag


# ─────────────────────────────────────────────
# TDA helpers
# ─────────────────────────────────────────────

def _takens_embed(x: np.ndarray, dim: int = TDA_DIM, lag: int = TDA_LAG) -> np.ndarray:
    n = len(x) - (dim - 1) * lag
    return np.stack([x[i * lag: n + i * lag] for i in range(dim)], axis=1)


def _persistence_entropy(lifetimes: np.ndarray) -> float:
    total = lifetimes.sum()
    if total < 1e-10:
        return 0.0
    p = lifetimes / total
    return float(-np.sum(p * np.log(p + 1e-9)))


def _prepare_cloud(x: np.ndarray) -> np.ndarray:
    """Subsample, embed, and normalise to unit std."""
    if len(x) > TDA_MAX_PTS:
        idx = np.linspace(0, len(x) - 1, TDA_MAX_PTS, dtype=int)
        x = x[idx]
    cloud = _takens_embed(x.astype(np.float64))
    cloud -= cloud.mean(axis=0)
    scale = cloud.std() + 1e-9
    return cloud / scale


def _extract_diagram_features(dgm: np.ndarray, hdim: int) -> Tuple[float, float, float, int]:
    """max_lifetime, sum_lifetimes, entropy, n_significant."""
    if hdim == 0:
        dgm = dgm[dgm[:, 1] < np.inf]   # remove the one infinite H0 bar
    if len(dgm) == 0:
        return 0.0, 0.0, 0.0, 0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    max_l = float(lifetimes.max())
    return (max_l,
            float(lifetimes.sum()),
            _persistence_entropy(lifetimes),
            int((lifetimes > max_l / 4 + 1e-10).sum()))


def _safe_bottleneck(dgm_a: np.ndarray, dgm_b: np.ndarray, hdim: int) -> float:
    """Bottleneck distance between two persistence diagrams; 0 on failure."""
    if hdim == 0:
        dgm_a = dgm_a[dgm_a[:, 1] < np.inf]
        dgm_b = dgm_b[dgm_b[:, 1] < np.inf]
    if len(dgm_a) == 0 and len(dgm_b) == 0:
        return 0.0
    # persim.bottleneck requires at least one non-empty diagram
    if len(dgm_a) == 0:
        dgm_a = np.zeros((1, 2))
    if len(dgm_b) == 0:
        dgm_b = np.zeros((1, 2))
    try:
        return float(persim_bottleneck(dgm_a, dgm_b))
    except Exception:
        return 0.0


def tda_window_features(test_x: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    """
    Compute 10 TDA features for a window and return shape-(10,) float32 array.
    All features are window-level scalars (broadcast to every point).
    """
    test_cloud  = _prepare_cloud(test_x)
    train_cloud = _prepare_cloud(train_x)

    test_dgms  = ripser(test_cloud,  maxdim=1)["dgms"]
    train_dgms = ripser(train_cloud, maxdim=1)["dgms"]

    t_h0_max, t_h0_sum, t_h0_ent, _      = _extract_diagram_features(test_dgms[0],  0)
    t_h1_max, t_h1_sum, _,        t_h1_n = _extract_diagram_features(test_dgms[1],  1)
    r_h0_max, *_                          = _extract_diagram_features(train_dgms[0], 0)
    r_h1_max, *_                          = _extract_diagram_features(train_dgms[1], 1)

    h0_bn = _safe_bottleneck(test_dgms[0], train_dgms[0], 0)
    h1_bn = _safe_bottleneck(test_dgms[1], train_dgms[1], 1)

    return np.array([
        t_h0_max, t_h0_sum, t_h0_ent,
        t_h1_max, t_h1_sum, float(t_h1_n),
        r_h0_max, r_h1_max,
        h0_bn,    h1_bn,
    ], dtype=np.float32)


# ─────────────────────────────────────────────
# FFT helpers (identical to v60)
# ─────────────────────────────────────────────

def _fft_reconstruct(x: np.ndarray, n_keep: int) -> np.ndarray:
    n = len(x)
    if n < 8:
        return np.full(n, np.mean(x))
    X = np.fft.rfft(x)
    top = np.argsort(np.abs(X)[1:])[::-1][:n_keep] + 1
    mask = np.zeros(len(X), dtype=bool)
    mask[0] = True
    mask[top] = True
    return np.fft.irfft(X * mask, n=n)


def _fft_anomaly_features(test_x: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    n = len(test_x)
    n_keep = max(1, min(10, n // 4))
    test_recon = _fft_reconstruct(test_x.astype(np.float64), n_keep)
    test_res   = test_x.astype(np.float64) - test_recon
    res_med  = float(np.median(test_res))
    res_mad  = float(np.median(np.abs(test_res - res_med))) + 1e-9
    test_res_z = np.clip((test_res - res_med) / (1.4826 * res_mad), -10, 10)

    n_keep_tr   = max(1, min(10, len(train_x) // 4))
    train_recon = _fft_reconstruct(train_x.astype(np.float64), n_keep_tr)
    train_res   = train_x.astype(np.float64) - train_recon
    train_res_std = float(np.std(train_res)) + 1e-9
    train_var  = float(np.var(train_x)) + 1e-9
    recon_var  = float(np.var(train_recon))
    periodicity_strength = float(np.clip(recon_var / train_var, 0., 1.))

    test_res_vs_train = np.clip(test_res / train_res_std, -10, 10)
    return np.column_stack([
        test_res, test_res_z, test_res_vs_train,
        np.full(n, periodicity_strength),
        np.full(n, np.log1p(train_res_std)),
    ]).astype(np.float32)


# ─────────────────────────────────────────────
# Global context
# ─────────────────────────────────────────────

def compute_window_global_stats(window_dirs, data_file="train.npy"):
    raw = {}
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        mt = info.get("metric_type", "Unknown")
        try:
            x = np.load(wdir / data_file).astype(np.float64)
        except FileNotFoundError:
            x = np.load(wdir / "train.npy").astype(np.float64)
        raw[wdir] = (mt, float(np.mean(x)), float(np.std(x)), float(np.max(x)))

    by_mt = defaultdict(list)
    for wdir, (mt, m, s, mx) in raw.items():
        by_mt[mt].append((wdir, m, s, mx))

    result = {}
    for mt, entries in by_mt.items():
        wdirs = [e[0] for e in entries]
        means = np.array([e[1] for e in entries])
        stds  = np.array([e[2] for e in entries])
        maxs  = np.array([e[3] for e in entries])

        def _z(arr):
            mu, sigma = arr.mean(), arr.std()
            return np.clip((arr - mu) / (sigma + 1e-9), -5., 5.)

        for i, wdir in enumerate(wdirs):
            result[wdir] = (float(_z(means)[i]), float(_z(stds)[i]), float(_z(maxs)[i]))
    return result


# ─────────────────────────────────────────────
# Rolling helpers (identical to v58/v60)
# ─────────────────────────────────────────────

def _rolling_mean_std(x, w):
    import pandas as pd
    s = pd.Series(x.astype(np.float64))
    r = s.rolling(w, center=True, min_periods=1)
    return r.mean().to_numpy(), r.std(ddof=0).fillna(0.).to_numpy()

def _rolling_minmax(x, w):
    import pandas as pd
    s = pd.Series(x.astype(np.float64))
    r = s.rolling(w, center=True, min_periods=1)
    return r.min().to_numpy(), r.max().to_numpy()

def _rolling_median_mad(x, w):
    half = w // 2; n = len(x)
    med = np.empty(n); mad = np.empty(n)
    for i in range(n):
        seg = x[max(0, i-half):min(n, i+half+1)]
        m = np.median(seg); med[i] = m; mad[i] = np.median(np.abs(seg - m))
    return med, mad

def _ewma(x, alpha=0.3):
    out = np.empty_like(x, dtype=np.float64); e = float(x[0])
    for i, v in enumerate(x):
        e = alpha * float(v) + (1 - alpha) * e; out[i] = e
    return out

def _percentile_rank_vs(values, reference):
    ref_sorted = np.sort(reference); n_ref = len(ref_sorted)
    if n_ref == 0: return np.full(len(values), 0.5)
    return np.searchsorted(ref_sorted, values, side="right").astype(np.float64) / n_ref

def _time_features(timestamps):
    n = len(timestamps); tod = np.empty(n); dow = np.empty(n)
    for i, t in enumerate(timestamps):
        d = datetime.fromtimestamp(int(t), tz=timezone.utc)
        tod[i] = (d.hour + d.minute/60 + d.second/3600) / 24
        dow[i] = d.weekday() / 7.
    return tod, dow

def parse_service(case_name):
    prefix = case_name.split("##", 1)[0] if "##" in case_name else case_name
    parts = prefix.split("_", 1)
    return parts[1] if len(parts) > 1 and "_" in prefix else prefix


# ─────────────────────────────────────────────
# Feature builders (77 base + 5 FFT + 10 TDA)
# ─────────────────────────────────────────────

def _base77(x, timestamps, train_x_ref, info, service, top_services, win_global):
    n = len(x); feats = []
    median = float(np.median(train_x_ref))
    mad    = float(np.median(np.abs(train_x_ref - median))) + 1e-9
    mu     = float(np.mean(train_x_ref))
    sd     = float(np.std(train_x_ref)) + 1e-9

    feats += [x.astype(np.float64),
              (x - median) / (1.4826 * mad),
              (x - mu) / sd,
              np.diff(x, prepend=x[0]),
              np.diff(x, n=2, prepend=[x[0], x[0]])]

    for w in (5, 11, 21):
        m, s = _rolling_mean_std(x, w); feats += [m, s]

    rmed11, rmad11 = _rolling_median_mad(x, 11)
    feats += [rmed11, rmad11, x - rmed11, (x - rmed11) / (1.4826 * rmad11 + 1e-9)]

    ewma = _ewma(x); res = x - ewma
    feats += [ewma, res, res / (np.std(res) + 1e-9)]
    feats += [_percentile_rank_vs(x, train_x_ref),
              np.arange(n, dtype=np.float64) / max(1, n - 1)]

    tod, dow = _time_features(timestamps); feats += [tod, dow]

    rmean41, rstd41 = _rolling_mean_std(x, 41)
    rmed41, rmad41  = _rolling_median_mad(x, 41)
    _, rs5 = _rolling_mean_std(x, 5)
    feats += [rmean41, rstd41, rmed41, rmad41, rs5 / (rstd41 + 1e-9)]

    rmin11, rmax11 = _rolling_minmax(x, 11)
    feats += [rmax11 - x, x - rmin11]
    for w_mm in (5, 21, 41):
        rmin_w, rmax_w = _rolling_minmax(x, w_mm)
        feats += [rmax_w - x, x - rmin_w]

    mz, sz, xz = win_global
    feats += [np.full(n, mz), np.full(n, sz), np.full(n, xz)]

    static = []
    mt = info.get("metric_type", "Unknown")
    for m in METRIC_TYPES:
        static.append(1. if mt == m else 0.)
    for ts in top_services:
        static.append(1. if service == ts else 0.)
    static += [0.] * (TOP_K_SERVICES - len(top_services))
    static += [float(info.get("intervals", 0)) / 3600.,
               float(info.get("training set anomaly ratio", 0.)),
               float(info.get("test set anomaly ratio", 0.))]

    point_feats  = np.column_stack(feats)
    feats_static = np.tile(np.array(static, dtype=np.float64), (n, 1))
    return np.hstack([point_feats, feats_static]).astype(np.float32)


def make_features(x, timestamps, train_x_ref, info, service, top_services,
                  win_global=(0., 0., 0.)):
    """92-feature P1 = 77 base + 5 FFT + 10 TDA."""
    base = _base77(x, timestamps, train_x_ref, info, service, top_services, win_global)
    fft  = _fft_anomaly_features(x, train_x_ref)
    tda  = tda_window_features(x, train_x_ref)
    n    = len(x)
    return np.hstack([base, fft, np.tile(tda, (n, 1))])


def make_features_shift(x, timestamps, ref_x, info, service, top_services,
                        win_global=(0., 0., 0.)):
    """99-feature P2 = 92 + 7 shift."""
    base = make_features(x, timestamps, ref_x, info, service, top_services, win_global)
    n = len(x)
    x_med = float(np.median(x)); x_mad = float(np.median(np.abs(x - x_med))) + 1e-9
    ref_std = float(np.std(ref_x)) + 1e-9
    shift = np.column_stack([
        _percentile_rank_vs(x, x),
        (x - x_med) / (1.4826 * x_mad),
        np.maximum(0., x - float(np.max(ref_x))),
        np.maximum(0., float(np.min(ref_x)) - x),
        np.full(n, float(np.mean(x)) - float(np.mean(ref_x))),
        np.full(n, float(np.std(x)) / ref_std),
        np.full(n, float(np.median(x)) - float(np.median(ref_x))),
    ]).astype(np.float32)
    return np.hstack([base, shift])


# ─────────────────────────────────────────────
# Top services / pseudo-label helpers
# ─────────────────────────────────────────────

def compute_top_services(window_dirs, k=TOP_K_SERVICES):
    counts = Counter()
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        counts[parse_service(info.get("case_name", ""))] += 1
    return [s for s, _ in counts.most_common(k)]

def load_pseudo_labels(path):
    data = json.loads(path.read_text())
    return {wid: np.array(v, dtype=np.int64) for wid, v in data["predictions"].items()}

def build_wid_map(window_dirs):
    return {wdir.name.split("_", 1)[0]: wdir for wdir in window_dirs}

def _load_test_arrays(wdir, info):
    test_x = np.load(wdir / "test.npy")
    try:
        test_ts = np.load(wdir / "test_timestamp.npy")
    except FileNotFoundError:
        test_ts = np.arange(len(test_x), dtype=np.int64) * info.get("intervals", 60)
    return test_x, test_ts


# ─────────────────────────────────────────────
# Pool builders
# ─────────────────────────────────────────────

def _build_pool_p1(window_dirs, top_services, target_mt,
                   train_gs, test_gs, pseudo_labels=None, wid_map=None):
    Xs, ys, ws = [], [], []
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        if info.get("metric_type") != target_mt: continue
        try:
            train_y = np.load(wdir / "train_label.npy")
        except FileNotFoundError: continue
        if train_y.sum() == 0: continue
        train_x = np.load(wdir / "train.npy")
        try:
            train_ts = np.load(wdir / "train_timestamp.npy")
        except FileNotFoundError:
            train_ts = np.arange(len(train_x), dtype=np.int64) * info.get("intervals", 60)
        service = parse_service(info.get("case_name", ""))
        wg = train_gs.get(wdir, (0., 0., 0.))
        Xs.append(make_features(train_x, train_ts, train_x, info, service, top_services, wg))
        ys.append(train_y)
        ws.append(np.ones(len(train_y), dtype=np.float32))

    if pseudo_labels and wid_map:
        for wid, pseudo_y in pseudo_labels.items():
            if pseudo_y.sum() == 0: continue
            wdir = wid_map.get(wid)
            if wdir is None: continue
            info = json.loads((wdir / "info.json").read_text())
            if info.get("metric_type") != target_mt: continue
            train_x = np.load(wdir / "train.npy")
            test_x, test_ts = _load_test_arrays(wdir, info)
            if len(test_x) != len(pseudo_y): continue
            service = parse_service(info.get("case_name", ""))
            wg = test_gs.get(wdir, (0., 0., 0.))
            Xs.append(make_features(test_x, test_ts, train_x, info, service, top_services, wg))
            ys.append(pseudo_y)
            ws.append(np.full(len(pseudo_y), PSEUDO_WEIGHT, dtype=np.float32))

    if not Xs:
        return np.zeros((0, N_FEATS_P1), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.vstack(Xs), np.hstack(ys), np.hstack(ws)


def _build_pool_p2(window_dirs, top_services, target_mt,
                   train_gs, test_gs, pseudo_labels=None, wid_map=None):
    Xs, ys, ws = [], [], []
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        if info.get("metric_type") != target_mt: continue
        try:
            train_y = np.load(wdir / "train_label.npy")
        except FileNotFoundError: continue
        if train_y.sum() == 0: continue
        train_x = np.load(wdir / "train.npy")
        n = len(train_x); cut = max(10, int(n * SPLIT_FRAC))
        if n - cut < 5: continue
        pseudo_y_split = train_y[cut:]
        if pseudo_y_split.sum() == 0: continue
        ref_x = train_x[:cut]; pseudo_x = train_x[cut:]
        try:
            train_ts = np.load(wdir / "train_timestamp.npy")
        except FileNotFoundError:
            train_ts = np.arange(n, dtype=np.int64) * info.get("intervals", 60)
        service = parse_service(info.get("case_name", ""))
        wg = train_gs.get(wdir, (0., 0., 0.))
        Xs.append(make_features_shift(pseudo_x, train_ts[cut:], ref_x, info, service, top_services, wg))
        ys.append(pseudo_y_split)
        ws.append(np.ones(len(pseudo_y_split), dtype=np.float32))

    if pseudo_labels and wid_map:
        for wid, pseudo_y in pseudo_labels.items():
            if pseudo_y.sum() == 0: continue
            wdir = wid_map.get(wid)
            if wdir is None: continue
            info = json.loads((wdir / "info.json").read_text())
            if info.get("metric_type") != target_mt: continue
            train_x = np.load(wdir / "train.npy")
            cut = max(1, int(len(train_x) * SPLIT_FRAC))
            ref_x = train_x[:cut]
            test_x, test_ts = _load_test_arrays(wdir, info)
            if len(test_x) != len(pseudo_y) or len(test_x) < 5: continue
            service = parse_service(info.get("case_name", ""))
            wg = test_gs.get(wdir, (0., 0., 0.))
            Xs.append(make_features_shift(test_x, test_ts, ref_x, info, service, top_services, wg))
            ys.append(pseudo_y)
            ws.append(np.full(len(pseudo_y), PSEUDO_WEIGHT, dtype=np.float32))

    if not Xs:
        return np.zeros((0, N_FEATS_P2), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.vstack(Xs), np.hstack(ys), np.hstack(ws)


# ─────────────────────────────────────────────
# Ensemble (HGBT + RF + LR, same as v58/v60)
# ─────────────────────────────────────────────

def _fit_models(X, y, label, sample_weight=None):
    scaler = StandardScaler().fit(X); X_sc = scaler.transform(X)
    t0 = time.time()
    rfs = [RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=10,
                                   class_weight="balanced", random_state=s, n_jobs=4).fit(
                                       X, y, sample_weight=sample_weight)
           for s in (0, 1, 2)]
    print(f"    {label} 3-seed RF  {time.time()-t0:.1f}s")
    t0 = time.time()
    hgbts = [HistGradientBoostingClassifier(max_iter=200, max_depth=8, learning_rate=0.05,
                                             min_samples_leaf=20, random_state=s,
                                             class_weight="balanced").fit(
                                                 X, y, sample_weight=sample_weight)
             for s in range(5)]
    print(f"    {label} 5-seed HGBT {time.time()-t0:.1f}s")
    t0 = time.time()
    lr = LogisticRegression(C=0.5, max_iter=500, class_weight="balanced",
                             solver="lbfgs").fit(X_sc, y, sample_weight=sample_weight)
    print(f"    {label} 1 LR        {time.time()-t0:.1f}s")
    return {"rfs": rfs, "hgbts": hgbts, "lr": lr, "scaler": scaler}


def fit_both_ensembles(window_dirs, top_services, train_gs, test_gs,
                       pseudo_labels=None, wid_map=None):
    ensembles = {}
    for mt in METRIC_TYPES:
        print(f"  [{mt}] building pools…", flush=True)
        t0 = time.time()
        X1, y1, w1 = _build_pool_p1(window_dirs, top_services, mt, train_gs, test_gs, pseudo_labels, wid_map)
        X2, y2, w2 = _build_pool_p2(window_dirs, top_services, mt, train_gs, test_gs, pseudo_labels, wid_map)
        print(f"    P1 X={X1.shape} pos={y1.mean():.3f}  "
              f"P2 X={X2.shape} pos={y2.mean() if len(y2) else 0:.3f}  "
              f"build={time.time()-t0:.1f}s")
        if len(X1) < 100:
            print(f"    SKIP P1"); continue
        bundle_p1 = _fit_models(X1, y1, "P1", sample_weight=w1)
        bundle_p2 = _fit_models(X2, y2, "P2", sample_weight=w2) if len(X2) >= 50 else None
        if bundle_p2 is None:
            print(f"    SKIP P2 ({len(X2)} samples)")
        ensembles[mt] = {"p1": bundle_p1, "p2": bundle_p2}
    return ensembles


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def _score_bundle(bundle, X):
    X_sc = bundle["scaler"].transform(X)
    rf_avg   = np.mean([c.predict_proba(X)[:, 1] for c in bundle["rfs"]], axis=0)
    hgbt_avg = np.mean([c.predict_proba(X)[:, 1] for c in bundle["hgbts"]], axis=0)
    lr_p     = bundle["lr"].predict_proba(X_sc)[:, 1]
    return 0.80 * hgbt_avg + 0.10 * rf_avg + 0.10 * lr_p


def predict_proba_window(test_x, test_ts, train_x_ref, info, service,
                          top_services, ensembles, win_global=(0., 0., 0.)):
    mt = info.get("metric_type", "Unknown")
    if mt not in ensembles: mt = next(iter(ensembles))
    bundle = ensembles[mt]
    X1 = make_features(test_x, test_ts, train_x_ref, info, service, top_services, win_global)
    p1 = _score_bundle(bundle["p1"], X1)
    if bundle["p2"] is not None:
        X2 = make_features_shift(test_x, test_ts, train_x_ref, info, service, top_services, win_global)
        p2 = _score_bundle(bundle["p2"], X2)
        return (1. - W_SHIFT) * p1 + W_SHIFT * p2
    return p1


def smooth_centered(p, w=SMOOTH_W):
    if w <= 1: return p.copy()
    kernel = np.ones(w) / w; half = w // 2
    padded = np.concatenate([p[half-1::-1] if half > 0 else p[:0], p,
                             p[-1:-half-1:-1] if half > 0 else p[:0]])
    out = np.convolve(padded, kernel, mode="valid")
    if len(out) > len(p): out = out[:len(p)]
    elif len(out) < len(p): out = np.convolve(p, kernel, mode="same")
    return out


def predict_window(test_x, test_ts, train_x_ref, info, service, top_services,
                   ensembles, win_global=(0., 0., 0.)):
    n = len(test_x)
    k = max(0, min(int(round(n * float(info.get("test set anomaly ratio", 0.)))), n))
    if k == 0: return np.zeros(n, dtype=int)
    prob = predict_proba_window(test_x, test_ts, train_x_ref, info, service,
                                top_services, ensembles, win_global)
    rm = smooth_centered(prob, SMOOTH_W)
    prob_f = (1. - SMOOTH_ALPHA) * prob + SMOOTH_ALPHA * rm
    order = np.lexsort((np.arange(n), -prob_f))
    pred = np.zeros(n, dtype=int); pred[order[:k]] = 1
    return pred


# ─────────────────────────────────────────────
# Validation + submission
# ─────────────────────────────────────────────

def run_validation(pseudo_labels, wid_map):
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=42)
    top_services = compute_top_services(train_pool)
    train_gs = compute_window_global_stats(train_pool, "train.npy")
    test_gs  = compute_window_global_stats(all_window_dirs(), "test.npy")

    print(">>> Fitting P1+P2 ensembles…")
    t0 = time.time()
    ensembles = fit_both_ensembles(train_pool, top_services, train_gs, test_gs,
                                   pseudo_labels, wid_map)
    print(f"    fit {time.time()-t0:.1f}s")

    def predictor(window):
        try:
            train_ts = np.load(window.wdir / "train_timestamp.npy")
        except FileNotFoundError:
            train_ts = np.arange(len(window.train_x)) * window.info.get("intervals", 60)
        service = parse_service(window.info.get("case_name", ""))
        wg = train_gs.get(window.wdir, (0., 0., 0.))
        return predict_window(window.train_x, train_ts, window.train_x, window.info,
                              service, top_services, ensembles, wg)

    print(">>> Cross-window LOO on holdout…")
    rep = cross_window_evaluate(predictor, holdout)
    print_summary_v2(rep, "v61 TDA+FFT (CW-LOO)")
    return rep


def generate_submission(ensembles, top_services, test_gs,
                        output=Path("submission_v61_tda.json")):
    print(f"\n>>> Generating predictions on 1000 test windows…")
    preds = {}; t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        try:
            test_ts = np.load(wdir / "test_timestamp.npy")
        except FileNotFoundError:
            test_ts = np.arange(len(w.test_x)) * w.info.get("intervals", 60)
        service = parse_service(w.info.get("case_name", ""))
        wg = test_gs.get(wdir, (0., 0., 0.))
        pred = predict_window(w.test_x, test_ts, w.train_x, w.info,
                              service, top_services, ensembles, wg)
        preds[w.wid] = pred.astype(int).tolist()
        if i % 100 == 0:
            print(f"    {i}/1000 ({time.time()-t0:.0f}s)")
    assert len(preds) == 1000
    output.write_text(json.dumps({"predictions": preds}, ensure_ascii=False,
                                  separators=(",", ":")), encoding="utf-8")
    print(f">>> Wrote {output}")
    return output


if __name__ == "__main__":
    print(f"Pseudo-label source: {PSEUDO_SOURCE}")
    print(f"N_FEATS_P1={N_FEATS_P1}  N_FEATS_P2={N_FEATS_P2}")
    print(f"TDA: maxdim=1, dim={TDA_DIM}, lag={TDA_LAG}, max_pts={TDA_MAX_PTS}\n")

    pseudo_labels = load_pseudo_labels(PSEUDO_SOURCE)
    n_with = sum(1 for v in pseudo_labels.values() if v.sum() > 0)
    print(f"Loaded {len(pseudo_labels)} pseudo-label windows, {n_with} with anomalies")

    wid_map = build_wid_map(all_window_dirs())

    # Quick benchmark: how long does ripser take on a typical window?
    _bx = np.load(list(all_window_dirs())[0] / "test.npy")
    _t = time.time()
    tda_window_features(_bx, _bx)
    print(f"TDA benchmark: n={len(_bx)} → {time.time()-_t:.3f}s per window\n")

    rep = run_validation(pseudo_labels, wid_map)

    print("\n>>> Re-training on ALL windows for final submission…")
    t0 = time.time()
    top_sv   = compute_top_services(all_window_dirs())
    train_gs = compute_window_global_stats(all_window_dirs(), "train.npy")
    test_gs  = compute_window_global_stats(all_window_dirs(), "test.npy")
    ensembles = fit_both_ensembles(all_window_dirs(), top_sv, train_gs, test_gs,
                                   pseudo_labels, wid_map)
    print(f"    full fit {time.time()-t0:.1f}s")
    generate_submission(ensembles, top_sv, test_gs)
    print("\nDone. Submit submission_v61_tda.json")
