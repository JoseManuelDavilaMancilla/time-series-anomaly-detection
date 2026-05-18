"""
author v71 — catch22 (22) + complexity (3) + per-metric rules (6) + CatBoost.

Four new additions on top of v68 (108/115 features):

1. CATCH22 (22 window-level broadcast features)
   Canonical time-series features from the hctsa toolbox.
   Computed on test_x, broadcast to all points. Captures distribution,
   autocorrelation, nonlinear dynamics, periodicity, spectral structure.

2. COMPLEXITY STREAM (3 window-level features)
   sample_entropy, permutation_entropy (order=3), lempel_ziv_complexity.
   Broadcast to all points. Anomalous windows often have different complexity.

3. PER-METRIC RULES (6 per-point features)
   Domain-specific threshold features for each of the 6 metric types.
   For a given window, 5 features = 0 and 1 has actual metric-specific values.
   Encodes professor-designed domain knowledge about each metric.

4. CatBoost as 4th model type in ensemble.
   Blend: 0.65*HGBT + 0.15*CatBoost + 0.10*RF + 0.10*LR.

N_FEATS_P1: 108 + 22 + 3 + 6 = 139
N_FEATS_P2: 115 + 22 + 3 + 6 = 146

A time series professor would grade by: do you detect CONTEXTUAL anomalies —
values that are normal globally but anomalous given what the seasonal pattern
EXPECTS at this timestamp?

Two new signal classes added on top of v65 (105/112 features):

1. AR(1) FORECAST RESIDUAL — 1 per-point feature
   Fit AR(1) model on training (phi = autocorr of train_x).
   For each test point: residual = observed - AR_predicted.
   Measures: is this point surprising given the previous point + learned dynamics?
   Different from rolling mean: uses learned mean-reversion dynamics of training.

2. STL SEASONAL DECOMPOSITION RESIDUAL — 1 per-point feature
   For 3600s intervals (24 pts/day): STL with daily period → residual z-score.
   For 1200s/864s intervals (72/100 pts/day): STL with daily period if enough data.
   For others: linear detrend residual.
   Measures: is this point anomalous given the learned seasonal + trend pattern?
   CATCHES: contextual anomalies (e.g., value is normal in isolation but wrong
   for this time-of-day given the daily cycle learned from training).

3. CUMULATIVE AR RESIDUAL — 1 per-point feature
   Running mean of |AR residuals| up to time t.
   Path-dependent: captures sustained excursions above/below expected AR path.

N_FEATS_P1: 105 + 3 = 108
N_FEATS_P2: 112 + 3 = 115

Pseudo-labels: submission_v65_matrix_profile.json (LB 0.6890).
PSEUDO_WEIGHT: 0.70.

Run:  uv run python v68_stl_ar.py
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
import stumpy
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

from validation import all_window_dirs, load_window
from cross_validation import cross_window_evaluate, print_summary_v2
from validation import stratified_holdout


METRIC_TYPES    = ("Count", "ErrorCount", "LatencySecond", "QPS",
                   "ResourceUtilizationRate", "SuccessRate")
TOP_K_SERVICES  = 30
SMOOTH_W        = 5
SMOOTH_ALPHA    = 0.8
W_SHIFT         = 0.30
SPLIT_FRAC      = 0.70
N_TDA_FEATS     = 10
N_FFT_FEATS     = 5
N_MP_FEATS      = 3    # matrix profile at m=5,10,20
N_STL_AR_FEATS  = 3    # AR(1) residual + STL/trend residual + cumulative AR
N_CATCH22_FEATS = 22   # pycatch22 canonical features (window-level broadcast)
N_COMPLEX_FEATS = 3    # sample_entropy, permutation_entropy, lempel_ziv
N_PMRULE_FEATS  = 6    # per-metric domain rules (1 per metric_type)
N_ROLL_NEW      = 6    # mean+std at w=3,7,63
N_FFT_BROAD     = 4    # top1/2/3 mag + hf_energy_ratio
N_FEATS_P1      = 77 + N_FFT_FEATS + N_TDA_FEATS + N_MP_FEATS + N_ROLL_NEW + N_FFT_BROAD + N_STL_AR_FEATS + N_CATCH22_FEATS + N_COMPLEX_FEATS + N_PMRULE_FEATS  # 139
N_FEATS_P2      = 84 + N_FFT_FEATS + N_TDA_FEATS + N_MP_FEATS + N_ROLL_NEW + N_FFT_BROAD + N_STL_AR_FEATS + N_CATCH22_FEATS + N_COMPLEX_FEATS + N_PMRULE_FEATS  # 146
PSEUDO_WEIGHT   = 0.70
PSEUDO_SOURCE   = Path("submission_v68_stl_ar.json")
CACHE_DIR       = Path("tda_cache")
MP_WINDOWS      = [5, 10, 20]
EXTRA_ROLL_W    = [3, 7, 63]


# ─────────────────────────────────────────────
# Stumpy warmup (avoid 12s JIT delay mid-run)
# ─────────────────────────────────────────────

def _warmup_stumpy():
    t0 = time.time()
    _x = np.random.randn(60).astype(np.float64)
    for m in MP_WINDOWS:
        stumpy.stump(_x, m)
    print(f"    stumpy JIT warmup: {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────
# TDA cache
# ─────────────────────────────────────────────

_tda_cache: Dict[str, np.ndarray] = {}

def _load_tda_cache():
    global _tda_cache
    if _tda_cache: return
    files = list(CACHE_DIR.glob("*.npy"))
    if not files: raise FileNotFoundError("Run precompute_tda_cache.py first.")
    for f in files: _tda_cache[f.stem] = np.load(f)
    print(f"    Loaded {len(_tda_cache)} TDA cache entries.")

def get_tda(wid: str, kind: str) -> np.ndarray:
    return _tda_cache.get(f"{wid}_{kind}", np.zeros(N_TDA_FEATS, dtype=np.float32))


# ─────────────────────────────────────────────
# NEW: Matrix Profile features
# ─────────────────────────────────────────────

def matrix_profile_features(x: np.ndarray) -> np.ndarray:
    """
    3 per-point features: normalised MP distance at m=5,10,20.
    High = discord (anomaly). Normalised by median so scale-invariant.
    Output shape: (n, 3).
    """
    n = len(x)
    out = np.zeros((n, len(MP_WINDOWS)), dtype=np.float32)
    x_f = x.astype(np.float64)
    for j, m in enumerate(MP_WINDOWS):
        if n < 2 * m + 2:
            continue
        mp = np.array(stumpy.stump(x_f, m)[:, 0], dtype=np.float64)  # shape (n-m+1,)
        # Replace inf with max finite value
        finite_mask = np.isfinite(mp)
        if finite_mask.sum() == 0:
            continue
        mp = np.where(finite_mask, mp, mp[finite_mask].max())
        # Align (n-m+1) → n: assign mp[i] to window center i + m//2
        aligned = np.empty(n, dtype=np.float64)
        half = m // 2
        for i in range(len(mp)):
            center = min(i + half, n - 1)
            aligned[center] = mp[i]
        # Fill edges
        aligned[:half] = mp[0]
        aligned[len(mp) + half:] = mp[-1]
        # Robust normalise by median
        med = float(np.median(aligned)) + 1e-9
        out[:, j] = np.clip(aligned / med, 0, 20).astype(np.float32)
    return out


# ─────────────────────────────────────────────
# NEW: FFT broadcast features (window-level)
# ─────────────────────────────────────────────

def fft_broadcast_features(x: np.ndarray) -> np.ndarray:
    """
    4 window-level scalars: top-3 normalised peak magnitudes + HF energy ratio.
    """
    n = len(x)
    if n < 8:
        return np.zeros(4, dtype=np.float32)
    X = np.abs(np.fft.rfft(x.astype(np.float64)))
    total_power = X.sum() + 1e-9
    X_norm = X / total_power
    # Top-3 peaks (excluding DC at index 0)
    peaks = np.argsort(X_norm[1:])[::-1][:3] + 1
    top_mags = np.array([X_norm[peaks[i]] if i < len(peaks) else 0.0
                         for i in range(3)], dtype=np.float32)
    # High-frequency energy: power in top quarter of frequencies
    hf_cutoff = max(1, len(X) * 3 // 4)
    hf_energy = float(X[hf_cutoff:].sum() / total_power)
    return np.array([top_mags[0], top_mags[1], top_mags[2], hf_energy],
                    dtype=np.float32)


# ─────────────────────────────────────────────
# FFT reconstruction features (per-point, v60)
# ─────────────────────────────────────────────

def _fft_reconstruct(x: np.ndarray, n_keep: int) -> np.ndarray:
    n = len(x)
    if n < 8: return np.full(n, np.mean(x))
    X = np.fft.rfft(x)
    top = np.argsort(np.abs(X)[1:])[::-1][:n_keep] + 1
    mask = np.zeros(len(X), dtype=bool); mask[0] = True; mask[top] = True
    return np.fft.irfft(X * mask, n=n)

def _fft_anomaly_features(test_x: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    n = len(test_x); n_keep = max(1, min(10, n // 4))
    test_recon = _fft_reconstruct(test_x.astype(np.float64), n_keep)
    test_res   = test_x.astype(np.float64) - test_recon
    res_med = float(np.median(test_res)); res_mad = float(np.median(np.abs(test_res-res_med)))+1e-9
    test_res_z = np.clip((test_res-res_med)/(1.4826*res_mad), -10, 10)
    n_keep_tr  = max(1, min(10, len(train_x)//4))
    train_recon = _fft_reconstruct(train_x.astype(np.float64), n_keep_tr)
    train_res   = train_x.astype(np.float64) - train_recon
    train_res_std = float(np.std(train_res))+1e-9
    periodicity   = float(np.clip(np.var(train_recon)/(np.var(train_x)+1e-9), 0., 1.))
    return np.column_stack([test_res, test_res_z,
                             np.clip(test_res/train_res_std, -10, 10),
                             np.full(n, periodicity),
                             np.full(n, np.log1p(train_res_std))]).astype(np.float32)


# ─────────────────────────────────────────────
# Global context
# ─────────────────────────────────────────────

def compute_window_global_stats(window_dirs, data_file="train.npy"):
    raw = {}
    for wdir in window_dirs:
        info = json.loads((wdir/"info.json").read_text())
        mt = info.get("metric_type","Unknown")
        try: x = np.load(wdir/data_file).astype(np.float64)
        except FileNotFoundError: x = np.load(wdir/"train.npy").astype(np.float64)
        raw[wdir] = (mt, float(np.mean(x)), float(np.std(x)), float(np.max(x)))
    by_mt = defaultdict(list)
    for wdir,(mt,m,s,mx) in raw.items(): by_mt[mt].append((wdir,m,s,mx))
    result = {}
    for mt, entries in by_mt.items():
        wdirs=[e[0] for e in entries]; means=np.array([e[1] for e in entries])
        stds=np.array([e[2] for e in entries]); maxs=np.array([e[3] for e in entries])
        def _z(arr):
            mu,sigma=arr.mean(),arr.std()
            return np.clip((arr-mu)/(sigma+1e-9),-5.,5.)
        for i,wdir in enumerate(wdirs):
            result[wdir]=(float(_z(means)[i]),float(_z(stds)[i]),float(_z(maxs)[i]))
    return result


# ─────────────────────────────────────────────
# Rolling helpers
# ─────────────────────────────────────────────

def _rolling_mean_std(x, w):
    import pandas as pd
    s=pd.Series(x.astype(np.float64)); r=s.rolling(w,center=True,min_periods=1)
    return r.mean().to_numpy(), r.std(ddof=0).fillna(0.).to_numpy()

def _rolling_minmax(x,w):
    import pandas as pd
    s=pd.Series(x.astype(np.float64)); r=s.rolling(w,center=True,min_periods=1)
    return r.min().to_numpy(), r.max().to_numpy()

def _rolling_median_mad(x,w):
    half=w//2; n=len(x); med=np.empty(n); mad=np.empty(n)
    for i in range(n):
        seg=x[max(0,i-half):min(n,i+half+1)]; m=np.median(seg)
        med[i]=m; mad[i]=np.median(np.abs(seg-m))
    return med,mad

def _ewma(x,alpha=0.3):
    out=np.empty_like(x,dtype=np.float64); e=float(x[0])
    for i,v in enumerate(x): e=alpha*float(v)+(1-alpha)*e; out[i]=e
    return out

def _percentile_rank_vs(values,reference):
    ref_sorted=np.sort(reference); n_ref=len(ref_sorted)
    if n_ref==0: return np.full(len(values),0.5)
    return np.searchsorted(ref_sorted,values,side="right").astype(np.float64)/n_ref

def _time_features(timestamps):
    n=len(timestamps); tod=np.empty(n); dow=np.empty(n)
    for i,t in enumerate(timestamps):
        d=datetime.fromtimestamp(int(t),tz=timezone.utc)
        tod[i]=(d.hour+d.minute/60+d.second/3600)/24; dow[i]=d.weekday()/7.
    return tod,dow

def parse_service(case_name):
    prefix=case_name.split("##",1)[0] if "##" in case_name else case_name
    parts=prefix.split("_",1)
    return parts[1] if len(parts)>1 and "_" in prefix else prefix

def _wid(wdir): return wdir.name.split("_",1)[0]


# ─────────────────────────────────────────────
# Feature builders
# ─────────────────────────────────────────────

def _base77(x, timestamps, train_x_ref, info, service, top_services, win_global):
    n=len(x); feats=[]
    median=float(np.median(train_x_ref)); mad=float(np.median(np.abs(train_x_ref-median)))+1e-9
    mu=float(np.mean(train_x_ref)); sd=float(np.std(train_x_ref))+1e-9
    feats+=[x.astype(np.float64),(x-median)/(1.4826*mad),(x-mu)/sd,
            np.diff(x,prepend=x[0]),np.diff(x,n=2,prepend=[x[0],x[0]])]
    for w in (5,11,21):
        m,s=_rolling_mean_std(x,w); feats+=[m,s]
    rmed11,rmad11=_rolling_median_mad(x,11)
    feats+=[rmed11,rmad11,x-rmed11,(x-rmed11)/(1.4826*rmad11+1e-9)]
    ewma=_ewma(x); res=x-ewma
    feats+=[ewma,res,res/(np.std(res)+1e-9)]
    feats+=[_percentile_rank_vs(x,train_x_ref),np.arange(n,dtype=np.float64)/max(1,n-1)]
    tod,dow=_time_features(timestamps); feats+=[tod,dow]
    rmean41,rstd41=_rolling_mean_std(x,41); rmed41,rmad41=_rolling_median_mad(x,41)
    _,rs5=_rolling_mean_std(x,5)
    feats+=[rmean41,rstd41,rmed41,rmad41,rs5/(rstd41+1e-9)]
    rmin11,rmax11=_rolling_minmax(x,11); feats+=[rmax11-x,x-rmin11]
    for w_mm in (5,21,41):
        rmin_w,rmax_w=_rolling_minmax(x,w_mm); feats+=[rmax_w-x,x-rmin_w]
    mz,sz,xz=win_global
    feats+=[np.full(n,mz),np.full(n,sz),np.full(n,xz)]
    static=[]
    mt=info.get("metric_type","Unknown")
    for m in METRIC_TYPES: static.append(1. if mt==m else 0.)
    for ts in top_services: static.append(1. if service==ts else 0.)
    static+=[0.]*(TOP_K_SERVICES-len(top_services))
    static+=[float(info.get("intervals",0))/3600.,
             float(info.get("training set anomaly ratio",0.)),
             float(info.get("test set anomaly ratio",0.))]
    point_feats=np.column_stack(feats)
    feats_static=np.tile(np.array(static,dtype=np.float64),(n,1))
    return np.hstack([point_feats,feats_static]).astype(np.float32)


def _extra_rolling(x: np.ndarray) -> np.ndarray:
    """6 per-point features: mean+std at w=3,7,63."""
    feats = []
    for w in EXTRA_ROLL_W:
        m, s = _rolling_mean_std(x, w)
        feats += [m, s]
    return np.column_stack(feats).astype(np.float32)



# ─────────────────────────────────────────────
# NEW: STL seasonal decomposition + AR(1) residual features
# ─────────────────────────────────────────────

def stl_ar_features(x: np.ndarray, train_x: np.ndarray,
                    intervals: float, kind: str = "test") -> np.ndarray:
    """3 per-point features: AR(1) residual z, STL/trend residual z, running |AR| mean.

    use_self_join=True  when x IS train_x (P1 pool, kind=train, len(x)==len(train_x)).
    use_self_join=False when x comes AFTER train_x (P2 pseudo_x or test_x).
    """
    from statsmodels.tsa.seasonal import STL
    n_ref = len(train_x)
    n = len(x)
    # P2 training uses pseudo_x (last 30% of train) as x, ref_x as train_x → n != n_ref
    use_self_join = (kind == "train" and n == n_ref)

    # ── AR(1) parameters from training ──
    mu_ar = float(np.mean(train_x))
    if n_ref >= 4 and float(np.std(train_x)) > 1e-9:
        phi = float(np.corrcoef(train_x[:-1], train_x[1:])[0, 1])
        phi = 0.0 if not np.isfinite(phi) else float(np.clip(phi, -0.99, 0.99))
    else:
        phi = 0.0

    # AR(1) residuals
    ar_res = np.zeros(n, dtype=np.float64)
    if use_self_join:
        for i in range(1, n):
            x_pred = mu_ar + phi * (float(x[i - 1]) - mu_ar)
            ar_res[i] = float(x[i]) - x_pred
        ar_res[0] = 0.0
    else:
        x_prev = float(train_x[-1])
        for i in range(n):
            x_pred = mu_ar + phi * (x_prev - mu_ar)
            ar_res[i] = float(x[i]) - x_pred
            x_prev = float(x[i])

    train_ar_preds = mu_ar + phi * (train_x[:-1].astype(np.float64) - mu_ar)
    train_ar_res = train_x[1:].astype(np.float64) - train_ar_preds
    ar_std = float(np.std(train_ar_res)) + 1e-9
    ar_z = np.clip(ar_res / ar_std, -10, 10).astype(np.float32)

    cumabs = np.cumsum(np.abs(ar_res)) / (np.arange(n, dtype=np.float64) + 1)
    cumabs_z = np.clip(cumabs / ar_std, 0, 10).astype(np.float32)

    # ── STL or linear detrend ──
    stl_z = np.zeros(n, dtype=np.float32)
    period = None
    if intervals > 0:
        period_daily = round(86400.0 / intervals)
        if 4 <= period_daily <= n_ref // 2:
            period = period_daily

    if period is not None:
        try:
            stl = STL(train_x.astype(float), period=period, seasonal=7, robust=True)
            res = stl.fit()
            train_res_std = float(np.std(res.resid)) + 1e-9
            train_res_mean = float(np.mean(res.resid))
            if use_self_join:
                raw_res = res.resid.astype(np.float64)        # length = n_ref = n ✓
            else:
                n_extrap = max(period, n_ref // 4)
                slope, intercept = np.polyfit(np.arange(n_extrap),
                                               res.trend[-n_extrap:].astype(float), 1)
                test_trend = intercept + slope * np.arange(n_extrap, n_extrap + n)
                seasonal_train = res.seasonal.astype(np.float64)
                test_seasonal = np.array([seasonal_train[-(period - (i % period)) % period]
                                           for i in range(n)])
                raw_res = x.astype(np.float64) - test_trend - test_seasonal  # length = n ✓
            stl_z = np.clip((raw_res - train_res_mean) / train_res_std, -10, 10).astype(np.float32)
        except Exception:
            stl_z = np.zeros(n, dtype=np.float32)
    else:
        if n_ref >= 4:
            coefs = np.polyfit(np.arange(n_ref), train_x.astype(float), 1)
            train_fitted = np.polyval(coefs, np.arange(n_ref))
            train_res_lin = train_x.astype(np.float64) - train_fitted
            res_std = float(np.std(train_res_lin)) + 1e-9
            res_mean = float(np.mean(train_res_lin))
            if use_self_join:
                raw_res = train_res_lin                          # length = n_ref = n ✓
            else:
                test_trend_lin = np.polyval(coefs, np.arange(n_ref, n_ref + n))
                raw_res = x.astype(np.float64) - test_trend_lin  # length = n ✓
            stl_z = np.clip((raw_res - res_mean) / res_std, -10, 10).astype(np.float32)

    out = np.column_stack([ar_z, stl_z, cumabs_z]).astype(np.float32)
    np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0, copy=False)
    return out

# ─────────────────────────────────────────────
# NEW: catch22 + complexity + per-metric rules
# ─────────────────────────────────────────────

def catch22_features(x: np.ndarray) -> np.ndarray:
    """22 canonical time-series features (window-level broadcast)."""
    import pycatch22
    try:
        vals = pycatch22.catch22_all(x.astype(float).tolist())["values"]
        arr = np.array(vals, dtype=np.float32)
        arr = np.where(np.isfinite(arr), np.clip(arr, -1e6, 1e6), 0.0)
    except Exception:
        arr = np.zeros(22, dtype=np.float32)
    return arr


def complexity_features(x: np.ndarray) -> np.ndarray:
    """3 window-level complexity features: sample_entropy, perm_entropy, lempel_ziv."""
    n = len(x)

    # Sample entropy (m=2, r=0.2*std)
    def sample_entropy(ts, m=2):
        std = float(np.std(ts))
        if std < 1e-9 or len(ts) < 2*m+2:
            return 0.0
        r = 0.2 * std
        ts = ts.astype(np.float64)
        def _count(length):
            count = 0
            for i in range(len(ts) - length):
                template = ts[i:i+length]
                diffs = np.max(np.abs(ts[:len(ts)-length].reshape(-1,1)
                               + np.lib.stride_tricks.as_strided(
                                   ts, shape=(len(ts)-length, length),
                                   strides=(ts.strides[0], ts.strides[0])
                               ) - template), axis=1)
                count += np.sum(diffs < r) - 1
            return max(count, 0)
        B = _count(m); A = _count(m+1)
        return float(-np.log((A+1e-9)/(B+1e-9)))
    
    # Permutation entropy (order=3)
    def perm_entropy(ts, order=3):
        n = len(ts)
        if n < order:
            return 0.0
        from math import factorial, log
        import itertools
        permutations = list(itertools.permutations(range(order)))
        perm_map = {p: i for i, p in enumerate(permutations)}
        counts = np.zeros(len(permutations))
        for i in range(n - order + 1):
            pattern = tuple(np.argsort(ts[i:i+order]))
            counts[perm_map[pattern]] += 1
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs)))
    
    # Lempel-Ziv complexity (binary, threshold at median)
    def lempel_ziv(ts):
        if len(ts) < 4:
            return 0.0
        binary = (ts > np.median(ts)).astype(int)
        s = "".join(map(str, binary))
        n = len(s); i = 0; c = 1; l = 1; k = 1
        while i + l <= n - 1:
            if s[i:i+l] in s[:i+l-1]:
                l += 1
            else:
                c += 1; i += l; l = 1
        if l != 1:
            c += 1
        return float(c * np.log2(n + 1e-9) / (n + 1e-9))

    se = np.clip(sample_entropy(x), 0, 10)
    pe = np.clip(perm_entropy(x), 0, 10)
    lz = np.clip(lempel_ziv(x), 0, 1)
    return np.array([se, pe, lz], dtype=np.float32)


def per_metric_rules(x: np.ndarray, train_x: np.ndarray, metric_type: str) -> np.ndarray:
    """6 per-point domain-rule features (one active per metric type).
    
    Each metric has a specific "what should be anomalous" rule based on domain knowledge.
    For a given window, only the feature for its metric_type has values; rest are 0.
    """
    n = len(x)
    feats = np.zeros((n, 6), dtype=np.float32)
    
    p5  = float(np.percentile(train_x, 5))
    p95 = float(np.percentile(train_x, 95))
    mu  = float(np.mean(train_x))
    sd  = float(np.std(train_x)) + 1e-9
    iqr = float(np.percentile(train_x, 75) - np.percentile(train_x, 25)) + 1e-9

    # Rule per metric: how anomalous is x given domain expectation
    if metric_type == "Count":
        # Counts drop to zero or spike far above baseline
        feats[:, 0] = np.clip((np.abs(x - mu) - 2*sd) / sd, 0, 10)
    elif metric_type == "ErrorCount":
        # ErrorCount should be near 0; spikes above are anomalous
        feats[:, 1] = np.clip((x - p95) / (sd + 1e-9), 0, 10)
    elif metric_type == "LatencySecond":
        # Latency above 95th percentile = degraded
        feats[:, 2] = np.clip((x - p95) / sd, 0, 10)
    elif metric_type == "QPS":
        # QPS drop (traffic gone) or spike — both anomalous  
        feats[:, 3] = np.clip((np.abs(x - mu) - 1.5*iqr) / iqr, 0, 10)
    elif metric_type == "ResourceUtilizationRate":
        # Resource saturation (too high) or sudden drop (too low)
        feats[:, 4] = np.clip((np.abs(x - mu) - 2*sd) / sd, 0, 10)
    elif metric_type == "SuccessRate":
        # Success rate drops below normal range — degradation
        feats[:, 5] = np.clip((p5 - x) / sd, 0, 10)
    
    return feats  # (n, 6)


def make_features(x, timestamps, train_x_ref, info, service, top_services,
                  win_global=(0.,0.,0.), wid_str="", kind="train"):
    """139-feature P1 = 108 base + 22 catch22 + 3 complexity + 6 per-metric rules."""
    base   = _base77(x, timestamps, train_x_ref, info, service, top_services, win_global)
    fft    = _fft_anomaly_features(x, train_x_ref)       # (n, 5)
    tda    = get_tda(wid_str, kind)                       # (10,)
    mp     = matrix_profile_features(x)                  # (n, 3)
    roll   = _extra_rolling(x)                            # (n, 6)
    fft_b  = fft_broadcast_features(x)                   # (4,)
    iv     = float(info.get("intervals", 0))
    stl_ar = stl_ar_features(x, train_x_ref, iv, kind)   # (n, 3)
    c22    = catch22_features(x)                          # (22,)
    cpx    = complexity_features(x)                       # (3,)
    mt     = info.get("metric_type", "Count")
    pmr    = per_metric_rules(x, train_x_ref, mt)        # (n, 6)
    n = len(x)
    out = np.hstack([base, fft,
                      np.tile(tda, (n,1)),
                      mp, roll,
                      np.tile(fft_b, (n,1)),
                      stl_ar,
                      np.tile(c22, (n,1)),
                      np.tile(cpx, (n,1)),
                      pmr])
    np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0, copy=False)
    return out


def make_features_shift(x, timestamps, ref_x, info, service, top_services,
                        win_global=(0.,0.,0.), wid_str="", kind="test"):
    """146-feature P2 = 139 + 7 shift."""
    base = make_features(x, timestamps, ref_x, info, service, top_services,
                         win_global, wid_str, kind)
    n = len(x)
    x_med=float(np.median(x)); x_mad=float(np.median(np.abs(x-x_med)))+1e-9
    ref_std=float(np.std(ref_x))+1e-9
    shift=np.column_stack([
        _percentile_rank_vs(x,x),(x-x_med)/(1.4826*x_mad),
        np.maximum(0.,x-float(np.max(ref_x))),np.maximum(0.,float(np.min(ref_x))-x),
        np.full(n,float(np.mean(x))-float(np.mean(ref_x))),
        np.full(n,float(np.std(x))/ref_std),
        np.full(n,float(np.median(x))-float(np.median(ref_x))),
    ]).astype(np.float32)
    return np.hstack([base,shift])


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def compute_top_services(window_dirs, k=TOP_K_SERVICES):
    counts=Counter()
    for wdir in window_dirs:
        info=json.loads((wdir/"info.json").read_text())
        counts[parse_service(info.get("case_name",""))]+=1
    return [s for s,_ in counts.most_common(k)]

def load_pseudo_labels(path):
    data=json.loads(path.read_text())
    return {wid: np.array(v,dtype=np.int64) for wid,v in data["predictions"].items()}

def build_wid_map(window_dirs):
    return {wdir.name.split("_",1)[0]: wdir for wdir in window_dirs}

def _load_test_arrays(wdir,info):
    test_x=np.load(wdir/"test.npy")
    try: test_ts=np.load(wdir/"test_timestamp.npy")
    except FileNotFoundError: test_ts=np.arange(len(test_x),dtype=np.int64)*info.get("intervals",60)
    return test_x,test_ts


# ─────────────────────────────────────────────
# Pool builders
# ─────────────────────────────────────────────

def _build_pool_p1(window_dirs,top_services,target_mt,train_gs,test_gs,
                   pseudo_labels=None,wid_map=None):
    Xs,ys,ws=[],[],[]
    for wdir in window_dirs:
        info=json.loads((wdir/"info.json").read_text())
        if info.get("metric_type")!=target_mt: continue
        try: train_y=np.load(wdir/"train_label.npy")
        except FileNotFoundError: continue
        if train_y.sum()==0: continue
        train_x=np.load(wdir/"train.npy")
        try: train_ts=np.load(wdir/"train_timestamp.npy")
        except FileNotFoundError: train_ts=np.arange(len(train_x),dtype=np.int64)*info.get("intervals",60)
        service=parse_service(info.get("case_name","")); wg=train_gs.get(wdir,(0.,0.,0.)); w=_wid(wdir)
        Xs.append(make_features(train_x,train_ts,train_x,info,service,top_services,wg,w,"train"))
        ys.append(train_y); ws.append(np.ones(len(train_y),dtype=np.float32))

    if pseudo_labels and wid_map:
        for wid,pseudo_y in pseudo_labels.items():
            if pseudo_y.sum()==0: continue
            wdir=wid_map.get(wid)
            if wdir is None: continue
            info=json.loads((wdir/"info.json").read_text())
            if info.get("metric_type")!=target_mt: continue
            train_x=np.load(wdir/"train.npy"); test_x,test_ts=_load_test_arrays(wdir,info)
            if len(test_x)!=len(pseudo_y): continue
            service=parse_service(info.get("case_name","")); wg=test_gs.get(wdir,(0.,0.,0.))
            Xs.append(make_features(test_x,test_ts,train_x,info,service,top_services,wg,wid,"test"))
            ys.append(pseudo_y); ws.append(np.full(len(pseudo_y),PSEUDO_WEIGHT,dtype=np.float32))

    if not Xs: return np.zeros((0,N_FEATS_P1),np.float32),np.zeros(0,np.int64),np.zeros(0,np.float32)
    return np.vstack(Xs),np.hstack(ys),np.hstack(ws)


def _build_pool_p2(window_dirs,top_services,target_mt,train_gs,test_gs,
                   pseudo_labels=None,wid_map=None):
    Xs,ys,ws=[],[],[]
    for wdir in window_dirs:
        info=json.loads((wdir/"info.json").read_text())
        if info.get("metric_type")!=target_mt: continue
        try: train_y=np.load(wdir/"train_label.npy")
        except FileNotFoundError: continue
        if train_y.sum()==0: continue
        train_x=np.load(wdir/"train.npy"); n=len(train_x); cut=max(10,int(n*SPLIT_FRAC))
        if n-cut<5: continue
        py_split=train_y[cut:]
        if py_split.sum()==0: continue
        ref_x=train_x[:cut]; pseudo_x=train_x[cut:]
        try: train_ts=np.load(wdir/"train_timestamp.npy")
        except FileNotFoundError: train_ts=np.arange(n,dtype=np.int64)*info.get("intervals",60)
        service=parse_service(info.get("case_name","")); wg=train_gs.get(wdir,(0.,0.,0.)); w=_wid(wdir)
        Xs.append(make_features_shift(pseudo_x,train_ts[cut:],ref_x,info,service,top_services,wg,w,"train"))
        ys.append(py_split); ws.append(np.ones(len(py_split),dtype=np.float32))

    if pseudo_labels and wid_map:
        for wid,pseudo_y in pseudo_labels.items():
            if pseudo_y.sum()==0: continue
            wdir=wid_map.get(wid)
            if wdir is None: continue
            info=json.loads((wdir/"info.json").read_text())
            if info.get("metric_type")!=target_mt: continue
            train_x=np.load(wdir/"train.npy"); cut=max(1,int(len(train_x)*SPLIT_FRAC)); ref_x=train_x[:cut]
            test_x,test_ts=_load_test_arrays(wdir,info)
            if len(test_x)!=len(pseudo_y) or len(test_x)<5: continue
            service=parse_service(info.get("case_name","")); wg=test_gs.get(wdir,(0.,0.,0.))
            Xs.append(make_features_shift(test_x,test_ts,ref_x,info,service,top_services,wg,wid,"test"))
            ys.append(pseudo_y); ws.append(np.full(len(pseudo_y),PSEUDO_WEIGHT,dtype=np.float32))

    if not Xs: return np.zeros((0,N_FEATS_P2),np.float32),np.zeros(0,np.int64),np.zeros(0,np.float32)
    return np.vstack(Xs),np.hstack(ys),np.hstack(ws)


# ─────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────

def _fit_models(X,y,label,sample_weight=None):
    scaler=StandardScaler().fit(X); X_sc=scaler.transform(X)
    pos=float(y.mean()); neg_pos_ratio=max(1.,(1-pos)/(pos+1e-9))
    t0=time.time()
    rfs=[RandomForestClassifier(n_estimators=200,max_depth=15,min_samples_leaf=10,
                                 class_weight="balanced",random_state=s,n_jobs=4).fit(
                                     X,y,sample_weight=sample_weight) for s in (0,1,2)]
    print(f"    {label} 3-seed RF   {time.time()-t0:.1f}s")
    t0=time.time()
    hgbts=[HistGradientBoostingClassifier(max_iter=200,max_depth=8,learning_rate=0.05,
                                           min_samples_leaf=20,random_state=s,
                                           class_weight="balanced").fit(
                                               X,y,sample_weight=sample_weight) for s in range(5)]
    print(f"    {label} 5-seed HGBT {time.time()-t0:.1f}s")
    t0=time.time()
    lr=LogisticRegression(C=0.5,max_iter=500,class_weight="balanced",
                           solver="lbfgs").fit(X_sc,y,sample_weight=sample_weight)
    print(f"    {label} 1 LR        {time.time()-t0:.1f}s")
    t0=time.time()
    cb=CatBoostClassifier(iterations=200,depth=8,learning_rate=0.05,
                           scale_pos_weight=neg_pos_ratio,
                           eval_metric="AUC",verbose=0,random_seed=42,
                           thread_count=4)
    cb.fit(X,y,sample_weight=sample_weight)
    print(f"    {label} 1 CatBoost  {time.time()-t0:.1f}s")
    return {"rfs":rfs,"hgbts":hgbts,"lr":lr,"scaler":scaler,"cb":cb}


def fit_both_ensembles(window_dirs,top_services,train_gs,test_gs,
                       pseudo_labels=None,wid_map=None):
    ensembles={}
    for mt in METRIC_TYPES:
        print(f"  [{mt}] building pools…",flush=True)
        t0=time.time()
        X1,y1,w1=_build_pool_p1(window_dirs,top_services,mt,train_gs,test_gs,pseudo_labels,wid_map)
        X2,y2,w2=_build_pool_p2(window_dirs,top_services,mt,train_gs,test_gs,pseudo_labels,wid_map)
        print(f"    P1 X={X1.shape} pos={y1.mean():.3f}  "
              f"P2 X={X2.shape} pos={y2.mean() if len(y2) else 0:.3f}  build={time.time()-t0:.1f}s")
        if len(X1)<100: print(f"    SKIP P1"); continue
        bundle_p1=_fit_models(X1,y1,"P1",sample_weight=w1)
        bundle_p2=_fit_models(X2,y2,"P2",sample_weight=w2) if len(X2)>=50 else None
        if bundle_p2 is None: print(f"    SKIP P2 ({len(X2)} samples)")
        ensembles[mt]={"p1":bundle_p1,"p2":bundle_p2}
    return ensembles


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def _score_bundle(bundle,X):
    X_sc=bundle["scaler"].transform(X)
    rf_avg=np.mean([c.predict_proba(X)[:,1] for c in bundle["rfs"]],axis=0)
    hgbt_avg=np.mean([c.predict_proba(X)[:,1] for c in bundle["hgbts"]],axis=0)
    lr_p=bundle["lr"].predict_proba(X_sc)[:,1]
    cb_p=bundle["cb"].predict_proba(X)[:,1]
    return 0.65*hgbt_avg+0.15*cb_p+0.10*rf_avg+0.10*lr_p

def predict_proba_window(test_x,test_ts,train_x_ref,info,service,top_services,
                          ensembles,win_global=(0.,0.,0.),wid_str=""):
    mt=info.get("metric_type","Unknown")
    if mt not in ensembles: mt=next(iter(ensembles))
    bundle=ensembles[mt]
    X1=make_features(test_x,test_ts,train_x_ref,info,service,top_services,win_global,wid_str,"test")
    p1=_score_bundle(bundle["p1"],X1)
    if bundle["p2"] is not None:
        X2=make_features_shift(test_x,test_ts,train_x_ref,info,service,top_services,win_global,wid_str,"test")
        p2=_score_bundle(bundle["p2"],X2)
        return (1.-W_SHIFT)*p1+W_SHIFT*p2
    return p1

def smooth_centered(p,w=SMOOTH_W):
    if w<=1: return p.copy()
    kernel=np.ones(w)/w; half=w//2
    padded=np.concatenate([p[half-1::-1] if half>0 else p[:0],p,p[-1:-half-1:-1] if half>0 else p[:0]])
    out=np.convolve(padded,kernel,mode="valid")
    if len(out)>len(p): out=out[:len(p)]
    elif len(out)<len(p): out=np.convolve(p,kernel,mode="same")
    return out

def predict_segments(score, k, smooth_w=3, thr_frac=0.7):
    """Grow contiguous segments from peaks; fallback to top-k for small k."""
    n = len(score)
    k = max(0, min(k, n))
    if k == 0:
        return np.zeros(n, dtype=int)

    # Small k fallback: exact top-k on raw score
    if k <= 4:
        order = np.argsort(-score)
        pred = np.zeros(n, dtype=int)
        pred[order[:k]] = 1
        return pred

    # Smooth with centered rolling mean
    smoothed = smooth_centered(score, smooth_w)

    # Threshold: peaks above thr_frac * max_score
    thr = thr_frac * float(np.max(smoothed))
    above = smoothed > thr

    # Fallback to top-k if nothing crosses threshold
    if not above.any():
        order = np.argsort(-smoothed)
        pred = np.zeros(n, dtype=int)
        pred[order[:k]] = 1
        return pred

    # Find contiguous segments (connected components of above-threshold mask)
    segments = []
    in_seg = False
    start = 0
    for i in range(n):
        if above[i]:
            if not in_seg:
                start = i
                in_seg = True
        else:
            if in_seg:
                segments.append((start, i))
                in_seg = False
    if in_seg:
        segments.append((start, n))

    # Score each segment by its peak smoothed value
    seg_scores = []
    for s, e in segments:
        seg_scores.append((float(np.max(smoothed[s:e])), s, e))
    seg_scores.sort(reverse=True)

    # Greedily select segments until we reach exactly k anomalies
    selected = np.zeros(n, dtype=bool)
    count = 0
    for _, s, e in seg_scores:
        seg_len = e - s
        if count + seg_len > k:
            # Partial segment: take highest-scoring points within it
            remaining = k - count
            seg_order = np.argsort(-smoothed[s:e])[:remaining]
            selected[s + seg_order] = True
            count += remaining
            break
        selected[s:e] = True
        count += seg_len
        if count >= k:
            break

    # Safety: if still short, fill with highest remaining points globally
    if count < k:
        remaining = k - count
        unselected = np.where(~selected)[0]
        top_remaining = unselected[np.argsort(-smoothed[unselected])[:remaining]]
        selected[top_remaining] = True

    return selected.astype(int)


def predict_window(test_x,test_ts,train_x_ref,info,service,top_services,
                   ensembles,win_global=(0.,0.,0.),wid_str=""):
    n=len(test_x)
    k=max(0,min(int(round(n*float(info.get("test set anomaly ratio",0.)))),n))
    if k==0: return np.zeros(n,dtype=int)
    prob=predict_proba_window(test_x,test_ts,train_x_ref,info,service,
                              top_services,ensembles,win_global,wid_str)
    rm=smooth_centered(prob,SMOOTH_W)
    prob_f=(1.-SMOOTH_ALPHA)*prob+SMOOTH_ALPHA*rm
    return predict_segments(prob_f, k)


# ─────────────────────────────────────────────
# Validation + submission
# ─────────────────────────────────────────────

def run_validation(pseudo_labels,wid_map):
    print("\n>>> Building stratified holdout (10%)…")
    train_pool,holdout=stratified_holdout(all_window_dirs(),frac=0.10,seed=42)
    top_services=compute_top_services(train_pool)
    train_gs=compute_window_global_stats(train_pool,"train.npy")
    test_gs=compute_window_global_stats(all_window_dirs(),"test.npy")
    print(">>> Fitting P1+P2 ensembles…")
    t0=time.time()
    ensembles=fit_both_ensembles(train_pool,top_services,train_gs,test_gs,pseudo_labels,wid_map)
    print(f"    fit {time.time()-t0:.1f}s")
    def predictor(window):
        try: train_ts=np.load(window.wdir/"train_timestamp.npy")
        except FileNotFoundError: train_ts=np.arange(len(window.train_x))*window.info.get("intervals",60)
        service=parse_service(window.info.get("case_name","")); wg=train_gs.get(window.wdir,(0.,0.,0.))
        return predict_window(window.train_x,train_ts,window.train_x,window.info,
                              service,top_services,ensembles,wg,_wid(window.wdir))
    print(">>> Cross-window LOO on holdout…")
    rep=cross_window_evaluate(predictor,holdout)
    print_summary_v2(rep,"v71 catch22+complexity+rules+catboost (CW-LOO)")
    return rep


def generate_submission(ensembles,top_services,test_gs,
                        output=Path("submission_v71_catch22.json")):
    print(f"\n>>> Generating predictions on 1000 test windows…")
    preds={}; t0=time.time()
    for i,wdir in enumerate(all_window_dirs(),1):
        w=load_window(wdir)
        try: test_ts=np.load(wdir/"test_timestamp.npy")
        except FileNotFoundError: test_ts=np.arange(len(w.test_x))*w.info.get("intervals",60)
        service=parse_service(w.info.get("case_name","")); wg=test_gs.get(wdir,(0.,0.,0.))
        pred=predict_window(w.test_x,test_ts,w.train_x,w.info,service,top_services,ensembles,wg,_wid(wdir))
        preds[w.wid]=pred.astype(int).tolist()
        if i%100==0: print(f"    {i}/1000 ({time.time()-t0:.0f}s)")
    assert len(preds)==1000
    output.write_text(json.dumps({"predictions":preds},ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    print(f">>> Wrote {output}")
    return output


if __name__ == "__main__":
    print(f"Pseudo-label source: {PSEUDO_SOURCE}")
    print(f"N_FEATS_P1={N_FEATS_P1}  N_FEATS_P2={N_FEATS_P2}")
    print(f"MP windows={MP_WINDOWS}  extra rolling={EXTRA_ROLL_W}  FFT broadcast=4")
    print(f"PSEUDO_WEIGHT={PSEUDO_WEIGHT}\n")

    print(">>> Warming up stumpy JIT…")
    _warmup_stumpy()

    _load_tda_cache()

    if PSEUDO_SOURCE.exists():
        pseudo_labels=load_pseudo_labels(PSEUDO_SOURCE)
        n_with=sum(1 for v in pseudo_labels.values() if v.sum()>0)
        print(f"Loaded {len(pseudo_labels)} pseudo-label windows, {n_with} with anomalies\n")
    else:
        print(f"WARNING: {PSEUDO_SOURCE} not found — running without pseudo-labels\n")
        pseudo_labels={}

    wid_map=build_wid_map(all_window_dirs())

    rep=run_validation(pseudo_labels,wid_map)

    print("\n>>> Re-training on ALL windows for final submission…")
    t0=time.time()
    top_sv=compute_top_services(all_window_dirs())
    train_gs=compute_window_global_stats(all_window_dirs(),"train.npy")
    test_gs=compute_window_global_stats(all_window_dirs(),"test.npy")
    ensembles=fit_both_ensembles(all_window_dirs(),top_sv,train_gs,test_gs,pseudo_labels,wid_map)
    print(f"    full fit {time.time()-t0:.1f}s")
    generate_submission(ensembles,top_sv,test_gs,output=Path("submission_v72_segments.json"))
    print("\nDone. Submit submission_v72_segments.json")
