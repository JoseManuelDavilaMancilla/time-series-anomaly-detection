"""
author v56 — higher pseudo-label weight experiment (PSEUDO_WEIGHT=0.50).

Same architecture as v54 (74-feature P1, 81-feature P2, iterated pseudo-labels)
but PSEUDO_WEIGHT raised from 0.30 to 0.50 to give pseudo-labeled test windows
more influence during training.

PSEUDO_SOURCE: submission_v54_iter_pseudo2.json (0.6810 LB — current best)

Run:  uv run python v56_pw50.py
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter
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
W_SHIFT = 0.30
SPLIT_FRAC = 0.70
N_FEATS_P1 = 74          # 68 base + 6 extra rolling min/max
N_FEATS_P2 = 81          # 74 + 7 shift features
PSEUDO_WEIGHT = 0.50    # raised from 0.30 → 0.50 (more pseudo-label influence)
PSEUDO_SOURCE = Path("submission_v54_iter_pseudo2.json")  # 0.6810 LB best


# ─────────────────────────────────────────────
# Rolling helpers — all centered (identical to v43)
# ─────────────────────────────────────────────

def _rolling_mean_std(x: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    s = pd.Series(x.astype(np.float64))
    r = s.rolling(w, center=True, min_periods=1)
    mean = r.mean().to_numpy()
    std = r.std(ddof=0).fillna(0.0).to_numpy()
    return mean, std


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
        start = max(0, i - half)
        end = min(n, i + half + 1)
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
# Feature builders (identical to v43)
# ─────────────────────────────────────────────

def make_features(
    x: np.ndarray, timestamps: np.ndarray, train_x_ref: np.ndarray,
    info: dict, service: str, top_services: List[str],
) -> np.ndarray:
    """68-feature matrix (P1). Identical to v43."""
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
) -> np.ndarray:
    """75-feature matrix (P2) = 68 standard + 7 shift features."""
    base = make_features(x, timestamps, ref_x, info, service, top_services)
    n = len(x)

    x_med = float(np.median(x))
    x_mad = float(np.median(np.abs(x - x_med))) + 1e-9
    ref_std = float(np.std(ref_x)) + 1e-9

    rank_in_self = _percentile_rank_vs(x, x)
    self_robust_z = (x - x_med) / (1.4826 * x_mad)
    above_ref_max = np.maximum(0.0, x - float(np.max(ref_x)))
    below_ref_min = np.maximum(0.0, float(np.min(ref_x)) - x)
    mean_shift_bc = np.full(n, float(np.mean(x)) - float(np.mean(ref_x)))
    std_ratio_bc = np.full(n, float(np.std(x)) / ref_std)
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
    """Load binary predictions from a submission JSON as pseudo-labels."""
    data = json.loads(path.read_text())
    return {wid: np.array(v, dtype=np.int64)
            for wid, v in data["predictions"].items()}


def build_wid_map(window_dirs) -> Dict[str, Path]:
    """Map wid (3-digit prefix) to wdir."""
    return {wdir.name.split("_", 1)[0]: wdir for wdir in window_dirs}


def _load_test_arrays(wdir: Path, info: dict):
    """Load test_x and test_ts for a window."""
    test_x = np.load(wdir / "test.npy")
    try:
        test_ts = np.load(wdir / "test_timestamp.npy")
    except FileNotFoundError:
        test_ts = np.arange(len(test_x), dtype=np.int64) * info.get("intervals", 60)
    return test_x, test_ts


# ─────────────────────────────────────────────
# Training pool builders (with pseudo-labels)
# ─────────────────────────────────────────────

def _build_pool_p1(window_dirs, top_services, target_mt,
                   pseudo_labels=None, wid_map=None):
    """P1 pool: full train_x (74 feats) + optional pseudo-labeled test windows."""
    Xs, ys, sample_ws = [], [], []

    # True labeled training windows
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
        sample_ws.append(np.ones(len(train_y), dtype=np.float32))

    # Pseudo-labeled test windows
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
            feats = make_features(test_x, test_ts, train_x, info, service, top_services)
            Xs.append(feats)
            ys.append(pseudo_y)
            sample_ws.append(np.full(len(pseudo_y), PSEUDO_WEIGHT, dtype=np.float32))

    if not Xs:
        return (np.zeros((0, N_FEATS_P1), np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.float32))
    return np.vstack(Xs), np.hstack(ys), np.hstack(sample_ws)


def _build_pool_p2(window_dirs, top_services, target_mt, split_frac=SPLIT_FRAC,
                   pseudo_labels=None, wid_map=None):
    """P2 pool: temporal splits (81 feats) + optional pseudo-labeled test windows.

    For pseudo-labeled test windows: ref_x = train_x[:split_frac] (first 70% of
    training data). This EXACTLY matches the inference distribution of P2, because
    at inference P2 sees test_x relative to train_x[:split_frac].
    """
    Xs, ys, sample_ws = [], [], []

    # True labeled training windows (temporal split)
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
        feats = make_features_shift(pseudo_x, pseudo_ts, ref_x, info, service, top_services)
        Xs.append(feats)
        ys.append(pseudo_y_split)
        sample_ws.append(np.ones(len(pseudo_y_split), dtype=np.float32))

    # Pseudo-labeled test windows: ref_x = train_x[:split_frac]
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
            if len(test_x) != len(pseudo_y):
                continue
            if len(test_x) < 5:
                continue
            service = parse_service(info.get("case_name", ""))
            feats = make_features_shift(test_x, test_ts, ref_x, info, service, top_services)
            Xs.append(feats)
            ys.append(pseudo_y)
            sample_ws.append(np.full(len(pseudo_y), PSEUDO_WEIGHT, dtype=np.float32))

    if not Xs:
        return (np.zeros((0, N_FEATS_P2), np.float32),
                np.zeros(0, np.int64), np.zeros(0, np.float32))
    return np.vstack(Xs), np.hstack(ys), np.hstack(sample_ws)


# ─────────────────────────────────────────────
# Ensemble fitting
# ─────────────────────────────────────────────

def _fit_models(X: np.ndarray, y: np.ndarray, label: str,
                sample_weight: np.ndarray | None = None) -> dict:
    """Train 3 RF + 5 HGBT + 1 LR. Supports sample_weight for pseudo-labeling."""
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

    t0 = time.time()
    hgbts = []
    for s in range(5):
        hgbt = HistGradientBoostingClassifier(
            max_iter=200, max_depth=8, learning_rate=0.05,
            min_samples_leaf=20, random_state=s, class_weight="balanced",
        )
        hgbt.fit(X, y, sample_weight=sample_weight)
        hgbts.append(hgbt)
    print(f"    {label} 5-seed HGBT fit {time.time() - t0:.1f}s")

    t0 = time.time()
    lr = LogisticRegression(C=0.5, max_iter=500, class_weight="balanced", solver="lbfgs")
    lr.fit(X_scaled, y, sample_weight=sample_weight)
    print(f"    {label} 1 LR fit {time.time() - t0:.1f}s")

    return {"rfs": rfs, "hgbts": hgbts, "lr": lr, "scaler": scaler}


def fit_both_ensembles(window_dirs, top_services,
                       pseudo_labels=None, wid_map=None) -> Dict[str, dict]:
    """For each metric_type: fit P1 (74 feats) and P2 (81 feats) bundles."""
    ensembles: Dict[str, dict] = {}
    for mt in METRIC_TYPES:
        print(f"  [{mt}] building pools…", flush=True)
        t0 = time.time()
        X1, y1, w1 = _build_pool_p1(window_dirs, top_services, mt,
                                     pseudo_labels, wid_map)
        X2, y2, w2 = _build_pool_p2(window_dirs, top_services, mt,
                                     pseudo_labels=pseudo_labels, wid_map=wid_map)

        n_pseudo_p1 = sum(1 for wid, py in (pseudo_labels or {}).items()
                          if py.sum() > 0 and wid_map and wid_map.get(wid) is not None
                          and json.loads((wid_map[wid] / "info.json").read_text()).get("metric_type") == mt)
        print(f"    P1 X={X1.shape} pos={y1.mean():.3f} pseudo_windows≈{n_pseudo_p1}  "
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
# Inference (identical to v43)
# ─────────────────────────────────────────────

def _score_bundle(bundle: dict, X: np.ndarray) -> np.ndarray:
    X_scaled = bundle["scaler"].transform(X)
    rf_avg = np.mean([clf.predict_proba(X)[:, 1] for clf in bundle["rfs"]], axis=0)
    hgbt_avg = np.mean([clf.predict_proba(X)[:, 1] for clf in bundle["hgbts"]], axis=0)
    lr_p = bundle["lr"].predict_proba(X_scaled)[:, 1]
    return 0.80 * hgbt_avg + 0.10 * rf_avg + 0.10 * lr_p


def predict_proba_window(
    test_x, test_ts, train_x_ref, info, service, top_services, ensembles,
) -> np.ndarray:
    mt = info.get("metric_type", "Unknown")
    if mt not in ensembles:
        mt = next(iter(ensembles))
    bundle = ensembles[mt]

    X1 = make_features(test_x, test_ts, train_x_ref, info, service, top_services)
    prob_p1 = _score_bundle(bundle["p1"], X1)

    if bundle["p2"] is not None:
        X2 = make_features_shift(test_x, test_ts, train_x_ref, info, service, top_services)
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
                   ensembles) -> np.ndarray:
    n = len(test_x)
    ratio = float(info.get("test set anomaly ratio", 0.0))
    k = max(0, min(int(round(n * ratio)), n))
    if k == 0:
        return np.zeros(n, dtype=int)

    prob_mean = predict_proba_window(test_x, test_ts, train_x_ref, info, service,
                                     top_services, ensembles)
    rm = smooth_centered(prob_mean, SMOOTH_W)
    prob_final = (1.0 - SMOOTH_ALPHA) * prob_mean + SMOOTH_ALPHA * rm

    order = np.lexsort((np.arange(n), -prob_final))
    top_idx = order[:k]
    pred = np.zeros(n, dtype=int)
    pred[top_idx] = 1
    return pred


# ─────────────────────────────────────────────
# Validation + submission
# ─────────────────────────────────────────────

def run_validation(pseudo_labels, wid_map, seed: int = 42):
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}  "
          f"pseudo_windows={sum(1 for py in pseudo_labels.values() if py.sum()>0)}")

    print(">>> Computing top-30 services from train_pool…")
    top_services = compute_top_services(train_pool, k=TOP_K_SERVICES)

    print(">>> Fitting P1+P2 ensembles on train_pool + pseudo-labeled test…")
    t0 = time.time()
    ensembles = fit_both_ensembles(train_pool, top_services, pseudo_labels, wid_map)
    print(f"    total fit time {time.time() - t0:.1f}s")

    def predictor(window):
        try:
            train_ts = np.load(window.wdir / "train_timestamp.npy")
        except FileNotFoundError:
            train_ts = np.arange(len(window.train_x), dtype=np.int64) * window.info.get("intervals", 60)
        service = parse_service(window.info.get("case_name", ""))
        return predict_window(window.train_x, train_ts, window.train_x, window.info,
                              service, top_services, ensembles)

    print(">>> Cross-window LOO evaluation on holdout train_x…")
    rep = cross_window_evaluate(predictor, holdout)
    print_summary_v2(rep, "v56 pw50 (CW-LOO)")

    from validation import save_report
    save_report(rep, "v56_pw50_loo")
    return rep, ensembles, top_services


def generate_submission(ensembles, top_services,
                        output: Path = Path("submission_v56_pw50.json")) -> Path:
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
                              top_services, ensembles)
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
    print(f">>> Loading pseudo-labels from {PSEUDO_SOURCE}…")
    pseudo_labels = load_pseudo_labels(PSEUDO_SOURCE)
    n_with_anomalies = sum(1 for py in pseudo_labels.values() if py.sum() > 0)
    print(f"    {len(pseudo_labels)} windows, {n_with_anomalies} with predicted anomalies")

    wid_map = build_wid_map(all_window_dirs())

    rep, ensembles, top_services = run_validation(pseudo_labels, wid_map)

    print("\n>>> Re-training on ALL 1000 windows + pseudo-labels for final submission…")
    t0 = time.time()
    top_services_full = compute_top_services(all_window_dirs(), k=TOP_K_SERVICES)
    ensembles_full = fit_both_ensembles(all_window_dirs(), top_services_full,
                                        pseudo_labels, wid_map)
    print(f"    full fit {time.time() - t0:.1f}s")
    generate_submission(ensembles_full, top_services_full)
