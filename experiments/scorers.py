"""
Anomaly scorers for time-series anomaly detection.

Each scorer is a callable with signature:
    scorer(train_x, test_x, train_y=None) -> np.ndarray

The returned array has the same length as test_x, with higher values
indicating more anomalous.
"""

from typing import Callable, Dict
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.neighbors import KernelDensity
import torch
import torch.nn as nn


def normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    scores = np.asarray(scores).ravel()
    mn, mx = np.min(scores), np.max(scores)
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


# ─────────────────────────────────────────────
# Statistical scorers (global, train-based)
# ─────────────────────────────────────────────

def score_zscore(train_x: np.ndarray, test_x: np.ndarray, **kwargs) -> np.ndarray:
    """Z-score relative to training mean and std."""
    mean, std = np.mean(train_x), np.std(train_x)
    if std < 1e-9:
        return np.zeros(len(test_x))
    return normalize(np.abs((test_x - mean) / std))


def score_mad(train_x: np.ndarray, test_x: np.ndarray, **kwargs) -> np.ndarray:
    """Modified Z-score using Median Absolute Deviation."""
    median = np.median(train_x)
    mad = np.median(np.abs(train_x - median))
    if mad < 1e-9:
        return np.zeros(len(test_x))
    return normalize(np.abs(0.6745 * (test_x - median) / mad))


def score_iqr(train_x: np.ndarray, test_x: np.ndarray, **kwargs) -> np.ndarray:
    """IQR-based score: how far outside the interquartile range."""
    q1, q3 = np.percentile(train_x, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-9:
        return np.zeros(len(test_x))
    return normalize(np.maximum((test_x - q3) / iqr, (q1 - test_x) / iqr))


def score_percentile(train_x: np.ndarray, test_x: np.ndarray, **kwargs) -> np.ndarray:
    """How extreme is each test point in the training distribution."""
    sorted_train = np.sort(train_x)
    n = len(sorted_train)
    if n == 0:
        return np.zeros(len(test_x))
    # Find rank of each test point in training distribution
    ranks = np.searchsorted(sorted_train, test_x)
    ranks = np.clip(ranks, 0, n - 1)
    # Distance from median rank
    median_rank = n // 2
    return normalize(np.abs(ranks - median_rank))


# ─────────────────────────────────────────────
# Temporal / differencing scorers
# ─────────────────────────────────────────────

def score_diff_zscore(train_x: np.ndarray, test_x: np.ndarray, **kwargs) -> np.ndarray:
    """Z-score of first differences relative to training diffs."""
    train_diffs = np.abs(np.diff(train_x, prepend=train_x[0]))
    test_diffs = np.abs(np.diff(test_x, prepend=test_x[0]))
    mean_d, std_d = np.mean(train_diffs), np.std(train_diffs)
    if std_d < 1e-9:
        return np.zeros(len(test_x))
    return normalize(np.abs((test_diffs - mean_d) / std_d))


def score_jerk(train_x: np.ndarray, test_x: np.ndarray, **kwargs) -> np.ndarray:
    """Z-score of second differences (jerk) relative to training."""
    pad = [train_x[0], train_x[0]]
    train_j = np.abs(np.diff(train_x, n=2, prepend=pad))
    test_j = np.abs(np.diff(test_x, n=2, prepend=[test_x[0], test_x[0]]))
    mean_j, std_j = np.mean(train_j), np.std(train_j)
    if std_j < 1e-9:
        return np.zeros(len(test_x))
    return normalize(np.abs((test_j - mean_j) / std_j))


def score_ema_diff(train_x: np.ndarray, test_x: np.ndarray, span: int = 5, **kwargs) -> np.ndarray:
    """Deviation from exponential moving average."""
    alpha = 2.0 / (span + 1)
    ema = test_x[0]
    ema_vals = np.zeros(len(test_x))
    for i in range(len(test_x)):
        ema = alpha * test_x[i] + (1 - alpha) * ema
        ema_vals[i] = ema
    return normalize(np.abs(test_x - ema_vals))


# ─────────────────────────────────────────────
# Online / adaptive scorers (test-window based)
# ─────────────────────────────────────────────

def score_online_rolling_zscore(train_x: np.ndarray, test_x: np.ndarray, window: int = 15, **kwargs) -> np.ndarray:
    """Rolling z-score computed only on past test points (no training data)."""
    scores = np.zeros(len(test_x))
    for i in range(len(test_x)):
        start = max(0, i - window)
        ref = test_x[start:i]
        if len(ref) < 2 or np.std(ref) < 1e-9:
            scores[i] = 0.0
        else:
            scores[i] = abs((test_x[i] - np.mean(ref)) / np.std(ref))
    return normalize(scores)


def score_online_diff_zscore(train_x: np.ndarray, test_x: np.ndarray, window: int = 15, **kwargs) -> np.ndarray:
    """Rolling z-score of first differences within the test window."""
    diffs = np.zeros(len(test_x))
    diffs[0] = 0.0
    for i in range(1, len(test_x)):
        diffs[i] = abs(test_x[i] - test_x[i - 1])
    scores = np.zeros(len(test_x))
    for i in range(1, len(test_x)):
        start = max(1, i - window)
        ref = diffs[start:i]
        if len(ref) < 2 or np.std(ref) < 1e-9:
            scores[i] = 0.0
        else:
            scores[i] = abs((diffs[i] - np.mean(ref)) / np.std(ref))
    return normalize(scores)


def score_online_jerk(train_x: np.ndarray, test_x: np.ndarray, window: int = 15, **kwargs) -> np.ndarray:
    """Rolling z-score of second differences within the test window."""
    diffs2 = np.zeros(len(test_x))
    diffs2[0] = diffs2[1] = 0.0
    for i in range(2, len(test_x)):
        diffs2[i] = abs(test_x[i] - 2 * test_x[i - 1] + test_x[i - 2])
    scores = np.zeros(len(test_x))
    for i in range(2, len(test_x)):
        start = max(2, i - window)
        ref = diffs2[start:i]
        if len(ref) < 2 or np.std(ref) < 1e-9:
            scores[i] = 0.0
        else:
            scores[i] = abs((diffs2[i] - np.mean(ref)) / np.std(ref))
    return normalize(scores)


# ─────────────────────────────────────────────
# Distance-based scorers
# ─────────────────────────────────────────────

def score_knn_distance(train_x: np.ndarray, test_x: np.ndarray, k: int = 5, **kwargs) -> np.ndarray:
    """Mean distance to k nearest neighbors in training data."""
    k = min(k, len(train_x) - 1)
    if k < 1:
        return np.zeros(len(test_x))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(train_x.reshape(-1, 1))
    dists, _ = nn.kneighbors(test_x.reshape(-1, 1))
    return normalize(np.mean(dists, axis=1))


def score_hist_rare(train_x: np.ndarray, test_x: np.ndarray, bins: int = 10, **kwargs) -> np.ndarray:
    """Score based on how rare a test point's histogram bin is in training data."""
    counts, edges = np.histogram(train_x, bins=min(bins, len(train_x)))
    idx = np.digitize(test_x, edges[:-1]) - 1
    idx = np.clip(idx, 0, len(counts) - 1)
    return normalize(-counts[idx])


# ─────────────────────────────────────────────
# Density-based scorers
# ─────────────────────────────────────────────

def score_kde_ll(train_x: np.ndarray, test_x: np.ndarray, **kwargs) -> np.ndarray:
    """Negative log-likelihood under a KDE fit on training data."""
    bw = max(np.std(train_x) * 0.3, 1e-6)
    kde = KernelDensity(bandwidth=bw).fit(train_x.reshape(-1, 1))
    ll = kde.score_samples(test_x.reshape(-1, 1))
    return normalize(-ll)


# ─────────────────────────────────────────────
# Isolation-based scorers
# ─────────────────────────────────────────────

def score_isolation_forest(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray = None, **kwargs) -> np.ndarray:
    """Isolation Forest trained on training data."""
    contamination = 0.1
    if train_y is not None and np.sum(train_y) > 0:
        contamination = max(min(float(np.mean(train_y)), 0.5), 0.01)
    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=1,
    )
    clf.fit(train_x.reshape(-1, 1))
    scores = -clf.decision_function(test_x.reshape(-1, 1))
    return normalize(scores)


def score_lof(train_x: np.ndarray, test_x: np.ndarray, **kwargs) -> np.ndarray:
    """Local Outlier Factor trained on training data."""
    n_neighbors = min(20, len(train_x) - 1)
    if n_neighbors < 1:
        return np.zeros(len(test_x))
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=1)
    clf.fit(train_x.reshape(-1, 1))
    scores = -clf.decision_function(test_x.reshape(-1, 1))
    return normalize(scores)


# ─────────────────────────────────────────────
# Supervised scorers
# ─────────────────────────────────────────────

def _extract_pointwise_features(series: np.ndarray) -> np.ndarray:
    """Extract pointwise feature matrix for a single series."""
    n = len(series)
    feats = {}
    feats["val"] = series
    for lag in [1, 2, 3, 5, 10]:
        shifted = np.roll(series, lag)
        shifted[:lag] = series[:lag]
        feats[f"diff_{lag}"] = series - shifted
        feats[f"ratio_{lag}"] = series / (shifted + 1e-9)
    for w in [3, 5, 10]:
        for i in range(n):
            window = series[max(0, i - w + 1):i + 1]
            if f"roll_mean_{w}" not in feats:
                feats[f"roll_mean_{w}"] = np.zeros(n)
                feats[f"roll_std_{w}"] = np.zeros(n)
                feats[f"roll_min_{w}"] = np.zeros(n)
                feats[f"roll_max_{w}"] = np.zeros(n)
            feats[f"roll_mean_{w}"][i] = np.mean(window)
            feats[f"roll_std_{w}"][i] = np.std(window)
            feats[f"roll_min_{w}"][i] = np.min(window)
            feats[f"roll_max_{w}"][i] = np.max(window)
    for w in [3, 5, 10]:
        feats[f"zscore_{w}"] = (series - feats[f"roll_mean_{w}"]) / (feats[f"roll_std_{w}"] + 1e-9)
    feats["second_diff"] = np.diff(series, n=2, prepend=[series[0], series[0]])
    # EMA
    alpha = 0.3
    ema = series[0]
    ema_vals = np.zeros(n)
    for i in range(n):
        ema = alpha * series[i] + (1 - alpha) * ema
        ema_vals[i] = ema
    feats["ema_diff"] = series - ema_vals
    return np.column_stack([feats[k] for k in sorted(feats.keys())])


def score_supervised_rf(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray, **kwargs) -> np.ndarray:
    """RandomForest trained on pointwise features from training data."""
    if np.sum(train_y) == 0:
        return np.zeros(len(test_x))
    X_train = _extract_pointwise_features(train_x)
    X_test = _extract_pointwise_features(test_x)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X_train, train_y)
    proba = clf.predict_proba(X_test)
    if proba.shape[1] > 1:
        return proba[:, 1]
    return np.zeros(len(test_x))


# ─────────────────────────────────────────────
# Deep learning scorers
# ─────────────────────────────────────────────

def score_autoencoder(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray = None, epochs: int = 50, **kwargs) -> np.ndarray:
    """Simple 1D autoencoder reconstruction error."""
    device = torch.device("cpu")
    mean, std = np.mean(train_x), np.std(train_x)
    if std < 1e-9:
        std = 1.0
    train_norm = (np.asarray(train_x, dtype=np.float32) - mean) / std
    test_norm = (np.asarray(test_x, dtype=np.float32) - mean) / std
    train_tensor = torch.tensor(train_norm, dtype=torch.float32).unsqueeze(1).to(device)
    test_tensor = torch.tensor(test_norm, dtype=torch.float32).unsqueeze(1).to(device)

    model = nn.Sequential(
        nn.Linear(1, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 16), nn.ReLU(),
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(train_tensor)
        loss = criterion(pred.squeeze(), train_tensor.squeeze())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(test_tensor)
        scores = torch.abs(pred.squeeze() - test_tensor.squeeze()).cpu().numpy()
    return normalize(scores)


# ─────────────────────────────────────────────
# Registry of all scorers
# ─────────────────────────────────────────────

ALL_SCORERS: Dict[str, Callable] = {
    "zscore": score_zscore,
    "mad": score_mad,
    "iqr": score_iqr,
    "percentile": score_percentile,
    "diff_zscore": score_diff_zscore,
    "jerk": score_jerk,
    "ema_diff": score_ema_diff,
    "online_rolling_zscore_15": lambda tx, te, **kw: score_online_rolling_zscore(tx, te, window=15),
    "online_rolling_zscore_20": lambda tx, te, **kw: score_online_rolling_zscore(tx, te, window=20),
    "online_diff_zscore_15": lambda tx, te, **kw: score_online_diff_zscore(tx, te, window=15),
    "online_jerk_15": lambda tx, te, **kw: score_online_jerk(tx, te, window=15),
    "knn_dist_3": lambda tx, te, **kw: score_knn_distance(tx, te, k=3),
    "knn_dist_5": lambda tx, te, **kw: score_knn_distance(tx, te, k=5),
    "hist_rare": score_hist_rare,
    "kde_ll": score_kde_ll,
    "isolation_forest": score_isolation_forest,
    "lof": score_lof,
    "supervised_rf": score_supervised_rf,
    "autoencoder": score_autoencoder,
}
