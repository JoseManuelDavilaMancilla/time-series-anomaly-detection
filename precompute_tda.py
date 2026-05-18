"""
Precompute TDA (persistent homology) features for all windows and cache to disk.

Runs ripser once on every window (train.npy and test.npy).
Saves results to tda_cache/ directory as .npy files.
Subsequent versions load from cache — pool building returns to normal speed.

Cache files:
  tda_cache/{wid}_train.npy  — 10 TDA features from train.npy
  tda_cache/{wid}_test.npy   — 10 TDA features from test.npy

Feature order (10 values):
  [h0_max, h0_sum, h0_entropy, h1_max, h1_sum, h1_n_sig,
   ref_h0_max, ref_h1_max, h0_bottleneck, h1_bottleneck]

For train windows: ref = train_x itself → h0_bottleneck/h1_bottleneck = 0.
For test windows:  ref = train_x       → bottleneck measures train→test shift.

Run:  python precompute_tda.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ripser import ripser
from persim import bottleneck as persim_bottleneck

from validation import all_window_dirs

TDA_MAX_PTS = 150   # faster than v61's 300; quality loss is minimal
TDA_DIM     = 2
TDA_LAG     = 1
CACHE_DIR   = Path("tda_cache")


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
    if len(x) > TDA_MAX_PTS:
        idx = np.linspace(0, len(x) - 1, TDA_MAX_PTS, dtype=int)
        x = x[idx]
    cloud = _takens_embed(x.astype(np.float64))
    cloud -= cloud.mean(axis=0)
    scale = cloud.std() + 1e-9
    return cloud / scale


def _extract(dgm: np.ndarray, hdim: int):
    if hdim == 0:
        dgm = dgm[dgm[:, 1] < np.inf]
    if len(dgm) == 0:
        return 0.0, 0.0, 0.0, 0
    lt = dgm[:, 1] - dgm[:, 0]
    mx = float(lt.max())
    return mx, float(lt.sum()), _persistence_entropy(lt), int((lt > mx / 4 + 1e-10).sum())


def _safe_bn(a: np.ndarray, b: np.ndarray, hdim: int) -> float:
    if hdim == 0:
        a = a[a[:, 1] < np.inf]
        b = b[b[:, 1] < np.inf]
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0: a = np.zeros((1, 2))
    if len(b) == 0: b = np.zeros((1, 2))
    try:
        return float(persim_bottleneck(a, b))
    except Exception:
        return 0.0


def compute_tda_features(series_x: np.ndarray, ref_x: np.ndarray) -> np.ndarray:
    """Return 10-element float32 array of TDA features."""
    sc  = _prepare_cloud(series_x)
    ref = _prepare_cloud(ref_x)

    dgms_s = ripser(sc,  maxdim=1)["dgms"]
    dgms_r = ripser(ref, maxdim=1)["dgms"]

    h0_max, h0_sum, h0_ent, _     = _extract(dgms_s[0], 0)
    h1_max, h1_sum, _,      h1_n  = _extract(dgms_s[1], 1)
    r0_max, *_                     = _extract(dgms_r[0], 0)
    r1_max, *_                     = _extract(dgms_r[1], 1)

    h0_bn = _safe_bn(dgms_s[0], dgms_r[0], 0)
    h1_bn = _safe_bn(dgms_s[1], dgms_r[1], 1)

    return np.array([h0_max, h0_sum, h0_ent,
                     h1_max, h1_sum, float(h1_n),
                     r0_max, r1_max, h0_bn, h1_bn], dtype=np.float32)


if __name__ == "__main__":
    CACHE_DIR.mkdir(exist_ok=True)

    wdirs = list(all_window_dirs())
    n_total = len(wdirs)
    print(f"Precomputing TDA cache for {n_total} windows → {CACHE_DIR}/")
    print(f"max_pts={TDA_MAX_PTS}  dim={TDA_DIM}  lag={TDA_LAG}\n")

    # Benchmark
    _x = np.load(wdirs[0] / "train.npy")
    t0 = time.time()
    compute_tda_features(_x, _x)
    bm = time.time() - t0
    print(f"Benchmark: {bm:.3f}s/window → est. total {bm*n_total*2/60:.1f} min "
          f"(train + test for each window)\n")

    t_start = time.time()
    skipped = 0

    for i, wdir in enumerate(wdirs, 1):
        wid = wdir.name.split("_", 1)[0]
        train_cache = CACHE_DIR / f"{wid}_train.npy"
        test_cache  = CACHE_DIR / f"{wid}_test.npy"

        # Skip if both already cached
        if train_cache.exists() and test_cache.exists():
            skipped += 1
            continue

        train_x = np.load(wdir / "train.npy").astype(np.float64)
        test_x  = np.load(wdir / "test.npy").astype(np.float64)

        # Train features: series=train_x, ref=train_x (self-reference)
        if not train_cache.exists():
            feats = compute_tda_features(train_x, train_x)
            np.save(train_cache, feats)

        # Test features: series=test_x, ref=train_x (shift detection)
        if not test_cache.exists():
            feats = compute_tda_features(test_x, train_x)
            np.save(test_cache, feats)

        if i % 50 == 0 or i == n_total:
            elapsed = time.time() - t_start
            rate = (i - skipped) / elapsed if elapsed > 0 else 0
            remaining = (n_total - i) / rate if rate > 0 else 0
            print(f"  [{i}/{n_total}]  elapsed={elapsed/60:.1f}min  "
                  f"eta={remaining/60:.1f}min  skipped={skipped}")

    print(f"\nDone. {n_total - skipped} windows computed, {skipped} skipped (cached).")
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")
    print(f"Cache size: {sum(f.stat().st_size for f in CACHE_DIR.glob('*.npy'))/1e6:.1f} MB")
