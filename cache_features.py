"""
cache_features.py — Precompute expensive features for all windows to speed up pipeline.

Caches:
  - matrix_profile_features (3 per-point) — stumpy.stump, expensive
  - catch22_features (22 window-level) — pycatch22.catch22_all
  - complexity_features (3 window-level) — sample_entropy, perm_entropy, lempel_ziv
  - stl_ar_features (3 per-point) — statsmodels STL + AR(1)

Cache files: feature_cache/<wid>_<kind>_<feat_name>.npy
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Import from pipeline
from pipeline import (
    all_window_dirs,
    matrix_profile_features,
    catch22_features,
    complexity_features,
    stl_ar_features,
    MP_WINDOWS,
)

CACHE_DIR = Path("feature_cache")


def _wid(wdir: Path) -> str:
    return wdir.name.split("_", 1)[0]


def cache_window_features(wdir: Path, kind: str = "train") -> Dict[str, np.ndarray]:
    """Compute and return expensive features for a single window."""
    info = json.loads((wdir / "info.json").read_text())
    iv = float(info.get("intervals", 0))

    if kind == "train":
        x = np.load(wdir / "train.npy")
        train_x = x
    else:
        x = np.load(wdir / "test.npy")
        train_x = np.load(wdir / "train.npy")

    features = {}
    features["mp"] = matrix_profile_features(x)
    features["c22"] = catch22_features(x)
    features["cpx"] = complexity_features(x)
    features["stl_ar"] = stl_ar_features(x, train_x, iv, kind)
    return features


def save_cache(wid: str, kind: str, features: Dict[str, np.ndarray]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    for name, arr in features.items():
        np.save(CACHE_DIR / f"{wid}_{kind}_{name}.npy", arr)


def load_cache(wid: str, kind: str) -> Dict[str, np.ndarray] | None:
    features = {}
    for name in ["mp", "c22", "cpx", "stl_ar"]:
        path = CACHE_DIR / f"{wid}_{kind}_{name}.npy"
        if not path.exists():
            return None
        features[name] = np.load(path)
    return features


def build_all_cache(window_dirs: List[Path]) -> None:
    """Precompute and cache expensive features for all windows (train + test)."""
    CACHE_DIR.mkdir(exist_ok=True)
    total = len(window_dirs) * 2  # train + test
    done = 0
    t0 = time.time()

    for wdir in window_dirs:
        wid = _wid(wdir)
        for kind in ["train", "test"]:
            if load_cache(wid, kind) is not None:
                done += 1
                continue
            features = cache_window_features(wdir, kind)
            save_cache(wid, kind, features)
            done += 1
            if done % 50 == 0:
                print(f"  Cached {done}/{total} ({time.time()-t0:.0f}s)")

    print(f"Done. Cached {done}/{total} windows in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    print(">>> Building feature cache for all windows...")
    build_all_cache(all_window_dirs())
