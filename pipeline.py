"""
author v24 — 3-seed CW ensemble (variance reduction at the CW level).

We've ensembled CNNs (3-seed +0.011 over single-seed) but never ensembled
the RF cross-window model. Each RF is itself a 500-tree bagging ensemble,
but two RFs with different `random_state` see different bootstrap samples
and different feature subsets at each split, so their errors aren't fully
correlated. Averaging their predicted probabilities should give a small
variance reduction — same mechanism that worked for the CNN.

Test stacked on the v22 winning architecture (metric_type hybrid + intervals
as feature). Three seeds (42, 123, 7), predictions averaged at the proba
level (not at the decision level).

Run:  uv run python v24_cw_ensemble.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from shared_lib import (
    categorize_window,
    global_distance_score,
    isolation_forest_test,
    normalize_scores,
    online_ensemble,
    per_window_rf_score,
    predict_segments,
)
from v6_cnn import SPECIALIZED, build_training_pool as v6_pool
from v7_cnn_ensemble import CNN_SEEDS, ensemble_cnn_score, fit_cnn_with_seed
from v22_metadata_features import (
    MetadataCrossWindowModel, MetadataHybridCW, build_metadata_hybrid,
    scores_v14_with_meta,
)
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)
CNN_WEIGHT = 0.35
CW_SEEDS = (42, 123, 7)


def build_metadata_hybrid_with_seed(window_dirs, mode: str, seed: int):
    g = MetadataCrossWindowModel(mode=mode, per_metric=False,
                                 n_estimators=500, max_depth=15, seed=seed).fit(window_dirs)
    p = MetadataCrossWindowModel(mode=mode, per_metric=True,
                                 n_estimators=500, max_depth=15, seed=seed).fit(window_dirs)
    return MetadataHybridCW(g, p, SPECIALIZED)


class CWEnsemble:
    """Wraps multiple MetadataHybridCW models and averages their predictions."""

    def __init__(self, models: List[MetadataHybridCW]):
        self.models = models

    def predict_proba(self, test_x: np.ndarray, metric_type: str = "ALL",
                      info: dict = None) -> np.ndarray:
        preds = [m.predict_proba(test_x, metric_type=metric_type, info=info)
                 for m in self.models]
        return np.mean(preds, axis=0)


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training 3-seed CNN ensemble…")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Training v22 baseline (single CW with intervals feature)…")
    t0 = time.time()
    cw_v22 = build_metadata_hybrid(train_pool, mode="intervals")
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training 3-seed CW ensemble (each = metadata hybrid)…")
    cw_models = []
    for s in CW_SEEDS:
        t0 = time.time()
        cw_models.append(build_metadata_hybrid_with_seed(train_pool, "intervals", s))
        print(f"    cw seed={s} fit {time.time() - t0:.1f}s")
    cw_v24 = CWEnsemble(cw_models)

    def predictor(cw_obj):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_v14_with_meta(sub_tr_x, sub_tr_y, sub_te_x, cw_obj,
                                          cnn_models, info, info.get("metric_type", "ALL"))
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    print("\n>>> Eval [v22 single CW + intervals feat]…")
    rep_v22 = evaluate(predictor(cw_v22), holdout)
    print_summary(rep_v22, name="v22 single CW")

    print(">>> Eval [v24 3-seed CW ensemble + intervals feat]…")
    rep_v24 = evaluate(predictor(cw_v24), holdout)
    print_summary(rep_v24, name="v24 3-seed CW")

    delta = rep_v24["overall_f1"] - rep_v22["overall_f1"]
    print(f"\n  Δ (v24 − v22) = {delta:+.4f}")

    report = {
        "v22_baseline_f1": rep_v22["overall_f1"],
        "v24_f1": rep_v24["overall_f1"],
        "delta": delta,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v24_cw_ensemble")
    return report, cw_v24, cnn_models


def generate_submission(cw_v24, cnn_models,
                        output: Path = Path("submission_cw_ensemble.json")) -> Path:
    print(f"\n>>> Re-training on ALL 1000 windows (3 seeds)…")
    full_models = []
    for s in CW_SEEDS:
        t0 = time.time()
        full_models.append(build_metadata_hybrid_with_seed(all_window_dirs(), "intervals", s))
        print(f"    cw seed={s} fit {time.time() - t0:.1f}s")
    cw_full = CWEnsemble(full_models)

    Xc, yc = v6_pool(all_window_dirs())
    cnn_full = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_full.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_v14_with_meta(w.train_x, w.train_y, w.test_x, cw_full, cnn_full,
                                      w.info, w.metric_type)
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
    rep, cw_v24, cnn_models = run_validation()
    if rep["delta"] > 0.001:
        print(f"\n3-seed CW ensemble beats single CW by {rep['delta']:+.4f}; generating submission.")
        generate_submission(cw_v24, cnn_models)
    else:
        print(f"\n3-seed CW ensemble did not meaningfully help "
              f"(Δ = {rep['delta']:+.4f}); submission NOT generated.")
