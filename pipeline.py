"""
author v4 — CUSUM channel for disjoint / constant_train windows.

Hypothesis: for windows where the test series sits at a level the training
never visited (disjoint) or where training is flat (constant_train), the
real anomaly signal is "the level shifted persistently", not "this single
point is far from its neighbors". The online rolling-z and global-distance
scorers we already use are both pointwise; they miss sustained shifts.

CUSUM accumulates departures from train mean/std, so a sustained level
change drives the CUSUM statistic up monotonically and lights up the
entire shifted region.

Builds on top of v3's hybrid cross-window + segment selection. Activates
CUSUM only for the two categories above (where it should help) so we don't
disturb the other 730 windows that already work.

Run:  uv run python v4_cusum.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from shared_lib import (
    CrossWindowModel,
    HybridCrossWindowModel,
    predict_segments,
    v8_style_scores,
)
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

SEG_KWARGS = dict(smooth=5, thr_frac=0.6, small_k_cutoff=4, max_seg=80)
SPECIALIZED = frozenset({"ErrorCount", "ResourceUtilizationRate", "SuccessRate"})


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

    def make_predictor(use_cusum: bool):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores, _ = v8_style_scores(
                sub_tr_x, sub_tr_y, sub_te_x, cw,
                metric_type=info.get("metric_type", "ALL"),
                use_cusum=use_cusum,
            )
            k = int(round(len(sub_te_x) * ratio))
            return predict_segments(scores, k, **SEG_KWARGS)
        return pred

    print(">>> Eval: hybrid + segments  (v3 baseline)…")
    rep_v3 = evaluate(make_predictor(use_cusum=False), holdout)
    print_summary(rep_v3, name="v3 hybrid + segments")

    print(">>> Eval: hybrid + segments + CUSUM (v4)…")
    rep_v4 = evaluate(make_predictor(use_cusum=True), holdout)
    print_summary(rep_v4, name="v4 hybrid + segments + CUSUM")

    delta = rep_v4["overall_f1"] - rep_v3["overall_f1"]
    print(f"    Δ (v4 − v3) = {delta:+.4f}")

    report = {
        "v3_hybrid": rep_v3,
        "v4_with_cusum": rep_v4,
        "delta_v4_vs_v3": delta,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v4_cusum")
    return report


def generate_submission(output: Path = Path("submission_cusum.json")) -> Path:
    print("\n>>> Training hybrid on ALL 1000 windows…")
    t0 = time.time()
    cw = build_hybrid(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Generating predictions with CUSUM…")
    preds: dict[str, list[int]] = {}
    cusum_categories = {"disjoint": 0, "constant_train": 0}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores, cat = v8_style_scores(
            w.train_x, w.train_y, w.test_x, cw,
            metric_type=w.metric_type, use_cusum=True,
        )
        if cat in cusum_categories:
            cusum_categories[cat] += 1
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
    print(f">>> CUSUM-affected windows: {cusum_categories}")
    return output


if __name__ == "__main__":
    rep = run_validation()
    if rep["delta_v4_vs_v3"] >= -0.003:
        print("\nCUSUM ≥ no-CUSUM (within noise). Generating submission.")
        generate_submission()
    else:
        print(f"\n!! CUSUM hurt by {-rep['delta_v4_vs_v3']:.4f}; submission NOT generated.")
