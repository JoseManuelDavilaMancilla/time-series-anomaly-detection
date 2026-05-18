"""
author v8 — 5-seed CNN ensemble (extension of v7).

v7 (3 seeds) gave +0.0110 over hybrid. Does adding 2 more seeds help, or are
we already at the variance-reduction asymptote?

If 5-seed beats 3-seed by ≥0.002, this becomes the new best submission.

Run:  uv run python v8_cnn5.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from shared_lib import (
    CrossWindowModel,
    HybridCrossWindowModel,
    predict_segments,
    v8_style_scores,
)
from v6_cnn import (
    SEG_KWARGS,
    SPECIALIZED,
    build_contexts,
    build_training_pool,
    cnn_score,
    fit_cnn,
)
from v7_cnn_ensemble import (
    build_hybrid,
    fit_cnn_with_seed,
    ensemble_cnn_score,
    scores_weighted,
)
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)

SEEDS_3 = (42, 123, 7)
SEEDS_5 = (42, 123, 7, 999, 2024)
SEEDS_7 = (42, 123, 7, 999, 2024, 31337, 1729)


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training hybrid cross-window…")
    t0 = time.time()
    cw = build_hybrid(train_pool)
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Building CNN pool…")
    X, y = build_training_pool(train_pool)
    print(f"    X.shape={X.shape}  y.mean={y.mean():.3f}")

    print(">>> Training 7 CNNs (we'll evaluate 3/5/7-seed variants on the same models)…")
    models = []
    for s in SEEDS_7:
        print(f">>> CNN seed={s}")
        t0 = time.time()
        models.append(fit_cnn_with_seed(X, y, s))
        print(f"    fit {time.time() - t0:.1f}s")

    def predictor_for(model_subset):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_weighted(sub_tr_x, sub_tr_y, sub_te_x, cw, model_subset,
                                     info.get("metric_type", "ALL"))
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        return pred

    # Baseline: hybrid + segments only
    def pred_baseline(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
        scores, _ = v8_style_scores(sub_tr_x, sub_tr_y, sub_te_x, cw,
                                    metric_type=info.get("metric_type", "ALL"))
        return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

    print("\n>>> Eval: v3 hybrid + segments (baseline)…")
    rep_v3 = evaluate(pred_baseline, holdout)
    print_summary(rep_v3, name="v3 baseline")

    print(">>> Eval: 1-seed CNN…")
    rep_1 = evaluate(predictor_for(models[:1]), holdout)
    print_summary(rep_1, name="1-seed")

    print(">>> Eval: 3-seed CNN…")
    rep_3 = evaluate(predictor_for(models[:3]), holdout)
    print_summary(rep_3, name="3-seed")

    print(">>> Eval: 5-seed CNN…")
    rep_5 = evaluate(predictor_for(models[:5]), holdout)
    print_summary(rep_5, name="5-seed")

    print(">>> Eval: 7-seed CNN…")
    rep_7 = evaluate(predictor_for(models[:7]), holdout)
    print_summary(rep_7, name="7-seed")

    print(f"\n  Δ (1-seed − v3) = {rep_1['overall_f1'] - rep_v3['overall_f1']:+.4f}")
    print(f"  Δ (3-seed − v3) = {rep_3['overall_f1'] - rep_v3['overall_f1']:+.4f}")
    print(f"  Δ (5-seed − v3) = {rep_5['overall_f1'] - rep_v3['overall_f1']:+.4f}")
    print(f"  Δ (7-seed − v3) = {rep_7['overall_f1'] - rep_v3['overall_f1']:+.4f}")
    print(f"  Δ (5-seed − 3-seed) = {rep_5['overall_f1'] - rep_3['overall_f1']:+.4f}")
    print(f"  Δ (7-seed − 5-seed) = {rep_7['overall_f1'] - rep_5['overall_f1']:+.4f}")

    # Pick winner from {1, 3, 5, 7}-seed
    candidates = [(1, rep_1), (3, rep_3), (5, rep_5), (7, rep_7)]
    candidates.sort(key=lambda kv: -kv[1]["overall_f1"])
    best_n, best_rep = candidates[0]
    print(f"\n  Best ensemble size: {best_n} seeds  F1={best_rep['overall_f1']:.4f}")

    report = {
        "v3_baseline": rep_v3,
        "seed_1": rep_1,
        "seed_3": rep_3,
        "seed_5": rep_5,
        "seed_7": rep_7,
        "winner_n": best_n,
        "deltas": {
            "1_minus_v3": rep_1["overall_f1"] - rep_v3["overall_f1"],
            "3_minus_v3": rep_3["overall_f1"] - rep_v3["overall_f1"],
            "5_minus_v3": rep_5["overall_f1"] - rep_v3["overall_f1"],
            "7_minus_v3": rep_7["overall_f1"] - rep_v3["overall_f1"],
        },
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v8_cnn5")
    return report


def generate_submission(n_seeds: int,
                        output: Path = Path("submission_cnn_5seed.json")) -> Path:
    seeds = SEEDS_7[:n_seeds]
    print(f"\n>>> Training hybrid on ALL 1000 windows…")
    t0 = time.time()
    cw = build_hybrid(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Building full CNN pool…")
    X, y = build_training_pool(all_window_dirs())
    print(f"    X.shape={X.shape}  y.mean={y.mean():.3f}")

    models = []
    for s in seeds:
        print(f">>> CNN seed={s} (full data)")
        t0 = time.time()
        models.append(fit_cnn_with_seed(X, y, s))
        print(f"    fit {time.time() - t0:.1f}s")

    print(f">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_weighted(w.train_x, w.train_y, w.test_x, cw, models,
                                 w.metric_type)
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
    return output


if __name__ == "__main__":
    rep = run_validation()
    best_n = rep["winner_n"]
    # Only generate if a bigger ensemble beats 3-seed by ≥0.001
    if best_n > 3 and rep[f"seed_{best_n}"]["overall_f1"] - rep["seed_3"]["overall_f1"] > 0.001:
        out_name = f"submission_cnn_{best_n}seed.json"
        print(f"\n{best_n}-seed beats 3-seed; generating {out_name}…")
        generate_submission(best_n, output=Path(out_name))
    else:
        print(f"\nMore seeds did not meaningfully help (best={best_n}); "
              "keeping submission_cnn_ensemble.json (3-seed) as best.")
