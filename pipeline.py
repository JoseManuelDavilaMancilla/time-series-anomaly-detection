"""
author v3 — Per-metric_type cross-window models.

Hypothesis: SuccessRate / LatencySecond / ResourceUtilizationRate / ErrorCount /
QPS / Count are very different distributions. One global RF cannot learn that
"a SuccessRate anomaly is a tiny dip below 1.0" while "an ErrorCount anomaly
is a spike above 0". Training 6 separate models — one per metric_type — gives
each model a coherent target. Average ~125 anomalous windows per type, plenty
of data to fit on.

We try both RF and LGBM backends to pick the winner. Segment selection from v1
is applied to both.

Run:  uv run python v3_per_metric.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

# Silence the harmless feature-name warning from sklearn when feeding numpy arrays to LGBM.
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


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training RF (global) — control…")
    t0 = time.time()
    cw_rf_global = CrossWindowModel(backend="rf", per_metric=False,
                                    n_estimators=200, max_depth=12).fit(train_pool)
    print(f"    rf global fit {time.time() - t0:.1f}s")

    print(">>> Training RF (per metric_type)…")
    t0 = time.time()
    cw_rf_per = CrossWindowModel(backend="rf", per_metric=True,
                                 n_estimators=200, max_depth=12).fit(train_pool)
    print(f"    rf per-metric fit {time.time() - t0:.1f}s "
          f"({len(cw_rf_per._models)} sub-models: {list(cw_rf_per._models)})")

    print(">>> Training LGBM (per metric_type)…")
    t0 = time.time()
    cw_lgbm_per = CrossWindowModel(backend="lgbm", per_metric=True,
                                   n_estimators=400, max_depth=8,
                                   min_samples_leaf=10).fit(train_pool)
    print(f"    lgbm per-metric fit {time.time() - t0:.1f}s")

    def predictor_for(cw):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores, _ = v8_style_scores(sub_tr_x, sub_tr_y, sub_te_x, cw,
                                        metric_type=info.get("metric_type", "ALL"))
            k = int(round(len(sub_te_x) * ratio))
            return predict_segments(scores, k, **SEG_KWARGS)
        return pred

    print(">>> Eval: RF global + segments (v1 baseline)…")
    rep_global = evaluate(predictor_for(cw_rf_global), holdout)
    print_summary(rep_global, name="RF global + segments")

    print(">>> Eval: RF per-metric + segments…")
    rep_rf_per = evaluate(predictor_for(cw_rf_per), holdout)
    print_summary(rep_rf_per, name="RF per-metric + segments")

    print(">>> Eval: LGBM per-metric + segments…")
    rep_lgbm_per = evaluate(predictor_for(cw_lgbm_per), holdout)
    print_summary(rep_lgbm_per, name="LGBM per-metric + segments")

    # ── Hybrid: route to per-metric only for the metric_types where it actually wins.
    # Threshold = +0.005 to ignore noise-level wins.
    specialized = frozenset(
        mt for mt in rep_rf_per["by_metric_type"]
        if rep_rf_per["by_metric_type"][mt]["mean_f1"]
           - rep_global["by_metric_type"].get(mt, {"mean_f1": 0})["mean_f1"]
           > 0.005
    )
    print(f"\n>>> Hybrid specialization set (per-metric > global by >0.005): {sorted(specialized)}")
    cw_hybrid = HybridCrossWindowModel(
        global_model=cw_rf_global,
        per_metric_model=cw_rf_per,
        specialized_types=specialized,
    )
    print(">>> Eval: Hybrid (per-metric for winners, global otherwise) + segments…")
    rep_hybrid = evaluate(predictor_for(cw_hybrid), holdout)
    print_summary(rep_hybrid, name="Hybrid + segments")

    print(f"\n    Δ (RF per-metric − RF global)   = {rep_rf_per['overall_f1'] - rep_global['overall_f1']:+.4f}")
    print(f"    Δ (LGBM per-metric − RF global) = {rep_lgbm_per['overall_f1'] - rep_global['overall_f1']:+.4f}")
    print(f"    Δ (Hybrid − RF global)          = {rep_hybrid['overall_f1'] - rep_global['overall_f1']:+.4f}")

    # Pick the winner
    candidates = [
        ("rf_global", rep_global["overall_f1"], "rf", False, None),
        ("rf_per_metric", rep_rf_per["overall_f1"], "rf", True, None),
        ("lgbm_per_metric", rep_lgbm_per["overall_f1"], "lgbm", True, None),
        ("hybrid", rep_hybrid["overall_f1"], "rf", "hybrid", specialized),
    ]
    candidates.sort(key=lambda x: -x[1])
    winner_name, winner_f1, winner_backend, winner_mode, winner_special = candidates[0]
    print(f"\n    Winner: {winner_name} (F1={winner_f1:.4f})")

    report = {
        "rf_global": rep_global,
        "rf_per_metric": rep_rf_per,
        "lgbm_per_metric": rep_lgbm_per,
        "hybrid": rep_hybrid,
        "winner": winner_name,
        "winner_config": {
            "backend": winner_backend,
            "mode": winner_mode,
            "specialized_types": sorted(winner_special) if winner_special else None,
        },
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v3_per_metric")
    return report


def _train_full(backend: str, mode):
    """mode is True/False for plain per-metric, or 'hybrid' for the routed model."""
    if mode == "hybrid":
        print(">>> Training full data: RF global + RF per-metric (for hybrid)…")
        t0 = time.time()
        g = CrossWindowModel(backend="rf", per_metric=False,
                             n_estimators=200, max_depth=12).fit(all_window_dirs())
        p = CrossWindowModel(backend="rf", per_metric=True,
                             n_estimators=200, max_depth=12).fit(all_window_dirs())
        print(f"    fit time {time.time() - t0:.1f}s")
        return g, p
    print(f"\n>>> Training cross-window {backend} (per_metric={mode}) on ALL 1000 windows…")
    t0 = time.time()
    cw = CrossWindowModel(
        backend=backend, per_metric=bool(mode),
        n_estimators=400 if backend == "lgbm" else 200,
        max_depth=8 if backend == "lgbm" else 12,
        min_samples_leaf=10 if backend == "lgbm" else 3,
    ).fit(all_window_dirs())
    print(f"    fit time {time.time() - t0:.1f}s  models: {list(cw._models)}")
    return cw


def generate_submission(backend: str, mode, specialized_types,
                        output: Path = Path("submission_per_metric.json")) -> Path:
    trained = _train_full(backend, mode)
    if mode == "hybrid":
        g, p = trained
        cw = HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                    specialized_types=frozenset(specialized_types or []))
    else:
        cw = trained

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores, _ = v8_style_scores(w.train_x, w.train_y, w.test_x, cw,
                                    metric_type=w.metric_type)
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
    cfg = rep["winner_config"]
    if rep["winner"] == "rf_global":
        print("\n!! No variant beat the global model; submission NOT generated.")
    else:
        out_name = "submission_per_metric.json" if cfg["mode"] != "hybrid" else "submission_hybrid_per_metric.json"
        print(f"\nGenerating {out_name} with {rep['winner']}…")
        generate_submission(cfg["backend"], cfg["mode"], cfg.get("specialized_types"),
                            output=Path(out_name))
