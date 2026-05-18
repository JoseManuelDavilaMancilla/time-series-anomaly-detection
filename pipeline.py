"""
author v11 — Ablation of teammate's v12 changes inside the author pipeline.

teammate's v12 (0.5665 LB) changed three things from v8:
  A. Stronger CW: 500 trees / depth 15 (vs 200 / 12)
  B. IsolationForest-on-test replaces online_ensemble for `disjoint` AND
     `constant_train` categories.
  C. Reweighting: disjoint = 0.35 cw + 0.30 g + 0.35 if (vs 0.30 / 0.30 / 0.40),
                   constant_train = 0.50 cw + 0.50 if (vs 0.40 / 0.60).

We don't know which of A/B/C contributed teammate's +0.018, and we don't know
whether they still help on top of the author stack (hybrid CW + 3-seed CNN
ensemble + tuned segments, currently 0.6111 LB).

Six variants tested, all using the v9 segtuned segment params:

  v_base       — current author best (0.6111 LB baseline)
  +stronger_cw — A only
  +if_disjoint — B (disjoint windows only)
  +if_both     — B (disjoint + constant_train)
  +v12_weights — C only
  +all_v12     — A + B(both) + C

Run:  uv run python v11_kimi_v12_ablation.py
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
    categorize_window,
    global_distance_score,
    isolation_forest_test,
    normalize_scores,
    online_ensemble,
    per_window_rf_score,
    predict_segments,
)
from v6_cnn import (
    SPECIALIZED,
    build_training_pool,
    cnn_score,
)
from v7_cnn_ensemble import (
    CNN_SEEDS,
    ensemble_cnn_score,
    fit_cnn_with_seed,
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


def build_hybrid_strong(window_dirs, n_estimators: int, max_depth: int,
                        min_samples_leaf: int = 3):
    """Hybrid CW with configurable strength."""
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


def scores_for_config(train_x, train_y, test_x, cw, cnn_models, metric_type, *,
                      if_on_disjoint: bool, if_on_constant: bool,
                      use_v12_weights: bool) -> np.ndarray:
    """Compute ensemble scores under a given v11 configuration."""
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw.predict_proba(test_x, metric_type=metric_type))
    cnn_s = normalize_scores(ensemble_cnn_score(cnn_models, test_x))

    if category == "constant_train":
        if if_on_constant:
            if_s = normalize_scores(isolation_forest_test(test_x, train_y))
            # v12 weights: 0.5 cw + 0.5 if; we additionally fold in the CNN
            if use_v12_weights:
                scores = (0.50 - CNN_WEIGHT) * cw_s + 0.50 * if_s + CNN_WEIGHT * cnn_s
            else:
                # Substitute IF for online but keep author weights (0.4 cw + 0.6 local)
                scores = 0.40 * cw_s + (0.60 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
        else:
            local = normalize_scores(online_ensemble(test_x, window=15))
            scores = 0.40 * cw_s + (0.60 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn_s

    elif category == "disjoint":
        g_s = normalize_scores(global_distance_score(train_x, test_x))
        if if_on_disjoint:
            if_s = normalize_scores(isolation_forest_test(test_x, train_y))
            if use_v12_weights:
                # v12: 0.35 cw + 0.30 global + 0.35 if; fold in CNN by trimming if
                scores = 0.35 * cw_s + 0.30 * g_s + (0.35 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
            else:
                # Keep author weights (0.30 / 0.30 / 0.40) but substitute IF for online
                scores = 0.30 * cw_s + 0.30 * g_s + (0.40 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
        else:
            local = normalize_scores(online_ensemble(test_x, window=15))
            scores = 0.30 * cw_s + 0.30 * g_s + (0.40 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn_s

    elif category == "partial_overlap":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        scores = 0.35 * cw_s + 0.35 * pw + (0.30 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn_s

    elif category == "test_within_train":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        scores = (0.50 - CNN_WEIGHT) * cw_s + 0.50 * pw + CNN_WEIGHT * cnn_s
    else:
        scores = np.zeros(len(test_x))
    return scores


CONFIGS = [
    # name              if_disj   if_const  v12_w
    ("base",            False,    False,    False),
    ("stronger_cw",     False,    False,    False),   # special: bigger CW
    ("if_disjoint",     True,     False,    False),
    ("if_both",         True,     True,     False),
    ("v12_weights",     False,    False,    True),    # weights only, no IF
    ("all_v12",         True,     True,     True),    # full v12 + CNN + segments
]


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training BASELINE hybrid CW (200/12/3)…")
    t0 = time.time()
    cw_base = build_hybrid_strong(train_pool, n_estimators=200, max_depth=12)
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Training STRONGER hybrid CW (500/15/3)…")
    t0 = time.time()
    cw_strong = build_hybrid_strong(train_pool, n_estimators=500, max_depth=15)
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Building CNN pool + training 3 CNNs…")
    X, y = build_training_pool(train_pool)
    print(f"    pool X.shape={X.shape}  y.mean={y.mean():.3f}")
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(X, y, s))
        print(f"    cnn seed={s}  fit {time.time() - t0:.1f}s")

    print(f"\n>>> Running {len(CONFIGS)} configurations…")
    results = {}
    for name, if_d, if_c, v12_w in CONFIGS:
        cw = cw_strong if name in ("stronger_cw", "all_v12") else cw_base

        def predictor(sub_tr_x, sub_tr_y, sub_te_x, info, ratio,
                      _cw=cw, _if_d=if_d, _if_c=if_c, _v12_w=v12_w):
            scores = scores_for_config(
                sub_tr_x, sub_tr_y, sub_te_x, _cw, cnn_models,
                info.get("metric_type", "ALL"),
                if_on_disjoint=_if_d, if_on_constant=_if_c, use_v12_weights=_v12_w,
            )
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)

        print(f"\n>>> [{name}]")
        rep = evaluate(predictor, holdout)
        print_summary(rep, name=name)
        results[name] = rep

    print("\n──────  ablation summary ──────")
    base_f1 = results["base"]["overall_f1"]
    rows = sorted(results.items(), key=lambda kv: -kv[1]["overall_f1"])
    for name, rep in rows:
        d = rep["overall_f1"] - base_f1
        print(f"  {name:<14}  F1={rep['overall_f1']:.4f}  Δ_vs_base={d:+.4f}")

    winner_name, winner_rep = rows[0]
    print(f"\n  Winner: {winner_name}  F1={winner_rep['overall_f1']:.4f}")

    report = {
        "configs": {name: r["overall_f1"] for name, r in results.items()},
        "per_config_per_window": {name: r["per_window"] for name, r in results.items()},
        "winner": winner_name,
        "delta_winner_vs_base": winner_rep["overall_f1"] - base_f1,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v11_kimi_v12_ablation")
    return report


def generate_submission(config_name: str,
                        output: Path = Path("submission_kimi_combo.json")) -> Path:
    cfg = next(c for c in CONFIGS if c[0] == config_name)
    name, if_d, if_c, v12_w = cfg

    print(f"\n>>> Training hybrid CW on ALL 1000 windows…")
    t0 = time.time()
    if name in ("stronger_cw", "all_v12"):
        cw = build_hybrid_strong(all_window_dirs(), n_estimators=500, max_depth=15)
    else:
        cw = build_hybrid_strong(all_window_dirs(), n_estimators=200, max_depth=12)
    print(f"    fit {time.time() - t0:.1f}s")

    print(">>> Building full CNN pool + 3 CNNs…")
    X, y = build_training_pool(all_window_dirs())
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(X, y, s))
        print(f"    cnn seed={s}  fit {time.time() - t0:.1f}s")

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_for_config(
            w.train_x, w.train_y, w.test_x, cw, cnn_models, w.metric_type,
            if_on_disjoint=if_d, if_on_constant=if_c, use_v12_weights=v12_w,
        )
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
    if rep["winner"] != "base" and rep["delta_winner_vs_base"] > 0.001:
        out_name = f"submission_kimi_combo_{rep['winner']}.json"
        print(f"\nWinner {rep['winner']} beats base by {rep['delta_winner_vs_base']:+.4f}; "
              f"generating {out_name}…")
        generate_submission(rep["winner"], output=Path(out_name))
    else:
        print(f"\nNo v12 change improved on the current author pipeline. "
              "submission NOT generated.")
