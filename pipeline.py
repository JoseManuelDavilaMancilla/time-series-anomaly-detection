"""
author v13 — Gradient-boosting cross-window ensemble with seed/hyperparam hacking.

Background: single LightGBM cross-window lost to single RF by −0.013 on validation
(v2_lgbm). Single XGBoost per-window lost to single RF per-window by −0.018
(teammate v14). So at single-model level, RF dominates.

But: RF is itself a bagging ensemble. A 5-seed GBM ensemble might catch up via
variance reduction (analogous to how 3-seed CNN +0.011 over single-seed CNN).
Worth checking before fully writing off GBM at this scale.

Three configurations tested:
  A. 5-seed LGBM (homogeneous params, just different `random_state`)
  B. 5-seed LGBM (diverse params — different lr, num_leaves, feature_fraction)
  C. 5-seed XGBoost (homogeneous params)

For each: compute predictions, normalize, average across seeds, and use as the
"CW" channel in the v11 all_v12 ensemble (otherwise unchanged).

Compare against the current best (single hybrid RF CW + 3-seed CNN, val 0.9639).

Run:  uv run python v13_gbm_seedhack.py
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
warnings.filterwarnings("ignore", category=UserWarning)

from shared_lib import (
    CrossWindowModel,
    HybridCrossWindowModel,
    categorize_window,
    extract_features,
    global_distance_score,
    isolation_forest_test,
    normalize_scores,
    online_ensemble,
    per_window_rf_score,
    predict_segments,
)
from v6_cnn import build_training_pool as v6_pool, SPECIALIZED
from v7_cnn_ensemble import CNN_SEEDS, ensemble_cnn_score, fit_cnn_with_seed
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
N_GBM_SEEDS = 5
GBM_SEEDS = (42, 123, 7, 999, 2024)

# Hyperparameter variants for the diverse LGBM ensemble
# Hyperparameter diversity via learning_rate × num_leaves × min_child_samples
# (feature/bagging-fraction removed — they triggered a libomp hang on macOS)
LGBM_DIVERSE_CONFIGS = [
    dict(learning_rate=0.05, num_leaves=31, min_child_samples=40, n_estimators=600),
    dict(learning_rate=0.03, num_leaves=63, min_child_samples=20, n_estimators=800),
    dict(learning_rate=0.07, num_leaves=15, min_child_samples=80, n_estimators=500),
    dict(learning_rate=0.05, num_leaves=47, min_child_samples=10, n_estimators=600),
    dict(learning_rate=0.04, num_leaves=23, min_child_samples=60, n_estimators=700),
]


def _build_pool(window_dirs):
    X_all, y_all = [], []
    for wdir in window_dirs:
        train_y = np.load(wdir / "train_label.npy")
        if train_y.sum() == 0:
            continue
        train_x = np.load(wdir / "train.npy")
        X_all.append(extract_features(train_x, include_value=False))
        y_all.append(train_y)
    return np.vstack(X_all), np.hstack(y_all)


def fit_lgbm(X, y, seed: int, *, learning_rate=0.05, num_leaves=31,
             n_estimators=600, max_depth=8, min_child_samples=40):
    """Train one LightGBM. Single-threaded (n_jobs=1) to avoid the thread-pool
    deadlock we hit with n_jobs=2 + feature_fraction<1.0 on this machine.
    Single-threaded training of 218k samples × 27 features × 600 trees is
    ~30–60s, so 5 seeds total is 3–5 min."""
    import lightgbm as lgb
    import sys
    t0 = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        learning_rate=learning_rate,
        # NOTE: do not pass feature_fraction or bagging_fraction here — they
        # combined with multi-threading cause hangs on macOS via the libomp build.
        is_unbalance=True,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )
    clf.fit(X, y)
    print(f"      lgbm seed={seed} fit_internal {time.time() - t0:.1f}s", flush=True)
    sys.stdout.flush()
    return clf


def fit_xgb(X, y, seed: int, *, n_estimators=500, max_depth=6,
            learning_rate=0.05):
    from xgboost import XGBClassifier
    import sys
    t0 = time.time()
    pos_weight = float((y == 0).sum()) / max(1, (y == 1).sum())
    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=pos_weight,
        random_state=seed,
        n_jobs=1,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X, y)
    print(f"      xgb seed={seed} fit_internal {time.time() - t0:.1f}s", flush=True)
    sys.stdout.flush()
    return clf


def gbm_ensemble_predict(models: List, test_x: np.ndarray) -> np.ndarray:
    X = extract_features(test_x, include_value=False)
    probs = []
    for m in models:
        p = m.predict_proba(X)
        probs.append(p[:, 1] if p.shape[1] > 1 else np.zeros(len(test_x)))
    return np.mean(probs, axis=0)


def scores_with_cw(cw_score_fn, train_x, train_y, test_x, cnn_models,
                   metric_type) -> np.ndarray:
    """v11 all_v12 ensemble template where the CW channel is parameterized."""
    category = categorize_window(train_x, test_x)
    cw_s = normalize_scores(cw_score_fn(test_x, metric_type))
    cnn_s = normalize_scores(ensemble_cnn_score(cnn_models, test_x))

    if category == "constant_train":
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * if_s + CNN_WEIGHT * cnn_s
    if category == "disjoint":
        g_s = normalize_scores(global_distance_score(train_x, test_x))
        if_s = normalize_scores(isolation_forest_test(test_x, train_y))
        return 0.35 * cw_s + 0.30 * g_s + (0.35 - CNN_WEIGHT) * if_s + CNN_WEIGHT * cnn_s
    if category == "partial_overlap":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        local = normalize_scores(online_ensemble(test_x, window=15))
        return 0.35 * cw_s + 0.35 * pw + (0.30 - CNN_WEIGHT) * local + CNN_WEIGHT * cnn_s
    if category == "test_within_train":
        pw = normalize_scores(per_window_rf_score(train_x, train_y, test_x))
        return (0.50 - CNN_WEIGHT) * cw_s + 0.50 * pw + CNN_WEIGHT * cnn_s
    return np.zeros(len(test_x))


def build_hybrid_rf(window_dirs):
    g = CrossWindowModel(backend="rf", per_metric=False,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    p = CrossWindowModel(backend="rf", per_metric=True,
                         n_estimators=500, max_depth=15).fit(window_dirs)
    return HybridCrossWindowModel(global_model=g, per_metric_model=p,
                                  specialized_types=SPECIALIZED)


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training BASELINE hybrid RF (500/15/3)…")
    t0 = time.time()
    cw_rf = build_hybrid_rf(train_pool)
    print(f"    rf fit {time.time() - t0:.1f}s")

    print(">>> Training 3-seed CNN ensemble…")
    Xc, yc = v6_pool(train_pool)
    print(f"    cnn pool X={Xc.shape}  y.mean={yc.mean():.3f}")
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    print(">>> Building GBM training pool (scale-invariant features)…")
    Xg, yg = _build_pool(train_pool)
    print(f"    GBM pool X={Xg.shape}  y.mean={yg.mean():.3f}")

    print(f">>> Training 5-seed LGBM (HOMOGENEOUS)…")
    lgbm_hom = []
    for s in GBM_SEEDS:
        t0 = time.time()
        lgbm_hom.append(fit_lgbm(Xg, yg, seed=s))
        print(f"    lgbm_hom seed={s}  fit {time.time() - t0:.1f}s")

    print(f">>> Training 5-seed LGBM (DIVERSE hyperparameters)…")
    lgbm_div = []
    for s, cfg in zip(GBM_SEEDS, LGBM_DIVERSE_CONFIGS):
        t0 = time.time()
        lgbm_div.append(fit_lgbm(Xg, yg, seed=s, **cfg))
        print(f"    lgbm_div seed={s} cfg={cfg}  fit {time.time() - t0:.1f}s")

    # XGBoost segfaults on this macOS install — skipping. See EXPERIMENTS.md.
    xgb_hom = []

    # CW score callables — each returns proba for the test_x window
    def cw_rf_score(test_x, metric_type):
        return cw_rf.predict_proba(test_x, metric_type=metric_type)

    def cw_lgbm_hom_score(test_x, metric_type):
        return gbm_ensemble_predict(lgbm_hom, test_x)

    def cw_lgbm_div_score(test_x, metric_type):
        return gbm_ensemble_predict(lgbm_div, test_x)

    def cw_xgb_hom_score(test_x, metric_type):
        return gbm_ensemble_predict(xgb_hom, test_x)

    def cw_rf_lgbm_blend_score(test_x, metric_type):
        # 50/50 RF + LGBM diverse
        rf_p = cw_rf.predict_proba(test_x, metric_type=metric_type)
        lgbm_p = gbm_ensemble_predict(lgbm_div, test_x)
        return 0.5 * normalize_scores(rf_p) + 0.5 * normalize_scores(lgbm_p)

    variants = [
        ("rf_baseline", cw_rf_score),
        ("lgbm_5seed_hom", cw_lgbm_hom_score),
        ("lgbm_5seed_diverse", cw_lgbm_div_score),
        # ("xgb_5seed_hom", cw_xgb_hom_score),  # XGBoost segfaults on this build
        ("rf_plus_lgbm_blend", cw_rf_lgbm_blend_score),
    ]

    results = {}
    for name, fn in variants:
        def predictor(sub_tr_x, sub_tr_y, sub_te_x, info, ratio, _fn=fn):
            scores = scores_with_cw(_fn, sub_tr_x, sub_tr_y, sub_te_x, cnn_models,
                                    info.get("metric_type", "ALL"))
            return predict_segments(scores, int(round(len(sub_te_x) * ratio)), **SEG_KWARGS)
        print(f"\n>>> Eval [{name}]")
        rep = evaluate(predictor, holdout)
        print_summary(rep, name=name)
        results[name] = rep

    print("\n──────  variant summary ──────")
    base = results["rf_baseline"]["overall_f1"]
    rows = sorted(results.items(), key=lambda kv: -kv[1]["overall_f1"])
    for name, rep in rows:
        d = rep["overall_f1"] - base
        print(f"  {name:<22}  F1={rep['overall_f1']:.4f}  Δ_vs_rf={d:+.4f}")

    winner_name = rows[0][0]
    winner_f1 = rows[0][1]["overall_f1"]
    delta_vs_base = winner_f1 - base

    report = {
        "variants": {name: r["overall_f1"] for name, r in results.items()},
        "winner": winner_name,
        "delta_winner_vs_rf_baseline": delta_vs_base,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v13_gbm_seedhack")
    return report


def generate_submission(strategy: str,
                        output: Path = Path("submission_gbm_seedhack.json")) -> Path:
    print(f"\n>>> Training all models on ALL 1000 windows (strategy={strategy})…")
    t0 = time.time()
    cw_rf = build_hybrid_rf(all_window_dirs())
    print(f"    rf fit {time.time() - t0:.1f}s")

    Xc, yc = v6_pool(all_window_dirs())
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    Xg, yg = _build_pool(all_window_dirs())
    gbm_models = []
    if strategy in ("lgbm_5seed_hom", "rf_plus_lgbm_blend"):
        cfgs = [{} for _ in GBM_SEEDS] if strategy == "lgbm_5seed_hom" else LGBM_DIVERSE_CONFIGS
        for s, cfg in zip(GBM_SEEDS, cfgs):
            t0 = time.time()
            gbm_models.append(fit_lgbm(Xg, yg, seed=s, **cfg))
            print(f"    gbm seed={s}  fit {time.time() - t0:.1f}s")
    elif strategy == "lgbm_5seed_diverse":
        for s, cfg in zip(GBM_SEEDS, LGBM_DIVERSE_CONFIGS):
            t0 = time.time()
            gbm_models.append(fit_lgbm(Xg, yg, seed=s, **cfg))
            print(f"    gbm seed={s}  fit {time.time() - t0:.1f}s")
    elif strategy == "xgb_5seed_hom":
        for s in GBM_SEEDS:
            t0 = time.time()
            gbm_models.append(fit_xgb(Xg, yg, seed=s))
            print(f"    xgb seed={s}  fit {time.time() - t0:.1f}s")

    def cw_score_fn(test_x, metric_type):
        if strategy == "rf_baseline":
            return cw_rf.predict_proba(test_x, metric_type=metric_type)
        if strategy == "rf_plus_lgbm_blend":
            rf_p = cw_rf.predict_proba(test_x, metric_type=metric_type)
            lgbm_p = gbm_ensemble_predict(gbm_models, test_x)
            return 0.5 * normalize_scores(rf_p) + 0.5 * normalize_scores(lgbm_p)
        return gbm_ensemble_predict(gbm_models, test_x)

    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores = scores_with_cw(cw_score_fn, w.train_x, w.train_y, w.test_x,
                                cnn_models, w.metric_type)
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
    if rep["winner"] != "rf_baseline" and rep["delta_winner_vs_rf_baseline"] > 0.002:
        out_name = f"submission_{rep['winner']}.json"
        print(f"\nWinner {rep['winner']} beats RF baseline by "
              f"{rep['delta_winner_vs_rf_baseline']:+.4f}; generating {out_name}…")
        generate_submission(rep["winner"], output=Path(out_name))
    else:
        print(f"\nNo GBM variant beat RF baseline (best Δ = "
              f"{rep['delta_winner_vs_rf_baseline']:+.4f}); submission NOT generated.")
