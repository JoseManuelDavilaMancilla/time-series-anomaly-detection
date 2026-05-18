"""
author v32 — Edge-fix in predict_segments.

v31 investigation revealed: ResourceUtilizationRate's 0.95 ceiling is driven
by TWO specific failure modes in `predict_segments` at window boundaries, not
by the model:

  1. **End-of-window false positives**: when the time series naturally drifts
     up at the end (e.g., wid=944), the smoothed score at the last 5 points
     becomes a local maximum and gets selected as anomaly.
  2. **Start-of-window misalignment**: convolution smoothing with mode='same'
     downweights position 0 (fewer samples in the kernel), so segments that
     truly start at position 0 get shifted rightward (e.g., wid=967: truth
     [0, 24], pred [4, 28]).

Fixes:
  A. **Reflect-padding for smoothing**: replace mode='same' with explicit
     edge reflection so positions 0 and N-1 see the same kernel contribution
     as interior positions.
  B. **Boundary penalty**: optionally suppress segment centers that fall in
     the first/last few positions UNLESS the smoothed score is sharply higher
     there than its interior peak (i.e., a real boundary anomaly).

Sweep both axes against v22 (no edge fix). Compare across all categories,
not just ResourceUtil — the fix should be neutral or positive on others.

Run:  uv run python v32_edge_fix.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Optional

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
    predict_topk,
)
from v6_cnn import SPECIALIZED, build_training_pool as v6_pool
from v7_cnn_ensemble import CNN_SEEDS, ensemble_cnn_score, fit_cnn_with_seed
from v22_metadata_features import build_metadata_hybrid, scores_v14_with_meta
from validation import (
    all_window_dirs,
    evaluate,
    load_window,
    print_summary,
    save_report,
    stratified_holdout,
)


def predict_segments_v32(
    scores: np.ndarray,
    k: int,
    *,
    smooth: int = 3,
    thr_frac: float = 0.7,
    small_k_cutoff: int = 4,
    max_seg: Optional[int] = None,
    edge_pad_mode: str = "reflect",          # 'reflect' or 'same' (legacy)
    boundary_penalty: float = 0.0,           # multiply scores by (1 - bp) within edge_width of boundaries unless they're a clear winner
    edge_width: int = 5,
) -> np.ndarray:
    """Like predict_segments but with edge-fix options.

    edge_pad_mode='reflect': use reflective padding for the smoothing
        convolution so positions 0 and N-1 see a full kernel.
    boundary_penalty>0: scale scores within `edge_width` of either boundary
        DOWN by (1 - boundary_penalty), UNLESS those scores are higher than
        the interior peak (in which case keep them).
    """
    n = len(scores)
    pred = np.zeros(n, dtype=int)
    if k <= 0:
        return pred
    if k <= small_k_cutoff:
        return predict_topk(scores, k)

    s_raw = np.asarray(scores, dtype=np.float64).copy()

    # Edge-aware smoothing
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        if edge_pad_mode == "reflect":
            half = smooth // 2
            padded = np.concatenate([s_raw[half-1::-1] if half > 0 else s_raw[:0],
                                     s_raw,
                                     s_raw[-1:-half-1:-1] if half > 0 else s_raw[:0]])
            # Convolve in 'valid' over the padded series → length n
            s = np.convolve(padded, kernel, mode="valid")
            # Handle off-by-one from asymmetric kernel
            if len(s) > n:
                s = s[: n]
            elif len(s) < n:
                # Fall back to same-mode
                s = np.convolve(s_raw, kernel, mode="same")
        else:
            s = np.convolve(s_raw, kernel, mode="same")
    else:
        s = s_raw.copy()

    # Boundary penalty
    if boundary_penalty > 0.0 and edge_width > 0 and edge_width * 2 < n:
        interior_max = s[edge_width:-edge_width].max() if n > 2 * edge_width else s.max()
        # Penalize edges only if they're below interior_max (i.e., not a true boundary anomaly)
        for j in range(min(edge_width, n)):
            if s[j] < interior_max:
                s[j] *= (1.0 - boundary_penalty)
            if n - 1 - j >= 0 and s[n - 1 - j] < interior_max:
                s[n - 1 - j] *= (1.0 - boundary_penalty)

    budget = int(k)
    order = np.argsort(-s)
    for center in order:
        if budget <= 0:
            break
        if pred[center] == 1:
            continue
        peak = s[center]
        if peak <= 0:
            break
        thr = thr_frac * peak

        pred[center] = 1
        budget -= 1

        L = R = int(center)
        grew = True
        while grew and budget > 0:
            grew = False
            if L > 0 and pred[L - 1] == 0 and s[L - 1] >= thr:
                L -= 1
                pred[L] = 1
                budget -= 1
                grew = True
                if budget <= 0:
                    break
            if R < n - 1 and pred[R + 1] == 0 and s[R + 1] >= thr:
                R += 1
                pred[R] = 1
                budget -= 1
                grew = True
            if max_seg is not None and (R - L + 1) >= max_seg:
                break

    return pred


CNN_WEIGHT = 0.35
SEG_KWARGS_BASE = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)


def scores_v22(train_x, train_y, test_x, cw, cnn_models, info, metric_type):
    return scores_v14_with_meta(train_x, train_y, test_x, cw, cnn_models, info, metric_type)


def run_validation(seed: int = 42) -> dict:
    print("\n>>> Building stratified holdout (10%)…")
    train_pool, holdout = stratified_holdout(all_window_dirs(), frac=0.10, seed=seed)
    print(f"    train_pool={len(train_pool)}  holdout={len(holdout)}")

    print(">>> Training metadata CW + 3-seed CNN…")
    t0 = time.time()
    cw = build_metadata_hybrid(train_pool, mode="intervals")
    print(f"    cw fit {time.time() - t0:.1f}s")
    Xc, yc = v6_pool(train_pool)
    cnn_models = []
    for s in CNN_SEEDS:
        t0 = time.time()
        cnn_models.append(fit_cnn_with_seed(Xc, yc, s))
        print(f"    cnn seed={s} fit {time.time() - t0:.1f}s")

    def make_predictor(seg_kwargs):
        def pred(sub_tr_x, sub_tr_y, sub_te_x, info, ratio):
            scores = scores_v22(sub_tr_x, sub_tr_y, sub_te_x, cw, cnn_models, info,
                                info.get("metric_type", "ALL"))
            return predict_segments_v32(scores, int(round(len(sub_te_x) * ratio)),
                                        **seg_kwargs)
        return pred

    variants = {
        "baseline (same, no penalty)": dict(**SEG_KWARGS_BASE, edge_pad_mode="same",
                                            boundary_penalty=0.0, edge_width=5),
        "reflect (no penalty)":        dict(**SEG_KWARGS_BASE, edge_pad_mode="reflect",
                                            boundary_penalty=0.0, edge_width=5),
        "reflect + pen=0.10":          dict(**SEG_KWARGS_BASE, edge_pad_mode="reflect",
                                            boundary_penalty=0.10, edge_width=5),
        "reflect + pen=0.20":          dict(**SEG_KWARGS_BASE, edge_pad_mode="reflect",
                                            boundary_penalty=0.20, edge_width=5),
        "reflect + pen=0.30":          dict(**SEG_KWARGS_BASE, edge_pad_mode="reflect",
                                            boundary_penalty=0.30, edge_width=5),
    }

    results = {}
    for name, kwargs in variants.items():
        print(f"\n>>> Eval [{name}]…")
        rep = evaluate(make_predictor(kwargs), holdout)
        print_summary(rep, name=name)
        results[name] = rep

    print("\n──────  summary ──────")
    base = results["baseline (same, no penalty)"]["overall_f1"]
    rows = sorted(results.items(), key=lambda kv: -kv[1]["overall_f1"])
    for name, rep in rows:
        d = rep["overall_f1"] - base
        print(f"  {name:<32}  F1={rep['overall_f1']:.4f}  Δ_vs_base={d:+.4f}")

    # Per-metric breakdown for top 2
    print("\n>>> Per-metric for top 2 variants:")
    for name, rep in rows[:2]:
        print(f"\n  [{name}]")
        for mt, st in sorted(rep["by_metric_type"].items()):
            base_f1 = results["baseline (same, no penalty)"]["by_metric_type"][mt]["mean_f1"]
            d_mt = st["mean_f1"] - base_f1
            print(f"    {mt:<28}  F1={st['mean_f1']:.4f}  Δ={d_mt:+.4f}")

    winner = rows[0][0]
    report = {
        "results": {name: r["overall_f1"] for name, r in results.items()},
        "winner": winner,
        "delta_vs_baseline": rows[0][1]["overall_f1"] - base,
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v32_edge_fix")
    return report, cw, cnn_models, variants


def generate_submission(seg_kwargs: dict, cw, cnn_models,
                        output: Path = Path("submission_edge_fix.json")) -> Path:
    print(f"\n>>> Re-training on ALL 1000 windows (seg_kwargs={seg_kwargs})…")
    t0 = time.time()
    cw_full = build_metadata_hybrid(all_window_dirs(), mode="intervals")
    print(f"    cw fit {time.time() - t0:.1f}s")

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
        scores = scores_v22(w.train_x, w.train_y, w.test_x, cw_full, cnn_full, w.info,
                            w.metric_type)
        k = int(round(len(w.test_x) * ratio))
        preds[w.wid] = predict_segments_v32(scores, k, **seg_kwargs).astype(int).tolist()
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
    rep, cw, cnn_models, variants = run_validation()
    if rep["winner"] != "baseline (same, no penalty)" and rep["delta_vs_baseline"] > 0.0005:
        winner_kwargs = variants[rep["winner"]]
        print(f"\n{rep['winner']} beats baseline by {rep['delta_vs_baseline']:+.4f}; "
              "generating submission.")
        generate_submission(winner_kwargs, cw, cnn_models)
    else:
        print(f"\nEdge fix did not meaningfully help "
              f"(best Δ = {rep['delta_vs_baseline']:+.4f}); submission NOT generated.")
