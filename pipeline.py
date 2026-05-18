"""
author v5 — Sweep the per-window RF weight.

Hypothesis: teammate's PW weights (0.35 for partial_overlap, 0.50 for test_within_train)
overweight a model that overfits inside a single window. v11d's removal of PW
(0.5255 leaderboard) was *worse* than v8 (0.5485) → PW does add value. The
question is which weight is optimal.

We sweep PW weight ∈ {0.10, 0.20, 0.35, 0.50, 0.65}. The CW weight absorbs the
difference. We pick the validation winner (with a tiebreak preference for lower
PW, since validation is biased toward PW being useful — 70/30 time-split keeps
the test data inside the training distribution).

Run:  uv run python v5_low_pw.py
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

# Sweep: each tuple is (label, pw_weight_partial, pw_weight_within).
# CW takes 1 − pw − online_share where online_share is the same as v8 (0.30 for
# partial_overlap and 0 for test_within_train).
SWEEP = [
    ("pw_0.10", 0.10, 0.20),
    ("pw_0.20", 0.20, 0.30),
    ("pw_0.35_baseline", 0.35, 0.50),
    ("pw_0.50", 0.50, 0.60),
    ("pw_0.65", 0.65, 0.75),
]


def weights_for(pw_partial: float, pw_within: float) -> dict:
    """Build the weights dict consumed by v8_style_scores."""
    # partial_overlap: 30% online is fixed (matches v8), remainder split between cw and pw
    cw_partial = max(0.0, 1.0 - pw_partial - 0.30)
    # test_within_train: no online, remainder is all cw
    cw_within = max(0.0, 1.0 - pw_within)
    return {
        "partial_overlap": (cw_partial, pw_partial, 0.30),
        "test_within_train": (cw_within, pw_within),
    }


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

    sweep_results = {}
    for label, pw_p, pw_w in SWEEP:
        weights = weights_for(pw_p, pw_w)
        print(f"\n>>> Eval: pw_partial={pw_p:.2f} pw_within={pw_w:.2f}  ({label})")

        def predictor(sub_tr_x, sub_tr_y, sub_te_x, info, ratio, _w=weights):
            scores, _ = v8_style_scores(
                sub_tr_x, sub_tr_y, sub_te_x, cw,
                metric_type=info.get("metric_type", "ALL"),
                pw_weight_override=_w,
            )
            k = int(round(len(sub_te_x) * ratio))
            return predict_segments(scores, k, **SEG_KWARGS)

        rep = evaluate(predictor, holdout)
        print_summary(rep, name=label)
        sweep_results[label] = {"overall_f1": rep["overall_f1"], "pw_partial": pw_p,
                                "pw_within": pw_w, "report": rep}

    print("\n──────  sweep summary ──────")
    rows = sorted(sweep_results.items(), key=lambda kv: -kv[1]["overall_f1"])
    for label, info in rows:
        baseline_delta = info["overall_f1"] - sweep_results["pw_0.35_baseline"]["overall_f1"]
        print(f"  {label:<20}  pw=({info['pw_partial']:.2f}, {info['pw_within']:.2f})  "
              f"F1={info['overall_f1']:.4f}  Δvs_baseline={baseline_delta:+.4f}")

    winner_label, winner_info = rows[0]
    print(f"\n  Winner: {winner_label}  F1={winner_info['overall_f1']:.4f}")

    report = {
        "sweep": {l: {k: v for k, v in d.items() if k != "report"} for l, d in sweep_results.items()},
        "per_label_per_window": {l: d["report"]["per_window"] for l, d in sweep_results.items()},
        "winner": winner_label,
        "winner_pw_partial": winner_info["pw_partial"],
        "winner_pw_within": winner_info["pw_within"],
        "seed": seed,
        "n_holdout": len(holdout),
    }
    save_report(report, "v5_low_pw")
    return report


def generate_submission(pw_partial: float, pw_within: float,
                        output: Path = Path("submission_low_pw.json")) -> Path:
    print(f"\n>>> Training hybrid on ALL 1000 windows (pw_partial={pw_partial:.2f} pw_within={pw_within:.2f})…")
    t0 = time.time()
    cw = build_hybrid(all_window_dirs())
    print(f"    fit {time.time() - t0:.1f}s")

    weights = weights_for(pw_partial, pw_within)
    print(">>> Generating predictions…")
    preds: dict[str, list[int]] = {}
    t0 = time.time()
    for i, wdir in enumerate(all_window_dirs(), 1):
        w = load_window(wdir)
        test_ratio = float(w.info.get("test set anomaly ratio", 0.0))
        scores, _ = v8_style_scores(
            w.train_x, w.train_y, w.test_x, cw,
            metric_type=w.metric_type, pw_weight_override=weights,
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
    baseline = rep["sweep"]["pw_0.35_baseline"]["overall_f1"]
    winner_label = rep["winner"]
    winner_f1 = rep["sweep"][winner_label]["overall_f1"]
    if winner_label != "pw_0.35_baseline" and winner_f1 - baseline > 0.001:
        print(f"\nGenerating submission with {winner_label}…")
        generate_submission(rep["winner_pw_partial"], rep["winner_pw_within"])
    else:
        print(f"\n!! Baseline pw=0.35 still best (or sweep gain <0.001); submission NOT generated.")
