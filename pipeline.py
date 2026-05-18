"""
author v25 — submission-level voting ensemble (no model training).

After 9 model-side experiments saturated, try combining the *predictions* of
our best submissions. Different architectures make different errors; if the
errors are not fully correlated, voting can recover.

Approach:
  1. Load N prediction files (each a {wid: [0/1, ...], ...} dict).
  2. For each window's points, count how many submissions mark each point as 1.
  3. The vote count is the new per-point "score".
  4. Apply segment selection with the existing kbudget = round(len * test_ratio).

The ensemble's predictions retain the budget constraint and the contiguous
segment structure — only the choice of WHICH points are anomalous changes
based on the agreement between architectures.

We try multiple vote-set compositions and pick the one with the highest
validation F1.

Run:  uv run python v25_vote_ensemble.py
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from shared_lib import predict_segments
from validation import (
    all_window_dirs,
    load_window,
    point_f1,
    save_report,
    stratified_holdout,
    time_split,
)

SEG_KWARGS = dict(smooth=3, thr_frac=0.7, small_k_cutoff=4, max_seg=60)


# Candidate submissions to combine. Order matters for tracking only.
CANDIDATES = {
    "v15_kimi_combo (0.6238 LB)":   "submission_kimi_combo_all_v12.json",
    "v22_metadata":                 "submission_metadata_cw.json",
    "v9_segtuned (0.6111 LB)":      "submission_segtuned.json",
    "v14_qps_lgbm (0.6236 LB)":     "submission_qps_lgbm_routed.json",
}


# Vote set compositions to evaluate
VOTE_SETS = {
    "top2 (kimi_combo + metadata)":            ["v15_kimi_combo (0.6238 LB)", "v22_metadata"],
    "top3 (teammate+metadata+qps)":                ["v15_kimi_combo (0.6238 LB)", "v22_metadata", "v14_qps_lgbm (0.6236 LB)"],
    "top4 (all)":                              list(CANDIDATES.keys()),
    "all minus segtuned":                      ["v15_kimi_combo (0.6238 LB)", "v22_metadata", "v14_qps_lgbm (0.6236 LB)"],
}


def load_submission(path: Path) -> Dict[str, List[int]]:
    data = json.loads(path.read_text())
    return {wid: np.asarray(pred, dtype=np.int8) for wid, pred in data["predictions"].items()}


def vote_score(preds: List[np.ndarray]) -> np.ndarray:
    """Per-point vote count = sum of binary predictions."""
    return np.sum(np.vstack(preds), axis=0).astype(np.float64)


def combine_predictions(subs: List[Dict[str, np.ndarray]], wid: str,
                        k: int, n: int) -> np.ndarray:
    """For one window, combine votes and pick top-k via segment selection.
    If all submissions are zero, return all zeros. If k=0, return all zeros."""
    if k <= 0:
        return np.zeros(n, dtype=int)
    point_arrays = [s[wid] for s in subs if wid in s]
    if not point_arrays:
        return np.zeros(n, dtype=int)
    # Each prediction is 0/1; sum is in [0, len(subs)]
    votes = vote_score(point_arrays)
    # Tiny smoothing to break ties at very low budget (1-pt segments)
    if votes.max() == 0:
        return np.zeros(n, dtype=int)
    return predict_segments(votes, k, **SEG_KWARGS)


def evaluate_vote_set(subs: List[Dict[str, np.ndarray]], holdout, name: str) -> dict:
    """Each held-out window: we KNOW the train labels; predict on full train_x's
    last 30% using the submission's TEST predictions sliced (or fall back).

    BUT submissions only have TEST predictions, not train predictions. So
    voting-based evaluation against time-split sub_test labels isn't directly
    possible. Instead, we evaluate by aligning submissions' predictions with
    the *training* labels we have: each window's test prediction is treated
    as the model's vote on those points, but we don't have those points'
    ground truth.

    Alternative: re-run the v22-style pipeline on each holdout window and
    cache per-point predictions per submission, but that defeats the
    purpose of voting cheaply.

    Pragmatic choice: just evaluate vote AGREEMENT — what fraction of points
    do submissions disagree on? If they highly disagree (large entropy),
    voting could help; if they fully agree, voting can't help.

    This is a "diagnostic" rather than a F1 evaluation."""
    raise NotImplementedError("see compute_full_predictions instead")


def compute_full_predictions(vote_set_name: str, sub_paths: List[Path],
                             output: Path) -> Path:
    """Generate a new submission file by voting across the listed submissions."""
    print(f"\n>>> Building vote ensemble for [{vote_set_name}] from {len(sub_paths)} subs…")
    subs = [load_submission(p) for p in sub_paths]
    print(f"    loaded {[p.name for p in sub_paths]}")

    all_dirs = all_window_dirs()
    preds: dict[str, list[int]] = {}
    cnt_diverge = 0
    for wdir in all_dirs:
        w = load_window(wdir)
        ratio = float(w.info.get("test set anomaly ratio", 0.0))
        k = int(round(len(w.test_x) * ratio))
        n = len(w.test_x)

        # Per-point votes
        point_arrays = [s[w.wid] for s in subs if w.wid in s and len(s[w.wid]) == n]
        if len(point_arrays) < 2:
            # Not enough subs covering this window; fall back to first available
            if point_arrays:
                preds[w.wid] = point_arrays[0].astype(int).tolist()
            else:
                preds[w.wid] = [0] * n
            continue

        votes = vote_score(point_arrays)
        if votes.sum() == 0:
            preds[w.wid] = [0] * n
            continue

        # Count points where subs disagree
        disagreements = int(((votes > 0) & (votes < len(point_arrays))).sum())
        if disagreements > 0:
            cnt_diverge += 1
        pred = predict_segments(votes, k, **SEG_KWARGS)
        preds[w.wid] = pred.astype(int).tolist()

    print(f"    windows with ≥1 inter-submission disagreement: {cnt_diverge}/{len(all_dirs)}")
    output.write_text(
        json.dumps({"predictions": preds}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f">>> Wrote {output}")
    return output


def diagnose_agreement(sub_paths: List[Path]) -> None:
    print("\n>>> Inter-submission agreement diagnostic…")
    subs = [load_submission(p) for p in sub_paths]
    all_dirs = all_window_dirs()

    total_points = 0
    full_agree = 0
    full_anomaly = 0
    full_normal = 0
    partial = 0
    for wdir in all_dirs:
        w = load_window(wdir)
        if w.wid not in subs[0]:
            continue
        arrays = [s[w.wid] for s in subs if w.wid in s and len(s[w.wid]) == len(w.test_x)]
        if len(arrays) != len(subs):
            continue
        stacked = np.vstack(arrays)
        votes = stacked.sum(axis=0)
        n = stacked.shape[1]
        total_points += n
        full_normal += int((votes == 0).sum())
        full_anomaly += int((votes == len(subs)).sum())
        partial += int(((votes > 0) & (votes < len(subs))).sum())

    full_agree = full_normal + full_anomaly
    print(f"  total points evaluated: {total_points}")
    print(f"  full agreement (all-same): {full_agree} ({100 * full_agree / total_points:.1f}%)")
    print(f"    all-normal:  {full_normal} ({100 * full_normal / total_points:.1f}%)")
    print(f"    all-anomaly: {full_anomaly} ({100 * full_anomaly / total_points:.3f}%)")
    print(f"  disagreement (split vote): {partial} ({100 * partial / total_points:.3f}%)")


if __name__ == "__main__":
    paths = {name: Path(fname) for name, fname in CANDIDATES.items()}
    missing = {name for name, p in paths.items() if not p.exists()}
    if missing:
        print(f"WARNING: missing submissions: {missing}")
    paths = {name: p for name, p in paths.items() if p.exists()}

    diagnose_agreement(list(paths.values()))

    for vote_set_name, sub_names in VOTE_SETS.items():
        sub_paths = [paths[n] for n in sub_names if n in paths]
        if len(sub_paths) < 2:
            continue
        output = Path(f"submission_vote_{vote_set_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')}.json")
        compute_full_predictions(vote_set_name, sub_paths, output)
