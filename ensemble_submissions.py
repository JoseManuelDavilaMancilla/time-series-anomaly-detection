"""
Ensemble helper — blend multiple submission JSONs.

Usage:
    uv run python ensemble_submissions.py submission_a.json submission_b.json --weights 0.6 0.4 -o submission_blend.json

Blending strategies:
    - rank_avg: average normalized ranks (default)
    - vote: majority vote across binary predictions
    - prob_avg: average predicted probabilities (if scores available)
"""

import argparse
import json
import numpy as np
from pathlib import Path


def load_submission(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["predictions"]


def ensemble_rank_avg(submissions: list[dict], weights: list[float] | None = None) -> dict:
    """Average normalized ranks, then top-k calibrate per window."""
    wids = sorted(submissions[0].keys())
    result = {}
    if weights is None:
        weights = [1.0 / len(submissions)] * len(submissions)
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    for wid in wids:
        preds = [np.array(s[wid], dtype=np.int64) for s in submissions]
        n = len(preds[0])
        k = sum(preds[0])  # assume all have same k
        # Convert to scores via rank
        scores = []
        for p in preds:
            score = np.zeros(n, dtype=np.float64)
            anom_idx = np.where(p == 1)[0]
            score[anom_idx] = 1.0
            # Add small jitter by rank to break ties
            if len(anom_idx) > 0:
                score[anom_idx] += np.linspace(0.9, 0.1, len(anom_idx))
            scores.append(score)
        avg_score = sum(w * s for w, s in zip(weights, scores))
        # Top-k
        if k == 0:
            result[wid] = [0] * n
        else:
            order = np.lexsort((np.arange(n), -avg_score))
            out = np.zeros(n, dtype=int)
            out[order[:k]] = 1
            result[wid] = out.tolist()
    return result


def ensemble_vote(submissions: list[dict], threshold: float = 0.5) -> dict:
    """Majority vote across binary predictions."""
    wids = sorted(submissions[0].keys())
    result = {}
    for wid in wids:
        preds = [np.array(s[wid], dtype=np.int64) for s in submissions]
        n = len(preds[0])
        k = sum(preds[0])
        vote_sum = sum(preds)
        # Select top-k by vote count, tie-break by sum
        if k == 0:
            result[wid] = [0] * n
        else:
            order = np.lexsort((np.arange(n), -vote_sum))
            out = np.zeros(n, dtype=int)
            out[order[:k]] = 1
            result[wid] = out.tolist()
    return result


def compute_disagreement(submissions: list[dict]) -> float:
    """Fraction of points where submissions disagree."""
    wids = sorted(submissions[0].keys())
    total = 0
    disagree = 0
    for wid in wids:
        preds = [np.array(s[wid], dtype=np.int64) for s in submissions]
        n = len(preds[0])
        total += n
        for i in range(n):
            vals = [p[i] for p in preds]
            if len(set(vals)) > 1:
                disagree += 1
    return disagree / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submissions", nargs="+", type=Path, help="Submission JSON files")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Blend weights")
    parser.add_argument("-o", "--output", type=Path, default=Path("submission_ensemble.json"))
    parser.add_argument("--mode", choices=["rank_avg", "vote"], default="rank_avg")
    parser.add_argument("--disagreement-only", action="store_true", help="Only print disagreement, no output")
    args = parser.parse_args()

    subs = [load_submission(p) for p in args.submissions]
    print(f"Loaded {len(subs)} submissions")

    disc = compute_disagreement(subs)
    print(f"Disagreement rate: {disc*100:.2f}%")

    if args.disagreement_only:
        return

    if args.mode == "rank_avg":
        blended = ensemble_rank_avg(subs, args.weights)
    else:
        blended = ensemble_vote(subs)

    args.output.write_text(
        json.dumps({"predictions": blended}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
