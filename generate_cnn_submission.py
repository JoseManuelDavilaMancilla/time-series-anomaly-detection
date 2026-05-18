"""
Standalone CNN submission generator.

Trains per-metric-type CNN ensembles on labeled windows,
generates predictions for all 1000 test windows.

Usage:
    uv run python generate_cnn_submission.py -o submission_cnn.json
"""

import json
import time
from pathlib import Path
import numpy as np
from cnn_addon import CNNEnsemble

from validation import all_window_dirs, load_window

METRIC_TYPES = ("Count", "ErrorCount", "LatencySecond", "QPS",
                "ResourceUtilizationRate", "SuccessRate")


def build_training_data(window_dirs, target_mt):
    """Collect raw series and labels for a given metric type."""
    raw_list, labels_list = [], []
    for wdir in window_dirs:
        info = json.loads((wdir / "info.json").read_text())
        if info.get("metric_type") != target_mt:
            continue
        try:
            train_y = np.load(wdir / "train_label.npy")
        except FileNotFoundError:
            continue
        if train_y.sum() == 0:
            continue
        train_x = np.load(wdir / "train.npy")
        raw_list.append(train_x)
        labels_list.append(train_y)
    return raw_list, labels_list


def generate_cnn_submission(output_path: Path = Path("submission_cnn.json")):
    window_dirs = all_window_dirs()
    print(f"Training CNN ensembles for {len(METRIC_TYPES)} metric types...")

    cnn_models = {}
    for mt in METRIC_TYPES:
        raw_list, labels_list = build_training_data(window_dirs, mt)
        print(f"  [{mt}] {len(raw_list)} windows with labels")
        if not raw_list:
            cnn_models[mt] = None
            continue
        t0 = time.time()
        cnn = CNNEnsemble(context=32, n_seeds=3, epochs=20, batch_size=256, lr=1e-3)
        cnn.fit(raw_list, labels_list)
        print(f"    trained in {time.time() - t0:.1f}s")
        cnn_models[mt] = cnn

    print(f"\nGenerating predictions on 1000 test windows...")
    preds = {}
    t0 = time.time()
    for i, wdir in enumerate(window_dirs, 1):
        w = load_window(wdir)
        mt = w.info.get("metric_type", "Unknown")
        if mt not in cnn_models or cnn_models[mt] is None:
            preds[w.wid] = [0] * len(w.test_x)
            continue

        n = len(w.test_x)
        k = max(0, min(int(round(n * float(w.info.get("test set anomaly ratio", 0.0)))), n))
        if k == 0:
            preds[w.wid] = [0] * n
            continue

        # Combine train + test for continuous context at boundary
        combined = np.concatenate([w.train_x, w.test_x])
        cnn_probs = cnn_models[mt].predict_proba(combined)
        test_probs = cnn_probs[len(w.train_x):]

        # Top-k selection
        order = np.lexsort((np.arange(n), -test_probs))
        pred = np.zeros(n, dtype=int)
        pred[order[:k]] = 1
        preds[w.wid] = pred.tolist()

        if i % 100 == 0:
            print(f"  {i}/1000 ({time.time() - t0:.0f}s)")

    assert len(preds) == 1000
    output_path.write_text(
        json.dumps({"predictions": preds}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, default=Path("submission_cnn.json"))
    args = parser.parse_args()
    generate_cnn_submission(args.output)
