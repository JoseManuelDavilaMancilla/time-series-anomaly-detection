"""
Dataset-wide analysis script.

Computes and saves statistics about the entire dataset:
- Train/test lengths
- Anomaly rates
- Train-test value range overlap
- Contiguous segment lengths
- Metric type distribution
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_all_window_dirs, load_window, find_contiguous_segments, save_results


def analyze_dataset():
    window_dirs = get_all_window_dirs()
    stats = {
        "n_windows": len(window_dirs),
        "train_lengths": [],
        "test_lengths": [],
        "train_anomaly_rates": [],
        "test_anomaly_ratios": [],
        "n_train_anomalies": [],
        "zero_anomaly_windows": 0,
        "segment_counts": [],
        "segment_lengths": [],
        "train_test_overlap_ratios": [],
        "metric_name_counts": defaultdict(int),
        "metric_type_counts": defaultdict(int),
    }

    for wdir in window_dirs:
        window = load_window(wdir)
        train_x = window["train_x"]
        train_y = window["train_y"]
        test_x = window["test_x"]
        info = window["info"]

        stats["train_lengths"].append(len(train_x))
        stats["test_lengths"].append(len(test_x))
        stats["train_anomaly_rates"].append(float(np.mean(train_y)))
        stats["n_train_anomalies"].append(int(np.sum(train_y)))
        stats["test_anomaly_ratios"].append(info.get("test set anomaly ratio", 0.0))

        if np.sum(train_y) == 0:
            stats["zero_anomaly_windows"] += 1
        else:
            segments = find_contiguous_segments(train_y)
            stats["segment_counts"].append(len(segments))
            for start, end in segments:
                stats["segment_lengths"].append(end - start + 1)

        # Train-test overlap
        train_min, train_max = np.min(train_x), np.max(train_x)
        test_min, test_max = np.min(test_x), np.max(test_x)
        overlap_min = max(train_min, test_min)
        overlap_max = min(train_max, test_max)
        if overlap_max > overlap_min:
            overlap_len = overlap_max - overlap_min
            train_len = train_max - train_min
            test_len = test_max - test_min
            union_len = max(train_max, test_max) - min(train_min, test_min)
            overlap_ratio = overlap_len / union_len if union_len > 0 else 0.0
        else:
            overlap_ratio = 0.0
        stats["train_test_overlap_ratios"].append(overlap_ratio)

        # Metric info
        parts = window["window_name"].split("##")
        if len(parts) == 2:
            metric_type = parts[0].split("_", 1)[1]
            metric_name = parts[1]
            stats["metric_type_counts"][metric_type] += 1
            stats["metric_name_counts"][metric_name] += 1

    # Convert to summary
    summary = {
        "n_windows": stats["n_windows"],
        "train_length": {
            "min": int(np.min(stats["train_lengths"])),
            "max": int(np.max(stats["train_lengths"])),
            "mean": float(np.mean(stats["train_lengths"])),
            "median": float(np.median(stats["train_lengths"])),
        },
        "test_length": {
            "min": int(np.min(stats["test_lengths"])),
            "max": int(np.max(stats["test_lengths"])),
            "mean": float(np.mean(stats["test_lengths"])),
            "median": float(np.median(stats["test_lengths"])),
        },
        "train_anomaly_rate": {
            "min": float(np.min(stats["train_anomaly_rates"])),
            "max": float(np.max(stats["train_anomaly_rates"])),
            "mean": float(np.mean(stats["train_anomaly_rates"])),
            "median": float(np.median(stats["train_anomaly_rates"])),
        },
        "test_anomaly_ratio": {
            "min": float(np.min(stats["test_anomaly_ratios"])),
            "max": float(np.max(stats["test_anomaly_ratios"])),
            "mean": float(np.mean(stats["test_anomaly_ratios"])),
            "median": float(np.median(stats["test_anomaly_ratios"])),
        },
        "zero_anomaly_windows": stats["zero_anomaly_windows"],
        "segment_count_mean": float(np.mean(stats["segment_counts"])) if stats["segment_counts"] else 0.0,
        "segment_length_mean": float(np.mean(stats["segment_lengths"])) if stats["segment_lengths"] else 0.0,
        "segment_length_median": float(np.median(stats["segment_lengths"])) if stats["segment_lengths"] else 0.0,
        "train_test_overlap": {
            "mean": float(np.mean(stats["train_test_overlap_ratios"])),
            "median": float(np.median(stats["train_test_overlap_ratios"])),
            "zero_overlap": int(np.sum(np.array(stats["train_test_overlap_ratios"]) == 0)),
        },
        "top_metric_names": dict(Counter(stats["metric_name_counts"]).most_common(10)),
        "top_metric_types": dict(Counter(stats["metric_type_counts"]).most_common(10)),
    }

    print(json.dumps(summary, indent=2))

    save_results("dataset_analysis", {
        "description": "Dataset-wide statistics and characteristics",
        "summary": summary,
    })


if __name__ == "__main__":
    analyze_dataset()
