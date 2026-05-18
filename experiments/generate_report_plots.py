"""
Generate plots for the homework report.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_all_window_dirs, load_window, find_contiguous_segments


def plot_dataset_overview(output_dir: Path = Path("report_plots")):
    """Generate overview plots of dataset characteristics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    window_dirs = get_all_window_dirs()

    train_lengths = []
    test_lengths = []
    train_rates = []
    test_ratios = []
    overlap_ratios = []
    segment_lengths = []

    for wdir in window_dirs:
        window = load_window(wdir)
        train_x = window["train_x"]
        train_y = window["train_y"]
        test_x = window["test_x"]
        info = window["info"]

        train_lengths.append(len(train_x))
        test_lengths.append(len(test_x))
        train_rates.append(float(np.mean(train_y)))
        test_ratios.append(info.get("test set anomaly ratio", 0.0))

        train_min, train_max = np.min(train_x), np.max(train_x)
        test_min, test_max = np.min(test_x), np.max(test_x)
        overlap_min = max(train_min, test_min)
        overlap_max = min(train_max, test_max)
        if overlap_max > overlap_min:
            overlap_len = overlap_max - overlap_min
            union_len = max(train_max, test_max) - min(train_min, test_min)
            overlap_ratios.append(overlap_len / union_len if union_len > 0 else 0.0)
        else:
            overlap_ratios.append(0.0)

        if np.sum(train_y) > 0:
            segments = find_contiguous_segments(train_y)
            for start, end in segments:
                segment_lengths.append(end - start + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Train lengths
    axes[0, 0].hist(train_lengths, bins=30, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Train Window Lengths")
    axes[0, 0].set_xlabel("Length")
    axes[0, 0].set_ylabel("Count")

    # Test lengths
    axes[0, 1].hist(test_lengths, bins=30, color="coral", edgecolor="black")
    axes[0, 1].set_title("Test Window Lengths")
    axes[0, 1].set_xlabel("Length")
    axes[0, 1].set_ylabel("Count")

    # Train vs test anomaly rates
    axes[0, 2].scatter(train_rates, test_ratios, alpha=0.3, s=10)
    axes[0, 2].plot([0, 0.7], [0, 0.7], "k--", alpha=0.5)
    axes[0, 2].set_title("Train Anomaly Rate vs Test Anomaly Ratio")
    axes[0, 2].set_xlabel("Train rate")
    axes[0, 2].set_ylabel("Test ratio")

    # Train-test overlap
    axes[1, 0].hist(overlap_ratios, bins=30, color="green", edgecolor="black")
    axes[1, 0].set_title("Train-Test Value Range Overlap")
    axes[1, 0].set_xlabel("Overlap ratio")
    axes[1, 0].set_ylabel("Count")

    # Segment lengths
    axes[1, 1].hist(segment_lengths, bins=30, color="purple", edgecolor="black")
    axes[1, 1].set_title("Anomalous Segment Lengths (Train)")
    axes[1, 1].set_xlabel("Segment length")
    axes[1, 1].set_ylabel("Count")

    # Category pie chart
    categories = {
        "constant_train": sum(1 for r in overlap_ratios if r == 0 and False),  # need recompute
    }
    # Simpler: just show zero vs non-zero overlap
    zero_overlap = sum(1 for r in overlap_ratios if r == 0)
    non_zero = len(overlap_ratios) - zero_overlap
    axes[1, 2].pie([zero_overlap, non_zero], labels=["Zero overlap", "Non-zero overlap"], autopct="%1.1f%%")
    axes[1, 2].set_title("Train-Test Overlap Distribution")

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_overview.png", dpi=150)
    print(f"Saved {output_dir / 'dataset_overview.png'}")
    plt.close()


def plot_experiment_comparison(output_dir: Path = Path("report_plots")):
    """Bar chart comparing experiment mean F1 scores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(__file__).parent.parent / "results"

    experiments = []
    for result_file in sorted(results_dir.glob("*.json")):
        data = json.loads(result_file.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        if isinstance(summary, dict) and "mean_f1" in summary:
            experiments.append((data["description"][:30], summary["mean_f1"]))
        elif isinstance(summary, dict):
            for sub_name, sub_summary in summary.items():
                if isinstance(sub_summary, dict) and "mean_f1" in sub_summary:
                    experiments.append((f"{data['description'][:20]} [{sub_name}]", sub_summary["mean_f1"]))

    if not experiments:
        print("No experiment results found")
        return

    experiments.sort(key=lambda x: x[1])
    names, scores = zip(*experiments)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = ax.barh(range(len(names)), scores, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean Validation F1", fontsize=10)
    ax.set_title("Experiment Comparison (Time-Based Validation Split)", fontsize=12)
    ax.set_xlim(0, max(scores) * 1.1)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "experiment_comparison.png", dpi=150)
    print(f"Saved {output_dir / 'experiment_comparison.png'}")
    plt.close()


def main():
    plot_dataset_overview()
    plot_experiment_comparison()


if __name__ == "__main__":
    main()
