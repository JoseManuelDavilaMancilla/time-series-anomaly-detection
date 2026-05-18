"""
Visualize a single window's train/test series with anomaly labels.

Usage:
    uv run python experiments/visualize_window.py 000
    uv run python experiments/visualize_window.py 000 --save plots/window_000.png
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import json

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Install with: uv pip install matplotlib")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_all_window_dirs, load_window


def plot_window(window_dir: Path, output_path: Path = None):
    """Plot train and test series with anomaly labels."""
    window = load_window(window_dir)
    train_x = window["train_x"]
    train_y = window["train_y"]
    test_x = window["test_x"]
    test_ts = window["test_ts"]
    info = window["info"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Train plot
    ax = axes[0]
    ax.plot(train_x, color="steelblue", linewidth=0.8, label="Train values")
    anom_idx = np.where(train_y == 1)[0]
    if len(anom_idx) > 0:
        ax.scatter(anom_idx, train_x[anom_idx], color="red", s=10, zorder=5, label="Train anomalies")
    ax.set_title(f"Train: {window['window_name']} | Anomalies: {np.sum(train_y)}/{len(train_y)} ({np.mean(train_y)*100:.1f}%)")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Test plot
    ax = axes[1]
    ax.plot(test_x, color="steelblue", linewidth=0.8, label="Test values")
    ax.set_title(f"Test: {window['window_name']} | Length: {len(test_x)} | Test ratio: {info.get('test set anomaly ratio', 'N/A')}")
    ax.set_ylabel("Value")
    ax.set_xlabel("Time step")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize a time-series window")
    parser.add_argument("window_id", help="Window ID, e.g. 000")
    parser.add_argument("--save", type=Path, default=None, help="Path to save plot")
    args = parser.parse_args()

    window_dirs = get_all_window_dirs()
    target = None
    for wdir in window_dirs:
        wid = wdir.name.split("_", 1)[0]
        if wid == args.window_id:
            target = wdir
            break

    if target is None:
        print(f"Window {args.window_id} not found")
        sys.exit(1)

    plot_window(target, args.save)


if __name__ == "__main__":
    main()
