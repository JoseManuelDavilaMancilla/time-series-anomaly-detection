"""
Run all experiments and produce a summary table.

Usage:
    cd experiments && uv run python run_all_experiments.py
"""

import sys
import json
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

# Import experiment runners
from exp_baseline_ensemble import run_experiment as run_baseline
from exp_supervised_rf import run_experiment as run_supervised_rf
from exp_xgboost_lgbm import run_experiment as run_xgboost_lgbm
from exp_model_selection import run_experiment as run_model_selection
from exp_stacking import run_experiment as run_stacking
from exp_online_adaptive import run_experiment as run_online_adaptive
from exp_subsequence import run_experiment as run_subsequence


def run_all(sample_size: int = 200):
    """Run every experiment script and print a summary table."""
    experiments = [
        ("Baseline Ensemble (v1)", run_baseline),
        ("Supervised RF (v3)", run_supervised_rf),
        ("XGBoost+LightGBM (v4)", run_xgboost_lgbm),
        ("Model Selection", run_model_selection),
        ("Stacking / Weighting", run_stacking),
        ("Online Adaptive", run_online_adaptive),
        ("Subsequence Discord", run_subsequence),
    ]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print(f"Running {len(experiments)} experiments with sample_size={sample_size}")
    print("=" * 80)

    for name, runner in experiments:
        print(f"\n>>> Running: {name}")
        start = time.time()
        try:
            runner(sample_size=sample_size)
            elapsed = time.time() - start
            print(f"    Completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"    FAILED: {e}")

    # Load all result files and print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Experiment':<35} {'Mean F1':>10} {'Median F1':>10} {'Count':>8}")
    print("-" * 80)

    for result_file in sorted(results_dir.glob("*.json")):
        data = json.loads(result_file.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        
        # Handle nested summaries (e.g., stacking has multiple strategies)
        if isinstance(summary, dict) and "mean_f1" not in summary:
            for sub_name, sub_summary in summary.items():
                if isinstance(sub_summary, dict) and "mean_f1" in sub_summary:
                    print(f"  {data['description'][:30]:<33} [{sub_name:<20}] {sub_summary['mean_f1']:>8.4f} {sub_summary['median_f1']:>10.4f} {sub_summary['count']:>8}")
        else:
            if isinstance(summary, dict) and "mean_f1" in summary:
                print(f"{data['description'][:35]:<35} {summary['mean_f1']:>10.4f} {summary['median_f1']:>10.4f} {summary['count']:>8}")


if __name__ == "__main__":
    run_all(sample_size=200)
