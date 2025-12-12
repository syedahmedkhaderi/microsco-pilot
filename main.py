import os
import argparse
import matplotlib.pyplot as plt

from src.dtm_controller import AdaptiveAFMController
from src.vision_evaluator import VisionEvaluator
from benchmarks.traditional_scan import run as run_baseline
from benchmarks.adaptive_scan import run as run_adaptive


def compare_and_plot(baseline_stats, adaptive_stats, result_dir="results"):
    os.makedirs(result_dir, exist_ok=True)
    labels = ["Baseline", "Adaptive"]
    times = [baseline_stats['total_time'], adaptive_stats['total_time']]
    qualities = [baseline_stats['avg_quality'], adaptive_stats['avg_quality']]

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].bar(labels, times, color=["#888", "#4CAF50"]) 
    ax[0].set_title("Total Scan Time (s)")
    ax[0].set_ylabel("seconds")
    ax[1].bar(labels, qualities, color=["#888", "#4CAF50"]) 
    ax[1].set_title("Average Quality (arb)")
    plt.tight_layout()
    outpath = os.path.join(result_dir, "benchmark_comparison.png")
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def main():
    parser = argparse.ArgumentParser(description="SmartScan Demo: Real-time adaptive AFM scanning")
    parser.add_argument("--grid_x", type=int, default=4)
    parser.add_argument("--grid_y", type=int, default=1)
    parser.add_argument("--region_size", type=int, default=300)
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--adaptive_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results, exist_ok=True)

    print("Running baseline...")
    baseline_stats = run_baseline(grid_x=args.grid_x, grid_y=args.grid_y, region_size=args.region_size, result_dir=args.results)
    print("Baseline stats:", baseline_stats)

    if args.baseline_only:
        return

    print("Running adaptive...")
    adaptive_stats = run_adaptive(grid_x=args.grid_x, grid_y=args.grid_y, region_size=args.region_size, result_dir=args.results)
    print("Adaptive stats:", adaptive_stats)

    if args.adaptive_only:
        return

    out = compare_and_plot(baseline_stats, adaptive_stats, result_dir=args.results)
    print(f"Saved benchmark comparison to {out}")


if __name__ == "__main__":
    main()