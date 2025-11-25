#!/usr/bin/env python3
"""
对比指定mode与FP32参考的输出差异
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path


def load_outputs(output_dir, mode, layer_idx):
    """Load saved layer outputs"""
    output_file = os.path.join(output_dir, f"mode_{mode}_layer_{layer_idx}.npy")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")

    outputs = np.load(output_file)
    return outputs


def load_stats(output_dir, mode):
    """Load saved statistics"""
    stats_file = os.path.join(output_dir, f"mode_{mode}_stats.json")
    if not os.path.exists(stats_file):
        return None

    with open(stats_file, 'r') as f:
        stats = json.load(f)
    return stats


def compute_comparison(fp_outputs, mode_outputs):
    """
    Compare two sets of outputs

    Returns:
        dict: Comparison metrics including mean, std, MSE, relative error, etc.
    """
    # Ensure same shape
    assert fp_outputs.shape == mode_outputs.shape, \
        f"Shape mismatch: FP {fp_outputs.shape} vs Mode {mode_outputs.shape}"

    # Compute differences
    diff = mode_outputs - fp_outputs
    abs_diff = np.abs(diff)

    # Compute metrics
    metrics = {
        # Basic statistics of FP reference
        "fp_mean": float(np.mean(fp_outputs)),
        "fp_std": float(np.std(fp_outputs)),
        "fp_min": float(np.min(fp_outputs)),
        "fp_max": float(np.max(fp_outputs)),

        # Basic statistics of mode outputs
        "mode_mean": float(np.mean(mode_outputs)),
        "mode_std": float(np.std(mode_outputs)),
        "mode_min": float(np.min(mode_outputs)),
        "mode_max": float(np.max(mode_outputs)),

        # Difference statistics
        "diff_mean": float(np.mean(diff)),
        "diff_std": float(np.std(diff)),
        "abs_diff_mean": float(np.mean(abs_diff)),
        "abs_diff_max": float(np.max(abs_diff)),

        # Error metrics
        "mse": float(np.mean(diff ** 2)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(abs_diff)),

        # Relative metrics
        "relative_mse": float(np.mean(diff ** 2) / (np.mean(fp_outputs ** 2) + 1e-8)),
        "relative_mae": float(np.mean(abs_diff) / (np.mean(np.abs(fp_outputs)) + 1e-8)),

        # Correlation
        "correlation": float(np.corrcoef(fp_outputs.flatten(), mode_outputs.flatten())[0, 1]),

        # Percentiles of absolute differences
        "abs_diff_p50": float(np.percentile(abs_diff, 50)),
        "abs_diff_p90": float(np.percentile(abs_diff, 90)),
        "abs_diff_p95": float(np.percentile(abs_diff, 95)),
        "abs_diff_p99": float(np.percentile(abs_diff, 99)),
    }

    return metrics


def print_comparison_table(layer_idx, metrics):
    """Print comparison table in a nice format"""
    print(f"\n{'='*80}")
    print(f"Layer {layer_idx} - Comparison Results")
    print(f"{'='*80}")

    # FP reference statistics
    print("\n📊 FP32 Reference Statistics:")
    print(f"  Mean:  {metrics['fp_mean']:>12.6f}")
    print(f"  Std:   {metrics['fp_std']:>12.6f}")
    print(f"  Range: [{metrics['fp_min']:>10.6f}, {metrics['fp_max']:>10.6f}]")

    # Mode statistics
    print("\n📊 Mode Output Statistics:")
    print(f"  Mean:  {metrics['mode_mean']:>12.6f}")
    print(f"  Std:   {metrics['mode_std']:>12.6f}")
    print(f"  Range: [{metrics['mode_min']:>10.6f}, {metrics['mode_max']:>10.6f}]")

    # Difference statistics
    print("\n📏 Difference (Mode - FP32):")
    print(f"  Mean Diff:     {metrics['diff_mean']:>12.6f}")
    print(f"  Std Diff:      {metrics['diff_std']:>12.6f}")
    print(f"  Mean Abs Diff: {metrics['abs_diff_mean']:>12.6f}")
    print(f"  Max Abs Diff:  {metrics['abs_diff_max']:>12.6f}")

    # Error metrics
    print("\n❌ Error Metrics:")
    print(f"  MSE:           {metrics['mse']:>12.6e}")
    print(f"  RMSE:          {metrics['rmse']:>12.6f}")
    print(f"  MAE:           {metrics['mae']:>12.6f}")

    # Relative metrics
    print("\n📈 Relative Metrics:")
    print(f"  Relative MSE:  {metrics['relative_mse']:>12.6e} ({metrics['relative_mse']*100:>6.3f}%)")
    print(f"  Relative MAE:  {metrics['relative_mae']:>12.6e} ({metrics['relative_mae']*100:>6.3f}%)")
    print(f"  Correlation:   {metrics['correlation']:>12.6f}")

    # Percentiles
    print("\n📊 Absolute Difference Percentiles:")
    print(f"  50th (Median): {metrics['abs_diff_p50']:>12.6f}")
    print(f"  90th:          {metrics['abs_diff_p90']:>12.6f}")
    print(f"  95th:          {metrics['abs_diff_p95']:>12.6f}")
    print(f"  99th:          {metrics['abs_diff_p99']:>12.6f}")

    print(f"{'='*80}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Compare mode outputs with FP32 reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare mode 2-1 with fp32 reference
  ./compare_with_fp.py 2-1

  # Compare mode 0 with custom output directory
  ./compare_with_fp.py 0 --output_dir custom_outputs

  # Use fp16 as reference instead of fp32
  ./compare_with_fp.py 2-1 --reference fp16
        """
    )
    parser.add_argument('mode', type=str,
                        help='Mode to compare (e.g., 0, 2-1, 2-4)')
    parser.add_argument('--reference', type=str, default='fp32',
                        help='Reference mode to compare against (default: fp32)')
    parser.add_argument('--output_dir', type=str, default='layer_outputs',
                        help='Directory containing saved outputs (default: layer_outputs)')
    parser.add_argument('--save_comparison', type=str, default=None,
                        help='Save comparison results to JSON file')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Comparing Mode '{args.mode}' vs Reference '{args.reference}'")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")

    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        print(f"\n❌ Error: Output directory not found: {args.output_dir}")
        print(f"Please run save_layer_outputs.py first to generate the reference data.")
        sys.exit(1)

    # Load statistics (optional, for display)
    fp_stats = load_stats(args.output_dir, args.reference)
    mode_stats = load_stats(args.output_dir, args.mode)

    # Determine layers to compare
    # We saved layer 0 and last layer (typically layer 23 for 130M model)
    # Try to find available layers
    available_layers = []
    for file in os.listdir(args.output_dir):
        if file.startswith(f"mode_{args.reference}_layer_") and file.endswith(".npy"):
            layer_idx = int(file.split("_")[-1].replace(".npy", ""))
            available_layers.append(layer_idx)

    available_layers.sort()
    print(f"\nAvailable layers: {available_layers}")

    if not available_layers:
        print(f"\n❌ Error: No layer outputs found for reference mode '{args.reference}'")
        print(f"Please run save_layer_outputs.py first:")
        print(f"  python3 save_layer_outputs.py quamba-130m-w8a8 --pretrained_dir <path> --mode {args.reference}")
        sys.exit(1)

    # Compare each layer
    all_comparisons = {}

    for layer_idx in available_layers:
        print(f"\n\n{'#'*80}")
        print(f"# Comparing Layer {layer_idx}")
        print(f"{'#'*80}")

        try:
            # Load outputs
            print(f"\nLoading outputs...")
            fp_outputs = load_outputs(args.output_dir, args.reference, layer_idx)
            mode_outputs = load_outputs(args.output_dir, args.mode, layer_idx)

            print(f"  FP reference shape: {fp_outputs.shape}")
            print(f"  Mode output shape:  {mode_outputs.shape}")

            # Compute comparison
            metrics = compute_comparison(fp_outputs, mode_outputs)

            # Print results
            print_comparison_table(layer_idx, metrics)

            # Store for later
            all_comparisons[f"layer_{layer_idx}"] = metrics

        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print(f"Please run save_layer_outputs.py for mode '{args.mode}' first:")
            print(f"  python3 save_layer_outputs.py quamba-130m-w8a8 --pretrained_dir <path> --mode {args.mode} --quantize")
            continue
        except Exception as e:
            print(f"\n❌ Error comparing layer {layer_idx}: {e}")
            continue

    # Print summary
    if all_comparisons:
        print(f"\n\n{'='*80}")
        print("SUMMARY - All Layers")
        print(f"{'='*80}")

        print(f"\n{'Layer':<10} {'MSE':<15} {'RMSE':<12} {'MAE':<12} {'Correlation':<12}")
        print("-" * 80)
        for layer_name, metrics in all_comparisons.items():
            layer_idx = layer_name.split("_")[1]
            print(f"{layer_idx:<10} {metrics['mse']:<15.6e} {metrics['rmse']:<12.6f} "
                  f"{metrics['mae']:<12.6f} {metrics['correlation']:<12.6f}")

        print(f"{'='*80}")

        # Average across all layers
        avg_mse = np.mean([m['mse'] for m in all_comparisons.values()])
        avg_mae = np.mean([m['mae'] for m in all_comparisons.values()])
        avg_corr = np.mean([m['correlation'] for m in all_comparisons.values()])

        print(f"\nAverage across all layers:")
        print(f"  MSE:         {avg_mse:.6e}")
        print(f"  MAE:         {avg_mae:.6f}")
        print(f"  Correlation: {avg_corr:.6f}")

        # Save results if requested
        if args.save_comparison:
            from datetime import datetime

            output = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reference_mode": args.reference,
                "compared_mode": args.mode,
                "output_dir": args.output_dir,
                "layers": all_comparisons,
                "summary": {
                    "avg_mse": float(avg_mse),
                    "avg_mae": float(avg_mae),
                    "avg_correlation": float(avg_corr)
                }
            }
            with open(args.save_comparison, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"\n✓ Comparison results saved to: {args.save_comparison}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
