#!/usr/bin/env python3
"""
分析已保存的第一层输入输出数据
"""

import json
import numpy as np
import argparse
from pathlib import Path


def load_data(npz_file, json_file):
    """Load NPZ and JSON data"""
    # Load numpy data
    data = np.load(npz_file)

    # Load JSON stats
    with open(json_file, 'r') as f:
        stats = json.load(f)

    return data, stats


def print_mode_summary(mode, stats_data):
    """Print summary for a single mode"""
    print(f"\n{'='*100}")
    print(f"MODE {mode}")
    print(f"{'='*100}")

    input_stats = stats_data['input']
    output_stats = stats_data['output']

    # Input
    print(f"\n📥 INPUT (Shape: {input_stats['shape']}):")
    print(f"  First 10 values: ", end="")
    print("[" + ", ".join([f"{v:>9.6f}" for v in input_stats['first_10']]) + "]")
    print(f"  Mean:  {input_stats['mean']:>12.6f}")
    print(f"  Std:   {input_stats['std']:>12.6f}")
    print(f"  Range: [{input_stats['min']:>10.6f}, {input_stats['max']:>10.6f}]")

    # Output
    print(f"\n📤 OUTPUT (Shape: {output_stats['shape']}):")
    print(f"  First 10 values: ", end="")
    print("[" + ", ".join([f"{v:>9.6f}" for v in output_stats['first_10']]) + "]")
    print(f"  Mean:  {output_stats['mean']:>12.6f}")
    print(f"  Std:   {output_stats['std']:>12.6f}")
    print(f"  Range: [{output_stats['min']:>10.6f}, {output_stats['max']:>10.6f}]")


def compare_modes(data, stats, reference_mode='fp32'):
    """Compare all modes with reference mode"""
    print(f"\n\n{'='*100}")
    print(f"COMPARISON WITH REFERENCE MODE: {reference_mode}")
    print(f"{'='*100}\n")

    # Get reference data
    ref_input_key = f'mode_{reference_mode}_input'
    ref_output_key = f'mode_{reference_mode}_output'

    if ref_input_key not in data or ref_output_key not in data:
        print(f"⚠️  Reference mode '{reference_mode}' not found in data")
        return

    ref_input = data[ref_input_key]
    ref_output = data[ref_output_key]

    # Header
    print(f"{'Mode':<10} {'Input MSE':<15} {'Input MAE':<12} {'Output MSE':<15} {'Output MAE':<12} {'Out Corr':<12}")
    print("-" * 100)

    # Compare each mode
    modes = []
    for key in sorted(data.files):
        if key.startswith('mode_') and key.endswith('_input'):
            mode = key.replace('mode_', '').replace('_input', '')
            if mode != reference_mode:
                modes.append(mode)

    for mode in sorted(modes):
        input_key = f'mode_{mode}_input'
        output_key = f'mode_{mode}_output'

        mode_input = data[input_key]
        mode_output = data[output_key]

        # Calculate metrics for input
        input_diff = mode_input - ref_input
        input_mse = np.mean(input_diff ** 2)
        input_mae = np.mean(np.abs(input_diff))

        # Calculate metrics for output
        output_diff = mode_output - ref_output
        output_mse = np.mean(output_diff ** 2)
        output_mae = np.mean(np.abs(output_diff))
        output_corr = np.corrcoef(ref_output.flatten(), mode_output.flatten())[0, 1]

        print(f"{mode:<10} {input_mse:<15.6e} {input_mae:<12.6f} {output_mse:<15.6e} {output_mae:<12.6f} {output_corr:<12.9f}")

    print("=" * 100)


def print_statistics_table(stats):
    """Print statistics table for all modes"""
    print(f"\n\n{'='*100}")
    print("STATISTICS SUMMARY TABLE")
    print(f"{'='*100}\n")

    modes = sorted(stats['modes'].keys())

    # Input statistics
    print("📥 INPUT STATISTICS:")
    print(f"{'Mode':<10} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-" * 100)
    for mode in modes:
        data = stats['modes'][mode]['input']
        print(f"{mode:<10} {data['mean']:<15.6f} {data['std']:<15.6f} {data['min']:<15.6f} {data['max']:<15.6f}")

    # Output statistics
    print(f"\n📤 OUTPUT STATISTICS:")
    print(f"{'Mode':<10} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-" * 100)
    for mode in modes:
        data = stats['modes'][mode]['output']
        print(f"{mode:<10} {data['mean']:<15.6f} {data['std']:<15.6f} {data['min']:<15.6f} {data['max']:<15.6f}")

    print("=" * 100)


def analyze_value_distribution(data, stats):
    """Analyze and print value distribution"""
    print(f"\n\n{'='*100}")
    print("VALUE DISTRIBUTION ANALYSIS")
    print(f"{'='*100}\n")

    modes = sorted(stats['modes'].keys())

    print("📊 OUTPUT VALUE PERCENTILES:")
    print(f"{'Mode':<10} {'P1':<12} {'P5':<12} {'P25':<12} {'P50':<12} {'P75':<12} {'P95':<12} {'P99':<12}")
    print("-" * 100)

    for mode in modes:
        output_key = f'mode_{mode}_output'
        if output_key in data:
            output_data = data[output_key].flatten()
            percentiles = np.percentile(output_data, [1, 5, 25, 50, 75, 95, 99])
            print(f"{mode:<10} ", end="")
            for p in percentiles:
                print(f"{p:<12.6f} ", end="")
            print()

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Analyze first layer I/O data")
    parser.add_argument('--npz_file', type=str, default='first_layer_io_all_modes.npz',
                        help='NPZ file path (default: first_layer_io_all_modes.npz)')
    parser.add_argument('--json_file', type=str, default='first_layer_io_all_modes_stats.json',
                        help='JSON file path (default: first_layer_io_all_modes_stats.json)')
    parser.add_argument('--reference', type=str, default='fp32',
                        help='Reference mode for comparison (default: fp32)')
    parser.add_argument('--detail', action='store_true',
                        help='Print detailed statistics for each mode')

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.npz_file).exists():
        print(f"❌ Error: NPZ file not found: {args.npz_file}")
        print(f"\nPlease run first:")
        print(f"  ./capture_first_layer.sh")
        return

    if not Path(args.json_file).exists():
        print(f"❌ Error: JSON file not found: {args.json_file}")
        return

    # Load data
    print("\n" + "="*100)
    print("FIRST LAYER I/O ANALYSIS")
    print("="*100)

    data, stats = load_data(args.npz_file, args.json_file)

    print(f"\n📁 Data loaded from:")
    print(f"  NPZ:  {args.npz_file}")
    print(f"  JSON: {args.json_file}")

    print(f"\n📊 Metadata:")
    print(f"  Timestamp:      {stats.get('timestamp', 'N/A')}")
    print(f"  Model:          {stats.get('model', 'N/A')}")
    print(f"  Pretrained dir: {stats.get('pretrained_dir', 'N/A')}")
    print(f"  Sequence length: {stats.get('seq_len', 'N/A')}")
    print(f"  Number of modes: {len(stats['modes'])}")

    # Print detailed statistics for each mode
    if args.detail:
        print("\n\n" + "="*100)
        print("DETAILED MODE STATISTICS")
        print("="*100)

        for mode in sorted(stats['modes'].keys()):
            print_mode_summary(mode, stats['modes'][mode])

    # Print statistics table
    print_statistics_table(stats)

    # Print value distribution
    analyze_value_distribution(data, stats)

    # Compare with reference mode
    compare_modes(data, stats, reference_mode=args.reference)

    # Summary insights
    print(f"\n\n{'='*100}")
    print("KEY INSIGHTS")
    print(f"{'='*100}\n")

    modes = sorted(stats['modes'].keys())

    # Find mode with highest/lowest output mean
    output_means = {mode: stats['modes'][mode]['output']['mean'] for mode in modes}
    max_mean_mode = max(output_means.items(), key=lambda x: x[1])
    min_mean_mode = min(output_means.items(), key=lambda x: x[1])

    print(f"📈 Highest output mean: Mode {max_mean_mode[0]} ({max_mean_mode[1]:.6f})")
    print(f"📉 Lowest output mean:  Mode {min_mean_mode[0]} ({min_mean_mode[1]:.6f})")

    # Find mode with highest/lowest output std
    output_stds = {mode: stats['modes'][mode]['output']['std'] for mode in modes}
    max_std_mode = max(output_stds.items(), key=lambda x: x[1])
    min_std_mode = min(output_stds.items(), key=lambda x: x[1])

    print(f"\n📊 Highest output std:  Mode {max_std_mode[0]} ({max_std_mode[1]:.6f})")
    print(f"📊 Lowest output std:   Mode {min_std_mode[0]} ({min_std_mode[1]:.6f})")

    # Calculate mean/std variation across modes
    mean_values = list(output_means.values())
    std_values = list(output_stds.values())

    print(f"\n📏 Output mean variation across modes: {np.std(mean_values):.6e}")
    print(f"📏 Output std variation across modes:  {np.std(std_values):.6e}")

    print(f"\n{'='*100}\n")


if __name__ == '__main__':
    main()
