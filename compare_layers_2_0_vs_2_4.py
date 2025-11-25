#!/usr/bin/env python3
"""
比较 Mode 2-0 vs 2-4 的第1层和第24层输出
分析量化误差在层间的累积效应
"""
import os
import sys
import json
import argparse
from pathlib import Path


def load_stats(output_dir, mode):
    """加载模式的统计数据"""
    stats_file = os.path.join(output_dir, f"mode_{mode}_stats.json")
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Statistics file not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        return json.load(f)


def compare_layer(stats1, stats2, layer_idx, mode1, mode2):
    """比较两个mode在同一层的输出"""
    layer_key = f"layer_{layer_idx}"
    
    if layer_key not in stats1 or layer_key not in stats2:
        print(f"❌ Layer {layer_idx} data not found in one or both modes")
        return None
    
    l1 = stats1[layer_key]
    l2 = stats2[layer_key]
    
    # 计算差异
    def calc_diff(a, b):
        return b - a
    
    def calc_rel_diff(a, b):
        if abs(a) < 1e-10:
            return 0.0
        return abs(b - a) / abs(a) * 100
    
    result = {
        'layer': layer_idx,
        'mode1': mode1,
        'mode2': mode2,
        'stats1': l1,
        'stats2': l2,
        'diff': {
            'mean': calc_diff(l1['mean'], l2['mean']),
            'std': calc_diff(l1['std'], l2['std']),
            'min': calc_diff(l1['min'], l2['min']),
            'max': calc_diff(l1['max'], l2['max']),
            'abs_mean': calc_diff(l1['abs_mean'], l2['abs_mean'])
        },
        'rel_diff': {
            'mean': calc_rel_diff(l1['mean'], l2['mean']),
            'std': calc_rel_diff(l1['std'], l2['std']),
            'min': calc_rel_diff(l1['min'], l2['min']),
            'max': calc_rel_diff(l1['max'], l2['max']),
            'abs_mean': calc_rel_diff(l1['abs_mean'], l2['abs_mean'])
        }
    }
    
    return result


def print_comparison_table(results, mode1, mode2):
    """打印格式化的对比表格"""
    print(f"\n{'='*100}")
    print(f"MODE {mode1} vs MODE {mode2} 层输出对比分析")
    print(f"{'='*100}\n")
    
    for result in results:
        layer_idx = result['layer']
        layer_name = "第1层 (Layer 0)" if layer_idx == 0 else f"第24层 (Layer {layer_idx})"
        
        print(f"{'#'*100}")
        print(f"# {layer_name}")
        print(f"{'#'*100}\n")
        
        # 统计对比表
        print("┌─────────────┬──────────────────┬──────────────────┬──────────────────┬─────────────────┐")
        print("│ 指标        │ Mode {:<11}│ Mode {:<11}│ 绝对差异 (Δ)    │ 相对差异 (%)    │".format(mode1, mode2))
        print("├─────────────┼──────────────────┼──────────────────┼──────────────────┼─────────────────┤")
        
        s1 = result['stats1']
        s2 = result['stats2']
        diff = result['diff']
        rel_diff = result['rel_diff']
        
        print(f"│ Mean        │ {s1['mean']:16.10f} │ {s2['mean']:16.10f} │ {diff['mean']:16.10f} │ {rel_diff['mean']:15.6f} │")
        print(f"│ Std         │ {s1['std']:16.10f} │ {s2['std']:16.10f} │ {diff['std']:16.10f} │ {rel_diff['std']:15.6f} │")
        print(f"│ Min         │ {s1['min']:16.10f} │ {s2['min']:16.10f} │ {diff['min']:16.10f} │ {rel_diff['min']:15.6f} │")
        print(f"│ Max         │ {s1['max']:16.10f} │ {s2['max']:16.10f} │ {diff['max']:16.10f} │ {rel_diff['max']:15.6f} │")
        print(f"│ Abs Mean    │ {s1['abs_mean']:16.10f} │ {s2['abs_mean']:16.10f} │ {diff['abs_mean']:16.10f} │ {rel_diff['abs_mean']:15.6f} │")
        print("└─────────────┴──────────────────┴──────────────────┴──────────────────┴─────────────────┘\n")


def analyze_error_propagation(results):
    """分析误差在层间的传播"""
    print(f"\n{'='*100}")
    print("🔍 误差传播分析")
    print(f"{'='*100}\n")
    
    layer0 = next((r for r in results if r['layer'] == 0), None)
    layer_last = next((r for r in results if r['layer'] != 0), None)
    
    if not layer0 or not layer_last:
        print("❌ 缺少必要的层数据")
        return
    
    last_idx = layer_last['layer']
    
    print("┌──────────────────────┬─────────────────┬─────────────────┬─────────────────┐")
    print("│ 指标                 │ 第1层 (相对%)   │ 第24层 (相对%)  │ 放大倍数        │")
    print("├──────────────────────┼─────────────────┼─────────────────┼─────────────────┤")
    
    for metric in ['mean', 'std', 'abs_mean']:
        rel0 = layer0['rel_diff'][metric]
        rel_last = layer_last['rel_diff'][metric]
        
        if abs(rel0) < 1e-10:
            amplification = float('inf') if abs(rel_last) > 1e-6 else 1.0
            amp_str = "∞" if amplification == float('inf') else f"{amplification:15.2f}"
        else:
            amplification = rel_last / rel0
            amp_str = f"{amplification:15.2f}"
        
        metric_name = {
            'mean': 'Mean',
            'std': 'Std',
            'abs_mean': 'Abs Mean'
        }[metric]
        
        print(f"│ {metric_name:20} │ {rel0:15.6f} │ {rel_last:15.6f} │ {amp_str} │")
    
    print("└──────────────────────┴─────────────────┴─────────────────┴─────────────────┘\n")
    
    # 分析结论
    mean_amp = layer_last['rel_diff']['mean'] / layer0['rel_diff']['mean'] if abs(layer0['rel_diff']['mean']) > 1e-10 else float('inf')
    std_amp = layer_last['rel_diff']['std'] / layer0['rel_diff']['std'] if abs(layer0['rel_diff']['std']) > 1e-10 else float('inf')
    
    print("📊 关键发现:\n")
    
    if layer0['rel_diff']['mean'] < 1e-3 and layer_last['rel_diff']['mean'] < 1e-3:
        print("  ✅ 误差极小: 两层的相对误差都 < 0.1%")
        print("     → Mode 2-0 和 2-4 在量化精度上几乎等价")
    elif layer_last['rel_diff']['mean'] > 10 * layer0['rel_diff']['mean']:
        print(f"  ⚠️ 误差显著放大: 第24层误差是第1层的 {mean_amp:.1f} 倍")
        print("     → 量化误差在深层累积")
        print("     → Mode 2-4 (FP32) 在深层更稳定")
    else:
        print(f"  📌 误差轻微增长: 第24层误差约为第1层的 {mean_amp:.1f} 倍")
        print("     → 量化误差有累积但可控")
    
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Compare layer outputs between Mode 2-0 and 2-4")
    parser.add_argument('--output_dir', type=str, default='layer_outputs',
                        help='Directory containing layer outputs (default: layer_outputs)')
    parser.add_argument('--mode1', type=str, default='2-0',
                        help='First mode to compare (default: 2-0)')
    parser.add_argument('--mode2', type=str, default='2-4',
                        help='Second mode to compare (default: 2-4)')
    
    args = parser.parse_args()
    
    # 加载统计数据
    try:
        print(f"加载 Mode {args.mode1} 统计数据...")
        stats1 = load_stats(args.output_dir, args.mode1)
        
        print(f"加载 Mode {args.mode2} 统计数据...")
        stats2 = load_stats(args.output_dir, args.mode2)
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print(f"\n请先运行以下命令生成层输出:")
        print(f"  python save_layer_outputs.py quamba-130m-w8a8 --pretrained_dir <path> --mode {args.mode1}")
        print(f"  python save_layer_outputs.py quamba-130m-w8a8 --pretrained_dir <path> --mode {args.mode2}")
        return 1
    
    # 确定要比较的层
    layer_keys = [k for k in stats1.keys() if k.startswith('layer_')]
    layer_indices = sorted([int(k.split('_')[1]) for k in layer_keys])
    
    print(f"发现 {len(layer_indices)} 层的数据: {layer_indices}")
    
    # 比较每一层
    results = []
    for layer_idx in layer_indices:
        result = compare_layer(stats1, stats2, layer_idx, args.mode1, args.mode2)
        if result:
            results.append(result)
    
    if not results:
        print("❌ 没有可比较的层数据")
        return 1
    
    # 打印对比表格
    print_comparison_table(results, args.mode1, args.mode2)
    
    # 分析误差传播
    if len(results) >= 2:
        analyze_error_propagation(results)
    
    # 保存结果
    output_file = os.path.join(args.output_dir, f"comparison_{args.mode1}_vs_{args.mode2}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 对比结果已保存到: {output_file}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
