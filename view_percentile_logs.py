#!/usr/bin/env python3
"""
Percentile日志查看工具

用法:
    python view_percentile_logs.py                    # 查看所有实验
    python view_percentile_logs.py --last 5           # 查看最近5次实验
    python view_percentile_logs.py --compare pa1.0,default  # 对比两个实验
    python view_percentile_logs.py --filter "mamba-370m"    # 筛选特定模型
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def load_experiments(log_file):
    """加载所有实验记录"""
    experiments = []
    if not Path(log_file).exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return experiments

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    experiments.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  解析JSON失败: {e}")
                    continue

    return experiments


def print_experiment_summary(exp, index=None):
    """打印单个实验的摘要"""
    header = f"实验 #{index}" if index is not None else "实验"
    print(f"\n{'='*80}")
    print(f"{header} - {exp['timestamp']}")
    print(f"{'='*80}")

    # 配置信息
    config = exp['config']
    print(f"\n📋 配置:")
    print(f"  模型: {config['model']}")
    print(f"  量化: W{config['w_bits']}A{config['a_bits']}")
    print(f"  Percentile Alpha: {config.get('percentile_alpha', 'default')}")
    print(f"  Group Heads: {config['group_heads']}")
    print(f"  GPTQ: {config['apply_gptq']}")

    # 激活统计（只显示前3层）
    if exp['activation_stats']:
        print(f"\n📊 激活统计 (前3层):")
        count = 0
        for layer_name, stats in exp['activation_stats'].items():
            if count >= 3:
                break
            print(f"\n  {layer_name}:")

            if 'before_percentile' in stats:
                before = stats['before_percentile']
                after = stats['after_percentile']
                print(f"    裁剪前: [{before['min']:.2f}, {before['max']:.2f}] 范围={before['range']:.2f}")
                print(f"    裁剪后: [{after['min']:.2f}, {after['max']:.2f}] 范围={after['range']:.2f}")

                if 'range_reduction' in stats:
                    print(f"    范围缩小: {stats['range_reduction']*100:.2f}%")
                if 'clipped_ratio' in stats:
                    print(f"    裁剪比例: {stats['clipped_ratio']*100:.4f}%")

            count += 1

        if len(exp['activation_stats']) > 3:
            print(f"\n  ... 共{len(exp['activation_stats'])}层")

    # Reorder效果
    reorder = exp.get('reorder_summary', {})
    if reorder.get('enabled'):
        print(f"\n🔄 Reorder效果:")
        print(f"  影响层数: {reorder.get('total_layers', 0)}")
        if reorder.get('avg_range_reduction') is not None:
            print(f"  平均范围缩小: {reorder['avg_range_reduction']:.2f}%")

    # 最终结果
    if 'results' in exp and exp['results']:
        results = exp['results']
        print(f"\n🎯 最终结果:")
        print(f"  Accuracy: {results.get('accuracy', 0)*100:.2f}%")
        print(f"  Perplexity: {results.get('perplexity', 0):.3f}")

    # 命令
    print(f"\n💻 命令:")
    print(f"  {exp['command']}")


def compare_experiments(exp1, exp2, label1="实验1", label2="实验2"):
    """对比两个实验"""
    print(f"\n{'='*80}")
    print(f"📊 对比: {label1} vs {label2}")
    print(f"{'='*80}")

    # 配置对比
    print(f"\n配置对比:")
    print(f"  {'项目':<25} {label1:<25} {label2:<25}")
    print(f"  {'-'*75}")

    config1 = exp1['config']
    config2 = exp2['config']

    fields = [
        ('模型', 'model'),
        ('量化', lambda c: f"W{c['w_bits']}A{c['a_bits']}"),
        ('Percentile Alpha', lambda c: c.get('percentile_alpha', 'default')),
        ('Group Heads', 'group_heads'),
        ('GPTQ', 'apply_gptq'),
    ]

    for label, key in fields:
        if callable(key):
            val1 = key(config1)
            val2 = key(config2)
        else:
            val1 = config1.get(key, 'N/A')
            val2 = config2.get(key, 'N/A')

        diff_marker = "" if val1 == val2 else " ⚠️"
        print(f"  {label:<25} {str(val1):<25} {str(val2):<25}{diff_marker}")

    # 结果对比
    if 'results' in exp1 and 'results' in exp2:
        r1 = exp1['results']
        r2 = exp2['results']

        print(f"\n🎯 结果对比:")
        acc1 = r1.get('accuracy', 0) * 100
        acc2 = r2.get('accuracy', 0) * 100
        acc_diff = acc2 - acc1

        ppl1 = r1.get('perplexity', 0)
        ppl2 = r2.get('perplexity', 0)
        ppl_diff = ppl2 - ppl1

        print(f"  Accuracy: {acc1:.2f}% vs {acc2:.2f}% (差异: {acc_diff:+.2f}%)")
        print(f"  Perplexity: {ppl1:.3f} vs {ppl2:.3f} (差异: {ppl_diff:+.3f})")

    # 激活范围对比（第一层）
    if exp1['activation_stats'] and exp2['activation_stats']:
        print(f"\n📊 激活范围对比（第一层）:")

        # 获取第一层的名称
        layer1 = list(exp1['activation_stats'].keys())[0]
        layer2 = list(exp2['activation_stats'].keys())[0]

        stats1 = exp1['activation_stats'][layer1]
        stats2 = exp2['activation_stats'][layer2]

        if 'before_percentile' in stats1 and 'before_percentile' in stats2:
            print(f"\n  裁剪前范围:")
            r1 = stats1['before_percentile']['range']
            r2 = stats2['before_percentile']['range']
            print(f"    {label1}: {r1:.2f}")
            print(f"    {label2}: {r2:.2f}")

            print(f"\n  裁剪后范围:")
            r1 = stats1['after_percentile']['range']
            r2 = stats2['after_percentile']['range']
            print(f"    {label1}: {r1:.2f}")
            print(f"    {label2}: {r2:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Percentile实验日志查看工具')
    parser.add_argument('--log_file', default='logs/percentile_experiments.jsonl',
                        help='日志文件路径')
    parser.add_argument('--last', type=int, help='显示最近N次实验')
    parser.add_argument('--filter', help='筛选包含关键词的实验')
    parser.add_argument('--compare', help='对比两个实验的索引，用逗号分隔（如: 1,2）')

    args = parser.parse_args()

    # 加载实验
    experiments = load_experiments(args.log_file)
    if not experiments:
        print("没有找到任何实验记录")
        return

    print(f"\n✅ 加载了 {len(experiments)} 个实验记录")

    # 筛选
    if args.filter:
        experiments = [
            exp for exp in experiments
            if args.filter.lower() in exp['config']['model'].lower() or
               args.filter.lower() in exp['command'].lower()
        ]
        print(f"✅ 筛选后剩余 {len(experiments)} 个实验")

    # 限制数量
    if args.last:
        experiments = experiments[-args.last:]
        print(f"✅ 显示最近 {len(experiments)} 个实验")

    # 对比模式
    if args.compare:
        indices = [int(i.strip()) for i in args.compare.split(',')]
        if len(indices) != 2:
            print("❌ --compare参数需要两个索引，用逗号分隔")
            return

        idx1, idx2 = indices
        if idx1 >= len(experiments) or idx2 >= len(experiments):
            print(f"❌ 索引超出范围（共{len(experiments)}个实验）")
            return

        compare_experiments(
            experiments[idx1], experiments[idx2],
            label1=f"实验#{idx1}", label2=f"实验#{idx2}"
        )
        return

    # 显示所有实验
    for i, exp in enumerate(experiments):
        print_experiment_summary(exp, index=i)


if __name__ == '__main__':
    main()
