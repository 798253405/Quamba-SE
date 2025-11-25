#!/usr/bin/env python3
"""
详细分析 Mode 2-0 vs Mode 2-4 的区别
"""
import json


# 读取统计数据
with open('first_layer_io_all_modes_stats.json', 'r') as f:
    stats = json.load(f)

mode_20 = stats['modes']['2-0']
mode_24 = stats['modes']['2-4']

print("=" * 100)
print("MODE 2-0 vs MODE 2-4 第一层输入输出对比分析")
print("=" * 100)
print()

# 根据 mode_config.py 查看配置差异
print("📋 模式配置定义:")
print("-" * 100)
print()
print("Mode 2-0 (CUDA, scale_factor=1.0):")
print("  - use_cuda_for_ssm: True")
print("  - scale_factor: 1.0")
print("  - 使用CUDA内核实现SSM，不做额外缩放")
print()
print("Mode 2-4 (CUDA, scale_factor=1.5):")
print("  - use_cuda_for_ssm: True")
print("  - scale_factor: 1.5")
print("  - 使用CUDA内核实现SSM，输入放大1.5倍")
print()

print("=" * 100)
print("📊 第一层输入 (Input) 对比")
print("=" * 100)
print()

print("┌─────────────────┬─────────────────────┬─────────────────────┬──────────────────┐")
print("│ 指标            │ Mode 2-0            │ Mode 2-4            │ 差异 (Δ)         │")
print("├─────────────────┼─────────────────────┼─────────────────────┼──────────────────┤")

# Input 对比
input_20 = mode_20['input']
input_24 = mode_24['input']

print(f"│ Shape           │ {str(input_20['shape']):19} │ {str(input_24['shape']):19} │ {'相同':16} │")
print(f"│ Mean            │ {input_20['mean']:19.10f} │ {input_24['mean']:19.10f} │ {input_24['mean']-input_20['mean']:16.10f} │")
print(f"│ Std             │ {input_20['std']:19.10f} │ {input_24['std']:19.10f} │ {input_24['std']-input_20['std']:16.10f} │")
print(f"│ Min             │ {input_20['min']:19.10f} │ {input_24['min']:19.10f} │ {input_24['min']-input_20['min']:16.10f} │")
print(f"│ Max             │ {input_20['max']:19.10f} │ {input_24['max']:19.10f} │ {input_24['max']-input_20['max']:16.10f} │")
print("└─────────────────┴─────────────────────┴─────────────────────┴──────────────────┘")
print()

print("前10个值对比:")
print(f"  Mode 2-0: {input_20['first_10'][:5]}")
print(f"  Mode 2-4: {input_24['first_10'][:5]}")
print(f"  相同: {input_20['first_10'] == input_24['first_10']}")
print()

print("=" * 100)
print("📊 第一层输出 (Output) 对比")
print("=" * 100)
print()

output_20 = mode_20['output']
output_24 = mode_24['output']

print("┌─────────────────┬─────────────────────┬─────────────────────┬──────────────────┐")
print("│ 指标            │ Mode 2-0            │ Mode 2-4            │ 差异 (Δ)         │")
print("├─────────────────┼─────────────────────┼─────────────────────┼──────────────────┤")
print(f"│ Shape           │ {str(output_20['shape']):19} │ {str(output_24['shape']):19} │ {'相同':16} │")
print(f"│ Mean            │ {output_20['mean']:19.10f} │ {output_24['mean']:19.10f} │ {output_24['mean']-output_20['mean']:16.10f} │")
print(f"│ Std             │ {output_20['std']:19.10f} │ {output_24['std']:19.10f} │ {output_24['std']-output_20['std']:16.10f} │")
print(f"│ Min             │ {output_20['min']:19.10f} │ {output_24['min']:19.10f} │ {output_24['min']-output_20['min']:16.10f} │")
print(f"│ Max             │ {output_20['max']:19.10f} │ {output_24['max']:19.10f} │ {output_24['max']-output_20['max']:16.10f} │")
print("└─────────────────┴─────────────────────┴─────────────────────┴──────────────────┘")
print()

print("前10个值对比:")
print(f"  Mode 2-0: {output_20['first_10'][:5]}")
print(f"  Mode 2-4: {output_24['first_10'][:5]}")
print(f"  相同: {output_20['first_10'] == output_24['first_10']}")
print()

# 计算相对误差
def calc_relative_diff(a, b):
    if abs(a) < 1e-10:
        return 0.0
    return abs(b - a) / abs(a) * 100

print("=" * 100)
print("📈 统计差异分析")
print("=" * 100)
print()

print("输入统计差异 (相对百分比):")
print(f"  Mean 相对差: {calc_relative_diff(input_20['mean'], input_24['mean']):8.6f}%")
print(f"  Std  相对差: {calc_relative_diff(input_20['std'], input_24['std']):8.6f}%")
print(f"  Min  相对差: {calc_relative_diff(input_20['min'], input_24['min']):8.6f}%")
print(f"  Max  相对差: {calc_relative_diff(input_20['max'], input_24['max']):8.6f}%")
print()

print("输出统计差异 (相对百分比):")
print(f"  Mean 相对差: {calc_relative_diff(output_20['mean'], output_24['mean']):8.6f}%")
print(f"  Std  相对差: {calc_relative_diff(output_20['std'], output_24['std']):8.6f}%")
print(f"  Min  相对差: {calc_relative_diff(output_20['min'], output_24['min']):8.6f}%")
print(f"  Max  相对差: {calc_relative_diff(output_20['max'], output_24['max']):8.6f}%")
print()

print("=" * 100)
print("🔍 关键发现")
print("=" * 100)
print()

# 判断输入是否完全相同
input_identical = (
    input_20['first_10'] == input_24['first_10'] and
    input_20['mean'] == input_24['mean'] and
    input_20['std'] == input_24['std']
)

# 判断输出是否完全相同
output_identical = (
    output_20['first_10'] == output_24['first_10'] and
    output_20['mean'] == output_24['mean'] and
    output_20['std'] == output_24['std']
)

if input_identical:
    print("✅ 输入完全相同")
    print("   → 两个mode接收相同的第一层输入张量")
else:
    print("❌ 输入存在差异")
    print(f"   → Mean差异: {abs(input_24['mean']-input_20['mean']):.2e}")
    print(f"   → Std差异:  {abs(input_24['std']-input_20['std']):.2e}")

print()

if output_identical:
    print("✅ 输出完全相同")
    print("   → scale_factor在第一层没有产生差异")
    print("   → 可能的原因:")
    print("      1. scale_factor只影响SSM模块，第一层可能不包含SSM")
    print("      2. scale_factor在后续层才生效")
else:
    mean_diff = abs(output_24['mean'] - output_20['mean'])
    std_diff = abs(output_24['std'] - output_20['std'])
    
    print("❌ 输出存在差异")
    print(f"   → Mean差异: {mean_diff:.2e}")
    print(f"   → Std差异:  {std_diff:.2e}")
    
    if mean_diff < 1e-6 and std_diff < 1e-6:
        print("   → 差异极小 (< 1e-6)，可能是数值精度误差")
    elif mean_diff < 1e-3 and std_diff < 1e-3:
        print("   → 差异较小 (< 1e-3)，scale_factor有轻微影响")
    else:
        print("   → 差异明显，scale_factor显著影响输出")

print()
print("=" * 100)
print("💡 结论与建议")
print("=" * 100)
print()

if input_identical and output_identical:
    print("第一层的输入输出完全相同，说明:")
    print("  • scale_factor 在第一层无效或未生效")
    print("  • 建议查看后续层的输出差异")
    print("  • 可以使用 save_layer_outputs.py 检查所有层")
elif input_identical and not output_identical:
    print("输入相同但输出不同，说明:")
    print("  • scale_factor 确实影响了第一层的计算")
    print("  • 1.5倍放大因子改变了量化行为")
    print("  • 建议分析这种差异对最终性能的影响")
else:
    print("输入就不相同，这不应该发生!")
    print("  • 建议检查代码逻辑")
    print("  • 确认数据加载是否使用了相同的seed")

print()
print("=" * 100)
