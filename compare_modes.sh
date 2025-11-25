#!/bin/bash

# ============================================================================
# 统一的 Mode 对比脚本
# ============================================================================
# 功能：
#   1. 保存所有 modes 的层输出
#   2. 对比所有 modes 与 FP32 参考
#   3. 对比任意两个 modes
#   4. 生成汇总表格
# ============================================================================

set -e

# 配置
PRETRAINED_DIR="${PRETRAINED_DIR:-pretrained_models/Quamba1-pa9999/pa-0.9999}"
MAMBA_MODEL="${MAMBA_MODEL:-pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m}"
OUTPUT_DIR="${OUTPUT_DIR:-layer_outputs}"
COMPARISON_DIR="${COMPARISON_DIR:-layer_outputs/comparisons}"
MODEL="quamba-130m-w8a8"
CALIB_NUM="${CALIB_NUM:-10}"
CALIB_LEN="${CALIB_LEN:-512}"

# 快速模式：layer comparison 只需要 1 个样本
QUICK_MODE="${QUICK_MODE:-false}"
if [ "$QUICK_MODE" = "true" ]; then
    CALIB_NUM=1
fi

# 所有可用的 modes
ALL_MODES="1 fp32 0 2-0 2-1 2-2 2-3 2-4 3"
QUANT_MODES="0 2-0 2-1 2-2 2-3 2-4 3"

# ============================================================================
# 辅助函数
# ============================================================================

usage() {
    echo "============================================================================"
    echo "Mode 对比脚本 - 统一工具"
    echo "============================================================================"
    echo ""
    echo "用法:"
    echo "  $0 save [modes]              - 保存指定 modes 的层输出"
    echo "  $0 compare <mode1> <mode2>   - 对比两个 modes"
    echo "  $0 compare_all               - 对比所有 modes 与 FP32 参考"
    echo "  $0 all                       - 完整流程：保存所有 + 对比所有"
    echo ""
    echo "示例:"
    echo "  $0 save                      - 保存所有 modes（默认 10 个样本）"
    echo "  $0 save fp32 0 2-4           - 只保存指定的 modes"
    echo "  $0 save fp_only              - 只保存 FP32 参考"
    echo "  $0 save quant_only           - 只保存量化 modes"
    echo "  $0 compare fp32 2-0          - 对比 FP32 vs Mode 2-0"
    echo "  $0 compare 2-0 2-4           - 对比 Mode 2-0 vs Mode 2-4"
    echo "  $0 compare_all               - 对比所有 modes 与 FP32"
    echo "  $0 all                       - 完整分析流程"
    echo ""
    echo "环境变量:"
    echo "  QUICK_MODE=true              - 快速模式，只用 1 个样本（用于 layer comparison）"
    echo "  CALIB_NUM=N                  - 自定义样本数量（默认：10）"
    echo "  CALIB_LEN=N                  - 自定义序列长度（默认：512）"
    echo ""
    echo "快速模式示例:"
    echo "  QUICK_MODE=true $0 save      - 用 1 个样本快速保存所有 modes"
    echo "  QUICK_MODE=true $0 all       - 快速完整流程（1 个样本）"
    echo ""
    echo "可用 modes: ${ALL_MODES}"
    echo ""
    exit 1
}

# 保存单个 mode 的输出
save_mode() {
    local mode=$1

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "保存 Mode $mode"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 检查是否已存在
    local stats_file="${OUTPUT_DIR}/mode_${mode}_stats.json"
    if [ -f "$stats_file" ]; then
        echo "ℹ️  发现已有数据: $stats_file"
        read -p "是否重新生成? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "✓ 跳过，使用现有数据"
            return 0
        fi
    fi

    # 根据 mode 类型选择不同的保存方式
    if [ "$mode" = "1" ]; then
        # Mode 1: 原始 Mamba FP16
        echo "使用原始 Mamba FP16 模型: ${MAMBA_MODEL}"
        python3 save_layer_outputs.py ${MAMBA_MODEL} \
            --output_dir ${OUTPUT_DIR} \
            --mode_name ${mode} \
            --calib_data_num ${CALIB_NUM} \
            --calib_seqlen ${CALIB_LEN}
    elif [ "$mode" = "fp32" ] || [ "$mode" = "fp16" ]; then
        # FP modes: Quamba模型，Linear INT8 + Conv1D/SSM FP32
        echo "保存混合精度模式: ${mode} (Linear INT8 + Conv1D/SSM FP32)"
        python3 save_layer_outputs.py ${MODEL} \
            --pretrained_dir ${PRETRAINED_DIR} \
            --mode ${mode} \
            --output_dir ${OUTPUT_DIR} \
            --calib_data_num ${CALIB_NUM} \
            --calib_seqlen ${CALIB_LEN}
    else
        # 量化 modes
        echo "保存量化模式: ${mode}"
        python3 save_layer_outputs.py ${MODEL} \
            --pretrained_dir ${PRETRAINED_DIR} \
            --mode ${mode} \
            --quantize \
            --output_dir ${OUTPUT_DIR} \
            --calib_data_num ${CALIB_NUM} \
            --calib_seqlen ${CALIB_LEN}
    fi

    if [ $? -eq 0 ]; then
        echo "✓ Mode $mode 保存成功"
    else
        echo "❌ Mode $mode 保存失败"
        return 1
    fi
}

# 对比两个 modes
compare_two_modes() {
    local mode1=$1
    local mode2=$2

    echo ""
    echo "============================================================================"
    echo "对比 Mode $mode1 vs Mode $mode2"
    echo "============================================================================"

    # 检查数据是否存在
    local stats1="${OUTPUT_DIR}/mode_${mode1}_stats.json"
    local stats2="${OUTPUT_DIR}/mode_${mode2}_stats.json"

    if [ ! -f "$stats1" ]; then
        echo "❌ 错误: Mode $mode1 数据不存在: $stats1"
        echo "请先运行: $0 save $mode1"
        return 1
    fi

    if [ ! -f "$stats2" ]; then
        echo "❌ 错误: Mode $mode2 数据不存在: $stats2"
        echo "请先运行: $0 save $mode2"
        return 1
    fi

    # 运行对比
    python3 compare_layers_2_0_vs_2_4.py \
        --output_dir ${OUTPUT_DIR} \
        --mode1 ${mode1} \
        --mode2 ${mode2}

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 对比完成!"
        echo "输出文件: ${OUTPUT_DIR}/comparison_${mode1}_vs_${mode2}.json"
    else
        echo "❌ 对比失败"
        return 1
    fi
}

# 对比所有 modes 与 FP32 参考
compare_all_with_fp() {
    local reference="${1:-fp32}"

    echo ""
    echo "============================================================================"
    echo "对比所有 modes 与 ${reference} 参考"
    echo "============================================================================"

    # 检查参考数据是否存在
    if [ ! -f "${OUTPUT_DIR}/mode_${reference}_stats.json" ]; then
        echo "❌ 错误: ${reference} 参考数据不存在"
        echo "请先运行: $0 save ${reference}"
        return 1
    fi

    # 创建对比目录
    mkdir -p ${COMPARISON_DIR}

    # 对比每个 mode
    echo ""
    echo "对比 modes: ${QUANT_MODES}"
    echo ""

    for mode in ${QUANT_MODES}; do
        if [ ! -f "${OUTPUT_DIR}/mode_${mode}_stats.json" ]; then
            echo "⚠️  跳过 Mode ${mode}: 数据不存在"
            continue
        fi

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "对比 Mode: ${mode}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        python3 compare_with_fp.py ${mode} \
            --reference ${reference} \
            --output_dir ${OUTPUT_DIR} \
            --save_comparison ${COMPARISON_DIR}/mode_${mode}_vs_${reference}.json

        if [ $? -eq 0 ]; then
            echo "✓ Mode ${mode} 对比完成"
        else
            echo "✗ Mode ${mode} 对比失败"
        fi
        echo ""
    done

    # 生成汇总表格
    echo ""
    echo "============================================================================"
    echo "生成汇总表格"
    echo "============================================================================"
    echo ""

    python3 - <<EOF
import json
import os

comparison_dir = "${COMPARISON_DIR}"
modes = "${QUANT_MODES}".split()
reference = "${reference}"

print("\n" + "="*100)
print(f"汇总表格 - 所有 Modes vs {reference}")
print("="*100)
print()

# Header
print(f"{'Mode':<10} {'Layer 0 MSE':<15} {'Layer 0 MAE':<12} {'Last Layer MSE':<15} {'Last Layer MAE':<12} {'Avg Corr':<12}")
print("-" * 100)

for mode in modes:
    comp_file = os.path.join(comparison_dir, f"mode_{mode}_vs_{reference}.json")
    if not os.path.exists(comp_file):
        print(f"{mode:<10} {'N/A':<15} {'N/A':<12} {'N/A':<15} {'N/A':<12} {'N/A':<12}")
        continue

    with open(comp_file, 'r') as f:
        data = json.load(f)

    layers = data['layers']
    layer_names = sorted(layers.keys(), key=lambda x: int(x.split('_')[1]))

    if len(layer_names) >= 1:
        first_layer = layers[layer_names[0]]
        last_layer = layers[layer_names[-1]]

        avg_corr = data['summary']['avg_correlation']

        print(f"{mode:<10} {first_layer['mse']:<15.6e} {first_layer['mae']:<12.6f} "
              f"{last_layer['mse']:<15.6e} {last_layer['mae']:<12.6f} {avg_corr:<12.6f}")

print("="*100)
print()

# Key insights
print("📊 关键发现:")
print()

# Find best mode (lowest last layer MSE)
best_mode = None
best_mse = float('inf')

for mode in modes:
    comp_file = os.path.join(comparison_dir, f"mode_{mode}_vs_{reference}.json")
    if not os.path.exists(comp_file):
        continue

    with open(comp_file, 'r') as f:
        data = json.load(f)

    layers = data['layers']
    layer_names = sorted(layers.keys(), key=lambda x: int(x.split('_')[1]))
    if len(layer_names) > 0:
        last_layer = layers[layer_names[-1]]
        if last_layer['mse'] < best_mse:
            best_mse = last_layer['mse']
            best_mode = mode

if best_mode:
    print(f"  ✓ 最佳模式: Mode {best_mode} (最低 MSE: {best_mse:.6e})")
print()

EOF

    echo ""
    echo "✓ 汇总表格生成完成!"
    echo ""
    echo "对比结果保存在: ${COMPARISON_DIR}/"
    echo ""
}

# ============================================================================
# 主流程
# ============================================================================

# 检查参数
if [ $# -eq 0 ]; then
    usage
fi

COMMAND=$1
shift

case $COMMAND in
    save)
        echo "============================================================================"
        echo "保存 Modes 层输出"
        echo "============================================================================"
        echo ""
        echo "输出目录: ${OUTPUT_DIR}"
        echo "样本数量: ${CALIB_NUM}"
        echo "序列长度: ${CALIB_LEN}"

        # 解析要保存的 modes
        if [ $# -eq 0 ]; then
            # 默认：保存所有 modes
            MODE_LIST="${ALL_MODES}"
        elif [ "$1" = "fp_only" ]; then
            MODE_LIST="fp32"
        elif [ "$1" = "quant_only" ]; then
            MODE_LIST="${QUANT_MODES}"
        elif [ "$1" = "essential" ]; then
            MODE_LIST="1 fp32 0 2-1 2-2 2-4"
        else
            MODE_LIST="$@"
        fi

        echo "Modes: ${MODE_LIST}"
        echo ""

        # 创建输出目录
        mkdir -p ${OUTPUT_DIR}

        # 保存每个 mode
        for mode in ${MODE_LIST}; do
            save_mode "$mode"
        done

        echo ""
        echo "============================================================================"
        echo "✓ 所有 modes 保存完成!"
        echo "============================================================================"
        echo ""
        echo "保存位置: ${OUTPUT_DIR}"
        echo ""
        echo "下一步:"
        echo "  对比两个 modes:  $0 compare fp32 2-0"
        echo "  对比所有 modes:  $0 compare_all"
        echo ""
        ;;

    compare)
        if [ $# -ne 2 ]; then
            echo "❌ 错误: compare 需要两个参数"
            echo "用法: $0 compare <mode1> <mode2>"
            exit 1
        fi
        compare_two_modes "$1" "$2"
        ;;

    compare_all)
        REFERENCE="${1:-fp32}"
        compare_all_with_fp "$REFERENCE"
        ;;

    all)
        echo "============================================================================"
        echo "完整分析流程"
        echo "============================================================================"
        echo ""
        echo "步骤 1: 保存所有 modes"
        echo "步骤 2: 对比所有 modes 与 FP32"
        echo ""

        # 保存所有 modes
        $0 save

        # 对比所有 modes
        $0 compare_all

        echo ""
        echo "============================================================================"
        echo "✓ 完整分析流程完成!"
        echo "============================================================================"
        echo ""
        ;;

    *)
        echo "❌ 错误: 未知命令 '$COMMAND'"
        usage
        ;;
esac
