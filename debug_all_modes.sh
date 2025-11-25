#!/bin/bash
################################################################################
# 统一调试脚本：所有 Mode 对比方案
################################################################################
# 功能：
#   1. 调试单个或多个 modes（添加打印）
#   2. 对比两个 modes 的数值差异
#   3. 保存和分析层输出
#   4. 生成 Markdown 报告
################################################################################

set -e

# ============================================================================
# 配置
# ============================================================================

FP16_MODEL="pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m"
PRETRAINED_DIR="pretrained_models/Quamba1-pa1/pa-1/quamba-130m-w8a8"
TASK="lambada_openai"
OUTPUT_DIR="debug_output"

# ============================================================================
# 颜色输出
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ============================================================================
# 辅助函数
# ============================================================================

usage() {
    cat << 'EOF'
================================================================================
统一调试脚本：所有 Mode 对比方案
================================================================================

用法:
  ./debug_all_modes.sh <command> [options]

命令:

  1. 快速调试（添加打印，运行，生成报告）
     quick <mode1> <mode2>        - 对比两个 modes 的数值差异
     示例: ./debug_all_modes.sh quick 2-0 2-1

  2. 运行单个 mode（不添加打印）
     run <mode>                   - 运行单个 mode
     示例: ./debug_all_modes.sh run 2-0

  3. 对比已有的日志文件
     compare <mode1> <mode2>      - 对比两个已存在的日志文件
     示例: ./debug_all_modes.sh compare 2-0 2-1

  4. 批量运行和对比
     batch <modes...>             - 运行多个 modes 并生成对比报告
     示例: ./debug_all_modes.sh batch 0 2-0 2-1 2-2

  5. 完整分析流程
     full                         - 运行所有 modes + 完整分析
     示例: ./debug_all_modes.sh full

可用 modes:
  0      - Baseline INT8 CUDA
  2-0    - CUDA INT8 + Requant
  2-1    - PyTorch INT8 Direct
  2-2    - FP32 PyTorch (INT8 Grid)
  2-3    - TRUE FP32 Conv + INT8 SSM
  2-4    - TRUE FP32 Conv + FP32 SSM
  3      - Hybrid Precision

常用对比组合:
  2-0 vs 2-1   - CUDA vs PyTorch (同样都是 INT8 SSM)
  2-0 vs 2-4   - INT8 grid vs TRUE FP32
  0 vs 2-0     - Baseline vs Dequant+Requant

环境变量:
  LIMIT=N                 - 测试样本数（默认：1）
  ENABLE_DEBUG_PRINT=true - 启用调试打印
  OUTPUT_DIR=path         - 输出目录（默认：debug_output）

示例:
  # 快速对比 Mode 2-0 vs 2-1（自动添加打印）
  ./debug_all_modes.sh quick 2-0 2-1

  # 批量运行并对比
  ./debug_all_modes.sh batch 0 2-0 2-1

  # 运行完整分析
  LIMIT=10 ./debug_all_modes.sh full

EOF
    exit 1
}

# ============================================================================
# 临时添加调试代码
# ============================================================================

add_debug_code() {
    print_header "临时添加调试代码"

    if [ -f "quamba/SoftEdgeSSM.py.backup" ]; then
        print_warning "发现已有备份，跳过"
        return 0
    fi

    cp quamba/SoftEdgeSSM.py quamba/SoftEdgeSSM.py.backup

    python3 << 'PYEOF'
import re

with open('quamba/SoftEdgeSSM.py', 'r') as f:
    content = f.read()

# Mode 2-0: 在 execute_mode_20_cuda_int8_requant 中 u_int8 赋值之前添加
pattern1 = r'(def execute_mode_20_cuda_int8_requant.*?""".*?""")(.*?)(# Quantize FP32 u back to INT8 using u_scale\s+u_int8 = )'
replacement1 = r'''\1\2
    # DEBUG: SSM 输入
    if layer_id in [0, 23] and os.environ.get('SSM_DEBUG_PRINT', 'false').lower() == 'true':
        print(f"\\n{'='*80}")
        print(f"[Mode 2-0] Layer {layer_id} - SSM Input")
        print(f"u: dtype={u.dtype}, range=[{u.min().item():.4f}, {u.max().item():.4f}], mean={u.mean().item():.4f}, std={u.std().item():.4f}")
        print(f"u_scale: {u_scale.item():.6f}")

    \3'''

content = re.sub(pattern1, replacement1, content, count=1, flags=re.DOTALL)

# Mode 2-0: 在 u_int8 赋值之后添加
pattern2 = r'(u_int8 = torch\.round\(u / u_scale\)\.clamp\(-128, 127\)\.to\(torch\.int8\))'
replacement2 = r'''\1

    # DEBUG: Requant 验证
    if layer_id in [0, 23] and os.environ.get('SSM_DEBUG_PRINT', 'false').lower() == 'true':
        u_dequant = u_int8.float() * u_scale
        diff = (u - u_dequant).abs()
        print(f"u_int8: range=[{u_int8.min().item()}, {u_int8.max().item()}]")
        print(f"u_dequant: mean={u_dequant.mean().item():.4f}, std={u_dequant.std().item():.4f}")
        print(f"Requant error: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")'''

content = re.sub(pattern2, replacement2, content, count=1)

# Mode 2-0: 在 return 之前添加输出
pattern3 = r'(def execute_mode_20_cuda_int8_requant.*?)(    return y)'
replacement3 = r'''\1
    # DEBUG: SSM 输出
    if layer_id in [0, 23] and os.environ.get('SSM_DEBUG_PRINT', 'false').lower() == 'true':
        y_out = y[0] if return_last_state else y
        print(f"y: dtype={y_out.dtype}, range=[{y_out.min().item():.4f}, {y_out.max().item():.4f}], mean={y_out.mean().item():.4f}, std={y_out.std().item():.4f}")
        print(f"{'='*80}\\n")

\2'''

content = re.sub(pattern3, replacement3, content, count=1, flags=re.DOTALL)

# Mode 2-1: 在函数开始处添加
pattern4 = r'(def execute_mode_21_pytorch_int8_direct.*?""".*?""")(.*?)(# Prepare A)'
replacement4 = r'''\1\2
    # DEBUG: SSM 输入
    if layer_id in [0, 23] and os.environ.get('SSM_DEBUG_PRINT', 'false').lower() == 'true':
        print(f"\\n{'='*80}")
        print(f"[Mode 2-1] Layer {layer_id} - SSM Input")
        print(f"u: dtype={u.dtype}, range=[{u.min().item()}, {u.max().item()}], mean={u.float().mean().item():.4f}")
        print(f"u_scale: {u_scale.item():.6f}")
        u_dequant = u.float() * u_scale
        print(f"u_dequant: mean={u_dequant.mean().item():.4f}, std={u_dequant.std().item():.4f}")

    \3'''

content = re.sub(pattern4, replacement4, content, count=1, flags=re.DOTALL)

# Mode 2-1: 在 return 之前添加输出
pattern5 = r'(def execute_mode_21_pytorch_int8_direct.*?)(    return y)'
replacement5 = r'''\1
    # DEBUG: SSM 输出
    if layer_id in [0, 23] and os.environ.get('SSM_DEBUG_PRINT', 'false').lower() == 'true':
        y_out = y[0] if return_last_state else y
        print(f"y: dtype={y_out.dtype}, range=[{y_out.min().item():.4f}, {y_out.max().item():.4f}], mean={y_out.mean().item():.4f}, std={y_out.std().item():.4f}")
        print(f"{'='*80}\\n")

\2'''

content = re.sub(pattern5, replacement5, content, count=1, flags=re.DOTALL)

with open('quamba/SoftEdgeSSM.py', 'w') as f:
    f.write(content)

print("✅ 调试代码已添加到 SoftEdgeSSM.py")
PYEOF

    print_success "调试代码已添加"
}

# ============================================================================
# 恢复原始代码
# ============================================================================

restore_code() {
    if [ -f "quamba/SoftEdgeSSM.py.backup" ]; then
        mv quamba/SoftEdgeSSM.py.backup quamba/SoftEdgeSSM.py
        print_success "原文件已恢复"
    fi
}

# ============================================================================
# 运行单个 mode
# ============================================================================

run_mode() {
    local mode=$1
    local enable_debug=${2:-false}
    local limit=${LIMIT:-1}

    print_header "运行 Mode $mode"

    mkdir -p ${OUTPUT_DIR}

    local log_file="${OUTPUT_DIR}/mode_${mode}.log"

    # 设置环境变量
    if [ "$enable_debug" = "true" ]; then
        export SSM_DEBUG_PRINT="true"
    fi

    # 运行测试
    python3 main.py quamba-130m-w8a8 \
        --model ${FP16_MODEL} \
        --mode ${mode} \
        --quantize \
        --eval_zero_shot \
        --task_list ${TASK} \
        --limit ${limit} \
        2>&1 | tee ${log_file}

    # 清理环境变量
    unset SSM_DEBUG_PRINT

    # 提取准确率
    local acc=$(grep -oP 'acc[^:]*:\s*\K[0-9.]+' ${log_file} | head -1)
    if [ -n "$acc" ]; then
        print_success "Mode $mode 准确率: $acc"
        echo "$acc" > ${OUTPUT_DIR}/mode_${mode}_acc.txt
    fi

    print_success "日志已保存: $log_file"
}

# ============================================================================
# 提取调试数据
# ============================================================================

extract_debug_data() {
    local mode=$1
    local log_file="${OUTPUT_DIR}/mode_${mode}.log"

    if [ ! -f "$log_file" ]; then
        print_error "日志文件不存在: $log_file"
        return 1
    fi

    # 提取 Layer 0 和 Layer 23 的数据
    grep -A 10 "\[Mode ${mode}\] Layer 0" ${log_file} > ${OUTPUT_DIR}/layer0_${mode}.txt 2>/dev/null || true
    grep -A 10 "\[Mode ${mode}\] Layer 23" ${log_file} > ${OUTPUT_DIR}/layer23_${mode}.txt 2>/dev/null || true

    if [ -s "${OUTPUT_DIR}/layer0_${mode}.txt" ]; then
        print_success "Layer 0 数据已提取"
    else
        print_warning "未找到 Layer 0 调试数据"
    fi

    if [ -s "${OUTPUT_DIR}/layer23_${mode}.txt" ]; then
        print_success "Layer 23 数据已提取"
    else
        print_warning "未找到 Layer 23 调试数据"
    fi
}

# ============================================================================
# 生成对比报告
# ============================================================================

generate_report() {
    local mode1=$1
    local mode2=$2
    local report_file="1123-debug-${mode1}-vs-${mode2}.md"

    print_header "生成对比报告"

    cat > ${report_file} << EOF
# Mode ${mode1} vs Mode ${mode2} 调试报告

**日期**: $(date '+%Y-%m-%d %H:%M:%S')
**任务**: ${TASK}
**样本数**: ${LIMIT:-1}

---

## 准确率对比

EOF

    # 添加准确率
    if [ -f "${OUTPUT_DIR}/mode_${mode1}_acc.txt" ]; then
        local acc1=$(cat ${OUTPUT_DIR}/mode_${mode1}_acc.txt)
        echo "- Mode ${mode1}: **${acc1}**" >> ${report_file}
    fi

    if [ -f "${OUTPUT_DIR}/mode_${mode2}_acc.txt" ]; then
        local acc2=$(cat ${OUTPUT_DIR}/mode_${mode2}_acc.txt)
        echo "- Mode ${mode2}: **${acc2}**" >> ${report_file}
    fi

    cat >> ${report_file} << 'EOF'

---

## Layer 0 (第1层)

### Mode ${mode1}
EOF

    if [ -f "${OUTPUT_DIR}/layer0_${mode1}.txt" ]; then
        echo '```' >> ${report_file}
        cat ${OUTPUT_DIR}/layer0_${mode1}.txt >> ${report_file}
        echo '```' >> ${report_file}
    else
        echo "无调试数据" >> ${report_file}
    fi

    cat >> ${report_file} << EOF

### Mode ${mode2}
EOF

    if [ -f "${OUTPUT_DIR}/layer0_${mode2}.txt" ]; then
        echo '```' >> ${report_file}
        cat ${OUTPUT_DIR}/layer0_${mode2}.txt >> ${report_file}
        echo '```' >> ${report_file}
    else
        echo "无调试数据" >> ${report_file}
    fi

    cat >> ${report_file} << 'EOF'

---

## Layer 23 (第24层)

### Mode ${mode1}
EOF

    if [ -f "${OUTPUT_DIR}/layer23_${mode1}.txt" ]; then
        echo '```' >> ${report_file}
        cat ${OUTPUT_DIR}/layer23_${mode1}.txt >> ${report_file}
        echo '```' >> ${report_file}
    else
        echo "无调试数据" >> ${report_file}
    fi

    cat >> ${report_file} << EOF

### Mode ${mode2}
EOF

    if [ -f "${OUTPUT_DIR}/layer23_${mode2}.txt" ]; then
        echo '```' >> ${report_file}
        cat ${OUTPUT_DIR}/layer23_${mode2}.txt >> ${report_file}
        echo '```' >> ${report_file}
    else
        echo "无调试数据" >> ${report_file}
    fi

    cat >> ${report_file} << 'EOF'

---

## 详细对比

### Layer 0 并排对比
```bash
diff -y debug_output/layer0_${mode1}.txt debug_output/layer0_${mode2}.txt
```

### Layer 23 并排对比
```bash
diff -y debug_output/layer23_${mode1}.txt debug_output/layer23_${mode2}.txt
```

---

## 日志文件

- Mode ${mode1}: `debug_output/mode_${mode1}.log`
- Mode ${mode2}: `debug_output/mode_${mode2}.log`

EOF

    # 替换变量
    sed -i "s/\${mode1}/${mode1}/g" ${report_file}
    sed -i "s/\${mode2}/${mode2}/g" ${report_file}

    print_success "报告已生成: $report_file"
}

# ============================================================================
# 主命令处理
# ============================================================================

cmd_quick() {
    local mode1=$1
    local mode2=$2

    if [ -z "$mode1" ] || [ -z "$mode2" ]; then
        print_error "用法: quick <mode1> <mode2>"
        exit 1
    fi

    print_header "快速对比: Mode $mode1 vs Mode $mode2"

    # 1. 添加调试代码
    add_debug_code

    # 2. 运行两个 modes
    run_mode $mode1 true
    run_mode $mode2 true

    # 3. 提取数据
    extract_debug_data $mode1
    extract_debug_data $mode2

    # 4. 生成报告
    generate_report $mode1 $mode2

    # 5. 恢复代码
    restore_code

    print_header "✅ 完成！"
    echo "查看报告: cat 1123-debug-${mode1}-vs-${mode2}.md"
    echo "查看对比: diff -y ${OUTPUT_DIR}/layer0_${mode1}.txt ${OUTPUT_DIR}/layer0_${mode2}.txt | less"
}

cmd_run() {
    local mode=$1

    if [ -z "$mode" ]; then
        print_error "用法: run <mode>"
        exit 1
    fi

    run_mode $mode false
}

cmd_compare() {
    local mode1=$1
    local mode2=$2

    if [ -z "$mode1" ] || [ -z "$mode2" ]; then
        print_error "用法: compare <mode1> <mode2>"
        exit 1
    fi

    extract_debug_data $mode1
    extract_debug_data $mode2
    generate_report $mode1 $mode2

    print_success "对比完成"
}

cmd_batch() {
    shift # 移除 'batch' 参数
    local modes=("$@")

    if [ ${#modes[@]} -eq 0 ]; then
        print_error "用法: batch <mode1> <mode2> ..."
        exit 1
    fi

    print_header "批量运行: ${modes[*]}"

    add_debug_code

    for mode in "${modes[@]}"; do
        run_mode $mode true
        extract_debug_data $mode
    done

    # 生成所有两两对比
    for ((i=0; i<${#modes[@]}-1; i++)); do
        for ((j=i+1; j<${#modes[@]}; j++)); do
            generate_report ${modes[i]} ${modes[j]}
        done
    done

    restore_code

    print_header "✅ 批量运行完成！"
}

cmd_full() {
    print_header "完整分析流程"

    local all_modes=(0 2-0 2-1 2-2 2-3 2-4 3)

    cmd_batch "${all_modes[@]}"

    print_header "✅ 完整分析完成！"
}

# ============================================================================
# 主入口
# ============================================================================

main() {
    local command=$1

    case $command in
        quick)
            shift
            cmd_quick "$@"
            ;;
        run)
            shift
            cmd_run "$@"
            ;;
        compare)
            shift
            cmd_compare "$@"
            ;;
        batch)
            cmd_batch "$@"
            ;;
        full)
            cmd_full
            ;;
        help|--help|-h|"")
            usage
            ;;
        *)
            print_error "未知命令: $command"
            usage
            ;;
    esac
}

# 捕获 Ctrl+C，确保恢复代码
trap restore_code EXIT

main "$@"
