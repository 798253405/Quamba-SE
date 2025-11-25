#!/bin/bash

# ============================================================================
# Quamba 快速运行脚本 - 所有模式（统一 --mode 参数版本）
# ============================================================================

MODEL="quamba-130m-w8a8"
PRETRAINED_DIR="pretrained_models/Quamba1-pa9999/pa-0.9999"
MAMBA_MODEL="pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m"
TASK="lambada_openai"
BASE_ARGS="--quantize --eval_zero_shot --task_list ${TASK} --testing"  # Add --testing for quick test

echo "============================================================================"
echo "Quamba Mode Quick Run"
echo "============================================================================"
echo ""
echo "Usage: $0 <mode>"
echo ""
echo "Available modes:"
echo "  1      - Original Mamba FP16 (Upper Bound)"
echo "  0      - Baseline INT8 CUDA"
echo "  2-0    - CUDA INT8 + Requantization"
echo "  2-1    - PyTorch INT8 Direct"
echo "  2-2    - FP32 PyTorch (INT8 Grid)"
echo "  2-3    - TRUE FP32 Conv + INT8 SSM"
echo "  2-4    - TRUE FP32 Conv + FP32 SSM"
echo "  3      - Hybrid Precision"
echo "  all    - Run all modes"
echo ""
#mode 0： total baseline CUDA INT8 CONV                         + CUDA INT8 SSM Input

#mode 2-0: CUDA INT8->FP32Inter->INT8CONVOut + INT8-FLOAT（Dequant）       + floattoINT&CUDA INT8 SSM Input
#mode 2-1: CUDA INT8->FP32Inter->INT8CONVOut                    + Torch Int8 SSM Kernel
#mode 2-2: CUDA INT8->FP32Inter->INT8CONVOut + INT8-FLOAT       + Torch FP32 SSM Kernel
#mode 2-3: CUDA INT8->FP32Inter->FP32CONV +                     + floattoINT Torch Int8 SSM Kernel
#mode 2-4: CUDA INT8->FP32Inter->FP32CONV +                     + Torch FP32 SSM Kernel
#mode 3: CUDA FP32->FP32Inter->FP32CONV +                       + Torch FP32 SSM Kernel             + int8 linear

if [ $# -eq 0 ]; then
    echo "Error: No mode specified"
    echo "Example: $0 2-4"
    exit 1
fi

MODE=$1

# 清除所有模式相关的环境变量（保留清空重置选项）
clear_env_vars() {
    unset FLOAT_SIM_ASIC_INT8
    unset SSM_USE_CUDA_FOR_FP32
    unset SSM_USE_PYTORCH_INT8
    unset CONV1D_MODE23_FP32
    unset CONV1D_MODE24_FP32
    unset CONV1D_MODE3_FP32
    unset FP32_SSM_INPUT
    unset QUAMBA_MODE
}

run_mode() {
    local mode=$1
    local log_dir="allmodeComparison/testing/mode${mode//-/}"

    echo "============================================================================"
    echo "Running Mode $mode"
    echo "============================================================================"

    # 清除所有环境变量，确保干净的状态
    clear_env_vars

    if [ "$mode" = "1" ]; then
        # Mode 1: Original Mamba FP16 (Upper Bound)
        echo "Using original Mamba FP16 model: ${MAMBA_MODEL}"
        python3 main.py ${MAMBA_MODEL} \
            --eval_zero_shot --task_list ${TASK} --testing \
            --log_dir ${log_dir}
    else
        # Quantized modes: Use unified --mode parameter
        python3 main.py ${MODEL} \
            --pretrained_dir ${PRETRAINED_DIR} \
            --mode ${mode} \
            ${BASE_ARGS} \
            --log_dir ${log_dir}
    fi
}

case $MODE in
    1|0|2-0|2-1|2-2|2-3|2-4|3)
        run_mode "$MODE"
        ;;
    all)
        echo "Running all modes..."
        run_mode "1"
        run_mode "0"
        run_mode "2-0"
        run_mode "2-1"
        run_mode "2-2"
        run_mode "2-3"
        run_mode "2-4"
        run_mode "3"
        echo ""
        echo "============================================================================"
        echo "All modes completed!"
        echo "============================================================================"
        ;;
    *)
        echo "Error: Invalid mode '$MODE'"
        echo "Use: 1, 0, 2-0, 2-1, 2-2, 2-3, 2-4, 3, or all"
        exit 1
        ;;
esac
