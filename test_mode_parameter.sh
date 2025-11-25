#!/bin/bash

# ============================================================================
# 测试新的 --mode 参数系统
# ============================================================================

echo "============================================================================"
echo "Testing new --mode parameter system"
echo "============================================================================"
echo ""

# 测试脚本语法
echo "1. Testing QUICK_RUN.sh syntax..."
bash -n QUICK_RUN.sh
if [ $? -eq 0 ]; then
    echo "   ✓ QUICK_RUN.sh syntax OK"
else
    echo "   ✗ QUICK_RUN.sh syntax error"
    exit 1
fi

echo ""
echo "2. Testing clear_env_vars function..."
source QUICK_RUN.sh

# 设置一些测试环境变量
export FLOAT_SIM_ASIC_INT8=true
export SSM_USE_CUDA_FOR_FP32=true
export CONV1D_MODE23_FP32=true

echo "   Before clear: FLOAT_SIM_ASIC_INT8=${FLOAT_SIM_ASIC_INT8}"
echo "   Before clear: SSM_USE_CUDA_FOR_FP32=${SSM_USE_CUDA_FOR_FP32}"
echo "   Before clear: CONV1D_MODE23_FP32=${CONV1D_MODE23_FP32}"

# 清除环境变量
clear_env_vars

if [ -z "$FLOAT_SIM_ASIC_INT8" ] && [ -z "$SSM_USE_CUDA_FOR_FP32" ] && [ -z "$CONV1D_MODE23_FP32" ]; then
    echo "   ✓ clear_env_vars() works correctly"
else
    echo "   ✗ clear_env_vars() failed"
    echo "   After clear: FLOAT_SIM_ASIC_INT8=${FLOAT_SIM_ASIC_INT8}"
    echo "   After clear: SSM_USE_CUDA_FOR_FP32=${SSM_USE_CUDA_FOR_FP32}"
    echo "   After clear: CONV1D_MODE23_FP32=${CONV1D_MODE23_FP32}"
    exit 1
fi

echo ""
echo "3. Testing mode_config.py..."
python3 -c "
from quamba.mode_config import MODE_CONFIG, setup_quamba_mode, clear_all_mode_env_vars
import os

# Test 1: Clear all env vars
clear_all_mode_env_vars()
print('   ✓ clear_all_mode_env_vars() works')

# Test 2: Setup mode 0
setup_quamba_mode('0', verbose=False)
assert all(os.environ.get(var, 'false').lower() == 'false' for var in ['FLOAT_SIM_ASIC_INT8', 'SSM_USE_CUDA_FOR_FP32'])
print('   ✓ Mode 0 setup correctly')

# Test 3: Setup mode 2-1
setup_quamba_mode('2-1', verbose=False)
assert os.environ.get('FLOAT_SIM_ASIC_INT8', 'false').lower() == 'true'
assert os.environ.get('SSM_USE_PYTORCH_INT8', 'false').lower() == 'true'
print('   ✓ Mode 2-1 setup correctly')

# Test 4: Setup mode 2-4
setup_quamba_mode('2-4', verbose=False)
assert os.environ.get('FLOAT_SIM_ASIC_INT8', 'false').lower() == 'true'
assert os.environ.get('CONV1D_MODE24_FP32', 'false').lower() == 'true'
print('   ✓ Mode 2-4 setup correctly')

# Test 5: Setup mode 3
setup_quamba_mode('3', verbose=False)
assert os.environ.get('CONV1D_MODE3_FP32', 'false').lower() == 'true'
print('   ✓ Mode 3 setup correctly')

print('')
print('   All mode configurations tested successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "All tests PASSED ✓"
    echo "============================================================================"
    echo ""
    echo "New usage examples:"
    echo "  python3 main.py quamba-130m-w8a8 --mode 0 --pretrained_dir ... --eval_zero_shot"
    echo "  python3 main.py quamba-130m-w8a8 --mode 2-1 --pretrained_dir ... --eval_zero_shot"
    echo "  ./QUICK_RUN.sh 2-4"
    echo "  ./QUICK_RUN.sh all"
    echo ""
else
    echo ""
    echo "============================================================================"
    echo "Tests FAILED ✗"
    echo "============================================================================"
    exit 1
fi
