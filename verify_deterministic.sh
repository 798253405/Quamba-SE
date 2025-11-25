#!/bin/bash

# ============================================================================
# 验证量化可重复性脚本
# ============================================================================

echo "🔍 验证量化的确定性（快速模式）"
echo "只运行到Layer 24，对比Conv1D和SSM输入数据"
echo ""

# 创建输出目录
mkdir -p debug_output/verify

echo "======================================================================"
echo "Run 1: Mode 2-0"
echo "======================================================================"
QUICK_VERIFY=true python3 main.py \
  pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize --eval_zero_shot --task_list lambada_openai --testing \
  --mode 2-0 2>&1 | tee debug_output/verify/run1.log

echo ""
echo "======================================================================"
echo "Run 2: Mode 2-0 (应该和Run 1完全相同)"
echo "======================================================================"
QUICK_VERIFY=true python3 main.py \
  pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize --eval_zero_shot --task_list lambada_openai --testing \
  --mode 2-0 2>&1 | tee debug_output/verify/run2.log

echo ""
echo "======================================================================"
echo "📊 对比结果"
echo "======================================================================"

echo ""
echo "🔹 Conv1D Output (前5个值):"
grep "first 5 values \[0,0,:5\]" debug_output/verify/run1.log | grep -A 1 "Conv1D Output"
grep "first 5 values \[0,0,:5\]" debug_output/verify/run2.log | grep -A 1 "Conv1D Output"

echo ""
echo "🔹 Conv1D Scales:"
grep "output_scale" debug_output/verify/run1.log | grep "0.0" | head -1
grep "output_scale" debug_output/verify/run2.log | grep "0.0" | head -1

echo ""
echo "🔹 SSM Input (u前5个值):"
grep "u first 5 values \[0,0,:5\]" debug_output/verify/run1.log
grep "u first 5 values \[0,0,:5\]" debug_output/verify/run2.log

echo ""
echo "🔹 dt前5个值:"
grep "dt first 5 values \[0,0,:5\]" debug_output/verify/run1.log
grep "dt first 5 values \[0,0,:5\]" debug_output/verify/run2.log

echo ""
echo "🔹 B前5个值:"
grep "B first 5 values \[0,0,:5\]" debug_output/verify/run1.log
grep "B first 5 values \[0,0,:5\]" debug_output/verify/run2.log

echo ""
echo "🔹 C前5个值:"
grep "C first 5 values \[0,0,:5\]" debug_output/verify/run1.log
grep "C first 5 values \[0,0,:5\]" debug_output/verify/run2.log

echo ""
echo "======================================================================"
echo "✅ 如果以上所有值都相同，说明量化是完全确定性的！"
echo "❌ 如果有任何值不同，说明还存在随机性"
echo "======================================================================"
