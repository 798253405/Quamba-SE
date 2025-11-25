#!/bin/bash

# ============================================================================
# 快速验证量化确定性
# ============================================================================

# 设置CUBLAS确定性行为（关键！）
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "🔍 快速验证量化确定性..."
echo "已设置: CUBLAS_WORKSPACE_CONFIG=:4096:8"
echo ""

# Run 1
echo "▶ Run 1..."
QUICK_VERIFY=true python3 main.py \
  pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize --eval_zero_shot --task_list lambada_openai --testing \
  --mode 2-0 2>&1 > /tmp/verify_run1.log

# 提取关键数据
CONV1D_RUN1=$(grep "first 5 values \[0,0,:5\]:" /tmp/verify_run1.log | head -1 | awk -F': ' '{print $2}')
DT_RUN1=$(grep "dt first 5 values \[0,0,:5\]:" /tmp/verify_run1.log | awk -F': ' '{print $2}')

# Run 2
echo "▶ Run 2..."
QUICK_VERIFY=true python3 main.py \
  pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize --eval_zero_shot --task_list lambada_openai --testing \
  --mode 2-0 2>&1 > /tmp/verify_run2.log

# 提取关键数据
CONV1D_RUN2=$(grep "first 5 values \[0,0,:5\]:" /tmp/verify_run2.log | head -1 | awk -F': ' '{print $2}')
DT_RUN2=$(grep "dt first 5 values \[0,0,:5\]:" /tmp/verify_run2.log | awk -F': ' '{print $2}')

echo ""
echo "======================================================================"
echo "📊 结果对比"
echo "======================================================================"
echo "Conv1D Output [0,0,:5]:"
echo "  Run 1: $CONV1D_RUN1"
echo "  Run 2: $CONV1D_RUN2"
echo ""
echo "dt [0,0,:5]:"
echo "  Run 1: $DT_RUN1"
echo "  Run 2: $DT_RUN2"
echo ""

if [ "$CONV1D_RUN1" == "$CONV1D_RUN2" ] && [ "$DT_RUN1" == "$DT_RUN2" ]; then
    echo "✅ 验证通过！量化是完全确定性的"
    echo ""
    echo "现在可以放心地对比不同mode，随机性已消除！"
else
    echo "❌ 验证失败！仍存在随机性"
    echo ""
    echo "详细日志："
    echo "  Run 1: /tmp/verify_run1.log"
    echo "  Run 2: /tmp/verify_run2.log"
fi
echo "======================================================================"
