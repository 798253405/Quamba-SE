#!/bin/bash

# Quamba2 - Mamba2 8B 系列实验（实验15-20）
# W4A8, W4A16, W8A8 各做 default + pa=1.0
# 总共 6 个实验
# 作者：YZ
# 日期：2025-11-04

set -e  # 遇到错误立即停止

LOG_FILE="run_mamba2_8b_$(date +%Y%m%d_%H%M%S).log"
exec 2>&1 | tee -a "$LOG_FILE"

echo "========================================================================"
echo "🚀 开始运行 Mamba2 8B 实验（实验15-20）"
echo "========================================================================"
echo "总实验数: 6 个"
echo "预计总时间: 3-3.5 小时"
echo "开始时间: $(date)"
echo "日志文件: $LOG_FILE"
echo ""

# ============================================================================
# Mamba2 8B W4A8
# ============================================================================

echo "========================================================================"
echo "📊 Quamba2: Mamba2 8B W4A8 - 默认 (15/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba2-8b-converted \
  --quantize \
  --group_heads \
  --apply_gptq \
  --quantize_embedding \
  --quantize_lm_head \
  --w_bits 4 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba2: Mamba2 8B W4A8 - pa=1.0 (16/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba2-8b-converted \
  --quantize \
  --group_heads \
  --apply_gptq \
  --quantize_embedding \
  --quantize_lm_head \
  --w_bits 4 \
  --a_bits 8 \
  --percentile_alpha 1.0 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

# ============================================================================
# Mamba2 8B W4A16
# ============================================================================

echo "========================================================================"
echo "📊 Quamba2: Mamba2 8B W4A16 - 默认 (17/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba2-8b-converted \
  --quantize \
  --group_heads \
  --apply_gptq \
  --quantize_embedding \
  --quantize_lm_head \
  --w_bits 4 \
  --a_bits 16 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba2: Mamba2 8B W4A16 - pa=1.0 (18/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba2-8b-converted \
  --quantize \
  --group_heads \
  --apply_gptq \
  --quantize_embedding \
  --quantize_lm_head \
  --w_bits 4 \
  --a_bits 16 \
  --percentile_alpha 1.0 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

# ============================================================================
# Mamba2 8B W8A8
# ============================================================================

echo "========================================================================"
echo "📊 Quamba2: Mamba2 8B W8A8 - 默认 (19/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba2-8b-converted \
  --quantize \
  --group_heads \
  --apply_gptq \
  --quantize_embedding \
  --quantize_lm_head \
  --w_bits 8 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba2: Mamba2 8B W8A8 - pa=1.0 (20/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba2-8b-converted \
  --quantize \
  --group_heads \
  --apply_gptq \
  --quantize_embedding \
  --quantize_lm_head \
  --w_bits 8 \
  --a_bits 8 \
  --percentile_alpha 1.0 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

# ============================================================================
# 完成
# ============================================================================

echo "========================================================================"
echo "🎉 Mamba2 8B 所有 6 个实验完成！"
echo "========================================================================"
echo "结束时间: $(date)"
echo "日志文件: $LOG_FILE"
echo ""
echo "📊 查看结果："
echo "  - 模型: pretrained_models/quamba2/"
echo "  - 日志: logs/"
echo ""
echo "实验完成统计："
echo "  🟢 Mamba2 8B W4A8:  2 个（默认 + pa=1.0）"
echo "  🟢 Mamba2 8B W4A16: 2 个（默认 + pa=1.0）"
echo "  🟢 Mamba2 8B W8A8:  2 个（默认 + pa=1.0）"
echo "  总计: 6 个实验"
echo ""
echo "文件夹结构："
echo "  pretrained_models/quamba2/default/"
echo "    ├── quamba2-8b-converted-w4a8/"
echo "    ├── quamba2-8b-converted-w4a16/"
echo "    └── quamba2-8b-converted-w8a8/"
echo "  pretrained_models/quamba2/pa-1/"
echo "    ├── quamba2-8b-converted-w4a8/"
echo "    ├── quamba2-8b-converted-w4a16/"
echo "    └── quamba2-8b-converted-w8a8/"
echo "========================================================================"
