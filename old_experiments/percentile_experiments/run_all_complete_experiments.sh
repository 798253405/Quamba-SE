#!/bin/bash

# Quamba 完整实验脚本 - 方案B (默认 + percentile_alpha=1.0 全对比)
# 包含所有模型（Mamba1 + Mamba2），每个做两次（默认和pa=1.0）
# 总共 24 个实验
# 作者：YZ
# 日期：2025-11-04

set -e  # 遇到错误立即停止

LOG_FILE="run_complete_experiments_$(date +%Y%m%d_%H%M%S).log"
exec 2>&1 | tee -a "$LOG_FILE"

echo "========================================================================"
echo "🚀 开始运行所有完整实验（默认 + percentile_alpha=1.0）"
echo "========================================================================"
echo "总实验数: 24 个"
echo "预计总时间: 8-10 小时"
echo "开始时间: $(date)"
echo "日志文件: $LOG_FILE"
echo ""

# ============================================================================
# Mamba1 系列实验 (W8A8) - 8个实验
# ============================================================================

echo "========================================================================"
echo "📊 Mamba1 130M W8A8 - 默认 (1/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize \
  --apply_gptq \
  --quantize_embedding \
  --quantize_lm_head \
  --w_bits 8 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba1 130M W8A8 - pa=1.0 (2/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba1 370M W8A8 - 默认 (3/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-370m \
  --quantize \
  --apply_gptq \
  --quantize_embedding \
  --quantize_lm_head \
  --w_bits 8 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba1 370M W8A8 - pa=1.0 (4/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-370m \
  --quantize \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba1 1.4B W8A8 - 默认 (5/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-1.4b \
  --quantize \
  --apply_gptq \
  --quantize_embedding \
  --quantize_lm_head \
  --w_bits 8 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba1 1.4B W8A8 - pa=1.0 (6/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-1.4b \
  --quantize \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba1 2.8B W8A8 - 默认 (7/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-2.8b \
  --quantize \
  --apply_gptq \
  --quantize_embedding \
  --quantize_lm_head \
  --w_bits 8 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba1 2.8B W8A8 - pa=1.0 (8/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-2.8b \
  --quantize \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

# ============================================================================
# Mamba2 130M 系列实验 - 4个实验
# ============================================================================

echo "========================================================================"
echo "📊 Mamba2 130M W4A8 - 默认 (9/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/ut-enyac/pretrained_models/mamba2-130m \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 130M W4A8 - pa=1.0 (10/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/ut-enyac/pretrained_models/mamba2-130m \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 130M W8A8 - 默认 (11/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/ut-enyac/pretrained_models/mamba2-130m \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 130M W8A8 - pa=1.0 (12/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/ut-enyac/pretrained_models/mamba2-130m \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

# ============================================================================
# Mamba2 2.7B 系列实验 - 6个实验
# ============================================================================

echo "========================================================================"
echo "📊 Mamba2 2.7B W4A8 - 默认 (13/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py state-spaces/mamba2-2.7b \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 2.7B W4A8 - pa=1.0 (14/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py state-spaces/mamba2-2.7b \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 2.7B W8A8 - 默认 (15/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py state-spaces/mamba2-2.7b \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 2.7B W8A8 - pa=1.0 (16/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py state-spaces/mamba2-2.7b \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 2.7B W4A16 - 默认 (17/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py state-spaces/mamba2-2.7b \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 2.7B W4A16 - pa=1.0 (18/24)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py state-spaces/mamba2-2.7b \
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

# ============================================================================
# Mamba2 8B 系列实验 - 6个实验
# ============================================================================

echo "========================================================================"
echo "📊 Mamba2 8B W4A8 - 默认 (19/24)"
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 8B W4A8 - pa=1.0 (20/24)"
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 8B W8A8 - 默认 (21/24)"
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 8B W8A8 - pa=1.0 (22/24)"
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 8B W4A16 - 默认 (23/24)"
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Mamba2 8B W4A16 - pa=1.0 (24/24)"
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
  --log_dir logs
echo "✅ 完成时间: $(date)"
echo ""

# ============================================================================
# 完成
# ============================================================================

echo "========================================================================"
echo "🎉 所有 24 个实验完成！"
echo "========================================================================"
echo "结束时间: $(date)"
echo "日志文件: $LOG_FILE"
echo ""
echo "📊 查看结果："
echo "  - 日志目录: logs/"
echo "  - 模型目录: pretrained_models/yzReproduceauthors/"
echo ""
echo "实验完成统计："
echo "  Mamba1 系列: 8 个实验 (4个模型 × 2配置)"
echo "  Mamba2 130M: 4 个实验 (2个量化 × 2配置)"
echo "  Mamba2 2.7B: 6 个实验 (3个量化 × 2配置)"
echo "  Mamba2 8B:   6 个实验 (3个量化 × 2配置)"
echo "  总计: 24 个实验"
echo "========================================================================"
