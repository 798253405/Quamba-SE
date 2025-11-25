#!/bin/bash

# Quamba 正确实验脚本 - 根据作者回复修正
# Quamba1: Mamba1 W8A8 (不加 embedding/lm_head/gptq)
# Quamba2: Mamba2 W4A8/W4A16/W8A8 (加所有参数)
# 每个配置做 default + pa=1.0 对比
# 总共 20 个实验
# 作者：YZ
# 日期：2025-11-04

set -e  # 遇到错误立即停止

LOG_FILE="run_correct_experiments_$(date +%Y%m%d_%H%M%S).log"
exec 2>&1 | tee -a "$LOG_FILE"

echo "========================================================================"
echo "🚀 开始运行正确的 Quamba1 + Quamba2 实验"
echo "========================================================================"
echo "Quamba1: Mamba1 W8A8 (无额外参数) - 8个实验"
echo "Quamba2: Mamba2 W4/W8 (有额外参数) - 12个实验"
echo "总实验数: 20 个"
echo "预计总时间: 6-8 小时"
echo "开始时间: $(date)"
echo "日志文件: $LOG_FILE"
echo ""

# ============================================================================
# Quamba1 部分：Mamba1 W8A8 (不加 embedding/lm_head/gptq)
# 保存路径：pretrained_models/quamba1/default/ 和 quamba1/pa-1/
# ============================================================================

echo "========================================================================"
echo "🔵 Quamba1 实验开始"
echo "========================================================================"
echo ""

echo "========================================================================"
echo "📊 Quamba1: Mamba1 130M W8A8 - 默认 (1/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize \
  --w_bits 8 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba1
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba1: Mamba1 130M W8A8 - pa=1.0 (2/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize \
  --w_bits 8 \
  --a_bits 8 \
  --percentile_alpha 1.0 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba1
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba1: Mamba1 370M W8A8 - 默认 (3/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-370m \
  --quantize \
  --w_bits 8 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba1
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba1: Mamba1 370M W8A8 - pa=1.0 (4/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-370m \
  --quantize \
  --w_bits 8 \
  --a_bits 8 \
  --percentile_alpha 1.0 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba1
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba1: Mamba1 1.4B W8A8 - 默认 (5/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-1.4b \
  --quantize \
  --w_bits 8 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba1
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba1: Mamba1 1.4B W8A8 - pa=1.0 (6/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-1.4b \
  --quantize \
  --w_bits 8 \
  --a_bits 8 \
  --percentile_alpha 1.0 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba1
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba1: Mamba1 2.8B W8A8 - 默认 (7/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-2.8b \
  --quantize \
  --w_bits 8 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba1
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba1: Mamba1 2.8B W8A8 - pa=1.0 (8/20)"
echo "========================================================================"
echo "开始时间: $(date)"
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-2.8b \
  --quantize \
  --w_bits 8 \
  --a_bits 8 \
  --percentile_alpha 1.0 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba1
echo "✅ 完成时间: $(date)"
echo ""

# ============================================================================
# Quamba2 部分：Mamba2 W4A8/W4A16/W8A8 (加 group_heads/gptq/embedding/lm_head)
# 保存路径：pretrained_models/quamba2/default/ 和 quamba2/pa-1/
# ============================================================================

echo "========================================================================"
echo "🟢 Quamba2 实验开始"
echo "========================================================================"
echo ""

# ---------- Mamba2 2.7B ----------

echo "========================================================================"
echo "📊 Quamba2: Mamba2 2.7B W4A8 - 默认 (9/20)"
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
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba2: Mamba2 2.7B W4A8 - pa=1.0 (10/20)"
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
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba2: Mamba2 2.7B W4A16 - 默认 (11/20)"
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
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba2: Mamba2 2.7B W4A16 - pa=1.0 (12/20)"
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
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba2: Mamba2 2.7B W8A8 - 默认 (13/20)"
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
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

echo "========================================================================"
echo "📊 Quamba2: Mamba2 2.7B W8A8 - pa=1.0 (14/20)"
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
  --log_dir logs \
  --output_subdir quamba2
echo "✅ 完成时间: $(date)"
echo ""

# ---------- Mamba2 8B ----------

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
echo "🎉 所有 20 个实验完成！"
echo "========================================================================"
echo "结束时间: $(date)"
echo "日志文件: $LOG_FILE"
echo ""
echo "📊 查看结果："
echo "  - Quamba1 模型: pretrained_models/quamba1/"
echo "  - Quamba2 模型: pretrained_models/quamba2/"
echo "  - 日志目录: logs/"
echo ""
echo "实验完成统计："
echo "  🔵 Quamba1 (Mamba1 W8A8, 无额外参数): 8 个实验"
echo "  🟢 Quamba2 (Mamba2 W4/W8, 有额外参数): 12 个实验"
echo "  总计: 20 个实验"
echo ""
echo "文件夹结构："
echo "  pretrained_models/quamba1/default/    - Quamba1 默认配置"
echo "  pretrained_models/quamba1/pa-1/       - Quamba1 pa=1.0"
echo "  pretrained_models/quamba2/default/    - Quamba2 默认配置"
echo "  pretrained_models/quamba2/pa-1/       - Quamba2 pa=1.0"
echo "========================================================================"
