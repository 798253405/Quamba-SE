#!/bin/bash
# 一键对比脚本：测试Mamba1-130M和Mamba2-2.7B在不同percentile下的表现
#
# 功能：
# 1. Mamba1-130M: 默认percentile vs pa=1.0
# 2. Mamba2-2.7B: 默认percentile vs pa=1.0
# 3. 所有结果保存到 percentileRangeResults/experiments.jsonl
# 4. 模型保存到 testPercentileRange/pa-*/

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Percentile效果对比实验${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "📋 实验计划:"
echo "  1. Mamba1-130M + 默认percentile (0.9995)"
echo "  2. Mamba1-130M + pa=1.0 (无裁剪)"
echo "  3. Mamba2-2.7B + 默认percentile (0.9995)"
echo "  4. Mamba2-2.7B + pa=1.0 (无裁剪)"
echo ""
echo "💾 输出位置:"
echo "  - 量化模型: pretrained_models/testPercentileRange/pa-*/"
echo "  - 统计日志: percentileRangeResults/experiments.jsonl"
echo "  - 激活值: percentileRangeResults/activations_*.npz"
echo ""
echo "⏱️  预计总时间: ~40-60分钟"
echo ""

# 询问是否继续
read -p "是否开始实验? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "已取消"
    exit 1
fi

# 基础配置
PRETRAINED_DIR="./pretrained_models"
LOG_DIR="logs"
OUTPUT_SUBDIR="testPercentileRange"

# ==============================================
# 实验1: Mamba1-130M + 默认percentile
# ==============================================
echo ""
echo -e "${GREEN}[1/4] Mamba1-130M + 默认percentile (0.9995)${NC}"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python3 main.py ${PRETRAINED_DIR}/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize \
  --w_bits 8 --a_bits 8 \
  --eval_zero_shot --task_list lambada_openai \
  --pretrained_dir ${PRETRAINED_DIR} \
  --log_dir ${LOG_DIR} \
  --output_subdir ${OUTPUT_SUBDIR}

echo -e "${GREEN}✅ 完成 [1/4]${NC}"
echo ""

# ==============================================
# 实验2: Mamba1-130M + pa=1.0
# ==============================================
echo ""
echo -e "${GREEN}[2/4] Mamba1-130M + pa=1.0 (无裁剪)${NC}"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python3 main.py ${PRETRAINED_DIR}/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize \
  --w_bits 8 --a_bits 8 \
  --percentile_alpha 1.0 \
  --eval_zero_shot --task_list lambada_openai \
  --pretrained_dir ${PRETRAINED_DIR} \
  --log_dir ${LOG_DIR} \
  --output_subdir ${OUTPUT_SUBDIR}

echo -e "${GREEN}✅ 完成 [2/4]${NC}"
echo ""

# ==============================================
# 实验3: Mamba2-2.7B + 默认percentile
# ==============================================
echo ""
echo -e "${GREEN}[3/4] Mamba2-2.7B + 默认percentile (0.9995)${NC}"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python3 main.py ${PRETRAINED_DIR}/mambaOriginalHuggingfaceDownload/mamba2-2.7b \
  --quantize \
  --w_bits 8 --a_bits 8 \
  --eval_zero_shot --task_list lambada_openai \
  --pretrained_dir ${PRETRAINED_DIR} \
  --log_dir ${LOG_DIR} \
  --output_subdir ${OUTPUT_SUBDIR}

echo -e "${GREEN}✅ 完成 [3/4]${NC}"
echo ""

# ==============================================
# 实验4: Mamba2-2.7B + pa=1.0
# ==============================================
echo ""
echo -e "${GREEN}[4/4] Mamba2-2.7B + pa=1.0 (无裁剪)${NC}"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

python3 main.py ${PRETRAINED_DIR}/mambaOriginalHuggingfaceDownload/mamba2-2.7b \
  --quantize \
  --w_bits 8 --a_bits 8 \
  --percentile_alpha 1.0 \
  --eval_zero_shot --task_list lambada_openai \
  --pretrained_dir ${PRETRAINED_DIR} \
  --log_dir ${LOG_DIR} \
  --output_subdir ${OUTPUT_SUBDIR}

echo -e "${GREEN}✅ 完成 [4/4]${NC}"
echo ""

# ==============================================
# 实验完成总结
# ==============================================
echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  🎉 所有实验完成！${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 查看结果:"
echo "  1. 查看所有实验记录:"
echo "     python3 view_percentile_logs.py"
echo ""
echo "  2. 对比最后两次实验:"
echo "     python3 view_percentile_logs.py --compare -1 -2"
echo ""
echo "  3. 查看JSON日志:"
echo "     cat percentileRangeResults/experiments.jsonl | jq ."
echo ""
echo "  4. 查看激活值文件:"
echo "     ls -lh percentileRangeResults/activations_*.npz"
echo ""
