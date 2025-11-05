# Percentile对比实验脚本使用说明

## 📋 功能概述

`compare_percentile_effects.sh` 是一键对比脚本，用于测试不同percentile设置对Mamba模型量化效果的影响。

## 🚀 快速开始

```bash
# 直接运行脚本
./compare_percentile_effects.sh
```

**提示**：脚本会要求确认后再开始（输入 `y` 继续，`n` 取消）

---

## 📊 实验内容

脚本会自动运行4个实验：

| # | 模型 | Percentile设置 | 说明 |
|---|------|--------------|------|
| 1 | Mamba1-130M | 默认 (0.9995) | 裁剪0.05%极值 |
| 2 | Mamba1-130M | pa=1.0 | 无裁剪 |
| 3 | Mamba2-2.7B | 默认 (0.9995) | 裁剪0.05%极值 |
| 4 | Mamba2-2.7B | pa=1.0 | 无裁剪 |

**量化配置**：所有实验使用 W8A8 + GPTQ + lambada_openai评测

---

## ⏱️ 时间估算

- **Mamba1-130M**：每个实验 ~8-10分钟
- **Mamba2-2.7B**：每个实验 ~15-20分钟
- **总计**：~40-60分钟

---

## 💾 输出文件

### 1. 量化模型
```
pretrained_models/testPercentileRange/
├── pa-default/
│   ├── mamba-130m/          # 实验1
│   └── mamba2-2.7b/         # 实验3
└── pa-1.0/
    ├── mamba-130m/          # 实验2
    └── mamba2-2.7b/         # 实验4
```

### 2. 统计日志
```
percentileRangeResults/
├── experiments.jsonl        # 所有实验的元数据（追加模式）
├── activations_mamba-130m_default_*.npz   # 激活值快照
├── activations_mamba-130m_pa1.0_*.npz
├── activations_mamba2-2.7b_default_*.npz
└── activations_mamba2-2.7b_pa1.0_*.npz
```

### 3. 评测结果
```
logs/
├── mamba-130m_w8a8.json     # 详细评测结果
└── mamba2-2.7b_w8a8.json
```

---

## 📈 查看结果

### 方法1：使用view工具（推荐）
```bash
# 查看所有实验
python3 view_percentile_logs.py

# 查看最后4次实验
python3 view_percentile_logs.py --last 4

# 对比最后两次实验
python3 view_percentile_logs.py --compare -1 -2

# 对比指定实验（如第2和第4个）
python3 view_percentile_logs.py --compare 1 3
```

### 方法2：直接查看JSONL
```bash
# 查看所有实验（需要安装jq）
cat percentileRangeResults/experiments.jsonl | jq .

# 查看最后一个实验
tail -n 1 percentileRangeResults/experiments.jsonl | jq .

# 提取所有accuracy
cat percentileRangeResults/experiments.jsonl | jq '.results.accuracy'
```

### 方法3：分析激活值
```python
import numpy as np

# 加载激活值
data = np.load("percentileRangeResults/activations_mamba-130m_default_*.npz")

# 查看包含的层
print(data.files)
# ['layer_0.x_proj:input', 'layer_0.x_proj:output', ...]

# 分析某层的激活值
layer0_input = data['layer_0.x_proj:input']
print(f"Shape: {layer0_input.shape}")
print(f"Min: {layer0_input.min()}, Max: {layer0_input.max()}")
print(f"Mean: {layer0_input.mean()}, Std: {layer0_input.std()}")
```

---

## 🎯 预期发现

根据历史实验结果：

### Mamba1-130M
- **pa=1.0 vs 默认**：预期准确率提升 **+1-2%**
- **原因**：Mamba1对激活范围敏感，保留极值有助于保持精度

### Mamba2-2.7B
- **pa=1.0 vs 默认**：预期准确率差异 **<0.5%**
- **原因**：Mamba2的reorder机制已有效降低范围，percentile影响较小

---

## 🔧 自定义实验

如果需要测试其他配置，可以修改脚本中的参数：

```bash
# 例如：测试W4A8量化
python3 main.py ${PRETRAINED_DIR}/state-spaces/mamba-130m \
  --quantize \
  --w_bits 4 --a_bits 8 \     # 改为W4A8
  --percentile_alpha 0.999 \   # 自定义percentile
  --eval_zero_shot --task_list lambada_openai \
  --pretrained_dir ${PRETRAINED_DIR} \
  --log_dir ${LOG_DIR} \
  --output_subdir ${OUTPUT_SUBDIR}
```

**可调参数**：
- `--w_bits`：权重位宽 (4/8)
- `--a_bits`：激活位宽 (8/16)
- `--percentile_alpha`：percentile阈值 (0.99-1.0)
- `--task_list`：评测任务 (lambada_openai, arc_easy, winogrande等)
- `--calib_data_num`：校准样本数 (默认512)

---

## ❓ 常见问题

### Q1: 为什么accuracy有波动？
**A**: GPTQ量化过程有随机性（未固定seed），导致 ±1-2% 的波动。但percentile统计是确定的（seed=42）。

### Q2: 激活值文件太大怎么办？
**A**: 每个文件只保存前3层的前10000个值，压缩后约50KB，不会占用太多空间。如果不需要可以关闭：
```python
# main.py 第24行
plogger = reset_percentile_logger(
    log_file="percentileRangeResults/experiments.jsonl",
    save_activations=False  # 关闭激活值保存
)
```

### Q3: 如何复现某次实验？
**A**: 查看 `experiments.jsonl` 中的 `command` 字段，复制命令行即可：
```bash
# 提取命令行
cat percentileRangeResults/experiments.jsonl | jq -r '.command'
```

### Q4: 实验中断了怎么办？
**A**: 脚本使用 `set -e`，遇到错误会自动停止。可以注释掉已完成的实验，只运行剩余部分。

---

## 📚 相关文档

- `CALIBRATION_INFO.md` - Calibration机制和统计信息详解
- `PERCENTILE_LOGGING.md` - 日志系统详细说明
- `view_percentile_logs.py` - 日志查看工具使用方法

---

**最后更新**：2025-11-05
