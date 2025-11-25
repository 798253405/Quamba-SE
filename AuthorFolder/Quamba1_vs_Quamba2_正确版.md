# Quamba1 vs Quamba2 正确对比（根据作者回复）

## 📧 作者关键回复

> If you'd like to reproduce the **Quamba1** results, please set the quantization bit-width to **W8A8** and quantize the Mamba1 models **without** --quantize_embedding, --quantize_lm_head, and --apply_gptq flags.

---

## 🎯 Quamba1 vs Quamba2 核心区别

| 对比项 | Quamba1 | Quamba2 |
|--------|---------|---------|
| **论文章节** | Table 9 | 主要结果 |
| **适用模型** | **Mamba1** | **Mamba2** |
| **量化精度** | **W8A8** | W4A8, W4A16, W8A8 |
| **--quantize** | ✅ | ✅ |
| **--group_heads** | ❌ | ✅ |
| **--apply_gptq** | ❌ | ✅ |
| **--quantize_embedding** | ❌ | ✅ |
| **--quantize_lm_head** | ❌ | ✅ |
| **保存路径** | `quamba1/` | `quamba2/` |
| **默认 percentile** | 0.9995 | 0.99999 |

---

## 📋 正确命令对比

### Quamba1 命令（Mamba1 W8A8）

```bash
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
  # 只有这些！不加其他！
```

**关键特点**:
- ❌ **没有** `--group_heads`
- ❌ **没有** `--apply_gptq`
- ❌ **没有** `--quantize_embedding`
- ❌ **没有** `--quantize_lm_head`
- ✅ **保存到** `quamba1/` 文件夹

---

### Quamba2 命令（Mamba2 W4A8）

```bash
python3 main.py state-spaces/mamba2-2.7b \
  --quantize \
  --group_heads \           # ✅ 加这个
  --apply_gptq \            # ✅ 加这个
  --quantize_embedding \    # ✅ 加这个
  --quantize_lm_head \      # ✅ 加这个
  --w_bits 4 \
  --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir quamba2
```

**关键特点**:
- ✅ **有** `--group_heads`
- ✅ **有** `--apply_gptq`
- ✅ **有** `--quantize_embedding`
- ✅ **有** `--quantize_lm_head`
- ✅ **保存到** `quamba2/` 文件夹

---

## 📊 实验矩阵

### Quamba1 实验（8个）

| 模型 | 量化 | percentile_alpha | Paper 结果 |
|------|------|-----------------|-----------|
| Mamba1 130M | W8A8 | 默认 (0.9995) | 40.61% |
| Mamba1 130M | W8A8 | **1.0** | - |
| Mamba1 370M | W8A8 | 默认 (0.9995) | 50.37% |
| Mamba1 370M | W8A8 | **1.0** | - |
| Mamba1 1.4B | W8A8 | 默认 (0.9995) | 60.43% |
| Mamba1 1.4B | W8A8 | **1.0** | - |
| Mamba1 2.8B | W8A8 | 默认 (0.9995) | 65.67% |
| Mamba1 2.8B | W8A8 | **1.0** | - |

---

### Quamba2 实验（12个）

| 模型 | 量化 | percentile_alpha | Paper 结果 |
|------|------|-----------------|-----------|
| Mamba2 2.7B | W4A8 | 默认 (0.99999) | 65.80% |
| Mamba2 2.7B | W4A8 | **1.0** | - |
| Mamba2 2.7B | W4A16 | 默认 (0.99999) | 67.50% |
| Mamba2 2.7B | W4A16 | **1.0** | - |
| Mamba2 2.7B | W8A8 | 默认 (0.99999) | 68.20% |
| Mamba2 2.7B | W8A8 | **1.0** | - |
| Mamba2 8B | W4A8 | 默认 (0.99999) | 69.50% |
| Mamba2 8B | W4A8 | **1.0** | - |
| Mamba2 8B | W4A16 | 默认 (0.99999) | 71.20% |
| Mamba2 8B | W4A16 | **1.0** | - |
| Mamba2 8B | W8A8 | 默认 (0.99999) | 72.10% |
| Mamba2 8B | W8A8 | **1.0** | - |

**总计**: 20 个实验

---

## 📂 文件夹结构

```
pretrained_models/
├── quamba1/
│   ├── default/              # Quamba1 默认配置
│   │   ├── quamba-130m-w8a8/
│   │   ├── quamba-370m-w8a8/
│   │   ├── quamba-1.4b-w8a8/
│   │   └── quamba-2.8b-w8a8/
│   └── pa-1/                 # Quamba1 pa=1.0
│       └── (同上4个模型)
│
└── quamba2/
    ├── default/              # Quamba2 默认配置
    │   ├── quamba2-2.7b-w4a8/
    │   ├── quamba2-2.7b-w4a16/
    │   ├── quamba2-2.7b-w8a8/
    │   ├── quamba2-8b-converted-w4a8/
    │   ├── quamba2-8b-converted-w4a16/
    │   └── quamba2-8b-converted-w8a8/
    └── pa-1/                 # Quamba2 pa=1.0
        └── (同上6个模型)
```

---

## ✅ 脚本验证

### 脚本名称
`run_correct_experiments.sh`

### 验证要点

**Quamba1 部分**:
```bash
grep -A 10 "Quamba1: Mamba1 370M W8A8 - 默认" run_correct_experiments.sh
```

应该看到：
```bash
python3 main.py .../mamba-370m \
  --quantize \
  --w_bits 8 \
  --a_bits 8 \
  ... (没有 gptq/embedding/lm_head)
  --output_subdir quamba1  # ← 关键！
```

**Quamba2 部分**:
```bash
grep -A 15 "Quamba2: Mamba2 2.7B W4A8 - 默认" run_correct_experiments.sh
```

应该看到：
```bash
python3 main.py state-spaces/mamba2-2.7b \
  --quantize \
  --group_heads \           # ← 有这个
  --apply_gptq \            # ← 有这个
  --quantize_embedding \    # ← 有这个
  --quantize_lm_head \      # ← 有这个
  --w_bits 4 \
  --a_bits 8 \
  ...
  --output_subdir quamba2  # ← 关键！
```

---

## 🚀 启动命令

```bash
cd /home/yz/myprojects/2025/logquamba/Quamba
nohup ./run_correct_experiments.sh > experiments_correct.log 2>&1 &
```

---

## 🎯 预期成果

### Quamba1 (Table 9 复现)
- ✅ 验证论文 Table 9 的 Mamba1 W8A8 结果
- ✅ 研究 percentile_alpha 对 Quamba1 的影响

### Quamba2 (主要结果复现)
- ✅ 验证论文主要的 Mamba2 量化结果
- ✅ 对比 W4A8 vs W4A16 vs W8A8
- ✅ 研究 percentile_alpha 对 Quamba2 的影响

### 方法对比
- ✅ 理解 embedding/lm_head/gptq 对准确率的贡献
- ✅ 分析 Quamba1 和 Quamba2 的技术差异

---

**总结**: 现在的脚本完全按照作者的指导修正，确保正确复现论文结果！
