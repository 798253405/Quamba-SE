# Quamba1 vs Quamba2 完全解析

## 🔍 核心发现（代码分析）

### 关键代码位置：`quamba/modelutils_mamba.py:814-832`

```python
if args.a_bits == 8:
    if args.group_heads:  # ← 这是区分 Quamba1 和 Quamba2 的关键！
        # ✅ 这是 Quamba2 方法
        logging.info(f"Reordering weights and activations for head grouping")
        reorder_params = get_reorder_params(...)  # 步骤1：聚类分析
        reorder_mamba(model, reorder_params)      # 步骤2：重排序权重
        act_scales = run_quamba2_calibration(...) # 步骤3：Quamba2 calibration
    else:
        # ✅ 这是 Quamba1 方法
        act_scales = run_quamba_calibration(...)  # 直接 calibration，不 reorder
```

---

## 📊 Quamba1 vs Quamba2 完整对比

| 对比项 | Quamba1 | Quamba2 |
|--------|---------|---------|
| **命令参数** | 不加 `--group_heads` | 加 `--group_heads` |
| **适用模型** | Mamba1 和 Mamba2 都可以 | 只用于 Mamba2 |
| **重排序** | ❌ 不做 reorder | ✅ 做 reorder（聚类） |
| **Calibration** | `run_quamba_calibration` | `run_quamba2_calibration` |
| **默认 percentile_alpha** | 0.9995 | 0.99999 |
| **Observer** | PerTensorPercentileObserver | CrossHeadMinmaxObserver |
| **论文列名** | "quamba1" | "quamba2" |

---

## 🧠 技术原理

### Quamba1 方法（基础量化）

```
FP16模型 → 收集激活统计 → 计算scale → 量化
           ↓
       所有层独立处理
```

**特点**：
- 简单直接
- 每层独立量化
- 不考虑层间或头间的相关性

### Quamba2 方法（高级量化 with reorder）

```
FP16模型 → 聚类分析 → 重排序 → 收集激活统计 → 量化
           ↓           ↓
       相似的头分组   相似头放一起
```

**特点**：
1. **聚类（Clustering）**：使用 AgglomerativeClustering + KMeans
2. **重排序（Reorder）**：将相似的头重新排列在一起
3. **跨头量化（Cross-Head）**：相似的头共享量化参数

**为什么 Quamba2 更好？**
- 相似的头使用相同的量化参数 → 减少量化误差
- 特别适合 Mamba2 的多头架构

---

## 🎯 我的脚本使用的是什么方法？

### Mamba1 系列（370M, 1.4B, 2.8B）

```bash
# 我的脚本（正确）
python3 main.py [mamba1-model] \
  --quantize \
  # ❌ 没有 --group_heads \        ← 使用 Quamba1 方法
  --apply_gptq \
  --w_bits 8 --a_bits 8 ...
```

**结果**：✅ 使用 **Quamba1 方法**（不 reorder）

**为什么正确？**
- Mamba1 是单头架构，不支持多头分组
- Quamba2 的 reorder 对 Mamba1 没有意义
- 论文的 "quamba1" 列就是用这个方法

---

### Mamba2 系列（2.7B, 8B）

```bash
# 我的脚本（正确）
python3 main.py [mamba2-model] \
  --quantize \
  --group_heads \                  ← 使用 Quamba2 方法
  --apply_gptq \
  --w_bits 8 --a_bits 8 ...
```

**结果**：✅ 使用 **Quamba2 方法**（with reorder）

**为什么正确？**
- Mamba2 是多头架构，支持头分组
- Quamba2 的 reorder 能显著提升精度
- 论文的 "quamba2" 列就是用这个方法

---

## 📋 你的表格解读

```
模型系列 | 模型大小 | baselin | WHT | quamba1 | quamba2-nopercentile | quamba2 | We reproduced
---------|---------|---------|-----|---------|---------------------|---------|---------------
Mamba1   | 130M    | 34.10%  | ... | 40.61%  | N/A                 | N/A     | 40.02% ✅
Mamba1   | 370M    | 45.78%  | ... | 50.37%  | N/A                 | N/A     | ？？% 🔄
Mamba2   | 2.7B    | N/A     | N/A | N/A     | ...                 | 68.20%  | ？？% 🔄
Mamba2   | 8B      | N/A     | N/A | N/A     | ...                 | 72.10%  | 69.03% ✅
```

**解读**：
- **Mamba1 模型** → 应该出现在 "quamba1" 列（论文的表9）
- **Mamba2 模型** → 应该出现在 "quamba2" 列
- **"We reproduced"** → 我们复现的结果，应该尽量接近 paper 报告

---

## ⚠️ 重要澄清

### 1. Mamba1/Mamba2 ≠ Quamba1/Quamba2

- **Mamba1/Mamba2**：模型架构（原始 FP16 模型）
  - Mamba1 (2023)：单头 SSM
  - Mamba2 (2024)：多头 SSM

- **Quamba1/Quamba2**：量化方法（论文提出的算法）
  - Quamba1：基础量化（无 reorder）
  - Quamba2：高级量化（有 reorder + head grouping）

### 2. 可以混用吗？

理论上可以，但不推荐：

| 模型 | 方法 | 结果 | 推荐 |
|------|------|------|------|
| Mamba1 | Quamba1 | ✅ 正常工作 | ✅ 推荐 |
| Mamba1 | Quamba2 | ⚠️ 可能工作，但无优势 | ❌ 不推荐 |
| Mamba2 | Quamba1 | ✅ 正常工作 | ⚠️ 可用但不如Quamba2 |
| Mamba2 | Quamba2 | ✅ 最佳效果 | ✅ 强烈推荐 |

---

## ✅ 我的脚本验证

### 对于 Mamba1：

```bash
# 脚本中的命令（正确）
python3 main.py pretrained_models/.../mamba-370m \
  --quantize --apply_gptq \
  # ❌ 没有 --group_heads
  --w_bits 8 --a_bits 8 ...
```

**代码执行路径**：
```python
if args.a_bits == 8:
    if args.group_heads:  # False，跳过这个分支
        ...
    else:
        act_scales = run_quamba_calibration(...)  # ✅ 执行这里！
```

**结果**：✅ 使用 Quamba1 方法（正确）

---

### 对于 Mamba2：

```bash
# 脚本中的命令（正确）
python3 main.py state-spaces/mamba2-2.7b \
  --quantize --group_heads \  # ✅ 有这个参数
  --apply_gptq \
  --w_bits 8 --a_bits 8 ...
```

**代码执行路径**：
```python
if args.a_bits == 8:
    if args.group_heads:  # True，执行这个分支
        reorder_params = get_reorder_params(...)
        reorder_mamba(model, reorder_params)
        act_scales = run_quamba2_calibration(...)  # ✅ 执行这里！
```

**结果**：✅ 使用 Quamba2 方法（正确）

---

## 🎉 最终结论

### 你的脚本完全正确！

| 实验 | 模型架构 | 量化方法 | 命令 | 正确性 |
|------|---------|---------|------|--------|
| Mamba1 370M W8A8 | Mamba1 | Quamba1 | 无 `--group_heads` | ✅ 正确 |
| Mamba1 1.4B W8A8 | Mamba1 | Quamba1 | 无 `--group_heads` | ✅ 正确 |
| Mamba1 2.8B W8A8 | Mamba1 | Quamba1 | 无 `--group_heads` | ✅ 正确 |
| Mamba2 2.7B W8A8 | Mamba2 | Quamba2 | 有 `--group_heads` | ✅ 正确 |
| Mamba2 2.7B W4A16 | Mamba2 | W4A16特殊 | 有 `--group_heads` | ✅ 正确 |
| Mamba2 8B W4A16 | Mamba2 | W4A16特殊 | 有 `--group_heads` | ✅ 正确 |

### 关键理解

1. **`--group_heads` 是区分 Quamba1 和 Quamba2 的关键参数**
2. **Mamba1 用 Quamba1 方法**（不加 `--group_heads`）
3. **Mamba2 用 Quamba2 方法**（加 `--group_heads`）
4. **你的脚本完美匹配了这个逻辑**

---

## 🚀 可以放心运行！

```bash
cd /home/yz/myprojects/2025/logquamba/Quamba
./run_all_missing_experiments.sh
```

**预计时间**: 2.5-3小时

**输出位置**:
- 模型: `pretrained_models/yzReproduceauthors/`
- 日志: `logs/*.json`
- 旧数据备份: `pretrained_models/yzreproduceSAFE/` ✅

---

## 📚 参考

### 代码位置
- Quamba1 calibration: `quamba/modelutils_mamba.py:112-236`
- Quamba2 calibration: `quamba/modelutils_mamba.py:237-363`
- 方法选择逻辑: `quamba/modelutils_mamba.py:814-832`
- Reorder实现: `quamba/reorder_utils.py`

### 论文
- Quamba: Efficient State Space Language Modeling on Low-Bit Mamba
- Table 9: Mamba1 系列用 Quamba1 方法
- Mamba2 部分: Mamba2 系列用 Quamba2 方法

---

**总结**: 你的担心是多余的，脚本的逻辑完全正确！可以直接运行！
