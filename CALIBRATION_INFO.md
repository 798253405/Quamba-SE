# Calibration 关键信息总结

## ⚠️ 核心要点

### Calibration统计信息只在运行时存在

**问题**：已有的量化模型无法直接获取percentile统计信息

**原因**：
```
量化流程：
1. 加载FP16模型
2. Calibration (512样本) ← 统计信息在这里产生！
   ├─ Observer收集激活值
   ├─ 计算min/max/percentile
   └─ 记录before/after范围
3. 计算scale/zero_point
4. 应用GPTQ量化
5. 保存量化模型 ← ❌ 统计信息丢失！
```

**保存的内容**：
- ✅ 量化后的权重 (INT4/INT8)
- ✅ scale和zero_point
- ❌ **没有**percentile裁剪前的min/max
- ❌ **没有**原始激活值分布

---

## 🚀 快速开始：一键对比脚本

### 对比不同percentile的效果（推荐）
```bash
# 一键运行4个实验：Mamba1-130M和Mamba2-2.7B的默认/无裁剪对比
./compare_percentile_effects.sh
```

**实验列表**：
1. Mamba1-130M + 默认percentile (0.9995)
2. Mamba1-130M + pa=1.0 (无裁剪)
3. Mamba2-2.7B + 默认percentile (0.9995)
4. Mamba2-2.7B + pa=1.0 (无裁剪)

**预计时间**：40-60分钟

**输出**：
- 量化模型：`testPercentileRange/pa-*/`
- 统计日志：`percentileRangeResults/experiments.jsonl`
- 激活值：`percentileRangeResults/activations_*.npz`

---

## 🔄 如何获取历史模型的统计

### 方案1：重新运行完整量化（推荐）
```bash
# 从FP16重新量化，自动记录统计
python3 main.py pretrained_models/.../mamba-370m \
  --quantize \
  --w_bits 8 --a_bits 8 \
  --percentile_alpha 1.0 \
  --eval_zero_shot --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs
```

**时间**：10-20分钟（取决于模型大小）

**输出**：
- 量化模型：`testPercentileRange/pa-1.0/mamba-370m/`
- 统计日志：`percentileRangeResults/experiments.jsonl`
- 激活值（可选）：`percentileRangeResults/activations_*.npz`

---

### 方案2：只运行Calibration（更快）
```bash
# 不运行eval，只收集统计
python3 main.py pretrained_models/.../mamba-370m \
  --quantize \
  --w_bits 8 --a_bits 8 \
  --calib_data_num 128 \  # 少量样本
  --percentile_alpha 1.0 \
  --pretrained_dir ./pretrained_models \
  --log_dir logs
  # 不加 --eval_zero_shot
```

**时间**：5-10分钟

**输出**：
- 量化模型：保存
- 统计日志：有（但没有accuracy）
- 激活值（可选）：有

---

## 🎲 随机性影响

### Calibration的随机性来源

#### 1. 数据采样随机性
```python
# quamba/modelutils_mamba.py:324, 351
calibration_dataset.shuffle(seed=42)  # ✅ 固定seed

# 但是从512个样本中随机选择
for i in tqdm(range(num_samples)):
    input_ids = preprocess_fn(calibration_dataset[i])  # 按顺序，不随机
```

**结论**：**Calibration本身是确定的**（seed=42）

---

#### 2. GPTQ随机性
```python
# quamba/modelutils_mamba.py:404
# GPTQ从wikitext2随机抽取128样本
# ❌ 没有固定seed！
```

**结论**：**GPTQ是随机的**（±1-2%变化）

---

### 是否影响percentile统计？

| 步骤 | 随机性 | 影响percentile统计 |
|------|--------|------------------|
| **Calibration** | ❌ 无（seed=42）| ❌ **不影响** |
| **GPTQ** | ✅ 有（无seed）| ❌ **不影响**（在calibration之后）|
| **最终Accuracy** | ✅ 有 | ✅ **影响**（通过GPTQ）|

**关键发现**：
- ✅ Percentile统计是**可复现的**（同一模型，同一数据）
- ✅ 激活值分布是**可复现的**
- ❌ 最终Accuracy不可复现（GPTQ随机性）

---

## 💾 激活值保存

### 保存内容
```python
# 保存第一批512个样本的激活值（每层前10000个值）
activation_samples = {
    "layer_0.x_proj:input": [...],  # 10000个float值
    "layer_0.x_proj:output": [...],
    "layer_1.x_proj:input": [...],
}
```

### 文件大小估算
```
每层：10000个值 × 4字节 = 40KB
前3层：40KB × 6 (input+output) = 240KB
压缩后：~50KB

多次实验：
- 5次 × 50KB = 250KB
- 20次 × 50KB = 1MB
```

**结论**：文件大小**可控**，不会占用太多空间

---

### 如何使用保存的激活值

#### 加载激活值
```python
import numpy as np

# 加载
data = np.load("percentileRangeResults/activations_mamba-370m_pa1.0_20251105_153000.npz")

# 查看包含的层
print(data.files)
# ['layer_0.x_proj:input', 'layer_0.x_proj:output', ...]

# 获取某层的激活值
layer0_input = data['layer_0.x_proj:input']
print(f"Shape: {layer0_input.shape}")
print(f"Min: {layer0_input.min()}, Max: {layer0_input.max()}")
```

#### 复现percentile计算
```python
# 模拟percentile裁剪
percentile_alpha = 0.9995

# 计算percentile值
threshold = np.quantile(np.abs(layer0_input), percentile_alpha)

# 对比
print(f"裁剪前范围: [{layer0_input.min():.2f}, {layer0_input.max():.2f}]")
print(f"裁剪后阈值: {threshold:.2f}")
print(f"被裁剪比例: {(np.abs(layer0_input) > threshold).mean()*100:.4f}%")
```

---

## 📊 实验可复现性总结

### 完全可复现
- ✅ Percentile裁剪前的min/max/range
- ✅ Percentile裁剪后的min/max/range
- ✅ 激活值分布（如果保存）
- ✅ Reorder效果

### 不可复现（需固定GPTQ seed）
- ❌ 最终accuracy（GPTQ随机性）
- ❌ 量化后的权重精确值

---

## 🔧 启用激活值保存

### 命令行开关（TODO）
```bash
# 未来可以添加
python3 main.py ... --save_activations
```

### 当前方法（修改代码）
```python
# main.py
plogger = reset_percentile_logger(
    log_file="percentileRangeResults/experiments.jsonl",
    save_activations=True  # ← 改为True
)
```

---

## 📂 文件组织

```
percentileRangeResults/
├── experiments.jsonl                          # 所有实验元数据
├── activations_mamba-130m_default_20251105.npz   # 激活值
├── activations_mamba-130m_pa1.0_20251105.npz
├── activations_mamba2-2.7b_default_20251105.npz
└── activations_mamba2-2.7b_pa1.0_20251105.npz
```

每个`.npz`文件包含：
- 前3层的input/output激活值
- 每层10000个采样值
- 压缩存储，~50KB/文件

---

## ⚡ 性能影响

### 记录统计（不保存激活值）
- 额外时间：<1秒
- 内存开销：可忽略
- 磁盘占用：~1KB/实验

### 保存激活值
- 额外时间：~2-3秒
- 内存开销：~10MB
- 磁盘占用：~50KB/实验

**结论**：**性能影响极小**

---

**最后更新**：2025-11-05
