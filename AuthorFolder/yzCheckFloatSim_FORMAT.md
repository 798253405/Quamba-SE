# yzCheckFloatSim 文件格式说明

## 生成的文件

运行 `python test_check_float_sim.py --quantize` 会在 `yzCheckFloatSim/` 目录下生成：

### 文件列表

1. **`int8_baseline.json`** - INT8基线（原始路径）
2. **`floatsim_samescale.json`** - Float simulation with same scale
3. **`floatsim_betterscale_2025.json`** - Float simulation with better scale (如果使用 `--test-better-scale`)

**总共：2-3个文件**（不是48个！）

每个文件包含所有24层的信息。

---

## JSON文件结构

### 完整示例

```json
{
  "config": {
    "float_sim_asic": true,
    "float_sim_better_scale": false
  },
  "layers": [
    {
      "layer_idx": 0,
      "effective_scale": 0.012345,
      "output_scale": 0.012345,
      "before_quant": [
        1.234567,
        2.345678,
        -0.123456,
        0.987654,
        ...  // 总共10个值
      ],
      "after_quant": [
        1.234000,
        2.346000,
        -0.123000,
        0.988000,
        ...  // 总共10个值
      ]
    },
    {
      "layer_idx": 1,
      "effective_scale": 0.023456,
      "output_scale": 0.023456,
      "before_quant": [...],
      "after_quant": [...]
    },
    ...  // 一直到 layer 23
    {
      "layer_idx": 23,
      "effective_scale": 0.034567,
      "output_scale": 0.034567,
      "before_quant": [...],
      "after_quant": [...]
    }
  ]
}
```

---

## 字段说明

### config (顶层配置)

| 字段 | 类型 | 说明 |
|------|------|------|
| `float_sim_asic` | boolean | 是否启用float simulation |
| `float_sim_better_scale` | boolean | 是否使用better scale (scale/N) |

### layers (层信息数组)

每个层包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `layer_idx` | int | 层索引 (0-23 for mamba-130m) |
| `effective_scale` | float | 实际使用的量化scale |
| `output_scale` | float | 原始output scale |
| `before_quant` | array[10] | 量化前的前10个值 |
| `after_quant` | array[10] | 量化后的前10个值 |

---

## before_quant 的含义

### 对于 `floatsim_samescale.json` 和 `floatsim_betterscale_*.json`

`before_quant` 是 **真实的FP32值**，来自：
```python
y_fp32 = Conv1D(x) + SiLU(Conv1D(x))  # 在PyTorch中计算
```

这是量化之前的精确值。

### 对于 `int8_baseline.json`

`before_quant` 是 **近似值**，通过dequantize得到：
```python
y_before_approx = y_int8 * output_scale
```

**为什么是近似？**
因为INT8路径直接调用CUDA kernel：
```
INT8 input → [CUDA内部: dequant → conv → silu → quant] → INT8 output
```

我们拿不到CUDA内部的FP32中间值，只能通过 `y_int8 * scale` 反推。

**近似的误差：**
```
真实值:    1.234567
量化后:    round(1.234567 / 0.01) * 0.01 = 1.23
反推近似:  1.23  (丢失了小数部分 0.004567)
```

---

## 三个文件的区别

### 1. `int8_baseline.json`

```json
{
  "config": {
    "float_sim_asic": false,        // ← INT8路径
    "float_sim_better_scale": false
  },
  "layers": [
    {
      "layer_idx": 0,
      "effective_scale": 0.012345,   // = output_scale
      "output_scale": 0.012345,
      "before_quant": [1.23, ...],   // ← 近似值 (y_int8 * scale)
      "after_quant": [1.23, ...]     // ← INT8 CUDA输出
    }
  ]
}
```

### 2. `floatsim_samescale.json`

```json
{
  "config": {
    "float_sim_asic": true,         // ← Float simulation
    "float_sim_better_scale": false // ← 使用相同scale
  },
  "layers": [
    {
      "layer_idx": 0,
      "effective_scale": 0.012345,   // = output_scale (相同!)
      "output_scale": 0.012345,
      "before_quant": [1.234567, ...],  // ← 真实FP32值
      "after_quant": [1.23, ...]        // ← 模拟量化后
    }
  ]
}
```

**预期：** `after_quant` 应该与 `int8_baseline.json` 的 `after_quant` **非常接近**（差异 < 1e-5）

### 3. `floatsim_betterscale_2025.json`

```json
{
  "config": {
    "float_sim_asic": true,
    "float_sim_better_scale": true  // ← 使用更精细的scale
  },
  "layers": [
    {
      "layer_idx": 0,
      "effective_scale": 0.000006098, // = output_scale / 2025
      "output_scale": 0.012345,
      "before_quant": [1.234567, ...],  // ← 真实FP32值 (与上面相同)
      "after_quant": [1.234568, ...]    // ← 更精细的量化 (更接近before_quant)
    }
  ]
}
```

**预期：** `after_quant` 应该与 `before_quant` **更接近**，因为scale更小，量化误差更小

---

## 如何使用这些文件进行检查

### Python脚本示例

```python
import json
import numpy as np

# 读取三个文件
baseline = json.load(open("yzCheckFloatSim/int8_baseline.json"))
floatsim_same = json.load(open("yzCheckFloatSim/floatsim_samescale.json"))
floatsim_better = json.load(open("yzCheckFloatSim/floatsim_betterscale_2025.json"))

# 检查第0层
layer0_baseline = baseline["layers"][0]
layer0_same = floatsim_same["layers"][0]
layer0_better = floatsim_better["layers"][0]

print("Layer 0 检查:")
print(f"  Baseline scale:     {layer0_baseline['effective_scale']}")
print(f"  FloatSim scale:     {layer0_same['effective_scale']}")
print(f"  Better scale:       {layer0_better['effective_scale']}")

# 对比 after_quant
baseline_vals = np.array(layer0_baseline["after_quant"])
floatsim_vals = np.array(layer0_same["after_quant"])

diff = np.abs(baseline_vals - floatsim_vals)
print(f"\nBaseline vs FloatSim (same scale):")
print(f"  Max diff:  {diff.max():.6e}")
print(f"  Mean diff: {diff.mean():.6e}")

# 检查量化误差
before_vals = np.array(layer0_better["before_quant"])
after_vals = np.array(layer0_better["after_quant"])
quant_error = np.abs(before_vals - after_vals)

print(f"\nBetter scale quantization error:")
print(f"  Max error:  {quant_error.max():.6e}")
print(f"  Mean error: {quant_error.mean():.6e}")
```

### 快速查看

```bash
# 查看文件列表
ls -lh yzCheckFloatSim/

# 查看第0层的scale
jq '.layers[0] | {layer_idx, effective_scale, output_scale}' yzCheckFloatSim/int8_baseline.json
jq '.layers[0] | {layer_idx, effective_scale, output_scale}' yzCheckFloatSim/floatsim_samescale.json
jq '.layers[0] | {layer_idx, effective_scale, output_scale}' yzCheckFloatSim/floatsim_betterscale_2025.json

# 查看第0层的前3个值
jq '.layers[0].after_quant[0:3]' yzCheckFloatSim/int8_baseline.json
jq '.layers[0].after_quant[0:3]' yzCheckFloatSim/floatsim_samescale.json
```

---

## 预期的检查结果

### ✓ 正确的实现

1. **Same scale vs Baseline:**
   - `effective_scale` 相同
   - `after_quant` 非常接近（差异 < 1e-5）

2. **Better scale:**
   - `effective_scale` = `output_scale / 2025`
   - `after_quant` 更接近 `before_quant`（量化误差更小）

3. **所有层:**
   - 24层都有数据
   - 每层的 `layer_idx` 从 0 到 23

### ✗ 有问题的情况

1. `effective_scale` 不对
2. `after_quant` 差异很大
3. 缺少某些层的数据
4. `before_quant` 全是 null

---

## 文件大小预估

每个文件包含：
- 24层
- 每层20个float值（10个before + 10个after）
- 约 **5-10 KB** per file

总共2-3个文件，约 **15-30 KB**。
