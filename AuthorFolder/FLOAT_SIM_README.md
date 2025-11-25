# Float Simulation Implementation - 详细说明

## 修改总结

### Offline vs Online 修改

#### ❌ Offline (校准/量化) - **完全没有修改**
- `from_fp16()` 方法生成的量化参数（scales, weights）与原版完全一致
- 校准过程不受影响
- 量化模型的保存和加载不受影响

#### ✅ Online (推理) - **只修改了forward路径**
以下文件的 `forward()` 方法被修改，添加了float simulation支持：

1. **`quamba/qConvLayer.py`**
   - 修改了 `forward()` - 添加float simulation路径选择
   - 添加了 `_forward_float_sim()` - FP32模拟量化
   - 添加了 `_log_conv1d_output()` - 记录输出用于检查
   - 添加了全局计数器 `_CONV1D_LAYER_COUNTER` 追踪层索引

2. **`quamba/qLinearLayer.py`** (W8A8B8O8Linear)
   - 修改了 `forward()` - 处理FP32输入
   - 修改了 `from_fp16()` - **仅仅存储了input_scale和output_scale**，不影响量化

3. **`quamba/qSelectiveScan.py`** (QSScan)
   - 修改了 `forward()` - 处理FP32输入的u参数

### 重要特性

**当 `FLOAT_SIM_ASIC=false` 时：**
- 代码路径与原版完全一致
- 没有任何性能或精度影响
- 默认行为不变

---

## 环境变量

### 1. `FLOAT_SIM_ASIC` (默认: `false`)
控制是否启用float simulation

```bash
export FLOAT_SIM_ASIC=true   # 启用float simulation
export FLOAT_SIM_ASIC=false  # 使用原版INT8路径（默认）
```

### 2. `FLOAT_SIM_BETTER_SCALE` (默认: `false`)
只在 `FLOAT_SIM_ASIC=true` 时有效，控制是否使用更精细的量化scale

```bash
export FLOAT_SIM_BETTER_SCALE=true   # 使用 scale / SCALE_FACTOR
export FLOAT_SIM_BETTER_SCALE=false  # 使用原始scale（默认）
```

### 3. `FLOAT_SIM_SCALE_FACTOR` (默认: `2025`)
只在 `FLOAT_SIM_ASIC=true` 且 `FLOAT_SIM_BETTER_SCALE=true` 时有效

```bash
export FLOAT_SIM_SCALE_FACTOR=2025  # effective_scale = output_scale / 2025
export FLOAT_SIM_SCALE_FACTOR=1000  # effective_scale = output_scale / 1000
```

### 4. `YZ_CHECK_FLOAT_SIM` (默认: `false`)
控制是否记录conv1d输出到 `yzCheckFloatSim/` 目录

```bash
export YZ_CHECK_FLOAT_SIM=true   # 启用记录
export YZ_CHECK_FLOAT_SIM=false  # 禁用记录（默认）
```

---

## 数据流

### 原始INT8路径 (FLOAT_SIM_ASIC=false)
```
in_proj (FP16 → INT8)
    ↓
conv1d (INT8 → INT8, CUDA kernel)
    ↓
x_proj (INT8 → INT8, CUDA kernel)
    ↓
SSM (INT8 → INT8, CUDA kernel)
    ↓
had → out_proj → FP16
```

### Float Simulation路径 (FLOAT_SIM_ASIC=true)
```
in_proj (FP16 → INT8)
    ↓
conv1d (INT8 → FP32*, PyTorch)
    │
    ├─ 步骤1: Dequantize INT8 → FP32
    ├─ 步骤2: Conv1D + SiLU (FP32)
    └─ 步骤3: Simulate quantization
               y_sim = round(y / scale) * scale
               (scale可以是原始scale或scale/N)
    ↓
x_proj (FP32 → INT8, CUDA kernel)
    │
    └─ round(x_fp32 / input_scale) → INT8
    ↓
SSM (FP32 u + INT8 dt,B,C,z → INT8, CUDA kernel)
    │
    ├─ round(u_fp32 / u_scale) → INT8
    └─ 使用原始INT8 CUDA kernel
    ↓
had → out_proj → FP16
```

**注意：** 虽然conv1d输出FP32，但下游层会将其转换为INT8后使用原始CUDA kernel，确保计算正确性。

---

## 输出日志格式

当 `YZ_CHECK_FLOAT_SIM=true` 时，每一层conv1d的输出会被记录到 `yzCheckFloatSim/` 目录。

### 文件命名规则

1. **INT8 baseline**: `layer{idx:02d}_int8_baseline.json`
   - 例: `layer00_int8_baseline.json`, `layer01_int8_baseline.json`

2. **Float sim (same scale)**: `layer{idx:02d}_floatsim_samescale.json`
   - 例: `layer00_floatsim_samescale.json`

3. **Float sim (better scale)**: `layer{idx:02d}_floatsim_betterscale_{factor}.json`
   - 例: `layer00_floatsim_betterscale_2025.json`

### JSON格式

```json
{
  "layer_idx": 0,
  "float_sim_asic": true,
  "float_sim_better_scale": false,
  "effective_scale": 0.123456,
  "output_scale": 0.123456,
  "before_quant": [1.234, 2.345, ...],  // 前10个值，量化前 (FP32)
  "after_quant": [1.200, 2.400, ...]    // 前10个值，量化后
}
```

**字段说明：**
- `layer_idx`: 层索引 (0-based)
- `float_sim_asic`: 是否启用float simulation
- `float_sim_better_scale`: 是否使用better scale
- `effective_scale`: 实际使用的量化scale
- `output_scale`: 原始output scale
- `before_quant`: 量化前的前10个值 (只有float sim模式才有，INT8模式为null)
- `after_quant`: 量化后的前10个值

---

## 使用示例

### 示例1: 基础测试（验证一致性）

```bash
cd /home/yz/myprojects/2025/logquamba/Quamba/temp-originalquamba

# 运行测试，验证float sim (same scale) 与 INT8 baseline 一致
python test_check_float_sim.py --quantize --seq-len 32
```

这会生成：
- `yzCheckFloatSim/layer00_int8_baseline.json`
- `yzCheckFloatSim/layer00_floatsim_samescale.json`
- `yzCheckFloatSim/layer01_int8_baseline.json`
- `yzCheckFloatSim/layer01_floatsim_samescale.json`
- ...（每一层都有对应文件）

### 示例2: 测试Better Scale

```bash
# 测试float sim with better scale (scale / 2025)
python test_check_float_sim.py \
    --quantize \
    --seq-len 32 \
    --test-better-scale \
    --scale-factor 2025
```

这会额外生成：
- `yzCheckFloatSim/layer00_floatsim_betterscale_2025.json`
- `yzCheckFloatSim/layer01_floatsim_betterscale_2025.json`
- ...

### 示例3: 在你自己的代码中使用

```python
import os
import torch
from quamba.quamba_mixer_seq import QuambaLMHeadModel

# 设置环境变量
os.environ['FLOAT_SIM_ASIC'] = 'true'
os.environ['FLOAT_SIM_BETTER_SCALE'] = 'true'
os.environ['FLOAT_SIM_SCALE_FACTOR'] = '2025'
os.environ['YZ_CHECK_FLOAT_SIM'] = 'true'  # 可选，用于记录输出

# 加载模型
model = QuambaLMHeadModel.from_pretrained("path/to/quamba-model", device="cuda")

# 推理
input_ids = torch.randint(1, 1000, (1, 64), device="cuda")
output = model(input_ids)

# 检查结果
# 输出会保存在 yzCheckFloatSim/ 目录
```

---

## 检查和验证

### 手动检查日志文件

```bash
# 查看第0层的所有日志
cat yzCheckFloatSim/layer00_int8_baseline.json
cat yzCheckFloatSim/layer00_floatsim_samescale.json
cat yzCheckFloatSim/layer00_floatsim_betterscale_2025.json
```

### 使用Python脚本对比

```python
import json
from pathlib import Path

# 读取日志
baseline = json.load(open("yzCheckFloatSim/layer00_int8_baseline.json"))
floatsim = json.load(open("yzCheckFloatSim/layer00_floatsim_samescale.json"))

# 对比
print("Baseline after_quant:", baseline["after_quant"])
print("FloatSim after_quant:", floatsim["after_quant"])

# 计算差异
import numpy as np
diff = np.abs(np.array(baseline["after_quant"]) - np.array(floatsim["after_quant"]))
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")
```

### 预期结果

1. **Float Sim (same scale) vs INT8 Baseline**:
   - `after_quant` 的值应该非常接近（差异 < 1e-5）
   - 因为使用相同的scale，只是计算路径不同

2. **Float Sim (better scale) vs INT8 Baseline**:
   - `after_quant` 的值应该不同
   - 因为使用了更精细的scale (scale / 2025)
   - `effective_scale` 应该是 `output_scale / 2025`

3. **Before vs After Quant (Float Sim)**:
   - `before_quant` 是Conv1D+SiLU后的FP32值
   - `after_quant` 是模拟量化后的值: `round(before / scale) * scale`
   - 差异取决于scale的粗细

---

## 未来扩展

### 动态Scale Factor

你可以修改 `qConvLayer.py` 中的scale计算逻辑，使其根据某些条件动态调整：

```python
# 在 _forward_float_sim() 中，替换：
if float_sim_better_scale:
    scale_factor = float(os.environ.get('FLOAT_SIM_SCALE_FACTOR', '2025'))
    effective_scale = self.output_scale / scale_factor

# 为：
if float_sim_better_scale:
    # 动态计算scale_factor，基于某些条件
    scale_factor = self._compute_dynamic_scale_factor(y_fp32, layer_idx)
    effective_scale = self.output_scale / scale_factor
```

### 记录更多信息

可以在 `_log_conv1d_output()` 中添加更多统计信息：

```python
log_entry["statistics"] = {
    "mean": float(y_after_quant.mean()),
    "std": float(y_after_quant.std()),
    "min": float(y_after_quant.min()),
    "max": float(y_after_quant.max()),
}
```

---

## 故障排除

### 问题1: 输出不一致

**症状**: Float Sim (same scale) 的输出与INT8 baseline差异很大

**可能原因**:
- 代码路径选择错误
- scale使用不正确

**检查方法**:
```bash
# 查看日志确认settings
cat yzCheckFloatSim/layer00_floatsim_samescale.json | grep -E "(effective_scale|output_scale)"
```

`effective_scale` 应该等于 `output_scale`

### 问题2: 没有生成日志文件

**症状**: `yzCheckFloatSim/` 目录为空

**可能原因**:
- 忘记设置 `YZ_CHECK_FLOAT_SIM=true`

**解决方法**:
```bash
export YZ_CHECK_FLOAT_SIM=true
python your_script.py
```

### 问题3: 层索引不连续

**症状**: 只有部分层有日志（比如只有layer00, layer02, ...）

**可能原因**:
- 某些层的conv1d可能没有被调用
- 或者使用了不同的Conv1D类（比如Quamba2Conv1D）

**检查方法**:
```bash
ls -la yzCheckFloatSim/
```

---

## 总结

这个实现提供了：

1. ✅ **完全向后兼容**: 默认行为与原版一致
2. ✅ **Offline不受影响**: 量化过程完全不变
3. ✅ **灵活的Scale控制**: 可以测试不同的量化粒度
4. ✅ **详细的日志记录**: 方便验证和调试
5. ✅ **易于扩展**: 可以添加动态scale计算逻辑

使用 `YZ_CHECK_FLOAT_SIM=true` 可以详细检查每一层conv1d的输出，确保实现符合预期！
