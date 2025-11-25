# 三种模拟模式说明

## 📊 三种模式对比

### Mode 1: INT8 Baseline（基线）
- **环境变量**: 所有flag都设为false（默认）
- **实现**: 使用原始INT8 CUDA kernel
- **例子**: 标准INT8量化流程
- **用途**: 作为对比基线

### Mode 2: FP32 SSM Input（上限）
- **命令行**: `--fp32-ssm-input`
- **环境变量**: `FP32_SSM_INPUT=true`
- **实现**: Conv1D输出保持FP32，**不做量化**
- **例子**: `0.5322` → `0.5322` (保持原值)
- **用途**: 看理论上限性能

### Mode 3: Float Sim ASIC INT8（验证）
- **命令行**: `--float-sim-asic-int8`
- **环境变量**: `FLOAT_SIM_ASIC_INT8=true`
- **实现**: 用FP32模拟INT8量化行为
- **例子**: `0.5322` → `round(0.5322/0.53) * 0.53 = 1 * 0.53 = 0.53`
- **用途**: 验证模拟正确性，应该与INT8 baseline一致

### Mode 4: Float Sim ASIC Research SE（研究）
- **命令行**: `--float-sim-asic-research-se`
- **环境变量**: `FLOAT_SIM_ASIC_RESEARCH_SE=true`
- **实现**: 用FP32模拟INT8量化，但**scale增强**
- **例子**: `0.5322` → `round(0.5322/0.53) * 0.53 * 2025 = 1 * 0.53 * 2025 = 1073.25`
- **参数**: `FLOAT_SIM_SCALE_FACTOR` (默认2025)
- **用途**: 你的研究重点

---

## 🚀 快速开始

### 方法1: 使用Quamba模型（推荐，更快）

```bash
# 使用已经量化好的Quamba模型
python3 test_three_modes.py \
    --model ut-enyac/quamba-130m-w8a8 \
    --pretrained-dir pretrained_models
```

这会运行所有4个测试并生成日志文件到 `yzCheckFloatSim/`:
- `int8_baseline.json`
- `fp32_upper_bound.json`
- `float_sim_int8.json`
- `float_sim_research_se_2025.json`

### 方法2: 使用Mamba模型（会现场量化，较慢）

```bash
# 加载FP16 Mamba模型并现场量化
python3 test_three_modes.py \
    --model pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
    --quantize
```

### 自定义scale factor

```bash
python3 test_three_modes.py \
    --model ut-enyac/quamba-130m-w8a8 \
    --pretrained-dir pretrained_models \
    --scale-factor 1000
```

生成文件: `float_sim_research_se_1000.json`

### 禁用日志（只看对比结果）

```bash
python3 test_three_modes.py \
    --model ut-enyac/quamba-130m-w8a8 \
    --pretrained-dir pretrained_models \
    --no-logging
```

---

## 📁 生成的日志文件

### 文件列表

```
yzCheckFloatSim/
├── int8_baseline.json                    # Mode 1: INT8 baseline
├── fp32_upper_bound.json                 # Mode 2: FP32 upper bound
├── float_sim_int8.json                   # Mode 3: Float sim INT8
└── float_sim_research_se_2025.json       # Mode 4: Research SE
```

### 文件格式

```json
{
  "config": {
    "mode": "float_sim_int8",
    "description": "FP32 simulation of INT8 quantization (should match baseline)"
  },
  "layers": [
    {
      "layer_idx": 0,
      "effective_scale": 0.012345,
      "output_scale": 0.012345,
      "before_quant": [0.5322, 0.6543, ...],  // 量化前 (FP32)
      "after_quant": [0.53, 0.65, ...]        // 量化后
    },
    {
      "layer_idx": 1,
      ...
    },
    ...  // 24 layers total
  ]
}
```

---

## 🔬 详细例子

假设某一层的某个值是 `0.5322`，output_scale = `0.53`

### Mode 1: INT8 Baseline
```
CUDA内部流程:
  y_fp32 = 0.5322  (Conv1D + SiLU 后的FP32值，我们拿不到)
  y_int8 = round(0.5322 / 0.53) = round(1.00415) = 1
  输出: y_int8 = 1 (INT8)

日志记录:
  before_quant: 0.53  (近似值，通过 1 * 0.53 反推)
  after_quant: 0.53   (通过 1 * 0.53 dequantize)
```

### Mode 2: FP32 Upper Bound
```
流程:
  y_fp32 = 0.5322  (Conv1D + SiLU)
  输出: 0.5322 (不量化，保持FP32)

日志记录:
  before_quant: 0.5322
  after_quant: 0.5322  (与before相同)
```

### Mode 3: Float Sim INT8
```
流程:
  y_fp32 = 0.5322  (Conv1D + SiLU)
  y_sim = round(0.5322 / 0.53) * 0.53 = 1 * 0.53 = 0.53
  输出: 0.53 (FP32，但模拟INT8行为)

日志记录:
  before_quant: 0.5322  (真实FP32值)
  after_quant: 0.53     (模拟量化后)

验证: after_quant 应该与 Mode 1 的结果一致
```

### Mode 4: Float Sim Research SE (factor=2025)
```
流程:
  y_fp32 = 0.5322  (Conv1D + SiLU)
  y_int8_value = round(0.5322 / 0.53) = 1  (INT8整数值)
  y_enhanced = 1 * 0.53 * 2025 = 1073.25
  输出: 1073.25 (FP32)

日志记录:
  before_quant: 0.5322     (真实FP32值)
  after_quant: 1073.25     (增强后的值)
  effective_scale: 1073.25 (= 0.53 * 2025)
```

---

## ✅ 预期结果

### 1. Float Sim INT8 vs Baseline
```
Max diff: 0.000000e+00
Mean diff: 0.000000e+00
```
**应该完全一致**，因为是用FP32模拟INT8

### 2. FP32 Upper Bound vs Baseline
```
Max diff: ~0.01-0.1
Mean diff: ~0.001-0.01
```
**应该更好**（diff > 0），因为保留了更多精度

### 3. Research SE vs Baseline
```
Max diff: 很大 (取决于scale factor)
Mean diff: 很大
```
**完全不同**，因为scale被放大了

---

## 🔧 环境变量对照表

| Mode | FP32_SSM_INPUT | FLOAT_SIM_ASIC_INT8 | FLOAT_SIM_ASIC_RESEARCH_SE | FLOAT_SIM_SCALE_FACTOR |
|------|----------------|---------------------|----------------------------|------------------------|
| Baseline | false | false | false | - |
| FP32 Upper Bound | **true** | false | false | - |
| Float Sim INT8 | false | **true** | false | - |
| Research SE | false | false | **true** | 2025 (可调) |

---

## 📝 在你自己的代码中使用

```python
import os

# Mode 1: Baseline (默认)
# 不需要设置任何环境变量

# Mode 2: FP32 Upper Bound
os.environ['FP32_SSM_INPUT'] = 'true'

# Mode 3: Float Sim INT8
os.environ['FLOAT_SIM_ASIC_INT8'] = 'true'

# Mode 4: Research SE
os.environ['FLOAT_SIM_ASIC_RESEARCH_SE'] = 'true'
os.environ['FLOAT_SIM_SCALE_FACTOR'] = '2025'

# 启用日志
os.environ['YZ_CHECK_FLOAT_SIM'] = 'true'

# 运行模型
model = ...
output = model(input_ids)
```

---

## 🎯 使用建议

1. **首先运行** Mode 3 (Float Sim INT8)，验证与 Baseline 一致
2. **然后运行** Mode 2 (FP32 Upper Bound)，看理论上限
3. **最后运行** Mode 4 (Research SE)，测试你的研究想法
4. **对比** Mode 4 与 Mode 2 的gap，看是否接近上限

---

## 📊 典型工作流

```bash
# 1. 运行所有模式，生成日志
python3 test_three_modes.py --quantize

# 2. 查看日志文件
ls -lh yzCheckFloatSim/

# 3. 检查某一层的值
cat yzCheckFloatSim/int8_baseline.json | python3 -m json.tool | grep -A 15 '"layer_idx": 0'
cat yzCheckFloatSim/float_sim_int8.json | python3 -m json.tool | grep -A 15 '"layer_idx": 0'

# 4. 对比不同模式的结果
python3 -c "
import json
baseline = json.load(open('yzCheckFloatSim/int8_baseline.json'))
fp32 = json.load(open('yzCheckFloatSim/fp32_upper_bound.json'))
research = json.load(open('yzCheckFloatSim/float_sim_research_se_2025.json'))

print('Layer 0 对比:')
print(f'Baseline:  {baseline[\"layers\"][0][\"after_quant\"][:3]}')
print(f'FP32:      {fp32[\"layers\"][0][\"after_quant\"][:3]}')
print(f'Research:  {research[\"layers\"][0][\"after_quant\"][:3]}')
"

# 5. 测试不同的scale factor
python3 test_three_modes.py --quantize --scale-factor 1000
python3 test_three_modes.py --quantize --scale-factor 5000
```

---

生成时间: 2025-11-07
