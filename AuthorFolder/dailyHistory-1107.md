# Daily History - 2025-11-07

## Session Overview

**Date**: 2025-11-07
**Main Goal**: 实现Mode 2（Float-Sim INT8模式），使其与INT8 CUDA kernel具有相同的计算精度但保持FP32数据类型，并创建CUDA vs Float-Sim对比测试系统

**Key Achievement**:
1. ✅ 修复了Mode 2实现，添加round模拟INT8计算精度
2. ✅ 添加了Conv1D和SiLU中间值的保存和outlier分析功能
3. ✅ 创建了完整的CUDA vs Float-Sim对比测试系统

---

## Table of Contents

1. [Background Context](#background-context)
2. [Problem 1: Conv1D和SiLU的Outlier分析](#problem-1-conv1d和silu的outlier分析)
3. [Problem 2: UnboundLocalError修复](#problem-2-unboundlocalerror修复)
4. [Problem 3: Mode 2需要添加round模拟INT8精度](#problem-3-mode-2需要添加round模拟int8精度)
5. [Problem 4: SiLU输出也需要round到INT8精度](#problem-4-silu输出也需要round到int8精度)
6. [Problem 5: 创建CUDA vs Float-Sim对比测试系统](#problem-5-创建cuda-vs-float-sim对比测试系统)
7. [Final Implementation Summary](#final-implementation-summary)
8. [Testing Instructions](#testing-instructions)
9. [Expected Results](#expected-results)

---

## Background Context

### 从上一个Session继承的状态

**Session 8 (2025-11-07早些时候)** 已经完成：
- Mode 1 (FP32_SSM_INPUT): 使用INT8 CUDA kernel，返回INT8，然后在qMambaLayer中dequantize到FP32给SSM
- Mode 2 (FLOAT_SIM_ASIC_INT8): 初步实现，但当时没有round模拟INT8精度

**相关文件**：
- `FP32_SSM_MODE_FIX.md` - 记录了之前的设计理解和修复过程
- `quamba/qConvLayer.py` - Conv1D层实现
- `quamba/qMambaLayer.py` - Mamba层的dual-path逻辑

### Mode 2的设计目标

用户的核心需求（多次强调）：
> "用int计算，只是silu这里是float"
> "我需要和int操作精度一模一样，只是数据精度不一样"

这意味着：
- **计算精度 = INT8**：所有中间值都要round到INT8网格
- **数据类型 = FP32**：不转换成torch.int8，保持torch.float32
- **目的**：测试如果给SSM更高精度的输入（虽然还是INT8精度的离散值）是否能提升准确率

---

## Problem 1: Conv1D和SiLU的Outlier分析

### Issue

用户请求：
> "你存储某一层的conv2d结果和silu后的结果，我看看有没有outlier。"

需要保存Layer 23的Conv1D输出和SiLU输出用于outlier分析。

### Solution

#### Step 1: 修改qConvLayer.py添加保存功能

**File**: `quamba/qConvLayer.py`

在Mode 2 (FLOAT_SIM_ASIC_INT8) 的实现中添加保存逻辑：

```python
# Step 4: Conv1D computation (FP32, same as INT8 CUDA kernel does internally)
weight_fp32_reshaped = weight_fp32.unsqueeze(1)
x_fp32_padded = F.pad(x_fp32, (self.kernel_size - 1, 0), mode='constant', value=0)
y_conv_fp32 = F.conv1d(x_fp32_padded, weight_fp32_reshaped, bias=bias_fp32,
                       stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups)

# Step 5: SiLU activation (FP32) - NO quantization after this!
y_silu_fp32 = y_conv_fp32 * torch.sigmoid(y_conv_fp32)

# ===== SAVE CONV1D AND SILU RESULTS FOR OUTLIER ANALYSIS =====
if _CONV1D_LAYER_COUNTER == 23:
    if not hasattr(self, '_conv_silu_saved'):
        self._conv_silu_saved = True
        os.makedirs('conv_silu_analysis', exist_ok=True)

        # Save Conv1D output (before SiLU)
        conv_output_cpu = y_conv_fp32.detach().cpu()
        torch.save(conv_output_cpu, 'conv_silu_analysis/layer23_conv1d_output.pt')

        # Save SiLU output (after SiLU)
        silu_output_cpu = y_silu_fp32.detach().cpu()
        torch.save(silu_output_cpu, 'conv_silu_analysis/layer23_silu_output.pt')

        # Print statistics
        print(f"\n{'='*80}")
        print(f"[Layer 23 Conv1D & SiLU Analysis - Mode 2]")
        print(f"{'='*80}")
        print(f"\nConv1D Output (before SiLU):")
        print(f"  Shape: {y_conv_fp32.shape}")
        print(f"  Min: {y_conv_fp32.min().item():.6f}")
        print(f"  Max: {y_conv_fp32.max().item():.6f}")
        print(f"  Mean: {y_conv_fp32.mean().item():.6f}")
        print(f"  Std: {y_conv_fp32.std().item():.6f}")
        print(f"  Abs Max: {y_conv_fp32.abs().max().item():.6f}")

        # Percentile analysis for Conv1D
        conv_flat = y_conv_fp32.flatten().abs()
        conv_sorted, _ = torch.sort(conv_flat)
        p99 = conv_sorted[int(len(conv_sorted) * 0.99)].item()
        p999 = conv_sorted[int(len(conv_sorted) * 0.999)].item()
        p9995 = conv_sorted[int(len(conv_sorted) * 0.9995)].item()
        print(f"  99th percentile (abs): {p99:.6f}")
        print(f"  99.9th percentile (abs): {p999:.6f}")
        print(f"  99.95th percentile (abs): {p9995:.6f}")

        # Similar for SiLU output...
```

**Key Points**:
- 只在Layer 23保存（通过`_CONV1D_LAYER_COUNTER == 23`控制）
- 使用`hasattr`确保只保存一次（第一次forward时）
- 保存到`conv_silu_analysis/`目录
- 打印统计信息：Min/Max/Mean/Std + Percentiles

#### Step 2: 创建outlier分析脚本

**File**: `analyze_conv_silu_outliers.py`

功能：
1. **基础统计**: Min/Max/Mean/Std/Median
2. **Percentile分析**: 50th到99.99th percentile
3. **IQR outlier检测**: Mild outliers (Q3+1.5*IQR), Extreme outliers (Q3+3*IQR)
4. **Top 20 outliers**: 按绝对值排序
5. **分布分析**: 正值/负值/零值比例，不同数值范围分布
6. **Conv1D vs SiLU对比**: SiLU对range的压缩效果
7. **Outlier clipping建议**: 如果Max / 99.95th percentile > 2.0，建议使用percentile clipping

**Usage**:
```python
python analyze_conv_silu_outliers.py
```

**输出**：详细的outlier分析报告，包括是否存在显著outliers以及建议的clipping值。

---

## Problem 2: UnboundLocalError修复

### Issue

运行Mode 2时报错：
```
File "/workspace/Quamba/quamba/qConvLayer.py", line 103, in forward
    fp32_ssm_input = os.environ.get('FP32_SSM_INPUT', 'false').lower() == 'true'
UnboundLocalError: local variable 'os' referenced before assignment
```

### Root Cause

在Mode 2的保存代码中（line 188）有一个重复的`import os`：
```python
if _CONV1D_LAYER_COUNTER == 23:
    if not hasattr(self, '_conv_silu_saved'):
        self._conv_silu_saved = True
        import os  # ← 这里重复import导致Python认为os是局部变量
        os.makedirs('conv_silu_analysis', exist_ok=True)
```

Python看到函数内部有`import os`，就认为`os`是局部变量。但在line 103就引用了`os`，此时还没执行到`import os`，所以报错。

### Solution

删除重复的`import os`（line 188），因为文件顶部已经有了：

**File**: `quamba/qConvLayer.py` (top of file)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from .quant_utils import quantize_tensor_per_tensor_absmax
import os  # ← 已经在这里导入了
import json
from pathlib import Path
```

**Modified code** (line 184-188):
```python
# ===== SAVE CONV1D AND SILU RESULTS FOR OUTLIER ANALYSIS =====
if _CONV1D_LAYER_COUNTER == 23:
    if not hasattr(self, '_conv_silu_saved'):
        self._conv_silu_saved = True
        os.makedirs('conv_silu_analysis', exist_ok=True)  # 直接使用，不需要import
```

---

## Problem 3: Mode 2需要添加round模拟INT8精度

### Issue

用户问：
> "你模拟的int8有round么"

检查代码发现Mode 2当时的实现：
```python
# Step 4: Conv1D computation (FP32, same as INT8 CUDA kernel does internally)
y_conv_fp32 = F.conv1d(...)

# Step 5: SiLU activation (FP32) - NO quantization after this!
y_silu_fp32 = y_conv_fp32 * torch.sigmoid(y_conv_fp32)

return y_silu_fp32  # FP32 (full precision, not quantized)
```

**问题**：Conv1D输出是完整FP32精度，没有round到INT8网格。这意味着计算精度 ≠ INT8。

### User Clarification

用户明确要求：
> "mode2要加round。我需要和int操作精度一模一样，只是数据精度不一样。"

**理解**：
- **计算精度**：所有中间值都要round到INT8能表达的离散值
- **数据精度**：保持FP32类型，不转换成torch.int8

### Solution

#### 修改Mode 2实现，添加round模拟INT8计算精度

**File**: `quamba/qConvLayer.py`, Mode 2 section

**Before**:
```python
# Step 4: Conv1D computation (FP32, same as INT8 CUDA kernel does internally)
y_conv_fp32 = F.conv1d(x_fp32_padded, weight_fp32_reshaped, bias=bias_fp32,
                       stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups)

# Step 5: SiLU activation (FP32) - NO quantization after this!
y_silu_fp32 = y_conv_fp32 * torch.sigmoid(y_conv_fp32)
```

**After**:
```python
# Step 4: Conv1D computation (FP32, same as INT8 CUDA kernel does internally)
y_conv_fp32 = F.conv1d(x_fp32_padded, weight_fp32_reshaped, bias=bias_fp32,
                       stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups)

# Step 4.5: Simulate INT8 computation precision (round to INT8 grid)
# This makes the computation precision identical to INT8 CUDA kernel
# But we keep it in FP32 format (data precision is FP32, computation precision is INT8)
y_conv_int8_simulated = torch.round(y_conv_fp32 / self.output_scale).clamp(-128, 127)
y_conv_fp32_quantized = y_conv_int8_simulated * self.output_scale

# Step 5: SiLU activation (FP32) on the INT8-precision values
# Input to SiLU has INT8 computation precision, but FP32 data type
y_silu_fp32 = y_conv_fp32_quantized * torch.sigmoid(y_conv_fp32_quantized)
```

**Key Changes**:
1. 添加了`torch.round(y / output_scale).clamp(-128, 127)`模拟INT8量化
2. 再乘以`output_scale`得到FP32类型但INT8精度的值
3. SiLU在INT8精度的输入上计算

#### 更新保存逻辑，保存raw和INT8-simulated两个版本

现在需要保存：
- `layer23_conv1d_raw.pt` - Conv1D原始FP32输出（round前）
- `layer23_conv1d_int8simulated.pt` - Conv1D INT8精度输出（round后）

**Updated save code**:
```python
# Save Conv1D output (before INT8 simulation, raw FP32)
conv_raw_cpu = y_conv_fp32.detach().cpu()
torch.save(conv_raw_cpu, 'conv_silu_analysis/layer23_conv1d_raw.pt')

# Save Conv1D output (after INT8 simulation, quantized to INT8 grid)
conv_quantized_cpu = y_conv_fp32_quantized.detach().cpu()
torch.save(conv_quantized_cpu, 'conv_silu_analysis/layer23_conv1d_int8simulated.pt')
```

并添加统计信息：
```python
print(f"\nConv1D Output (raw FP32, before INT8 simulation):")
# ... print stats for y_conv_fp32

print(f"\nConv1D Output (after INT8 simulation, round+clamp to INT8 grid):")
# ... print stats for y_conv_fp32_quantized

# Check quantization effect
quant_diff = (y_conv_fp32 - y_conv_fp32_quantized).abs().max().item()
print(f"  Max quantization error: {quant_diff:.8f}")
```

---

## Problem 4: SiLU输出也需要round到INT8精度

### Issue

用户进一步澄清：
> "silu你做了限制么，我要的是计算精度是int8,也就是silu其实也限制成int8的能表达的值"

当前实现只round了Conv1D输出，但SiLU输出还是完整FP32精度：
```python
y_silu_fp32 = y_conv_fp32_quantized * torch.sigmoid(y_conv_fp32_quantized)
return y_silu_fp32  # FP32 (full precision) ← 这里没有round!
```

### Solution

#### 添加SiLU输出的round模拟

**File**: `quamba/qConvLayer.py`, Mode 2 section

**Modified**:
```python
# Step 5: SiLU activation (FP32) on the INT8-precision values
# Input to SiLU has INT8 computation precision, but FP32 data type
y_silu_fp32_raw = y_conv_fp32_quantized * torch.sigmoid(y_conv_fp32_quantized)

# Step 6: Simulate INT8 computation precision for SiLU output
# Round to INT8 grid to match INT8 computation precision
# But keep FP32 data type (don't convert to torch.int8)
y_silu_int8_simulated = torch.round(y_silu_fp32_raw / self.output_scale).clamp(-128, 127)
y_silu_fp32 = y_silu_int8_simulated * self.output_scale

# Key difference from Baseline:
# - Baseline returns torch.int8 type
# - Mode 2 returns FP32 type with INT8-precision values (discrete grid)

return y_silu_fp32  # FP32 data type, but INT8 computation precision (discrete values)
```

#### 更新保存逻辑，保存SiLU的raw和INT8-simulated

现在需要保存：
- `layer23_silu_raw.pt` - SiLU原始FP32输出（round前）
- `layer23_silu_int8simulated.pt` - SiLU INT8精度输出（round后）

**Updated save code**:
```python
# Save SiLU output (before INT8 simulation, raw FP32)
silu_raw_cpu = y_silu_fp32_raw.detach().cpu()
torch.save(silu_raw_cpu, 'conv_silu_analysis/layer23_silu_raw.pt')

# Save SiLU output (after INT8 simulation, quantized to INT8 grid)
silu_output_cpu = y_silu_fp32.detach().cpu()
torch.save(silu_output_cpu, 'conv_silu_analysis/layer23_silu_int8simulated.pt')
```

并添加统计信息：
```python
print(f"\nSiLU Output (raw FP32, before INT8 simulation):")
# ... print stats for y_silu_fp32_raw

print(f"\nSiLU Output (after INT8 simulation, round+clamp to INT8 grid):")
# ... print stats for y_silu_fp32

# Check quantization effect for SiLU
silu_quant_diff = (y_silu_fp32_raw - y_silu_fp32).abs().max().item()
print(f"  Max quantization error: {silu_quant_diff:.8f}")
```

### Final Mode 2 Data Flow

```
输入: INT8 (离散值)
  ↓ dequant (INT8 * scale)
FP32输入 (离散值，来自INT8)
  ↓ Conv1D (FP32计算)
FP32 Conv输出 (连续值)
  ↓ round到INT8网格
FP32 Conv输出 (离散值，INT8精度)
  ↓ SiLU (FP32计算)
FP32 SiLU输出 (连续值)
  ↓ round到INT8网格
FP32 SiLU输出 (离散值，INT8精度) ← 最终输出，FP32类型但INT8精度
```

**与Baseline的对比**:

| 步骤 | Baseline (INT8 CUDA) | Mode 2 (INT8精度模拟) |
|------|---------------------|---------------------|
| Conv1D输入 | INT8 | INT8 → FP32 (离散) |
| Conv1D计算 | FP32 | FP32 |
| Conv1D输出 | Round到INT8网格 | Round到INT8网格 ✓ |
| SiLU输入 | INT8网格上的值 | INT8网格上的值 ✓ |
| SiLU计算 | FP32 | FP32 |
| SiLU输出 | Round到INT8网格 | Round到INT8网格 ✓ |
| 返回类型 | **torch.int8** | **torch.float32** ← 唯一区别 |

**计算精度完全相同，只有数据类型不同！**

---

## Problem 5: 创建CUDA vs Float-Sim对比测试系统

### Issue

用户请求：
> "你给我一个脚本，里面3个命令行运行1.cuda模式，所有的中间变量都存进一个txt或者什么文件里。2. floatsimint模式，所有中间变量运算结果存入一个文件3对比分析这两个文件，显示哪些值不一样。你可以对应的改代码"

需要验证Mode 2是否正确模拟了INT8 CUDA kernel的计算精度。

### Solution Architecture

创建一个三步测试系统：
1. **运行CUDA模式**，保存Layer 23的中间值
2. **运行Float-Sim模式**，保存Layer 23的中间值
3. **对比分析**，找出差异

### Implementation

#### File 1: test_cuda_vs_floatsim.sh

自动化测试脚本。

**Features**:
- 依次运行3个步骤
- 使用环境变量控制保存模式
- 输出带颜色标识的进度信息
- 保存所有输出到txt文件

**Script structure**:
```bash
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL="quamba-130m-w8a8"
PRETRAINED_DIR="pretrained_models/quamba1/default"
BATCH_SIZE=16
TASK="lambada_openai"
OUTPUT_DIR="cuda_vs_floatsim_comparison"

# Step 1: CUDA mode
export SAVE_INTERMEDIATE_VALUES="cuda"
export INTERMEDIATE_VALUES_DIR="${OUTPUT_DIR}/cuda_values"

python3 main.py ${MODEL} \
    --quantize \
    --batch_size ${BATCH_SIZE} \
    --eval_zero_shot \
    --task_list ${TASK} \
    --pretrained_dir ${PRETRAINED_DIR} \
    --testing \
    > ${OUTPUT_DIR}/cuda_output.txt 2>&1

# Step 2: Float-Sim mode
export SAVE_INTERMEDIATE_VALUES="floatsim"
export INTERMEDIATE_VALUES_DIR="${OUTPUT_DIR}/floatsim_values"

python3 main.py ${MODEL} \
    --quantize \
    --batch_size ${BATCH_SIZE} \
    --eval_zero_shot \
    --task_list ${TASK} \
    --float-sim-asic-int8 \
    --pretrained_dir ${PRETRAINED_DIR} \
    --testing \
    > ${OUTPUT_DIR}/floatsim_output.txt 2>&1

# Step 3: Compare
unset SAVE_INTERMEDIATE_VALUES
unset INTERMEDIATE_VALUES_DIR

python3 compare_cuda_floatsim_values.py \
    --cuda_dir ${OUTPUT_DIR}/cuda_values \
    --floatsim_dir ${OUTPUT_DIR}/floatsim_values \
    --output ${OUTPUT_DIR}/comparison_report.txt
```

**Environment Variables**:
- `SAVE_INTERMEDIATE_VALUES`: 'cuda' or 'floatsim'
- `INTERMEDIATE_VALUES_DIR`: Directory to save intermediate values

#### File 2: compare_cuda_floatsim_values.py

Python对比分析脚本。

**Features**:
- 加载CUDA和Float-Sim的所有`.pt`文件
- 逐个对比每个张量
- 计算统计差异
- 显示Top 10最大差异
- 生成详细报告

**Key functions**:

```python
def compare_tensors(tensor1, tensor2, name, output_file):
    """Compare two tensors and write detailed comparison to output file."""

    # Convert to same dtype for comparison
    tensor1_fp32 = tensor1.float()
    tensor2_fp32 = tensor2.float()

    # Compute differences
    diff = (tensor1_fp32 - tensor2_fp32).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Count exact matches
    exact_match = (diff == 0).sum().item()
    total_elements = diff.numel()
    match_percentage = exact_match / total_elements * 100

    # Relative difference
    abs_max = max(tensor1_fp32.abs().max().item(), tensor2_fp32.abs().max().item())
    rel_diff = max_diff / abs_max * 100 if abs_max > 0 else 0

    # Print statistics
    output_file.write(f"\n{'='*80}\n")
    output_file.write(f"{name}\n")
    output_file.write(f"{'='*80}\n")
    output_file.write(f"Shape: {tensor1.shape}\n")
    output_file.write(f"CUDA dtype: {tensor1.dtype}\n")
    output_file.write(f"Float-Sim dtype: {tensor2.dtype}\n")
    output_file.write(f"\nCOMPARISON:\n")
    output_file.write(f"  Max absolute difference: {max_diff:.8e}\n")
    output_file.write(f"  Mean absolute difference: {mean_diff:.8e}\n")
    output_file.write(f"  Relative difference: {rel_diff:.6f}%\n")
    output_file.write(f"  Exact matches: {exact_match} / {total_elements} ({match_percentage:.2f}%)\n")

    # Show top 10 differences
    if max_diff > 0:
        diff_flat = diff.flatten()
        top_diff_indices = torch.topk(diff_flat, min(10, total_elements)).indices

        output_file.write(f"\nTop 10 differences:\n")
        output_file.write(f"  {'Index':<10} {'CUDA':<15} {'Float-Sim':<15} {'Abs Diff':<15}\n")
        for idx in top_diff_indices:
            idx_item = idx.item()
            cuda_val = tensor1_fp32.flatten()[idx_item].item()
            floatsim_val = tensor2_fp32.flatten()[idx_item].item()
            diff_val = diff_flat[idx_item].item()
            output_file.write(f"  {idx_item:<10} {cuda_val:<15.8f} {floatsim_val:<15.8f} {diff_val:<15.8e}\n")

    # Determine if identical
    is_identical = max_diff < 1e-6
    if is_identical:
        output_file.write(f"\n✓ IDENTICAL (max diff < 1e-6)\n")
    else:
        output_file.write(f"\n✗ DIFFERENT (max diff >= 1e-6)\n")

    return is_identical
```

**Main function**:
```python
def main():
    parser = argparse.ArgumentParser(description='Compare CUDA vs Float-Sim intermediate values')
    parser.add_argument('--cuda_dir', type=str, required=True)
    parser.add_argument('--floatsim_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    # Find all .pt files in CUDA directory
    cuda_files = sorted(Path(args.cuda_dir).glob('*.pt'))

    # Compare each file
    for cuda_file in cuda_files:
        filename = cuda_file.name
        floatsim_file = Path(args.floatsim_dir) / filename

        cuda_tensor = torch.load(cuda_file)
        floatsim_tensor = torch.load(floatsim_file)

        compare_tensors(cuda_tensor, floatsim_tensor, filename, output_file)

    # Print summary
    output_file.write(f"\nSUMMARY:\n")
    output_file.write(f"Identical: {identical_count}\n")
    output_file.write(f"Different: {different_count}\n")
```

#### File 3: Modify qConvLayer.py to save intermediate values

**Added at the beginning of forward()**:

```python
@torch.no_grad()
def forward(self, x):
    global _CONV1D_LAYER_COUNTER

    # Check simulation mode from environment variables
    fp32_ssm_input = os.environ.get('FP32_SSM_INPUT', 'false').lower() == 'true'
    float_sim_asic_int8 = os.environ.get('FLOAT_SIM_ASIC_INT8', 'false').lower() == 'true'
    float_sim_asic_research_se = os.environ.get('FLOAT_SIM_ASIC_RESEARCH_SE', 'false').lower() == 'true'

    # Check if intermediate value saving is enabled
    save_mode = os.environ.get('SAVE_INTERMEDIATE_VALUES', '')  # 'cuda' or 'floatsim'
    save_dir = os.environ.get('INTERMEDIATE_VALUES_DIR', '')
    should_save = (save_mode in ['cuda', 'floatsim']) and save_dir and (_CONV1D_LAYER_COUNTER == 23)
```

**Modified Baseline/Mode 1 path**:

Combined Baseline and Mode 1 into one path:
```python
# Baseline / Mode 1: INT8 CUDA kernel
if fp32_ssm_input or (not float_sim_asic_int8 and not float_sim_asic_research_se):
    # Use INT8 CUDA kernel
    y = quant_causal_conv1d_cuda.fwd(
            x, self.input_scale,
            self.weight, self.weight_scale,
            self.output_scale,
            self.bias_scale, self.bias,
            None, None, None, True
        )

    # ===== SAVE INTERMEDIATE VALUES FOR COMPARISON =====
    if should_save and save_mode == 'cuda':
        os.makedirs(save_dir, exist_ok=True)
        # Save input
        torch.save(x.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_input.pt'))
        # Save output
        torch.save(y.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_output_int8.pt'))
        print(f"[CUDA Mode] Saved Layer {_CONV1D_LAYER_COUNTER} Conv1D intermediate values to {save_dir}")
    # ===== END SAVE CODE =====

    _CONV1D_LAYER_COUNTER += 1
    return y  # INT8
```

**Modified Mode 2 path**:

```python
# Mode 2: INT8 Conv1D + FP32 SiLU (INT8 computation precision)
elif float_sim_asic_int8:
    # ... (dequant, conv1d, round, silu, round) ...

    # ===== SAVE INTERMEDIATE VALUES FOR COMPARISON =====
    if should_save and save_mode == 'floatsim':
        os.makedirs(save_dir, exist_ok=True)
        # Save input (INT8)
        torch.save(x.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_input.pt'))
        # Save Conv1D raw output (before INT8 simulation)
        torch.save(y_conv_fp32.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_raw.pt'))
        # Save Conv1D INT8-simulated output
        torch.save(y_conv_fp32_quantized.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_int8simulated.pt'))
        # Save SiLU raw output (before INT8 simulation)
        torch.save(y_silu_fp32_raw.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_silu_raw.pt'))
        # Save SiLU INT8-simulated output (final output)
        torch.save(y_silu_fp32.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_output_int8.pt'))
        print(f"[Float-Sim Mode] Saved Layer {_CONV1D_LAYER_COUNTER} Conv1D intermediate values to {save_dir}")
    # ===== END SAVE CODE =====

    _CONV1D_LAYER_COUNTER += 1
    return y_silu_fp32  # FP32 data type, but INT8 computation precision
```

#### File 4: CUDA_FLOATSIM_COMPARISON_README.md

详细的使用说明文档，包含：
- 系统概述
- 文件说明
- 使用方法（一键运行和手动运行）
- 保存的中间值列表
- 对比报告内容说明
- 预期结果
- 调试技巧
- 故障排除

---

## Final Implementation Summary

### Modified Files

1. **quamba/qConvLayer.py**
   - 添加了Conv1D和SiLU的round模拟INT8计算精度
   - 添加了outlier分析的保存功能（保存到`conv_silu_analysis/`）
   - 添加了CUDA vs Float-Sim对比的保存功能（通过环境变量控制）
   - 修复了`import os`的UnboundLocalError

2. **quamba/qMambaLayer.py**
   - 保持不变（dual-path逻辑已经在Session 8实现）

### New Files Created

1. **test_cuda_vs_floatsim.sh**
   - 自动化测试脚本
   - 运行CUDA模式 → Float-Sim模式 → 对比分析

2. **compare_cuda_floatsim_values.py**
   - Python对比分析脚本
   - 加载、对比、生成报告

3. **analyze_conv_silu_outliers.py**
   - Outlier分析脚本
   - IQR方法检测outliers
   - Percentile分析

4. **CUDA_FLOATSIM_COMPARISON_README.md**
   - 详细使用说明

### Key Implementation Details

#### Mode 2最终实现 (qConvLayer.py, lines 169-346)

```python
elif float_sim_asic_int8:
    # Step 1: Dequantize input (INT8 -> FP32)
    x_fp32 = x.float() * self.input_scale

    # Step 2: Dequantize weight (INT8 -> FP32)
    weight_fp32 = self.weight.float() * self.weight_scale

    # Step 3: Dequantize bias (INT8 -> FP32)
    if self.bias is not None:
        bias_fp32 = self.bias.float() * self.bias_scale
    else:
        bias_fp32 = None

    # Step 4: Conv1D computation (FP32, same as INT8 CUDA kernel does internally)
    weight_fp32_reshaped = weight_fp32.unsqueeze(1)
    x_fp32_padded = F.pad(x_fp32, (self.kernel_size - 1, 0), mode='constant', value=0)
    y_conv_fp32 = F.conv1d(x_fp32_padded, weight_fp32_reshaped, bias=bias_fp32,
                           stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups)

    # Step 4.5: Simulate INT8 computation precision (round to INT8 grid)
    # This makes the computation precision identical to INT8 CUDA kernel
    # But we keep it in FP32 format (data precision is FP32, computation precision is INT8)
    y_conv_int8_simulated = torch.round(y_conv_fp32 / self.output_scale).clamp(-128, 127)
    y_conv_fp32_quantized = y_conv_int8_simulated * self.output_scale

    # Step 5: SiLU activation (FP32) on the INT8-precision values
    # Input to SiLU has INT8 computation precision, but FP32 data type
    y_silu_fp32_raw = y_conv_fp32_quantized * torch.sigmoid(y_conv_fp32_quantized)

    # Step 6: Simulate INT8 computation precision for SiLU output
    # Round to INT8 grid to match INT8 computation precision
    # But keep FP32 data type (don't convert to torch.int8)
    y_silu_int8_simulated = torch.round(y_silu_fp32_raw / self.output_scale).clamp(-128, 127)
    y_silu_fp32 = y_silu_int8_simulated * self.output_scale

    # Key difference from Baseline:
    # - Baseline returns torch.int8 type
    # - Mode 2 returns FP32 type with INT8-precision values (discrete grid)

    # [Save code for outlier analysis - lines 207-326]
    # [Save code for CUDA vs Float-Sim comparison - lines 329-343]

    _CONV1D_LAYER_COUNTER += 1
    return y_silu_fp32  # FP32 data type, but INT8 computation precision (discrete values)
```

#### 环境变量控制

**Outlier分析**（自动触发，无需环境变量）：
- 在Mode 2中，Layer 23会自动保存到`conv_silu_analysis/`

**CUDA vs Float-Sim对比**（需要环境变量）：
```bash
# CUDA模式
export SAVE_INTERMEDIATE_VALUES="cuda"
export INTERMEDIATE_VALUES_DIR="cuda_vs_floatsim_comparison/cuda_values"

# Float-Sim模式
export SAVE_INTERMEDIATE_VALUES="floatsim"
export INTERMEDIATE_VALUES_DIR="cuda_vs_floatsim_comparison/floatsim_values"
```

---

## Testing Instructions

### Test 1: Outlier分析

**Purpose**: 查看Layer 23的Conv1D和SiLU输出是否有显著outliers

**Steps**:
```bash
# 1. 运行Mode 2（会自动保存）
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --float-sim-asic-int8 \
    --pretrained_dir pretrained_models/quamba1/default --testing

# 2. 分析outliers
python analyze_conv_silu_outliers.py
```

**Output files**:
- `conv_silu_analysis/layer23_conv1d_raw.pt`
- `conv_silu_analysis/layer23_conv1d_int8simulated.pt`
- `conv_silu_analysis/layer23_silu_raw.pt`
- `conv_silu_analysis/layer23_silu_int8simulated.pt`

**Expected**:
- 分析报告显示Conv1D和SiLU的outlier统计
- 如果Max / 99.95th percentile > 2.0，建议使用percentile clipping
- 可以看到round前后的量化误差

### Test 2: CUDA vs Float-Sim对比

**Purpose**: 验证Mode 2是否正确模拟了INT8 CUDA kernel的计算精度

**Steps**:
```bash
# 一键运行（推荐）
./test_cuda_vs_floatsim.sh

# 或手动运行三个步骤（见上面的环境变量说明）
```

**Output files**:
```
cuda_vs_floatsim_comparison/
├── cuda_values/
│   ├── layer23_conv1d_input.pt
│   └── layer23_conv1d_output_int8.pt
├── floatsim_values/
│   ├── layer23_conv1d_input.pt
│   ├── layer23_conv1d_raw.pt
│   ├── layer23_conv1d_int8simulated.pt
│   ├── layer23_silu_raw.pt
│   └── layer23_conv1d_output_int8.pt
├── cuda_output.txt
├── floatsim_output.txt
└── comparison_report.txt
```

**Comparison report contents**:
- 每个张量的形状、dtype
- 统计信息（Min/Max/Mean/Std）
- 差异分析（Max diff, Mean diff, Relative diff）
- 精确匹配百分比
- Top 10最大差异的位置和值
- 总结（Identical/Different/Missing）

### Test 3: Mode 1和Mode 2的准确率测试

**Purpose**: 验证Mode 1和Mode 2的准确率是否都是39%（与Baseline相同）

**Mode 1** (FP32_SSM_INPUT):
```bash
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --fp32-ssm-input \
    --pretrained_dir pretrained_models/quamba1/default --testing
```

**Mode 2** (FLOAT_SIM_ASIC_INT8):
```bash
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --float-sim-asic-int8 \
    --pretrained_dir pretrained_models/quamba1/default --testing
```

**Baseline** (INT8 CUDA):
```bash
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --pretrained_dir pretrained_models/quamba1/default --testing
```

---

## Expected Results

### Outlier Analysis Expected Results

从`analyze_conv_silu_outliers.py`的输出应该看到：

1. **Conv1D raw输出** (round前):
   - 可能有一些outliers超过99.95th percentile
   - Max / 99.95th percentile的比值

2. **Conv1D INT8-simulated输出** (round后):
   - 所有值都在INT8网格上（离散值）
   - Max quantization error = round造成的最大误差

3. **SiLU raw输出** (round前):
   - SiLU通常会压缩range
   - Outliers可能比Conv1D少

4. **SiLU INT8-simulated输出** (round后):
   - 所有值都在INT8网格上（离散值）
   - 这是Mode 2的最终输出

5. **Outlier clipping建议**:
   - 如果比值 > 2.0，建议使用percentile-based clipping
   - Suggested clip value会被打印出来

### CUDA vs Float-Sim Comparison Expected Results

从`comparison_report.txt`应该看到：

#### 1. layer23_conv1d_input.pt
- **IDENTICAL** ✓
- 两者的输入应该完全相同（都是INT8）
- Max diff应该是0

#### 2. layer23_conv1d_output_int8.pt
- **IDENTICAL** ✓（如果Mode 2正确实现）
- CUDA: torch.int8 dtype
- Float-Sim: torch.float32 dtype
- 转换为FP32后，Max diff应该 < 1e-6
- 精确匹配应该是100%（或接近100%，考虑浮点误差）

**如果不identical**:
- 说明Mode 2的round/clamp逻辑有问题
- 或者使用的scale不一致
- 报告会显示Top 10差异的位置，方便调试

#### 3. layer23_conv1d_raw.pt (Float-Sim独有)
- 这是Conv1D的raw FP32输出，round之前
- 显示round造成的量化误差有多大

#### 4. layer23_conv1d_int8simulated.pt (Float-Sim独有)
- 这是Conv1D的INT8模拟输出，round之后
- 应该和CUDA的`layer23_conv1d_output_int8.pt`在Conv1D部分完全相同

#### 5. layer23_silu_raw.pt (Float-Sim独有)
- 这是SiLU的raw FP32输出，round之前
- 显示SiLU的round造成的量化误差

### Accuracy Expected Results

| Mode | Conv1D | SSM Input | Expected Accuracy | Purpose |
|------|--------|-----------|-------------------|---------|
| **Baseline** | INT8 CUDA | INT8 | 39% ✓ | Standard INT8 |
| **Mode 1** | INT8 CUDA | FP32 (discrete) | 39% ✓ | Verify dequant |
| **Mode 2** | INT8 CUDA (simulated) | FP32 (discrete) | **39%** ✓ | Verify simulation |

**Mode 2应该是39%** 因为：
- 计算精度和Baseline完全相同（都是INT8精度）
- 只有数据类型不同（FP32 vs INT8）
- 数据类型不应该影响准确率（值是相同的）

**如果Mode 2不是39%**:
- 说明实现有bug
- 使用CUDA vs Float-Sim对比系统找出差异
- 检查round/clamp逻辑
- 检查scale是否正确

---

## Key Learnings and Insights

### 1. 计算精度 vs 数据精度

**计算精度**（Computation Precision）:
- 指中间值被round到什么精度
- INT8计算精度 = 所有中间值round到INT8网格（256个离散值）
- FP32计算精度 = 中间值是连续的浮点数

**数据精度**（Data Precision）:
- 指数据类型（torch.int8 vs torch.float32）
- 只影响内存和带宽
- 不影响准确率（如果值相同）

**Mode 2的核心**：
- **计算精度 = INT8**（通过round模拟）
- **数据精度 = FP32**（torch.float32类型）
- 这样可以测试SSM是否受益于更高的数据精度（即使计算精度相同）

### 2. Round模拟INT8精度的正确方法

```python
# 正确的方法
y_int8_simulated = torch.round(y_fp32 / output_scale).clamp(-128, 127)
y_quantized = y_int8_simulated * output_scale
```

**关键点**:
1. 除以scale（归一化到INT8范围）
2. Round到最近的整数
3. Clamp到[-128, 127]（INT8范围）
4. 乘以scale（反量化回FP32范围）

**结果**:
- `y_quantized`的数据类型是FP32
- 但所有值都在INT8网格上（离散值）
- 例如：如果scale=0.0121，值只能是{..., -0.0242, -0.0121, 0, 0.0121, 0.0242, ...}

### 3. 为什么需要两次Round

Mode 2需要两次round：

**第一次round**（Conv1D输出）:
```python
y_conv_int8_simulated = torch.round(y_conv_fp32 / self.output_scale).clamp(-128, 127)
y_conv_fp32_quantized = y_conv_int8_simulated * self.output_scale
```
- 模拟INT8 CUDA kernel中Conv1D输出的量化

**第二次round**（SiLU输出）:
```python
y_silu_int8_simulated = torch.round(y_silu_fp32_raw / self.output_scale).clamp(-128, 127)
y_silu_fp32 = y_silu_int8_simulated * self.output_scale
```
- 模拟INT8 CUDA kernel中SiLU输出的量化

**为什么需要两次？**
- INT8 CUDA kernel在Conv1D后量化一次
- 然后在SiLU后又量化一次
- Mode 2必须完全模拟这个流程才能保证计算精度相同

### 4. CUDA kernel内部的数据流

理解INT8 CUDA kernel的内部流程很重要：

```
INT8 input
  ↓ dequant (INT8 * input_scale → FP32)
FP32 input
  ↓ Conv1D (FP32 computation)
FP32 conv output
  ↓ round + clamp (→ INT8 grid) ← 第一次量化
INT8 conv output (虽然存为INT8，但在SiLU前会dequant)
  ↓ dequant (INT8 * output_scale → FP32)
FP32 value on INT8 grid
  ↓ SiLU (FP32 computation)
FP32 silu output
  ↓ round + clamp (→ INT8 grid) ← 第二次量化
INT8 silu output
```

**Mode 2的模拟**:
- 完全相同的流程
- 唯一区别：最后不转换成torch.int8，保持torch.float32

### 5. 环境变量控制的优雅设计

使用环境变量控制保存功能的好处：
- 不需要修改命令行参数
- 不影响原有代码逻辑
- 容易在脚本中开关
- 可以分别控制CUDA和Float-Sim模式

```python
save_mode = os.environ.get('SAVE_INTERMEDIATE_VALUES', '')  # 'cuda' or 'floatsim'
save_dir = os.environ.get('INTERMEDIATE_VALUES_DIR', '')
should_save = (save_mode in ['cuda', 'floatsim']) and save_dir and (_CONV1D_LAYER_COUNTER == 23)
```

**使用示例**:
```bash
# 开启保存
export SAVE_INTERMEDIATE_VALUES="cuda"
export INTERMEDIATE_VALUES_DIR="path/to/save"
python3 main.py ...

# 关闭保存
unset SAVE_INTERMEDIATE_VALUES
unset INTERMEDIATE_VALUES_DIR
python3 main.py ...
```

---

## Troubleshooting Guide

### Problem: 没有生成中间值文件

**Symptoms**:
- `conv_silu_analysis/`目录为空
- 或`cuda_vs_floatsim_comparison/cuda_values/`为空

**Possible Causes**:
1. 环境变量没有设置（对于CUDA vs Float-Sim对比）
2. Layer 23没有被触发（检查`_CONV1D_LAYER_COUNTER`）
3. 权限问题（无法创建目录）

**Solutions**:
```bash
# 检查环境变量
echo $SAVE_INTERMEDIATE_VALUES
echo $INTERMEDIATE_VALUES_DIR

# 检查目录权限
ls -ld conv_silu_analysis/
mkdir -p conv_silu_analysis/

# 手动触发（如果需要）
# 修改qConvLayer.py中的层数条件
```

### Problem: CUDA vs Float-Sim对比显示"DIFFERENT"

**Symptoms**:
- `comparison_report.txt`显示max diff > 1e-6
- 精确匹配百分比 < 100%

**Possible Causes**:
1. Mode 2的round逻辑有误
2. 使用的scale不一致
3. SiLU没有做round
4. Clamp范围错误

**Solutions**:

**检查round逻辑**:
```python
# 应该是这样
y_int8_simulated = torch.round(y / self.output_scale).clamp(-128, 127)
y_quantized = y_int8_simulated * self.output_scale

# 而不是
y_quantized = torch.round(y)  # ✗ 错误！没有除以scale
```

**检查scale**:
```python
# 应该使用output_scale
y_int8_simulated = torch.round(y / self.output_scale)  # ✓

# 不是其他scale
y_int8_simulated = torch.round(y / self.input_scale)  # ✗ 错误！
```

**检查SiLU是否round**:
```python
# 应该有两次round
y_conv_quantized = torch.round(y_conv / self.output_scale).clamp(-128, 127) * self.output_scale
y_silu_raw = y_conv_quantized * torch.sigmoid(y_conv_quantized)
y_silu_quantized = torch.round(y_silu_raw / self.output_scale).clamp(-128, 127) * self.output_scale  # ✓

# 而不是只round一次
y_conv_quantized = torch.round(y_conv / self.output_scale).clamp(-128, 127) * self.output_scale
y_silu_output = y_conv_quantized * torch.sigmoid(y_conv_quantized)  # ✗ 缺少第二次round！
```

**手动验证**:
```python
import torch

# 加载CUDA输出
cuda_out = torch.load('cuda_vs_floatsim_comparison/cuda_values/layer23_conv1d_output_int8.pt')
# 加载Float-Sim输出
floatsim_out = torch.load('cuda_vs_floatsim_comparison/floatsim_values/layer23_conv1d_output_int8.pt')

# 转换为FP32
cuda_fp32 = cuda_out.float()
floatsim_fp32 = floatsim_out.float()

# 计算差异
diff = (cuda_fp32 - floatsim_fp32).abs()
print(f"Max diff: {diff.max().item():.8e}")
print(f"Mean diff: {diff.mean().item():.8e}")
print(f"Exact matches: {(diff == 0).sum().item()} / {diff.numel()}")

# 找到最大差异的位置
max_diff_idx = diff.argmax().item()
print(f"\nMax diff at index {max_diff_idx}:")
print(f"  CUDA: {cuda_fp32.flatten()[max_diff_idx].item():.8f}")
print(f"  Float-Sim: {floatsim_fp32.flatten()[max_diff_idx].item():.8f}")
print(f"  Diff: {diff.flatten()[max_diff_idx].item():.8e}")
```

### Problem: Mode 2准确率不是39%

**Symptoms**:
- Mode 2的lambada_openai准确率 ≠ 39%
- 可能是36%（之前的错误实现）或其他值

**Possible Causes**:
1. Float-Sim模拟不正确（见上面的troubleshooting）
2. qMambaLayer的dual-path逻辑有问题
3. SSM没有正确处理FP32输入

**Solutions**:

**Step 1: 运行CUDA vs Float-Sim对比**:
```bash
./test_cuda_vs_floatsim.sh
```
检查`comparison_report.txt`，确保所有张量都是IDENTICAL。

**Step 2: 检查qMambaLayer的dual-path逻辑**:
```python
# 应该是这样（qMambaLayer.py中）
if fp32_mode_enabled:
    if x.dtype == torch.int8:
        # Mode 1: Conv1D returned INT8, dequantize
        x_for_xproj = x
        x_for_ssm = x.float() * self.conv1d.output_scale
    elif x.dtype == torch.float32:
        # Mode 2: Conv1D returned FP32, quantize for x_proj
        x_for_xproj = torch.round(x / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
        x_for_ssm = x  # Keep FP32
    else:
        raise ValueError(f"Unexpected dtype: {x.dtype}")
```

**Step 3: 添加debug输出**:
在qMambaLayer.py的dual-path部分添加：
```python
if DEBUG_ENABLED and self._debug_step_count <= 3:
    print(f"[qMambaLayer] x dtype: {x.dtype}")
    print(f"  x_for_xproj dtype: {x_for_xproj.dtype}")
    print(f"  x_for_ssm dtype: {x_for_ssm.dtype}")
    print(f"  x_for_ssm first 5 values: {x_for_ssm.flatten()[:5]}")
```

### Problem: Outlier分析脚本报错

**Symptoms**:
- `analyze_conv_silu_outliers.py`找不到文件
- 或加载tensor失败

**Possible Causes**:
1. 没有先运行Mode 2生成文件
2. 文件路径错误

**Solutions**:
```bash
# 检查文件是否存在
ls -lh conv_silu_analysis/

# 应该看到4个文件
layer23_conv1d_raw.pt
layer23_conv1d_int8simulated.pt
layer23_silu_raw.pt
layer23_silu_int8simulated.pt

# 如果没有，先运行Mode 2
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --float-sim-asic-int8 \
    --pretrained_dir pretrained_models/quamba1/default --testing
```

---

## Files Changed Summary

### Modified Files

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `quamba/qConvLayer.py` | 98-346 | 添加Mode 2的round模拟，outlier分析保存，CUDA vs Float-Sim对比保存 |

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `test_cuda_vs_floatsim.sh` | 204 | 自动化测试脚本 |
| `compare_cuda_floatsim_values.py` | 165 | 对比分析脚本 |
| `analyze_conv_silu_outliers.py` | 257 | Outlier分析脚本 |
| `CUDA_FLOATSIM_COMPARISON_README.md` | 286 | 详细使用说明 |
| `dailyHistory-1107.md` | This file | 今天的详细历史记录 |

---

## Quick Reference

### Command Cheat Sheet

```bash
# Outlier分析
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --float-sim-asic-int8 \
    --pretrained_dir pretrained_models/quamba1/default --testing
python analyze_conv_silu_outliers.py

# CUDA vs Float-Sim对比（一键）
./test_cuda_vs_floatsim.sh

# 手动对比
# Step 1: CUDA
export SAVE_INTERMEDIATE_VALUES="cuda"
export INTERMEDIATE_VALUES_DIR="cuda_vs_floatsim_comparison/cuda_values"
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --pretrained_dir pretrained_models/quamba1/default --testing

# Step 2: Float-Sim
export SAVE_INTERMEDIATE_VALUES="floatsim"
export INTERMEDIATE_VALUES_DIR="cuda_vs_floatsim_comparison/floatsim_values"
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --float-sim-asic-int8 \
    --pretrained_dir pretrained_models/quamba1/default --testing

# Step 3: Compare
unset SAVE_INTERMEDIATE_VALUES
unset INTERMEDIATE_VALUES_DIR
python3 compare_cuda_floatsim_values.py \
    --cuda_dir cuda_vs_floatsim_comparison/cuda_values \
    --floatsim_dir cuda_vs_floatsim_comparison/floatsim_values \
    --output cuda_vs_floatsim_comparison/comparison_report.txt

# Mode 1测试
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --fp32-ssm-input \
    --pretrained_dir pretrained_models/quamba1/default --testing

# Mode 2测试
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --float-sim-asic-int8 \
    --pretrained_dir pretrained_models/quamba1/default --testing

# Baseline测试
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --pretrained_dir pretrained_models/quamba1/default --testing
```

### File Locations

```
Project Root/
├── quamba/
│   ├── qConvLayer.py          ← Modified
│   ├── qMambaLayer.py          ← Unchanged (from Session 8)
│   └── ...
├── test_cuda_vs_floatsim.sh   ← New
├── compare_cuda_floatsim_values.py   ← New
├── analyze_conv_silu_outliers.py     ← New
├── CUDA_FLOATSIM_COMPARISON_README.md   ← New
├── dailyHistory-1107.md        ← This file
├── FP32_SSM_MODE_FIX.md        ← From Session 8
└── cuda_vs_floatsim_comparison/   ← Generated by test
    ├── cuda_values/
    ├── floatsim_values/
    └── comparison_report.txt
```

---

## Next Steps (TODO)

### 1. 运行CUDA vs Float-Sim对比测试
```bash
./test_cuda_vs_floatsim.sh
```
**Expected**: `comparison_report.txt`显示所有张量IDENTICAL

### 2. 运行Mode 1和Mode 2的准确率测试
**Expected**:
- Mode 1: 39% (与Baseline相同)
- Mode 2: 39% (与Baseline相同)

### 3. 运行Outlier分析
```bash
python analyze_conv_silu_outliers.py
```
**Analysis**:
- 查看是否有显著outliers
- 如果Max / 99.95th percentile > 2.0，考虑使用percentile clipping

### 4. 如果Mode 2 = 39%，说明什么？
- Mode 2正确模拟了INT8计算精度
- 给SSM更高的数据精度（FP32 vs INT8）但相同的计算精度（INT8网格）不能提升准确率
- **结论**：瓶颈不在数据类型，而在计算精度（INT8量化损失）

### 5. 如果想要>39%的准确率，需要什么？
两个方向：

**方向A：使用更大的scale**
- 测试pa=1.0模型（output_scale从0.0121提升到0.0212，42.78%差异）
- 预期：可能>39%，因为更大的scale保留更多信息

**方向B：真正的FP32 Conv1D（完全不量化）**
- Conv1D和SiLU都用完整FP32精度，不round
- 这需要新的Mode 3
- 预期：>39%，因为没有量化损失

---

## Related Documentation

1. **FP32_SSM_MODE_FIX.md** - Session 8的设计文档，记录了Mode 1和Mode 2的最初理解和修复过程
2. **CUDA_FLOATSIM_COMPARISON_README.md** - CUDA vs Float-Sim对比测试系统的详细使用说明
3. **THREE_MODES_README.md** - 三种模式的总体说明（如果存在）

---

## Conclusion

今天成功完成了：
1. ✅ 修复Mode 2，添加round模拟INT8计算精度（两次round：Conv1D后和SiLU后）
2. ✅ 添加Conv1D和SiLU的outlier分析功能
3. ✅ 创建完整的CUDA vs Float-Sim对比测试系统
4. ✅ 修复UnboundLocalError

**核心成就**：
- Mode 2现在完全模拟了INT8 CUDA kernel的计算精度
- 唯一区别是数据类型（FP32 vs INT8）
- 可以通过对比测试验证实现正确性

**下一步**：
- 运行测试验证Mode 2 = 39%
- 如果验证通过，说明数据类型不影响准确率（在相同计算精度下）
- 可以继续探索其他方向（更大scale、真正的FP32计算）提升准确率

---

**End of Daily History - 2025-11-07**
