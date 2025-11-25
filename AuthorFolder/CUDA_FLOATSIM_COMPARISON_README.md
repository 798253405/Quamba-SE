# CUDA vs Float-Sim INT8 对比测试

## 概述

这个测试系统用于对比INT8 CUDA模式和Float-Sim INT8模式的中间值，验证Float-Sim实现是否正确模拟了CUDA kernel的计算精度。

## 文件说明

### 1. test_cuda_vs_floatsim.sh
自动化测试脚本，依次运行：
1. **CUDA模式** - 运行Baseline INT8 CUDA，保存Layer 23的所有中间值
2. **Float-Sim模式** - 运行Float-Sim INT8，保存Layer 23的所有中间值
3. **对比分析** - 比较两组中间值，找出差异

### 2. compare_cuda_floatsim_values.py
Python对比脚本，功能：
- 加载CUDA和Float-Sim的中间张量
- 逐个对比每个张量
- 计算统计差异（max diff, mean diff, relative diff）
- 显示Top 10最大差异的位置
- 生成详细对比报告

### 3. qConvLayer.py（已修改）
添加了中间值保存功能：
- 通过环境变量`SAVE_INTERMEDIATE_VALUES`控制保存模式（'cuda'或'floatsim'）
- 通过环境变量`INTERMEDIATE_VALUES_DIR`指定保存目录
- 只保存Layer 23的中间值（可修改`_CONV1D_LAYER_COUNTER == 23`来改变层数）

## 使用方法

### 一键运行（推荐）
```bash
./test_cuda_vs_floatsim.sh
```

这个脚本会自动完成所有3个步骤，最终生成对比报告。

### 手动运行

如果需要单独运行某个步骤：

#### Step 1: 运行CUDA模式
```bash
export SAVE_INTERMEDIATE_VALUES="cuda"
export INTERMEDIATE_VALUES_DIR="cuda_vs_floatsim_comparison/cuda_values"

python3 main.py quamba-130m-w8a8 \
    --quantize \
    --batch_size 16 \
    --eval_zero_shot \
    --task_list lambada_openai \
    --pretrained_dir pretrained_models/quamba1/default \
    --testing
```

#### Step 2: 运行Float-Sim模式
```bash
export SAVE_INTERMEDIATE_VALUES="floatsim"
export INTERMEDIATE_VALUES_DIR="cuda_vs_floatsim_comparison/floatsim_values"

python3 main.py quamba-130m-w8a8 \
    --quantize \
    --batch_size 16 \
    --eval_zero_shot \
    --task_list lambada_openai \
    --float-sim-asic-int8 \
    --pretrained_dir pretrained_models/quamba1/default \
    --testing
```

#### Step 3: 对比分析
```bash
python3 compare_cuda_floatsim_values.py \
    --cuda_dir cuda_vs_floatsim_comparison/cuda_values \
    --floatsim_dir cuda_vs_floatsim_comparison/floatsim_values \
    --output cuda_vs_floatsim_comparison/comparison_report.txt
```

## 保存的中间值

### CUDA模式保存（Layer 23）：
- `layer23_conv1d_input.pt` - Conv1D输入（INT8）
- `layer23_conv1d_output_int8.pt` - Conv1D输出（INT8，包含SiLU）

### Float-Sim模式保存（Layer 23）：
- `layer23_conv1d_input.pt` - Conv1D输入（INT8）
- `layer23_conv1d_raw.pt` - Conv1D原始输出（FP32，round前）
- `layer23_conv1d_int8simulated.pt` - Conv1D INT8模拟输出（FP32，round后）
- `layer23_silu_raw.pt` - SiLU原始输出（FP32，round前）
- `layer23_conv1d_output_int8.pt` - SiLU INT8模拟输出（FP32，round后）

## 对比报告内容

对比报告包含：

### 1. 基础统计
- 张量形状
- 数据类型
- Min/Max/Mean/Std

### 2. 差异分析
- 最大绝对差异
- 平均绝对差异
- 相对差异百分比
- 精确匹配的元素数量和百分比

### 3. Top 10差异
- 显示差异最大的10个位置
- 包含CUDA值、Float-Sim值、绝对差异

### 4. 总结
- 相同的张量数量
- 不同的张量数量
- 缺失的张量数量

## 预期结果

如果Float-Sim正确模拟了INT8计算精度，对比`layer23_conv1d_output_int8.pt`应该：
- **CUDA模式**：torch.int8类型
- **Float-Sim模式**：torch.float32类型，但值在INT8网格上（离散值）
- **差异**：应该完全相同（转换为FP32后比较）

如果发现差异，说明Float-Sim的round/clamp逻辑或scale使用有问题。

## 调试技巧

### 1. 检查是否保存成功
```bash
ls -lh cuda_vs_floatsim_comparison/cuda_values/
ls -lh cuda_vs_floatsim_comparison/floatsim_values/
```

### 2. 手动加载查看张量
```python
import torch
cuda_out = torch.load('cuda_vs_floatsim_comparison/cuda_values/layer23_conv1d_output_int8.pt')
floatsim_out = torch.load('cuda_vs_floatsim_comparison/floatsim_values/layer23_conv1d_output_int8.pt')

print(f"CUDA dtype: {cuda_out.dtype}, shape: {cuda_out.shape}")
print(f"Float-Sim dtype: {floatsim_out.dtype}, shape: {floatsim_out.shape}")

# 转换为FP32比较
cuda_fp32 = cuda_out.float()
floatsim_fp32 = floatsim_out.float()
diff = (cuda_fp32 - floatsim_fp32).abs()
print(f"Max diff: {diff.max().item():.8e}")
print(f"Mean diff: {diff.mean().item():.8e}")
```

### 3. 修改保存的层数
在`qConvLayer.py`中修改：
```python
should_save = (save_mode in ['cuda', 'floatsim']) and save_dir and (_CONV1D_LAYER_COUNTER == 23)
```
将`23`改为其他层数即可。

## 故障排除

### 问题：没有生成中间值文件
**解决**：检查环境变量是否正确设置
```bash
echo $SAVE_INTERMEDIATE_VALUES
echo $INTERMEDIATE_VALUES_DIR
```

### 问题：对比报告显示"MISSING"
**解决**：确保CUDA和Float-Sim模式都运行完成，检查两个目录都有文件

### 问题：差异很大
**解决**：
1. 检查Float-Sim的round逻辑（应该是`torch.round(x / scale).clamp(-128, 127) * scale`）
2. 检查使用的scale是否一致（`self.output_scale`）
3. 检查SiLU是否也做了round（Mode 2应该做了两次round：Conv1D后和SiLU后）

## 相关文件

- `quamba/qConvLayer.py` - Conv1D层实现，包含保存逻辑
- `quamba/qMambaLayer.py` - Mamba层实现
- `FP32_SSM_MODE_FIX.md` - Mode 1和Mode 2的设计文档
