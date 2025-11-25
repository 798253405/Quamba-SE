# 第一层输入输出捕获工具

## 📋 功能说明

这个工具可以捕获所有modes的第一层（Layer 0）的输入和输出，并保存到单个文件中。同时打印每个mode的：
- 前10个值
- Mean（均值）
- Std（标准差）
- Min/Max（最小/最大值）

## 🚀 快速开始

### 方式1：使用Shell脚本（推荐）

```bash
# 捕获所有modes (fp32 + 7个量化modes)
./capture_first_layer.sh

# 只捕获关键modes (fp32, 0, 2-1, 2-2, 2-4)
./capture_first_layer.sh essential

# 只捕获量化modes
./capture_first_layer.sh quant_only

# 捕获指定modes
./capture_first_layer.sh fp32 0 2-1 2-4
```

### 方式2：直接使用Python脚本

```bash
# 所有modes
python3 save_first_layer_io.py \
    --pretrained_dir pretrained_models/Quamba1-pa9999/pa-0.9999

# 指定modes
python3 save_first_layer_io.py \
    --pretrained_dir pretrained_models/Quamba1-pa9999/pa-0.9999 \
    --modes fp32 0 2-1 2-4

# 自定义输出文件和序列长度
python3 save_first_layer_io.py \
    --pretrained_dir pretrained_models/Quamba1-pa9999/pa-0.9999 \
    --output_file custom_output.npz \
    --seq_len 1024
```

## 📊 输出格式

### 屏幕输出示例

```
==================================================================================================
MODE 0
==================================================================================================

📥 INPUT (Shape: [1, 512, 768]):
  First 10 values: ['0.123456', '-0.234567', '0.345678', ...]
  Mean: 0.001234
  Std:  0.234567
  Range: [-2.345678, 3.456789]

📤 OUTPUT (Shape: [1, 512, 768]):
  First 10 values: ['0.234567', '-0.345678', '0.456789', ...]
  Mean: 0.002345
  Std:  0.345678
  Range: [-3.456789, 4.567890]

==================================================================================================
MODE 2-1
==================================================================================================

📥 INPUT (Shape: [1, 512, 768]):
  First 10 values: ['0.123450', '-0.234560', '0.345670', ...]
  Mean: 0.001230
  Std:  0.234560
  Range: [-2.345600, 3.456700]

📤 OUTPUT (Shape: [1, 512, 768]):
  First 10 values: ['0.234560', '-0.345670', '0.456780', ...]
  Mean: 0.002340
  Std:  0.345670
  Range: [-3.456700, 4.567800]
```

### 输出文件

执行后会生成两个文件：

1. **first_layer_io_all_modes.npz** - 完整数据（numpy压缩格式）
   - 包含所有modes的完整输入/输出数组
   - 可以用numpy加载：`data = np.load('first_layer_io_all_modes.npz')`

2. **first_layer_io_all_modes_stats.json** - 统计信息（JSON格式）
   ```json
   {
       "timestamp": "2025-01-10 16:00:00",
       "model": "quamba-130m-w8a8",
       "pretrained_dir": "pretrained_models/Quamba1-pa9999/pa-0.9999",
       "seq_len": 512,
       "modes": {
           "0": {
               "input": {
                   "shape": [1, 512, 768],
                   "first_10": [0.123456, -0.234567, ...],
                   "mean": 0.001234,
                   "std": 0.234567,
                   "min": -2.345678,
                   "max": 3.456789
               },
               "output": {
                   "shape": [1, 512, 768],
                   "first_10": [0.234567, -0.345678, ...],
                   "mean": 0.002345,
                   "std": 0.345678,
                   "min": -3.456789,
                   "max": 4.567890
               }
           },
           "2-1": {...}
       }
   }
   ```

## 📖 数据加载和分析

### 加载NPZ文件

```python
import numpy as np

# 加载数据
data = np.load('first_layer_io_all_modes.npz')

# 查看所有keys
print("Available keys:", data.files)

# 获取特定mode的输入/输出
mode0_input = data['mode_0_input']    # Shape: [1, 512, 768]
mode0_output = data['mode_0_output']

mode21_input = data['mode_2-1_input']
mode21_output = data['mode_2-1_output']

# 计算差异
diff = mode21_output - mode0_output
print(f"Max difference: {np.max(np.abs(diff))}")
```

### 加载JSON统计

```python
import json

# 加载统计
with open('first_layer_io_all_modes_stats.json', 'r') as f:
    stats = json.load(f)

# 查看Mode 0的统计
mode0_stats = stats['modes']['0']
print(f"Mode 0 input mean: {mode0_stats['input']['mean']}")
print(f"Mode 0 output mean: {mode0_stats['output']['mean']}")

# 对比所有modes的输出均值
for mode, data in stats['modes'].items():
    print(f"Mode {mode}: output mean = {data['output']['mean']:.6f}")
```

### 对比分析示例

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.load('first_layer_io_all_modes.npz')

# 对比Mode 0和Mode 2-1的输出
fp32_output = data['mode_fp32_output']
mode0_output = data['mode_0_output']
mode21_output = data['mode_2-1_output']

# 计算差异
diff_0 = mode0_output - fp32_output
diff_21 = mode21_output - fp32_output

# 打印统计
print(f"Mode 0 vs FP32:")
print(f"  MSE: {np.mean(diff_0**2):.6e}")
print(f"  MAE: {np.mean(np.abs(diff_0)):.6f}")

print(f"\nMode 2-1 vs FP32:")
print(f"  MSE: {np.mean(diff_21**2):.6e}")
print(f"  MAE: {np.mean(np.abs(diff_21)):.6f}")

# 绘制差异分布
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(diff_0.flatten(), bins=100, alpha=0.7)
plt.title('Mode 0 - FP32')
plt.xlabel('Difference')

plt.subplot(1, 2, 2)
plt.hist(diff_21.flatten(), bins=100, alpha=0.7)
plt.title('Mode 2-1 - FP32')
plt.xlabel('Difference')

plt.tight_layout()
plt.savefig('first_layer_output_diff.png')
```

## 🔧 参数说明

### Python脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称 | `quamba-130m-w8a8` |
| `--pretrained_dir` | 预训练模型路径 | `pretrained_models/Quamba1-pa9999/pa-0.9999` |
| `--modes` | 要测试的modes | `fp32 0 2-0 2-1 2-2 2-3 2-4 3` |
| `--output_file` | 输出文件名 | `first_layer_io_all_modes.npz` |
| `--seq_len` | 序列长度 | `512` |

### Shell脚本选项

```bash
./capture_first_layer.sh              # 所有modes
./capture_first_layer.sh essential    # 关键modes
./capture_first_layer.sh quant_only   # 仅量化modes
./capture_first_layer.sh fp32 0 2-1   # 自定义modes
```

## 📈 应用场景

1. **验证量化精度**
   - 对比量化mode与FP32的第一层输出差异
   - 识别量化引入的误差大小

2. **调试模型行为**
   - 检查不同modes的输入是否一致
   - 分析输出分布的变化

3. **数值稳定性分析**
   - 检查是否存在数值溢出/下溢
   - 分析激活值的范围

4. **Mode对比研究**
   - 对比不同量化策略的影响
   - 评估INT8 vs FP32的差异

## ⚠️ 注意事项

1. **内存占用**：每个mode约占用 150MB（取决于序列长度和模型大小）
2. **运行时间**：捕获所有8个modes约需要 10-15分钟
3. **单一样本**：使用单个样本进行测试，结果可能因样本而异
4. **第一层特殊性**：第一层最接近输入，能反映量化的初始影响

## 🎯 最佳实践

1. **先运行essential模式**
   ```bash
   ./capture_first_layer.sh essential
   ```

2. **检查关键modes的差异**
   - Mode 0 vs FP32：评估baseline量化精度
   - Mode 2-1 vs Mode 0：评估PyTorch实现准确性
   - Mode 2-2 vs Mode 2-1：评估FP32 SSM的改进

3. **使用Python进行深度分析**
   ```python
   # 加载并分析
   data = np.load('first_layer_io_all_modes.npz')
   # ... 进行自定义分析
   ```

## 📚 相关工具

- `save_layer_outputs.py` - 保存多层输出（第1层和最后一层）
- `compare_with_fp.py` - 对比层输出与FP32参考
- `./QUICK_RUN.sh` - 运行完整评估测试

---

**版本**: 1.0
**更新**: 2025-01-10
