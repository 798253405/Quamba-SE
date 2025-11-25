# Layer Output Comparison Tools

这套工具用于对比不同量化mode与FP32参考模型的层输出差异。

## 📋 文件说明

| 文件 | 用途 |
|------|------|
| `save_layer_outputs.py` | 主脚本：保存指定mode的第1层和最后一层输出 |
| `compare_with_fp.py` | 主脚本：对比指定mode与FP32参考的差异 |
| `save_all_modes.sh` | 批量保存所有modes的输出 |
| `compare_all_modes.sh` | 批量对比所有modes |
| `comparewithfp` | 简化命令：快速对比单个mode |

## 🚀 快速开始

### 1. 保存FP32参考输出

```bash
# 方式1：使用Python脚本
python3 save_layer_outputs.py quamba-130m-w8a8 \
    --pretrained_dir pretrained_models/Quamba1-pa9999/pa-0.9999 \
    --mode fp32 \
    --output_dir layer_outputs \
    --calib_data_num 10

# 方式2：使用Shell脚本（推荐）
./save_all_modes.sh fp_only
```

### 2. 保存特定mode的输出

```bash
# Mode 2-1
python3 save_layer_outputs.py quamba-130m-w8a8 \
    --pretrained_dir pretrained_models/Quamba1-pa9999/pa-0.9999 \
    --mode 2-1 \
    --quantize \
    --output_dir layer_outputs

# Mode 0 (Baseline)
python3 save_layer_outputs.py quamba-130m-w8a8 \
    --pretrained_dir pretrained_models/Quamba1-pa9999/pa-0.9999 \
    --mode 0 \
    --quantize \
    --output_dir layer_outputs
```

### 3. 对比mode与FP32参考

```bash
# 方式1：使用简化命令（推荐）
./comparewithfp 2-1

# 方式2：使用Python脚本
python3 compare_with_fp.py 2-1 --reference fp32 --output_dir layer_outputs
```

## 📊 批量操作

### 保存所有modes

```bash
# 保存FP32 + 所有量化modes (0, 2-0, 2-1, 2-2, 2-3, 2-4, 3)
./save_all_modes.sh

# 只保存FP32参考
./save_all_modes.sh fp_only

# 只保存量化modes
./save_all_modes.sh quant_only

# 保存关键modes (FP32, 0, 2-1, 2-2, 2-4)
./save_all_modes.sh essential

# 保存指定modes
./save_all_modes.sh fp32 0 2-1 2-4
```

### 对比所有modes

```bash
# 对比所有量化modes与FP32参考，生成汇总表格
./compare_all_modes.sh
```

## 📈 输出说明

### 1. 保存的文件结构

```
layer_outputs/
├── mode_fp32_layer_0.npy          # FP32第1层输出 (numpy array)
├── mode_fp32_layer_23.npy         # FP32最后一层输出
├── mode_fp32_stats.json           # FP32统计信息
├── mode_0_layer_0.npy             # Mode 0第1层输出
├── mode_0_layer_23.npy            # Mode 0最后一层输出
├── mode_0_stats.json              # Mode 0统计信息
├── mode_2-1_layer_0.npy
├── mode_2-1_layer_23.npy
├── mode_2-1_stats.json
└── ...
```

### 2. 对比输出指标

执行 `./comparewithfp 2-1` 会输出：

```
================================================================================
Layer 0 - Comparison Results
================================================================================

📊 FP32 Reference Statistics:
  Mean:       0.123456
  Std:        0.234567
  Range:     [-1.234567,  2.345678]

📊 Mode Output Statistics:
  Mean:       0.123450
  Std:        0.234560
  Range:     [-1.234500,  2.345600]

📏 Difference (Mode - FP32):
  Mean Diff:         0.000006
  Std Diff:          0.000007
  Mean Abs Diff:     0.012345
  Max Abs Diff:      0.123456

❌ Error Metrics:
  MSE:               1.234567e-04
  RMSE:              0.011111
  MAE:               0.012345

📈 Relative Metrics:
  Relative MSE:      1.234567e-03 ( 0.123%)
  Relative MAE:      5.678901e-03 ( 0.568%)
  Correlation:       0.999999

📊 Absolute Difference Percentiles:
  50th (Median):     0.008765
  90th:              0.023456
  95th:              0.034567
  99th:              0.056789
================================================================================
```

**指标说明：**
- **MSE (Mean Squared Error)**: 均方误差，越小越好
- **RMSE (Root Mean Squared Error)**: 均方根误差
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **Relative MSE/MAE**: 相对误差（相对于FP32的值）
- **Correlation**: 相关系数，越接近1.0越好
- **Percentiles**: 绝对差值的百分位数，用于了解误差分布

### 3. 汇总表格

执行 `./compare_all_modes.sh` 会生成：

```
================================================================================
SUMMARY TABLE - All Modes vs fp32
================================================================================

Mode       Layer 0 MSE     Layer 0 MAE  Last Layer MSE  Last Layer MAE  Avg Corr
----------------------------------------------------------------------------------------------------
0          1.234567e-04    0.012345     2.345678e-04    0.023456        0.999999
2-0        1.234568e-04    0.012346     2.345679e-04    0.023457        0.999998
2-1        2.345678e-04    0.023456     3.456789e-04    0.034567        0.999997
2-2        3.456789e-04    0.034567     4.567890e-04    0.045678        0.999996
2-3        1.234567e-04    0.012345     2.345678e-04    0.023456        0.999999
2-4        1.234568e-04    0.012346     2.345679e-04    0.023457        0.999998
3          2.345678e-04    0.023456     3.456789e-04    0.034567        0.999997
================================================================================
```

## 🔧 高级用法

### 自定义参数

```bash
# 使用更多样本
python3 save_layer_outputs.py quamba-130m-w8a8 \
    --pretrained_dir pretrained_models/Quamba1-pa9999/pa-0.9999 \
    --mode fp32 \
    --calib_data_num 100 \
    --calib_seqlen 1024

# 使用fp16作为参考
python3 compare_with_fp.py 2-1 --reference fp16

# 保存对比结果到JSON
python3 compare_with_fp.py 2-1 --save_comparison results/mode_2-1_comparison.json
```

### 修改脚本配置

编辑 `save_all_modes.sh` 修改默认参数：

```bash
PRETRAINED_DIR="pretrained_models/Quamba1-pa9999/pa-0.9999"  # 模型路径
OUTPUT_DIR="layer_outputs"                                   # 输出目录
CALIB_DATA_NUM=10                                            # 样本数量
CALIB_SEQLEN=512                                             # 序列长度
```

## 📝 使用示例

### 示例1：对比Mode 2-1的精度

```bash
# Step 1: 保存FP32参考（只需运行一次）
./save_all_modes.sh fp_only

# Step 2: 保存Mode 2-1输出
python3 save_layer_outputs.py quamba-130m-w8a8 \
    --pretrained_dir pretrained_models/Quamba1-pa9999/pa-0.9999 \
    --mode 2-1 --quantize

# Step 3: 对比
./comparewithfp 2-1
```

### 示例2：对比所有modes

```bash
# Step 1: 保存所有modes（耗时较长）
./save_all_modes.sh

# Step 2: 批量对比
./compare_all_modes.sh

# 结果会保存在 comparisons/ 目录
```

### 示例3：分析特定layer的差异

```bash
# 保存后，可以直接加载numpy文件分析
python3

>>> import numpy as np
>>> fp32_layer0 = np.load('layer_outputs/mode_fp32_layer_0.npy')
>>> mode21_layer0 = np.load('layer_outputs/mode_2-1_layer_0.npy')
>>> diff = mode21_layer0 - fp32_layer0
>>> print(f"Max diff: {np.max(np.abs(diff))}")
```

## ⚠️ 注意事项

1. **内存占用**：每个mode的输出约占用 100-500MB（取决于样本数和序列长度）
2. **首次运行**：必须先保存FP32参考，然后才能对比其他modes
3. **模型路径**：确保 `--pretrained_dir` 指向正确的模型目录
4. **样本数量**：`calib_data_num` 越大结果越准确，但耗时越长（默认10个样本已足够）

## 🎯 常见问题

**Q: 如何选择参考模式？**
- 默认使用 `fp32` 作为精度参考
- 如果没有fp32模型，可以用 `fp16` 或 mode `0` (baseline)

**Q: 为什么只保存第1层和最后一层？**
- 第1层最接近输入，能反映量化的初始影响
- 最后一层最接近输出，能反映累积误差
- 可以修改代码保存更多层

**Q: MSE多少算好？**
- < 1e-4: 非常接近FP32
- 1e-4 ~ 1e-3: 良好
- 1e-3 ~ 1e-2: 一般
- > 1e-2: 差异较大

**Q: Correlation多少算好？**
- > 0.9999: 几乎完美
- 0.999 ~ 0.9999: 优秀
- 0.99 ~ 0.999: 良好
- < 0.99: 需要关注

## 📚 相关文档

- Mode配置说明: `SSM_MODE_GUIDE.md`
- Mode统一接口: `quamba/mode_config.py`
- 使用说明: 本README

---

**版本**: 1.0
**更新**: 2025-01-10
