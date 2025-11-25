# Four-Mode Comparison Test

## 概述

这个脚本自动测试四种模式并比较它们的accuracy：

1. **Baseline (INT8 CUDA)**: 原始Quamba INT8实现，使用优化的CUDA kernel
2. **Mode 2 (Float Sim INT8)**: PyTorch模拟INT8量化行为（验证实现正确性）
3. **Mode 1 (FP32 Upper Bound)**: SSM输入保持FP32，不量化（理论精度上限）
4. **Mode 3 (Scale Enhancement)**: 双尺度量化，inlier用scale1，outlier用scale2（研究方法）

## 使用方法

### 快速测试（推荐，约10-30分钟）
使用 `--testing` flag，只跑100个样本：
```bash
./test_four_modes.sh
```

### 完整评估（数小时）
跑完整数据集：
```bash
./test_four_modes.sh --full
```

## 输出

### 1. 终端输出
脚本运行时会显示进度：
```
=================================
Four-Mode Comparison Test
Test mode: quick (100 samples)
=================================

[1/4] Running Baseline (INT8 CUDA kernel)...
✓ Baseline completed in 45s
  Accuracy: 0.6234

[2/4] Running Mode 2 (Float Sim INT8)...
✓ Mode 2 completed in 523s
  Accuracy: 0.6234
  Diff from baseline: 0.000000

[3/4] Running Mode 1 (FP32 SSM Input)...
✓ Mode 1 completed in 518s
  Accuracy: 0.6456
  Improvement: +0.0222 (+3.56%)

[4/4] Running Mode 3 (Scale Enhancement)...
✓ Mode 3 completed in 521s
  Accuracy: 0.6345
  Improvement: +0.0111 (+1.78%)

=================================
Test Complete!
=================================
```

### 2. 结果文件 `four_modes_results.txt`
包含详细的结果和分析：
```
================================================================================
Four-Mode Comparison Test Results
================================================================================
Date: 2025-11-07 12:45:32
Model: quamba-130m-w8a8
Task: lambada_openai
Test mode: quick (100 samples)
================================================================================

1. Baseline (INT8 CUDA kernel)
   - Description: Original Quamba INT8 implementation with CUDA kernel
   - Accuracy: 0.6234
   - Time: 45s
   - Purpose: Reference baseline for comparison

2. Mode 2 (Float Sim INT8 - Verification)
   - Description: PyTorch simulation of INT8 quantization behavior
   - Accuracy: 0.6234
   - Time: 523s
   - Diff from baseline: 0.000000
   - Purpose: Verify PyTorch simulation matches CUDA implementation
   - Expected: Should be identical to baseline (diff ≈ 0)

3. Mode 1 (FP32 SSM Input - Upper Bound)
   - Description: SSM input kept in FP32 without quantization
   - Accuracy: 0.6456
   - Time: 518s
   - Improvement over baseline: +0.0222 (+3.56%)
   - Purpose: Theoretical upper bound of precision improvement
   - Expected: Should be better than baseline

4. Mode 3 (Scale Enhancement - Research)
   - Description: Dual-scale quantization (scale1 for inliers, scale2 for outliers)
   - Scale factor: 2025.0
   - Accuracy: 0.6345
   - Time: 521s
   - Improvement over baseline: +0.0111 (+1.78%)
   - Purpose: Research approach for handling outliers
   - Expected: Should be between baseline and Mode 1

================================================================================
SUMMARY
================================================================================

Baseline (INT8):           0.6234
Mode 2 (Verification):     0.6234  (diff: 0.000000)
Mode 1 (FP32 Upper Bound): 0.6456
Mode 3 (Scale Enhancement):0.6345

Key Findings:
- Mode 2 should match baseline exactly (verification of implementation)
- Mode 1 shows theoretical upper bound of precision improvement
- Mode 3 explores dual-scale quantization approach
```

### 3. 详细日志文件
- `baseline_output.log`: Baseline运行的完整输出
- `mode1_output.log`: Mode 1运行的完整输出
- `mode2_output.log`: Mode 2运行的完整输出
- `mode3_output.log`: Mode 3运行的完整输出

## 预期结果

### Mode 2验证
**关键**：Mode 2应该与Baseline**完全一致**（diff ≈ 0），这验证了PyTorch实现的正确性。

如果Mode 2与Baseline有差异：
- 小差异（< 0.001）：可能是浮点精度误差，可接受
- 大差异（> 0.01）：说明实现有问题，需要检查

### Mode 1上限
Mode 1应该比Baseline好，显示精度改进的理论上限。

### Mode 3研究
Mode 3应该介于Baseline和Mode 1之间，验证dual-scale方法的有效性。

## 速度说明

| 模式 | 实现方式 | 相对速度 |
|------|---------|---------|
| Baseline | CUDA kernel | 1x (最快) |
| Mode 1/2/3 | PyTorch循环 | ~10-50x 慢 |

**为什么慢？**
- Baseline使用高度优化的CUDA kernel，并行计算
- Mode 1/2/3使用纯PyTorch实现，逐时间步循环计算
- 这是研究代码，优先验证正确性而非速度

## 自定义配置

编辑 `test_four_modes.sh` 文件中的配置：

```bash
# 修改模型
MODEL="quamba-130m-w8a8"

# 修改任务
TASK="lambada_openai"

# 修改batch size
BATCH_SIZE=16

# 修改scale factor（Mode 3）
# 在Mode 3测试部分找到：
--float-sim-scale-factor 2025.0
```

## 故障排除

### 1. 权限错误
```bash
chmod +x test_four_modes.sh
```

### 2. 模型路径错误
检查 `PRETRAINED_DIR` 是否正确：
```bash
PRETRAINED_DIR="pretrained_models/quamba1/default"
```

### 3. 内存不足
降低batch size：
```bash
BATCH_SIZE=8  # 或 4
```

### 4. 某个模式失败
查看对应的log文件：
```bash
cat mode1_output.log  # 查看Mode 1的错误
```

## 结果解读

### 成功的测试应该显示：

✅ **Mode 2 ≈ Baseline**: 验证实现正确
✅ **Mode 1 > Baseline**: 有精度改进空间
✅ **Baseline < Mode 3 < Mode 1**: Dual-scale有效

### 如果结果异常：

❌ **Mode 2 ≠ Baseline**: 检查实现，可能有bug
❌ **Mode 1 ≤ Baseline**: 不太可能，检查数据
❌ **Mode 3 < Baseline**: Dual-scale可能有问题

## 下一步

1. **验证通过（Mode 2 = Baseline）**：
   - 分析Mode 1和Mode 3的改进幅度
   - 决定是否值得优化为CUDA kernel

2. **验证失败（Mode 2 ≠ Baseline）**：
   - 检查selective_scan_SE实现
   - 对比CUDA kernel和PyTorch实现的差异

3. **研究发现（Mode 3有改进）**：
   - 优化dual-scale实现
   - 考虑实现为高效的CUDA kernel
   - 调整scale_factor参数

## 快速命令参考

```bash
# 快速测试（推荐）
./test_four_modes.sh

# 完整评估
./test_four_modes.sh --full

# 查看结果
cat four_modes_results.txt

# 查看特定模式的详细日志
cat mode1_output.log
cat mode2_output.log
cat mode3_output.log
```
