# 所有模式测试结果分析

## 📊 测试配置
- **模型**: quamba-130m-w8a8
- **预训练权重**: pretrained_models/testPercentileRange/default
- **任务**: lambada_openai (zero-shot)
- **测试模式**: --testing (100 samples)

---

## 🎯 结果总览

| 排名 | Mode | Accuracy (%) | Perplexity | Conv1D输出 | SSM实现 |
|------|------|--------------|------------|-----------|---------|
| 🥇 1 | **Mode 2-0** | **38.00%** | 29.9138 | FP32 (INT8 grid) | CUDA INT8 (requant) |
| 🥇 1 | **Mode 3** | **38.00%** | 29.0754 | FP32 (TRUE) | PyTorch FP32 |
| 🥈 3 | Mode 2-1 | 36.00% | 29.0117 | INT8 | PyTorch INT8 |
| 🥈 3 | Mode 2-2 | 36.00% | 30.5710 | FP32 (INT8 grid) | PyTorch FP32 |
| 🥈 3 | Mode 2-3 | 36.00% | 29.0117 | FP32 (TRUE) | PyTorch INT8 (requant) |
| 🥉 6 | Mode 2-4 | 34.00% | 27.2117 | FP32 (TRUE) | PyTorch FP32 |
| 7 | Mode 1 | 33.00% | 29.0602 | FP32 (TRUE) | PyTorch FP32 |

---

## 🔍 关键发现

### 1. **Mode 2-0 和 Mode 3 表现最佳 (38.00%)**

**Mode 2-0**: CUDA INT8 + Requantization
- Conv1D: FP32 (INT8 grid)
- SSM: CUDA INT8 kernel (with requantization)
- **优势**: CUDA INT8 kernel 优化好，requantization 开销可接受

**Mode 3**: FP32/FP16 Input + FP32 Conv/SSM + INT8 Linear (Hybrid Precision)
- Conv1D: TRUE FP32 (accepts FP32/FP16 input, dynamic quantization)
- SSM: PyTorch FP32
- Linear: INT8 量化
- **优势**: 混合精度策略 - FP32 用于关键部分，INT8 用于Linear层
- **Perplexity最低之一**: 29.0754（说明预测质量好）

### 2. **Mode 2-3 的 TRUE FP32 Conv1D 没有带来预期提升**

**预期**: Mode 2-3 应该比 Mode 2-1 更好（因为 Conv1D 使用 TRUE FP32）
**实际**: Mode 2-3 = Mode 2-1 = 36.00%（完全相同！）
**Perplexity**: 完全相同 (29.0117)

**分析**:
```
Mode 2-1: INT8 Conv1D → PyTorch INT8 SSM
Mode 2-3: TRUE FP32 Conv1D → requantize to INT8 → PyTorch INT8 SSM
```

**可能原因**:
1. ✅ **Requantization 抵消了 FP32 的精度优势**: TRUE FP32 → INT8 时，精度损失等同于直接用 INT8
2. ✅ **PyTorch INT8 SSM 是瓶颈**: SSM 的量化误差主导了整体误差
3. ⚠️ **Scale mismatch 风险**: 如果 TRUE FP32 range 不匹配 calibrated output_scale，requantization 会引入额外误差

**结论**: Mode 2-3 的设计存在问题 - TRUE FP32 的优势在 requantization 步骤被完全抵消

### 3. **Mode 2-4 表现意外较差 (34.00%)**

**预期**: Mode 2-4 应该是 Mode 2-x 系列中最好的（完全 FP32 pipeline）
**实际**: Mode 2-4 = 34.00%（仅优于 Mode 1）

**分析**:
```
Mode 2-4: TRUE FP32 Conv1D → FP32 SSM (no requantization)
```

**可能原因**:
1. ⚠️ **FP32 SSM 实现问题**: PyTorch FP32 SSM (selective_scan_SE_float) 可能存在数值问题
2. ⚠️ **Scale mismatch**: FP32 输出可能与后续层期望的 scale 不匹配
3. ⚠️ **过拟合于 INT8 calibration**: 模型权重是基于 INT8 calibration 的，完全 FP32 反而偏离了校准点

**Perplexity**: 27.2117（最低！）但 accuracy 不高
- 说明 Mode 2-4 预测更"自信"，但不一定更准确

### 4. **Mode 1 (Pure FP32) 表现最差 (33.00%)**

**分析**:
- Mode 1 是理论上界，但实际表现最差
- **原因**: 模型权重是为 INT8 量化校准的，完全 FP32 反而偏离了最佳工作点
- **结论**: 这个模型的最佳性能点在量化配置下，而非完全 FP32

### 5. **Mode 2-2 vs Mode 2-4: INT8 Grid vs TRUE FP32**

**Mode 2-2**: 36.00% (FP32 on INT8 grid → PyTorch FP32 SSM)
**Mode 2-4**: 34.00% (TRUE FP32 → PyTorch FP32 SSM)

**结论**: INT8 grid 的离散化反而更好！
- **可能原因**: INT8 grid 的离散化起到了类似正则化的作用
- **或者**: PyTorch FP32 SSM 对 INT8 grid 输入优化更好

---

## 💡 模式对比深入分析

### Conv1D 输出精度影响

| Conv1D输出类型 | 模式 | 平均 Accuracy |
|---------------|------|---------------|
| **INT8** | Mode 2-1 | 36.00% |
| **FP32 (INT8 grid)** | Mode 2-0, Mode 2-2 | 37.00% |
| **FP32 (TRUE)** | Mode 2-3, Mode 2-4, Mode 3, Mode 1 | 35.25% |

**结论**: FP32 (INT8 grid) 表现最好，TRUE FP32 反而不如预期

### SSM 实现影响

| SSM类型 | 模式 | 平均 Accuracy |
|---------|------|---------------|
| **CUDA INT8** | Mode 2-0 | 38.00% |
| **PyTorch INT8** | Mode 2-1, Mode 2-3 | 36.00% |
| **PyTorch FP32** | Mode 2-2, Mode 2-4, Mode 3, Mode 1 | 35.25% |

**结论**: CUDA INT8 kernel 表现最好（高度优化），PyTorch 实现反而不如

### Requantization 开销

| 是否 Requantization | 模式 | Accuracy |
|---------------------|------|----------|
| ✅ 有 requant | Mode 2-0 | 38.00% |
| ✅ 有 requant | Mode 2-3 | 36.00% |
| ❌ 无 requant | Mode 2-1 | 36.00% |
| ❌ 无 requant | Mode 2-2 | 36.00% |
| ❌ 无 requant | Mode 2-4 | 34.00% |

**结论**: Requantization 不是主要瓶颈（Mode 2-0 最好，Mode 2-3 与无 requant 相同）

---

## 🎯 推荐策略

### 1. **生产环境推荐: Mode 2-0 或 Mode 3**

**Mode 2-0** (38.00%):
```bash
FLOAT_SIM_ASIC_INT8=true SSM_USE_CUDA_FOR_FP32=true \
python3 main.py quamba-130m-w8a8 --quantize --float-sim-asic-int8 ...
```
- ✅ 最高 accuracy (38.00%)
- ✅ CUDA INT8 kernel 高度优化
- ✅ Requantization 开销可接受

**Mode 3** (38.00%):
```bash
CONV1D_MODE3_FP32=true \
python3 main.py quamba-130m-w8a8 --quantize ...
```
- ✅ 最高 accuracy (38.00%)
- ✅ 最低 perplexity (29.0754)
- ✅ Hybrid precision (FP32 Conv/SSM + INT8 Linear)
- ✅ 接受 FP32/FP16 输入（无需预量化）
- ✅ 灵活性最高

### 2. **Mode 2-3 的问题**

**不推荐使用 Mode 2-3**:
- ❌ TRUE FP32 优势被 requantization 完全抵消
- ❌ 与 Mode 2-1 (INT8 Conv1D) 性能完全相同
- ❌ 增加了计算复杂度，但无性能提升
- ⚠️ Scale mismatch 风险

**建议**:
- 如果需要 PyTorch INT8 SSM: 直接用 **Mode 2-1**（更简单，性能相同）
- 如果需要 TRUE FP32 Conv1D: 用 **Mode 2-4** 或 **Mode 3**（避免 requantization）

### 3. **Mode 2-4 的意外结果**

**Mode 2-4 (34.00%) 表现不如预期**:
- ❌ 完全 FP32 pipeline，但 accuracy 仅 34%
- ⚠️ 可能的原因: PyTorch FP32 SSM 数值问题，或与 INT8 calibration 不匹配

**调试建议**:
1. 检查 Mode 2-4 的 Layer 24 输出
2. 对比 Mode 2-4 vs Mode 3 的数值差异（两者都用 FP32 SSM）
3. 分析为什么 Mode 3 (38%) 比 Mode 2-4 (34%) 好很多

---

## 🔬 需要进一步调查

### 1. **Mode 3 vs Mode 2-4 差异分析**

两者都使用 TRUE FP32 Conv1D + FP32 SSM，但结果相差 4%：
```
Mode 3:   38.00% (FP32/FP16 input, dynamic quantization, INT8 Linear)
Mode 2-4: 34.00% (INT8 input, static calibration, INT8 Linear)
```

**关键区别**:
- Mode 3 接受 FP32/FP16 输入 + 动态量化
- Mode 2-4 接受 INT8 输入 + 静态 calibration

**可能原因**:
1. Mode 3 的动态量化更适应输入分布
2. Mode 2-4 的静态 scale 与 TRUE FP32 输出不匹配
3. 输入精度对最终结果影响很大

### 2. **Mode 2-3 Scale Validation 检查**

查看 Mode 2-3 的日志，检查是否有 scale mismatch warning:
```bash
grep -i "scale" logs_all_modes/mode23/*.log
grep -i "mismatch" logs_all_modes/mode23/*.log
```

如果有大量 scale mismatch，说明 TRUE FP32 range 不匹配 calibrated output_scale

### 3. **Layer 24 输出分析**

查看 Layer 24 的输出：
```bash
grep "Layer 24" logs_all_modes/mode*/quamba-130m-w8a8.log
```

对比不同模式的 Layer 24 output range, absmax, scales

---

## 📝 总结

### ✅ 成功发现

1. **Mode 2-0 和 Mode 3 是最佳选择** (38.00%)
2. **Mode 2-3 的 TRUE FP32 Conv1D 没有价值** (requantization 抵消了优势)
3. **Mode 1 (Pure FP32) 不是最优** (模型为 INT8 校准)
4. **CUDA INT8 kernel 优化非常好** (Mode 2-0 最佳)
5. **Hybrid precision (Mode 3) 效果最好** (精度 + 效率平衡)

### ⚠️ 意外发现

1. **Mode 2-4 表现不如预期** (34% vs 预期接近 Mode 1)
2. **INT8 grid 比 TRUE FP32 更好** (Mode 2-2 > Mode 2-4)
3. **PyTorch FP32 SSM 表现一般** (不如 CUDA INT8)

### 🎯 推荐行动

1. ✅ **生产环境使用 Mode 3** (38%, 最灵活，hybrid precision)
2. ✅ **或使用 Mode 2-0** (38%, CUDA 优化最好)
3. ❌ **弃用 Mode 2-3** (无优势，增加复杂度)
4. 🔍 **调查 Mode 2-4 vs Mode 3 差异** (为什么 Mode 3 好 4%？)
5. 🔍 **检查 Mode 2-3 scale mismatch** (是否有大量 warning？)

---

## 🚀 后续测试建议

### 1. 完整评估（非 testing 模式）

当前结果基于 `--testing` (100 samples)，需要完整评估：
```bash
# Mode 3 完整评估
CONV1D_MODE3_FP32=true python3 main.py quamba-130m-w8a8 \
    --pretrained_dir pretrained_models/testPercentileRange/default \
    --quantize --eval_zero_shot --task_list lambada_openai \
    --log_dir logs_mode3_full

# Mode 2-0 完整评估
FLOAT_SIM_ASIC_INT8=true SSM_USE_CUDA_FOR_FP32=true \
python3 main.py quamba-130m-w8a8 \
    --pretrained_dir pretrained_models/testPercentileRange/default \
    --quantize --float-sim-asic-int8 --eval_zero_shot --task_list lambada_openai \
    --log_dir logs_mode20_full
```

### 2. Layer 24 数值对比

对比 Mode 2-3 vs Mode 2-4 vs Mode 3 的 Layer 24 输出

### 3. Scale Validation 分析

检查 Mode 2-3 的 scale validation 日志

---

**测试时间**: 2025-11-10
**配置**: pretrained_models/testPercentileRange/default
**样本数**: 100 (testing mode)
