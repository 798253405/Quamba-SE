# 所有模式完整路径对比

## 总览表

| Mode | Conv1D 输出 | SSM 输入 | SSM 实现 | 环境变量 |
|------|------------|----------|---------|---------|
| **Mode 0** | INT8 | INT8 | CUDA INT8 | (默认) |
| **Mode 2-0** | FP32 (INT8 grid) | INT8 (requantize) | CUDA INT8 | `FLOAT_SIM_ASIC_INT8=true SSM_USE_CUDA_FOR_FP32=true` |
| **Mode 2-1** | INT8 | INT8 (direct) | PyTorch INT8 | `FLOAT_SIM_ASIC_INT8=true SSM_USE_PYTORCH_INT8=true` |
| **Mode 2-2** | FP32 (INT8 grid) | FP32 (INT8 grid) | PyTorch FP32 | `FLOAT_SIM_ASIC_INT8=true` |
| **Mode 2-3** ✅ | FP32 (TRUE) | INT8 (requantize) | PyTorch INT8 | `FLOAT_SIM_ASIC_INT8=true CONV1D_MODE23_FP32=true` |
| **Mode 2-4** ✨ | FP32 (TRUE) | FP32 (TRUE) | PyTorch FP32 | `FLOAT_SIM_ASIC_INT8=true CONV1D_MODE24_FP32=true` |
| **Mode 3** 🌟 | FP32 (TRUE) | FP32 (TRUE) | PyTorch FP32 + INT8 Linear | `CONV1D_MODE3_FP32=true` |
| **Mode 1** | FP32 (TRUE) | FP32 (TRUE) | PyTorch FP32 | `FP32_SSM_INPUT=true` |

---

## 详细路径

### **Mode 0: Baseline INT8 CUDA**

```
环境变量: (无)

Conv1D:
  📁 quamba/qConvLayer.py:112-148
  🔧 quant_causal_conv1d_cuda.fwd() → CUDA INT8 kernel
  📊 输入: INT8 → 输出: INT8

SSM:
  📁 quamba/qSelectiveScan.py:325-373
  🔧 quant_selective_scan_fn() → CUDA INT8 kernel
  📊 输入: INT8 → 输出: INT8 → half

精度: INT8 computation (256 discrete values)
```

---

### **Mode 2-0: CUDA INT8 + Requantization**

```
环境变量: FLOAT_SIM_ASIC_INT8=true SSM_USE_CUDA_FOR_FP32=true

Conv1D:
  📁 quamba/qConvLayer.py:150-199
  🔧 quant_causal_conv1d_cuda.fwd() → INT8 kernel
       → y_int8.float() * output_scale
  📊 输入: INT8 → CUDA计算: FP32 → 量化到INT8 → 反量化到FP32
  ⚠️ 输出: FP32 (on INT8 grid - 256 discrete values)

SSM:
  📁 quamba/qSelectiveScan.py:285-291
  📁 quamba/SoftEdgeSSM.py:28-99
  🔧 execute_mode_20_cuda_int8_requant()
       → FP32 requantize to INT8
       → quant_selective_scan_fn() (CUDA INT8 kernel)
  📊 输入: FP32 (INT8 grid) → Requantize to INT8 → CUDA INT8 → FP32 → half

测试: Requantization 是否影响精度
```

---

### **Mode 2-1: PyTorch INT8 Direct**

```
环境变量: FLOAT_SIM_ASIC_INT8=true SSM_USE_PYTORCH_INT8=true

Conv1D:
  📁 quamba/qConvLayer.py:112-148 (同 Mode 0)
  🔧 quant_causal_conv1d_cuda.fwd() → CUDA INT8 kernel
  📊 输入: INT8 → 输出: INT8 (no dequantization)

SSM:
  📁 quamba/qSelectiveScan.py:293-299
  📁 quamba/SoftEdgeSSM.py:102-213
  🔧 execute_mode_21_pytorch_int8_direct()
       → selective_scan_SE_int8Torch() (PyTorch INT8)
  📊 输入: INT8 (direct pass, no requantization) → PyTorch INT8 → FP32 → half

测试: PyTorch INT8 vs CUDA INT8 实现差异
```

---

### **Mode 2-2: FP32 PyTorch (INT8 Grid)**

```
环境变量: FLOAT_SIM_ASIC_INT8=true

Conv1D:
  📁 quamba/qConvLayer.py:150-199
  🔧 quant_causal_conv1d_cuda.fwd() → INT8 kernel
       → y_int8.float() * output_scale
  📊 输入: INT8 → CUDA计算: FP32 → 量化到INT8 → 反量化到FP32
  ⚠️ 输出: FP32 (on INT8 grid - 256 discrete values)

SSM:
  📁 quamba/qSelectiveScan.py:318-332
  📁 quamba/SoftEdgeSSM.py:344-419 (line 387-396)
  🔧 execute_fp32_modes('mode22_fp32_replicates_mode21')
       → selective_scan_SE_mode22_fp32_replicates_mode21()
  📊 输入: FP32 (on INT8 grid) → 完全FP32计算 → FP32 → half

特点: SSM 用 FP32 复制 Mode 2-1 的逻辑
当前最佳性能: ✅ (对于 INT8 grid 输入)
```

---

### **Mode 2-3: TRUE FP32 Conv1D + PyTorch INT8 SSM** ⭐ (修正后)

```
环境变量: FLOAT_SIM_ASIC_INT8=true CONV1D_MODE23_FP32=true

Conv1D:
  📁 quamba/qConvLayer.py:201-230
  🔧 quant_causal_conv1d_cuda.fwd_fp32() → NEW FP32 CUDA kernel
  📁 csrc/causal_conv1d/quant_causal_conv1d_fwd_fp32_kernel.cuh

  关键代码 (line 155-157):
    float out_vals_store[kNElts];
    for (int i = 0; i < kNElts; ++i) {
        out_vals_store[i] = out_vals[i];  // 不量化！
    }

  📊 输入: INT8 → CUDA计算: FP32 (完整精度) → 输出: FP32 (TRUE continuous)
  ✅ 跳过量化步骤 (保留 CUDA 内部 FP32 精度)

SSM: (✅ 修正后的路由)
  📁 quamba/qSelectiveScan.py:274-280
  🔧 execute_mode_21_legacy_pytorch_int8_requant()
  📁 quamba/SoftEdgeSSM.py:216-342

  步骤:
    1. 接收: FP32 (TRUE continuous values)
    2. Requantize: torch.round(u / u_scale).clamp(-128, 127).to(torch.int8)
    3. 计算: selective_scan_SE_int8Torch() (PyTorch INT8)
    4. 输出: FP32 → half

  📊 输入: FP32 (TRUE) → Requantize to INT8 → PyTorch INT8 → FP32 → half

测试目的: Conv1D 的高精度 FP32 输出是否能改善最终结果
预期性能: 与 Mode 2-1 legacy 相当或更好
```

---

### **Mode 2-4: TRUE FP32 Conv1D + PyTorch FP32 SSM** ✨ (完全FP32)

```
环境变量: FLOAT_SIM_ASIC_INT8=true CONV1D_MODE24_FP32=true

Conv1D:
  📁 quamba/qConvLayer.py:301-329
  🔧 quant_causal_conv1d_cuda.fwd_fp32() → FP32 CUDA kernel
  📁 csrc/causal_conv1d/quant_causal_conv1d_fwd_fp32_kernel.cuh

  关键代码 (line 155-157):
    float out_vals_store[kNElts];
    for (int i = 0; i < kNElts; ++i) {
        out_vals_store[i] = out_vals[i];  // 不量化！
    }

  📊 输入: INT8 → CUDA计算: FP32 (完整精度) → 输出: FP32 (TRUE continuous)
  ✅ 跳过量化步骤 (保留 CUDA 内部 FP32 精度)

SSM:
  📁 quamba/qSelectiveScan.py:277-284
  🔧 execute_fp32_modes('fp32_upper_bound')
  📁 quamba/SoftEdgeSSM.py:344-419 (line 378-386)

  步骤:
    1. 接收: FP32 (TRUE continuous values)
    2. 直接使用: 不需要 requantization！
    3. 计算: selective_scan_SE_float() (PyTorch FP32)
    4. 输出: FP32 → half

  📊 输入: FP32 (TRUE) → 直接使用 → PyTorch FP32 → FP32 → half

测试目的: 完全 FP32 流程，测试 Conv1D FP32 的真正上界
预期性能: 应该是所有 Mode 2-x 中最好的，接近 Mode 1

关键特征:
  ✅ No scale mismatch issues (不需要 requantization)
  ✅ No quantization error in Conv1D
  ✅ No quantization error in SSM
  ✅ Complete FP32 pipeline: Conv1D → SSM
```

---

### **Mode 3: FP32/FP16 Input + FP32 Conv/SSM + INT8 Linear** 🌟 (Hybrid Precision)

```
环境变量: CONV1D_MODE3_FP32=true

输入特征:
  📊 模型输入: FP32 或 FP16 (完整精度)
  ⚡ 动态量化: 运行时将 FP32/FP16 输入量化为 INT8

Conv1D:
  📁 quamba/qConvLayer.py:341-399

  步骤:
    1. 接收 FP32/FP16 输入
    2. 计算动态 scale: x_dynamic_scale = x.abs().max() / 127.0
    3. 量化: x_int8 = round(x / x_dynamic_scale).clamp(-128, 127)
    4. CUDA 计算: quant_causal_conv1d_cuda.fwd_fp32() → FP32 kernel
    5. 输出: FP32 (TRUE continuous values)

  📁 csrc/causal_conv1d/quant_causal_conv1d_fwd_fp32_kernel.cuh

  关键代码 (line 155-157):
    float out_vals_store[kNElts];
    for (int i = 0; i < kNElts; ++i) {
        out_vals_store[i] = out_vals[i];  // 不量化！
    }

  📊 输入: FP32/FP16 → 动态量化到 INT8 → CUDA计算: FP32 → 输出: FP32 (TRUE)

SSM:
  📁 quamba/qSelectiveScan.py:274-289
  🔧 execute_fp32_modes('fp32_upper_bound')
  📁 quamba/SoftEdgeSSM.py:344-419 (line 378-386)

  步骤:
    1. 接收: FP32 (TRUE continuous values)
    2. 直接使用: 不需要 requantization！
    3. 计算: selective_scan_SE_float() (PyTorch FP32)
    4. 输出: FP32 → half

  📊 输入: FP32 (TRUE) → 直接使用 → PyTorch FP32 → FP32 → half

Linear层:
  📊 保持 INT8 量化 (baseline 配置)
  ⚡ 这是与 Mode 2-4 和 Mode 1 的主要区别

测试目的:
  - Hybrid precision: 测试 FP32 Conv/SSM + INT8 Linear 的组合
  - 对比 Mode 2-4: 测试 FP32 输入是否比 INT8 输入更好
  - 实用性: 比完全 FP32 更节省内存和计算

预期性能:
  - Conv/SSM 与 Mode 2-4 相同（FP32）
  - Linear 使用 INT8（减少内存/计算）
  - 整体应该接近 Mode 2-4 或 Mode 1

关键特征:
  ✅ Accepts FP32/FP16 input (no pre-quantization needed)
  ✅ Dynamic quantization at runtime
  ✅ Complete FP32 pipeline for Conv1D → SSM
  ✅ INT8 Linear (memory/compute efficient)
  🎯 Best of both worlds: FP32 precision + INT8 efficiency
```

---

### **Mode 1: Pure FP32 Upper Bound**

```
环境变量: FP32_SSM_INPUT=true

Conv1D:
  📁 (通常使用 PyTorch FP32 fallback，如果没有量化权重)
  📊 输入: FP32 → 输出: FP32 (full precision)

SSM:
  📁 quamba/qSelectiveScan.py:318-332
  📁 quamba/SoftEdgeSSM.py:344-419 (line 378-386)
  🔧 execute_fp32_modes('fp32_upper_bound')
       → selective_scan_SE_float()
  📊 输入: FP32 → 完全FP32计算 → FP32 → half

精度: 完全 FP32 (理论上界)
```

---

## 关键对比

### Conv1D 输出精度

```
Mode 0:      INT8 (256 values)
Mode 2-0:    FP32 on INT8 grid (256 discrete FP32 values)
Mode 2-1:    INT8 (256 values)
Mode 2-2:    FP32 on INT8 grid (256 discrete FP32 values)
Mode 2-3: ✅ FP32 TRUE continuous (unlimited precision)
Mode 2-4: ✨ FP32 TRUE continuous (unlimited precision)
Mode 3:   🌟 FP32 TRUE continuous (unlimited precision, accepts FP32/FP16 input)
Mode 1:      FP32 TRUE continuous (unlimited precision)
```

### SSM 计算精度

```
Mode 0:      CUDA INT8 kernel
Mode 2-0:    CUDA INT8 kernel (with requantization overhead)
Mode 2-1:    PyTorch INT8
Mode 2-2:    PyTorch FP32 (designed for INT8 grid input)
Mode 2-3: ✅ PyTorch INT8 (with requantization from TRUE FP32)
Mode 2-4: ✨ PyTorch FP32 (full precision, no requantization)
Mode 3:   🌟 PyTorch FP32 (full precision, same as Mode 2-4)
Mode 1:      PyTorch FP32 (full precision)
```

### 数值流图

```
┌──────────────┬──────────────────────────────────────────────────────────────┐
│              │                    Conv1D Output → SSM                        │
├──────────────┼──────────────────────────────────────────────────────────────┤
│ Mode 0       │ INT8 ──────────────→ INT8 SSM                               │
│ Mode 2-0     │ INT8 grid (FP32) ──→ INT8 SSM (requant)                     │
│ Mode 2-1     │ INT8 ──────────────→ INT8 SSM (direct)                      │
│ Mode 2-2     │ INT8 grid (FP32) ──→ FP32 SSM (INT8 grid logic)            │
│ Mode 2-3 ✅  │ TRUE FP32 ─────────→ INT8 SSM (requant from FP32)          │
│ Mode 2-4 ✨  │ TRUE FP32 ─────────→ FP32 SSM (full precision)             │
│ Mode 3   🌟  │ FP32/FP16 input → TRUE FP32 → FP32 SSM + INT8 Linear      │
│ Mode 1       │ TRUE FP32 ─────────→ FP32 SSM (full precision)             │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 修正历史

### 修正前的 Mode 2-3 (错误):

```
Conv1D: TRUE FP32 ✅
SSM:    execute_fp32_modes('mode22_fp32_replicates_mode21') ❌
        → selective_scan_SE_mode22_fp32_replicates_mode21()
        (为 INT8 grid 的 FP32 设计，不适合 TRUE continuous FP32)

结果: 比 Mode 2-2 更差 ❌
```

### 修正后的 Mode 2-3 (正确):

```
Conv1D: TRUE FP32 ✅
SSM:    execute_mode_21_legacy_pytorch_int8_requant() ✅
        → requantize FP32 to INT8
        → selective_scan_SE_int8Torch()

结果: 预期与 Mode 2-1 legacy 相当或更好 ✅
```

---

## 运行命令总结

```bash
# Mode 0: Baseline
python3 main.py quamba-130m-w8a8 --pretrained_dir pretrained_models/quamba1/default --quantize --eval_zero_shot --task_list lambada_openai --testing --log_dir logs_mode0

# Mode 2-0: CUDA INT8 + Requantization
FLOAT_SIM_ASIC_INT8=true SSM_USE_CUDA_FOR_FP32=true python3 main.py quamba-130m-w8a8 --pretrained_dir pretrained_models/quamba1/default --quantize --float-sim-asic-int8 --eval_zero_shot --task_list lambada_openai --testing --log_dir logs_mode20

# Mode 2-1: PyTorch INT8 Direct
FLOAT_SIM_ASIC_INT8=true SSM_USE_PYTORCH_INT8=true python3 main.py quamba-130m-w8a8 --pretrained_dir pretrained_models/quamba1/default --quantize --float-sim-asic-int8 --eval_zero_shot --task_list lambada_openai --testing --log_dir logs_mode21

# Mode 2-2: FP32 PyTorch (INT8 Grid)
FLOAT_SIM_ASIC_INT8=true python3 main.py quamba-130m-w8a8 --pretrained_dir pretrained_models/quamba1/default --quantize --float-sim-asic-int8 --eval_zero_shot --task_list lambada_openai --testing --log_dir logs_mode22

# Mode 2-3: TRUE FP32 Conv1D + PyTorch INT8 SSM (CORRECTED)
FLOAT_SIM_ASIC_INT8=true CONV1D_MODE23_FP32=true python3 main.py quamba-130m-w8a8 --pretrained_dir pretrained_models/quamba1/default --quantize --float-sim-asic-int8 --eval_zero_shot --task_list lambada_openai --testing --log_dir logs_mode23

# Mode 2-4: TRUE FP32 Conv1D + PyTorch FP32 SSM (完全FP32)
FLOAT_SIM_ASIC_INT8=true CONV1D_MODE24_FP32=true python3 main.py quamba-130m-w8a8 --pretrained_dir pretrained_models/quamba1/default --quantize --float-sim-asic-int8 --eval_zero_shot --task_list lambada_openai --testing --log_dir logs_mode24

# Mode 3: FP32/FP16 Input + FP32 Conv/SSM + INT8 Linear (Hybrid Precision)
CONV1D_MODE3_FP32=true python3 main.py quamba-130m-w8a8 --pretrained_dir pretrained_models/quamba1/default --quantize --eval_zero_shot --task_list lambada_openai --testing --log_dir logs_mode3

# Mode 1: Pure FP32 Upper Bound
FP32_SSM_INPUT=true python3 main.py quamba-130m-w8a8 --pretrained_dir pretrained_models/quamba1/default --quantize --fp32-ssm-input --eval_zero_shot --task_list lambada_openai --testing --log_dir logs_mode1
```

---

## 调试命令

```bash
# Mode 2-3 with debug
FLOAT_SIM_ASIC_INT8=true CONV1D_MODE23_FP32=true SSM_DEBUG_MODE23=true python3 main.py quamba-130m-w8a8 --pretrained_dir pretrained_models/quamba1/default --quantize --float-sim-asic-int8 --eval_zero_shot --task_list lambada_openai --testing --log_dir logs_mode23_debug

# 检查 debug_mode_comparison/ 目录
ls -lh debug_mode_comparison/
```
