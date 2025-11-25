# Quamba 模型层结构说明

## 我们追踪的是什么层？

### Scale 的完整路径

以 Layer 0 为例，完整的 key 名称：

```python
backbone.layers.0.mixer.conv1d.x_scale       # Conv1d 输入 x 的 scale
backbone.layers.0.mixer.conv1d.wx_scale      # weight × x 的联合 scale
backbone.layers.0.mixer.conv1d.x_out_scales  # Conv1d+SiLU 输出 x 的 scale
```

### 层次结构

```
Model (Quamba2)
├── backbone
│   └── layers (24层 for 130m model)
│       ├── Layer 0
│       │   └── mixer (Mamba2Mixer / SSM block)
│       │       ├── in_proj (Linear: 输入投影)
│       │       ├── conv1d ← 🎯 我们关注的层!
│       │       │   ├── x_scale      (输入 scale)
│       │       │   ├── wx_scale     (Conv1d 理论输出 scale)
│       │       │   └── x_out_scales (Conv1d+SiLU 实际输出 scale) ← 🔥 Percentile scale!
│       │       ├── x_proj (Linear: SSM 参数投影)
│       │       ├── dt_proj (Linear)
│       │       ├── selective_scan (SSM 核心计算)
│       │       └── out_proj (Linear: 输出投影)
│       ├── Layer 1
│       │   └── mixer.conv1d ...
│       ⋮
│       └── Layer 23
│           └── mixer.conv1d ...
└── norm_f (Final LayerNorm)
```

## Conv1d 在 Mamba 架构中的位置

### 完整的数据流 (单个 Mamba block)

```
输入: hidden_states [B, L, D]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1️⃣ in_proj (Linear)                                         │
│    输入:  FP16 [B, L, D]                                     │
│    输出:  INT8 [B, L, 2*D]  (量化后)                        │
│    Split: x [B, D, L], z [B, D, L]                          │
└─────────────────────────────────────────────────────────────┘
    ↓ x
┌─────────────────────────────────────────────────────────────┐
│ 2️⃣ conv1d ← 🎯 我们分析的核心层!                            │
│    输入:  INT8 x [B, D, L]                                   │
│    scale: x_scale (反量化输入)                               │
│                                                              │
│    CUDA Kernel 内部:                                         │
│    ┌────────────────────────────────────────────────────┐  │
│    │ • 读取 INT8 x, INT8 weight                         │  │
│    │ • 转为 FP32                                        │  │
│    │ • Conv1d 计算 (FP32)                               │  │
│    │   out = conv(x) * wx_scale + bias                 │  │
│    │ • SiLU 激活 (FP32)                                 │  │
│    │   out = out / (1 + exp(-out))                     │  │
│    │ • 量化为 INT8                                      │  │
│    │   q = round(out / x_out_scale) ← 使用 percentile! │  │
│    └────────────────────────────────────────────────────┘  │
│                                                              │
│    输出:  INT8 [B, D, L]                                     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3️⃣ x_proj (Linear)                                          │
│    输入:  INT8 x [B, L, D]  (rearrange 后)                  │
│    输出:  INT8 [B, L, dt_rank + 2*dstate]                   │
│    Split: dt, B, C                                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4️⃣ dt_proj (Linear)                                         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 5️⃣ selective_scan (SSM 核心)                                │
│    输入:  x (INT8), dt, B, C, z                             │
│    输出:  FP16 y [B, D, L]                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 6️⃣ out_proj (Linear)                                        │
│    输出:  FP16 [B, L, D]                                     │
└─────────────────────────────────────────────────────────────┘
```

## 为什么关注 Conv1d？

### 1. Conv1d 是唯一融合 SiLU 的层

```
其他层:        Linear (INT8×INT8 → 直接量化)
Conv1d 特殊:   Conv (INT8×INT8) + SiLU (FP32) → 需要重新量化
```

### 2. SiLU 改变了值的分布

```
Conv1d 输出 (理论):  wx_scale = x_scale × weight_scale
                    值域由 INT8×INT8 的点积决定

SiLU 输出 (实际):    x_out_scale = percentile(SiLU(conv_output))
                    值域被 SiLU 非线性变换改变

关键: x_out_scale ≠ wx_scale  (ratio = 1.7 - 12.7)
```

### 3. Percentile 只用在这里

整个 Mamba block 中，**只有 Conv1d 的输出使用 percentile scale**：

| 层 | 输入量化 | 输出量化 | 使用 Percentile? |
|----|---------|---------|-----------------|
| in_proj | MinMax | MinMax | ❌ |
| **conv1d** | MinMax | **Percentile** | ✅ 🔥 |
| x_proj | MinMax | MinMax | ❌ |
| dt_proj | MinMax | MinMax | ❌ |
| selective_scan | MinMax | N/A (FP16) | ❌ |
| out_proj | MinMax | N/A (FP16) | ❌ |

## 我们追踪的 24 层是什么？

### Quamba2-130m 架构

```python
Total layers: 24 Mamba blocks

每个 block 包含:
  • 1x RMSNorm
  • 1x Mamba2Mixer
      - in_proj
      - conv1d ← 我们追踪的
      - x_proj
      - dt_proj
      - selective_scan
      - out_proj
```

### 对应关系

```
我们的 "Layer 0" = backbone.layers.0.mixer.conv1d
我们的 "Layer 1" = backbone.layers.1.mixer.conv1d
我们的 "Layer 2" = backbone.layers.2.mixer.conv1d
...
我们的 "Layer 23" = backbone.layers.23.mixer.conv1d
```

**每一层都是独立的 Mamba block，有自己独立的 conv1d 和 scales。**

## 追踪的 Scale 含义

### x_scale (Conv1d 输入)

```python
来源: in_proj 的输出 scale
用途: Conv1d CUDA kernel 中反量化输入
      x_fp32 = x_int8 * x_scale

代码位置: quant_causal_conv1d_fwd_kernel.cuh:57
          float scale_x = params.scale_x;
```

### wx_scale (Conv1d 理论输出)

```python
定义: wx_scale = weight_scale × x_scale
含义: 如果直接用 INT8×INT8 的结果，理论上的反量化 scale
问题: 经过 SiLU 后，这个 scale 不再适用!

计算: 在模型构建时预计算并保存
      qconv.wx_scale = qconv.weight_scale * input_scale
```

### x_out_scales (Conv1d+SiLU 实际输出) 🔥

```python
来源: Calibration 时用 PerTensorPercentileObserver 观察 SiLU 输出
计算: torch.quantile(silu_output, percentile_alpha)
      scale = percentile_value / 127

用途: Conv1d CUDA kernel 中量化输出
      q_out = round(silu_output_fp32 / x_out_scale)

代码位置:
  • 计算: observer.py:92
  • 保存: qMambaLayer.py:852
  • 使用: quant_causal_conv1d_fwd_kernel.cuh:149

Shape: (1, 4, 4) for Quamba2
       可能是 per-group 量化 (head_groups × dim_groups)
```

## 数据示例解读

```
Layer 0 | x_scale: 0.0380 | wx_scale: 0.0092 | x_out_scale: 0.0158 | ratio: 1.72
```

**含义**:
1. **x_scale = 0.0380**: in_proj 输出的 INT8 值，乘以 0.0380 得到真实值
2. **wx_scale = 0.0092**: Conv1d 权重和输入的联合 scale
3. **x_out_scale = 0.0158**:
   - SiLU 输出的 FP32 值，除以 0.0158 量化为 INT8
   - 来自 calibration 时观察 99.95% 分位数
4. **ratio = 1.72**:
   - x_out_scale / wx_scale = 1.72
   - 说明 SiLU 扩大了值域 1.72 倍
   - 不能直接用 wx_scale，必须重新 calibrate!

## 总结

### 我们追踪的是

✅ **24 个 Mamba block 的 conv1d 层**
- 每层独立的 1D 卷积 + SiLU 激活
- 唯一使用 percentile scale 的地方
- 对 SSM 性能影响最大的量化节点

### 为什么重要

1. **SiLU 改变分布** → 必须重新 calibrate
2. **不同层差异大** → ratio 从 1.7 到 12.7
3. **Percentile 的核心** → 只在这里用，且效果显著

### 关键文件位置

| 功能 | 文件 | 行号 |
|-----|------|-----|
| Percentile 计算 | observer.py | 92 |
| Conv1d 构建 | qMambaLayer.py | 848-852 |
| CUDA SiLU | quant_causal_conv1d_fwd_kernel.cuh | 139-144 |
| CUDA 量化 | quant_causal_conv1d_fwd_kernel.cuh | 149 |
| Scale 读取 | analyze_scales.py | 43-55 |
