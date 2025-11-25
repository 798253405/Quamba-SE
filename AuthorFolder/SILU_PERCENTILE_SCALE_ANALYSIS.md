# SiLU 与 Percentile Scale 的完整分析

## 核心问题

**为什么 Conv1d 输出需要用 Percentile Scale 重新量化？**

## 背景：Conv1d 的精度流程

### Online Inference 中的实际流程

```
CUDA Kernel: quant_causal_conv1d_fwd_kernel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

输入:  INT8 x, INT8 weight, INT8 bias
      FP32 scale_x, scale_w, scale_b, scale_out

Step 1: 读取并转换为 FP32
  x_vals[i] = float(x_vals_load[i])        // INT8 → FP32 (类型转换)
  weight_vals[i] = float(weight[i])        // INT8 → FP32 (类型转换)
  bias_val = float(bias) * scale_b         // INT8 → FP32 并反量化

Step 2: Conv1d 计算 (FP32)
  out_tmp = Σ(weight_vals[w] × x_vals[...])  // INT8×INT8 以 FP32 计算
  out_vals[i] = scale_wx × out_tmp + bias_val // 反量化到真实值域

Step 3: SiLU Activation (FP32 → FP32) 🔥 关键!
  out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]))

Step 4: 量化为 INT8 (使用 percentile scale)
  q = clamp(round(out_vals[i] / scale_out), -128, 127)

输出:  INT8 tensor (直接写回 global memory)
```

**关键发现：Conv1d 内部是 FP32 计算，但直接输出 INT8！**

## 核心：SiLU 函数为什么是关键

### SiLU 的数学定义

```
SiLU(x) = x / (1 + e^(-x))

特性:
  • x → -∞: SiLU(x) → 0    (负数被压缩到接近0)
  • x = 0:  SiLU(0) = 0    (原点)
  • x → +∞: SiLU(x) → x    (大正数几乎不变)
  • 非线性、非对称
```

### SiLU 如何改变值的分布

假设 Conv1d 输出范围 `[-5.0, 5.0]`, 间距 = 0.1 (由 scale_wx 决定)

```
Conv1d 输出 (FP32)  →  SiLU 输出 (FP32)  →  说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  -5.00           →     -0.033          →  负数被严重压缩
  -4.00           →     -0.072          →  压缩到 [-0.27, 0]
  -3.00           →     -0.142          →  范围仅 0.27
  -2.00           →     -0.238          →
  -1.00           →     -0.269          →  最小值
   0.00           →      0.000          →  中心点
   1.00           →      0.731          →
   2.00           →      1.762          →  正数相对保持
   3.00           →      2.858          →
   4.00           →      3.928          →  接近线性
   5.00           →      4.967          →  范围约 5.0

输入范围: 10.0  (从 -5 到 5)
输出范围: 5.24  (从 -0.27 到 4.97)  ← 压缩了 47.6%!
分布:     不对称，大部分值集中在 [0, 3] 区间
```

### 为什么不能用 Conv1d 的 scale？

#### 场景1: 用 scale_wx = 0.1 (Conv1d 的 scale)

```
SiLU 输出 (FP32)  →  量化 (INT8)  →  反量化 (FP32)  →  误差
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  -0.269          →       -3       →      -0.300       →  0.031
   0.000          →        0       →       0.000       →  0.000
   0.731          →        7       →       0.700       →  0.031
   1.762          →       18       →       1.800       →  0.038
   2.858          →       29       →       2.900       →  0.042
   4.967          →       50       →       5.000       →  0.033

INT8 使用范围: [-3, 50]
INT8 利用率:   50/127 = 39.4%  ← 浪费 60% 的表示能力!
平均误差:      0.027
```

**问题：**
- SiLU 输出范围是 `[~0, 5]`，但 scale_wx 是为 Conv1d 的 `[-5, 5]` 设计的
- Scale 太大 → INT8 的 256 个离散级别被浪费
- 量化精度低

#### 场景2: 用 Percentile scale = 0.0387

```
SiLU 输出 (FP32)  →  量化 (INT8)  →  反量化 (FP32)  →  误差
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  -0.269          →       -7       →      -0.271       →  0.002
   0.000          →        0       →       0.000       →  0.000
   0.731          →       19       →       0.736       →  0.005
   1.762          →       46       →       1.781       →  0.019
   2.858          →       74       →       2.865       →  0.007
   4.967          →      127       →       4.917       →  0.050 (饱和)

INT8 使用范围: [-7, 127]
INT8 利用率:   127/127 = 100%  ← 充分利用!
平均误差:      0.011
误差改善:      2.51x
```

**优势：**
- Scale 根据 SiLU 输出的实际分布计算
- Scale 更小 (0.0387 vs 0.1) → 量化间距更细
- INT8 的 256 个级别充分利用
- 量化精度提高 2.5 倍

### 为什么需要 Percentile 而不是 MinMax？

假设 SiLU 输出有 outliers:
- 99.95% 的值在 `[0, 3.0]`
- 0.05% 的 outliers 到达 `10.0`

```
MinMax scale      = 10.0 / 127 = 0.0787
Percentile scale  = 3.0 / 127 = 0.0236  ← 小了 3.3 倍!

对于正常值 1.0:
  MinMax:      q = round(1.0/0.0787) = 13  → 精度低
  Percentile:  q = round(1.0/0.0236) = 42  → 精度高 3.3x!

对于 outlier 10.0:
  MinMax:      q = 127  → 正常表示
  Percentile:  q = 127 (饱和) → 有损失，但只影响 0.05%
```

**Trade-off：牺牲 0.05% 的 outliers，换取 99.95% 正常值的高精度！**

## 完整的 Percentile Scale 工作流程

### Offline Calibration (生成 scale)

```python
# 位置: modelutils_mamba.py:161-165
if is_x(op) or is_ssm_state(op):
    observers[i][op + ":input"] = PerTensorPercentileObserver(
        n_bits=8, clip_ratio=1.0, sym=True,
        percentile_alpha=0.9995  # 忽略 top 0.05%
    )

# Hook 捕获: x_proj 的输入 (即 conv1d+silu 的输出)
# 位置: qMambaLayer.py:105-109 (FP16 版本)
x = self.conv1d(x)           # FP16
x = self.act(x[...,:seqlen]) # FP16 SiLU
x_reshape = rearrange(x, "b d l -> b l d")  # Hook 在这里捕获 FP16
x_dbl = self.x_proj(x_reshape)

# Observer 计算 percentile (observer.py:90-92)
w = w.clone().to(torch.float32)  # FP16 → FP32
cur_max = torch.quantile(w.abs().reshape(-1), self.percentile_alpha)
# cur_max: 99.95% 分位数 (FP32)

# EMA 更新 (observer.py:98-101)
if self.w_max is None:
    self.w_max = cur_max
else:
    self.w_max = self.w_max + 0.01 * (cur_max - self.w_max)

# 计算 scale (observer.py:112-118)
scales = w_max / 127  # FP32 scalar
return scales.to(torch.float32).clamp(min=1e-6)

# 保存到 state_dict (qMambaLayer.py:852)
qconv.output_scale = act_scales["x_proj:input"].item()  # FP32 scalar
```

### Online Inference (使用 scale)

```cpp
// 位置: quant_causal_conv1d_fwd_kernel.cuh:57-62
float scale_x = params.scale_x;      // Conv1d 输入 scale
float scale_w = params.scale_w;      // Conv1d 权重 scale
float scale_b = params.scale_b;      // Conv1d bias scale
float scale_out = params.scale_out;  // Conv1d 输出 scale (来自 percentile!)
float scale_wx = scale_w * scale_x;  // 联合 scale

// Conv1d 计算 (Line 126-137)
float out_vals[kNElts];
for (int i = 0; i < kNElts; ++i) {
    float out_tmp = 0;
    for (int w = 0; w < kWidth; ++w) {
        out_tmp += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
    }
    out_vals[i] = scale_wx * out_tmp + bias_val;  // FP32
}

// SiLU activation (Line 139-144)
if (params.silu_activation) {
    for (int i = 0; i < kNElts; ++i) {
        out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));  // FP32
    }
}

// 量化为 INT8 (Line 146-151) - 使用 percentile scale!
input_t out_vals_store[kNElts];
for (int i = 0; i < kNElts; ++i) {
    int tmp = int(roundf(out_vals[i] / scale_out));  // FP32 / FP32
    out_vals_store[i] = tmp > 127 ? 127 : tmp < -128 ? -128 : static_cast<input_t>(tmp);
}
// 输出: INT8 tensor
```

## 实际模型的 Scale 数据

### Quamba2-130m-w8a8 (default 配置)

运行 `python3 analyze_scales.py` 的结果：

```
Layer |     Config |         x_scale |        wx_scale |     x_out_scale |      ratio |     shape
     0 |    default |      0.03795522 |      0.00918891 |      0.01578744 |     1.7181 | (1, 4, 4)
     1 |    default |      0.03491019 |      0.00804318 |      0.06717040 |     8.3512 | (1, 4, 4)
     2 |    default |      0.05035064 |      0.00715890 |      0.07035671 |     9.8279 | (1, 4, 4)
     3 |    default |      0.03946235 |      0.00817391 |      0.07583546 |     9.2778 | (1, 4, 4)
     4 |    default |      0.05570251 |      0.00752799 |      0.06240292 |     8.2895 | (1, 4, 4)
```

### 🔥 关键发现

1. **ratio > 1**: SiLU **扩大了值域**，而不是压缩！
   - 理论分析基于简化假设（Conv1d 输出 `[-5, 5]`）
   - 实际模型中，Conv1d 输出可能本身就很小
   - SiLU 对小负数的压缩 + 对正数的保持 = 总体扩大

2. **x_out_scales 是 tensor**: Shape `(1, 4, 4)`
   - 不是 per-tensor 量化（单个 scalar）
   - 可能是 per-channel 或 per-group 量化
   - Dim 1 和 Dim 2 可能对应 head groups 和 dim groups

3. **ratio 差异很大**: 从 1.7 到 9.8
   - 不同层的 SiLU 影响不同
   - 深层网络 (Layer 1-4) 的 ratio 更大 (8-10x)
   - 浅层网络 (Layer 0) 的 ratio 较小 (1.7x)

### 修正的理解

**原始假设**: SiLU 压缩值域 → percentile scale 更小 → ratio < 1

**实际情况**:
- Conv1d 输出经过量化后，`wx_scale` 已经很小 (0.007-0.009)
- SiLU 激活后，输出值域实际上**扩大**了
- `x_out_scale` (0.015-0.076) > `wx_scale` → ratio > 1
- Percentile 的作用：**防止 SiLU 输出的 outliers 让 scale 更大**

### 为什么仍然需要 Percentile？

即使 ratio > 1，Percentile 仍然重要：

1. **SiLU 改变了值的分布** - 需要重新观察实际输出
2. **Outliers 问题依然存在** - 如果用 MinMax，scale 会更大
3. **不同层差异巨大** - ratio 从 1.7 到 9.8，说明每层都需要独立 calibration

## 总结

### 核心答案

**Q: 为什么需要 Percentile Scale？**

**A: 因为 SiLU 是非线性函数，它改变了值的范围和分布：**

1. **Conv1d 输出**: 量化后的 `wx_scale` 很小 (0.007-0.009)
2. **SiLU 输出**: 扩大了值域，需要更大的 scale (0.015-0.076)
3. **如果用 Conv1d 的 scale**:
   - Scale 太小，无法表示 SiLU 输出
   - 会导致严重的饱和（clipping）
   - 量化误差巨大
4. **用 Percentile scale**:
   - 观察 SiLU 输出的实际分布
   - 计算适合 SiLU 输出的 scale
   - 忽略 top 0.05% outliers，防止 scale 过大
   - 充分利用 INT8 范围

### 关键公式

```
量化公式: q = clamp(round(x / scale), -128, 127)

Conv1d scale:    scale_wx = (w_max / 127) * (x_max / 127)
                 为 Conv1d 输出范围设计

Percentile scale: scale_out = percentile(SiLU_output, 99.95%) / 127
                  为 SiLU 输出分布优化
                  忽略 top 0.05% outliers
```

### 精度提升数据

| 指标 | 用 scale_wx | 用 Percentile scale | 提升 |
|-----|------------|-------------------|------|
| INT8 利用率 | 39.4% | 100% | 2.5x |
| 平均量化误差 | 0.027 | 0.011 | 2.5x |
| 受 outliers 影响 | 大 | 小 (只影响 0.05%) | - |

### 本质

**SiLU 把 FP32 值"重新排列"了，需要重新选择最优的量化间距来充分利用 INT8 的 256 个离散级别！**

这不是"增加信息量"，而是"减少表示误差" - 用有限的 INT8 范围去更好地拟合实际值分布。

## 代码位置索引

| 功能 | 文件 | 行号 |
|-----|------|-----|
| Percentile Observer 定义 | observer.py | 75-118 |
| Percentile 计算 (torch.quantile) | observer.py | 92 |
| Calibration 注册 observer | modelutils_mamba.py | 161-165 |
| Conv1d output_scale 设置 | qMambaLayer.py | 848-852 |
| CUDA kernel SiLU 计算 | quant_causal_conv1d_fwd_kernel.cuh | 139-144 |
| CUDA kernel 量化 | quant_causal_conv1d_fwd_kernel.cuh | 146-151 |
| Conv1d forward (Python) | qMambaLayer.py | 920 |
| SSM 接收 INT8 | qMambaLayer.py | 933 |
