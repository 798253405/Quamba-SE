# Quantization Granularity (量化粒度) 详解

## "整个层" vs "Channel" - 精确定义

### Conv1d 层的维度

```python
# Conv1d 输入
x.shape = [Batch, Channels, Length]
        = [B, D, L]

例如 Mamba2-130m:
  D = 768 (channels/dim)
  L = sequence length
```

## Quamba1 vs Quamba2 的量化粒度

### 1️⃣ Quamba1 (Per-Tensor)

```python
x_out_scales = scalar  # 一个数值

含义:
  • 整个 Conv1d 层的所有输出共享 1 个 scale
  • 所有 Channels (D=768) 用同一个 scale
  • 所有 Sequence positions (L) 用同一个 scale

量化公式:
  for b in range(B):
    for d in range(D):      # 所有 channels
      for l in range(L):    # 所有 positions
        q[b,d,l] = round(x[b,d,l] / scale_out)  # 同一个 scale!

粒度: Per-Tensor (整个 3D tensor 共享)
```

**问题**:
- Channel 0 的值可能在 [0, 1] 范围
- Channel 767 的值可能在 [0, 10] 范围
- 强制用同一个 scale → Channel 0 精度损失 10x!

---

### 2️⃣ Quamba2 (Group Quantization)

#### Shape: (1, 4, 4) for 130m/2.7b

```python
x_out_scales.shape = (1, 4, 4)

维度解释:
  Dim 0 (SSD groups): 1 group
    → 所有 sequence positions 共享 (不分 token)

  Dim 1 (Head groups): 4 groups
    → 768 channels 分成 4 组
    → 每组 192 channels

  Dim 2 (Dim groups): 4 groups
    → 每个 head group 的 192 channels 再分 4 组
    → 每组 48 channels

总共: 1 × 4 × 4 = 16 个 scales
```

**具体映射**:

```python
# Head groups (4组)
head_group_0: channels [0:192]     → 4 个 dim scales
head_group_1: channels [192:384]   → 4 个 dim scales
head_group_2: channels [384:576]   → 4 个 dim scales
head_group_3: channels [576:768]   → 4 个 dim scales

# 每个 head group 内部的 Dim groups (4组)
以 head_group_0 为例:
  dim_group_0: channels [0:48]     → 1 个 scale
  dim_group_1: channels [48:96]    → 1 个 scale
  dim_group_2: channels [96:144]   → 1 个 scale
  dim_group_3: channels [144:192]  → 1 个 scale

量化时:
  scale_idx = (ssd_group, head_group, dim_group)
  scale = x_out_scales[0, head_group, dim_group]

  q[b, c, l] = round(x[b, c, l] / scale)
  其中:
    head_group = c // 192
    dim_group = (c % 192) // 48
```

**粒度**: Per-Channel-Group (每 48 个 channels 共享一个 scale)

---

#### Shape: (8, 4, 4) for 8b

```python
x_out_scales.shape = (8, 4, 4)

维度解释:
  Dim 0 (SSD groups): 8 groups
    → 可能对应不同的 State Space 维度
    → 或者更细粒度的 token/position 分组

  Dim 1 (Head groups): 4 groups (同上)
  Dim 2 (Dim groups): 4 groups (同上)

总共: 8 × 4 × 4 = 128 个 scales
```

**粒度**: 更细 - 增加了 SSD 维度的分组

---

## 对比总结表

| 方法 | 粒度名称 | Scale 数量 | 每个 scale 覆盖的 channels | 说明 |
|------|---------|-----------|------------------------|------|
| **Quamba1** | Per-Tensor | 1 | **所有 768 channels** | 整个层共享 |
| **Quamba2 (130m)** | Per-Channel-Group | 16 | **48 channels** | 每 48 个 channels 一组 |
| **Quamba2 (8b)** | Per-Channel-Group + SSD | 128 | **48 channels (per SSD group)** | 更细粒度 |

## 为什么不是 Per-Channel？

### Per-Channel 量化

```python
# 如果是 per-channel
x_out_scales.shape = (D,) = (768,)  # 每个 channel 一个 scale

量化:
  for d in range(768):
    q[:,d,:] = round(x[:,d,:] / x_out_scales[d])

优点: 最精细
缺点:
  • 768 个 scales，开销大
  • Calibration 时需要观察每个 channel 的分布
  • CUDA kernel 需要频繁切换 scale
```

### Quamba2 的 Group 方案

```python
# Group quantization (折中方案)
x_out_scales.shape = (1, 4, 4)  # 16 scales

量化:
  每 48 个 channels 用一个 scale

优点:
  • 比 per-tensor 精细 48x
  • 比 per-channel 开销小 48x
  • 充分利用 Mamba2 的 multi-head 结构
```

## 具体例子

### 130m 模型 (D=768)

```
Channel Range    | Head Group | Dim Group | Scale Index
-----------------+------------+-----------+-------------
[0:48]           |     0      |     0     | x_out_scales[0,0,0]
[48:96]          |     0      |     1     | x_out_scales[0,0,1]
[96:144]         |     0      |     2     | x_out_scales[0,0,2]
[144:192]        |     0      |     3     | x_out_scales[0,0,3]
[192:240]        |     1      |     0     | x_out_scales[0,1,0]
[240:288]        |     1      |     1     | x_out_scales[0,1,1]
...              |    ...     |    ...    | ...
[720:768]        |     3      |     3     | x_out_scales[0,3,3]

如果用 Quamba1:
  所有 [0:768] 用同一个 scale!
```

### 实际数据 (130m Layer 0)

```python
# Quamba2 的 16 个 scales (从前面的输出):
x_out_scales[0] = [
  [0.0078, 0.0112, 0.0316, 0.0518],  # Head 0 的 4 个 dim groups
  [0.0065, 0.0063, 0.0208, 0.0166],  # Head 1 的 4 个 dim groups
  [0.0066, 0.0111, 0.0214, 0.0093],  # Head 2 的 4 个 dim groups
  [0.0057, 0.0191, 0.0131, 0.0137],  # Head 3 的 4 个 dim groups
]

观察:
  • Head 0, Dim 3: scale=0.0518 (最大)
  • Head 3, Dim 0: scale=0.0057 (最小)
  • 差异: 0.0518 / 0.0057 = 9.1x!

如果用 Quamba1 (per-tensor):
  • 统一 scale = 0.0518 (取最大，避免饱和)
  • Channels [576:624] (对应 Head 3, Dim 0) 的值最大只到 0.0057
  • 量化后: round(0.0057 / 0.0518 * 127) ≈ 14
  • 只用了 INT8 的 14/127 = 11% 范围!
  • 精度损失: 9.1x

如果用 Quamba2:
  • Channels [576:624] 用自己的 scale = 0.0057
  • 量化后: round(0.0057 / 0.0057 * 127) = 127
  • 充分利用 INT8 的 100% 范围!
```

## 回答原问题

### "整个层" 指什么？

**Quamba1 的"整个层"**:
- 指整个 Conv1d 层的所有输出
- 包括所有 channels (D=768)
- 包括所有 sequence positions (L)
- 所有 B×D×L 个数值用 1 个 scale

**Quamba2 没有"整个层"**:
- 层被分成 16 或 128 个 groups
- 每个 group 是 **48 个 channels 的一段**
- 例如: channels [0:48] 是一个 group

### "Channel" 指什么？

- Channel 是 Conv1d 的一个输出维度 (D 维的一个元素)
- 768 channels 对应 768 个不同的特征
- **Quamba2 不是 per-channel**，而是 **per-channel-group** (48 channels/group)

## 最终答案

```
Quamba1 对 Mamba2:
  "整个层" = 所有 768 channels 共享 1 个 scale

Quamba2 对 Mamba2:
  "整个层" 被分成 16 groups
  每个 group = 48 channels
  每个 group 用自己的 scale

不是:
  ❌ Per-channel (768 scales)

是:
  ✅ Per-channel-group (16 scales, 每组 48 channels)
```

## 代码验证位置

| 信息 | 文件 | 说明 |
|-----|------|------|
| Group 划分 | qMamba2.py | x_head_group_range, x_dim_group_range |
| CUDA 索引 | quant_causal_conv1d_fwd_kernel.cuh | 根据 channel 索引到对应 scale |
| Calibration | observer.py:121-180 | PerSSDGroupObserver 观察每个 group |
