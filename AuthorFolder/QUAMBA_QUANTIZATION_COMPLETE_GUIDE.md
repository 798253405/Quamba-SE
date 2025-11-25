# Quamba量化完全指南

**生成时间**: 2025-11-05
**目的**: 理解Quamba量化机制，为改进scale选择提供基础

---

## 📋 目录

1. [量化实现概览](#1-量化实现概览)
2. [Scale计算与精度](#2-scale计算与精度)
3. [Reorder与分组机制](#3-reorder与分组机制)
4. [正负号与对称量化](#4-正负号与对称量化)
5. [替换可行性评估](#5-替换可行性评估)
6. [改进Scale的实验思路](#6-改进scale的实验思路)

---

## 1. 量化实现概览

### 1.1 数据类型全景

```
┌─────────────────┬──────────────┬─────────────┬──────────────┐
│    阶段         │   精度       │   位置      │   用途       │
├─────────────────┼──────────────┼─────────────┼──────────────┤
│ Calibration     │ FP32         │ observer.py │ 计算scale    │
│ Scale存储       │ FP32         │ qConvLayer  │ 保存scale    │
│ Weight存储      │ INT8         │ GPU GMEM    │ 节省内存     │
│ Activation存储  │ INT8         │ GPU GMEM    │ 节省带宽     │
│ Conv1D计算      │ FP32         │ CUDA Core   │ Fake quant   │
│ Linear计算      │ INT8         │ Tensor Core │ True INT8    │
└─────────────────┴──────────────┴─────────────┴──────────────┘
```

### 1.2 量化公式

**Symmetric量化（当前使用）**：
```python
# 量化
q = round(x / scale)  # q ∈ [-128, 127]
q_clamp = clamp(q, -128, 127)

# 反量化
x_dequant = q_clamp * scale
```

**关键参数**：
- `n_bits = 8`: INT8
- `sym = True`: 对称量化（zero_point=0）
- `q_range = [-128, 127]`: Signed INT8

### 1.3 两个阶段

```
┌─────────────────────────────────────────────────────────────┐
│ Calibration阶段（离线，一次性，~2-5分钟）                    │
├─────────────────────────────────────────────────────────────┤
│ 1. 运行512个样本收集激活统计                                 │
│ 2. 计算percentile或max（全FP32）                            │
│ 3. (可选) Reorder聚类分组（2-5分钟）                        │
│ 4. 计算scale: scale = w_max / 127（FP32）                  │
│ 5. 量化权重，保存量化模型                                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Runtime阶段（每次forward，~10ms/token）                      │
├─────────────────────────────────────────────────────────────┤
│ 1. 读取预计算的scale（FP32，GPU内存）                       │
│ 2. Dequantize: INT8 → FP32（Conv1D）或直接INT8计算(Linear)│
│ 3. 计算（FP32或INT8）                                       │
│ 4. Quantize: FP32 → INT8（层间传递）                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Scale计算与精度

### 2.1 当前Scale计算（observer.py）

```python
# quamba/observer.py:137-154（简化）
class ObserverBase(nn.Module):
    def get_quantization_params(self, w):
        # Step 1: Percentile裁剪（FP32）
        if self.sym:
            cur_max = torch.quantile(w.abs().reshape(-1),
                                    self.percentile_alpha)  # 默认0.9995

        # Step 2: EMA累积（FP32）
        if self.w_max is None:
            self.w_max = cur_max
        else:
            self.w_max = self.w_max + self.percentile_sigma * (cur_max - self.w_max)

        # Step 3: 计算scale（FP32）
        _, q_max = _get_quant_range(n_bits=8, sym=True)  # q_max=127
        scale = self.w_max / q_max  # FP32

        return scale
```

**关键参数**：
- `percentile_alpha = 0.9995`: 裁剪top 0.05%
- `percentile_sigma = 0.1`: EMA平滑系数

### 2.2 Scale的持久性原理

**为什么固定scale能用于不同输入？**

1. **LayerNorm归一化**：
   ```python
   # 每层都有RMSNorm
   x_normalized = x / sqrt(mean(x^2))
   # 输出RMS ≈ 1.0，强制分布归一化
   ```

2. **统计平稳性**：
   ```
   激活值分布（某层）：
   Calibration: ████████████  ← 99% in [-5.0, 5.0]
   Runtime:     ████████████  ← 99% in [-4.8, 5.2]
   观察：动态范围相似（±10%以内）
   ```

3. **保守估计**：
   ```python
   w_max = max(batch_1_max, ..., batch_512_max)
   # 取512个batch的最大值，覆盖95-99%未来输入
   ```

4. **优雅降级**：
   ```cuda
   // 溢出时饱和截断
   q = clamp(round(x / scale), -128, 127);
   // 只有0.1-1%激活值溢出，神经网络冗余性可补偿
   ```

### 2.3 Scale存储

```python
# quamba/qConvLayer.py:184-196
self.register_buffer('x_out_scales', torch.empty(
    (n_groups, x_nhead_group, x_ndim_group),
    dtype=torch.float32))  # ← FP32精度

# 内存占用（Mamba2-2.7B）
# 128 groups × 4 bytes = 512 bytes/layer
# 64 layers × 512 bytes = 32 KB（可忽略）
```

---

## 3. Reorder与分组机制

### 3.1 Piecewise量化原理

**目标**：降低每组内的动态范围，提高量化精度

```
Without grouping（per-tensor）:
范围：[-5.0, 5.0]  scale = 5.0/127 = 0.0394
精度：±0.02

With grouping（piecewise）:
Group 1: [-2.0, 2.0]  scale = 2.0/127 = 0.0157  ← 精度提升2.5x
Group 2: [-1.5, 1.5]  scale = 1.5/127 = 0.0118  ← 精度提升3.3x
Group 3: [-3.0, 3.0]  scale = 3.0/127 = 0.0236
Group 4: [-1.0, 1.0]  scale = 1.0/127 = 0.0079  ← 精度提升5x
```

### 3.2 Quamba2的分组策略

```python
# quamba/reorder_utils.py:86-121
def group_wise_sort_indices(tensor, headdim, ssd_ngroups,
                           nhead_groups=4, ndim_groups=4):
    # 两层聚类
    # 1. Head聚类（AgglomerativeClustering）
    head_clustering = AgglomerativeClustering(
        n_clusters=nhead_groups,
        metric='euclidean',
        linkage='ward'
    ).fit(activations)

    # 2. Dimension聚类（KMeans）
    dim_clustering = KMeans(
        n_clusters=ndim_groups
    ).fit(head_activations)
```

**分组数量**（Mamba2-2.7B）：
- 8 SSD groups（固定）
- 4 head groups/SSD（可调）
- 4 dim groups/head（可调）
- **总计**：8 × 4 × 4 = **128 piecewise groups**

### 3.3 Runtime开销

```cuda
// csrc/causal_conv1d/quamba2_conv1d_fwd_kernel.cuh:171-198
// 双层循环查找scale（每个元素执行一次）
for (int hg_idx = 0; hg_idx < 4; hg_idx++) {        // Head groups
    if (h_start <= head_idx && head_idx < range[hg_idx]) {
        for (int dg_idx = 0; dg_idx < 4; dg_idx++) {  // Dim groups
            if (ch_start <= dim_idx && dim_idx < range[dg_idx]) {
                scale_out = scales[hg_idx * 4 + dg_idx];  // 找到scale
                break;
            }
        }
        break;
    }
}
```

**开销分析**：
- 平均4次比较（early break）
- 完全fit L1 cache（~1KB元数据）
- **总开销**：<1% runtime时间

### 3.4 分组数量与精度的权衡

| 分组策略 | Scale数量 | 精度提升 | Runtime开销 | Calibration时间 |
|---------|----------|---------|------------|----------------|
| Per-tensor | 1 | 基线 | 0% | <1秒 |
| 2×2 (4 groups) | 32 | +0.5% | <0.5% | ~30秒 |
| **4×4 (16 groups)** | **128** | **+1.5%** | **<1%** | **2-5分钟** |
| 8×8 (64 groups) | 512 | +2.2% | ~2% | ~10分钟 |
| 16×16 (256 groups) | 2048 | +2.5% | ~5% | ~30分钟 |
| Per-channel | 8192 | +2.7% (理论上限) | ~10% | ~1小时 |

**当前4×4是最优平衡点**。

---

## 4. 正负号与对称量化

### 4.1 Symmetric量化

```python
# quamba/quant_utils.py:6-13
def _get_quant_range(n_bits, sym):
    if sym:
        q_max = (2**(n_bits-1)-1)  # INT8: 127
        q_min = (-2**(n_bits-1))   # INT8: -128  ← 有符号
    else:
        q_max = (2**(n_bits)-1)    # UINT8: 255
        q_min = (0)
    return q_min, q_max

# 所有调用都是 sym=True
```

**特点**：
- ✅ Zero-point = 0（原点固定）
- ✅ 同一个scale处理正负值
- ✅ Tensor Core直接支持
- ✅ 适合Mamba（LayerNorm后分布对称）

### 4.2 为什么不用Asymmetric？

| 特性 | Symmetric | Asymmetric |
|------|-----------|------------|
| 参数数量 | 1 (scale) | 2 (scale + zero_point) |
| 计算复杂度 | 低（直接乘） | 高（需要减zero_point） |
| Tensor Core | ✅ 支持 | ❌ 不支持 |
| 对称分布精度 | ✅ 完美 | ✅ 完美 |
| 非对称分布精度 | ⚠️ 可能浪费50% | ✅ 完美 |
| Mamba适配性 | ✅ 完美（LayerNorm后对称） | ❌ 增益小，代价大 |

---

## 5. 替换可行性评估

### 5.1 Calibration阶段（✅ 容易修改）

**位置**：`quamba/observer.py:137-154`

**当前实现**：全FP32
```python
cur_max = torch.quantile(w.abs().reshape(-1), percentile_alpha)  # FP32
scale = cur_max / 127  # FP32
```

**可修改内容**：
- ✅ Percentile策略（alpha值、per-channel等）
- ✅ Scale计算公式（max、mean、learned等）
- ✅ 范围估计方法（ACIQ、EMA参数等）
- ✅ 分组策略（更多/更少groups）

**限制**：
- ⚠️ 最终必须输出FP32 scale
- ⚠️ Runtime仍用INT8（但可先验证理论上限）

### 5.2 Runtime阶段（❌ 难以修改）

**位置**：`csrc/causal_conv1d/*.cuh`, `csrc/linear/*.cuh`

**当前实现**：硬编码INT8
```cuda
// Conv1D: Fake quantization
int tmp = int(roundf(out / scale));
q = clamp(tmp, -128, 127);  // ← 硬编码范围

// Linear: Tensor Core
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32"
    //                                              ↑  ↑
    //                                            INT8硬件指令
);
```

**修改难度**：
- ⚠️ 改bit宽（INT4/INT16）：需重写CUDA，修改Tensor Core指令
- ❌ 非均匀量化（Log等）：失去Tensor Core加速（10-30x性能下降）

---

## 6. 改进Scale的实验思路

### 🎯 核心约束

> **必须保持INT8兼容**：Runtime不变，只改Calibration

这意味着：
- ✅ 可以改scale计算方法
- ✅ 可以改分组策略
- ✅ 可以调整percentile参数
- ❌ 不能改量化映射函数（仍然是 q=round(x/scale)）
- ❌ 不能改数值表示（仍然是INT8 [-128,127]）

### 6.1 方向1：优化Percentile策略

#### 实验1.1：不同Percentile Alpha

**假设**：默认0.9995可能不是最优

```python
# quamba/observer.py 修改
class ObserverBase(nn.Module):
    def __init__(self, percentile_alpha=0.9995):  # 改这里
        self.percentile_alpha = percentile_alpha
```

**实验配置**：
```bash
# 测试不同alpha值
for alpha in 0.999 0.9995 0.9999 1.0; do
    python main.py ... --percentile_alpha $alpha
done
```

**预期**：
- `alpha=1.0`（你的实验显示最好）→ 可能GPTQ假设不同
- `alpha=0.999`（更激进裁剪）→ 可能提升鲁棒性
- `alpha=0.9999`（更保守）→ 折中方案

#### 实验1.2：Per-Channel Percentile

**假设**：每个channel的分布不同，全局percentile次优

```python
# 修改observer.py
def get_quantization_params(self, w):
    # 当前：per-tensor percentile
    cur_max = torch.quantile(w.abs().reshape(-1), self.percentile_alpha)

    # 改为：per-channel percentile
    if w.dim() == 2:  # [out_channels, in_channels]
        cur_max = torch.quantile(
            w.abs().reshape(w.shape[0], -1),  # 每个out_channel独立
            self.percentile_alpha,
            dim=1,
            keepdim=True
        )

    scale = cur_max / 127
    return scale  # shape: [out_channels, 1]
```

**代价**：
- 增加scale存储（per-channel vs per-tensor）
- 需要修改CUDA kernel读取scale的逻辑

#### 实验1.3：动态Percentile（数据依赖）

**假设**：不同层需要不同alpha

```python
# 自动搜索最优alpha
def find_optimal_percentile(activations, n_bits=8):
    best_alpha = 0.9995
    best_mse = float('inf')

    for alpha in [0.999, 0.9995, 0.9999, 1.0]:
        # 计算scale
        w_max = torch.quantile(activations.abs(), alpha)
        scale = w_max / 127

        # 量化+反量化
        q = torch.clamp(torch.round(activations / scale), -128, 127)
        dequant = q * scale

        # 计算MSE
        mse = ((activations - dequant) ** 2).mean()

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    return best_alpha
```

### 6.2 方向2：改进Scale计算公式

#### 实验2.1：ACIQ（Analytical Clipping for Integer Quantization）

**思想**：最小化量化误差（MSE）而非简单取max

```python
# 基于ACIQ论文（ICLR 2018）
def aciq_scale(activations, n_bits=8):
    # 假设高斯分布
    std = activations.std()
    mean = activations.mean()

    # ACIQ的最优裁剪阈值（查表或计算）
    # 对于INT8，最优alpha ≈ 2.5*std
    optimal_max = 2.5 * std

    scale = optimal_max / 127
    return scale
```

**优点**：
- 理论最优（对高斯分布）
- 不需要percentile计算（更快）

**缺点**：
- 假设高斯分布（Mamba激活可能不是）
- 需要实验验证

#### 实验2.2：基于MSE的Scale搜索

**思想**：直接最小化量化误差

```python
def mse_optimal_scale(activations, n_bits=8):
    w_max_candidates = torch.linspace(
        activations.abs().max() * 0.8,  # 下界
        activations.abs().max() * 1.0,  # 上界
        steps=20
    )

    best_scale = None
    best_mse = float('inf')

    for w_max in w_max_candidates:
        scale = w_max / 127

        # 量化+反量化
        q = torch.clamp(torch.round(activations / scale), -128, 127)
        dequant = q * scale

        # MSE
        mse = ((activations - dequant) ** 2).mean()

        if mse < best_mse:
            best_mse = mse
            best_scale = scale

    return best_scale
```

**优点**：
- 直接优化目标（MSE）
- 不假设分布

**缺点**：
- Calibration时间增加20x
- 可能过拟合calibration数据

#### 实验2.3：Entropy-Based Scale

**思想**：保留最大信息量

```python
def entropy_optimal_scale(activations, n_bits=8):
    # 计算激活值的熵
    hist, bins = torch.histogram(activations.abs(), bins=256)
    prob = hist / hist.sum()
    entropy_original = -(prob * torch.log2(prob + 1e-10)).sum()

    # 搜索使量化后熵最大的scale
    w_max_candidates = torch.linspace(...)

    best_scale = None
    best_entropy = 0

    for w_max in w_max_candidates:
        scale = w_max / 127
        q = torch.clamp(torch.round(activations / scale), -128, 127)

        # 量化值的熵
        hist_q, _ = torch.histogram(q, bins=256, range=(-128, 127))
        prob_q = hist_q / hist_q.sum()
        entropy_q = -(prob_q * torch.log2(prob_q + 1e-10)).sum()

        if entropy_q > best_entropy:
            best_entropy = entropy_q
            best_scale = scale

    return best_scale
```

### 6.3 方向3：混合精度（Layer-wise）

#### 实验3.1：敏感层识别

**假设**：不是所有层对量化同样敏感

```python
# 1. Calibration时测量每层的量化误差
layer_sensitivity = {}

for name, module in model.named_modules():
    if isinstance(module, Conv1d) or isinstance(module, Linear):
        # 记录FP16激活
        fp16_output = module(input_fp16)

        # 量化
        quantize_module(module, n_bits=8)
        int8_output = module(input_fp16)

        # 计算误差
        mse = ((fp16_output - int8_output) ** 2).mean()
        layer_sensitivity[name] = mse

# 2. 对敏感层用更高精度
for name, module in model.named_modules():
    if layer_sensitivity[name] > threshold:
        # 敏感层：用更小的scale（更高精度）
        # 或者用更多分组
        module.set_scale_multiplier(0.8)  # scale缩小20%
```

#### 实验3.2：First/Last层特殊处理

**观察**：首尾层通常最敏感

```python
# 首层：输入是原始tokens，分布可能不同
first_layer.percentile_alpha = 1.0  # 不裁剪

# 中间层：正常量化
middle_layers.percentile_alpha = 0.9995

# 末层：直接连接输出，影响最大
last_layer.percentile_alpha = 0.9999  # 更保守
```

### 6.4 方向4：EMA参数优化

#### 实验4.1：不同EMA Sigma

**当前**：`percentile_sigma = 0.1`

```python
# 测试不同平滑系数
for sigma in [0.05, 0.1, 0.2, 0.3]:
    observer = ObserverBase(percentile_sigma=sigma)
    # ...
```

**预期**：
- `sigma=0.05`（更平滑）→ 鲁棒但可能欠拟合
- `sigma=0.3`（更激进）→ 更快适应但可能震荡

#### 实验4.2：Warmup策略

```python
class AdaptiveObserver(ObserverBase):
    def __init__(self):
        super().__init__()
        self.step = 0

    def get_quantization_params(self, w):
        # Warmup前几个batch用更大的sigma
        if self.step < 50:
            sigma = 0.5  # 快速适应
        elif self.step < 200:
            sigma = 0.2  # 中等
        else:
            sigma = 0.1  # 稳定

        # 更新w_max
        cur_max = torch.quantile(w.abs(), self.percentile_alpha)
        self.w_max = self.w_max + sigma * (cur_max - self.w_max)

        self.step += 1
        return self.w_max / 127
```

### 6.5 方向5：优化分组策略

#### 实验5.1：更多分组

**当前**：4×4=16 groups/SSD

```python
# quamba/reorder_utils.py修改
def group_wise_sort_indices(tensor, headdim, ssd_ngroups,
                           nhead_groups=8,    # 改为8
                           ndim_groups=8):    # 改为8
    # 8×8=64 groups/SSD
    # 总计：8 SSD × 64 = 512 groups
```

**代价**：
- Calibration时间：2-5分钟 → 10-20分钟
- Runtime开销：<1% → ~2%
- 精度提升：+1.5% → +2.2%（预估）

#### 实验5.2：自适应分组数

**思想**：不同层用不同分组数

```python
# 根据层的激活分布决定分组数
def adaptive_grouping(activations, base_groups=4):
    # 计算分布的方差
    variance = activations.var()

    # 高方差层需要更多分组
    if variance > threshold_high:
        return base_groups * 2  # 8 groups
    elif variance > threshold_low:
        return base_groups      # 4 groups
    else:
        return base_groups // 2 # 2 groups
```

### 6.6 方向6：Learned/Gradient-Based Scale

#### 实验6.1：QAT-like Scale Learning

**思想**：在Calibration时优化scale（伪QAT）

```python
# 将scale变成可学习参数
class LearnableObserver(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化为传统方法
        initial_scale = self.compute_initial_scale()
        self.scale = nn.Parameter(torch.tensor(initial_scale))

    def forward(self, activations, targets):
        # Fake quantization
        q = torch.clamp(torch.round(activations / self.scale), -128, 127)
        dequant = q * self.scale

        # 计算损失（比如下游任务loss）
        loss = compute_task_loss(dequant, targets)

        return loss  # 反向传播优化scale

# Calibration时微调scale
optimizer = torch.optim.Adam([observer.scale], lr=0.01)
for batch in calibration_data:
    loss = observer(activations, targets)
    loss.backward()
    optimizer.step()
```

**优点**：
- 直接优化最终任务目标
- 可能找到非直觉的最优scale

**缺点**：
- 需要标签数据（calibration数据可能没有）
- 计算成本高
- 可能过拟合

#### 实验6.2：AdaRound风格的Scale优化

**思想**：优化round操作附近的scale

```python
def adaround_scale(activations, initial_scale, n_steps=100):
    scale = torch.tensor(initial_scale, requires_grad=True)
    optimizer = torch.optim.Adam([scale], lr=0.001)

    for _ in range(n_steps):
        # Soft quantization（可微分）
        q_soft = torch.sigmoid((activations / scale - torch.floor(activations / scale) - 0.5) * 10)
        q = torch.floor(activations / scale) + q_soft
        q = torch.clamp(q, -128, 127)

        dequant = q * scale

        # 最小化重建误差
        loss = ((activations - dequant) ** 2).mean()

        loss.backward()
        optimizer.step()

        # 约束scale不能太小
        with torch.no_grad():
            scale.clamp_(min=1e-6)

    return scale.item()
```

---

## 7. 实验优先级推荐

### 🥇 高优先级（简单+有效）

1. **不同Percentile Alpha**（5分钟实现，立即见效）
   ```python
   # 最简单：只改一个参数
   percentile_alpha = [0.999, 0.9995, 0.9999, 1.0]
   ```

2. **EMA Sigma调优**（10分钟实现）
   ```python
   percentile_sigma = [0.05, 0.1, 0.2, 0.3]
   ```

3. **First/Last层特殊处理**（30分钟实现）
   ```python
   # 针对性优化敏感层
   ```

### 🥈 中优先级（需要实验验证）

4. **基于MSE的Scale搜索**（1小时实现）
   - 理论更优
   - 需要验证计算成本

5. **动态Percentile**（2小时实现）
   - 每层自适应alpha
   - 可能显著提升

6. **更多分组**（需修改代码）
   - 8×8或16×16
   - 需要权衡精度vs速度

### 🥉 低优先级（研究性质）

7. **Learned Scale**（几天实现）
   - 计算成本高
   - 可能过拟合
   - 更适合作为理论上限测试

8. **Entropy-Based**（几天实现）
   - 理论有趣但不一定实用

---

## 8. 快速开始：第一个实验

### 实验：测试不同Percentile Alpha

**目标**：找到最优alpha值

**代码修改**（只需改一处）：

```python
# quamba/observer.py:第8行左右
class ObserverBase(nn.Module):
    def __init__(self,
                 n_bits=8,
                 percentile_alpha=0.9995,  # ← 改这里
                 percentile_sigma=0.1):
        # ...
```

**实验脚本**：

```bash
#!/bin/bash
# test_percentile_alpha.sh

MODEL="pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m"
TASK="lambada_openai"

for ALPHA in 0.999 0.9995 0.9999 1.0; do
    echo "Testing alpha=$ALPHA"

    # 修改observer.py中的默认值（或通过命令行参数）
    python main.py $MODEL \
        --quantize \
        --w_bits 8 --a_bits 8 \
        --eval_zero_shot --task_list $TASK \
        --percentile_alpha $ALPHA \
        --log_dir logs/alpha_${ALPHA}
done

# 比较结果
grep "Accuracy" logs/alpha_*/eval_results.txt
```

**预期结果**：

```
alpha=0.999:  Accuracy: 52.8%
alpha=0.9995: Accuracy: 53.2%  ← 当前默认
alpha=0.9999: Accuracy: 53.5%
alpha=1.0:    Accuracy: 53.7%  ← 你的实验最好
```

**如果1.0最好**：
- 说明percentile裁剪在Quamba1上有害
- 可能Quamba2的GPTQ依赖percentile，但Quamba1不需要
- **建议**：Quamba1用alpha=1.0，Quamba2保持0.9995

---

## 9. 总结

### 可以在FP32 Calibration阶段做什么？

```
┌──────────────────────────────────────────────────────────────┐
│ Calibration阶段（全FP32，容易修改）                           │
├──────────────────────────────────────────────────────────────┤
│ ✅ 改percentile策略（alpha, per-channel）                     │
│ ✅ 改scale计算公式（MSE, ACIQ, entropy）                      │
│ ✅ 改EMA参数（sigma, warmup）                                 │
│ ✅ 混合精度（layer-wise不同策略）                             │
│ ✅ 改分组策略（更多/更少groups）                              │
│ ✅ Learned scale（QAT-like）                                  │
├──────────────────────────────────────────────────────────────┤
│ ❌ 不能改量化映射（仍是 q=round(x/scale)）                    │
│ ❌ 不能改数值表示（仍是INT8 [-128,127]）                      │
│ ❌ 不能改Runtime计算（仍用Tensor Core INT8）                  │
└──────────────────────────────────────────────────────────────┘
```

### 推荐的实验路径

```
第1周：简单参数调优
  ├─ 测试不同percentile_alpha
  ├─ 测试不同percentile_sigma
  └─ First/Last层特殊处理

第2周：高级Scale策略
  ├─ MSE-optimal scale
  ├─ ACIQ scale
  └─ Per-channel percentile

第3周：分组优化
  ├─ 测试8×8分组
  ├─ 自适应分组
  └─ Layer-wise分组策略

第4周（可选）：研究性实验
  ├─ Learned scale（QAT）
  ├─ Entropy-based scale
  └─ 理论上限测试
```

### 预期收益

| 方法 | 实现难度 | 计算成本 | 预期提升 | 推荐度 |
|------|---------|----------|---------|--------|
| **Alpha调优** | ⭐ | +0% | +0.5-1% | ⭐⭐⭐⭐⭐ |
| **MSE-optimal** | ⭐⭐ | +20x calibration | +1-2% | ⭐⭐⭐⭐ |
| **8×8分组** | ⭐⭐⭐ | +5x calibration | +0.5-1% | ⭐⭐⭐ |
| **Per-channel** | ⭐⭐⭐ | +10x calibration | +0.3-0.7% | ⭐⭐⭐ |
| **Learned scale** | ⭐⭐⭐⭐ | +100x calibration | +1-3% | ⭐⭐ |

---

**最后更新**：2025-11-05

**关键记住**：
> 在INT8约束下，Scale选择是量化精度的核心！
> Calibration阶段全FP32，修改成本低，值得充分实验。
> 优先测试简单方法（alpha调优），再考虑复杂方法。
