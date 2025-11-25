# Mamba量化方法文献综述

**创建时间**: 2025-11-05
**数据来源**: arXiv搜索 + 论文原文
**目的**: 总结现有Mamba PTQ方法及Quamba的优势

---

## 📚 现有Mamba量化方法总览

### 时间线

```
2024-07  Mamba-PTQ        (首次探索，识别outlier问题)
2024-10  Quamba          (首个完整Language Mamba PTQ方案)
2024-12  PTQ4VM          (Visual Mamba)
2025-01  QMamba          (Visual Mamba，21%提升)
2025-01  MambaQuant      (KLT rotation方法)
2025-03  Quamba2         (支持Mamba2，3×加速)
```

---

## 🔬 各方法详细分析

### 1. Mamba-PTQ (Jul 2024)

**论文**: *Mamba-PTQ: Outlier Channels in Recurrent Large Language Models* (arXiv 2407.12397)

**关键发现**:
- ⭐ **首次识别**：Mamba模型存在与Transformer类似的outlier channels
- 问题：Activation outliers导致量化困难
- 贡献：提供baseline结果，提出outlier-aware量化的初步方案

**局限**:
- 只是初步探索（ICML 2024 workshop）
- 没有完整的解决方案
- 性能未达到实用水平

---

### 2. Quamba (Oct 2024)

**论文**: *Quamba: A Post-Training Quantization Recipe for Selective State Space Models* (arXiv 2410.13229)

#### 核心贡献

**技术创新**:
1. **Input Activation处理**：Percentile clipping (99.999th)
   - 抑制selective SSM输入的最大值
   - 获得更精细的量化精度

2. **Output Activation处理**：Hadamard transform
   - 在outlier-free空间量化输出
   - Fused operation，无额外开销

**对比实验** (Mamba 2.8B):

| 方法 | Perplexity (↓) | Zero-shot Acc (↑) | Latency (ms) |
|------|---------------|-----------------|-------------|
| FP16 Baseline | 9.45 | 63.1% | 103.56 |
| **SmoothQuant-SSM** | 13.59 | 57.3% | 56.53 |
| **QuaRot-SSM** | 9.89 | 62.4% | 67.76 |
| **Quamba** | **9.91** | **62.2%** | **60.17** |

**关键优势**:
- ✅ **精度**: 只有0.9%准确率下降（vs FP16）
- ✅ **速度**: 1.72× speedup (vs FP16)
- ✅ **鲁棒性**: 在Jamba-52B混合模型上成功（naive方法失败）

**局限**:
- ❌ 只支持Mamba1
- ❌ 只支持W8A8
- ❌ 没有针对Mamba2的优化

---

### 3. MambaQuant (Jan 2025)

**论文**: *MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods* (arXiv 2501.13484)

#### 核心技术

**创新方法**:
1. **KLT-Enhanced Rotation**:
   - Karhunen-Loeve Transform
   - 自适应channel分布的variance

2. **Smooth-Fused Rotation**:
   - 平衡weights和activations的channel variance
   - 将额外参数融合到模型权重

**识别的问题**:
- Gate projections、output projections、matmul存在显著outliers
- Parallel scan机制放大outliers
- 产生不均匀、重尾分布

**性能**:
- ✅ W8A8精度损失 < 1%（vision + language）
- ✅ 对比：QuaRot在Vim-T上损失21%，MambaQuant几乎无损

**定位**:
- "首个comprehensive PTQ framework for Mamba family"

---

### 4. QMamba (Jan 2025)

**论文**: *QMamba: Post-Training Quantization for Vision State Space Models* (arXiv 2501.13624)

#### 针对Vision Mamba的挑战

**识别的问题**:
1. **离散参数分布**: Long-tailed skewness
2. **Hidden state动态性**: 高度动态变化

**创新技术**:
1. **Long-tailed Skewness Quantization (LtSQ)**:
   - 处理skewed distribution
   - 减少离散参数的量化误差

2. **Temporal Group Quantization (TGQ)**:
   - 处理hidden state的动态变化
   - 时序分组量化

**性能**:
- ✅ ImageNet 4-bit activation: **+21.0%** vs 其他方法
- ✅ 多种模型架构和尺寸上超越现有PTQ方法

**特点**:
- 专门针对Vision Mamba（不是Language）
- 首个Vision SSM PTQ框架

---

### 5. PTQ4VM (Dec 2024)

**论文**: *PTQ4VM: Post-Training Quantization for Visual Mamba* (arXiv 2412.20386)

#### 首个Visual Mamba量化comprehensive study

**识别的问题**:
1. Token-wise variance
2. Channel-wise outliers
3. Long tail of activations

**技术方法**:
1. **Per-Token Static Quantization (PTS)**
2. **Joint Learning of Smoothing Scale and Step Size (JLSS)**

**定位**:
- 首个Visual Mamba量化的全面研究
- 为后续工作（QMamba等）奠定基础

---

### 6. Quamba2 (Mar 2025)

**论文**: *Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models* (arXiv 2503.22879)

#### Quamba的升级版

**扩展支持**:
- ✅ Mamba1 + Mamba2
- ✅ W8A8, W4A8, W4A16

**核心创新**:
1. **Input Quantization**:
   - Sorting + Clustering
   - 利用channel-order preservation
   - 利用activation persistence

2. **Parameter Quantization**:
   - Per-state-group量化（B和C参数）

3. **Compute-Invariant优化**:
   - Offline重排权重
   - 保持SSM output一致性

**性能提升** (Quamba2-8B):
- ✅ Prefilling: **1.3× speedup**
- ✅ Generation: **3× speedup**
- ✅ Memory: **4× reduction**
- ✅ Accuracy: 只损失1.6%

**对比**:
- 超越"two state-of-the-art SSM quantization methods"
- 论文未明确指出是哪两个（可能是MambaQuant和QMamba）

---

## 🆚 方法对比总结

### Language Mamba PTQ

| 方法 | 时间 | 技术路线 | 主要优势 | 局限 |
|------|------|---------|---------|------|
| **Mamba-PTQ** | 2024-07 | Outlier识别 | 首次探索 | 不完整 |
| **Quamba** | 2024-10 | Percentile + Hadamard | 精度+速度平衡 | 只支持Mamba1 W8A8 |
| **MambaQuant** | 2025-01 | KLT rotation | <1%精度损失 | 计算开销？ |
| **Quamba2** | 2025-03 | Clustering + Piecewise | **3×加速，支持多配置** | - |

### Vision Mamba PTQ

| 方法 | 时间 | 技术路线 | 主要优势 |
|------|------|---------|---------|
| **PTQ4VM** | 2024-12 | PTS + JLSS | 首个comprehensive study |
| **QMamba** | 2025-01 | LtSQ + TGQ | **+21%精度 (4-bit)** |

---

## 🎯 Quamba系列的独特优势

### 1. **首个Language Mamba完整方案** (Quamba)

**vs Mamba-PTQ**:
- ✅ 完整解决方案（不只是识别问题）
- ✅ 实用性能（0.9%精度损失，1.72×加速）
- ✅ 在Jamba混合模型上成功

**vs SmoothQuant/QuaRot改编版**:
- ✅ SSM特定设计（不是Transformer改编）
- ✅ 更好的精度-速度权衡
- ✅ 更低的overhead（fused operations）

### 2. **系统级优化** (Quamba2)

**vs MambaQuant**:
- ✅ 更快的推理速度（3× generation）
- ✅ 支持多种bit-width配置（W4A8, W4A16）
- ✅ 同时支持Mamba1和Mamba2

**技术差异**:
```
MambaQuant路线：
  Rotation-based (KLT + Smooth-fused)
  → 精度优先（<1%损失）
  → 可能有rotation计算开销

Quamba2路线：
  Clustering-based (Sorting + Piecewise scale)
  → 速度优先（3×加速）
  → Offline优化，runtime无额外开销
```

### 3. **部署友好性**

**Quamba系列的工程优势**:
- ✅ **Fused operations**: Hadamard transform融合，无额外开销
- ✅ **Static quantization**: 完全静态，无runtime计算
- ✅ **边缘设备实测**: Orin Nano 8G实时生成（13 tokens/sec）
- ✅ **开源完整**: 代码 + 预训练模型 + CUDA kernels

**对比**:
- MambaQuant: 学术方法，未见部署数据
- QMamba: 针对Vision，不是Language
- PTQ4VM: 针对Visual Mamba

---

## 📊 性能数据汇总

### Language Mamba 2.8B (Zero-shot Average)

| 方法 | Accuracy | vs FP16 | Latency (Orin Nano) | Speedup |
|------|----------|---------|-------------------|---------|
| FP16 Baseline | 63.1% | - | 103.56 ms | 1.0× |
| SmoothQuant-SSM | 57.3% | -5.8% | 56.53 ms | 1.83× |
| QuaRot-SSM | 62.4% | -0.7% | 67.76 ms | 1.53× |
| **Quamba** | **62.2%** | **-0.9%** | **60.17 ms** | **1.72×** |

### Mamba2-8B (Quamba2)

| 阶段 | Speedup | Memory | Accuracy Drop |
|------|---------|--------|--------------|
| Prefilling | 1.3× | 4× reduction | 1.6% |
| Generation | **3×** | 4× reduction | 1.6% |

### Vision Mamba (QMamba, 4-bit activation)

| 方法 | ImageNet Acc | Improvement |
|------|-------------|-------------|
| Existing PTQ | ~XX% | - |
| **QMamba** | **+21.0%** | 显著提升 |

---

## 💡 技术路线对比

### Rotation-based方法

**代表**: MambaQuant, QuaRot-SSM

**原理**:
```
激活值 → Hadamard/KLT rotation → 量化 → 反rotation
```

**优势**:
- ✅ 精度高（理论上optimal）
- ✅ 通用性强

**劣势**:
- ⚠️ Runtime开销（rotation计算）
- ⚠️ 内存开销（rotation矩阵）

### Smoothing-based方法

**代表**: SmoothQuant-SSM

**原理**:
```
激活值 → Smoothing (scale equalization) → 量化
```

**优势**:
- ✅ 简单
- ✅ 开销小

**劣势**:
- ❌ 精度损失大（Mamba上-5.8%）
- ❌ 不适合SSM特性

### Clustering-based方法

**代表**: Quamba2

**原理**:
```
Offline: 分析激活值分布 → Clustering → 生成piecewise scales
Runtime: Lookup scale → 量化（无额外计算）
```

**优势**:
- ✅ **Runtime无开销**（scales预计算）
- ✅ **速度快**（3× generation）
- ✅ **细粒度控制**（128个scales）

**劣势**:
- ⚠️ 依赖Calibration质量
- ⚠️ 静态scale（分布偏移时性能下降）

### Percentile-based方法

**代表**: Quamba (original)

**原理**:
```
输入激活值 → Percentile clipping (99.999th) → 量化
输出激活值 → Hadamard transform → 量化
```

**优势**:
- ✅ 简单有效
- ✅ SSM特定优化

**劣势**:
- ⚠️ Percentile选择敏感（您的发现：alpha=1.0 > 0.9995）

---

## 🔍 Quamba的核心差异化

### 1. **架构特定设计**

**其他方法的问题**:
- SmoothQuant/QuaRot: 为Transformer设计，改编到Mamba效果不佳
- 通用PTQ方法: 未考虑SSM的selective scan特性

**Quamba的优势**:
> "Existing quantization techniques are poorly suited for SSMs due to unique architectural characteristics"

- ✅ 针对selective scan的敏感性设计
- ✅ 处理SSM特有的outlier pattern（不同于Attention）
- ✅ 利用SSM的activation persistence

### 2. **实用性优先**

**学术vs工程**:

| 维度 | MambaQuant | QMamba | Quamba系列 |
|------|-----------|--------|-----------|
| **精度** | 最优 (<1%) | 优 (+21%) | 良 (~1.6%) |
| **速度** | 未知 | 未知 | **最优 (3×)** |
| **部署** | 未知 | 未知 | **完整 (Orin实测)** |
| **开源** | 未知 | 未知 | **代码+模型+CUDA** |

**Quamba的定位**:
- 不追求理论最优精度
- **追求工程可用性**：速度 + 精度 + 部署的平衡

### 3. **混合模型支持**

**Quamba在Jamba-52B上的成功**:
- Jamba = Transformer + Mamba混合架构
- Quamba量化Mamba部分 + LLM.int8量化Transformer部分
- **成功**: 只有1.1%精度损失
- **对比**: Naive quantization完全失败

**意义**:
- 证明Quamba与其他量化方法兼容
- 为混合架构量化提供方案

---

## 📈 研究趋势分析

### 时间线总结

```
2024-07  Mamba-PTQ     → 识别问题
         ↓
2024-10  Quamba       → 首个完整方案（Language）
         ↓
2024-12  PTQ4VM       → 探索Vision Mamba
         ↓
2025-01  QMamba       → Vision优化 (+21%)
         MambaQuant   → Rotation方法
         ↓
2025-03  Quamba2      → 速度优化 (3×)
```

### 研究分化

**两条路线**:
1. **Language Mamba**: Quamba系列 vs MambaQuant
2. **Vision Mamba**: QMamba vs PTQ4VM

**未来方向**:
- 更低bit-width（W4A4, W2A8）
- 混合架构量化（Jamba-like）
- QAT for Mamba
- 硬件加速器（FPGA, ASIC）

---

## 🎯 对您研究的启示

### Quamba的真实优势

**不是**:
- ❌ 最高精度（MambaQuant可能更好）
- ❌ 新的量化理论
- ❌ 通用方案

**而是**:
- ✅ **首个Language Mamba完整方案**
- ✅ **工程实用性**：速度+精度+部署的最佳平衡
- ✅ **SSM特定优化**：不是Transformer改编
- ✅ **边缘设备实测**：Orin Nano实时生成
- ✅ **开源生态**：代码+模型+CUDA kernels

### 改进方向

基于文献分析，您可以考虑：

1. **借鉴MambaQuant的rotation方法**:
   - KLT-enhanced rotation可能提升精度
   - 但需要评估runtime开销

2. **优化Percentile策略**:
   - 您的发现：alpha=1.0 > 0.9995
   - Quamba用99.999th，可能过于保守
   - 建议：Per-layer adaptive percentile

3. **结合QMamba的temporal方法**:
   - TGQ处理hidden state动态性
   - 可能适用于language Mamba的长序列

4. **扩展混合架构支持**:
   - Quamba在Jamba上的成功
   - 可以探索更多混合架构

---

## 📚 参考文献

### Quamba系列
1. Chiang et al., "Quamba: A Post-Training Quantization Recipe for Selective State Space Models", arXiv:2410.13229, Oct 2024
2. Chiang et al., "Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models", arXiv:2503.22879, Mar 2025

### 其他Mamba PTQ
3. "Mamba-PTQ: Outlier Channels in Recurrent Large Language Models", arXiv:2407.12397, Jul 2024
4. "MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods", arXiv:2501.13484, Jan 2025

### Vision Mamba PTQ
5. "QMamba: Post-Training Quantization for Vision State Space Models", arXiv:2501.13624, Jan 2025
6. "PTQ4VM: Post-Training Quantization for Visual Mamba", arXiv:2412.20386, Dec 2024

### 通用方法（Mamba改编）
7. SmoothQuant: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", 2023
8. QuaRot: Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", 2024

---

**创建时间**: 2025-11-05
**数据来源**: arXiv搜索 (2024-2025)
**总结**: Quamba的优势在于**首个Language Mamba完整方案 + 工程实用性**，而非理论精度最优。
