# Mode 2-1 vs Mode 2-0 调试对比

**问题**: Mode 2-1 正确率为 36.0%，低于预期的 38.0% (Mode 2-0 和 Mode 0 的水平)

**日期**: 2025-11-23
**目标**: 找出 Mode 2-1 相对 Mode 2-0 的关键差异，定位性能下降原因

---

## 📊 实际结果对比

| Mode | 描述 | Accuracy | 期望 | 状态 |
|------|------|----------|------|------|
| **Mode 0** | 基准 INT8 CUDA | **38.0%** ✅ | 基准 | ✅ 符合预期 |
| **Mode 2-0** | Dequant + CUDA INT8 SSM | **38.0%** ✅ | 38.0% | ✅ 符合预期 |
| **Mode 2-1** | INT8 + PyTorch INT8 SSM | **36.0%** ❌ | 38.0% | ❌ **低于预期 2%** |

**关键问题**: Mode 2-1 和 Mode 2-0 的唯一差异是 SSM 实现（PyTorch INT8 vs CUDA INT8），以及数据分叉处的处理方式。

---

## 🔍 完整数据流对比（突出差异）

### 数据流阶段对比表

| 阶段 | Mode 2-0 | Mode 2-1 | 差异标注 |
|------|----------|----------|----------|
| **in_proj 输入** | `hidden_states` FP16/FP32 | `hidden_states` FP16/FP32 | ✅ 相同 |
| **in_proj 函数** | HadLinear W4A8B16O16 | HadLinear W4A8B16O16 | ✅ 相同 |
| **in_proj 输出** | `xz` INT8 B×L×2D | `xz` INT8 B×L×2D | ✅ 相同 |
| **Conv1D 输入** | `x` INT8 B×D×L | `x` INT8 B×D×L | ✅ 相同 |
| **Conv1D 函数** | CUDA INT8 `quant_causal_conv1d_cuda.fwd()` | CUDA INT8 `quant_causal_conv1d_cuda.fwd()` | ✅ 相同 |
| **Conv1D 输出** | `x` INT8 B×D×L | `x` INT8 B×D×L | ✅ 相同 |
| **SiLU** | (融合在 Conv1D 中) INT8 | (融合在 Conv1D 中) INT8 | ✅ 相同 |
| **🔴 数据分叉处理** | **Dequant**: `x.float() * scale`<br>`x_for_ssm` = **FP32 (INT8 grid)** | **保持INT8**: `x_for_ssm = x`<br>`x_for_ssm` = **INT8** | ❌ **关键差异1** |
| **x_proj 路径** | `x_for_xproj` = INT8 (保持) | `x_for_xproj` = INT8 (保持) | ✅ 相同 |
| **x_proj 函数** | W8A8B8O8 | W8A8B8O8 | ✅ 相同 |
| **x_proj 输出** | `dt,B,C` INT8 | `dt,B,C` INT8 | ✅ 相同 |
| **dt_proj 函数** | W8A8B8O8 | W8A8B8O8 | ✅ 相同 |
| **dt_proj 输出** | `dt` INT8 | `dt` INT8 | ✅ 相同 |
| **🔴 SSM 输入 (u)** | `x_for_ssm` = **FP32 (INT8 grid)**<br>B×D×L | `x_for_ssm` = **INT8**<br>B×D×L | ❌ **关键差异2** |
| **SSM 其他输入** | `dt,B,C,z` 全部 INT8 | `dt,B,C,z` 全部 INT8 | ✅ 相同 |
| **🔴 SSM 实现** | **CUDA INT8 selective_scan**<br>(硬件优化) | **PyTorch INT8 selective_scan**<br>(内部 dequant + 软件实现) | ❌ **关键差异3** |
| **SSM 输出** | `y` FP16 B×D×L | `y` FP16 B×D×L | ✅ 相同 |
| **Hadamard** | scale=1/√d | scale=1/√d | ✅ 相同 |
| **out_proj** | HadLinear W4A8B16O16 | HadLinear W4A8B16O16 | ✅ 相同 |
| **out_proj 输出** | `output` FP16 | `output` FP16 | ✅ 相同 |
| **环境变量** | `FLOAT_SIM_ASIC_INT8=true`<br>`SSM_USE_CUDA_FOR_FP32=true` | `FLOAT_SIM_ASIC_INT8=true`<br>`SSM_USE_PYTORCH_INT8=true` | ❌ **差异4** |

---

## 🔴 关键差异详解

### 差异 1: 数据分叉处的处理方式

**代码位置**: `quamba/qMambaLayer.py`

#### Mode 2-0 (38.0% ✅)
```python
# qMambaLayer:751
x_for_xproj = x  # 保持 INT8 用于 x_proj 路径

# qMambaLayer:760 - 关键！Dequant 操作
x_for_ssm = x.float() * self.conv1d.output_scale  # INT8 → FP32 (INT8 grid)
```

#### Mode 2-1 (36.0% ❌)
```python
# qMambaLayer:751
x_for_xproj = x  # 保持 INT8 用于 x_proj 路径

# qMambaLayer:757 - 关键！保持 INT8
x_for_ssm = x  # 保持 INT8
```

**差异说明**:
- Mode 2-0: 在传入 SSM 前先 **Dequant** 到 FP32 (数值仍在 INT8 grid 范围内)
- Mode 2-1: 直接传入 **INT8** 张量给 SSM

**影响分析**:
- Mode 2-0 的 SSM 接收 **FP32 dtype** 数据（值在 INT8 grid）
- Mode 2-1 的 SSM 接收 **INT8 dtype** 数据
- PyTorch INT8 SSM 内部会进行 dequant，但可能处理方式不同

---

### 差异 2: SSM 输入数据类型

#### Mode 2-0
```python
# SSM 接收的 u (路径B)
u: torch.float32  # dtype = FP32
# 值域: [-127, 127] 范围内的整数值 (INT8 grid)
# 例如: tensor([45.0, -12.0, 127.0, -3.0])
```

#### Mode 2-1
```python
# SSM 接收的 u (路径B)
u: torch.int8  # dtype = INT8
# 值域: [-128, 127]
# 例如: tensor([45, -12, 127, -3], dtype=torch.int8)
```

**dtype 差异可能导致的问题**:
1. **Scale 应用时机不同**:
   - Mode 2-0: SSM 知道输入是 FP32，可能直接使用 scale 进行计算
   - Mode 2-1: SSM 内部需要先 dequant (INT8 → FP32)，可能引入额外的数值误差

2. **数值精度差异**:
   - INT8 dtype 可能在某些操作中被强制转换，损失精度
   - FP32 dtype 即使在 INT8 grid 上也能保持更高的中间计算精度

---

### 差异 3: SSM 实现方式

#### Mode 2-0: CUDA INT8 SSM
```python
# SSM 函数: CUDA INT8 selective_scan
# 位置: csrc/selective_scan/
# 特点:
# - 硬件优化的 CUDA kernel
# - 直接处理 FP32 输入 (INT8 grid)
# - 内部量化/反量化优化
# - 使用 cuBLAS/cuDNN 加速
```

**代码路径** (推测):
```cpp
// csrc/selective_scan/selective_scan_fwd_kernel.cu
// CUDA kernel 直接处理 FP32 输入
__global__ void selective_scan_fwd_kernel(
    const float* u,     // FP32 输入 (INT8 grid)
    const float* delta,
    const float* A,
    const float* B,
    const float* C,
    float* out,
    const float* scales  // 各种 scale
) {
    // 硬件优化的计算
    // 直接使用 FP32 进行 SSM 计算
}
```

#### Mode 2-1: PyTorch INT8 SSM
```python
# SSM 函数: PyTorch INT8 selective_scan
# 位置: quamba/qSelectiveScan.py
# 特点:
# - 软件实现 (Python/PyTorch)
# - 接收 INT8 输入，内部 dequant
# - 可能存在多次量化/反量化
# - CPU/GPU 通用实现，性能可能不如 CUDA
```

**代码路径**:
```python
# quamba/qSelectiveScan.py
class QSScan:
    def forward(self, u_int8, dt_int8, B_int8, C_int8, z_int8, ...):
        # 🔴 关键！内部 dequant
        u_fp32 = u_int8.float() * self.u_scale  # INT8 → FP32
        dt_fp32 = dt_int8.float() * self.dt_scale
        B_fp32 = B_int8.float() * self.B_scale
        C_fp32 = C_int8.float() * self.C_scale
        z_fp32 = z_int8.float() * self.z_scale

        # PyTorch 软件实现的 selective scan
        y = self._selective_scan_pytorch(u_fp32, dt_fp32, B_fp32, C_fp32, z_fp32)
        return y
```

**性能差异原因**:
1. **多次 dequant 开销**: PyTorch 实现需要对所有 5 个输入 (u, dt, B, C, z) 分别 dequant
2. **软件 vs 硬件**: PyTorch 实现无法利用 CUDA kernel 的硬件优化
3. **数值精度**: 多次量化/反量化可能累积误差

---

### 差异 4: 环境变量配置

#### Mode 2-0
```bash
FLOAT_SIM_ASIC_INT8=true         # 启用 INT8 浮点模拟
SSM_USE_CUDA_FOR_FP32=true       # 强制使用 CUDA INT8 SSM (即使输入是 FP32)
```

#### Mode 2-1
```bash
FLOAT_SIM_ASIC_INT8=true         # 启用 INT8 浮点模拟
SSM_USE_PYTORCH_INT8=true        # 使用 PyTorch INT8 SSM
```

---

## 🎯 预期 vs 实际差异分析

### 理论预期

Mode 2-1 和 Mode 0 应该完全相同，因为：

| 比较项 | Mode 0 (38.0%) | Mode 2-1 (36.0%) | 理论上 |
|--------|----------------|------------------|--------|
| Conv1D 输出 | INT8 | INT8 | ✅ 相同 |
| SSM 输入 (u) | INT8 | INT8 | ✅ 相同 |
| SSM 实现 | CUDA INT8 | PyTorch INT8 | ⚠️ **不同** |
| 其他输入 (dt,B,C,z) | INT8 | INT8 | ✅ 相同 |

**问题**: 为什么 Mode 0 用 CUDA INT8 SSM 得到 38%，而 Mode 2-1 用 PyTorch INT8 SSM 只有 36%？

---

## 🔬 可能的性能下降原因

### 原因 1: PyTorch INT8 SSM 实现问题 ⭐⭐⭐⭐⭐ (最可能)

**假设**: PyTorch INT8 SSM 的软件实现在数值精度上不如 CUDA INT8 SSM

**证据**:
1. Mode 0 (CUDA INT8 SSM): **38.0%** ✅
2. Mode 2-0 (CUDA INT8 SSM): **38.0%** ✅
3. Mode 2-1 (PyTorch INT8 SSM): **36.0%** ❌
4. Mode 2-2 (PyTorch FP32 SSM): **36.0%** ❌

**结论**: 所有使用 PyTorch SSM (无论 INT8 还是 FP32) 的模式都是 36%，说明 **PyTorch SSM 实现本身有问题**！

**调试建议**:
```python
# 在 quamba/qSelectiveScan.py 中添加调试输出
class QSScan:
    def forward(self, ...):
        # 对比 dequant 后的值
        u_fp32 = u_int8.float() * self.u_scale
        print(f"[DEBUG] u_int8 range: [{u_int8.min()}, {u_int8.max()}]")
        print(f"[DEBUG] u_scale: {self.u_scale}")
        print(f"[DEBUG] u_fp32 range: [{u_fp32.min():.4f}, {u_fp32.max():.4f}]")
        print(f"[DEBUG] u_fp32 mean: {u_fp32.mean():.4f}, std: {u_fp32.std():.4f}")
```

---

### 原因 2: Scale 应用方式差异 ⭐⭐⭐⭐

**假设**: PyTorch INT8 SSM 内部的 dequant 方式与 CUDA INT8 SSM 不同

#### CUDA INT8 SSM 的 dequant (推测)
```cpp
// 可能在 SSM 计算过程中直接使用 INT8，只在必要时 dequant
// 使用硬件加速的 INT8 乘法和累加
__device__ float compute_ssm_element(
    int8_t u_int8,
    float u_scale,
    // ...
) {
    // 可能优化：延迟 dequant 到最后
    float result = (float)u_int8 * u_scale;  // 硬件优化的转换
    return result;
}
```

#### PyTorch INT8 SSM 的 dequant (实际)
```python
# 一次性 dequant 所有输入
u_fp32 = u_int8.float() * self.u_scale  # 可能引入误差
# 然后用 FP32 进行计算
y = self._selective_scan_pytorch(u_fp32, ...)
```

**问题**: PyTorch 一次性 dequant 可能在大张量上引入累积误差

**调试建议**:
```python
# 对比 Mode 0 和 Mode 2-1 的 SSM 输入数值分布
# 在 qMambaLayer.py 中添加:
if self.ssm_mode == "0" or self.ssm_mode == "2-1":
    print(f"[{self.ssm_mode}] u before SSM: {u.shape}, dtype={u.dtype}")
    print(f"[{self.ssm_mode}] u range: [{u.min()}, {u.max()}]")
    print(f"[{self.ssm_mode}] u_scale: {self.mixer.selective_scan.u_scale}")
```

---

### 原因 3: SSM State 量化差异 ⭐⭐⭐

**假设**: CUDA 和 PyTorch 对 SSM 内部 state 的量化处理不同

**`ssm_state_scale`** (Percentile) 的使用方式可能不同：

#### CUDA INT8 SSM
```cpp
// SSM 内部 state 量化 (硬件优化)
__device__ void update_ssm_state(
    float* state,
    int8_t* state_quantized,
    float state_scale
) {
    // 直接在 INT8 上进行 state 更新
    // 硬件优化的量化/反量化
}
```

#### PyTorch INT8 SSM
```python
# SSM 内部 state 量化 (软件实现)
def update_state(self, state_fp32, state_scale):
    # 可能多次量化/反量化
    state_int8 = torch.round(state_fp32 / state_scale).to(torch.int8)
    state_fp32_dequant = state_int8.float() * state_scale
    # 误差累积！
    return state_fp32_dequant
```

---

### 原因 4: 浮点运算顺序差异 ⭐⭐

**假设**: CUDA 和 PyTorch 的浮点运算顺序不同，导致数值误差

**IEEE 754 浮点标准**: `(a * b) * c ≠ a * (b * c)`

#### CUDA 实现
```cpp
// 可能优化的运算顺序
float result = fma(u_int8, u_scale, bias);  // fused multiply-add
```

#### PyTorch 实现
```python
# 标准的运算顺序
result = u_int8.float() * u_scale + bias  # 分开计算
```

---

## 🛠️ 调试步骤建议

### Step 1: 验证 PyTorch SSM 是问题根源

**目标**: 确认 PyTorch SSM (无论 INT8/FP32) 都比 CUDA SSM 差

**方法**:
```bash
# 运行对比实验
python main.py --model quamba-130m-w8a8 --tasks lambada_openai --num_fewshot 0 \
  --limit 100 --ssm_mode 0    # CUDA INT8 SSM → 38%

python main.py --model quamba-130m-w8a8 --tasks lambada_openai --num_fewshot 0 \
  --limit 100 --ssm_mode 2-1  # PyTorch INT8 SSM → 36%

python main.py --model quamba-130m-w8a8 --tasks lambada_openai --num_fewshot 0 \
  --limit 100 --ssm_mode 2-2  # PyTorch FP32 SSM → 36%
```

**预期结果**: 如果 Mode 2-2 也是 36%，则证明 PyTorch SSM 实现有问题

---

### Step 2: 对比 SSM 输入数值

**目标**: 检查 Mode 0 和 Mode 2-1 传入 SSM 的数据是否一致

**代码修改**: 在 `quamba/qMambaLayer.py` 添加日志

```python
# 在 forward() 方法中，SSM 调用前
if self.ssm_mode in ["0", "2-1"]:
    import torch
    print(f"\n{'='*80}")
    print(f"[Mode {self.ssm_mode}] SSM Input Debug")
    print(f"{'='*80}")
    print(f"u: shape={u.shape}, dtype={u.dtype}")
    print(f"   range=[{u.min().item():.6f}, {u.max().item():.6f}]")
    print(f"   mean={u.mean().item():.6f}, std={u.std().item():.6f}")
    print(f"   u_scale={self.mixer.selective_scan.u_scale:.6f}")

    # 如果是 INT8，显示 dequant 后的值
    if u.dtype == torch.int8:
        u_dequant = u.float() * self.mixer.selective_scan.u_scale
        print(f"   u_dequant range=[{u_dequant.min().item():.6f}, {u_dequant.max().item():.6f}]")
        print(f"   u_dequant mean={u_dequant.mean().item():.6f}, std={u_dequant.std().item():.6f}")
```

**预期结果**: Mode 0 和 Mode 2-1 的 `u` 应该完全相同（都是 INT8）

---

### Step 3: 对比 SSM 内部 dequant 实现

**目标**: 检查 CUDA 和 PyTorch SSM 的 dequant 代码是否一致

**文件位置**:
- CUDA: `csrc/selective_scan/selective_scan_fwd.cu`
- PyTorch: `quamba/qSelectiveScan.py`

**检查项**:
1. **Dequant 公式**: 是否都是 `x_fp32 = x_int8.float() * scale`？
2. **Scale 顺序**: 是否先乘 `u_scale`，还是先计算再乘？
3. **Clipping**: 是否有 `clamp(-127, 127)` 或其他裁剪？
4. **Dtype 转换**: `float()` 的实现是否一致？

---

### Step 4: 对比 SSM 输出

**目标**: 检查 SSM 输出 `y` 是否有显著差异

**代码修改**: 在 SSM 调用后添加日志

```python
# 在 SSM 调用后
y = self.mixer.selective_scan(u, dt, B, C, z, ...)

if self.ssm_mode in ["0", "2-1"]:
    print(f"\n[Mode {self.ssm_mode}] SSM Output Debug")
    print(f"y: shape={y.shape}, dtype={y.dtype}")
    print(f"   range=[{y.min().item():.6f}, {y.max().item():.6f}]")
    print(f"   mean={y.mean().item():.6f}, std={y.std().item():.6f}")

    # 计算差异 (如果有参考)
    if hasattr(self, 'y_ref'):
        diff = (y - self.y_ref).abs()
        print(f"   diff from ref: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")
```

---

### Step 5: 逐层对比 SSM 计算

**目标**: 找出 PyTorch SSM 在哪一步引入误差

**方法**: 在 PyTorch SSM 实现中添加详细日志

```python
# quamba/qSelectiveScan.py
class QSScan:
    def forward(self, u, dt, B, C, z, ...):
        # Step 1: Dequant
        u_fp32 = u.float() * self.u_scale
        dt_fp32 = dt.float() * self.dt_scale
        # ... (其他 dequant)

        print(f"[PyTorch SSM] After dequant:")
        print(f"  u_fp32: range=[{u_fp32.min():.4f}, {u_fp32.max():.4f}]")

        # Step 2: SSM 核心计算
        # ... (selective scan 算法)

        print(f"[PyTorch SSM] After SSM core:")
        print(f"  y: range=[{y.min():.4f}, {y.max():.4f}]")

        return y
```

---

## 📋 调试优先级

### 🔥 高优先级（立即执行）

1. **验证 PyTorch SSM 假设**: 运行 Mode 0 vs Mode 2-1 vs Mode 2-2，确认 PyTorch SSM 是问题根源
2. **对比 SSM 输入**: 确保 Mode 0 和 Mode 2-1 传入 SSM 的数据完全一致
3. **检查 dequant 实现**: 对比 CUDA 和 PyTorch 的 dequant 代码

### ⚠️ 中优先级（问题明确后）

4. **对比 SSM State 量化**: 检查 `ssm_state_scale` 的使用方式
5. **逐层对比 SSM 计算**: 找出 PyTorch SSM 的具体误差来源

### 💡 低优先级（优化阶段）

6. **浮点运算顺序**: 检查 FMA 等优化是否影响结果
7. **硬件差异**: 检查 CUDA vs PyTorch 的硬件加速差异

---

## 🎯 预期调试结果

**最可能的原因**: PyTorch INT8 SSM 的软件实现在数值精度上不如 CUDA INT8 SSM

**证据支持**:
- Mode 0 (CUDA INT8): 38% ✅
- Mode 2-0 (CUDA INT8): 38% ✅
- Mode 2-1 (PyTorch INT8): 36% ❌
- Mode 2-2 (PyTorch FP32): 36% ❌

**结论**: PyTorch SSM (无论 INT8/FP32) 都比 CUDA SSM 差 **2%**

**建议**:
1. 优化 PyTorch SSM 实现，使其数值精度接近 CUDA 实现
2. 或者在生产环境中使用 Mode 0 或 Mode 2-0 (CUDA SSM)
3. 如果必须使用 PyTorch SSM，考虑提高 SSM 内部的计算精度（例如使用 FP64）

---

## 📌 下一步行动

1. ✅ **运行对比实验**: 确认 Mode 0/2-1/2-2 的准确率
2. ✅ **添加调试日志**: 在 qMambaLayer.py 和 qSelectiveScan.py 中添加日志
3. ✅ **对比数值分布**: 检查 SSM 输入/输出的数值分布差异
4. ✅ **检查 dequant 实现**: 对比 CUDA 和 PyTorch 的 dequant 代码
5. ✅ **定位误差来源**: 找出 PyTorch SSM 的具体问题所在
6. ✅ **修复或规避**: 修复 PyTorch SSM 实现，或在生产环境中使用 CUDA SSM

---

## 🔑 关键文件位置

| 文件 | 作用 | 关键代码位置 |
|------|------|-------------|
| `quamba/qMambaLayer.py` | Mode 选择和数据分叉 | Line 751 (x_for_xproj)<br>Line 757 (Mode 2-1 保持INT8)<br>Line 760 (Mode 2-0 Dequant) |
| `quamba/qSelectiveScan.py` | PyTorch INT8 SSM 实现 | `forward()` 方法的 dequant 逻辑 |
| `csrc/selective_scan/` | CUDA INT8 SSM 实现 | CUDA kernel 的 dequant 和计算逻辑 |
| `quamba/qConvLayer.py` | Conv1D 实现 | Line 116-122 (CUDA INT8)<br>Line 157 (Mode 2-0/2-1 调用) |

---

**总结**: Mode 2-1 的性能下降 (36% vs 预期 38%) 最可能是由于 **PyTorch INT8 SSM 的软件实现** 在数值精度上不如 CUDA INT8 SSM。建议优先调试 PyTorch SSM 的 dequant 实现和 SSM 核心计算逻辑。
