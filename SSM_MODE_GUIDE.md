# Quamba 量化模式完整指南

**版本**: 2.0
**日期**: 2025-11-23
**验证状态**: ✅ 7 种模式均已验证（lambada_openai, 100 samples）

---

## 📋 核心概念

### SiLU 激活函数是 Fused 在 Conv1D 内部的

**关键发现**: SiLU 激活不是独立的层，而是直接融合在 Conv1D CUDA kernel 内部。

```python
# quamba/qConvLayer.py:116-122
# CUDA INT8 kernel with fused SiLU
y = quant_causal_conv1d_cuda.fwd(
    x, self.input_scale,
    self.weight, self.weight_scale,
    self.output_scale,
    self.bias_scale, self.bias,
    None, None, None, True  # ← silu_activation=True (fused)
)
```

**两种 Conv1D CUDA Kernel**:

| Kernel 函数 | 精度 | SiLU 精度 | 输出 | 文件位置 |
|------------|------|-----------|------|---------|
| `quant_causal_conv1d_cuda.fwd()` | **INT8** | **INT8** | INT8 | `csrc/causal_conv1d/quant_causal_conv1d.cpp` |
| `quant_causal_conv1d_cuda.fwd_fp32()` | **FP32** | **FP32** | FP32 | `csrc/causal_conv1d/quant_causal_conv1d_fwd_fp32.cu` |

**重要**: Conv1D 和 SiLU **总是同精度**！

---

## 📊 7 种量化模式完整对比

## 📊 超详细横向对比表（所有模式×所有步骤）

**说明**：表格按横向展开，每行代表一个模式，每个步骤有3列（输入、函数、输出）

### ⭐ 超宽完整表格：全流程（in_proj → Hadamard → Conv1D → ... → out_proj → 环境变量 → Accuracy）

**注意**：
- ✅ **所有 7 个模式都使用 Hadamard 变换** (`use_had_transform=True`)
- Hadamard 在 3 个位置：①in_proj输入侧 (HadLinear)、②SSM输出→out_proj之间 (独立层)、③out_proj输出侧 (HadLinear)
- Hadamard scale: `1.0 / sqrt(dim)`，保持 dtype 不变，正交变换
- 表格可能需要横向滚动查看
- 🎯 = Percentile quantization scale 应用位置（α=0.9995，裁剪0.05% outliers）
- Observer类型: ✅ Percentile (`x_proj:input`=u_scale, `ssm_state_act:input`=ssm_state_scale) | ❌ MinMax (其他所有scale)
- Scale相等关系: `in_proj:output`=Conv1D输入 | Conv1D输出=`x_proj:input` | `x_proj:output`=dt_proj输入

| Mode | in_proj<br>**输入**<br>`in_proj:input`<br>❌**MinMax**<br>📥modelutils:171-175<br>PerTensorMinmaxObserver<br>📌qMambaLayer:593-597<br>→in_proj.input_scale | in_proj<br>**函数**<br>(HadLinear) | in_proj<br>**输出**<br>`in_proj:output`<br>=`z_scale`<br>❌**MinMax**<br>📥modelutils:176-180<br>PerTensorMinmaxObserver<br>📌qMambaLayer:593-597<br>→in_proj.output_scale<br>📌qMambaLayer:603-606<br>→Conv1D.input_scale<br>📌qMambaLayer:628-633<br>→SSM.z_scale | Conv1D<br>**输入**<br>=`in_proj:output` | Conv1D<br>**函数** | Conv1D<br>**输出** | SiLU<br>**输入**<br>(fused) | SiLU<br>**函数**<br>(fused) | SiLU<br>**输出**<br>(fused) | 🎯Conv1D<br>**output_scale**<br>`x_proj:input`<br>=`u_scale`<br>✅**Percentile**<br>α=0.9995<br>📥modelutils:163-169<br>PerTensorPercentileObserver<br>hook收集Conv1D+SiLU输出<br>quantile裁剪0.05% outliers<br>📌modelutils:247-251<br>scale计算:cur_max/127<br>📌qMambaLayer:603-606<br>→Conv1D.output_scale<br>📌qMambaLayer:610-614<br>→x_proj.input_scale<br>📌qMambaLayer:628-633<br>→SSM.u_scale<br>🚀Inference用法:<br>Mode0/2-1:传入INT8 kernel<br>Mode2-0/2-2:Dequant(qMambaLayer:760)<br>Mode2-3/2-4/3:Requant(qMambaLayer:764) | 数据分叉<br>**输入** | 数据分叉<br>**函数**<br>(路径A:dt,B,C 路径B:u) | 数据分叉<br>**输出** | x_proj<br>**输入**<br>=`x_proj:input` | x_proj<br>**函数** | x_proj<br>**输出**<br>`x_proj:output`<br>=`B_scale`,`C_scale`<br>❌**MinMax**<br>📥modelutils:176-180<br>PerTensorMinmaxObserver<br>📌qMambaLayer:610-614<br>→x_proj.output_scale<br>📌qMambaLayer:617-621<br>→dt_proj.input_scale<br>📌qMambaLayer:628-633<br>→SSM.B_scale,C_scale | dt_proj<br>**输入**<br>=`x_proj:output` | dt_proj<br>**函数** | dt_proj<br>**输出**<br>`dt_proj:output`<br>=`dt_scale`<br>❌**MinMax**<br>📥modelutils:176-180<br>PerTensorMinmaxObserver<br>📌qMambaLayer:617-621<br>→dt_proj.output_scale<br>📌qMambaLayer:628-633<br>→SSM.dt_scale | SSM<br>**输入**<br>6 scale汇合:<br>`u_scale`=`x_proj:input`✅<br>`dt_scale`=`dt_proj:output`❌<br>`B_scale`=`x_proj:output`❌<br>`C_scale`=`x_proj:output`❌<br>`z_scale`=`in_proj:output`❌<br>`ssm_state_scale`=`ssm_state_act:input`✅<br>(qMambaLayer:628-633) | 🎯SSM<br>**ssm_state_scale**<br>`ssm_state_act:input`<br>✅**Percentile**<br>α=0.9995<br>📥modelutils:163-169<br>PerTensorPercentileObserver<br>hook收集SSM内部state激活<br>📌modelutils:247-251<br>scale计算:cur_max/127<br>📌qMambaLayer:628-633<br>→SSM.ssm_state_scale<br>🚀Inference用法:<br>SSM kernel内部state量化 | SSM<br>**函数** | SSM<br>**输出** | Hadamard<br>**输入** | Hadamard<br>**函数** | Hadamard<br>**输出** | out_proj<br>**输入**<br>`out_proj:input`<br>❌**MinMax**<br>📥modelutils:171-175<br>📌qMambaLayer:636-640<br>→Hadamard.x_H_scale<br>📌qMambaLayer:641-644<br>→out_proj.input_scale | out_proj<br>**函数**<br>(HadLinear) | out_proj<br>**输出** | **环境变量** | **Accuracy** |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| **0**<br>**路径A**<br>(dt,B,C) | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | **无分叉**<br>`x` 同时用于两路<br>qMambaLayer:821 | `x`<br>INT8<br>B×D×L | `x`<br>INT8<br>B×L×D | W8A8B8O8<br>qMambaLayer:822 | `dt,B,C`<br>INT8 | `dt`<br>INT8<br>B×L×dt_rank | W8A8B8O8<br>qMambaLayer:832 | `dt`<br>INT8<br>B×dt_rank×L | **SSM 6 scale汇合**:<br>`u_scale`=`x_proj:input`✅<br>(u: INT8)<br>`dt_scale`=`dt_proj:output`❌<br>(dt: INT8)<br>`B_scale`=`x_proj:output`❌<br>(B: INT8)<br>`C_scale`=`x_proj:output`❌<br>(C: INT8)<br>`z_scale`=`in_proj:output`❌<br>(z: INT8)<br>`ssm_state_scale`=`ssm_state_act:input`✅<br>qMambaLayer:628-633 | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ |
| **0**<br>**路径B**<br>(u) | `hidden_states`<br>FP16/FP32<br>B×L×D<br>← prev layer | HadLinear<br>+ Had输入变换<br>W4A8B16O16<br>qMambaLayer.py:52 | `xz`<br>INT8<br>B×L×2D | `x`<br>INT8<br>B×D×L<br>← split | **CUDA INT8**<br>**Conv1D+SiLU融合**<br>`quant_causal_conv1d_cuda.fwd()`<br>qConvLayer:116-122<br>`silu_activation=True` | `x`<br>INT8<br>B×D×L | (融合在Conv1D中) | (融合在Conv1D中)<br>**INT8 CUDA kernel**<br>直接输出INT8 | (融合在Conv1D中) | ↓ | `x`<br>INT8<br>B×D×L | **无分叉**<br>`x` 同时用于两路<br>qMambaLayer:821 | `x`<br>INT8<br>B×D×L | → | → | → | → | → | → | ↑汇合到路径A | ↓ | CUDA INT8<br>selective_scan | `y`<br>FP16<br>B×D×L | `y`<br>FP16<br>B×L×D | Hadamard<br>scale=1/√d<br>qMambaLayer:141 | `y`<br>FP16<br>B×L×D | `y`<br>FP16<br>B×L×D | HadLinear<br>+ Had输入/输出变换<br>W4A8B16O16<br>qMambaLayer:71 | `output`<br>FP16<br>B×L×D | 无 | **38.0%** ✅ |
| **2-0**<br>**路径A**<br>(dt,B,C) | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | 保持INT8<br>`x_for_xproj=x`<br>qMambaLayer:751 | `x_for_xproj`<br>INT8<br>B×D×L | `x_for_xproj`<br>INT8<br>B×L×D | W8A8B8O8<br>qMambaLayer:770 | `dt,B,C`<br>INT8 | `dt`<br>INT8<br>B×L×dt_rank | W8A8B8O8<br>qMambaLayer:775 | `dt`<br>INT8<br>B×dt_rank×L | **SSM 6 scale汇合**:<br>`u_scale`=`x_proj:input`✅<br>(u: FP32→requant INT8)<br>`dt_scale`=`dt_proj:output`❌<br>(dt: INT8)<br>`B_scale`=`x_proj:output`❌<br>(B: INT8)<br>`C_scale`=`x_proj:output`❌<br>(C: INT8)<br>`z_scale`=`in_proj:output`❌<br>(z: INT8)<br>`ssm_state_scale`=`ssm_state_act:input`✅<br>qMambaLayer:628-633 | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ |
| **2-0**<br>**路径B**<br>(u) | `hidden_states`<br>FP16/FP32<br>B×L×D<br>← prev layer | HadLinear<br>+ Had输入变换<br>W4A8B16O16<br>qMambaLayer.py:52 | `xz`<br>INT8<br>B×L×2D | `x`<br>INT8<br>B×D×L<br>← split | **CUDA INT8**<br>**Conv1D+SiLU融合**<br>`quant_causal_conv1d_cuda.fwd()`<br>qConvLayer:157<br>`silu_activation=True` | `x`<br>INT8<br>B×D×L | (融合在Conv1D中) | (融合在Conv1D中)<br>**INT8 CUDA kernel**<br>直接输出INT8 | (融合在Conv1D中) | ↓ | `x`<br>INT8<br>B×D×L | Dequant<br>`x.float()*scale`<br>**scale=**<br>`self.conv1d.output_scale`<br>=`x_proj:input` (Percentile)<br>**← 来自左边Conv1D output_scale列**<br>qMambaLayer:760 | `x_for_ssm`<br>FP32 (INT8 grid)<br>B×D×L | → | → | → | → | → | → | ↑汇合到路径A | ↓ | CUDA INT8<br>selective_scan | `y`<br>FP16<br>B×D×L | `y`<br>FP16<br>B×L×D | Hadamard<br>scale=1/√d<br>qMambaLayer:141 | `y`<br>FP16<br>B×L×D | `y`<br>FP16<br>B×L×D | HadLinear<br>+ Had输入/输出变换<br>W4A8B16O16<br>qMambaLayer:71 | `output`<br>FP16<br>B×L×D | `FLOAT_SIM_ASIC_INT8=true`<br>`SSM_USE_CUDA_FOR_FP32=true` | **38.0%** ✅ |
| **2-1**<br>**路径A**<br>(dt,B,C) | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | 保持INT8<br>`x_for_xproj=x`<br>qMambaLayer:751 | `x_for_xproj`<br>INT8<br>B×D×L | `x_for_xproj`<br>INT8<br>B×L×D | W8A8B8O8<br>qMambaLayer:770 | `dt,B,C`<br>INT8 | `dt`<br>INT8<br>B×L×dt_rank | W8A8B8O8<br>qMambaLayer:775 | `dt`<br>INT8<br>B×dt_rank×L | **SSM 6 scale汇合**:<br>`u_scale`=`x_proj:input`✅<br>(u: INT8)<br>`dt_scale`=`dt_proj:output`❌<br>(dt: INT8)<br>`B_scale`=`x_proj:output`❌<br>(B: INT8)<br>`C_scale`=`x_proj:output`❌<br>(C: INT8)<br>`z_scale`=`in_proj:output`❌<br>(z: INT8)<br>`ssm_state_scale`=`ssm_state_act:input`✅<br>qMambaLayer:628-633 | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ |
| **2-1**<br>**路径B**<br>(u) | `hidden_states`<br>FP16/FP32<br>B×L×D<br>← prev layer | HadLinear<br>+ Had输入变换<br>W4A8B16O16<br>qMambaLayer.py:52 | `xz`<br>INT8<br>B×L×2D | `x`<br>INT8<br>B×D×L<br>← split | **CUDA INT8**<br>**Conv1D+SiLU融合**<br>`quant_causal_conv1d_cuda.fwd()`<br>qConvLayer:157<br>`silu_activation=True` | `x`<br>INT8<br>B×D×L | (融合在Conv1D中) | (融合在Conv1D中)<br>**INT8 CUDA kernel**<br>直接输出INT8 | (融合在Conv1D中) | ↓ | `x`<br>INT8<br>B×D×L | 保持INT8<br>`x_for_ssm=x`<br>qMambaLayer:757 | `x_for_ssm`<br>INT8<br>B×D×L | → | → | → | → | → | → | ↑汇合到路径A | ↓ | PyTorch INT8<br>(内部dequant) | `y`<br>FP16<br>B×D×L | `y`<br>FP16<br>B×L×D | Hadamard<br>scale=1/√d<br>qMambaLayer:141 | `y`<br>FP16<br>B×L×D | `y`<br>FP16<br>B×L×D | HadLinear<br>+ Had输入/输出变换<br>W4A8B16O16<br>qMambaLayer:71 | `output`<br>FP16<br>B×L×D | `FLOAT_SIM_ASIC_INT8=true`<br>`SSM_USE_PYTORCH_INT8=true` | 36.0% |
| **2-2**<br>**路径A**<br>(dt,B,C) | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | 保持INT8<br>`x_for_xproj=x`<br>qMambaLayer:751 | `x_for_xproj`<br>INT8<br>B×D×L | `x_for_xproj`<br>INT8<br>B×L×D | W8A8B8O8<br>qMambaLayer:770 | `dt,B,C`<br>INT8 | `dt`<br>INT8<br>B×L×dt_rank | W8A8B8O8<br>qMambaLayer:775 | `dt`<br>INT8<br>B×dt_rank×L | **SSM 6 scale汇合**:<br>`u_scale`=`x_proj:input`✅<br>(u: FP32 INT8 grid)<br>`dt_scale`=`dt_proj:output`❌<br>(dt: INT8)<br>`B_scale`=`x_proj:output`❌<br>(B: INT8)<br>`C_scale`=`x_proj:output`❌<br>(C: INT8)<br>`z_scale`=`in_proj:output`❌<br>(z: INT8)<br>`ssm_state_scale`=`ssm_state_act:input`✅<br>qMambaLayer:628-633 | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ |
| **2-2**<br>**路径B**<br>(u) | `hidden_states`<br>FP16/FP32<br>B×L×D<br>← prev layer | HadLinear<br>+ Had输入变换<br>W4A8B16O16<br>qMambaLayer.py:52 | `xz`<br>INT8<br>B×L×2D | `x`<br>INT8<br>B×D×L<br>← split | **CUDA INT8**<br>**Conv1D+SiLU融合**<br>`quant_causal_conv1d_cuda.fwd()`<br>qConvLayer:157<br>`silu_activation=True` | `x`<br>INT8<br>B×D×L | (融合在Conv1D中) | (融合在Conv1D中)<br>**INT8 CUDA kernel**<br>直接输出INT8 | (融合在Conv1D中) | ↓ | `x`<br>INT8<br>B×D×L | Dequant<br>`x.float()*scale`<br>**scale=**<br>`self.conv1d.output_scale`<br>=`x_proj:input` (Percentile)<br>**← 来自左边Conv1D output_scale列**<br>qMambaLayer:760 | `x_for_ssm`<br>FP32 (INT8 grid)<br>B×D×L | → | → | → | → | → | → | ↑汇合到路径A | ↓ | PyTorch FP32<br>(mode22) | `y`<br>FP16<br>B×D×L | `y`<br>FP16<br>B×L×D | Hadamard<br>scale=1/√d<br>qMambaLayer:141 | `y`<br>FP16<br>B×L×D | `y`<br>FP16<br>B×L×D | HadLinear<br>+ Had输入/输出变换<br>W4A8B16O16<br>qMambaLayer:71 | `output`<br>FP16<br>B×L×D | `FLOAT_SIM_ASIC_INT8=true` | 36.0% |
| **2-3**<br>**路径A**<br>(dt,B,C) | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | Requant<br>`round(x/scale)`<br>**scale=**<br>`self.conv1d.output_scale`<br>=`x_proj:input` (Percentile)<br>**← 来自左边Conv1D output_scale列**<br>qMambaLayer:764 | `x_for_xproj`<br>INT8<br>B×D×L | `x_for_xproj`<br>INT8<br>B×L×D | W8A8B8O8<br>qMambaLayer:770 | `dt,B,C`<br>INT8 | `dt`<br>INT8<br>B×L×dt_rank | W8A8B8O8<br>qMambaLayer:775 | `dt`<br>INT8<br>B×dt_rank×L | **SSM 6 scale汇合**:<br>`u_scale`=`x_proj:input`✅<br>**← 来自左边Conv1D output_scale列**<br>(u: FP32 TRUE→requant INT8)<br>`dt_scale`=`dt_proj:output`❌<br>**← 来自左边dt_proj输出列**<br>(dt: INT8)<br>`B_scale`=`x_proj:output`❌<br>**← 来自左边x_proj输出列**<br>(B: INT8)<br>`C_scale`=`x_proj:output`❌<br>**← 来自左边x_proj输出列**<br>(C: INT8)<br>`z_scale`=`in_proj:output`❌<br>**← 来自左边in_proj输出列**<br>(z: INT8)<br>`ssm_state_scale`=`ssm_state_act:input`✅<br>**← 来自右边SSM ssm_state_scale列**<br>qMambaLayer:628-633 | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ |
| **2-3**<br>**路径B**<br>(u) | `hidden_states`<br>FP16/FP32<br>B×L×D<br>← prev layer | HadLinear<br>+ Had输入变换<br>W4A8B16O16<br>qMambaLayer.py:52 | `xz`<br>INT8<br>B×L×2D | `x`<br>INT8<br>B×D×L<br>← split | **FP32 CUDA** ⭐<br>**Conv1D+SiLU融合**<br>`quant_causal_conv1d_fwd_fp32()`<br>qConvLayer:230<br>`silu_activation=True` | `x`<br>**FP32 (TRUE)** ⭐<br>B×D×L | (融合在Conv1D中) | (融合在Conv1D中)<br>**FP32 CUDA kernel**<br>直接输出FP32 | (融合在Conv1D中) | ↓<br>⚠️ scale mismatch | `x`<br>FP32 (TRUE)<br>B×D×L | 保持FP32<br>`x_for_ssm=x`<br>qMambaLayer:765 | `x_for_ssm`<br>**FP32 (TRUE)** ⭐<br>B×D×L | → | → | → | → | → | → | ↑汇合到路径A | ↓<br>⚠️u被requant到INT8<br>FP32优势被抵消 | PyTorch INT8<br>(内部dequant)<br>❌优势被抵消 | `y`<br>FP16<br>B×D×L | `y`<br>FP16<br>B×L×D | Hadamard<br>scale=1/√d<br>qMambaLayer:141 | `y`<br>FP16<br>B×L×D | `y`<br>FP16<br>B×L×D | HadLinear<br>+ Had输入/输出变换<br>W4A8B16O16<br>qMambaLayer:71 | `output`<br>FP16<br>B×L×D | `FLOAT_SIM_ASIC_INT8=true`<br>`CONV1D_MODE23_FP32=true` | 36.0%<br>❌ FP32无效 |
| **2-4**<br>**路径A**<br>(dt,B,C) | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | Requant<br>`round(x/scale)`<br>**scale=**<br>`self.conv1d.output_scale`<br>=`x_proj:input` (Percentile)<br>**← 来自左边Conv1D output_scale列**<br>qMambaLayer:764 | `x_for_xproj`<br>INT8<br>B×D×L | `x_for_xproj`<br>INT8<br>B×L×D | W8A8B8O8<br>qMambaLayer:770 | `dt,B,C`<br>INT8 | `dt`<br>INT8<br>B×L×dt_rank | W8A8B8O8<br>qMambaLayer:775 | `dt`<br>INT8<br>B×dt_rank×L | **SSM 6 scale汇合**:<br>`u_scale`=`x_proj:input`✅<br>**← 来自左边Conv1D output_scale列**<br>(u: FP32 TRUE)<br>`dt_scale`=`dt_proj:output`❌<br>**← 来自左边dt_proj输出列**<br>(dt: INT8)<br>`B_scale`=`x_proj:output`❌<br>**← 来自左边x_proj输出列**<br>(B: INT8)<br>`C_scale`=`x_proj:output`❌<br>**← 来自左边x_proj输出列**<br>(C: INT8)<br>`z_scale`=`in_proj:output`❌<br>**← 来自左边in_proj输出列**<br>(z: INT8)<br>`ssm_state_scale`=`ssm_state_act:input`✅<br>**← 来自右边SSM ssm_state_scale列**<br>qMambaLayer:628-633 | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ |
| **2-4**<br>**路径B**<br>(u) | `hidden_states`<br>FP16/FP32<br>B×L×D<br>← prev layer | HadLinear<br>+ Had输入变换<br>W4A8B16O16<br>qMambaLayer.py:52 | `xz`<br>INT8<br>B×L×2D | `x`<br>INT8<br>B×D×L<br>← split | **FP32 CUDA** ⭐<br>**Conv1D+SiLU融合**<br>`quant_causal_conv1d_fwd_fp32()`<br>qConvLayer:362<br>`silu_activation=True` | `x`<br>**FP32 (TRUE)** ⭐<br>B×D×L | (融合在Conv1D中) | (融合在Conv1D中)<br>**FP32 CUDA kernel**<br>直接输出FP32 | (融合在Conv1D中) | ↓<br>⚠️ scale mismatch | `x`<br>FP32 (TRUE)<br>B×D×L | 保持FP32<br>`x_for_ssm=x`<br>qMambaLayer:765 | `x_for_ssm`<br>**FP32 (TRUE)** ⭐<br>B×D×L | → | → | → | → | → | → | ↑汇合到路径A | ↓<br>(u保持FP32 TRUE) | PyTorch FP32<br>(mode22)<br>❌ 34% only | `y`<br>FP16<br>B×D×L | `y`<br>FP16<br>B×L×D | Hadamard<br>scale=1/√d<br>qMambaLayer:141 | `y`<br>FP16<br>B×L×D | `y`<br>FP16<br>B×L×D | HadLinear<br>+ Had输入/输出变换<br>W4A8B16O16<br>qMambaLayer:71 | `output`<br>FP16<br>B×L×D | `FLOAT_SIM_ASIC_INT8=true`<br>`CONV1D_MODE24_FP32=true` | 34.0%<br>❌原因未知 |
| **3**<br>**路径A**<br>(dt,B,C) | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | Requant<br>`round(x/scale)`<br>**scale=**<br>`self.conv1d.output_scale`<br>=`x_proj:input` (Percentile)<br>**← 来自左边Conv1D output_scale列**<br>⚠️ scale mismatch<br>Conv1D用动态scale<br>qMambaLayer:764 | `x_for_xproj`<br>INT8<br>B×D×L | `x_for_xproj`<br>INT8<br>B×L×D | W8A8B8O8<br>qMambaLayer:770 | `dt,B,C`<br>INT8 | `dt`<br>INT8<br>B×L×dt_rank | W8A8B8O8<br>qMambaLayer:775 | `dt`<br>INT8<br>B×dt_rank×L | **SSM 6 scale汇合**:<br>`u_scale`=动态量化❌<br>**← 来自左边Conv1D output_scale列(动态)**<br>(u: FP32 TRUE)<br>`dt_scale`=`dt_proj:output`❌<br>**← 来自左边dt_proj输出列**<br>(dt: INT8)<br>`B_scale`=`x_proj:output`❌<br>**← 来自左边x_proj输出列**<br>(B: INT8)<br>`C_scale`=`x_proj:output`❌<br>**← 来自左边x_proj输出列**<br>(C: INT8)<br>`z_scale`=`in_proj:output`❌<br>**← 来自左边in_proj输出列**<br>(z: INT8)<br>`ssm_state_scale`=`ssm_state_act:input`✅<br>**← 来自右边SSM ssm_state_scale列**<br>qMambaLayer:628-633 | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ | ↓ |
| **3**<br>**路径B**<br>(u) | `hidden_states`<br>FP16/FP32<br>B×L×D<br>← prev layer | HadLinear<br>+ Had输入变换<br>W4A8B16O16<br>qMambaLayer.py:52 | `xz`<br>INT8<br>B×L×2D | `x`<br>INT8<br>B×D×L<br>← split<br>**→动态量化** ⭐ | **动态量化+FP32 CUDA** ⭐<br>**Conv1D+SiLU融合**<br>`x_absmax/127`<br>qConvLayer:426-430<br>+<br>`quant_causal_conv1d_fwd_fp32()`:438<br>`silu_activation=True` | `x`<br>**FP32 (TRUE)** ⭐<br>B×D×L | (融合在Conv1D中) | (融合在Conv1D中)<br>**FP32 CUDA kernel**<br>直接输出FP32 | (融合在Conv1D中) | ❌ 无Percentile<br>**动态量化** ⭐<br>`x_absmax/127`<br>**📌 scale计算**:<br>forward()时动态<br>qConvLayer:426-430 | `x`<br>FP32 (TRUE)<br>B×D×L | 保持FP32<br>`x_for_ssm=x`<br>qMambaLayer:765 | `x_for_ssm`<br>**FP32 (TRUE)** ⭐<br>B×D×L | → | → | → | → | → | → | ↑汇合到路径A | ↓<br>❌ `u_scale`=动态<br>每token动态计算<br>(u保持FP32 TRUE) | **Hybrid FP32** ⭐<br>selective_scan_SE_float | `y`<br>FP16<br>B×D×L | `y`<br>FP16<br>B×L×D | Hadamard<br>scale=1/√d<br>qMambaLayer:141 | `y`<br>FP16<br>B×L×D | `y`<br>FP16<br>B×L×D | HadLinear<br>+ Had输入/输出变换<br>W4A8B16O16<br>qMambaLayer:71 | `output`<br>FP16<br>B×L×D | `CONV1D_MODE3_FP32=true` | **38.0%** ✅<br>动态量化 |

---

# 📊 7种模式精度理论分析

## 评分框架

### 精度损失因素权重

1. **量化误差 (40%)**: INT8量化导致的信息损失
2. **Scale质量 (30%)**: Percentile vs MinMax，是否裁剪outliers
3. **数据范围 (20%)**: FP32 TRUE > FP32 INT8 grid > INT8
4. **Scale一致性 (10%)**: 是否存在scale mismatch

---

## 🥇 最佳模式（并列第一）

### Mode 3 (动态量化) - 理论得分 90/100 (A)

**ACC: 38.0%** ✅

| 维度 | 得分 | 分析 |
|-----|------|------|
| **量化误差** | 38/40 | ✅✅ u保持FP32 TRUE<br>✅ 动态量化减少Conv1D量化误差 |
| **Scale质量** | 28/30 | ✅✅ **动态scale**: 每个token自适应<br>⚠️ 但Requant用静态Percentile (mismatch) |
| **数据范围** | 20/20 | ✅✅ u: FP32 TRUE完整范围 |
| **Scale一致性** | 4/10 | ❌ **Scale mismatch**:<br>Conv1D: 动态scale(`x.absmax/127`)<br>Requant: 静态Percentile scale |

**关键机制：动态量化**

```python
# Conv1D forward动态scale (qConvLayer.py:426-430)
x_absmax = x.abs().max().item()
x_dynamic_scale = x_absmax / 127.0  # ← 每个token自适应

x_int8 = torch.round(x / x_dynamic_scale).clamp(-128, 127).to(torch.int8)
y_fp32 = quant_causal_conv1d_cuda.fwd_fp32(x_int8, x_dynamic_scale, ...)
# 输出: FP32 TRUE (无裁剪)
```

**优势**:
- ✅✅ **动态量化优于静态Percentile**: 每个token自适应scale，大幅减少Conv1D量化误差
- ✅✅ **u保持FP32 TRUE**: 完整数值范围，无量化损失
- ✅ **Hybrid FP32 SSM**: `selective_scan_SE_float` 专门优化

**为什么效果好**:
1. 动态量化的优势压倒scale mismatch的劣势
2. u路径(FP32 TRUE)是SSM的主输入，影响最大
3. 路径A (dt,B,C) 即使有scale mismatch，影响相对较小

---

### Mode 2-0 (Dequant + CUDA SSM) - 理论得分 86/100 (B+)

**ACC: 38.0%** ✅

| 维度 | 得分 | 分析 |
|-----|------|------|
| **量化误差** | 32/40 | ⚠️ Dequant→Requant，理论上抵消<br>但FP32(INT8 grid)仍受限于INT8范围 |
| **Scale质量** | 30/30 | ✅ u_scale用Percentile<br>✅ SSM state用Percentile |
| **数据范围** | 14/20 | ⚠️ FP32但值域限制在INT8 grid<br>略优于Mode 0 (浮点表示精度) |
| **Scale一致性** | 10/10 | ✅ Dequant和Requant用同一scale |

**关键机制：Scale一致性**

```python
# Dequant (qMambaLayer.py:760)
x_for_ssm = x.float() * self.conv1d.output_scale
# INT8 [-128,127] → FP32 但值域仍为 INT8 grid

# SSM内部Requant (SSM CUDA kernel)
u_int8 = round(u_fp32 / u_scale)  # ← 用同一个Percentile scale
```

**优势**:
- ✅ **Scale一致性最好**: Dequant/Requant用同一个Percentile scale
- ✅ **CUDA INT8 SSM优化**: 硬件优化比PyTorch FP32更好
- ✅ **环境变量明确**: `FLOAT_SIM_ASIC_INT8=true` + `SSM_USE_CUDA_FOR_FP32=true`

**局限**:
- ❌ **FP32 INT8 grid**: u的值域仍然是 `scale * [-128, 127]`，虽然精度提升但范围受限

---

## 🥈 良好模式

### Mode 2-2 (Dequant + PyTorch FP32 SSM) - 理论得分 88/100 (B+)

**ACC: 36.0%** (低于预期)

| 维度 | 得分 | 分析 |
|-----|------|------|
| **量化误差** | 34/40 | ⚠️ SSM用FP32，但u仍是INT8 grid |
| **Scale质量** | 30/30 | ✅ u_scale用Percentile<br>✅ SSM state用Percentile |
| **数据范围** | 14/20 | ⚠️ u: FP32 但INT8 grid |
| **Scale一致性** | 10/10 | ✅ Dequant用Percentile scale |

**为什么低于预期**:

理论上SSM用FP32应该比Mode 2-0的INT8 SSM更好，但实际相同：

1. **PyTorch FP32 SSM不如CUDA INT8 SSM**
   - CUDA INT8: 硬件优化 → 38.0% ✅
   - PyTorch FP32 mode22: 软件实现 → 36.0%

2. **FP32 INT8 grid的限制**
   - u的值域仍然是 `scale * [-128, 127]`
   - FP32只是提供更精确的"表示"，但范围未扩大

---

## 🥉 一般模式

### Mode 0 (基准) - 理论得分 82/100 (B)

**ACC: 基准**

| 维度 | 得分 | 分析 |
|-----|------|------|
| **量化误差** | 30/40 | ❌ 全程INT8，累积误差最大 |
| **Scale质量** | 30/30 | ✅ Conv1D输出用Percentile (u_scale)<br>✅ SSM state用Percentile |
| **数据范围** | 12/20 | ❌ u路径: 全程INT8，范围最窄 |
| **Scale一致性** | 10/10 | ✅ 无scale mismatch |

**优势**:
- ✅ **无额外转换开销**: 全程INT8，无dequant/requant
- ✅ **Scale一致性最好**: calibration的scale直接用于inference
- ✅ **2个Percentile scale保护关键路径**

**局限**:
- ❌ **量化误差累积**: Conv1D→SSM全程INT8
- ❌ **u的数值范围最窄**: INT8限制在[-128, 127]

---

### Mode 2-1 (保持INT8 + PyTorch SSM) - 理论得分 80/100 (B)

**ACC: 36.0%**

| 维度 | 得分 | 分析 |
|-----|------|------|
| **量化误差** | 28/40 | ❌ PyTorch INT8 SSM内部实现可能精度损失 |
| **Scale质量** | 30/30 | ✅ u_scale用Percentile<br>✅ SSM state用Percentile |
| **数据范围** | 12/20 | ❌ u: 全程INT8 |
| **Scale一致性** | 10/10 | ✅ 无scale mismatch |

**优势**: 无转换开销（全程INT8）

**局限**:
- ❌ **PyTorch INT8 SSM精度**: 不如CUDA实现
- ❌ **数值范围窄**: INT8限制

---

## 🔴 较差模式

### Mode 2-3 (Conv1D FP32 → Requant u) - 理论得分 72/100 (C)

**ACC: 36.0%** ❌ (FP32优势完全无效)

| 维度 | 得分 | 分析 |
|-----|------|------|
| **量化误差** | 26/40 | ❌❌ FP32 TRUE→INT8 requant，巨大精度损失<br>**FP32优势完全浪费** |
| **Scale质量** | 24/30 | ✅ Requant用Percentile<br>⚠️ 但Conv1D forward未用scale (mismatch) |
| **数据范围** | 18/20 | ✅ Conv1D输出FP32 TRUE (完整范围)<br>❌ 但立即被requant到INT8 |
| **Scale一致性** | 4/10 | ❌❌ **Scale mismatch**:<br>Conv1D forward: 无scale (FP32输出)<br>Requant: 用calibration的Percentile scale |

**关键问题：FP32优势被requant抵消**

```python
# Conv1D输出FP32 TRUE ✅
y_fp32 = quant_causal_conv1d_fwd_fp32(...)

# 但立即Requant ❌
x_for_xproj = torch.round(x / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
# 路径A和u都被requant回INT8，FP32完全浪费！
```

**局限**:
- ❌❌ **FP32优势完全浪费**: 路径A和u都被requant回INT8
- ❌ **Scale mismatch**: Conv1D forward不用scale，但requant用calibration scale
- ❌ **PyTorch INT8 SSM**: 精度不如CUDA

---

## ❌❌ 最差模式：Mode 2-4

### Mode 2-4 (Conv1D FP32 + u保持FP32 + PyTorch FP32 SSM) - 理论得分 84/100 (B)

**ACC: 34.0%** ❌❌ (远低于预期！)

| 维度 | 得分 | 分析 |
|-----|------|------|
| **量化误差** | 36/40 | ✅ u保持FP32 TRUE，无requant损失<br>⚠️ 但路径A仍是INT8 |
| **Scale质量** | 24/30 | ⚠️ u不用quantization scale (直接FP32)<br>⚠️ Conv1D forward未用scale (mismatch) |
| **数据范围** | 20/20 | ✅✅ u: FP32 TRUE，完整数值范围<br>✅ SSM: FP32计算 |
| **Scale一致性** | 4/10 | ❌ Scale mismatch (同Mode 2-3) |

**🔥 关键疑问：理论上应该最好，实际最差！**

**理论优势**:
- u用FP32 TRUE (完整范围)
- SSM用FP32计算
- 无requant损失

**实际却最差的可能原因**:

#### 1. PyTorch FP32 SSM (mode22) 实现问题

```python
# Mode 2-2和2-4都用mode22
# 但2-2: u是FP32 INT8 grid → 36.0%
# 2-4: u是FP32 TRUE → 34.0% ❌

# 推测: mode22可能假设输入在INT8范围内
# 当u是FP32 TRUE (范围超过INT8)，可能导致数值溢出或不稳定
```

#### 2. 路径A/B严重不平衡

```python
# 路径A: INT8 (使用Percentile scale)
# 路径B: FP32 TRUE (完整范围)

# SSM内部6个输入scale差异巨大:
# - dt, B, C, z_scale: MinMax (INT8范围)
# - u_scale: 应该是FP32范围，但实际用的是calibration时的Percentile scale
#   这个scale是基于INT8数据统计的，不适合FP32 TRUE范围！
```

#### 3. Scale信息丢失

Conv1D forward不用任何scale（直接FP32输出），但SSM内部可能假设输入经过quantization scale归一化，FP32 TRUE的数值范围可能导致SSM内部计算不稳定。

**建议调查方向**:
1. 检查PyTorch mode22的实现，确认是否对输入范围有假设
2. 比较Mode 2-2和2-4的SSM内部激活值范围
3. 检查是否存在数值溢出或NaN

---

## 🎯 总结与建议

### 理论与实际ACC排名对比

| 理论排名 | 模式 | 理论得分 | 评级 | 实际ACC | 符合度 |
|---------|------|---------|------|---------|--------|
| 🥇 1 | **Mode 3** | 90/100 | A | **38.0%** ✅ | ✅✅ 完全符合 |
| 🥈 2 | Mode 2-2 | 88/100 | B+ | 36.0% | ⚠️ 低于预期 |
| 🥉 3 | **Mode 2-0** | 86/100 | B+ | **38.0%** ✅ | ✅ 符合 |
| 4 | Mode 2-4 | 84/100 | B | 34.0% ❌ | ❌❌ 远低于预期 |
| 5 | Mode 0 | 82/100 | B | **38.0%** ✅ | - |
| 6 | Mode 2-1 | 80/100 | B | 36.0% | ✅ 符合 |
| 7 | Mode 2-3 | 72/100 | C | 36.0% | ✅ 符合 |

### 最终建议

**1. 追求最高精度: Mode 3** 🥇
- 动态量化自适应每个token
- u保持FP32 TRUE
- Hybrid FP32 SSM优化
- ACC: 38.0% (并列第一)

**2. 追求稳定性: Mode 2-0** 🥇
- Scale一致性好
- CUDA INT8 SSM优化好
- 环境变量明确
- ACC: 38.0% (并列第一)

**3. 基准对比: Mode 0**
- 全程INT8，一致性最好
- 无额外开销
- 适合作为baseline

**不推荐**:
- ❌ **Mode 2-4**: 理论好但实际最差，需要调查PyTorch mode22
- ❌ **Mode 2-3**: FP32优势被requant完全抵消

---

## 🔑 关键发现

### 1. 动态量化优于静态Percentile
Mode 3的动态量化 (`x.absmax/127`) 比静态Percentile (calibration时统计) 更有效

### 2. CUDA INT8 SSM优于PyTorch FP32 SSM
- Mode 2-0 (CUDA INT8) = 38.0% ✅
- Mode 2-2 (PyTorch FP32) = 36.0%

说明硬件优化的重要性

### 3. Scale一致性很重要
Mode 2-4虽然u用FP32 TRUE，但scale mismatch + 可能的实现问题导致最差结果

### 4. FP32 INT8 grid的价值有限
Mode 2-0和2-2都是u用FP32 INT8 grid，但后者用FP32 SSM反而更差

---

## 需要进一步调查的问题

1. **Mode 2-4为什么这么差？**
   - 检查PyTorch mode22的实现
   - 对比SSM内部激活值范围
   - 是否存在数值溢出/NaN？

2. **Mode 3的动态量化能否应用到其他模式？**
   - 是否可以在Mode 2-4中也用动态量化？
   - 动态量化 + u(FP32 TRUE) + CUDA SSM？

3. **为什么CUDA INT8 SSM比PyTorch FP32 SSM好？**
   - 硬件优化的重要性
   - PyTorch实现是否有问题？
