# 所有 Scale 的完整生命周期分析

## 概览：7 个主要 Scale

| Scale 名称 | Observer 类型 | 用途 | SSM 使用 |
|-----------|--------------|------|----------|
| `x_proj:input` | ✅ Percentile | Conv1D输出→x_proj输入 | ✅ u_scale |
| `ssm_state_act:input` | ✅ Percentile | SSM内部state量化 | ✅ ssm_state_scale |
| `in_proj:input` | ❌ MinMax | in_proj输入 | ❌ 不直接用于SSM |
| `in_proj:output` | ❌ MinMax | in_proj输出→Conv1D输入 | ✅ z_scale |
| `x_proj:output` | ❌ MinMax | x_proj输出→dt_proj输入 | ✅ B_scale, C_scale |
| `dt_proj:output` | ❌ MinMax | dt_proj输出 | ✅ dt_scale |
| `out_proj:input` | ❌ MinMax | SSM输出→out_proj输入 | ❌ 不直接用于SSM |

---

## 1️⃣ `x_proj:input` (Conv1D output_scale / u_scale)

### 📥 Calibration 阶段

**Observer 注册** (modelutils_mamba.py:163-169):
```python
if is_x(op) or is_ssm_state(op):  # is_x = lambda op: op == "x_proj"
    observers[i]["x_proj:input"] = PerTensorPercentileObserver(
        n_bits=8,
        percentile_alpha=0.9995  # 裁剪0.05% outliers
    )
```

**数据收集** (modelutils_mamba.py:141-149):
```python
# Hook 在 x_proj 层的 forward 时触发
def stat_hook(m, inputs, outputs, op="x_proj", block_idx=i):
    observers[block_idx]["x_proj:input"].update(inputs.clone().detach())
    # 收集的是 Conv1D+SiLU 输出（INT8），即将输入 x_proj
```

**Scale 计算** (modelutils_mamba.py:247-251):
```python
scale, base = observer.get_quantization_parameters()
# 内部调用 observer.py:92-110:
#   cur_max = torch.quantile(w.abs().reshape(-1), 0.9995)
#   scale = cur_max / 127
act_scales[i]["x_proj:input"] = scale.to(torch.float32)
```

### 🔄 Model 构建阶段

**赋值到 Conv1D** (qMambaLayer.py:603-606):
```python
qmixer.conv1d = QCausalConv1D.from_fp16(
    output_scale=act_scales["x_proj:input"].item(),  # ← 存储为 self.output_scale
)
```

**赋值到 x_proj** (qMambaLayer.py:610-614):
```python
qmixer.x_proj = W4A8B8O8Linear.from_fp16(
    input_scale=act_scales["x_proj:input"],  # ← 存储为 self.input_scale
)
```

**赋值到 SSM** (qMambaLayer.py:628-633):
```python
qmixer.selective_scan = QSScan.from_fp16(
    u_scale=act_scales["x_proj:input"],  # ← 存储为 self.u_scale
)
```

### 🚀 Inference 阶段

**使用1: Conv1D forward - Mode 0, 2-1** (qConvLayer.py:116-122 / 157):
```python
# INT8 Conv1D kernel 直接输出 INT8，不使用 output_scale
y = quant_causal_conv1d_cuda.fwd(
    x, self.input_scale,
    self.weight, self.weight_scale,
    self.output_scale,  # ← 传入但 INT8 模式下不用于 dequant
    self.bias_scale, self.bias,
    None, None, None, True
)
```

**使用2: Dequant INT8→FP32 - Mode 2-0, 2-2** (qMambaLayer.py:760):
```python
x_for_ssm = x.float() * self.conv1d.output_scale  # ← Dequant 使用
```

**使用3: Requant FP32→INT8 - Mode 2-3, 2-4** (qMambaLayer.py:764):
```python
x_for_xproj = torch.round(x / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
# ← Requant 使用
```

**使用4: Requant FP32→INT8 - Mode 3 ⚠️ Scale Mismatch** (qMambaLayer.py:764):
```python
# Conv1D forward 时用动态 scale (qConvLayer.py:426-430):
#   x_dynamic_scale = x.abs().max().item() / 127.0
#   y_fp32 = quant_causal_conv1d_cuda.fwd_fp32(x_int8, x_dynamic_scale, ...)
# 但 Requant 时用的是 calibration 的 Percentile scale:
x_for_xproj = torch.round(x / self.conv1d.output_scale).clamp(-128, 127).to(torch.int8)
# ⚠️ 两个 scale 不匹配！
```

**使用5: x_proj forward** (W4A8B8O8Linear):
```python
# x_proj.input_scale 用于将输入 dequant（如果需要）
# 具体使用在 linear 层内部
```

**使用6: SSM forward - u 输入** (qSelectiveScan.py / CUDA):
```python
# self.u_scale 用于量化/反量化 u 输入
# 具体使用在 SSM kernel 内部
```

---

## 2️⃣ `ssm_state_act:input` (SSM state scale)

### 📥 Calibration 阶段

**Observer 注册** (modelutils_mamba.py:163-169):
```python
if is_x(op) or is_ssm_state(op):  # is_ssm_state = lambda op: op == "ssm_state_act"
    observers[i]["ssm_state_act:input"] = PerTensorPercentileObserver(
        n_bits=8,
        percentile_alpha=0.9995
    )
```

**数据收集** (通过 SSM 内部 hook):
```python
# Hook 在 selective_scan 内部收集 SSM state 激活值
def stat_hook(m, inputs, outputs, op="ssm_state_act", block_idx=i):
    observers[block_idx]["ssm_state_act:input"].update(inputs.clone().detach())
    # 收集的是 SSM 内部计算的 state 激活值
```

**Scale 计算** (modelutils_mamba.py:247-251):
```python
scale, base = observer.get_quantization_parameters()
act_scales[i]["ssm_state_act:input"] = scale.to(torch.float32)
```

### 🔄 Model 构建阶段

**赋值到 SSM** (qMambaLayer.py:628-633):
```python
qmixer.selective_scan = QSScan.from_fp16(
    ssm_state_scale=act_scales["ssm_state_act:input"],  # ← 存储为 self.ssm_state_scale
)
```

### 🚀 Inference 阶段

**使用: SSM forward - state 量化** (qSelectiveScan.py / CUDA):
```python
# self.ssm_state_scale 用于 SSM 内部 state 的量化
# 具体使用在 selective_scan kernel 内部
# 对 SSM 状态进行 INT8 量化以节省内存和计算
```

---

## 3️⃣ `in_proj:input` (in_proj 输入 scale)

### 📥 Calibration 阶段

**Observer 注册** (modelutils_mamba.py:171-175):
```python
else:  # 不是 x_proj 或 ssm_state_act
    observers[i]["in_proj:input"] = PerTensorMinmaxObserver(
        n_bits=8,
        clip_ratio=1.0,
        sym=True
    )
```

**数据收集** (modelutils_mamba.py:141-149):
```python
def stat_hook(m, inputs, outputs, op="in_proj", block_idx=i):
    observers[block_idx]["in_proj:input"].update(inputs.clone().detach())
    # 收集的是前一层输出（hidden_states），即将输入 in_proj
```

**Scale 计算** (modelutils_mamba.py:247-251):
```python
scale, base = observer.get_quantization_parameters()
# MinmaxObserver 计算方法 (observer.py):
#   cur_max = w.abs().max()
#   scale = cur_max / 127
act_scales[i]["in_proj:input"] = scale.to(torch.float32)
```

### 🔄 Model 构建阶段

**赋值到 in_proj** (qMambaLayer.py:593-597):
```python
qmixer.in_proj = W4A8B8O8Linear.from_fp16(
    input_scale=act_scales["in_proj:input"],  # ← 存储为 self.input_scale
    output_scale=act_scales["in_proj:output"],
)
```

### 🚀 Inference 阶段

**使用: in_proj forward** (W4A8B8O8Linear):
```python
# self.input_scale 用于将 FP16 hidden_states 量化为 INT8
# 或在已经是 INT8 时用于记录 scale 信息
```

---

## 4️⃣ `in_proj:output` (z_scale / Conv1D 输入 scale)

### 📥 Calibration 阶段

**Observer 注册** (modelutils_mamba.py:176-180):
```python
observers[i]["in_proj:output"] = PerTensorMinmaxObserver(
    n_bits=8,
    clip_ratio=1.0,
    sym=True
)
```

**数据收集** (modelutils_mamba.py:141-149):
```python
def stat_hook(m, inputs, outputs, op="in_proj", block_idx=i):
    observers[block_idx]["in_proj:output"].update(outputs.clone().detach())
    # 收集的是 in_proj 的输出（xz拼接），即将 split 并送入 Conv1D 和作为 z
```

**Scale 计算** (modelutils_mamba.py:247-251):
```python
scale, base = observer.get_quantization_parameters()
act_scales[i]["in_proj:output"] = scale.to(torch.float32)
```

### 🔄 Model 构建阶段

**赋值到 in_proj** (qMambaLayer.py:593-597):
```python
qmixer.in_proj = W4A8B8O8Linear.from_fp16(
    input_scale=act_scales["in_proj:input"],
    output_scale=act_scales["in_proj:output"],  # ← 存储为 self.output_scale
)
```

**赋值到 Conv1D** (qMambaLayer.py:603-606):
```python
qmixer.conv1d = QCausalConv1D.from_fp16(
    input_scale=act_scales["in_proj:output"].item(),  # ← 存储为 self.input_scale
    output_scale=act_scales["x_proj:input"].item(),
)
```

**赋值到 SSM (作为 z_scale)** (qMambaLayer.py:628-633):
```python
qmixer.selective_scan = QSScan.from_fp16(
    z_scale=act_scales["in_proj:output"],  # ← 存储为 self.z_scale
)
```

### 🚀 Inference 阶段

**使用1: in_proj forward** (W4A8B8O8Linear):
```python
# self.output_scale 用于记录输出的量化 scale
```

**使用2: Conv1D forward** (qConvLayer.py):
```python
# self.input_scale 用于量化输入（如果需要）
# 或在 kernel 中用于计算
```

**使用3: SSM forward - z 输入** (qSelectiveScan.py):
```python
# self.z_scale 用于量化/反量化 z 输入
```

---

## 5️⃣ `x_proj:output` (B_scale, C_scale / dt_proj 输入 scale)

### 📥 Calibration 阶段

**Observer 注册** (modelutils_mamba.py:176-180):
```python
observers[i]["x_proj:output"] = PerTensorMinmaxObserver(
    n_bits=8,
    clip_ratio=1.0,
    sym=True
)
```

**数据收集** (modelutils_mamba.py:141-149):
```python
def stat_hook(m, inputs, outputs, op="x_proj", block_idx=i):
    observers[block_idx]["x_proj:output"].update(outputs.clone().detach())
    # 收集的是 x_proj 输出（dt, B, C 拼接），即将 split 后分别使用
```

**Scale 计算** (modelutils_mamba.py:247-251):
```python
scale, base = observer.get_quantization_parameters()
act_scales[i]["x_proj:output"] = scale.to(torch.float32)
```

### 🔄 Model 构建阶段

**赋值到 x_proj** (qMambaLayer.py:610-614):
```python
qmixer.x_proj = W4A8B8O8Linear.from_fp16(
    input_scale=act_scales["x_proj:input"],
    output_scale=act_scales["x_proj:output"],  # ← 存储为 self.output_scale
)
```

**赋值到 dt_proj** (qMambaLayer.py:617-621):
```python
qmixer.dt_proj = W8A8B8O8Linear.from_fp16(
    input_scale=act_scales["x_proj:output"].item(),  # ← 存储为 self.input_scale
    output_scale=act_scales["dt_proj:output"].item(),
)
```

**赋值到 SSM (作为 B_scale, C_scale)** (qMambaLayer.py:628-633):
```python
qmixer.selective_scan = QSScan.from_fp16(
    B_scale=act_scales["x_proj:output"],  # ← 存储为 self.B_scale
    C_scale=act_scales["x_proj:output"],  # ← 存储为 self.C_scale（同一个值）
)
```

### 🚀 Inference 阶段

**使用1: x_proj forward** (W4A8B8O8Linear):
```python
# self.output_scale 用于记录输出的量化 scale
```

**使用2: dt_proj forward** (W8A8B8O8Linear):
```python
# self.input_scale 用于量化输入（已经是 INT8，所以主要用于记录）
```

**使用3: SSM forward - B, C 输入** (qSelectiveScan.py):
```python
# self.B_scale 和 self.C_scale 用于量化/反量化 B, C 输入
```

---

## 6️⃣ `dt_proj:output` (dt_scale)

### 📥 Calibration 阶段

**Observer 注册** (modelutils_mamba.py:176-180):
```python
observers[i]["dt_proj:output"] = PerTensorMinmaxObserver(
    n_bits=8,
    clip_ratio=1.0,
    sym=True
)
```

**数据收集** (modelutils_mamba.py:141-149):
```python
def stat_hook(m, inputs, outputs, op="dt_proj", block_idx=i):
    observers[block_idx]["dt_proj:output"].update(outputs.clone().detach())
    # 收集的是 dt_proj 输出，即将输入 SSM 作为 dt
```

**Scale 计算** (modelutils_mamba.py:247-251):
```python
scale, base = observer.get_quantization_parameters()
act_scales[i]["dt_proj:output"] = scale.to(torch.float32)
```

### 🔄 Model 构建阶段

**赋值到 dt_proj** (qMambaLayer.py:617-621):
```python
qmixer.dt_proj = W8A8B8O8Linear.from_fp16(
    input_scale=act_scales["x_proj:output"].item(),
    output_scale=act_scales["dt_proj:output"].item(),  # ← 存储为 self.output_scale
)
```

**赋值到 SSM (作为 dt_scale)** (qMambaLayer.py:628-633):
```python
qmixer.selective_scan = QSScan.from_fp16(
    dt_scale=act_scales["dt_proj:output"],  # ← 存储为 self.dt_scale
)
```

### 🚀 Inference 阶段

**使用1: dt_proj forward** (W8A8B8O8Linear):
```python
# self.output_scale 用于记录输出的量化 scale
```

**使用2: SSM forward - dt 输入** (qSelectiveScan.py):
```python
# self.dt_scale 用于量化/反量化 dt 输入
```

---

## 7️⃣ `out_proj:input` (out_proj 输入 scale)

### 📥 Calibration 阶段

**Observer 注册** (modelutils_mamba.py:171-175):
```python
observers[i]["out_proj:input"] = PerTensorMinmaxObserver(
    n_bits=8,
    clip_ratio=1.0,
    sym=True
)
```

**数据收集** (modelutils_mamba.py:141-149):
```python
def stat_hook(m, inputs, outputs, op="out_proj", block_idx=i):
    observers[block_idx]["out_proj:input"].update(inputs.clone().detach())
    # 收集的是 SSM 输出（经过 Hadamard），即将输入 out_proj
```

**Scale 计算** (modelutils_mamba.py:247-251):
```python
scale, base = observer.get_quantization_parameters()
act_scales[i]["out_proj:input"] = scale.to(torch.float32)
```

### 🔄 Model 构建阶段

**赋值到 Hadamard** (qMambaLayer.py:636-640):
```python
if use_had_transform:
    qmixer.had.x_H_scale = act_scales["out_proj:input"].item()  # ← HadLinear 内部
else:
    qmixer.had.scale = act_scales["out_proj:input"].item()  # ← QAct
```

**赋值到 out_proj** (qMambaLayer.py:641-644):
```python
qmixer.out_proj = W4A8B16O16Linear.from_fp16(
    input_scale=act_scales["out_proj:input"],  # ← 存储为 self.input_scale
)
```

### 🚀 Inference 阶段

**使用1: Hadamard forward** (qMambaLayer.py:141):
```python
# scale 用于量化 SSM 输出为 INT8（如果需要）
```

**使用2: out_proj forward** (W4A8B16O16Linear):
```python
# self.input_scale 用于量化输入（从 FP16 或已有 INT8）
```

---

## 🔑 总结：Scale 使用模式

### Percentile Scale (2个)
- `x_proj:input`: Conv1D输出 → x_proj/SSM
- `ssm_state_act:input`: SSM内部state

### MinMax Scale (5个)
- `in_proj:input`: 前一层输出 → in_proj
- `in_proj:output`: in_proj输出 → Conv1D/z
- `x_proj:output`: x_proj输出 → dt_proj/B/C
- `dt_proj:output`: dt_proj输出 → SSM dt
- `out_proj:input`: SSM输出 → out_proj

### SSM 6-scale 输入
1. `u_scale` = `x_proj:input` ✅ Percentile
2. `dt_scale` = `dt_proj:output` ❌ MinMax
3. `B_scale` = `x_proj:output` ❌ MinMax
4. `C_scale` = `x_proj:output` ❌ MinMax
5. `z_scale` = `in_proj:output` ❌ MinMax
6. `ssm_state_scale` = `ssm_state_act:input` ✅ Percentile
