# Percentile Scale 使用位置完整追踪

## 核心问题回答

### Q1: 每一层的 conv1d 都去 SSM 吗？

**答案：是的！** 每个 Mamba block 的数据流：

```
输入 → RMSNorm → in_proj → conv1d → x_proj → SSM (selective_scan)
                                 ↓
                            这里去 SSM!
```

**24 层结构 (Quamba2-130m)**:
```
Layer 0: conv1d → SSM
Layer 1: conv1d → SSM
Layer 2: conv1d → SSM
...
Layer 23: conv1d → SSM
```

每一层的 conv1d 输出都是对应层 SSM 的输入！

---

## Q2: Percentile 用在哪些 Scale？

### Quamba1 (Mamba1 模型)

**Percentile 使用位置** (modelutils_mamba.py:163-169):

```python
if is_x(op) or is_ssm_state(op):
    observers[i][op + ":input"] = PerTensorPercentileObserver(
        percentile_alpha=percentile_alpha  # 0.9995
    )
```

**判断条件** (modelutils_mamba.py:122-123):
```python
is_x = lambda op: op == "x_proj"
is_ssm_state = lambda op: op == "ssm_state_act"
```

#### ✅ 使用 Percentile 的 2 个 Scale:

1. **`x_proj:input`** ← Conv1d+SiLU 的输出
   - 来源: `conv1d` 输出 (fused with SiLU)
   - 去向: `x_proj` 输入
   - 位置: qMambaLayer.py:851
   ```python
   qmixer.conv1d = QCausalConv1D.from_fp16(
       output_scale=act_scales["x_proj:input"].item(),  # ← Percentile!
   )
   ```

2. **`ssm_state_act:input`** ← SSM 内部状态激活
   - 来源: SSM 内部 state 计算
   - 去向: SSM state quantization
   - 位置: qMambaLayer.py:876
   ```python
   qmixer.selective_scan = QSScan.from_fp16(
       ssm_state_scale=act_scales["ssm_state_act:input"],  # ← Percentile!
   )
   ```

#### ❌ 不使用 Percentile 的其他 Scale (用 MinMax):

- `in_proj:input` (MinMax)
- `in_proj:output` (MinMax)
- `x_proj:output` (MinMax)
- `dt_proj:output` (MinMax)
- `out_proj:input` (MinMax)

---

### Quamba2 (Mamba2 模型) - 不用 Percentile!

**Quamba2 使用不同的 Observer** (modelutils_mamba.py:256-273):

```python
def run_quamba2_calibration(...):
    # Quamba2 不用 PerTensorPercentileObserver
    # 而是用 CrossHeadMinmaxObserver (Group Quantization)
```

**关键区别**:
```
Quamba1: PerTensorPercentileObserver  → 1 个 scalar scale (percentile)
Quamba2: CrossHeadMinmaxObserver      → (1,4,4) 或 (8,4,4) scales (minmax per group)
```

---

## Q3: 如何监测和修改特定的 Percentile Scale？

### 方法 1: 修改 Observer (Calibration 阶段)

**位置**: `quamba/observer.py:92`

```python
class PerTensorPercentileObserver:
    def get_quantization_parameters(self):
        cur_max = torch.quantile(
            w.abs().reshape(-1),
            self.percentile_alpha  # ← 这里计算 percentile
        )

        # 🔥 你可以在这里修改 scale!
        # 例如: cur_max = cur_max + 2025  (你之前提到的 +2025)
        scale = cur_max / (2 ** (self.n_bits - 1) - 1)
        return scale, zero
```

**影响范围**:
- 会影响所有使用 `PerTensorPercentileObserver` 的层
- 即: 所有 24 层的 `x_proj:input` 和 `ssm_state_act:input`

---

### 方法 2: 修改 act_scales (Model 构建阶段)

**位置**: `quamba/modelutils_mamba.py:247-251`

```python
for i in range(len(layers) + 1):
    for name, observer in observers[i].items():
        scale, base = observer.get_quantization_parameters()
        act_scales[i][name] = scale.to(torch.float32)

        # 🔥 你可以在这里修改特定层的 scale!
        # 例如:
        if name == "x_proj:input" and i == 0:  # 只修改 Layer 0
            act_scales[i][name] = scale * 1.5  # 乘以 1.5
```

**优点**: 可以精确控制每一层

---

### 方法 3: 修改已保存的模型 (加载后)

**位置**: 加载模型后直接修改

```python
model = load_quamba_model(...)

# 遍历所有层
for layer_idx, layer in enumerate(model.backbone.layers):
    mixer = layer.mixer

    # 修改 conv1d 的 output_scale (这就是 x_proj:input)
    if hasattr(mixer.conv1d, 'output_scale'):
        old_scale = mixer.conv1d.output_scale
        new_scale = old_scale * 1.5  # 你的修改
        mixer.conv1d.output_scale = new_scale
        print(f"Layer {layer_idx}: {old_scale:.4f} → {new_scale:.4f}")

    # 修改 SSM 的 ssm_state_scale
    if hasattr(mixer.selective_scan, 'ssm_state_scale'):
        old_scale = mixer.selective_scan.ssm_state_scale
        new_scale = old_scale * 1.5
        mixer.selective_scan.ssm_state_scale = new_scale
```

**优点**: 不需要重新 calibrate

---

## Q4: 如何标记和监测 Percentile Scale 的使用？

### 监测脚本

我给你写一个脚本来追踪每一层的 percentile scale 使用情况：

```python
import torch
from safetensors import safe_open

def trace_percentile_scales(model_path):
    """
    追踪模型中所有使用 percentile scale 的位置
    """

    with safe_open(model_path, framework="pt", device="cpu") as f:
        print("=" * 80)
        print("Percentile Scale 使用位置追踪")
        print("=" * 80)

        # 统计信息
        total_layers = 0
        x_proj_scales = []
        ssm_state_scales = []

        for key in f.keys():
            # 追踪 conv1d 的 output_scale (对应 x_proj:input)
            if "mixer.conv1d.output_scale" in key:
                total_layers += 1
                layer_idx = int(key.split(".")[2])  # backbone.layers.0.mixer...
                value = f.get_tensor(key)

                print(f"\n🎯 Layer {layer_idx} Conv1d Output (去 SSM)")
                print(f"   Key: {key}")
                print(f"   Shape: {value.shape}")
                print(f"   Value: {value}")
                print(f"   ✅ 使用 Percentile (x_proj:input)")

                x_proj_scales.append({
                    'layer': layer_idx,
                    'scale': float(value.mean())
                })

            # 追踪 SSM 的 ssm_state_scale
            if "selective_scan.ssm_state_scale" in key:
                layer_idx = int(key.split(".")[2])
                value = f.get_tensor(key)

                print(f"\n🎯 Layer {layer_idx} SSM State")
                print(f"   Key: {key}")
                print(f"   Value: {value}")
                print(f"   ✅ 使用 Percentile (ssm_state_act:input)")

                ssm_state_scales.append({
                    'layer': layer_idx,
                    'scale': float(value)
                })

        print("\n" + "=" * 80)
        print("统计总结")
        print("=" * 80)
        print(f"总层数: {total_layers}")
        print(f"使用 Percentile 的 Scale:")
        print(f"  • x_proj:input (Conv1d → SSM): {len(x_proj_scales)} 层")
        print(f"  • ssm_state_act:input (SSM 内部): {len(ssm_state_scales)} 层")
        print(f"\n每一层的 Conv1d 都使用 Percentile Scale 去 SSM!")

        # 显示 scale 分布
        print("\n" + "=" * 80)
        print("Conv1d → SSM Scale 分布 (Percentile-computed)")
        print("=" * 80)
        for item in x_proj_scales[:5]:  # 显示前 5 层
            print(f"Layer {item['layer']:2d}: {item['scale']:.6f}")
        print("...")

# 使用
trace_percentile_scales("./pretrained_models/quamba1-130m-w8a8/model.safetensors")
```

---

## 总结

### ✅ Percentile 用在这 2 个地方:

| Scale 名称 | 来源 | 去向 | 每层都用? | 去 SSM? |
|-----------|------|------|----------|---------|
| **x_proj:input** | Conv1d+SiLU 输出 | x_proj 输入 → SSM | ✅ 24 层全部 | ✅ 是 |
| **ssm_state_act:input** | SSM 内部 state | SSM 量化 | ✅ 24 层全部 | ✅ 是 |

### 🔥 关键点:

1. **每一层的 conv1d 都去 SSM** (24 层全部)
2. **Percentile 只用在 2 个位置**:
   - Conv1d 输出 (x_proj:input)
   - SSM 内部 state (ssm_state_act:input)
3. **Quamba2 不用 Percentile** (改用 Group Quantization)

### 📍 修改 Percentile Scale 的 3 个位置:

1. **Observer** (observer.py:92) - 影响所有层
2. **act_scales** (modelutils_mamba.py:251) - 可控制单层
3. **加载后修改** - 最灵活

### 🎯 监测方法:

使用上面的 `trace_percentile_scales()` 脚本追踪每一层的 scale 使用情况。
