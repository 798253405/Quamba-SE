# Quamba1 模型加载Bug修复详解

**创建时间**: 2025-11-06
**问题类型**: 代码兼容性Bug
**影响范围**: Quamba1模型（不使用 `--quantize_lm_head` 的场景）
**重要性**: ⭐⭐⭐ **非常重要，不要删除此文件**

---

## 🔍 问题背景

### 作者的量化建议

根据作者的邮件回复：

> If you'd like to reproduce the Quamba1 results, please set the quantization bit-width to W8A8 and quantize the Mamba1 models **without** `--quantize_embedding`, `--quantize_lm_head`, and `--apply_gptq` flags.

**核心要求**：
- Quamba1（Mamba1）：W8A8，**不加** `--quantize_lm_head`
- Quamba2（Mamba2）：W4A8，**必须加** `--quantize_lm_head`, `--quantize_embedding`, `--apply_gptq`

### 发现的问题

当按照作者建议量化 Mamba1-130M 模型后，尝试加载评估时遇到了 **4个连续的代码Bug**，导致无法正常加载和运行模型。

---

## 🐛 Bug详情与修复

### Bug 1: `qNorm.py` - KeyError: 'output_scale'

**错误信息**：
```python
KeyError: 'backbone.norm_f.output_scale'
```

**错误位置**：`quamba/qNorm.py:43`

**根本原因**：
```python
# 原始代码（第42-44行）
def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    self.output_scale = state_dict[prefix + 'output_scale']  # 直接访问，键不存在时报错
    del state_dict[prefix + 'output_scale']
```

当不加 `--quantize_lm_head` 时：
1. `norm_f` 不会被量化，保持 FP16 的 `RMSNorm`
2. 保存模型时 `norm_f` 的 state_dict 中**没有** `output_scale` 键
3. 但 config 错误地把所有 norm 类型写成了 `QRMSNorm`
4. 加载时创建 `QRMSNorm`，`load_hook` 找不到 `output_scale` 键，报错

**修复方案**：
```python
# 修复后的代码（第42-48行）
def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    # Handle backward compatibility: if output_scale is not in state_dict, set to None
    if prefix + 'output_scale' in state_dict:
        self.output_scale = state_dict[prefix + 'output_scale']
        del state_dict[prefix + 'output_scale']
    else:
        self.output_scale = None  # 动态量化模式
```

**同样修复了 `QRMSNormGated`**（第143-155行）：
```python
def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    # Handle backward compatibility: if scales are not in state_dict, set to default values
    if prefix + 'z_scale' in state_dict:
        self.z_scale = state_dict[prefix + 'z_scale']
        del state_dict[prefix + 'z_scale']
    else:
        self.z_scale = 0.0

    if prefix + 'output_scale' in state_dict:
        self.output_scale = state_dict[prefix + 'output_scale']
        del state_dict[prefix + 'output_scale']
    else:
        self.output_scale = None
```

---

### Bug 2: `quamba_mixer_seq.py` - Missing key: 'lm_head.bias'

**错误信息**：
```python
RuntimeError: Error(s) in loading state_dict for QuambaLMHeadModel:
    Missing key(s) in state_dict: "lm_head.bias".
```

**错误位置**：`quamba/quamba_mixer_seq.py:417`

**根本原因**：
```python
# 原始代码（第416-417行）
if lm_head_layer == "Linear":
    self.lm_head = torch.nn.Linear(d_model, vocab_size)  # 默认 bias=True
```

- 原始 Mamba 模型的 `lm_head` **没有 bias**
- 但 PyTorch 的 `nn.Linear` 默认 `bias=True`
- 保存的模型没有 `lm_head.bias`，加载时报错

**修复方案**：
```python
# 修复后的代码（第416-417行）
if lm_head_layer == "Linear":
    self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)  # 匹配原始Mamba
```

---

### Bug 3: `quamba_mixer_seq.py` - TypeError: linear() argument 'input' must be Tensor, not tuple

**错误信息**：
```python
TypeError: linear(): argument 'input' (position 1) must be Tensor, not tuple
```

**错误位置**：Forward pass 中 `lm_head` 接收输入时

**根本原因**：

Config 中 `norm_cfg` 被设置为 `{"norm": "QRMSNorm"}`，这是全局的，导致：
1. 加载时 `norm_f` 被创建为 `QRMSNorm`
2. 但保存的模型中 `norm_f` 是 FP16 的 `RMSNorm`（因为没有 `--quantize_lm_head`）
3. `QRMSNorm` 当 `output_scale=None` 时返回 **tuple**: `(y, per_token_scale)`
4. 但 `lm_head` 期望单个 Tensor，导致类型错误

**代码逻辑**（`qNorm.py:86-91`）：
```python
else:
    # output per_token scaling factor if output_scale is None
    y = y.reshape(x_shape_og)
    residual_out = residual_out.reshape(x_shape_og)
    per_token_scale = per_token_scale.reshape(x_shape_og[0:-1])
    return (y, per_token_scale) if not prenorm else (y, residual_out, per_token_scale)
```

**修复方案**：

当 `lm_head` 是 FP16 `Linear` 时，强制 `norm_f` 也使用 FP16 `RMSNorm`：

```python
# 修复后的代码（第416-424行）
if lm_head_layer == "Linear":
    self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
    # For Quamba1 (no quantized lm_head), norm_f should also be FP16 RMSNorm
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
        norm_epsilon = getattr(config, 'norm_epsilon', 1e-5)
        self.backbone.norm_f = RMSNorm(d_model, eps=norm_epsilon, **factory_kwargs)
    except ImportError:
        pass  # Keep QRMSNorm if RMSNorm is not available
```

**逻辑**：
- 如果 `lm_head_cfg["layer"]` 是 `"Linear"`（FP16），说明没有量化 lm_head
- 此时 `norm_f` 也应该是 FP16 的 `RMSNorm`（返回单个Tensor）
- 加载后强制替换 `norm_f` 为 `RMSNorm`

---

### Bug 4: `quamba_mixer_seq.py` - RuntimeError: expected mat1 and mat2 to have the same dtype

**错误信息**：
```python
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::Half != float
```

**错误位置**：Forward pass 中矩阵乘法

**根本原因**：

- `norm_f` (FP16 `RMSNorm`) 输出 FP16 Tensor
- 但保存的 `lm_head.weight` 可能是 FP32
- 矩阵乘法时类型不匹配

**修复方案**：

加载模型后，强制 `lm_head` 转换为 FP16：

```python
# 修复后的代码（第430-444行）
@classmethod
def from_pretrained(cls, pretrained_model_name, device=None, **kwargs):
    cache_dir = kwargs.pop("cache_dir", None)
    config_data = load_config_hf(pretrained_model_name, cache_dir=cache_dir)
    config = QuambaConfig(**config_data)
    model = cls(config, device="cpu", **kwargs)
    loaded_model = load_state_dict_hf(pretrained_model_name, device="cpu", cache_dir=cache_dir)
    model.load_state_dict(loaded_model)
    del loaded_model
    torch.cuda.empty_cache()
    gc.collect()
    # Ensure lm_head is FP16 for compatibility
    if hasattr(model, 'lm_head') and isinstance(model.lm_head, torch.nn.Linear):
        model.lm_head = model.lm_head.half()
    return model.to(device)
```

---

## 📊 修复总结表

| Bug | 文件 | 行数 | 问题 | 修复方式 |
|-----|------|------|------|---------|
| 1 | `qNorm.py` | 42-44 | 直接访问不存在的键 | 先检查键是否存在 |
| 1b | `qNorm.py` | 143-143 | 同上（`QRMSNormGated`） | 同上 |
| 2 | `quamba_mixer_seq.py` | 417 | `Linear` 默认有 bias | 显式设置 `bias=False` |
| 3 | `quamba_mixer_seq.py` | 416-424 | `norm_f` 返回 tuple | 替换为 FP16 `RMSNorm` |
| 4 | `quamba_mixer_seq.py` | 441-443 | dtype 不匹配 | 强制转换为 FP16 |

---

## 🔄 回档方案

如果需要恢复原始代码，请按以下步骤操作：

### 1. 恢复 `qNorm.py`

**恢复第42-44行**：
```python
def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    self.output_scale = state_dict[prefix + 'output_scale']
    del state_dict[prefix + 'output_scale']
```

**恢复第143-143行**（`QRMSNormGated`）：
```python
def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    self.z_scale = state_dict[prefix + 'z_scale']
    self.output_scale = state_dict[prefix + 'output_scale']
    del state_dict[prefix + 'z_scale']
    del state_dict[prefix + 'output_scale']
```

### 2. 恢复 `quamba_mixer_seq.py`

**恢复第417行**：
```python
if lm_head_layer == "Linear":
    self.lm_head = torch.nn.Linear(d_model, vocab_size)
```

**删除第418-424行**（norm_f 替换逻辑）：
```python
# 删除这些行
# For Quamba1 (no quantized lm_head), norm_f should also be FP16 RMSNorm
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
    norm_epsilon = getattr(config, 'norm_epsilon', 1e-5)
    self.backbone.norm_f = RMSNorm(d_model, eps=norm_epsilon, **factory_kwargs)
except ImportError:
    pass  # Keep QRMSNorm if RMSNorm is not available
```

**恢复第430-444行**（删除 lm_head.half()）：
```python
@classmethod
def from_pretrained(cls, pretrained_model_name, device=None, **kwargs):
    cache_dir = kwargs.pop("cache_dir", None)
    config_data = load_config_hf(pretrained_model_name, cache_dir=cache_dir)
    config = QuambaConfig(**config_data)
    model = cls(config, device="cpu", **kwargs)
    loaded_model = load_state_dict_hf(pretrained_model_name, device="cpu", cache_dir=cache_dir)
    model.load_state_dict(loaded_model)
    del loaded_model
    torch.cuda.empty_cache()
    gc.collect()
    return model.to(device)
```

### 3. Git 回档命令

如果使用 Git 管理：
```bash
# 查看修改
git diff quamba/qNorm.py quamba/quamba_mixer_seq.py

# 回档单个文件
git checkout HEAD~1 quamba/qNorm.py
git checkout HEAD~1 quamba/quamba_mixer_seq.py

# 或者回档到特定commit
git log --oneline  # 找到修改前的commit
git checkout <commit-hash> quamba/qNorm.py quamba/quamba_mixer_seq.py
```

---

## 🎯 根本原因分析

### 代码设计缺陷

**问题1：保存 config 时的逻辑错误**

`modelutils_mamba.py:923`：
```python
model.config.norm_cfg = {"norm": model.backbone.layers[0].norm.__class__.__name__}
```

这行代码把 **Block 内部的 norm 类型**（`QRMSNorm`）应用到了**所有 norm**（包括 `norm_f`），但：
- 当不加 `--quantize_lm_head` 时，`norm_f` 实际是 FP16 `RMSNorm`
- Config 应该分别保存 block norm 和 final norm 的类型

**建议改进**：
```python
# 更好的设计
model.config.norm_cfg = {
    "norm": model.backbone.layers[0].norm.__class__.__name__,  # Block内的norm
    "final_norm": model.backbone.norm_f.__class__.__name__     # 最终的norm_f
}
```

**问题2：量化逻辑与加载逻辑不一致**

- **量化时**（`quantize_fp16_model`）：
  - Block 内的 norm 总是被量化为 `QRMSNorm`（第645行）
  - `norm_f` 只有在 `quantize_lm_head=True` 时才被量化（第696行）

- **加载时**（`QuambaMixerModel.__init__`）：
  - 从 config 的 `norm_cfg["norm"]` 读取类型
  - **没有区分 block norm 和 final norm**

**设计问题**：
- 量化逻辑知道何时量化 `norm_f`
- 但保存/加载逻辑不知道这个区别
- 导致加载时创建了错误的 norm 类型

---

## 🔧 正确的使用方式

### Quamba1（Mamba1-130M）量化命令

```bash
cd /workspace/Quamba
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize --w_bits 8 --a_bits 8 \
  --eval_zero_shot --task_list lambada_openai \
  --batch_size 16 --log_dir logs \
  --pretrained_dir ./pretrained_models \
  --output_subdir 1106YzResearchQuamba1
```

**关键点**：
- ✅ 使用 `--w_bits 8 --a_bits 8`
- ❌ **不要** 加 `--quantize_lm_head`
- ❌ **不要** 加 `--quantize_embedding`
- ❌ **不要** 加 `--apply_gptq`

### Quamba1 评估命令

```bash
cd /workspace/Quamba
python3 main.py 1106YzResearchQuamba1/default/quamba-130m-w8a8 \
  --pretrained_dir ./pretrained_models \
  --eval_zero_shot --task_list lambada_openai \
  --batch_size 16 --log_dir logs
```

### Quamba2（Mamba2-130M）量化命令

```bash
cd /workspace/Quamba
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba2-130m \
  --quantize --w_bits 4 --a_bits 8 \
  --group_heads --apply_gptq --quantize_embedding --quantize_lm_head \
  --eval_zero_shot --task_list lambada_openai \
  --batch_size 16 --log_dir logs \
  --pretrained_dir ./pretrained_models \
  --output_subdir 1106YzResearchQuamba2
```

**关键点**：
- ✅ 使用 `--w_bits 4 --a_bits 8`
- ✅ **必须** 加 `--quantize_lm_head`
- ✅ **必须** 加 `--quantize_embedding`
- ✅ **必须** 加 `--apply_gptq`
- ✅ **必须** 加 `--group_heads`（Mamba2特有）

---

## ✅ 验证修复

修复后，以下命令应该能正常运行：

```bash
# 1. 量化
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize --w_bits 8 --a_bits 8 \
  --eval_zero_shot --task_list lambada_openai \
  --batch_size 16 --log_dir logs \
  --pretrained_dir ./pretrained_models \
  --output_subdir 1106YzResearchQuamba1

# 2. 评估
python3 main.py 1106YzResearchQuamba1/default/quamba-130m-w8a8 \
  --pretrained_dir ./pretrained_models \
  --eval_zero_shot --task_list lambada_openai \
  --batch_size 16 --log_dir logs
```

**预期结果**：
- ✅ 模型加载成功
- ✅ 评估正常运行
- ✅ 输出 lambada_openai 的准确率和困惑度

---

## 📚 相关文档

- `SESSION_HISTORY.md` - Session 6 记录了完整的 debug 过程
- `QUAMBA_QUANTIZATION_COMPLETE_GUIDE.md` - 量化机制完整指南
- `DOCUMENTATION_INDEX.md` - 文档索引

---

## 🔖 标签

`#bug-fix` `#quamba1` `#model-loading` `#compatibility` `#critical`

---

**维护者**: Claude (Sonnet 4.5) + Yizhi Chen
**最后更新**: 2025-11-06
**状态**: ✅ 已修复并验证
