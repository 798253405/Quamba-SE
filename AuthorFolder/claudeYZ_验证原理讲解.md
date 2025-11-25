# Percentile Alpha 验证原理讲解

## 1. percentile_alpha 是如何影响量化的？

### 量化的核心：确定 scale（量化步长）

在 PTQ (Post-Training Quantization) 中，最关键的参数是 **scale**（量化步长）：

```python
# 量化公式
quantized_value = round(float_value / scale)
# 反量化公式
dequantized_value = quantized_value * scale
```

**scale 越大** → 量化范围越大 → 单个 bin 越宽 → 量化误差越大
**scale 越小** → 量化范围越小 → 单个 bin 越窄 → 量化误差越小，但可能截断极端值

---

### percentile_alpha 如何决定 scale？

`percentile_alpha` 控制用哪个分位数来确定 scale：

```python
# quamba/observer.py:92
cur_max = torch.quantile(w.abs().reshape(-1), self.percentile_alpha)
```

**示例**：假设激活值为 `[1, 2, 3, ..., 98, 99, 500]`（最后一个是极端 outlier）

| percentile_alpha | 取哪个分位数 | cur_max | scale (cur_max/127) | 覆盖范围 | 量化误差 |
|------------------|-------------|---------|---------------------|----------|---------|
| 0.99             | 99% 分位数   | 99      | 0.78                | 99%      | 小       |
| 0.99999          | 99.999%     | ~499    | 3.93                | 99.999%  | 较大     |
| 1.0              | 100% (max)  | 500     | 3.94                | 100%     | 最大     |

**关键差异**：
- `pa=0.99999`：裁剪掉最极端的 0.001%，scale ≈ 3.93
- `pa=1.0`：包含所有值，scale = 3.94

虽然 scale 差异只有 0.25%，但对于整个网络的累积误差可能有影响。

---

## 2. 量化模型中存储了什么？

### 文件结构

```bash
pretrained_models/yzreproduceauthors/quamba2-130m-w4a8/
├── config.json           # 模型配置（但不包含 percentile_alpha）
└── pytorch_model.bin     # 模型权重 + 量化参数
```

### pytorch_model.bin 中的内容

**不同于 QAT (Quantization-Aware Training)**，PTQ 模型中：
- **量化权重**：以 int8/uint8 格式存储（W4 打包成 uint8）
- **量化 scale**：以 float32 存储，用于反量化

```python
# 示例：第0层的参数
state_dict = {
    # 量化权重（4-bit 打包成 uint8）
    "backbone.layers.0.mixer.in_proj.weight": tensor([...], dtype=torch.uint8),

    # 权重的 scale（用于反量化）
    "backbone.layers.0.mixer.in_proj.weight_scale": tensor(0.012345, dtype=torch.float32),

    # 激活的 scale（用于运行时量化）
    "backbone.layers.0.mixer.x_conv_out_scale": tensor(0.056789, dtype=torch.float32),
    "backbone.layers.0.mixer.ssm_state_act_scale": tensor(0.023456, dtype=torch.float32),

    # 其他参数...
}
```

**关键点**：
- `percentile_alpha` **不会**直接存储在模型中
- `percentile_alpha` 的影响体现在 **scale 参数的值** 上
- 不同的 `percentile_alpha` → 不同的 scale 值 → 不同的量化误差

---

## 3. 验证方案原理

### 方案 1：claudeYZ_verify_percentile_alpha.py

**原理**：独立测试 `PerTensorPercentileObserver` 类

```python
# 创建测试数据（包含 outlier）
activation = torch.randn(1024, 512) * 10
outlier_indices = torch.randperm(activation.numel())[:1%]
activation[outlier_indices] = torch.randn(...) * 1000  # 极端值

# 测试不同的 percentile_alpha
for alpha in [0.99, 0.999, 0.9999, 0.99999, 1.0]:
    observer = PerTensorPercentileObserver(..., percentile_alpha=alpha)
    observer.update(activation)
    scale, base = observer.get_quantization_parameters()

    # 验证：scale 应该单调递增
    # 验证：pa=1.0 应该覆盖 100% 的值
```

**验证点**：
1. ✅ scale 随 alpha 单调递增？
2. ✅ pa=1.0 的覆盖率是否为 100%？
3. ✅ pa=0.99999 是否裁剪了 ~0.001% 的极端值？

**优点**：快速（30秒），不需要运行完整量化

---

### 方案 2：claudeYZ_inspect_model.py

**原理**：直接读取 `pytorch_model.bin`，对比不同模型的 scale 值

```python
# 加载两个模型
state_dict1 = torch.load("yzreproduceauthors/quamba2-130m-w4a8/pytorch_model.bin")
state_dict2 = torch.load("yz100percent/quamba2-130m-w4a8/pytorch_model.bin")

# 找出所有 scale 参数
scale_keys = [k for k in state_dict1.keys() if 'scale' in k.lower()]

# 对比 scale 值
for key in scale_keys:
    scale1 = state_dict1[key].item()
    scale2 = state_dict2[key].item()
    diff_pct = (scale2 - scale1) / scale1 * 100
    print(f"{key}: {scale1:.6f} vs {scale2:.6f} (差异 {diff_pct:.2f}%)")
```

**验证点**：
1. ✅ 两个模型的 scale 是否有差异？
2. ✅ 差异的平均值/中位数/最大值是多少？
3. ✅ 如果差异 > 1%，说明 percentile_alpha 有显著影响

**优点**：直接从模型文件验证，不需要重新量化

**前提**：你已经有两个用不同 `percentile_alpha` 量化的模型

---

### 方案 3：添加日志输出

**原理**：在 `quamba/observer.py` 中添加 `logging.debug()`，量化时实时显示参数

```python
# quamba/observer.py:114
def get_quantization_parameters(self):
    logging.debug(f"PerTensorPercentileObserver: "
                  f"percentile_alpha={self.percentile_alpha:.5f}, "
                  f"w_max={self.w_max.item():.6f}")
    ...
```

**运行时**：
```bash
python main.py ... --percentile_alpha 1.0 --verbose 2>&1 | grep percentile
```

**预期输出**：
```
DEBUG ... PerTensorPercentileObserver: percentile_alpha=1.00000, w_max=123.456789
DEBUG ... PerTensorPercentileObserver: percentile_alpha=1.00000, w_max=234.567890
...
```

**验证点**：
1. ✅ 日志中显示的 `percentile_alpha` 是否等于命令行参数？
2. ✅ 不同 `percentile_alpha` 是否产生不同的 `w_max`？

**优点**：实时验证，可以看到参数传递的完整路径

---

## 4. 为什么 percentile_alpha 不直接存储在模型中？

### 原因 1：PTQ 的特性

PTQ (Post-Training Quantization) 是**一次性**的过程：

```
FP16 模型 → [Calibration + GPTQ] → 量化模型
           ↑
      percentile_alpha 只在这里使用
```

量化完成后，`percentile_alpha` 已经"烧入"到 scale 参数中了，不再需要。

---

### 原因 2：推理时不需要

推理时只需要：
1. 量化权重（uint8）
2. scale 参数（float32）
3. 反量化公式：`dequantized = quantized * scale`

**不需要知道** scale 是怎么计算出来的（用了哪个分位数）。

---

### 类比：烘焙蛋糕

- **percentile_alpha** = 烤箱温度（180°C 还是 200°C）
- **scale** = 烤好的蛋糕
- **模型文件** = 装蛋糕的盒子

你不需要在盒子上写"用 180°C 烤的"，因为蛋糕已经烤好了。
但是，用不同温度烤出来的蛋糕**口感不同**（就像不同 scale 导致不同的准确率）。

---

## 5. 验证流程推荐

### 快速验证（5 分钟）

```bash
# 1. 测试 observer 基本功能
python3 claudeYZ_verify_percentile_alpha.py

# 2. 检查已保存模型的结构
python3 claudeYZ_inspect_model.py pretrained_models/yzreproduceauthors/quamba2-130m-w4a8
```

---

### 深度验证（20 分钟）

```bash
# 1. 运行一次量化，使用特殊值 + verbose
python3 main.py ... --percentile_alpha 0.5 --verbose 2>&1 | tee log_pa05.txt

# 2. 从日志中验证参数
grep "percentile_alpha" log_pa05.txt

# 3. 检查生成的模型
python3 claudeYZ_inspect_model.py pretrained_models/yzreproduceauthors/quamba2-130m-w4a8
```

---

### 终极验证（2 小时）

```bash
# 1. 量化 3 个不同 percentile_alpha 的模型
python3 main.py ... --percentile_alpha 0.99 --pretrained_dir ./models_pa099
python3 main.py ... --percentile_alpha 0.99999 --pretrained_dir ./models_pa099999
python3 main.py ... --percentile_alpha 1.0 --pretrained_dir ./models_pa100

# 2. 对比它们的 scale 参数
python3 claudeYZ_inspect_model.py models_pa099/... models_pa099999/...
python3 claudeYZ_inspect_model.py models_pa099999/... models_pa100/...

# 3. 对比准确率
# 如果 scale 差异 > 1%，但准确率差异 < 0.1%，说明量化对这个模型很鲁棒
# 如果 scale 差异 > 1%，且准确率差异 > 0.5%，说明 percentile_alpha 很重要
```

---

## 6. 关键代码位置

### percentile_alpha 的传递路径

```
main.py:91
  args = parse_options()  # 解析命令行参数
  args.percentile_alpha = 1.0

main.py:28
  model = quantize_model_mamba(model, ..., args)

quamba/modelutils_mamba.py:816
  act_scales = run_quamba2_calibration(..., percentile_alpha=args.percentile_alpha)

quamba/modelutils_mamba.py:168
  observer = PerTensorPercentileObserver(..., percentile_alpha=percentile_alpha)

quamba/observer.py:77
  self.percentile_alpha = percentile_alpha  # 存储到 observer 实例

quamba/observer.py:92
  cur_max = torch.quantile(w.abs(), self.percentile_alpha)  # 实际使用！
```

### scale 的存储位置

```
quamba/observer.py:117
  return _get_minmax_quantization_params(w_max=self.w_max, ...)
    ↓
  scales = w_max / q_max  # 计算 scale

quamba/modelutils_mamba.py:367-374
  act_scales[layer_idx]["x_conv_out:input"] = scale  # 保存到 act_scales 字典

quamba/qMamba2.py:from_fp16()
  mixer = W4A8QMamba2(
      ...
      x_conv_out_scale=act_scales["x_conv_out:input"],  # 传递给量化模块
  )

quamba/quamba_mixer_seq.py:445
  torch.save(self.state_dict(), model_path)  # scale 作为模型参数保存
    ↓
  pytorch_model.bin: {
      "backbone.layers.0.mixer.x_conv_out_scale": tensor(0.056789),
      ...
  }
```

---

## 7. 常见问题

### Q1: 为什么我改了 percentile_alpha，准确率没变化？

**可能原因**：
1. **GPTQ 随机性**掩盖了影响（±1.75% >> percentile_alpha 的影响）
2. 模型对量化误差不敏感（高度鲁棒）
3. 参数没有正确传递（用 `--verbose` 检查日志）

**验证方法**：
```bash
python3 claudeYZ_inspect_model.py model_pa099/... model_pa100/...
```
如果 scale 有显著差异（>1%），但准确率没变，说明模型很鲁棒。

---

### Q2: 如何确认 scale 的差异确实影响了准确率？

**需要固定 GPTQ 的随机种子**：

```python
# 修改 quamba/data_loaders.py:19
random.seed(42)  # 固定 seed
```

然后重新量化 3 次，对比结果。

---

### Q3: 能否从模型文件反推 percentile_alpha？

**不能**。因为：
- scale 只是一个浮点数，没有"来源"信息
- 相同的 scale 可能来自不同的 (w_max, percentile_alpha) 组合

这就像从蛋糕的口感，无法精确反推烤箱温度一样。

---

## 8. 总结

| 验证方法 | 速度 | 可靠性 | 需要重新量化 | 推荐场景 |
|---------|------|--------|------------|---------|
| claudeYZ_verify_percentile_alpha.py | ⚡ 快 | ⭐⭐⭐ | ❌ | 快速验证 observer 功能 |
| claudeYZ_inspect_model.py | ⚡ 快 | ⭐⭐⭐⭐⭐ | ❌ | 对比已保存的模型 |
| --verbose 日志 | 🐢 慢 | ⭐⭐⭐⭐ | ✅ | 实时查看参数传递 |
| 对比准确率 | 🐢 很慢 | ⭐⭐⭐⭐⭐ | ✅ | 终极验证 |

**最佳实践**：
1. 先运行 `claudeYZ_verify_percentile_alpha.py` 确认 observer 没问题
2. 然后运行 `claudeYZ_inspect_model.py` 对比已有模型的 scale
3. 如果 scale 差异 > 1%，说明 percentile_alpha 有影响
4. 如果 scale 差异 < 0.1%，说明参数可能没生效，用 `--verbose` 检查日志

---

## 9. 最新研究进展 (2025-11-06)

### W4A5-SE Dual-Scale方案

**核心思路**：4bit activation + 1bit scale选择器
```
Normal值:  q = round(x / scale1)         → 4bit
Outlier值: q = round(x / scale1 / scale2) → 4bit + flag=1
存储: 5bit/value (vs 8bit当前)
```

**关键发现**：
- ✅ **SSM可以支持dual-scale**（先dequant再FP32计算）
- ❌ **GEMM不支持**（accumulator混用，无法区分来源）
- ✅ **应用位置**：Conv1D输出量化（SSM输入x）
- ✅ **收益**：节省50%带宽（10MB→5MB per layer）

**SSM必须FP32的原因**（数学原理，非实现选择）：
1. Exponential函数：动态范围[0.001, 1000+]
2. 递归状态累积：误差线性累积
3. 非线性激活：exp/log无法用INT近似

**量化收益来源**（即使SSM是FP32）：
- DRAM带宽是瓶颈（占70%能耗）
- GEMM仍然INT8加速（占95% FLOPs）
- SSM只占5%计算，FP32影响有限

**实现位置**：
- `csrc/causal_conv1d/quant_causal_conv1d_fwd_kernel.cuh:149` - Conv输出量化
- `csrc/selective_scan/quant_sscan_fwd_kernel.cuh:157` - SSM输入dequant

**硬件开销**（ASIC）：
- Dequant MUX: +50 gates per lane
- Flag存储: +12.5% DRAM访问
- 总开销: <5%

详见：`SESSION_HISTORY.md` Session 5
