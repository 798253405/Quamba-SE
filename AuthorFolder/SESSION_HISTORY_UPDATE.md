# SESSION_HISTORY.md 更新内容

请将以下内容**替换**原SESSION_HISTORY.md的"临时调试代码位置清单"部分（从3490行开始到文件末尾）：

---

## 临时调试代码位置清单 (待删除)

### 第一轮调试代码（已禁用）

#### 目的
用于对比CUDA kernel和PyTorch模拟的SSM输出，验证实现正确性。

#### 已禁用的代码块

##### 1. qSelectiveScan.py
**位置**: 第243-249行 (DISABLED)
```python
# ===== TEMPORARY DEBUG CODE - DISABLED (use qMambaLayer debug instead) =====
# if not hasattr(self, '_debug_call_count'):
#     self._debug_call_count = 0
# ...
```
**功能**: 打印Mode 1/2/3的SSM输入/输出

**位置**: 第279-285行 (DISABLED)
```python
# ===== TEMPORARY DEBUG CODE - DISABLED (use qMambaLayer debug instead) =====
# if not hasattr(self, '_debug_call_count'):
# ...
```
**功能**: 打印Baseline的SSM输入/输出

##### 2. qConvLayer.py
**位置**: 第129-133行 (DISABLED)
```python
# ===== TEMPORARY DEBUG CODE - DISABLED (use qMambaLayer debug instead) =====
# if _CONV1D_LAYER_COUNTER == 0:
#     print(f"\n[Conv1D Mode 2] Scales:")
```
**功能**: 打印Mode 2的Conv1D scales

**位置**: 第166-170行 (DISABLED)
```python
# ===== TEMPORARY DEBUG CODE - DISABLED (use qMambaLayer debug instead) =====
# if _CONV1D_LAYER_COUNTER == 0:
#     print(f"\n[Conv1D Baseline] Scales:")
```
**功能**: 打印Baseline的Conv1D scales

##### 3. selective_scan_SE.py
**位置**: 第184-201行 (保留，用于Mode 3调试)
```python
# ===== TEMPORARY DEBUG CODE - TO BE DELETED =====
num_outliers = overflow_mask.sum().item()
print(f"\n[SSM DEBUG] Dual-scale dequantization:")
...
# ===== END TEMPORARY DEBUG CODE =====
```
**功能**: 打印dual-scale dequantization的详细信息

---

### 第二轮调试代码（当前活跃） - 完整数据流追踪

#### 目的
追踪完整数据流（Conv1D → x_proj → dt_proj → SSM → had → out_proj），对比Baseline和Mode 2在每个阶段的差异。

**原因**: 虽然SSM单次调用输入/输出完全一致，但最终accuracy仍有差异（Baseline=0.39, Mode 2=0.36-0.37）。需要检查是否其他层引入了差异。

#### 当前活跃的代码块

##### qMambaLayer.py - 完整数据流调试

**调试范围设置**:
```python
# 位置: 第962行
DEBUG_ENABLED = self.layer_idx is not None and self.layer_idx >= 21
```
- **层范围**: 仅最后3层（layer_idx >= 21，假设24层模型）
- **调用次数**: 每层仅打印前3次调用
- **打印格式**: dtype + 前3个值（最小化输出）

**1. Conv1D输出** (第964-973行)
```python
if DEBUG_ENABLED and self._debug_step_count <= 3:
    print(f"\n{'='*80}")
    print(f"[Layer {self.layer_idx} Call #{self._debug_step_count}] After Conv1D")
    print(f"  x dtype: {x.dtype}, shape: {x.shape}")
    print(f"  x first 3 values: {x.flatten()[:3].tolist()}")
    if hasattr(x, '_dual_scale_overflow_mask'):
        print(f"  x has dual-scale metadata")
```
**功能**: 打印Conv1D输出（所有模式共享）

**2. Dual-path分离** (第979-983行, FP32模式路径)
```python
if DEBUG_ENABLED and self._debug_step_count <= 3:
    print(f"  x_for_xproj dtype: {x_for_xproj.dtype}, first 3: ...")
    print(f"  x_for_ssm dtype: {x_for_ssm.dtype}, first 3: ...")
```
**功能**: 打印dual-path分离后的INT8和FP32版本（仅Mode 1/2/3）

**3. x_proj输出** (第989-993行, FP32模式; 第1016-1020行, Baseline)
```python
if DEBUG_ENABLED and self._debug_step_count <= 3:
    print(f"  After x_proj: x_dbl dtype: {x_dbl.dtype}, first 3: ...")
    print(f"  dt (before dt_proj) dtype: {dt.dtype}, first 3: ...")
```
**功能**: 打印x_proj输出和dt（dt_proj之前）

**4. dt_proj + B/C/z** (第1000-1006行, FP32模式; 第1026-1032行, Baseline)
```python
if DEBUG_ENABLED and self._debug_step_count <= 3:
    print(f"  After dt_proj: dt dtype: {dt.dtype}, first 3: ...")
    print(f"  B dtype: {B.dtype}, first 3: ...")
    print(f"  C dtype: {C.dtype}, first 3: ...")
    print(f"  z dtype: {z.dtype if z is not None else 'None'}, first 3: ...")
```
**功能**: 打印所有SSM输入参数

**5. SSM输出** (第1041-1045行)
```python
if DEBUG_ENABLED and self._debug_step_count <= 3:
    print(f"  After SSM: y dtype: {y.dtype}, first 3: ...")
```
**功能**: 打印SSM输出

**6. Hadamard变换** (第1050-1054行)
```python
if DEBUG_ENABLED and self._debug_step_count <= 3:
    print(f"  After had: y dtype: {y.dtype}, first 3: ...")
```
**功能**: 打印Hadamard变换输出

**7. out_proj最终输出** (第1059-1063行)
```python
if DEBUG_ENABLED and self._debug_step_count <= 3:
    print(f"  After out_proj: out dtype: {out.dtype}, first 3: ...")
    print(f"{'='*80}\n")
```
**功能**: 打印最终输出

#### 覆盖范围总结

**完整数据流**:
```
Conv1D output (x)
    ↓
[FP32 modes only] x_for_xproj (INT8) / x_for_ssm (FP32)
    ↓
x_proj output (x_dbl) → split → dt, B, C
    ↓
dt_proj output (dt)
    ↓
SSM (u=x, dt, B, C, z) → y
    ↓
Hadamard (y) → y
    ↓
out_proj (y) → out (final)
```

**对比点**:
- Baseline: x (INT8) → SSM
- Mode 2: x_for_ssm (FP32, 模拟INT8) → SSM
- 其他层（x_proj, dt_proj, had, out_proj）应该完全相同（都用INT8版本的x）

---

### 验证结果总结

#### Mode 2 vs Baseline对比 - 第一轮（SSM层级）
- **SSM输入**: 完全一致 ✅
- **SSM输出**: 完全一致 ✅
- **结论**: PyTorch模拟INT8的SSM实现正确

#### Mode 2 vs Baseline对比 - 第二轮（完整数据流）
**待运行**: 用户正在运行测试，查看是否其他层（x_proj, dt_proj, had, out_proj）引入了差异

**预期**:
- 如果所有中间值都一致 → 差异可能来自浮点累积误差或其他非确定性因素
- 如果某层开始出现差异 → 定位到具体问题层进行修复

#### Mode 3 Dual-Scale验证
- **Outliers检测**: 1759/2826240 (0.06%) ✅
- **INT8量化**: 正确 ✅
- **Dequantization**: 正确，但outliers精度损失严重 ⚠️
- **SSM输出**: 与Baseline差异巨大（outliers导致）
- **结论**: Dual-scale实现正确，但scale_factor=3.0太大导致精度损失

---

### 调试代码删除计划

#### 第一阶段（已完成）
- ✅ 禁用第一轮调试代码（qSelectiveScan.py, qConvLayer.py scales打印）

#### 第二阶段（待定）
- 等待第二轮测试结果
- 如果找到问题根因，删除所有qMambaLayer.py的调试代码
- 如果需要进一步调试，可能保留部分代码或添加新的调试点

#### 第三阶段（最终清理）
- 删除所有临时调试代码
- 更新文档记录最终结论
- 提交干净的代码版本
