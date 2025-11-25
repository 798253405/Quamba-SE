# FP32 SSM Modes Final Fix - Session 8

**Date**: 2025-11-07
**Issue**: Mode 1 (FP32_SSM_INPUT) and Mode 2 (FLOAT_SIM_ASIC_INT8) both showing 36% accuracy instead of expected 39%

---

## Problem Summary

### Original Issue
- **Baseline (INT8 CUDA)**: 39% accuracy ✓
- **Mode 2 (FLOAT_SIM_ASIC_INT8)**: 36% accuracy ✗ (Expected: 39%)
- **Mode 1 (FP32_SSM_INPUT)**: 36% accuracy ✗ (Expected: 39%)

### Root Cause

**Misunderstanding of design goals:**

The original buggy implementation had:
- Mode 1: Conv1D with CUTLASS FP32 kernel → FP32 output
- Mode 2: Conv1D with FP32 simulation → FP32 output (quantized values)

This caused both modes to use PyTorch SSM implementation instead of INT8 CUDA kernel, leading to 36% accuracy.

---

## Correct Design (Clarified)

### **Key Insight: Both modes use INT8 CUDA kernel!**

**User's requirement:**
> "我让你都用int8cuda 你懂么。只有输出会有变化。"
> (Use INT8 CUDA for all. Only the output changes.)

### **Correct Understanding:**

| Mode | Conv1D运算 | Conv1D输出 | SSM输入 | 预期 |
|------|----------|-----------|---------|------|
| **Baseline** | INT8 CUDA | INT8 | INT8 | 39% ✓ |
| **Mode 1** | INT8 CUDA | INT8 → FP32反量化 | FP32 | 39% |
| **Mode 2** | INT8 CUDA | INT8 → FP32反量化 | FP32 | 39% |

**All three modes:**
- Use the same INT8 CUDA kernel for Conv1D
- Use the same quantization scale (0.0121 from default model)
- **Only difference**: SSM input type (INT8 vs FP32)

**Why Mode 1 & Mode 2 should both be 39%:**
- Conv1D computation is identical to Baseline
- INT8 → FP32 dequantization doesn't add information (precision already lost)
- Only tests whether FP32 SSM implementation matches INT8 CUDA kernel

---

## Code Changes

### 1. qConvLayer.py (lines 124-172)

**Key change**: Both Mode 1 and Mode 2 use INT8 CUDA kernel

```python
# Mode 1: FP32 SSM Input (use INT8 CUDA kernel, return INT8)
if fp32_ssm_input:
    y = quant_causal_conv1d_cuda.fwd(
            x, self.input_scale,
            self.weight, self.weight_scale,
            self.output_scale,
            self.bias_scale, self.bias,
            None, None, None, True
        )
    return y  # INT8

# Mode 2: FP32 SSM (use INT8 CUDA kernel, return INT8)
elif float_sim_asic_int8:
    y = quant_causal_conv1d_cuda.fwd(
            x, self.input_scale,
            self.weight, self.weight_scale,
            self.output_scale,
            self.bias_scale, self.bias,
            None, None, None, True
        )
    return y  # INT8
```

### 2. qMambaLayer.py (lines 742 & 1133)

**Key change**: Simplified logic - both modes return INT8, both dequantize

```python
# Dual-path logic: Both Mode 1 and Mode 2 use INT8 CUDA kernel
# Conv1D returns INT8, dequantize to FP32 for SSM
# Only difference from Baseline: SSM receives FP32 instead of INT8
if fp32_mode_enabled:
    if x.dtype == torch.int8:
        # Conv1D returned INT8 from CUDA kernel (same as Baseline)
        # Dequantize to FP32 for SSM
        x_for_xproj = x  # INT8 for x_proj (unchanged from Baseline)
        x_for_ssm = x.float() * self.conv1d.output_scale  # Dequantize to FP32 for SSM
    else:
        raise ValueError(f"Unexpected dtype in fp32_mode_enabled: {x.dtype}, expected torch.int8")
```

**Applied to 2 classes:**
- W4A8QMamba (line ~742)
- W8A8QMamba (line ~1133)

---

## Data Flow Comparison

### Baseline (INT8 CUDA):
```
hidden_states (FP16)
  → in_proj (W8A8) → INT8
  → Conv1D (INT8 CUDA kernel) → INT8
  → x_proj → INT8
  → SSM (INT8 CUDA kernel, QSScan) → FP16
  → out_proj → FP16

Result: 39% accuracy
```

### Mode 1 & Mode 2 (FP32 SSM):
```
hidden_states (FP16)
  → in_proj (W8A8) → INT8
  → Conv1D (INT8 CUDA kernel, SAME AS BASELINE) → INT8
  → Dequantize: INT8 * output_scale → FP32
  → Split:
      • x_proj: use INT8 (unchanged)
      • SSM: use FP32 (dequantized)
  → SSM (FP32 PyTorch, selective_scan_SE_float) → FP16
  → out_proj → FP16

Expected: 39% (same as Baseline, only type conversion)
Actual before fix: 36% (using wrong PyTorch implementation)
```

**The ONLY difference from Baseline:**
- SSM input: INT8 vs FP32 (dequantized)
- SSM implementation: CUDA kernel vs PyTorch

---

## Why Both Modes Should Be 39%

### **No precision gain from dequantization:**
```
Original FP32 value: 0.05322
  ↓ (Conv1D quantization)
INT8 value: round(0.05322 / 0.0121) = 4
  ↓ (Dequantization)
FP32 value: 4 * 0.0121 = 0.0484 (precision lost!)
```

**Information is already lost in INT8 quantization. Dequantization doesn't recover it.**

### **Purpose of these modes:**
- **Test whether PyTorch SSM matches CUDA SSM numerically**
- If both are 39%, PyTorch implementation is correct
- If they differ, indicates numerical difference between implementations

---

## Expected Results After Fix

| Mode | Conv1D | SSM Impl | Expected Accuracy | Purpose |
|------|--------|----------|-------------------|---------|
| **Baseline** | INT8 CUDA | CUDA kernel | 39% ✓ | Standard INT8 |
| **Mode 1** | INT8 CUDA | PyTorch | **39%** 🎯 | Verify PyTorch SSM |
| **Mode 2** | INT8 CUDA | PyTorch | **39%** 🎯 | Verify PyTorch SSM |

**Both Mode 1 and Mode 2 have the same expected result (39%).**

They are essentially the same mode - both test FP32 SSM with dequantized INT8 inputs.

---

## Testing Commands

### Run all three modes sequentially:
```bash
bash test_three_modes_debug.sh
```

### Individual modes:

**Baseline**:
```bash
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --pretrained_dir pretrained_models/quamba1/default --testing
```

**Mode 1 (FP32_SSM_INPUT)**:
```bash
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --fp32-ssm-input \
    --pretrained_dir pretrained_models/quamba1/default --testing
```

**Mode 2 (FLOAT_SIM_ASIC_INT8)**:
```bash
python3 main.py quamba-130m-w8a8 --quantize --batch_size 16 \
    --eval_zero_shot --task_list lambada_openai \
    --float-sim-asic-int8 \
    --pretrained_dir pretrained_models/quamba1/default --testing
```

---

## Debug Output (After Fix)

### Mode 1 & Mode 2 Debug Output (Layer 23):

```
[Conv1D Layer 23] Parsed: fp32=True, int8_sim=False
  Conv1D Mode 1 output y dtype: torch.int8
  First 10 values (INT8): [-2, 11, -7, -4, ...]

[Layer 23 Call #1] After Conv1D
  x_for_xproj dtype: torch.int8, first 10: [-2, 11, -7, ...]
  x_for_ssm dtype: torch.float32, first 10 values:
    x_for_ssm[0] = -0.024324646220  # -2 * 0.0121623231
    x_for_ssm[1] = 0.133785560727   # 11 * 0.0121623231

[QSScan Debug] Will use FP32 SSM: True
```

---

## What if we want >39% accuracy?

To achieve accuracy higher than Baseline (>39%), we would need to:

### **Option 1: Use larger scale (pa=1.0 model)**
```python
# Load model with larger output_scale (0.0212 instead of 0.0121)
--pretrained_dir pretrained_models/quamba1/pa-1
```

**Expected:** Potentially >39% if larger scale preserves more information

### **Option 2: Implement true FP32 Conv1D (no quantization)**
```python
# Conv1D computation in full FP32, no quantization
y_fp32 = F.conv1d(x_fp32, weight_fp32, bias_fp32)
y_fp32 = F.silu(y_fp32)  # No quantization here
# Feed to SSM directly
```

**Expected:** >39% as SiLU output has full FP32 precision

**Current implementation does NOT do this** - it uses INT8 CUDA kernel which already quantizes.

---

## Related Files

- `quamba/qConvLayer.py` - Conv1D layer (Mode 1 & 2 now both use INT8 CUDA)
- `quamba/qMambaLayer.py` - Mamba layer with dequantization logic
- `quamba/qSelectiveScan.py` - SSM implementation (QSScan)
- `quamba/selective_scan_SE.py` - PyTorch SSM implementations
- `test_three_modes_debug.sh` - Automated testing script
- `compare_scales.py` - Scale comparison between percentile settings

---

## Session History

**Session 8 - Final Understanding:**

1. ✅ Both Mode 1 and Mode 2 use INT8 CUDA kernel (NOT FP32 simulation)
2. ✅ Only difference from Baseline: SSM receives FP32 instead of INT8
3. ✅ Both modes should achieve 39% (testing PyTorch SSM vs CUDA SSM)
4. ✅ No precision gain expected (information already lost in INT8 quantization)

**Previous misunderstandings (corrected):**
- ❌ Mode 1 uses FP32 Conv1D computation → ✅ Mode 1 uses INT8 CUDA
- ❌ Mode 2 uses FP32 simulation → ✅ Mode 2 uses INT8 CUDA
- ❌ Mode 1 should achieve >39% → ✅ Mode 1 should achieve 39%

**Key learning:**
> "我让你都用int8cuda 你懂么。只有输出会有变化。"

Only the output type changes (INT8 → FP32 dequantization), not the computation.

---

## Next Steps

1. Run `test_three_modes_debug.sh` to verify fix
2. All three modes should show 39% accuracy
3. If Mode 1/2 still show 36%, investigate PyTorch SSM vs CUDA SSM numerical differences
4. For achieving >39%, consider:
   - Testing pa=1.0 model (larger scale)
   - Implementing true FP32 Conv1D (no quantization)

---

## Notes

- The `_forward_fp32_upper_bound` and `_forward_float_sim_int8` functions are now UNUSED
- Both Mode 1 and Mode 2 use the same code path (INT8 CUDA kernel)
- Debug prints verify INT8 output and FP32 dequantization
- All changes preserve Baseline behavior (INT8 path unchanged)
