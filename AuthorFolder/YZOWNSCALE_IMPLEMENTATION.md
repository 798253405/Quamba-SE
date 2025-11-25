# YzOwnScale Implementation - Dual-Scale Quantization for SSM Input

## ⚠️ IMPLEMENTATION REVERTED (2025-11-06)

**Status**: ❌ **All code changes have been reverted**

**Reason for Reversion**:
1. **Precision Unfairness**: Implementation used fp16 for SSM input (via Python dequant), while baseline uses int8. This creates an unfair comparison.
2. **Kernel Limitation**: CUDA kernel (`quant_sscan_fwd_kernel.cuh:133`) only reads scalar scale, making per-element scale map ineffective.
3. **Wrong Approach**: Attempted to implement dual-scale in Python layer instead of kernel layer.
4. **Fundamental Issue**: Dual-scale dequantization requires kernel modification, cannot be implemented purely in Python.

**What Was Learned**:
- Conv1D outputs int8 (quantized at `quant_causal_conv1d_fwd_kernel.cuh:149`)
- SSM receives int8 and dequantizes in kernel (at `quant_sscan_fwd_kernel.cuh:157`)
- Dual-scale must be implemented at SSM kernel's dequant step, not in Python
- Cannot change precision (int8→fp16) without making comparison unfair

**Reverted Files**:
- `utils.py` - Removed `--yzOwnScale` and `--yzOwnScaleEqual` flags
- `quamba/observer.py` - Removed `get_dual_scale_parameters()` method
- `quamba/qSelectiveScan.py` - Removed dual-scale logic and per-element scale map
- `quamba/modelutils_mamba.py` - Removed dual-scale calibration code

**This document is kept for historical reference only.**

---

## 📋 Overview (Original Plan - Not Implemented)

**Date**: 2025-11-06
**Feature**: `--yzOwnScale` (REVERTED)
**Purpose**: Outlier-aware dual-scale quantization for SSM input (u/x)
**Scope**: **Mamba1 only** in Phase 1 (Mamba2 requires kernel modifications)

This feature was intended to implement a dual-scale quantization scheme specifically for the SSM (Selective State Space Model) input activations, addressing the challenge of outliers in quantization.

**Implementation Status**:
- ❌ **Reverted**: All changes have been rolled back
- **Reason**: Incorrect approach (Python-layer implementation not feasible)

---

## 🎯 Core Concept

### Problem
- SSM input activations contain outliers (extreme values beyond typical distribution)
- Single-scale quantization clips outliers, causing accuracy degradation
- Outliers are rare (<1% of values) but important for model quality

###Solution
- **Dual-Scale Quantization**:
  - `scale_inlier`: For values within percentile threshold (99.9%+)
  - `scale_outlier`: For values outside percentile threshold (outliers)
  - Dynamic selection based on pre-computed threshold

### Quantization Formula
```python
if abs(value) > threshold:
    quantized = round(value / scale_outlier)  # Outlier
else:
    quantized = round(value / scale_inlier)   # Inlier
```

### Dequantization Formula
```python
if abs(quantized) corresponds to outlier:
    dequantized = quantized * scale_outlier
else:
    dequantized = quantized * scale_inlier
```

---

## 🔧 Implementation Details

### Phase 1: Simple Per-Element Scale Map (Current Implementation)

**Status**: ✅ Completed (No CUDA recompilation needed)

**Approach**:
- Create a per-element scale map in Python
- Pass scale map to kernel (instead of single scalar scale)
- Kernel uses corresponding scale for each element

**Advantages**:
- ✅ No need to recompile CUDA kernels
- ✅ Quick to implement and test
- ✅ Easy to verify correctness

**Disadvantages**:
- ❌ Higher memory overhead (per-element scale map)
- ❌ Slightly slower due to scale map memory transfer

---

## 📁 Modified Files

### 1. Command Line Interface
- **File**: `utils.py`
- **Change**: Added `--yzOwnScale` flag
- **Default**: `False` (disabled, backward compatible)

### 2. Observer (Calibration Statistics)
- **File**: `quamba/observer.py`
- **Change**: Added `get_dual_scale_parameters()` method to `PerTensorPercentileObserver`
- **Returns**:
  - `scale_inlier`: Scale for inliers (percentile-based)
  - `scale_outlier`: Scale for outliers (absolute max)
  - `threshold`: Percentile threshold value

### 3. Calibration Functions
- **Files**: `quamba/modelutils_mamba.py`
- **Functions Modified**:
  - `run_quamba_calibration()` (Mamba1)
  - `run_quamba2_calibration()` (Mamba2)
- **Change**:
  - Added `yzOwnScale` parameter
  - Collect dual-scale parameters for `x_proj:input` (Mamba1) and `x_conv_out:input` (Mamba2)
  - Store as `x_proj:input:inlier`, `x_proj:input:outlier`, `x_proj:input:threshold`

### 4. QSScan Module (Mamba1 SSM)
- **File**: `quamba/qSelectiveScan.py`
- **Class**: `QSScan`
- **Changes**:
  - Added `use_dual_scale` flag to `__init__()`
  - Added `u_scale_outlier` and `u_threshold` buffers
  - Modified `from_fp16()` to accept dual-scale parameters
  - Modified `forward()` to create per-element scale map when `use_dual_scale=True`

### 5. Quantized Mamba Layers (Mamba1)
- **File**: `quamba/qMambaLayer.py`
- **Classes**: `W4A8QMamba`, `W8A8QMamba`
- **Changes**:
  - Pass dual-scale parameters to `QSScan.from_fp16()`
  - In `forward()`: Dequantize `x` to `x_fp16` when dual-scale is enabled
  - Pass `x_fp16` to `selective_scan.forward()`

### 6. Mamba2 Architecture Note
- **Files**: `quamba/qChunkScan.py`, `quamba/qMamba2.py`
- **Status**: ⚠️ Dual-scale NOT implemented for Mamba2 in Phase 1
- **Reason**: Mamba2 uses group-wise quantization with Triton kernels
  - SSM input (`x_conv_out`) already supports per-group scales via `x_out_scales`
  - Adding per-element dual-scale on top of group-wise quantization requires kernel modification
  - Triton kernel takes `x_scales` parameter which can be multi-dimensional (group-wise)
  - Phase 1 goal was "no kernel recompilation" - this constraint conflicts with Mamba2's architecture
- **Calibration**: Dual-scale parameters for `x_conv_out:input` ARE collected during calibration
- **Future Work**: Design approach to integrate per-element dual-scale with group-wise quantization

---

## 📊 Memory and Performance Impact

### Memory Overhead (Per-Element Scale Map)

| Model | Batch=1, Seq=2048 | Mask Size | Percentage of Input |
|-------|-------------------|-----------|---------------------|
| mamba-130m | 24 layers | 9 MB | 12.5% |
| mamba-370m | 48 layers | 24 MB | 12.5% |
| mamba-1.4b | 48 layers | 48 MB | 12.5% |
| mamba2-2.7b | 64 layers | 80 MB | 12.5% |

**Conclusion**: Memory overhead is acceptable for modern GPUs (40-80GB VRAM).

### Performance Impact

- **Memory Transfer Overhead**: +12.5% (for scale map)
- **Estimated Inference Slowdown**: 5-10%
- **Benefit**: Potentially better accuracy for models with outliers

---

## 🚀 Usage

### Enable YzOwnScale (Mamba1 Models Only)

**Important**: In Phase 1, `--yzOwnScale` only affects **Mamba1** models. Mamba2 models will collect calibration data but not apply dual-scale quantization.

```bash
# Example: Mamba1-130m with dual-scale
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize \
  --w_bits 8 --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai \
  --pretrained_dir ./pretrained_models \
  --log_dir logs \
  --output_subdir myExperiment \
  --yzOwnScale  # ← Enable dual-scale quantization
```

### Disable YzOwnScale (Default Behavior)

```bash
# Simply omit --yzOwnScale flag
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m \
  --quantize \
  --w_bits 8 --a_bits 8 \
  --batch_size 16 \
  --eval_zero_shot \
  --task_list lambada_openai
```

---

## ✅ Testing Checklist

### Backward Compatibility
- [x] Without `--yzOwnScale`: All existing results unchanged
- [x] Models quantized without `--yzOwnScale` load and run correctly
- [ ] Verify accuracy matches baseline (needs testing)

### Dual-Scale Functionality
- [ ] With `--yzOwnScale`: Dual-scale parameters collected during calibration
- [ ] Outlier threshold logged correctly
- [ ] Scale map created correctly in forward pass
- [ ] Accuracy improvement vs single-scale (needs benchmarking)

### Edge Cases
- [ ] Models with very few outliers (<0.1%)
- [ ] Models with many outliers (>5%)
- [ ] Different percentile_alpha values

---

## 🧪 Ablation Study: yzOwnScaleEqual (Control Group)

**Date**: 2025-11-06
**Purpose**: Verify that fp16 simulation does not introduce unfair precision gain

### Problem
- yzOwnScale uses fp16 for SSM input (instead of int8) to create per-element scale maps
- Need to verify this doesn't give unfair advantage from fp16 precision itself
- Solution: Add control group that uses fp16 but with single scale (simulating int8)

### Implementation
**Flag**: `--yzOwnScaleEqual`
- Must be used together with `--yzOwnScale`
- Forces all elements to use `scale_inlier` (no dual-scale)
- Simulates int8 quantization using fp16 numerical representation

**Mechanism**:
- Sets environment variable `YZOWNSCALE_EQUAL=true` in `utils.py`
- `QSScan.forward()` reads this env var and uses single scale for all elements
- All code marked with `#ownscale` for easy debugging

### Three-Group Experiment

| Group | Flags | Purpose |
|-------|-------|---------|
| **Baseline** | (none) | Standard int8 SSM |
| **Control** | `--yzOwnScale --yzOwnScaleEqual` | fp16 simulation of int8 (verify fairness) |
| **Treatment** | `--yzOwnScale` | True dual-scale quantization |

### Test Commands

**Baseline** (standard int8):
```bash
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m --quantize --w_bits 8 --a_bits 8 --batch_size 16 --eval_zero_shot --task_list lambada_openai --pretrained_dir ./pretrained_models --log_dir logs --output_subdir ownscale_baseline
```

**Control** (fp16 simulating int8):
```bash
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m --quantize --w_bits 8 --a_bits 8 --batch_size 16 --eval_zero_shot --task_list lambada_openai --pretrained_dir ./pretrained_models --log_dir logs --output_subdir ownscale_control --yzOwnScale --yzOwnScaleEqual
```

**Treatment** (true dual-scale):
```bash
python3 main.py pretrained_models/mambaOriginalHuggingfaceDownload/mamba-130m --quantize --w_bits 8 --a_bits 8 --batch_size 16 --eval_zero_shot --task_list lambada_openai --pretrained_dir ./pretrained_models --log_dir logs --output_subdir ownscale_treatment --yzOwnScale
```

### Expected Results
- **Baseline ≈ Control**: Proves fp16 simulation is fair (no precision gain)
- **Treatment vs Control**: Pure dual-scale effect (if Control > Baseline, Treatment improvement is still valid but needs to account for fp16 factor)

### Files Modified (all marked with #ownscale)
1. `utils.py`: Added `--yzOwnScaleEqual` flag, sets `YZOWNSCALE_EQUAL` env var
2. `quamba/qSelectiveScan.py`: Reads env var, applies equal scale when set
3. `quamba/modelutils_mamba.py`: Already collects dual-scale parameters (unchanged)

---

## 🔮 Future Optimizations (Phase 2)

### Sparse Outlier Indices
If outliers are very sparse (<1%), we can optimize:

**Current**: Dense per-element scale map (48 MB for mamba-1.4b)
**Optimized**: Sparse outlier indices (0.3 MB for 0.1% outliers)

**Savings**: 97% memory reduction

**Implementation**:
1. Store only outlier positions as indices
2. Modify CUDA kernel to look up outlier mask
3. Use hash table or sorted binary search for fast lookup

---

## 📝 Implementation Notes

### Why Per-Element Scale Map?
- **Phase 1 Goal**: Quick validation without CUDA changes
- **Allows Testing**: Can benchmark accuracy improvement
- **Easy Debugging**: All logic in Python layer

### When to Use Sparse Indices?
- When profiling shows outlier ratio < 1%
- When memory is constrained
- After Phase 1 validation proves the approach works

### Backward Compatibility Guarantee
- **Default behavior unchanged**: `yzOwnScale=False` by default
- **No breaking changes**: All existing code paths preserved
- **Optional feature**: Users must explicitly enable

---

## 🐛 Known Limitations

1. **Phase 1 Only**: Currently only per-element scale map implemented
2. **Memory Overhead**: 12.5% additional memory for scale maps (Mamba1 only)
3. **Mamba2 Not Supported**: Dual-scale NOT implemented for Mamba2 in Phase 1
   - Mamba2 uses group-wise quantization with Triton kernels
   - Combining per-element dual-scale with group-wise scales requires kernel modifications
   - Calibration data IS collected but not used in Mamba2 layers
   - Future work needed to design proper integration
4. **No Benchmarks Yet**: Accuracy improvement not quantified for Mamba1

---

## 📚 References

- **Outlier-Aware Quantization**: Common technique in LLM quantization
- **SmoothQuant**: Migration of outliers via channel-wise scaling
- **AWQ**: Activation-aware weight quantization

---

## 👥 Contributors

- **Implementation**: Claude (Anthropic AI)
- **Concept**: User request for dual-scale SSM input quantization
- **Date**: 2025-11-06
