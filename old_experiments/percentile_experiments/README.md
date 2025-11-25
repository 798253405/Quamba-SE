# Percentile Experiments Archive

This folder contains experimental scripts from previous percentile alpha research.

## Archived Scripts

### 1. compare_percentile_effects.sh
- **Purpose**: Compare effects of different percentile alpha values
- **Date**: Nov 5, 2024
- **Models**: Mamba1-130M, Mamba2-2.7B
- **Experiments**: Default percentile (0.9995) vs pa=1.0 (no clipping)

### 2. run_all_complete_experiments.sh
- **Purpose**: Run 24 complete experiments (default + pa=1.0)
- **Date**: Nov 4, 2024
- **Coverage**: All Quamba1 and Quamba2 models
- **Total Experiments**: 24
- **Estimated Time**: 8-10 hours

### 3. run_correct_experiments.sh
- **Purpose**: Corrected experiments based on author's feedback
- **Date**: Nov 4, 2024
- **Quamba1**: Mamba1 W8A8 (no extra parameters)
- **Quamba2**: Mamba2 W4/W8 (with all parameters)
- **Total Experiments**: 20
- **Estimated Time**: 6-8 hours

### 4. run_mamba2_8b_experiments.sh
- **Purpose**: Mamba2 8B series experiments (experiments 15-20)
- **Date**: Nov 4, 2024
- **Variants**: W4A8, W4A16, W8A8
- **Total Experiments**: 6
- **Estimated Time**: 3-3.5 hours

## Why Archived?

These scripts are from previous percentile quantization research and are **not related** to the current FP32 SSM input research (Session 8). They are preserved here for reference but not actively used.

## Current Active Scripts

See the main project directory for current research scripts:
- `test_three_modes_debug.sh` - Current debugging script
- `test_four_modes.sh` - Four-mode comparison for FP32 SSM research
- `build_cutlass.sh` - Infrastructure (CUTLASS compilation)

---

**Archive Date**: 2025-11-07
**Reason**: Transitioned to FP32 SSM input research (Session 8)
