# Quamba Analysis Documentation Backup
**Generated: 2024-11-07**
**Purpose: Consolidated backup of all analysis and documentation generated during Quamba project investigation**

---

## Table of Contents
1. [Float Simulation Mode Analysis](#float-simulation-mode-analysis)
2. [Percentile Scale Usage Analysis](#percentile-scale-usage-analysis)
3. [Conv1D Connection Analysis](#conv1d-connection-analysis)
4. [Layer Structure Explanation](#layer-structure-explanation)
5. [Mamba2 Group Quantization Analysis](#mamba2-group-quantization-analysis)
6. [Three Modes Implementation](#three-modes-implementation)
7. [Test Files](#test-files)
8. [Python Analysis Scripts](#python-analysis-scripts)

---

# Float Simulation Mode Analysis
**Date: 2024-11-07 11:40**
**Files: FLOAT_SIM_README.md, FLOAT_SIM_CHANGES_SUMMARY.md, yzCheckFloatSim_FORMAT.md**

## Overview
Analysis of the float simulation mode (`testingflag=2`) in Quamba, which simulates quantization using float32 operations.

### Key Findings
- Float simulation mode uses fake quantization with float32 operations
- Implements asymmetric INT8 quantization simulation
- Three testing modes: 0 (normal INT8), 1 (force float32), 2 (float simulation)
- Modified files: qLinearLayer.py, qConvLayer.py, qSelectiveScan.py, utils.py

### Implementation Details
```python
# Float simulation formula
if testingflag == 2:
    x_quantized = torch.clamp(torch.round(x / scale + zero_point), 0, 255)
    x_dequantized = scale * (x_quantized - zero_point)
```

### Test Results
- Created test_float_sim.py for validating float simulation
- Verified quantization error patterns match INT8 behavior
- Confirmed scale and zero point calculations are correct

---

# Percentile Scale Usage Analysis
**Date: 2024-11-06 22:36 - 2024-11-07 10:52**
**Files: PERCENTILE_SCALE_USAGE_MAP.md, SILU_PERCENTILE_SCALE_ANALYSIS.md, PERCENTILE_ANALYSIS_CONCLUSION.md**

## Overview
Comprehensive analysis of how percentile-based scales are used throughout Quamba's quantization system.

### Key Components Using Percentile Scales

1. **qLinearLayer.py**
   - SiLU activation: Uses silu_percentile_mode (97th percentile)
   - GELU activation: Computed similarly to SiLU
   - Output projection: Uses gelu_input and gelu_output percentiles

2. **qConvLayer.py**
   - Conv1d operations: Uses conv_percentile_mode
   - Pre-computed scales for efficiency

3. **qSelectiveScan.py**
   - SSM operations: Uses ssm_percentile_mode
   - Critical for selective scan accuracy

### Scale Computation Method
```python
def getPercentileBasedScale(tensor, percentile, symmetric=False):
    if symmetric:
        max_val = torch.quantile(tensor.abs(), percentile/100)
        scale = max_val / 127
    else:
        min_val = torch.quantile(tensor, 1-percentile/100)
        max_val = torch.quantile(tensor, percentile/100)
        scale = (max_val - min_val) / 255
    return scale
```

### Analysis Scripts Created
- analyze_scales.py: Visualizes scale distributions
- trace_percentile_usage.py: Traces percentile scale usage
- inspect_tensor_scales.py: Inspects specific tensor scales

---

# Conv1D Connection Analysis
**Date: 2024-11-07 11:01**
**File: CONV1D_CONNECTION_ANALYSIS.md**

## Overview
Analysis of the Conv1D layer's role in Quamba/Mamba2 architecture and its quantization strategy.

### Key Findings
1. **Conv1D in Mamba2 Architecture**
   - Acts as local feature extractor before SSM
   - Kernel size typically 4, capturing short-term dependencies
   - Processes input sequence with causal padding

2. **Quantization Strategy**
   - Uses percentile-based scaling for robustness
   - Separate scales for weights and activations
   - Critical for maintaining temporal coherence

3. **Integration with SSM**
   - Conv1D output feeds into selective scan mechanism
   - Provides local context for state updates
   - Quantization errors can propagate through SSM

---

# Layer Structure Explanation
**Date: 2024-11-06 22:45**
**File: LAYER_STRUCTURE_EXPLANATION.md**

## Overview
Detailed explanation of Quamba's layer structure and quantization granularity.

### MambaBlock Structure
```
Input (x)
    ↓
[Linear Projection] → in_proj
    ↓
Split into (x, z)
    ↓
x → [Conv1D] → [SiLU] → [SSM] →
                                 × → [Out Projection]
z → [SiLU] ────────────────────→
```

### Quantization Points
1. **Input Projection**: Group-wise quantization
2. **Conv1D**: Percentile-based scales
3. **Activation Functions**: Pre-computed scales
4. **SSM Operations**: Dynamic quantization
5. **Output Projection**: Group-wise with residual

---

# Mamba2 Group Quantization Analysis
**Date: 2024-11-06 22:49**
**Files: MAMBA2_GROUP_QUANTIZATION_ANALYSIS.md, analyze_mamba2_scales.py**

## Overview
Analysis of Mamba2-specific quantization strategies and group-wise approaches.

### Group Quantization Strategy
- **Group Size**: 128 channels per group
- **Rationale**: Balance between accuracy and efficiency
- **Implementation**: Separate scales per group

### Benefits
1. Better captures channel-wise variations
2. Reduces quantization error in SSM operations
3. Maintains computational efficiency

### Analysis Results
- Created visualization of scale distributions across groups
- Identified optimal group sizes for different layers
- Validated group boundaries align with feature boundaries

---

# Three Modes Implementation
**Date: 2024-11-07 12:01**
**Files: THREE_MODES_README.md, test_three_modes.py**

## Overview
Documentation of the three testing modes in Quamba for debugging and validation.

### Testing Modes
1. **Mode 0 (INT8)**: Normal quantized inference
2. **Mode 1 (Float32)**: Force all operations to float32
3. **Mode 2 (Float Sim)**: Simulate quantization with float32

### Usage
```python
# Set mode via constructor
model = QuambaModel(..., testingflag=2)

# Or set globally
import utils
utils.testingflag = 2
```

### Validation Script
- test_three_modes.py: Compares outputs across all three modes
- Validates quantization error bounds
- Ensures mode consistency

---

# Test Files
**Date: 2024-11-07**
**Note: Latest 2 test files (test_check_float_sim.py, test_three_modes.py) are NOT included as they are actively being used**

## Test Files Created

### test_float_sim.py (2024-11-07 11:46)
```python
# Tests float simulation mode accuracy
# Validates quantization/dequantization cycle
# Compares with true INT8 operations
```

### Earlier Test Files
- Various unit tests for individual components
- Integration tests for full model inference
- Performance benchmarks

---

# Python Analysis Scripts
**Date: 2024-11-06 to 2024-11-07**

## Scripts Created

### analyze_scales.py (2024-11-06 22:38)
- Analyzes scale distributions across layers
- Generates histograms and statistics
- Identifies outliers and patterns

### trace_percentile_usage.py (2024-11-07 10:58)
- Traces percentile scale usage through model
- Identifies which operations use which percentiles
- Maps scale propagation paths

### analyze_mamba2_scales.py (2024-11-06 22:54)
- Specific analysis for Mamba2 architecture
- Group-wise scale analysis
- Channel correlation studies

### inspect_tensor_scales.py (2024-11-06 22:36)
- Low-level tensor scale inspection
- Debug tool for specific tensors
- Scale computation verification

---

## Summary of Current State

### What Has Been Accomplished
1. **Float Simulation Mode**: Successfully implemented and tested fake quantization mode
2. **Percentile Analysis**: Comprehensive understanding of percentile-based scaling
3. **Architecture Understanding**: Deep dive into Conv1D and SSM connections
4. **Testing Framework**: Three-mode testing system for validation
5. **Analysis Tools**: Suite of Python scripts for scale analysis

### Key Insights
1. Percentile-based scaling (97th percentile) is crucial for activation quantization
2. Group-wise quantization with 128-channel groups balances accuracy and efficiency
3. Float simulation mode accurately reproduces INT8 quantization behavior
4. Conv1D plays critical role in temporal feature extraction before SSM

### Current Work in Progress
- Testing testingflag implementation (DO NOT MODIFY)
- Latest test scripts are being actively used

### Important Files to Preserve
1. SESSION_HISTORY.md - Complete session history
2. test_check_float_sim.py - Active testing (DO NOT MODIFY)
3. test_three_modes.py - Active testing (DO NOT MODIFY)
4. Modified core files with testingflag support:
   - qLinearLayer.py
   - qConvLayer.py
   - qSelectiveScan.py
   - utils.py

---

**End of Consolidated Documentation Backup**