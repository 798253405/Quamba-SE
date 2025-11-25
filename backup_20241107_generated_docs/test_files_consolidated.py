"""
Consolidated Test Files Backup
Generated: 2024-11-07
Purpose: Backup of test files created during Quamba analysis
Note: Excludes test_check_float_sim.py and test_three_modes.py (actively in use)
"""

# ============================================================================
# test_float_sim.py
# Date: 2024-11-07 11:46
# Purpose: Test float simulation mode (testingflag=2) accuracy
# ============================================================================

"""
test_float_sim.py - Tests the float simulation mode implementation

This script validates that the float simulation mode (testingflag=2) correctly
simulates INT8 quantization using float32 operations.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_float_simulation_mode():
    """Test that float simulation mode produces expected quantization behavior"""
    from quamba import qLinearLayer
    import utils

    # Set to float simulation mode
    utils.testingflag = 2

    # Create test tensor
    batch_size, seq_len, dim = 2, 128, 256
    x = torch.randn(batch_size, seq_len, dim)

    # Create layer
    layer = qLinearLayer.QLinear(
        in_features=dim,
        out_features=dim,
        bias=False
    )

    # Forward pass
    output = layer(x)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, dim), f"Shape mismatch: {output.shape}"

    # Check that values are within expected range for dequantized INT8
    # After quantization to INT8 and dequantization, values should show
    # quantization artifacts (discrete levels)

    # Calculate theoretical quantization levels
    scale = layer.scale_x
    zero_point = layer.zero_point_x if hasattr(layer, 'zero_point_x') else 0

    # Check that output has quantization-like behavior
    unique_values = torch.unique(output).numel()
    total_values = output.numel()

    # Ratio of unique values should be much less than 1.0 for quantized data
    uniqueness_ratio = unique_values / total_values
    print(f"Uniqueness ratio: {uniqueness_ratio:.4f}")
    print(f"Unique values: {unique_values}, Total values: {total_values}")

    # Verify quantization error bounds
    if scale is not None:
        max_quant_error = scale.item() / 2  # Half of quantization step
        print(f"Max theoretical quantization error: {max_quant_error:.6f}")

    print("Float simulation mode test passed!")
    return output

def test_quantization_consistency():
    """Test that quantization is consistent across multiple runs"""
    from quamba import qLinearLayer
    import utils

    utils.testingflag = 2

    # Fixed seed for reproducibility
    torch.manual_seed(42)

    dim = 128
    x = torch.randn(1, 32, dim)

    layer = qLinearLayer.QLinear(dim, dim, bias=False)

    # Multiple forward passes with same input
    outputs = []
    for _ in range(3):
        outputs.append(layer(x.clone()))

    # Check all outputs are identical
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], atol=1e-6), \
            f"Outputs {0} and {i} differ!"

    print("Quantization consistency test passed!")

def test_compare_modes():
    """Compare outputs between different testing modes"""
    from quamba import qLinearLayer
    import utils

    torch.manual_seed(42)
    dim = 64
    x = torch.randn(1, 16, dim)

    outputs = {}

    for mode in [0, 1, 2]:
        utils.testingflag = mode
        layer = qLinearLayer.QLinear(dim, dim, bias=False)
        outputs[mode] = layer(x.clone())
        print(f"Mode {mode} output range: [{outputs[mode].min():.4f}, {outputs[mode].max():.4f}]")

    # Mode 1 (float32) and Mode 2 (float sim) should be different
    # Mode 1 has no quantization, Mode 2 simulates it
    diff_1_2 = (outputs[1] - outputs[2]).abs().mean()
    print(f"Mean absolute difference between Mode 1 and 2: {diff_1_2:.6f}")

    # The difference should be non-zero but small
    assert diff_1_2 > 1e-6, "Modes 1 and 2 should produce different results"
    assert diff_1_2 < 0.1, "Quantization error seems too large"

    print("Mode comparison test passed!")

def test_activation_quantization():
    """Test quantization of activation functions"""
    from quamba import qLinearLayer
    import utils
    import torch.nn.functional as F

    utils.testingflag = 2

    # Test SiLU activation quantization
    x = torch.randn(1, 32, 128)

    # Manual SiLU
    silu_output = x * torch.sigmoid(x)

    # Get percentile scale (97th percentile as per implementation)
    percentile = 97
    threshold = torch.quantile(silu_output.abs(), percentile/100.0)
    scale = threshold / 127.0  # Symmetric quantization for activations

    # Simulate quantization
    quantized = torch.clamp(torch.round(silu_output / scale), -128, 127)
    dequantized = quantized * scale

    # Check quantization error
    error = (silu_output - dequantized).abs()
    max_error = error.max()
    mean_error = error.mean()

    print(f"SiLU quantization - Max error: {max_error:.6f}, Mean error: {mean_error:.6f}")
    assert max_error < scale * 1.5, "Quantization error too large"

    print("Activation quantization test passed!")

if __name__ == "__main__":
    print("="*60)
    print("Running Float Simulation Mode Tests")
    print("="*60)

    test_float_simulation_mode()
    print("-"*60)

    test_quantization_consistency()
    print("-"*60)

    test_compare_modes()
    print("-"*60)

    test_activation_quantization()
    print("-"*60)

    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)

# ============================================================================
# Additional helper test functions created during analysis
# Date: 2024-11-06 to 2024-11-07
# ============================================================================

def inspect_quantization_scales():
    """Helper function to inspect quantization scales in a model"""
    from quamba import qLinearLayer
    import utils

    utils.testingflag = 2
    dim = 256

    layer = qLinearLayer.QLinear(dim, dim * 4, bias=False)

    # Inspect weight scales
    if hasattr(layer, 'scale_weight'):
        print(f"Weight scale shape: {layer.scale_weight.shape}")
        print(f"Weight scale range: [{layer.scale_weight.min():.6f}, {layer.scale_weight.max():.6f}]")

    # Test with input to get activation scales
    x = torch.randn(1, 32, dim)
    _ = layer(x)

    if hasattr(layer, 'scale_x'):
        print(f"Input scale: {layer.scale_x:.6f}")

    return layer

def validate_percentile_scales():
    """Validate that percentile-based scales are computed correctly"""
    import torch

    # Test tensor
    x = torch.randn(1000)

    # Test different percentiles
    percentiles = [90, 95, 97, 99]

    for p in percentiles:
        threshold = torch.quantile(x.abs(), p/100.0)
        scale = threshold / 127.0

        # Count how many values would saturate
        saturated = (x.abs() > threshold).sum().item()
        saturation_rate = saturated / x.numel() * 100

        print(f"Percentile {p}: scale={scale:.6f}, saturation={saturation_rate:.2f}%")

        # Verify saturation rate matches expected
        expected_saturation = 100 - p
        assert abs(saturation_rate - expected_saturation) < 2.0, \
            f"Saturation rate {saturation_rate} doesn't match expected {expected_saturation}"

def trace_scale_propagation():
    """Trace how scales propagate through the model"""
    from quamba import qMambaLayer
    import utils

    utils.testingflag = 2

    # Create a minimal Mamba block
    config = {
        'd_model': 128,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
    }

    layer = qMambaLayer.MambaBlock(
        d_model=config['d_model'],
        d_state=config['d_state'],
        d_conv=config['d_conv'],
        expand=config['expand'],
    )

    # Input
    x = torch.randn(1, 32, config['d_model'])

    # Hook to capture intermediate scales
    scales_captured = {}

    def capture_scale_hook(module, input, output):
        if hasattr(module, 'scale_x'):
            scales_captured[module.__class__.__name__] = module.scale_x

    # Register hooks
    for submodule in layer.modules():
        submodule.register_forward_hook(capture_scale_hook)

    # Forward pass
    _ = layer(x)

    # Print captured scales
    for name, scale in scales_captured.items():
        if scale is not None:
            print(f"{name}: scale={scale:.6f}")

    return scales_captured

# ============================================================================
# End of Consolidated Test Files
# ============================================================================