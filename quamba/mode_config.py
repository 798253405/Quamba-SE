"""
Quamba Mode Configuration System

Provides a unified interface for setting quantization modes using a single
QUAMBA_MODE environment variable instead of multiple mode flags.

Usage:
    from quamba.mode_config import setup_quamba_mode

    # In main.py or test script
    setup_quamba_mode('2-4')  # Set Mode 2-4

    # Or use environment variable
    export QUAMBA_MODE=2-4
    setup_quamba_mode()  # Auto-detect from env

Available Modes:
    0      : Baseline INT8 CUDA
    2-0    : CUDA INT8 + Requantization
    2-1    : PyTorch INT8 Direct
    2-2    : FP32 PyTorch (INT8 Grid)
    2-3    : TRUE FP32 Conv + INT8 SSM
    2-4    : TRUE FP32 Conv + FP32 SSM (Mode 2-2 same)
    3      : Hybrid Precision (FP32 Conv/SSM + INT8 Linear)
    5      : Dual-Path Mode 5 (FP32 Conv/SSM + Mode 0 comparison)
    6      : Dual-Path Mode 6 (FP32 output feeds both paths)
"""

import os
from typing import Optional


# Mode configuration mapping
MODE_CONFIG = {
    '0': {
        'name': 'Baseline INT8 CUDA',
        'env': {},
        'description': 'Original INT8 CUDA kernel baseline'
    },
    '2-0': {
        'name': 'CUDA INT8 + Requantization',
        'env': {
            'FLOAT_SIM_ASIC_INT8': 'true',
            'SSM_USE_CUDA_FOR_FP32': 'true'
        },
        'description': 'FP32 (INT8 grid) → requantize → CUDA INT8 SSM'
    },
    '2-1': {
        'name': 'PyTorch INT8 Direct',
        'env': {
            'FLOAT_SIM_ASIC_INT8': 'true',
            'SSM_USE_PYTORCH_INT8': 'true'
        },
        'description': 'FP32 (INT8 grid) → requantize → PyTorch INT8 SSM'
    },
    '2-2': {
        'name': 'FP32 PyTorch (INT8 Grid)',
        'env': {
            'FLOAT_SIM_ASIC_INT8': 'true'
        },
        'description': 'Conv1D: INT8 → FP32 (INT8 grid), SSM: Mode 2-2 FP32'
    },
    '2-3': {
        'name': 'TRUE FP32 Conv + INT8 SSM',
        'env': {
            'FLOAT_SIM_ASIC_INT8': 'true',
            'CONV1D_MODE23_FP32': 'true'
        },
        'description': 'Conv1D: TRUE FP32, SSM: PyTorch INT8 (with requantization)'
    },
    '2-4': {
        'name': 'TRUE FP32 Conv + FP32 SSM',
        'env': {
            'FLOAT_SIM_ASIC_INT8': 'true',
            'CONV1D_MODE24_FP32': 'true'
        },
        'description': 'Conv1D: TRUE FP32, SSM: Mode 2-2 same FP32'
    },
    '3': {
        'name': 'Hybrid Precision',
        'env': {
            'CONV1D_MODE3_FP32': 'true'
        },
        'description': 'FP32/FP16 input + FP32 Conv/SSM + INT8 Linear'
    },
    '4': {
        'name': 'Selective Grid FP32',
        'env': {
            'CONV1D_MODE24_FP32': 'true',
            'CONV1D_MODE4_SELECTIVE_GRID': 'true'
        },
        'description': 'Conv1D: Mixed (INT8 grid for normal, FP32 for overflow), SSM: FP32'
    },
    '5': {
        'name': 'Dual-Path Mode 5',
        'env': {},
        'description': 'Conv1D: FP32 output, SSM: FP32 input, dual-path comparison with Mode 0'
    },
    '5-0': {
        'name': 'Mode 5-0 Virtual INT8 Eval',
        'env': {},
        'description': 'Virtual INT8: Conv1D FP32→Virtual INT8 grid, x_proj INT8, SSM via fwd_mode5 (should match Mode 0)'
    },
    '5-1': {
        'name': 'Mode 5-1 FP32 Eval',
        'env': {},
        'description': 'Single path: FP32 no quantization, for eval without stats printing'
    },
    '5-2': {
        'name': 'Mode 5-2 VirtualQuant+Outlier Eval',
        'env': {},
        'description': 'Single path: Virtual quantization + Outlier protection, for eval without stats printing'
    },
    '5-3': {
        'name': 'Mode 5-3 Dual-Scale INT8 Eval',
        'env': {},
        'description': 'Single path: Dual-scale INT8 (normal: 1x scale, outlier: 2x scale), for eval without stats printing'
    },
    '5-4': {
        'name': 'Mode 5-4 QuarterScale 4x Precision Eval',
        'env': {},
        'description': 'Single path: QuarterScale (small |q|<32 uses scale/4 for 4x precision), x_proj INT8 (same as 5-0)'
    },
    '6': {
        'name': 'Dual-Path Mode 6',
        'env': {},
        'description': 'Conv1D: FP32 output (feeds both paths), SSM: FP32 input, dual-path comparison'
    },
    '6-0': {
        'name': 'Mode 6-0 INT8+FP32_x_proj Eval',
        'env': {},
        'description': 'Single path: Conv1D INT8, x_proj FP32 (dequant), SSM INT8 - x_proj uses F.linear'
    },
    '6-1': {
        'name': 'Mode 6-1 FP32+FP32_x_proj Eval',
        'env': {},
        'description': 'Single path: Full FP32, x_proj FP32 (dequant) - SSM and x_proj use identical FP32 values'
    },
    '6-2': {
        'name': 'Mode 6-2 VirtualQuant+FP32_x_proj Eval',
        'env': {},
        'description': 'Single path: VirtualQuant+Outlier, x_proj FP32 (dequant) - SSM and x_proj use identical mixed values'
    },
    '6-3': {
        'name': 'Mode 6-3 HalfScale 2x Precision Eval',
        'env': {},
        'description': 'Single path: HalfScale for small values (|q|<64 uses scale/2 for 2x precision), x_proj FP32'
    },
    '6-4': {
        'name': 'Mode 6-4 CalibratedDualScale+FP32_x_proj Eval',
        'env': {},
        'description': 'Single path: Calibrated DualScale INT8 (outlier uses α=1.0 scale), x_proj FP32 (dequant)'
    }
}


# All possible mode environment variables
ALL_MODE_ENV_VARS = [
    'FLOAT_SIM_ASIC_INT8',
    'SSM_USE_CUDA_FOR_FP32',
    'SSM_USE_PYTORCH_INT8',
    'CONV1D_MODE23_FP32',
    'CONV1D_MODE24_FP32',
    'CONV1D_MODE3_FP32',
    'CONV1D_MODE4_SELECTIVE_GRID'
]


def clear_all_mode_env_vars():
    """Clear all mode-related environment variables"""
    for var in ALL_MODE_ENV_VARS:
        os.environ.pop(var, None)


def setup_quamba_mode(mode: Optional[str] = None, verbose: bool = True):
    """
    Setup Quamba quantization mode using a single mode identifier.

    Args:
        mode: Mode identifier ('0', '2-0', '2-1', '2-2', '2-3', '2-4', '3')
              If None, reads from QUAMBA_MODE environment variable
        verbose: Print mode configuration info

    Raises:
        ValueError: If mode is invalid or not specified

    Examples:
        >>> setup_quamba_mode('2-4')  # Set Mode 2-4
        >>> setup_quamba_mode()       # Auto-detect from QUAMBA_MODE env var
    """
    # Auto-detect from environment if not specified
    if mode is None:
        mode = os.environ.get('QUAMBA_MODE', '').strip()
        if not mode:
            raise ValueError(
                "Mode not specified. Either pass mode argument or set QUAMBA_MODE environment variable.\n"
                f"Available modes: {', '.join(MODE_CONFIG.keys())}"
            )

    # Validate mode
    if mode not in MODE_CONFIG:
        raise ValueError(
            f"Invalid mode '{mode}'. Available modes: {', '.join(MODE_CONFIG.keys())}\n"
            "Use setup_quamba_mode('2-4') or set QUAMBA_MODE=2-4"
        )

    # Clear all existing mode variables
    clear_all_mode_env_vars()

    # Set new mode environment variables
    config = MODE_CONFIG[mode]
    for key, value in config['env'].items():
        os.environ[key] = value

    # Print configuration if verbose
    if verbose:
        print(f"\n{'='*80}")
        print(f"Quamba Mode Configuration")
        print(f"{'='*80}")
        print(f"  Mode:        {mode}")
        print(f"  Name:        {config['name']}")
        print(f"  Description: {config['description']}")
        if config['env']:
            print(f"  Environment variables set:")
            for key, value in config['env'].items():
                print(f"    {key} = {value}")
        else:
            print(f"  Environment variables: (none, using baseline)")

        # Print kernel function information
        print(f"\n  Kernel Functions:")
        conv1d_kernel = _get_conv1d_kernel_name(mode)
        ssm_kernel = _get_ssm_kernel_name(mode)
        print(f"    Conv1D: {conv1d_kernel}")
        print(f"    SSM:    {ssm_kernel}")

        print(f"{'='*80}\n")


def _get_conv1d_kernel_name(mode: str) -> str:
    """Get the Conv1D kernel function name for a given mode"""
    config = MODE_CONFIG.get(mode, {})
    env = config.get('env', {})

    if mode == '5':
        return "quant_causal_conv1d_cuda.fwd_mode5 (CUDA FP32 output, dual-path)"
    elif mode == '5-0':
        return "quant_causal_conv1d_cuda.fwd_mode5 → Virtual INT8 (CUDA FP32 → INT8 grid)"
    elif mode == '5-1':
        return "quant_causal_conv1d_cuda.fwd_mode5 (CUDA FP32 output, single-path for eval)"
    elif mode == '5-2':
        return "quant_causal_conv1d_cuda.fwd_mode5 + VirtualQuant (CUDA FP32 + INT8 grid + outlier)"
    elif mode == '5-3':
        return "quant_causal_conv1d_cuda.fwd_mode5 + DualScale (CUDA FP32 + dual INT8 grid)"
    elif mode == '5-4':
        return "quant_causal_conv1d_cuda.fwd_mode5 + QuarterScale (CUDA FP32 + scale/4 for small values)"
    elif mode == '6-3':
        return "quant_causal_conv1d_cuda.fwd_mode5 + HalfScale (CUDA FP32 + scale/2 for small values)"
    elif mode == '6':
        return "quant_causal_conv1d_cuda.fwd_mode6 (CUDA FP32 output, dual-path)"
    elif mode == '6-4':
        return "quant_causal_conv1d_cuda.fwd_mode5 + CalibratedDualScale (CUDA FP32 + α=1.0 outlier scale)"
    elif env.get('CONV1D_MODE4_SELECTIVE_GRID') == 'true':
        return "PyTorch F.conv1d (Selective Grid: FP32 for overflow, INT8 grid for normal)"
    elif env.get('CONV1D_MODE24_FP32') == 'true':
        return "PyTorch F.conv1d (FP32)"
    elif env.get('CONV1D_MODE23_FP32') == 'true':
        return "PyTorch F.conv1d (FP32)"
    elif env.get('CONV1D_MODE3_FP32') == 'true':
        return "PyTorch F.conv1d (FP32)"
    elif env.get('FLOAT_SIM_ASIC_INT8') == 'true':
        return "PyTorch F.conv1d + SiLU (FP32 with INT8 grid)"
    else:
        return "quant_causal_conv1d_cuda.fwd (CUDA INT8 kernel)"


def _get_ssm_kernel_name(mode: str) -> str:
    """Get the SSM kernel function name for a given mode"""
    config = MODE_CONFIG.get(mode, {})
    env = config.get('env', {})

    if mode == '5':
        return "quant_sscan_cuda.fwd_mode5 (CUDA FP32 input, dual-path)"
    elif mode == '5-0':
        return "quant_sscan_cuda.fwd (CUDA INT8 kernel, single-path for eval)"
    elif mode == '5-1':
        return "quant_sscan_cuda.fwd_mode5 (CUDA FP32 input, single-path for eval)"
    elif mode == '5-2':
        return "quant_sscan_cuda.fwd_mode5 (CUDA FP32 input from VirtualQuant+Outlier)"
    elif mode == '5-3':
        return "quant_sscan_cuda.fwd_mode5 (CUDA FP32 input from DualScale INT8)"
    elif mode == '5-4':
        return "quant_sscan_cuda.fwd_mode5 (CUDA FP32 input from QuarterScale with 4x precision for small values)"
    elif mode == '6-3':
        return "mamba_ssm selective_scan_fn (FP32 input from HalfScale mixed values)"
    elif mode == '6':
        return "quant_sscan_cuda.fwd_mode6 (CUDA FP32 input, dual-path)"
    elif mode == '6-4':
        return "quant_sscan_cuda.fwd_mode5 (CUDA FP32 input from CalibratedDualScale INT8 with α=1.0 outlier scale)"
    elif env.get('CONV1D_MODE4_SELECTIVE_GRID') == 'true' or env.get('CONV1D_MODE24_FP32') == 'true':
        return "PyTorch (FP32 simulation, no CUDA kernel)"
    elif env.get('SSM_USE_CUDA_FOR_FP32') == 'true':
        return "selective_scan_cuda.fwd (CUDA INT8 with requantization)"
    elif env.get('SSM_USE_PYTORCH_INT8') == 'true':
        return "PyTorch implementation (INT8 direct, no CUDA)"
    elif env.get('CONV1D_MODE23_FP32') == 'true':
        return "PyTorch implementation (INT8 with requantization, no CUDA)"
    elif env.get('FLOAT_SIM_ASIC_INT8') == 'true':
        return "PyTorch (FP32 simulation with INT8 grid, no CUDA)"
    else:
        return "selective_scan_cuda.fwd (CUDA INT8 kernel)"


def get_current_mode() -> Optional[str]:
    """
    Detect current mode based on environment variables.

    Returns:
        Mode identifier ('0', '2-0', etc.) or None if no mode is set
    """
    # Check QUAMBA_MODE first
    quamba_mode = os.environ.get('QUAMBA_MODE', '').strip()
    if quamba_mode and quamba_mode in MODE_CONFIG:
        return quamba_mode

    # Detect from environment variables
    for mode, config in MODE_CONFIG.items():
        env_vars = config['env']
        if not env_vars:
            # Mode 0 (baseline) - check if no other vars are set
            any_set = any(os.environ.get(var, 'false').lower() == 'true'
                         for var in ALL_MODE_ENV_VARS)
            if not any_set:
                return '0'
            continue

        # Check if all required env vars match
        match = all(
            os.environ.get(key, 'false').lower() == value.lower()
            for key, value in env_vars.items()
        )

        # Also check that no extra vars are set
        extra_vars_set = any(
            os.environ.get(var, 'false').lower() == 'true'
            for var in ALL_MODE_ENV_VARS
            if var not in env_vars
        )

        if match and not extra_vars_set:
            return mode

    return None


def print_mode_info(mode: Optional[str] = None):
    """
    Print information about a specific mode or all modes.

    Args:
        mode: Mode identifier, or None to print all modes
    """
    if mode is None:
        # Print all modes
        print(f"\n{'='*80}")
        print("Available Quamba Modes")
        print(f"{'='*80}\n")
        for m, config in MODE_CONFIG.items():
            print(f"Mode {m}: {config['name']}")
            print(f"  {config['description']}")
            if config['env']:
                print(f"  Env vars: {', '.join(f'{k}={v}' for k, v in config['env'].items())}")
            print()
    else:
        # Print specific mode
        if mode not in MODE_CONFIG:
            print(f"Error: Invalid mode '{mode}'")
            return

        config = MODE_CONFIG[mode]
        print(f"\n{'='*80}")
        print(f"Mode {mode}: {config['name']}")
        print(f"{'='*80}")
        print(f"Description: {config['description']}")
        if config['env']:
            print(f"\nEnvironment variables:")
            for key, value in config['env'].items():
                print(f"  {key} = {value}")
        else:
            print(f"\nEnvironment variables: (none, using baseline)")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    # CLI usage
    import sys

    if len(sys.argv) < 2:
        print_mode_info()
        print("Usage:")
        print("  python -m quamba.mode_config <mode>  # Print mode info")
        print("  python -m quamba.mode_config list    # List all modes")
    elif sys.argv[1] == 'list':
        print_mode_info()
    else:
        mode = sys.argv[1]
        print_mode_info(mode)
