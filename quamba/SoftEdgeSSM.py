"""
SoftEdge SSM - Research Modes Beyond Baseline

This file contains all SSM execution modes beyond Mode 0 (baseline CUDA INT8).
These are incremental improvements and research variants for comparing different
quantization and computation strategies.

Modes implemented here:
- Mode 2-0: CUDA INT8 kernel with FP32 input (requantization)
- Mode 2-1: PyTorch INT8 direct pass (no requantization)
- Mode 2-1 Legacy: PyTorch INT8 with requantization (for Mode 2-3)
- Mode 2-2: FP32 computation replicating Mode 2-1 logic
- Mode 1: Pure FP32 (upper bound)
- Mode 3: FP32 with dual-scale SE (research)
"""

import torch
import os
from .selective_scan_SE import (
    selective_scan_SE_float,
    selective_scan_SE_int8Torch,
    selective_scan_SE_mode22_fp32_replicates_mode21,
    selective_scan_SE_floatSimASIC_SoftEdge
)
import quant_sscan_cuda

#mode 0： total baseline CUDA INT8 CONV                         + CUDA INT8 SSM Input

#mode 2-0: CUDA INT8->FP32Inter->INT8CONVOut + INT8-FLOAT（Dequant）       + floattoINT&CUDA INT8 SSM Input
#mode 2-1: CUDA INT8->FP32Inter->INT8CONVOut                    + Torch Int8 SSM Kernel
#mode 2-2: CUDA INT8->FP32Inter->INT8CONVOut + INT8-FLOAT       + Torch FP32 SSM Kernel
#mode 2-3: CUDA INT8->FP32Inter->FP32CONV +                     + floattoINT Torch Int8 SSM Kernel
#mode 2-4: CUDA INT8->FP32Inter->FP32CONV +                     + Torch FP32 SSM Kernel
#mode 3: CUDA FP32->FP32Inter->FP32CONV +                       + Torch FP32 SSM Kernel             + int8 linear
def execute_mode_20_cuda_int8_requant(
    u, dt, B, C, z,
    u_scale, dt_scale, A_log, A_scale, B_scale, C_scale,
    ssm_state_scale, D, D_scale, z_scale, dt_bias, dt_bias_scale,
    delta_softplus, return_last_state, layer_id
):
    """
    Mode 2-0: CUDA INT8 kernel with requantization

    Conv1D outputs FP32 (on INT8 grid) → requantize to INT8 → CUDA INT8 kernel

    This mode tests the CUDA kernel with a dequant-requant cycle.
    """
    # Quantize FP32 u back to INT8 using u_scale
    u_int8 = torch.round(u / u_scale).clamp(-128, 127).to(torch.int8)

    # 🔍 DEBUG: Save Mode 2-0 inputs (all layers)
    if os.environ.get('SSM_DEBUG_MODE20', 'false').lower() == 'true':
        debug_dir = "debug_mode_comparison"
        os.makedirs(debug_dir, exist_ok=True)
        print(f"🔍 [Mode 2-0 DEBUG Layer {layer_id}] Saving inputs to {debug_dir}/")
        torch.save({
            'layer_id': layer_id,
            'u_int8': u_int8.cpu(),
            'u_scale': u_scale.cpu(),
            'dt': dt.cpu(),
            'dt_scale': dt_scale.cpu(),
            'A_log': A_log.cpu(),
            'A_scale': A_scale.cpu(),
            'B': B.cpu(),
            'B_scale': B_scale.cpu(),
            'C': C.cpu(),
            'C_scale': C_scale.cpu(),
            'ssm_state_scale': ssm_state_scale.cpu(),
            'D': D.cpu() if D is not None else None,
            'D_scale': D_scale.cpu() if D is not None else None,
            'z': z.cpu() if z is not None else None,
            'z_scale': z_scale.cpu(),
            'dt_bias': dt_bias.cpu() if dt_bias is not None else None,
            'dt_bias_scale': dt_bias_scale.cpu() if dt_bias is not None else None,
            'delta_softplus': delta_softplus,
        }, f"{debug_dir}/mode20_inputs_layer{layer_id}.pt")

    # Call CUDA INT8 kernel
    from .qSelectiveScan import quant_selective_scan_fn
    y = quant_selective_scan_fn(
        u_int8, u_scale[None],
        dt, dt_scale[None],
        A_log, A_scale[None],
        B, B_scale[None],
        C, C_scale[None],
        ssm_state_scale[None],
        D=D, scale_D=D_scale[None] if D is not None else None,
        z=z, scale_z=z_scale[None],
        delta_bias=dt_bias if dt_bias is not None else None,
        scale_delta_bias=dt_bias_scale[None] if dt_bias is not None else None,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )

    # 🔍 DEBUG: Save Mode 2-0 outputs (all layers)
    if os.environ.get('SSM_DEBUG_MODE20', 'false').lower() == 'true':
        print(f"✅ [Mode 2-0 DEBUG Layer {layer_id}] Saved output")
        torch.save({
            'layer_id': layer_id,
            'y': y[0].cpu() if return_last_state else y.cpu(),
            'last_state': y[1].cpu() if return_last_state else None,
            'y_dtype': str(y[0].dtype if return_last_state else y.dtype),
            'y_shape': y[0].shape if return_last_state else y.shape,
        }, f"{debug_dir}/mode20_output_layer{layer_id}.pt")

    return y


def execute_mode_21_pytorch_int8_direct(
    u, dt, B, C, z,
    u_scale, dt_scale, A_log, A_scale, B_scale, C_scale,
    ssm_state_scale, D, D_scale, z_scale, dt_bias, dt_bias_scale,
    delta_softplus, return_last_state, layer_id
):
    """
    Mode 2-1: PyTorch INT8 implementation (Direct Pass)

    Conv1D outputs INT8 → directly pass to PyTorch INT8 SSM (no requantization)

    This is the NEW path that avoids dequant-requant cycle.
    """
    # u is already INT8, no requantization needed!

    # Prepare A (same as CUDA: A = -exp(A_log * scale))
    A = A_log.float() * A_scale.float()
    A = -torch.exp(A)

    # 🔍 DEBUG: Save Mode 2-1 inputs (all layers)
    debug_mode21 = os.environ.get('SSM_DEBUG_MODE21', 'false').lower() == 'true'

    if debug_mode21:
        debug_dir = "debug_mode_comparison"
        os.makedirs(debug_dir, exist_ok=True)
        print(f"🔍 [Mode 2-1 DEBUG Layer {layer_id}] Saving inputs to {debug_dir}/")
        torch.save({
            'layer_id': layer_id,
            'u_int8': u.cpu(),  # Already INT8
            'u_scale': u_scale.cpu(),
            'dt': dt.cpu(),
            'dt_scale': dt_scale.cpu(),
            'A_log': A_log.cpu(),
            'A_scale': A_scale.cpu(),
            'A_computed': A.cpu(),
            'B': B.cpu(),
            'B_scale': B_scale.cpu(),
            'C': C.cpu(),
            'C_scale': C_scale.cpu(),
            'ssm_state_scale': ssm_state_scale.cpu(),
            'D': D.cpu() if D is not None else None,
            'D_scale': D_scale.cpu() if D is not None else None,
            'z': z.cpu() if z is not None else None,
            'z_scale': z_scale.cpu(),
            'dt_bias': dt_bias.cpu() if dt_bias is not None else None,
            'dt_bias_scale': dt_bias_scale.cpu() if dt_bias is not None else None,
            'delta_softplus': delta_softplus,
        }, f"{debug_dir}/mode21_inputs_layer{layer_id}.pt")

    # Check if we should use ssm_state_scale (Mode 2-3)
    use_state_scale = os.environ.get('SSM_USE_STATE_SCALE', 'false').lower() == 'true'

    # Call PyTorch INT8 SSM implementation
    y = selective_scan_SE_int8Torch(
        u, u_scale.item(),
        dt, dt_scale.item(),
        A, A_scale.item(),
        B, B_scale.item(),
        C, C_scale.item(),
        ssm_state_scale=ssm_state_scale.item() if use_state_scale else None,
        D_int8=D if D is not None else None,
        D_scale=D_scale.item() if D is not None else None,
        z_int8=z,
        z_scale=z_scale.item() if z is not None else None,
        delta_bias_int8=dt_bias if dt_bias is not None else None,
        delta_bias_scale=dt_bias_scale.item() if dt_bias is not None else None,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )

    # 🔍 DEBUG: Save Mode 2-1 outputs before post-processing
    if debug_mode21:
        y_before = y[0] if return_last_state else y
        torch.save({
            'layer_id': layer_id,
            'y_before_half': y_before.cpu().half(),
            'y_before_dtype': str(y_before.dtype),
            'y_before_shape': y_before.shape,
        }, f"{debug_dir}/mode21_output_before_layer{layer_id}.pt")

    # Post-process: Convert FP32 → FP16 to match CUDA kernel output
    if return_last_state:
        y_out, last_state = y
        if y_out.dtype == torch.float32:
            y_out = y_out.half()

        if debug_mode21:
            print(f"✅ [Mode 2-1 DEBUG Layer {layer_id}] Saved final output")
            torch.save({
                'layer_id': layer_id,
                'y': y_out.cpu(),
                'y_dtype': str(y_out.dtype),
                'y_shape': y_out.shape,
                'last_state': last_state.cpu(),
            }, f"{debug_dir}/mode21_output_layer{layer_id}.pt")

        return y_out, last_state
    else:
        if y.dtype == torch.float32:
            y = y.half()

        if debug_mode21:
            print(f"✅ [Mode 2-1 DEBUG Layer {layer_id}] Saved final output")
            torch.save({
                'layer_id': layer_id,
                'y': y.cpu(),
                'y_dtype': str(y.dtype),
                'y_shape': y.shape,
                'last_state': None,
            }, f"{debug_dir}/mode21_output_layer{layer_id}.pt")

        return y


def execute_mode_21_legacy_pytorch_int8_requant(
    u, dt, B, C, z,
    u_scale, dt_scale, A_log, A_scale, B_scale, C_scale,
    ssm_state_scale, D, D_scale, z_scale, dt_bias, dt_bias_scale,
    delta_softplus, return_last_state, layer_id
):
    """
    Mode 2-1 Legacy: PyTorch INT8 with requantization

    Conv1D outputs FP32 (on INT8 grid) → requantize to INT8 → PyTorch INT8 SSM

    This is for backwards compatibility with Mode 2-3.
    """
    # Quantize FP32 u back to INT8
    u_int8 = torch.round(u / u_scale).clamp(-128, 127).to(torch.int8)

    # Prepare A
    A = A_log.float() * A_scale.float()
    A = -torch.exp(A)

    # 🔍 DEBUG: Save inputs
    debug_mode21 = os.environ.get('SSM_DEBUG_MODE21', 'false').lower() == 'true'
    debug_mode23 = os.environ.get('SSM_DEBUG_MODE23', 'false').lower() == 'true'

    if debug_mode21 or debug_mode23:
        debug_dir = "debug_mode_comparison"
        os.makedirs(debug_dir, exist_ok=True)
        mode_name = "2-3" if debug_mode23 else "2-1"
        file_suffix = "mode23" if debug_mode23 else "mode21"

        print(f"🔍 [Mode {mode_name} DEBUG Layer {layer_id}] Saving inputs to {debug_dir}/")
        torch.save({
            'layer_id': layer_id,
            'u_int8': u_int8.cpu(),
            'u_scale': u_scale.cpu(),
            'dt': dt.cpu(),
            'dt_scale': dt_scale.cpu(),
            'A_log': A_log.cpu(),
            'A_scale': A_scale.cpu(),
            'A_computed': A.cpu(),
            'B': B.cpu(),
            'B_scale': B_scale.cpu(),
            'C': C.cpu(),
            'C_scale': C_scale.cpu(),
            'ssm_state_scale': ssm_state_scale.cpu(),
            'D': D.cpu() if D is not None else None,
            'D_scale': D_scale.cpu() if D is not None else None,
            'z': z.cpu() if z is not None else None,
            'z_scale': z_scale.cpu(),
            'dt_bias': dt_bias.cpu() if dt_bias is not None else None,
            'dt_bias_scale': dt_bias_scale.cpu() if dt_bias is not None else None,
            'delta_softplus': delta_softplus,
        }, f"{debug_dir}/{file_suffix}_inputs_layer{layer_id}.pt")

    # Check if we should use ssm_state_scale (Mode 2-3)
    use_state_scale = os.environ.get('SSM_USE_STATE_SCALE', 'false').lower() == 'true'

    # Call PyTorch INT8 SSM
    y = selective_scan_SE_int8Torch(
        u_int8, u_scale.item(),
        dt, dt_scale.item(),
        A, A_scale.item(),
        B, B_scale.item(),
        C, C_scale.item(),
        ssm_state_scale=ssm_state_scale.item() if use_state_scale else None,
        D_int8=D if D is not None else None,
        D_scale=D_scale.item() if D is not None else None,
        z_int8=z,
        z_scale=z_scale.item() if z is not None else None,
        delta_bias_int8=dt_bias if dt_bias is not None else None,
        delta_bias_scale=dt_bias_scale.item() if dt_bias is not None else None,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )

    # 🔍 DEBUG: Save outputs before post-processing
    if debug_mode21 or debug_mode23:
        mode_name = "2-3" if debug_mode23 else "2-1"
        file_suffix = "mode23" if debug_mode23 else "mode21"

        print(f"✅ [Mode {mode_name} DEBUG Layer {layer_id}] Saved output (before post-processing)")
        torch.save({
            'layer_id': layer_id,
            'y_before_half': y[0].cpu() if return_last_state else y.cpu(),
            'y_before_dtype': str(y[0].dtype if return_last_state else y.dtype),
            'y_before_shape': y[0].shape if return_last_state else y.shape,
            'last_state': y[1].cpu() if return_last_state else None,
        }, f"{debug_dir}/{file_suffix}_output_before_layer{layer_id}.pt")

    # Post-process output
    if not return_last_state:
        if y.dtype == torch.float32:
            y = y.half()

        if debug_mode21 or debug_mode23:
            mode_name = "2-3" if debug_mode23 else "2-1"
            file_suffix = "mode23" if debug_mode23 else "mode21"

            print(f"✅ [Mode {mode_name} DEBUG Layer {layer_id}] Saved final output")
            torch.save({
                'layer_id': layer_id,
                'y': y.cpu(),
                'y_dtype': str(y.dtype),
                'y_shape': y.shape,
            }, f"{debug_dir}/{file_suffix}_output_layer{layer_id}.pt")

        return y
    else:
        y_out, last_state = y
        if y_out.dtype == torch.float32:
            y_out = y_out.half()

        if debug_mode21 or debug_mode23:
            mode_name = "2-3" if debug_mode23 else "2-1"
            file_suffix = "mode23" if debug_mode23 else "mode21"

            print(f"✅ [Mode {mode_name} DEBUG Layer {layer_id}] Saved final output")
            torch.save({
                'layer_id': layer_id,
                'y': y_out.cpu(),
                'y_dtype': str(y_out.dtype),
                'y_shape': y_out.shape,
                'last_state': last_state.cpu(),
            }, f"{debug_dir}/{file_suffix}_output_layer{layer_id}.pt")

        return y_out, last_state


def execute_fp32_modes(
    mode_type, u, dt, B, C, z,
    u_scale, dt_scale, A_log, A_scale, B_scale, C_scale,
    D, D_scale, z_scale, dt_bias, dt_bias_scale,
    delta_softplus, return_last_state, layer_id
):
    """
    Modes 1/2-2/3: FP32 PyTorch SSM implementations

    mode_type: 'fp32_upper_bound', 'mode22_fp32_replicates_mode21', or 'dual_scale_se'

    All inputs are FP32, use pure PyTorch implementations.
    """
    # Dequantize weight buffers to FP32
    A = A_log.float() * A_scale.float()
    A = -torch.exp(A)

    dt_fp32 = dt.float() * dt_scale.float()
    B_fp32 = B.float() * B_scale.float()
    C_fp32 = C.float() * C_scale.float()

    D_fp32 = None
    if D is not None:
        D_fp32 = D.float() * D_scale.float()

    z_fp32 = None
    if z is not None:
        z_fp32 = z.float() * z_scale.float()

    delta_bias_fp32 = None
    if dt_bias is not None:
        delta_bias_fp32 = dt_bias.float() * dt_bias_scale.float()

    # Call the appropriate SE SSM version
    if mode_type == 'fp32_upper_bound':
        # Mode 1: Full FP32 precision (upper bound)
        y = selective_scan_SE_float(
            u, dt_fp32, A, B_fp32, C_fp32,
            D=D_fp32, z=z_fp32,
            delta_bias=delta_bias_fp32,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state
        )
    elif mode_type == 'mode22_fp32_replicates_mode21':
        # Mode 2-2: FP32 computation that exactly replicates Mode 2-1 logic
        y = selective_scan_SE_mode22_fp32_replicates_mode21(
            u, dt_fp32, A, B_fp32, C_fp32,
            D=D_fp32,
            z=z_fp32,
            delta_bias=delta_bias_fp32,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state
        )
    else:  # 'dual_scale_se'
        # Mode 3: FP32 computation with dual-scale SE for outliers
        scale_factor = float(os.environ.get('FLOAT_SIM_SCALE_FACTOR', '2025'))
        y = selective_scan_SE_floatSimASIC_SoftEdge(
            u, dt_fp32, A, B_fp32, C_fp32,
            D=D_fp32, z=z_fp32,
            delta_bias=delta_bias_fp32,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
            scale_factor=scale_factor,
            u_scale=u_scale.item()
        )

    # Post-process output
    if not return_last_state:
        if y.dtype == torch.float32:
            y = y.half()
        return y
    else:
        y_out, last_state = y
        if y_out.dtype == torch.float32:
            y_out = y_out.half()
        return y_out, last_state
