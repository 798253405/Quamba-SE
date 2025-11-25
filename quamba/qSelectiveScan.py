"""
The code is modfied from
https://github.com/state-spaces/mamba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os

import quant_sscan_cuda
from .triton.selective_state_update import quant_sscan_update_triton
from .SoftEdgeSSM import (
    execute_mode_20_cuda_int8_requant,
    execute_mode_21_pytorch_int8_direct,
    execute_mode_21_legacy_pytorch_int8_requant,
    execute_fp32_modes
)

# Global counter for tracking QSelectiveScan layers
_QSSCAN_LAYER_COUNTER = 0

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_()
    return w, scales


class QSScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, u_scale, delta, delta_scale, A, A_scale, B, B_scale, C, C_scale, ssm_state_scale,
                D=None, D_scale=None, z=None, z_scale=None, delta_bias=None, delta_bias_scale=None,
                delta_softplus=False, return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
        out, x = quant_sscan_cuda.fwd(
            u, u_scale, delta, delta_scale, A, A_scale, B, B_scale, C, C_scale, ssm_state_scale,
            D, D_scale, z, z_scale, delta_bias, delta_bias_scale, delta_softplus)
        # last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        last_state = x[:, :, -1, :]  # (batch, dim, dstate)
        return out if not return_last_state else (out, last_state)
        # if z is None:
        #     return out if not return_last_state else (out, last_state)
        # else: # has z
        #     ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
        #     out_z = rest[0]
        #     return out_z if not return_last_state else (out_z, last_state)


def quant_selective_scan_fn(
        u, scale_u, delta, scale_delta,
        A, scale_A, B, scale_B, C, scale_C, ssm_state_scale,
        D=None, scale_D=None, z=None, scale_z=None,
        delta_bias=None, scale_delta_bias=None,
        delta_softplus=False, return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return QSScanFn.apply(
        u, scale_u, delta, scale_delta, A, scale_A, 
        B, scale_B, C, scale_C, ssm_state_scale,
        D, scale_D, z, scale_z, delta_bias, scale_delta_bias, 
        delta_softplus, return_last_state)


#W8A8B8O8 QSScan
class QSScan(nn.Module):

    def __init__(self, d_state, d_inner, delta_softplus=True):
        global _QSSCAN_LAYER_COUNTER
        device = torch.device("cuda") # triton must be on Cuda

        super().__init__()
        # Assign unique layer ID to this instance
        self.layer_id = _QSSCAN_LAYER_COUNTER
        _QSSCAN_LAYER_COUNTER += 1

        self.delta_softplus = delta_softplus
        # allocate dt_bias
        self.register_buffer("dt_bias", torch.empty((d_inner), device=device, dtype=torch.int8))
        self.register_buffer("dt_bias_scale", torch.tensor(1.0, device=device, dtype=torch.float32))

        # allocate A_log
        self.register_buffer("A_log", torch.empty((d_inner, d_state), device=device, dtype=torch.int8))
        self.register_buffer("A_scale", torch.tensor(1.0, device=device, dtype=torch.float32))

        # allocate D "skip" parameter
        self.register_buffer("D", torch.empty((d_inner), device=device, dtype=torch.int8)) 
        self.register_buffer("D_scale", torch.tensor(1.0, device=device, dtype=torch.float32))

        # allocate scaling factors
        self.register_buffer('u_scale', torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer('dt_scale', torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer('B_scale', torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer('C_scale', torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer('z_scale', torch.tensor(1.0, device=device, dtype=torch.float32))
        self.register_buffer('ssm_state_scale', torch.tensor(1.0, device=device, dtype=torch.float32))

    @classmethod
    def from_fp16(cls, d_state, d_inner, A_log, D=None, dt_bias=None, delta_softplus=True,
                 ssm_state_scale=1.0, u_scale=1.0, dt_scale=1.0, B_scale=1.0, C_scale=1.0, z_scale=1.0):
        qsscan = cls(d_state, d_inner)

        A_log_quant, A_log_scale = quantize_weight_per_tensor_absmax(A_log, n_bits=8)
        A_log_quant = A_log_quant.to(torch.int8)
        qsscan.A_log = A_log_quant
        qsscan.A_scale = A_log_scale.float()

        if D is not None:
            D_quant, D_scale = quantize_weight_per_tensor_absmax(D, n_bits=8)
            D_quant = D_quant.to(torch.int8)
            qsscan.D = D_quant
            qsscan.D_scale = D_scale.float()
        else:
            qsscan.D = None
        
        if dt_bias is not None:
            dt_bias_quant, dt_bias_scale = quantize_weight_per_tensor_absmax(dt_bias, n_bits=8)
            dt_bias_quant = dt_bias_quant.to(torch.int8)
            qsscan.dt_bias = dt_bias_quant
            qsscan.dt_bias_scale = dt_bias_scale.float()
        else:
            qsscan.dt_bias = None
            qsscan.dt_bias_scale = torch.tensor(0.0)
        
        qsscan.u_scale = u_scale.float()
        qsscan.dt_scale = dt_scale.float()
        qsscan.B_scale = B_scale.float()
        qsscan.C_scale = C_scale.float()
        qsscan.z_scale = z_scale.float()
        qsscan.ssm_state_scale = ssm_state_scale.float()
        qsscan.delta_softplus = delta_softplus
        return qsscan

    #NOTE(HY): Only activate q_sscan when quamba is True,
    # since quantize_tensor_per_tensor_absmax only returns real scales when quamba is True.
    @torch.no_grad()
    def forward(self, u, dt, B, C, z=None, return_last_state=False):
        """
        Selective Scan forward with multiple execution modes.

        Environment Variables for Mode Control:
        ========================================

        SSM_USE_CUDA_FOR_FP32 (default: false)
            - Mode 2-0: FP32 input (u, on INT8 grid) → quantize → CUDA INT8 kernel
            - Verification mode: should match Mode 0 exactly

        SSM_USE_PYTORCH_INT8 (default: false)
            - Mode 2-1: INT8 input (u) → PyTorch INT8 implementation (Direct Pass)
            - Verification mode: should match Mode 0/2-0 results (no CUDA)
            - When used with FLOAT_SIM_ASIC_INT8: FP32 input → quantize → PyTorch INT8 (Legacy)

        FLOAT_SIM_ASIC_INT8 (default: false)
            - Mode 2-2: FP32 input (u, on INT8 grid) → FP32 PyTorch SSM (INT8 simulation)
            - Verification mode: should match Mode 0/2-0/2-1 results

        Default (all false):
            - Mode 0: INT8 input (u) → CUDA INT8 kernel (original baseline)

        Mode Selection Logic:
        ====================
        If u.dtype == torch.int8:
            if SSM_USE_PYTORCH_INT8 == true:
                → Mode 2-1: Pass INT8 directly to PyTorch INT8 implementation
            else:
                → Mode 0: CUDA INT8 kernel (original baseline)

        If u.dtype == torch.float32 (from conv1d with dequantization):
            if SSM_USE_CUDA_FOR_FP32 == true:
                → Mode 2-0: Quantize u to INT8, call CUDA kernel
            elif SSM_USE_PYTORCH_INT8 == true:
                → Mode 2-1 (Legacy): Quantize u to INT8, call PyTorch INT8 implementation
            elif FLOAT_SIM_ASIC_INT8 == true:
                → Mode 2-2: FP32 PyTorch SSM (INT8 simulation)
            else:
                → Should not reach here (conv1d would not dequantize)
        """
        # DEBUG: Track first call
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = False

        # ========================================================================
        # Parse environment variables for mode selection
        # ========================================================================
        float_sim_asic_int8 = os.environ.get('FLOAT_SIM_ASIC_INT8', 'false').lower() == 'true'
        ssm_use_cuda_for_fp32 = os.environ.get('SSM_USE_CUDA_FOR_FP32', 'false').lower() == 'true'
        ssm_use_pytorch_int8 = os.environ.get('SSM_USE_PYTORCH_INT8', 'false').lower() == 'true'
        conv1d_mode23_fp32 = os.environ.get('CONV1D_MODE23_FP32', 'false').lower() == 'true'
        conv1d_mode24_fp32 = os.environ.get('CONV1D_MODE24_FP32', 'false').lower() == 'true'
        conv1d_mode3_fp32 = os.environ.get('CONV1D_MODE3_FP32', 'false').lower() == 'true'

        # # DEBUG: Print on first call
        # if not self._debug_printed:
        #     print(f"\n{'='*80}")
        #     print(f"[QSScan Mode Selection] Environment variables:")
        #     print(f"  FP32_SSM_INPUT           = {os.environ.get('FP32_SSM_INPUT', 'NOT_SET')}")
        #     print(f"  FLOAT_SIM_ASIC_INT8      = {os.environ.get('FLOAT_SIM_ASIC_INT8', 'NOT_SET')}")
        #     print(f"  SSM_USE_CUDA_FOR_FP32    = {os.environ.get('SSM_USE_CUDA_FOR_FP32', 'NOT_SET')}")
        #     print(f"  SSM_USE_PYTORCH_INT8     = {os.environ.get('SSM_USE_PYTORCH_INT8', 'NOT_SET')}")
        #     print(f"  CONV1D_MODE23_FP32       = {os.environ.get('CONV1D_MODE23_FP32', 'NOT_SET')}")
        #     print(f"  CONV1D_MODE24_FP32       = {os.environ.get('CONV1D_MODE24_FP32', 'NOT_SET')}")
        #     print(f"  CONV1D_MODE3_FP32        = {os.environ.get('CONV1D_MODE3_FP32', 'NOT_SET')}")
        #     print(f"\n  Input tensor dtypes:")
        #     print(f"    u: {u.dtype}, dt: {dt.dtype}, B: {B.dtype}, C: {C.dtype}")

        #     # Determine which mode will be used
        #     if u.dtype == torch.float32:
        #         if conv1d_mode3_fp32:
        #             mode_str = "Mode 3: FP32/FP16 input → TRUE FP32 Conv1D → PyTorch FP32 SSM + INT8 Linear"
        #         elif conv1d_mode24_fp32:
        #             mode_str = "Mode 2-4: TRUE FP32 Conv1D output → PyTorch FP32 SSM (no requantization)"
        #         elif conv1d_mode23_fp32:
        #             mode_str = "Mode 2-3: TRUE FP32 Conv1D output → quantize to INT8 → PyTorch INT8 SSM"
        #         elif ssm_use_cuda_for_fp32:
        #             mode_str = "Mode 2-0: FP32 input (INT8 grid) → quantize to INT8 → CUDA INT8 kernel"
        #         elif ssm_use_pytorch_int8:
        #             mode_str = "Mode 2-1 (Legacy): FP32 input (INT8 grid) → quantize to INT8 → PyTorch INT8 implementation"
        #         elif fp32_ssm_input:
        #             mode_str = "Mode 1: FP32 input (full precision) → FP32 PyTorch SSM (upper bound)"
        #         elif float_sim_asic_int8:
        #             mode_str = "Mode 2-2: FP32 input (INT8 grid) → FP32 PyTorch SSM (INT8 simulation)"
        #         else:
        #             mode_str = "Mode ?: FP32 input but no mode selected (unexpected)"
        #     elif u.dtype == torch.int8:
        #         if ssm_use_pytorch_int8:
        #             mode_str = "Mode 2-1: INT8 input → PyTorch INT8 implementation (Direct Pass)"
        #         else:
        #             mode_str = "Mode 0: INT8 input → CUDA INT8 kernel (baseline)"
        #     else:
        #         mode_str = f"Unknown: u.dtype = {u.dtype}"

        #     print(f"\n  Selected Mode: {mode_str}")
        #     print(f"{'='*80}\n")
        #     self._debug_printed = True

        # ========================================================================
        # Mode 3: FP32/FP16 input → FP32 Conv1D → PyTorch FP32 SSM + INT8 Linear
        # ========================================================================
        # Purpose: Hybrid precision - FP32 for Conv/SSM, INT8 for Linear
        # Conv1D: Accepts FP32/FP16 input → CUDA FP32 kernel → FP32 output
        # SSM: Direct FP32 computation (same as Mode 2-4)
        # Linear: Keeps INT8 quantization (baseline)
        # ========================================================================
        if conv1d_mode3_fp32 and u.dtype == torch.float32:
            return execute_fp32_modes(
                'fp32_upper_bound',  # Use pure FP32 SSM (same as Mode 2-4)
                u, dt, B, C, z,
                self.u_scale, self.dt_scale, self.A_log, self.A_scale, self.B_scale, self.C_scale,
                self.D, self.D_scale, self.z_scale, self.dt_bias, self.dt_bias_scale,
                self.delta_softplus, return_last_state, self.layer_id
            )

        # ========================================================================
        # Mode 2-4: TRUE FP32 Conv1D → Mode 2-2 FP32 SSM
        # ========================================================================
        # Purpose: Test TRUE FP32 Conv1D output (vs INT8 grid) with Mode 2-2 SSM
        # Conv1D: CUDA FP32 kernel → TRUE FP32 output (same as Mode 2-3)
        # x: Requantize FP32 → INT8 for x_proj (dt/B/C are INT8)
        # SSM: Mode 2-2 same FP32 kernel (mode22_fp32_replicates_mode21)
        # Difference from Mode 2-2: Conv1D output is TRUE FP32 (not INT8 grid)
        # ========================================================================
        elif conv1d_mode24_fp32 and u.dtype == torch.float32:
            return execute_fp32_modes(
                'mode22_fp32_replicates_mode21',  # Use Mode 2-2 same FP32 SSM
                u, dt, B, C, z,
                self.u_scale, self.dt_scale, self.A_log, self.A_scale, self.B_scale, self.C_scale,
                self.D, self.D_scale, self.z_scale, self.dt_bias, self.dt_bias_scale,
                self.delta_softplus, return_last_state, self.layer_id
            )

        # ========================================================================
        # Mode 2-3: TRUE FP32 Conv1D → PyTorch INT8 SSM (requantization)
        # ========================================================================
        # Purpose: Test Conv1D FP32 output (preserving CUDA internal precision)
        #          with PyTorch INT8 SSM implementation
        # Conv1D: CUDA INT8 kernel with FP32 output (no quantization step)
        # SSM: Requantize FP32 → INT8 → PyTorch INT8 implementation
        # ========================================================================
        elif conv1d_mode23_fp32 and u.dtype == torch.float32:
            return execute_mode_21_legacy_pytorch_int8_requant(
                u, dt, B, C, z,
                self.u_scale, self.dt_scale, self.A_log, self.A_scale, self.B_scale, self.C_scale,
                self.ssm_state_scale, self.D, self.D_scale, self.z_scale, self.dt_bias, self.dt_bias_scale,
                self.delta_softplus, return_last_state, self.layer_id
            )

        # ========================================================================
        # Mode 2-0: CUDA INT8 with requantization (SoftEdge incremental improvement)
        # ========================================================================
        elif ssm_use_cuda_for_fp32 and u.dtype == torch.float32:
            return execute_mode_20_cuda_int8_requant(
                u, dt, B, C, z,
                self.u_scale, self.dt_scale, self.A_log, self.A_scale, self.B_scale, self.C_scale,
                self.ssm_state_scale, self.D, self.D_scale, self.z_scale, self.dt_bias, self.dt_bias_scale,
                self.delta_softplus, return_last_state, self.layer_id
            )

        # ========================================================================
        # Mode 2-1: PyTorch INT8 direct pass (SoftEdge incremental improvement)
        # ========================================================================
        elif ssm_use_pytorch_int8 and u.dtype == torch.int8:
            return execute_mode_21_pytorch_int8_direct(
                u, dt, B, C, z,
                self.u_scale, self.dt_scale, self.A_log, self.A_scale, self.B_scale, self.C_scale,
                self.ssm_state_scale, self.D, self.D_scale, self.z_scale, self.dt_bias, self.dt_bias_scale,
                self.delta_softplus, return_last_state, self.layer_id
            )

        # ========================================================================
        # Mode 2-1 (Legacy): FP32 input (INT8 grid) → Quantize to INT8 → PyTorch INT8 implementation
        # ========================================================================
        # Purpose: Backwards compatibility for Mode 2-3
        #          Conv1D outputs FP32 (on INT8 grid), SSM requantizes and uses PyTorch
        # ========================================================================
        elif ssm_use_pytorch_int8 and u.dtype == torch.float32:
            return execute_mode_21_legacy_pytorch_int8_requant(
                u, dt, B, C, z,
                self.u_scale, self.dt_scale, self.A_log, self.A_scale, self.B_scale, self.C_scale,
                self.ssm_state_scale, self.D, self.D_scale, self.z_scale, self.dt_bias, self.dt_bias_scale,
                self.delta_softplus, return_last_state, self.layer_id
            )

        # ========================================================================
        # Mode 2-2: FP32 input (INT8 grid) → FP32 PyTorch SSM (INT8 simulation)
        # ========================================================================
        # Purpose: FP32 computation that replicates Mode 2-1 logic
        # ========================================================================
        elif float_sim_asic_int8 and u.dtype == torch.float32:
            return execute_fp32_modes(
                'mode22_fp32_replicates_mode21', u, dt, B, C, z,
                self.u_scale, self.dt_scale, self.A_log, self.A_scale, self.B_scale, self.C_scale,
                self.D, self.D_scale, self.z_scale, self.dt_bias, self.dt_bias_scale,
                self.delta_softplus, return_last_state, self.layer_id
            )

        # ========================================================================
        # Mode 0: INT8 input → CUDA INT8 kernel (original baseline)
        # ========================================================================
        # Purpose: Standard quantized inference path (default)
        # All inputs are INT8, computation happens in CUDA kernel
        # ========================================================================
        else:
            # u, dt, B, C, z are all INT8
            # Kernel will dequantize internally, compute in FP32, then quantize output

            y = quant_selective_scan_fn(
                # Input u (INT8, from conv1d)
                u, self.u_scale[None],  # u_scale = x_proj:input_scale

                # Delta (time step, INT8, from dt_proj)
                dt, self.dt_scale[None],  # dt_scale = dt_proj:output_scale

                # State transition matrix A (INT8 weight buffer)
                self.A_log, self.A_scale[None],

                # Input matrix B (INT8, from x_proj split)
                B, self.B_scale[None],  # B_scale = x_proj:output_scale

                # Output matrix C (INT8, from x_proj split)
                C, self.C_scale[None],  # C_scale = x_proj:output_scale

                # SSM state scale for internal state quantization
                ssm_state_scale=self.ssm_state_scale[None],

                # Optional: Skip connection parameter D (INT8 weight buffer)
                D=self.D if self.D is not None else None,
                scale_D=self.D_scale[None] if self.D is not None else None,

                # Optional: Gate z (INT8, from in_proj split)
                z=z, scale_z=self.z_scale[None],  # z_scale = in_proj:output_scale

                # Optional: Delta bias (INT8 weight buffer)
                delta_bias=self.dt_bias if self.dt_bias is not None else None,
                scale_delta_bias=self.dt_bias_scale[None] if self.dt_bias is not None else None,

                # Activation function flag
                delta_softplus=self.delta_softplus,

                # Whether to return final SSM state
                return_last_state=return_last_state,
            )

            # CUDA kernel returns output in (B, L, D) format (already transposed)
            return y


    @torch.no_grad()
    def update(self, ssm_state, u, dt, B, C, z=None):

        y = quant_sscan_update_triton(
            ssm_state, u, self.u_scale[None],
            dt, self.dt_scale[None],
            self.A_log, self.A_scale[None],
            self.ssm_state_scale[None],
            B, self.B_scale[None], 
            C, self.C_scale[None],
            self.D if self.D is not None else None,
            self.D_scale[None] if self.D is not None else None,
            z if z is not None else None,
            self.z_scale[None] if z is not None else None,
            self.dt_bias if self.dt_bias is not None else None,
            self.dt_bias_scale[None] if self.dt_bias is not None else None,
            self.delta_softplus
        )
        return y, ssm_state
    
    def to(self, *args, **kwargs):
        super(QSScan, self).to(*args, **kwargs)
        self.u_scale = self.u_scale.to(*args, **kwargs)
        self.A_log = self.A_log.to(*args, **kwargs)
        self.A_scale = self.A_scale.to(*args, **kwargs)
        if self.D is not None:
            self.D = self.D.to(*args, **kwargs)
        if self.D_scale is not None:
            self.D_scale = self.D_scale.to(*args, **kwargs)
        if self.dt_bias is not None:
            self.dt_bias = self.dt_bias.to(*args, **kwargs)
        if self.dt_bias_scale is not None:
            self.dt_bias_scale = self.dt_bias_scale.to(*args, **kwargs)
        if self.B_scale is not None:
            self.B_scale = self.B_scale.to(*args, **kwargs)
        if self.C_scale is not None:
            self.C_scale = self.C_scale.to(*args, **kwargs)
        if self.z_scale is not None:
            self.z_scale = self.z_scale.to(*args, **kwargs)
        return self

    def __repr__(self):
        return f"QSScan()"