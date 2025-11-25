import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from .quant_utils import quantize_tensor_per_tensor_absmax
import os
import json
from pathlib import Path

import quant_causal_conv1d_cuda
import quamba2_conv1d_cuda

# Global counter for tracking conv1d layer calls
_CONV1D_LAYER_COUNTER = 0

class QCausalConv1D(nn.Module):

    def __init__ (self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()
        assert in_channels == out_channels == groups, "QCausalConv1D only supports in_channels == out_channels == groups"
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 1, "QCausalConv1D only supports 1D kernel_size"
            kernel_size = kernel_size[0]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.register_buffer('weight', torch.empty(
            (in_channels, kernel_size), dtype=torch.int8, **factory_kwargs))
        if bias is not None:
            self.register_buffer('bias', torch.empty(
                (in_channels), dtype=torch.int8, **factory_kwargs))
        else:
            self.bias = None
        self.weight_scale = 0.0
        self.bias_scale = 0.0
        self.input_scale = 0.0
        self.output_scale = 0.0
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)
        
    @classmethod
    def from_fp16(cls, originalLayer: nn.Conv1d, input_scale=1.0, output_scale=1.0):
        device = originalLayer.weight.device
        qconv = cls(originalLayer.in_channels, originalLayer.out_channels,
                    originalLayer.kernel_size, originalLayer.stride,
                    originalLayer.padding, originalLayer.dilation,
                    originalLayer.groups, originalLayer.bias, device=device)
        
        qconv.input_scale = input_scale
        qconv.output_scale = output_scale

        int8_weight, weight_scale = quantize_tensor_per_tensor_absmax(
            originalLayer.weight.clone().detach(),
            n_bits = 8,
            clip_ratio = 1.0
        )
        int8_weight = rearrange(int8_weight.to(torch.int8), "d 1 w -> d w").contiguous()
        qconv.weight = int8_weight.to(device)
        qconv.weight_scale = weight_scale.item()
        if originalLayer.bias is not None:
            int8_bias, bias_scale = quantize_tensor_per_tensor_absmax(
                originalLayer.bias.clone().detach(),
                n_bits = 8,
                clip_ratio = 1.0
            )
            int8_bias = int8_bias.to(torch.int8).contiguous().to(device)
            qconv.bias = int8_bias
            qconv.bias_scale = bias_scale.item()
        else:
            qconv.bias = None
            qconv.bias_scale = 1.0
        return qconv

    def store_hook(self, module, state_dict, prefix, local_metadata):
        state_dict[prefix + 'weight_scale'] = self.weight_scale
        state_dict[prefix + 'bias_scale'] = self.bias_scale
        state_dict[prefix + 'input_scale'] = self.input_scale
        state_dict[prefix + 'output_scale'] = self.output_scale
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.weight_scale = state_dict[prefix + 'weight_scale']
        self.bias_scale = state_dict[prefix + 'bias_scale']
        self.input_scale = state_dict[prefix + 'input_scale']
        self.output_scale = state_dict[prefix + 'output_scale']
        del state_dict[prefix + 'weight_scale']
        del state_dict[prefix + 'bias_scale']
        del state_dict[prefix + 'input_scale']
        del state_dict[prefix + 'output_scale']

    @torch.no_grad()
    def forward(self, x):
        global _CONV1D_LAYER_COUNTER

        # Check simulation mode
        float_sim_asic_int8 = os.environ.get('FLOAT_SIM_ASIC_INT8', 'false').lower() == 'true'
        conv1d_mode23_fp32 = os.environ.get('CONV1D_MODE23_FP32', 'false').lower() == 'true'
        conv1d_mode24_fp32 = os.environ.get('CONV1D_MODE24_FP32', 'false').lower() == 'true'
        conv1d_mode3_fp32 = os.environ.get('CONV1D_MODE3_FP32', 'false').lower() == 'true'
        conv1d_mode4_selective = os.environ.get('CONV1D_MODE4_SELECTIVE_GRID', 'false').lower() == 'true'

        # Check if intermediate value saving is enabled
        save_mode = os.environ.get('SAVE_INTERMEDIATE_VALUES', '')  # 'cuda' or 'floatsim'
        save_dir = os.environ.get('INTERMEDIATE_VALUES_DIR', '')
        should_save = (save_mode in ['cuda', 'floatsim']) and save_dir and (_CONV1D_LAYER_COUNTER == 23)

        # Mode 4: Use PyTorch (not CUDA), apply grid constraint on non-overflow values
        if conv1d_mode4_selective:
            # Dequantize INT8 to FP32
            x_fp32 = x.float() * self.input_scale  # (B, D, L)
            weight_fp32 = self.weight.float() * self.weight_scale  # (D, kernel_size)
            bias_fp32 = self.bias.float() * self.bias_scale if self.bias is not None else None  # (D,)

            # PyTorch Conv1d expects (B, C_in, L_in)
            # Our x_fp32 is already (B, D, L), weight is (D, kernel_size)
            # Use depthwise conv: groups=D
            weight_conv = weight_fp32.unsqueeze(1)  # (D, 1, kernel_size)

            # Causal conv: pad left, then remove right padding
            y_fp32 = F.conv1d(x_fp32, weight_conv, bias=bias_fp32,
                             groups=self.in_channels, padding=self.kernel_size - 1)
            y_fp32 = y_fp32[:, :, :x_fp32.shape[2]]  # Remove extra padding

            # SiLU activation
            y_fp32 = y_fp32 * torch.sigmoid(y_fp32)

            # For each value: if overflow, keep FP32; else quantize to grid
            y_quantized = torch.round(y_fp32 / self.output_scale)

            # Apply grid: quantize + dequantize
            y_grid = torch.clamp(y_quantized, -128, 127) * self.output_scale

            # For overflow values (|quantized| > 127), use original FP32
            overflow_mask = torch.abs(y_quantized) > 127
            y_output = torch.where(overflow_mask, y_fp32, y_grid)

            # Print statistics for first 3 layers and layer 23
            if _CONV1D_LAYER_COUNTER <= 2 or _CONV1D_LAYER_COUNTER == 23:
                overflow_count = overflow_mask.sum().item()
                total_count = overflow_mask.numel()

                print(f"\n{'='*80}")
                print(f"[Mode 4 Fake Mixed Precision] Layer {_CONV1D_LAYER_COUNTER}")
                print(f"{'='*80}")
                print(f"  True FP32 range: [{y_fp32.min().item():.6f}, {y_fp32.max().item():.6f}]")
                print(f"  Grid range: [{y_grid.min().item():.6f}, {y_grid.max().item():.6f}]")
                print(f"  Output range: [{y_output.min().item():.6f}, {y_output.max().item():.6f}]")
                print(f"  Overflow count: {overflow_count} / {total_count} ({100.0*overflow_count/total_count:.4f}%)")

                if overflow_count > 0:
                    print(f"  Example overflow values (first 3):")
                    overflow_idx = torch.nonzero(overflow_mask.flatten())[:3]
                    for idx in overflow_idx:
                        i = idx.item()
                        print(f"    [{i}] FP32={y_fp32.flatten()[i]:.6f}, Grid={y_grid.flatten()[i]:.6f}, Output={y_output.flatten()[i]:.6f}")
                else:
                    print(f"  No overflow values (all on grid)")
                print(f"{'='*80}\n")

            _CONV1D_LAYER_COUNTER += 1
            return y_output

        # Baseline: CUDA INT8 kernel (returns INT8)
        if not float_sim_asic_int8:
            # Use INT8 CUDA kernel (same as Baseline)
            y = quant_causal_conv1d_cuda.fwd(
                    x, self.input_scale,
                    self.weight, self.weight_scale,
                    self.output_scale,
                    self.bias_scale, self.bias,
                    None, None, None, True
                )

            # ===== SAVE INTERMEDIATE VALUES FOR COMPARISON =====
            if should_save and save_mode == 'cuda':
                os.makedirs(save_dir, exist_ok=True)

                # Save input (INT8)
                torch.save(x.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_input.pt'))

                # Save output (INT8)
                torch.save(y.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_output_int8.pt'))

                # Save dequantized output (FP32) for comparison
                y_dequant_fp32 = y.float() * self.output_scale
                torch.save(y_dequant_fp32.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_output_fp32.pt'))

                # Save scale for verification
                scale_info = {
                    'input_scale': self.input_scale.item() if hasattr(self.input_scale, 'item') else self.input_scale,
                    'weight_scale': self.weight_scale.item() if hasattr(self.weight_scale, 'item') else self.weight_scale,
                    'output_scale': self.output_scale.item() if hasattr(self.output_scale, 'item') else self.output_scale,
                    'bias_scale': self.bias_scale.item() if hasattr(self.bias_scale, 'item') else self.bias_scale,
                }
                torch.save(scale_info, os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_scales.pt'))

            # ===== END SAVE CODE =====

            _CONV1D_LAYER_COUNTER += 1
            return y  # INT8

        # Float-Sim INT8: CUDA INT8 kernel (returns INT8)
        # Mode 2-0, 2-1, 2-2, 2-3: Use CUDA INT8 kernel, return INT8
        # Dequantization will be handled in qMambaLayer based on specific mode
        else:  # float_sim_asic_int8 == True
            # Use CUDA INT8 kernel (100% identical to Mode 0 baseline)
            y = quant_causal_conv1d_cuda.fwd(
                    x, self.input_scale,
                    self.weight, self.weight_scale,
                    self.output_scale,
                    self.bias_scale, self.bias,
                    None, None, None, True  # silu_activation=True
                )

            # Return INT8 directly (same as Mode 0)
            # Dequantization to FP32 will happen in qMambaLayer if needed

            # ===== SAVE INTERMEDIATE VALUES FOR COMPARISON =====
            if should_save and save_mode == 'floatsim':
                os.makedirs(save_dir, exist_ok=True)

                # Save input (INT8)
                torch.save(x.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_input.pt'))

                # Save output (INT8)
                torch.save(y.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_output_int8.pt'))

                # Save dequantized output for comparison
                y_dequant_fp32 = y.float() * self.output_scale
                torch.save(y_dequant_fp32.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_output_fp32.pt'))

                # Save scale for verification
                scale_info = {
                    'input_scale': self.input_scale.item() if hasattr(self.input_scale, 'item') else self.input_scale,
                    'weight_scale': self.weight_scale.item() if hasattr(self.weight_scale, 'item') else self.weight_scale,
                    'output_scale': self.output_scale.item() if hasattr(self.output_scale, 'item') else self.output_scale,
                    'bias_scale': self.bias_scale.item() if hasattr(self.bias_scale, 'item') else self.bias_scale,
                }
                torch.save(scale_info, os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_scales.pt'))

                print(f"[Float-Sim INT8 Mode] Saved Layer {_CONV1D_LAYER_COUNTER} Conv1D intermediate values to {save_dir}")
                print(f"  - input (INT8)")
                print(f"  - output_int8 (INT8)")
                print(f"  - output_fp32 (FP32, for comparison)")
                print(f"  - scales (output_scale={self.output_scale:.10f})")
            # ===== END SAVE CODE =====

            # Print Layer 24 (counter=23) output and input_scale
            if _CONV1D_LAYER_COUNTER == 23:
                print(f"\n{'='*80}")
                print(f"[Layer 24 / Counter {_CONV1D_LAYER_COUNTER}] Conv1D Output (Mode 2-0/2-1/2-2/2-3)")
                print(f"{'='*80}")
                print(f"  Location: qConvLayer.py forward() - FLOAT_SIM_ASIC_INT8 path")
                print(f"  Conv1D Kernel: CUDA INT8 (same as Mode 0)")
                print(f"  ")
                print(f"  Output:")
                print(f"    dtype: {y.dtype}")
                print(f"    shape: {y.shape}")
                print(f"    range: [{y.min().item()}, {y.max().item()}]")
                print(f"    absmax: {y.abs().max().item()}")
                print(f"    first 5 values [0,0,:5]: {y[0, 0, :5].tolist()}")
                print(f"  ")
                print(f"  Scales:")
                print(f"    input_scale  = {self.input_scale:.10f}  (used by Conv1D CUDA kernel for input)")
                print(f"    output_scale = {self.output_scale:.10f}  (for dequantization in qMambaLayer)")
                print(f"  ")
                print(f"  Next Step:")
                print(f"    → qMambaLayer.py: decides whether to dequantize based on mode")
                print(f"    → Mode 2-0/2-2: dequantize to FP32 for SSM")
                print(f"    → Mode 2-1: keep INT8 for PyTorch INT8 SSM")
                print(f"    → Mode 2-3: keep INT8, but Conv1D output is already FP32 in Mode 2-3")
                print(f"{'='*80}\n")

                # Quick verification mode: exit after printing Layer 24
                if os.environ.get('QUICK_VERIFY', 'false').lower() == 'true':
                    print("🔍 QUICK_VERIFY mode: Exiting after Layer 24 Conv1D output print")
                    import sys
                    sys.exit(0)

            _CONV1D_LAYER_COUNTER += 1
            return y  # INT8 (same as Mode 0)

        # Mode 2-3: True FP32 Conv1D (CUDA kernel with FP32 output)
        # This mode preserves CUDA's internal FP32 precision (skips quantization step)
        if conv1d_mode23_fp32:
            # Use CUDA INT8 kernel but output FP32 (preserve internal precision)
            y_fp32 = quant_causal_conv1d_cuda.fwd_fp32(
                    x, self.input_scale,
                    self.weight, self.weight_scale,
                    self.bias_scale, self.bias,
                    True  # silu_activation=True
                )

            # ===== SAVE INTERMEDIATE VALUES (if needed) =====
            if should_save and save_mode == 'floatsim':
                os.makedirs(save_dir, exist_ok=True)

                torch.save(x.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_input.pt'))
                torch.save(y_fp32.detach().cpu(), os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_conv1d_output_fp32_true.pt'))

                scale_info = {
                    'input_scale': self.input_scale.item() if hasattr(self.input_scale, 'item') else self.input_scale,
                    'weight_scale': self.weight_scale.item() if hasattr(self.weight_scale, 'item') else self.weight_scale,
                    'bias_scale': self.bias_scale.item() if hasattr(self.bias_scale, 'item') else self.bias_scale,
                }
                torch.save(scale_info, os.path.join(save_dir, f'layer{_CONV1D_LAYER_COUNTER}_scales.pt'))

                print(f"[Mode 2-3 FP32] Saved Layer {_CONV1D_LAYER_COUNTER} Conv1D (TRUE FP32 from CUDA)")
            # ===== END SAVE CODE =====

            # ===== MODE 2-3 SCALE VALIDATION =====
            # Check if u_scale (output_scale) matches the actual FP32 output range
            # This is critical because SSM will use u_scale to requantize FP32 → INT8

            # Calculate simple MinMax scale based on actual FP32 output
            y_fp32_absmax = y_fp32.abs().max().item()
            simpleMinMaxScale = y_fp32_absmax / 127.0

            # Get all scales
            input_scale_val = self.input_scale.item() if hasattr(self.input_scale, 'item') else self.input_scale
            output_scale_val = self.output_scale.item() if hasattr(self.output_scale, 'item') else self.output_scale

            # SSM will use output_scale as u_scale (this is what gets used for requantization)
            used_scale = output_scale_val
            scale_name = "output_scale (used as u_scale)"

            # Calculate scale ratio
            scale_ratio = simpleMinMaxScale / used_scale if used_scale > 0 else float('inf')

            # Simulate requantization to check value distribution
            u_requant_simulated = torch.round(y_fp32 / used_scale).clamp(-128, 127)
            unique_values = torch.unique(u_requant_simulated).numel()

            # Print scale info for first layer only (to avoid spam)
            if _CONV1D_LAYER_COUNTER == 0:
                print(f"\n{'='*80}")
                print(f"[Mode 2-3 Scale Validation] Layer {_CONV1D_LAYER_COUNTER}")
                print(f"{'='*80}")
                print(f"  FP32 Output Range: [{y_fp32.min().item():.6f}, {y_fp32.max().item():.6f}]")
                print(f"  FP32 Output AbsMax: {y_fp32_absmax:.6f}")
                print(f"")
                print(f"  Available Scales:")
                print(f"    input_scale          = {input_scale_val:.6f}  (from Conv1D input)")
                print(f"    output_scale         = {output_scale_val:.6f}  (from Conv1D output, calibrated)")
                print(f"    simpleMinMaxScale    = {simpleMinMaxScale:.6f}  (calculated from FP32 output)")
                print(f"")
                print(f"  ✅ SSM will use: {scale_name} = {used_scale:.6f}")
                print(f"  Scale Ratio (simpleMinMax / used): {scale_ratio:.4f}x")
                print(f"")
                print(f"  Requantization Check (using {scale_name}):")
                print(f"    Requantized INT8 range: [{u_requant_simulated.min().item():.0f}, {u_requant_simulated.max().item():.0f}]")
                print(f"    Unique INT8 values used: {unique_values}/256")
                print(f"{'='*80}\n")

            # Error if scale mismatch is severe
            if scale_ratio > 2.0 or scale_ratio < 0.5:
                print(f"\n{'='*80}")
                print(f"❌ ERROR: Severe scale mismatch detected in Mode 2-3!")
                print(f"{'='*80}")
                print(f"  Layer: {_CONV1D_LAYER_COUNTER}")
                print(f"  simpleMinMaxScale: {simpleMinMaxScale:.6f}")
                print(f"  Used Scale:        {used_scale:.6f}")
                print(f"  Scale Ratio:       {scale_ratio:.4f}x")
                print(f"")
                print(f"  This means the FP32 output range doesn't match the calibrated scale!")
                print(f"  SSM will requantize with wrong scale, causing poor performance.")
                print(f"")
                print(f"  Possible causes:")
                print(f"    1. Mode 2-3 FP32 kernel produces different values than Mode 2-2 INT8 kernel")
                print(f"    2. output_scale was calibrated for Mode 2-2, not Mode 2-3")
                print(f"    3. Different numerical precision/rounding in the kernels")
                print(f"")
                print(f"  Solutions:")
                print(f"    1. Use Mode 2-4 (FP32 Conv + FP32 SSM) instead - no requantization needed")
                print(f"    2. Recalibrate scales specifically for Mode 2-3")
                print(f"    3. Use dynamic scaling (calculate simpleMinMaxScale at runtime)")
                print(f"{'='*80}\n")
                raise RuntimeError(f"Mode 2-3 scale mismatch: ratio={scale_ratio:.4f}x (expected ~1.0x)")

            # Warning if scale mismatch is moderate
            if scale_ratio > 1.5 or scale_ratio < 0.67:
                print(f"\n⚠️  Warning: Moderate scale mismatch in Mode 2-3 Layer {_CONV1D_LAYER_COUNTER}")
                print(f"    simpleMinMaxScale: {simpleMinMaxScale:.6f}, Used Scale: {used_scale:.6f}, Ratio: {scale_ratio:.4f}x")
                print(f"    This may cause suboptimal performance.\n")

            # ===== END SCALE VALIDATION =====

            # Print Layer 24 (counter=23) output
            if _CONV1D_LAYER_COUNTER == 23:
                print(f"\n{'='*80}")
                print(f"[Layer 24 / Counter {_CONV1D_LAYER_COUNTER}] Conv1D Output (Mode 2-3)")
                print(f"{'='*80}")
                print(f"  Location: qConvLayer.py forward() - Mode 2-3 path")
                print(f"  Conv1D Kernel: CUDA FP32 kernel (fwd_fp32) → TRUE FP32 output")
                print(f"  ")
                print(f"  Output:")
                print(f"    dtype: {y_fp32.dtype}")
                print(f"    shape: {y_fp32.shape}")
                print(f"    range: [{y_fp32.min().item():.6f}, {y_fp32.max().item():.6f}]")
                print(f"    absmax: {y_fp32.abs().max().item():.6f}")
                print(f"  ")
                print(f"  Scales:")
                print(f"    input_scale  = {self.input_scale:.10f}  (used by Conv1D CUDA FP32 kernel for input)")
                print(f"    output_scale = {self.output_scale:.10f}  (will be used by SSM for requantization to INT8)")
                print(f"  ")
                print(f"  Next Step:")
                print(f"    → qMambaLayer.py: x requantize to INT8 (using output_scale)")
                print(f"    → x_proj (W8A8B8O8Linear): compute dt/B/C (INT8)")
                print(f"    → qSelectiveScan.py: Requantize FP32 u → INT8, then PyTorch INT8 SSM")
                print(f"{'='*80}\n")

            _CONV1D_LAYER_COUNTER += 1
            return y_fp32  # TRUE FP32 (preserves CUDA internal precision)

        # Mode 2-4: True FP32 Conv1D + FP32 SSM (no requantization)
        # This mode preserves complete FP32 precision throughout (Conv1D → SSM)
        if conv1d_mode24_fp32:
            # Use CUDA INT8 kernel but output FP32 (preserve internal precision)
            y_fp32 = quant_causal_conv1d_cuda.fwd_fp32(
                    x, self.input_scale,
                    self.weight, self.weight_scale,
                    self.bias_scale, self.bias,
                    True  # silu_activation=True
                )

            # ===== MODE 2-4 INFO =====
            # No scale validation needed - SSM will use FP32 directly (no requantization)
            if _CONV1D_LAYER_COUNTER == 0:
                print(f"\n{'='*80}")
                print(f"[Mode 2-4 Info] Layer {_CONV1D_LAYER_COUNTER}")
                print(f"{'='*80}")
                print(f"  Conv1D: CUDA FP32 kernel → TRUE FP32 output")
                print(f"  SSM:    Mode 2-2 same FP32 (mode22_fp32_replicates_mode21)")
                print(f"")
                print(f"  FP32 Output Range: [{y_fp32.min().item():.6f}, {y_fp32.max().item():.6f}]")
                print(f"  FP32 Output AbsMax: {y_fp32.abs().max().item():.6f}")
                print(f"")
                print(f"  ✅ Conv1D: TRUE FP32 (vs Mode 2-2 INT8 grid)")
                print(f"{'='*80}\n")

            # Print Layer 24 (counter=23) output
            if _CONV1D_LAYER_COUNTER == 23:
                print(f"\n{'='*80}")
                print(f"[Layer 24 / Counter {_CONV1D_LAYER_COUNTER}] Conv1D Output (Mode 2-4)")
                print(f"{'='*80}")
                print(f"  Location: qConvLayer.py forward() - Mode 2-4 path")
                print(f"  Conv1D Kernel: CUDA FP32 kernel (fwd_fp32) → TRUE FP32 output (same as Mode 2-3)")
                print(f"  ")
                print(f"  Output:")
                print(f"    dtype: {y_fp32.dtype}")
                print(f"    shape: {y_fp32.shape}")
                print(f"    range: [{y_fp32.min().item():.6f}, {y_fp32.max().item():.6f}]")
                print(f"    absmax: {y_fp32.abs().max().item():.6f}")
                print(f"  ")
                print(f"  Scales:")
                print(f"    input_scale  = {self.input_scale:.10f}  (used by Conv1D CUDA FP32 kernel for input)")
                print(f"    output_scale = {self.output_scale:.10f}  (used by qMambaLayer.py for x requantization)")
                print(f"  ")
                print(f"  Next Step:")
                print(f"    → qMambaLayer.py: x requantize to INT8 (using output_scale)")
                print(f"    → x_proj (W8A8B8O8Linear): compute dt/B/C (INT8)")
                print(f"    → qSelectiveScan.py: Mode 2-2 same FP32 SSM (FP32 u, INT8 dt/B/C)")
                print(f"{'='*80}\n")
            # ===== END MODE 2-4 INFO =====

            _CONV1D_LAYER_COUNTER += 1
            return y_fp32  # TRUE FP32 (will be used directly by FP32 SSM)

        # Mode 3: FP32/FP16 input → FP32 Conv1D + FP32 SSM + INT8 Linear (hybrid precision)
        # This mode accepts FP32/FP16 input and uses FP32 for Conv1D and SSM, but keeps Linear layers as INT8
        if conv1d_mode3_fp32:
            # ===== Handle FP32/FP16 input =====
            # If input is FP32/FP16, quantize it to INT8 for CUDA kernel
            if x.dtype in [torch.float32, torch.float16]:
                if _CONV1D_LAYER_COUNTER == 0:
                    print(f"\n{'='*80}")
                    print(f"[Mode 3 Conv1D] Layer {_CONV1D_LAYER_COUNTER}")
                    print(f"{'='*80}")
                    print(f"  Input dtype: {x.dtype} (FP32/FP16)")
                    print(f"  Input range: [{x.min().item():.6f}, {x.max().item():.6f}]")

                # Calculate dynamic scale from FP32 input
                x_absmax = x.abs().max().item()
                x_dynamic_scale = x_absmax / 127.0 if x_absmax > 0 else 1.0

                # Quantize FP32 → INT8
                x_int8 = torch.round(x / x_dynamic_scale).clamp(-128, 127).to(torch.int8)

                if _CONV1D_LAYER_COUNTER == 0:
                    print(f"  Dynamic scale: {x_dynamic_scale:.6f}")
                    print(f"  Quantized to INT8 for CUDA kernel")
                    print(f"{'='*80}\n")

                # Use CUDA FP32 kernel with quantized input
                y_fp32 = quant_causal_conv1d_cuda.fwd_fp32(
                        x_int8, x_dynamic_scale,  # Use dynamic scale
                        self.weight, self.weight_scale,
                        self.bias_scale, self.bias,
                        True  # silu_activation=True
                    )
            else:
                # Input is already INT8 (use original input_scale)
                y_fp32 = quant_causal_conv1d_cuda.fwd_fp32(
                        x, self.input_scale,
                        self.weight, self.weight_scale,
                        self.bias_scale, self.bias,
                        True  # silu_activation=True
                    )

            # ===== MODE 3 INFO =====
            if _CONV1D_LAYER_COUNTER == 0:
                print(f"\n{'='*80}")
                print(f"[Mode 3 Output] Layer {_CONV1D_LAYER_COUNTER}")
                print(f"{'='*80}")
                print(f"  Conv1D: CUDA FP32 kernel → TRUE FP32 output")
                print(f"  SSM:    Will use FP32 directly (no requantization)")
                print(f"  Linear: Keeps INT8 quantization (baseline)")
                print(f"")
                print(f"  FP32 Output Range: [{y_fp32.min().item():.6f}, {y_fp32.max().item():.6f}]")
                print(f"  FP32 Output AbsMax: {y_fp32.abs().max().item():.6f}")
                print(f"")
                print(f"  ✅ Hybrid precision: FP32 for Conv/SSM, INT8 for Linear")
                print(f"{'='*80}\n")
            # ===== END MODE 3 INFO =====

            _CONV1D_LAYER_COUNTER += 1
            return y_fp32  # TRUE FP32 (will be used directly by FP32 SSM)

    def update(self, x, conv_state):
        # update conv_state in-place
        y = quant_causal_conv1d_cuda.update(
            x, conv_state, self.input_scale,
            self.weight, self.weight_scale,
            self.output_scale,
            self.bias_scale, self.bias, True
        ) 
        return y

    # def to(self, *args, **kwargs):
    #     super(QCausalConv1D, self).to(*args, **kwargs)
    #     # Move all parameters to the specified device
    #     for name, param in self.named_parameters():
    #         setattr(self, name, param.to(*args, **kwargs))
    #     # Move all buffers to the specified device
    #     for name, buf in self.named_buffers():
    #         setattr(self, name, buf.to(*args, **kwargs))
    #     return self

    def __repr__(self):
        return f"QCausalConv1D({self.out_channels}, {self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride})"



class Quamb2Conv1D(nn.Module):

    def __init__ (self, x_dim, x_headdim, d_state, n_groups, x_nhead_group, x_ndim_group,
                  in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device=None, **kwargs):
        factory_kwargs = {"device": device}
        super().__init__()
        # x and d_state
        self.x_dim = x_dim
        self.x_headdim = x_headdim
        self.d_state = d_state
        self.n_groups = n_groups
        self.x_nhead_group = x_nhead_group
        self.x_ndim_group = x_ndim_group

        assert in_channels == out_channels == groups, "QCausalConv1D only supports in_channels == out_channels == groups"
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 1, "QCausalConv1D only supports 1D kernel_size"
            kernel_size = kernel_size[0]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.x_scale = 0.0
        self.B_scale = 0.0
        self.C_scale = 0.0
        self.wx_scale = 0.0
        self.wB_scale = 0.0
        self.wC_scale = 0.0
        self.register_buffer('weight', torch.empty(
            (in_channels, kernel_size), dtype=torch.int8, **factory_kwargs))
        if bias is not None:
            self.register_buffer('bias', torch.empty(
                (in_channels), dtype=torch.int8, **factory_kwargs))
            self.bx_scale = 0.0
            self.bB_scale = 0.0
            self.bC_scale = 0.0
        else:
            self.bias = None
            self.bx_scale = None
            self.bB_scale = None
            self.bC_scale = None
        self._register_state_dict_hook(self.store_hook)
        self._register_load_state_dict_pre_hook(self.load_hook)

        if x_nhead_group > 0 and x_ndim_group > 0:
            self.register_buffer('x_head_group_range', torch.empty(
                (n_groups, x_nhead_group), dtype=torch.int32, **factory_kwargs))
            self.register_buffer('x_dim_group_range', torch.empty(
                (n_groups, x_nhead_group, x_ndim_group), dtype=torch.int32, **factory_kwargs))
            self.register_buffer('x_out_scales', torch.empty(
                (n_groups, x_nhead_group, x_ndim_group), dtype=torch.float32, **factory_kwargs))
        elif x_nhead_group == 0 and x_ndim_group == 0:
            self.x_head_group_range = None
            self.x_dim_group_range = None
            self.register_buffer('x_out_scales', torch.empty(
                (1), dtype=torch.float32, **factory_kwargs))
        else:
            raise ValueError("""x_nhead_group and x_ndim_group must be both zero or both non-zero""")
        
        self.register_buffer('B_out_scales', torch.empty(
            (n_groups), dtype=torch.float32, **factory_kwargs))
        self.register_buffer('C_out_scales', torch.empty(
            (n_groups), dtype=torch.float32, **factory_kwargs))

    @classmethod
    def from_fp16(
        cls,
        originalLayer: nn.Conv1d,
        x_dim, x_headdim, d_state, n_groups,
        x_scale, B_scale, C_scale,
        x_out_scales, B_out_scales, C_out_scales,
        x_head_group_range, x_dim_group_range,
    ):

        if x_head_group_range is not None and x_dim_group_range is not None:
            assert len(x_head_group_range.shape) == 2, "x_head_group_range must have shape [n_ssd_group, x_nhead_group]"
            assert len(x_dim_group_range.shape) == 3, "x_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
            x_nhead_group = x_head_group_range.shape[1] # [n_ssd_group, x_nhead_group]
            x_ndim_group = x_dim_group_range.shape[2]   # [n_ssd_group, x_nhead_group, n_dim_group]
        elif x_head_group_range is None and x_dim_group_range is None:
            x_nhead_group = 0
            x_ndim_group = 0 
        else:
            raise ValueError("""x_head_group_range and x_dim_group_range must be both None or both not None""")
    
        device = originalLayer.weight.device
        qconv = cls(x_dim, x_headdim, d_state, n_groups, x_nhead_group, x_ndim_group,
                    originalLayer.in_channels, originalLayer.out_channels,
                    originalLayer.kernel_size, originalLayer.stride,
                    originalLayer.padding, originalLayer.dilation,
                    originalLayer.groups, originalLayer.bias, device=device)

        # input scales for x, B, and C
        qconv.x_scale = x_scale.item()   # float scalar
        qconv.B_scale = B_scale.item()   # float scalar
        qconv.C_scale = C_scale.item()   # float scalar

        # output grouping params
        qconv.x_head_group_range = x_head_group_range.to(device) if x_head_group_range is not None else None # scale tensor must be on cuda
        qconv.x_dim_group_range = x_dim_group_range.to(device) if x_dim_group_range is not None else None    # scale tensor must be on cuda

        # output scales for x, B, and C
        qconv.x_out_scales = x_out_scales.to(device) # scale tensor must be on cuda
        qconv.B_out_scales = B_out_scales.to(device) # scale tensor must be on cuda
        qconv.C_out_scales = C_out_scales.to(device) # scale tensor must be on cuda

        weight = rearrange(originalLayer.weight.clone().detach().to(torch.float32), "d 1 w -> d w")
        d_start = 0
        split_int8_weight = []
        split_weight_scales = []
        for dim in [qconv.x_dim, qconv.d_state*qconv.n_groups, qconv.d_state*qconv.n_groups]:
            d_end = d_start + dim
            w_split = weight[d_start:d_end].contiguous()
            w_split_i8, w_split_scale = quantize_tensor_per_tensor_absmax(
                w_split, n_bits = 8, clip_ratio = 1.0)
            split_int8_weight.append(w_split_i8.to(torch.int8).contiguous())
            split_weight_scales.append(w_split_scale.item())
            d_start = d_end
        cat_int8_weight = torch.cat(split_int8_weight, dim=0).contiguous()
        qconv.weight = cat_int8_weight
        qconv.wx_scale, qconv.wB_scale, qconv.wC_scale = split_weight_scales

        bias = originalLayer.bias.clone().detach().to(torch.float32) if originalLayer.bias is not None else None
        if bias is not None:
            d_start = 0
            split_int8_bias = []
            split_bias_scales = []
            for dim in [qconv.x_dim, qconv.d_state*qconv.n_groups, qconv.d_state*qconv.n_groups]:
                d_end = d_start + dim
                b_split = bias[d_start:d_end].contiguous()
                bias_split_i8, bias_split_scale = quantize_tensor_per_tensor_absmax(
                    b_split, n_bits = 8, clip_ratio = 1.0)
                split_int8_bias.append(bias_split_i8.to(torch.int8).contiguous())
                split_bias_scales.append(bias_split_scale.item())
                d_start = d_end
            cat_int8_bias = torch.cat(split_int8_bias, dim=0).contiguous()
            qconv.bias = cat_int8_bias
            qconv.bx_scale, qconv.bB_scale, qconv.bC_scale = split_bias_scales
        else:
            qconv.bias = None
            qconv.bx_scale, qconv.bB_scale, qconv.bC_scale = None, None, None
        return qconv

    def store_hook(self, module, state_dict, prefix, local_metadata):
        # Define all scales to store
        scale_names = ['x_scale', 'B_scale', 'C_scale', 'wx_scale', 'wB_scale', 'wC_scale']
        # Store each scale dynamically
        for name in scale_names:
            state_dict[prefix + name] = getattr(self, name)
        # Handle bias-related scales if bias exists
        if self.bias is not None:
            for name in ['bx_scale', 'bB_scale', 'bC_scale']:
                state_dict[prefix + name] = getattr(self, name)
        return state_dict
        
    def load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Define all scale names
        scales = ['x_scale', 'B_scale', 'C_scale', 'wx_scale', 'wB_scale', 'wC_scale']
        bias_scales = ['bx_scale', 'bB_scale', 'bC_scale']
        # Load and remove regular scales
        for scale in scales:
            setattr(self, scale, state_dict[prefix + scale])
            del state_dict[prefix + scale]
        # Handle bias-related scales if bias exists
        if self.bias is not None:
            for scale in bias_scales:
                setattr(self, scale, state_dict[prefix + scale])
                del state_dict[prefix + scale]

    @torch.no_grad()
    def forward(self, xBC):
        x, B, C = quamba2_conv1d_cuda.fwd(
                xBC, self.x_scale, self.B_scale, self.C_scale,
                self.x_dim, self.x_headdim, self.d_state, self.n_groups,
                self.x_head_group_range, self.x_dim_group_range,
                self.x_out_scales, self.B_out_scales, self.C_out_scales,
                self.weight, self.wx_scale, self.wB_scale, self.wC_scale,
                self.bias, self.bx_scale, self.bB_scale, self.bC_scale,
                None, None, None, True
            )
        return x, B, C

    @torch.no_grad()
    def update(self, xBC, conv_state):
        # update conv_state in-place
        x, B, C = quamba2_conv1d_cuda.update(
                xBC, conv_state, self.x_scale, self.B_scale, self.C_scale,
                self.x_dim, self.x_headdim, self.d_state, self.n_groups,
                self.x_head_group_range, self.x_dim_group_range,
                self.x_out_scales, self.B_out_scales, self.C_out_scales,
                self.weight, self.wx_scale, self.wB_scale, self.wC_scale,
                self.bias, self.bx_scale, self.bB_scale, self.bC_scale,
                None, None, None, True
        ) 
        return x, B, C

    # def to(self, *args, **kwargs):
    #     super(QCausalConv1D, self).to(*args, **kwargs)
    #     # Move all parameters to the specified device
    #     for name, param in self.named_parameters():
    #         setattr(self, name, param.to(*args, **kwargs))
    #     # Move all buffers to the specified device
    #     for name, buf in self.named_buffers():
    #         setattr(self, name, buf.to(*args, **kwargs))
    #     return self

    def __repr__(self):
        return f"Quamb2Conv1D({self.out_channels}, {self.in_channels}, kernel_size={self.kernel_size}, stride={self.stride})"