#!/usr/bin/env python3
"""
保存模型各层输出的脚本
支持FP16/FP32和所有量化mode，保存第1层和第24层（最后一层）的输出
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path

from utils import build_mamba_and_tokenizer, set_deterministic, parse_options
from quamba.modelutils_mamba import quantize_model_mamba
from quamba.mode_config import setup_quamba_mode
from datasets import load_dataset


class LayerOutputHook:
    """Hook to capture detailed layer outputs with monkey patching"""
    def __init__(self):
        self.outputs = {}
        self.original_forwards = {}

    def wrap_mixer_forward(self, mixer, layer_idx):
        """Wrap mixer's forward to capture intermediate values"""
        original_forward = mixer.forward
        self.original_forwards[layer_idx] = original_forward

        def wrapped_forward(hidden_states, inference_params=None):
            # Capture input
            self.outputs[f"layer_{layer_idx}_mixer_input"] = hidden_states.detach().cpu().float()

            # Call original forward and capture intermediate steps
            # We'll monkey-patch the conv1d and selective_scan calls
            import types

            # Save original conv1d
            original_conv1d = mixer.conv1d.forward if hasattr(mixer, 'conv1d') else None

            # Wrap conv1d
            if original_conv1d is not None:
                def wrapped_conv1d(x):
                    conv_input = x.detach().cpu().float()
                    conv_output = original_conv1d(x)
                    self.outputs[f"layer_{layer_idx}_conv1d_input"] = conv_input
                    self.outputs[f"layer_{layer_idx}_conv1d_output"] = conv_output.detach().cpu().float()
                    return conv_output
                mixer.conv1d.forward = wrapped_conv1d

            # Save original selective_scan if it exists as a module
            original_ssm = None
            ssm_attr_name = None
            for attr_name in ['selective_scan', 'qsscan', 'QSScan']:
                if hasattr(mixer, attr_name):
                    ssm_module = getattr(mixer, attr_name)
                    if hasattr(ssm_module, 'forward'):
                        original_ssm = ssm_module.forward
                        ssm_attr_name = attr_name
                        break

            # Wrap selective_scan
            if original_ssm is not None:
                def wrapped_ssm(*args, **kwargs):
                    # Capture SSM input (first arg is usually u)
                    if len(args) > 0:
                        ssm_input = args[0].detach().cpu().float()
                        self.outputs[f"layer_{layer_idx}_ssm_input"] = ssm_input

                    ssm_output = original_ssm(*args, **kwargs)

                    # Capture SSM output
                    if isinstance(ssm_output, tuple):
                        self.outputs[f"layer_{layer_idx}_ssm_output"] = ssm_output[0].detach().cpu().float()
                    else:
                        self.outputs[f"layer_{layer_idx}_ssm_output"] = ssm_output.detach().cpu().float()

                    return ssm_output

                getattr(mixer, ssm_attr_name).forward = wrapped_ssm

            # Call original forward
            try:
                output = original_forward(hidden_states, inference_params)
            finally:
                # Restore original methods
                if original_conv1d is not None:
                    mixer.conv1d.forward = original_conv1d
                if original_ssm is not None:
                    getattr(mixer, ssm_attr_name).forward = original_ssm

            # Capture output
            if isinstance(output, tuple):
                self.outputs[f"layer_{layer_idx}_mixer_output"] = output[0].detach().cpu().float()
            else:
                self.outputs[f"layer_{layer_idx}_mixer_output"] = output.detach().cpu().float()

            return output

        mixer.forward = wrapped_forward

    def restore_all(self):
        """Restore all original forward methods"""
        # This is handled in the wrapped_forward function
        pass

    def get_output(self, key):
        """Get output for a specific key"""
        return self.outputs.get(key, None)

    def clear(self):
        """Clear stored outputs"""
        self.outputs.clear()


def get_calibration_data(tokenizer, calib_data_num=10, calib_seqlen=512):
    """Load calibration data for testing"""
    print(f"Loading calibration data: {calib_data_num} samples, max_length={calib_seqlen}")

    dataset = load_dataset("c4", "en", split="train", streaming=True)
    dataset_iter = iter(dataset)

    samples = []
    for _ in range(calib_data_num):
        data = next(dataset_iter)
        text = data["text"]

        # Tokenize
        if hasattr(tokenizer, 'encode'):  # For custom tokenizer
            tokens = tokenizer.encode(text)
            if len(tokens) > calib_seqlen:
                tokens = tokens[:calib_seqlen]
        else:  # For HuggingFace tokenizer
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=calib_seqlen,
                truncation=True
            ).input_ids[0]

        samples.append(tokens)

    return samples


def save_layer_outputs(model, tokenizer, mode, output_dir, calib_data_num=10, calib_seqlen=512, device="cuda"):
    """
    Run model and save layer outputs

    Args:
        model: The model to run
        tokenizer: Tokenizer
        mode: Mode identifier (e.g., 'fp16', 'fp32', '0', '2-1', etc.)
        output_dir: Directory to save outputs
        calib_data_num: Number of samples to process
        calib_seqlen: Maximum sequence length
        device: Device to run on
    """
    model.eval()
    model.to(device)

    # Determine which layers to hook
    # For Quamba-130M: 24 layers (0-23), we want layer 0 and layer 23 (last)
    n_layers = len(model.backbone.layers)
    print(f"Model has {n_layers} layers")

    target_layers = [0, n_layers - 1]  # First and last layer
    print(f"Capturing outputs from layers: {target_layers}")

    # Setup hooks for mixer (which contains conv1d and ssm)
    hook = LayerOutputHook()

    print("\nWrapping mixer forward methods:")
    for layer_idx in target_layers:
        layer = model.backbone.layers[layer_idx]

        # Wrap the mixer module's forward
        if hasattr(layer, 'mixer'):
            hook.wrap_mixer_forward(layer.mixer, layer_idx)
            print(f"  ✓ Wrapped layer {layer_idx} mixer ({type(layer.mixer).__name__})")
        else:
            print(f"  ✗ Layer {layer_idx} has no mixer")

    # Get calibration data
    samples = get_calibration_data(tokenizer, calib_data_num, calib_seqlen)

    # Storage for all outputs - tracking full flow: mixer_input → conv1d → ssm → mixer_output
    component_keys = []
    for layer_idx in target_layers:
        component_keys.extend([
            f"layer_{layer_idx}_mixer_input",
            f"layer_{layer_idx}_conv1d_input",
            f"layer_{layer_idx}_conv1d_output",
            f"layer_{layer_idx}_ssm_input",
            f"layer_{layer_idx}_ssm_output",
            f"layer_{layer_idx}_mixer_output"
        ])
    all_outputs = {key: [] for key in component_keys}

    # Run model on each sample
    print(f"\nProcessing {len(samples)} samples...")
    with torch.no_grad():
        for idx, tokens in enumerate(samples):
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens, dtype=torch.long)

            tokens = tokens.unsqueeze(0).to(device)  # Add batch dimension

            # Forward pass
            hook.clear()
            try:
                _ = model(tokens)

                # Collect outputs for all components
                for key in component_keys:
                    output = hook.get_output(key)
                    if output is not None:
                        all_outputs[key].append(output.numpy())

                if (idx + 1) % 5 == 0:
                    print(f"  Processed {idx + 1}/{len(samples)} samples")
            except Exception as e:
                print(f"  Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Remove hooks
    hook.remove_hooks()

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    stats = {}

    # Process each layer's complete flow
    for layer_idx in target_layers:
        layer_stats = {}

        # Process all components in order
        for component in ['mixer_input', 'conv1d_input', 'conv1d_output', 'ssm_input', 'ssm_output', 'mixer_output']:
            key = f"layer_{layer_idx}_{component}"
            outputs = all_outputs.get(key, [])

            if not outputs:
                print(f"Warning: No data collected for {key}")
                continue

            # Concatenate all samples along batch dimension
            outputs_array = np.concatenate(outputs, axis=0)

            # Save raw outputs
            output_file = os.path.join(output_dir, f"mode_{mode}_{key}.npy")
            np.save(output_file, outputs_array)
            print(f"Saved {key}: {output_file} (shape: {outputs_array.shape})")

            # Get original dtype before conversion to float
            original_dtype = str(outputs[0].dtype) if outputs else "unknown"

            # Compute statistics
            layer_stats[component] = {
                "shape": list(outputs_array.shape),
                "dtype": original_dtype,
                "mean": float(np.mean(outputs_array)),
                "std": float(np.std(outputs_array)),
                "min": float(np.min(outputs_array)),
                "max": float(np.max(outputs_array)),
                "abs_mean": float(np.mean(np.abs(outputs_array)))
            }

        stats[f"layer_{layer_idx}"] = layer_stats

    # Add timestamp and metadata
    from datetime import datetime
    stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats['mode'] = mode
    stats['num_samples'] = calib_data_num
    stats['seq_length'] = calib_seqlen
    # Add model dtype info
    model_dtype = str(next(model.parameters()).dtype)
    stats['model_dtype'] = model_dtype

    # Save statistics
    stats_file = os.path.join(output_dir, f"mode_{mode}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics: {stats_file}")

    return stats


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Save layer outputs for different modes")
    parser.add_argument('model', type=str, help='Model name or path (e.g., quamba-130m-w8a8 or pretrained_models/mamba-130m)')
    parser.add_argument('--pretrained_dir', type=str, default=None, help='Path to pretrained model (required for Quamba models)')
    parser.add_argument('--mode', type=str, default='0',
                        choices=['fp16', 'fp32', '0', '2-0', '2-1', '2-2', '2-3', '2-4', '3'],
                        help='Mode to run (fp16/fp32 for floating point, or quantized mode)')
    parser.add_argument('--mode_name', type=str, default=None,
                        help='Custom mode name for saving (e.g., "1" for original Mamba)')
    parser.add_argument('--output_dir', type=str, default='layer_outputs',
                        help='Directory to save outputs (default: layer_outputs)')
    parser.add_argument('--calib_data_num', type=int, default=10,
                        help='Number of calibration samples (default: 10)')
    parser.add_argument('--calib_seqlen', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--quantize', action='store_true', default=False,
                        help='Use quantized model (required for mode 0-3)')
    parser.add_argument('--w_bits', type=int, default=8,
                        help='Weight bits for quantization (default: 8)')
    parser.add_argument('--a_bits', type=int, default=8,
                        help='Activation bits for quantization (default: 8)')
    parser.add_argument('--group_heads', action='store_true', default=False,
                        help='Whether to group heads during reordering (default: False)')
    parser.add_argument('--apply_gptq', action='store_true', default=False,
                        help='Whether to apply GPTQ quantizer (default: False)')
    parser.add_argument('--hybrid_blocks', action='store_true', default=False,
                        help='Whether to create hybrid blocks (default: False)')
    parser.add_argument('--hybrid_blocks_config', type=str, default=None,
                        help='Path to hybrid blocks config (default: None)')
    parser.add_argument('--percentile_alpha', type=float, default=None,
                        help='Percentile alpha for calibration (default: None)')

    args = parser.parse_args()

    # Setup deterministic behavior
    set_deterministic(1234)

    # Setup mode if quantized
    # 如果使用 --mode_name 但没有 --quantize，说明是 FP 模式（如 mode 1）
    if args.mode not in ['fp16', 'fp32'] and not args.mode_name:
        if not args.quantize:
            print("Warning: mode is set but --quantize is not enabled. Enabling --quantize.")
            args.quantize = True
        setup_quamba_mode(args.mode, verbose=True)
    elif args.quantize:
        # 如果明确指定了 --quantize，则设置 mode
        setup_quamba_mode(args.mode, verbose=True)
    else:
        # FP32/FP16模式：不设置量化kernel环境变量
        print(f"\n{'='*80}")
        print(f"Running in {args.mode.upper()} mode")
        print(f"{'='*80}")
        print(f"  Quamba model weights: INT8")
        print(f"  Conv1D/SSM computation: {args.mode.upper()} (no quantization kernel)")
        print(f"  No quantization environment variables set")
        print(f"{'='*80}\n")

    # Build model
    model_name = args.model.lower().split('/')[-1]
    model_type = model_name.split('-')[0]
    print(f"\nLoading model: {model_name} (type: {model_type})")

    model, tokenizer, is_quamba = build_mamba_and_tokenizer(args, model_type)
    model.config.use_cache = False

    # Quantize if needed
    if args.quantize and not is_quamba:
        print("Quantizing model...")
        model = quantize_model_mamba(model, model_type, tokenizer, "cuda", args)
    else:
        # 显示使用的精度
        dtype = next(model.parameters()).dtype
        print(f"Using floating point model with dtype: {dtype}")

    # Determine mode name for saving
    if args.mode_name:
        # Use custom mode name if provided (e.g., "1" for original Mamba)
        mode_name = args.mode_name
    else:
        mode_name = args.mode
        if args.mode == 'fp16' and not args.quantize:
            mode_name = 'fp16'
        elif args.mode == 'fp32' and not args.quantize:
            mode_name = 'fp32'

    print(f"\n{'='*80}")
    print(f"Running in mode: {mode_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    # Save layer outputs
    stats = save_layer_outputs(
        model,
        tokenizer,
        mode_name,
        args.output_dir,
        calib_data_num=args.calib_data_num,
        calib_seqlen=args.calib_seqlen
    )

    print(f"\n{'='*80}")
    print("Summary:")
    print(json.dumps(stats, indent=2))
    print(f"{'='*80}\n")
    print(f"Done! Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
