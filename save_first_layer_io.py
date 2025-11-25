#!/usr/bin/env python3
"""
保存所有modes的第一层输入输出到单个文件
并打印前10个值、mean、std
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from utils import build_mamba_and_tokenizer, set_deterministic
from quamba.modelutils_mamba import quantize_model_mamba
from quamba.mode_config import setup_quamba_mode
from datasets import load_dataset


class FirstLayerIOHook:
    """Hook to capture first layer input and output"""
    def __init__(self):
        self.input = None
        self.output = None
        self.hook = None

    def register_hook(self, module):
        """Register forward hook for the first layer"""
        def hook_fn(module, input, output):
            # Store input (detach and move to CPU)
            if isinstance(input, tuple):
                self.input = input[0].detach().cpu().float()
            else:
                self.input = input.detach().cpu().float()

            # Store output (detach and move to CPU)
            if isinstance(output, tuple):
                self.output = output[0].detach().cpu().float()
            else:
                self.output = output.detach().cpu().float()

        self.hook = module.register_forward_hook(hook_fn)
        return self.hook

    def remove_hook(self):
        """Remove the registered hook"""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def get_io(self):
        """Get input and output"""
        return self.input, self.output

    def clear(self):
        """Clear stored data"""
        self.input = None
        self.output = None


def get_test_sample(tokenizer, seq_len=512):
    """Get a single test sample"""
    dataset = load_dataset("c4", "en", split="train", streaming=True, trust_remote_code=True)
    dataset_iter = iter(dataset)

    data = next(dataset_iter)
    text = data["text"]

    # Tokenize
    if hasattr(tokenizer, 'encode'):  # For custom tokenizer
        tokens = tokenizer.encode(text)
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        tokens = torch.tensor(tokens, dtype=torch.long)
    else:  # For HuggingFace tokenizer
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True
        ).input_ids[0]

    return tokens


def save_first_layer_io(mode, model, tokenizer, output_dict, device="cuda", seq_len=512):
    """
    Run model and save first layer input/output

    Args:
        mode: Mode identifier
        model: The model to run
        tokenizer: Tokenizer
        output_dict: Dictionary to store results
        device: Device to run on
        seq_len: Sequence length
    """
    model.eval()
    model.to(device)

    # Setup hook for first layer
    first_layer = model.backbone.layers[0]
    hook = FirstLayerIOHook()
    hook.register_hook(first_layer)

    # Get test sample
    tokens = get_test_sample(tokenizer, seq_len)
    tokens = tokens.unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    print(f"  Running mode {mode}...")
    with torch.no_grad():
        _ = model(tokens)

    # Get input/output
    layer_input, layer_output = hook.get_io()

    if layer_input is None or layer_output is None:
        print(f"  Warning: Failed to capture I/O for mode {mode}")
        hook.remove_hook()
        return

    # Convert to numpy
    layer_input = layer_input.numpy()
    layer_output = layer_output.numpy()

    # Store results
    output_dict[mode] = {
        'input': {
            'data': layer_input,
            'shape': list(layer_input.shape),
            'first_10': layer_input.flatten()[:10].tolist(),
            'mean': float(np.mean(layer_input)),
            'std': float(np.std(layer_input)),
            'min': float(np.min(layer_input)),
            'max': float(np.max(layer_input))
        },
        'output': {
            'data': layer_output,
            'shape': list(layer_output.shape),
            'first_10': layer_output.flatten()[:10].tolist(),
            'mean': float(np.mean(layer_output)),
            'std': float(np.std(layer_output)),
            'min': float(np.min(layer_output)),
            'max': float(np.max(layer_output))
        }
    }

    print(f"  ✓ Mode {mode} captured successfully")

    # Remove hook
    hook.remove_hook()


def print_summary(results):
    """Print summary of all modes"""
    print("\n" + "="*100)
    print("FIRST LAYER INPUT/OUTPUT SUMMARY")
    print("="*100)

    for mode in sorted(results.keys()):
        data = results[mode]

        print(f"\n{'='*100}")
        print(f"MODE {mode}")
        print(f"{'='*100}")

        # Input
        print(f"\n📥 INPUT (Shape: {data['input']['shape']}):")
        print(f"  First 10 values: {[f'{v:.6f}' for v in data['input']['first_10']]}")
        print(f"  Mean: {data['input']['mean']:.6f}")
        print(f"  Std:  {data['input']['std']:.6f}")
        print(f"  Range: [{data['input']['min']:.6f}, {data['input']['max']:.6f}]")

        # Output
        print(f"\n📤 OUTPUT (Shape: {data['output']['shape']}):")
        print(f"  First 10 values: {[f'{v:.6f}' for v in data['output']['first_10']]}")
        print(f"  Mean: {data['output']['mean']:.6f}")
        print(f"  Std:  {data['output']['std']:.6f}")
        print(f"  Range: [{data['output']['min']:.6f}, {data['output']['max']:.6f}]")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Save first layer I/O for all modes")
    parser.add_argument('--model', type=str, default='quamba-130m-w8a8',
                        help='Model name (default: quamba-130m-w8a8)')
    parser.add_argument('--pretrained_dir', type=str,
                        default='pretrained_models/Quamba1-pa9999/pa-0.9999',
                        help='Path to pretrained model')
    parser.add_argument('--mamba_model_path', type=str,
                        default='pretrained_models/mambaOriginalHuggingfaceDownload',
                        help='Path to original Mamba FP16 model for mode 1 (default: pretrained_models/mambaOriginalHuggingfaceDownload)')
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['1', 'fp32', '0', '2-0', '2-1', '2-2', '2-3', '2-4', '3'],
                        help='Modes to test (default: all modes including mode 1)')
    parser.add_argument('--output_file', type=str, default='first_layer_io_all_modes.npz',
                        help='Output file path (default: first_layer_io_all_modes.npz)')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Sequence length (default: 512)')
    parser.add_argument('--calib_data_num', type=int, default=512)
    parser.add_argument('--calib_seqlen', type=int, default=512)

    args = parser.parse_args()

    # Setup deterministic behavior
    set_deterministic(1234)

    print("\n" + "="*100)
    print("FIRST LAYER INPUT/OUTPUT CAPTURE")
    print("="*100)
    print(f"\nModel: {args.model}")
    print(f"Pretrained dir: {args.pretrained_dir}")
    print(f"Modes: {args.modes}")
    print(f"Output file: {args.output_file}")
    print(f"Sequence length: {args.seq_len}")
    print("="*100 + "\n")

    model_name = args.model.lower().split('/')[-1]
    model_type = model_name.split('-')[0]

    results = {}

    for mode in args.modes:
        print(f"\n{'#'*100}")
        print(f"# Processing Mode: {mode}")
        print(f"{'#'*100}")

        try:
            # Setup mode
            if mode == '1':
                # Mode 1: Original Mamba FP16 model
                print(f"  Mode 1: Using original Mamba FP16 model from {args.mamba_model_path}")
                use_mamba = True
                use_quantize = False
            elif mode in ['fp16', 'fp32']:
                # FP modes: Quamba model without quantization
                use_mamba = False
                use_quantize = False
            else:
                # Quantized modes
                setup_quamba_mode(mode, verbose=True)
                use_mamba = False
                use_quantize = True

            # Build model
            print(f"\nLoading model...")

            # Create a minimal args object for model loading
            class ModelArgs:
                def __init__(self):
                    if use_mamba:
                        # Mode 1: Original Mamba
                        self.model = args.mamba_model_path
                        self.pretrained_dir = None
                    else:
                        # Quamba models
                        self.model = args.model
                        self.pretrained_dir = args.pretrained_dir
                    self.quantize = use_quantize
                    self.calib_data_num = args.calib_data_num
                    self.calib_seqlen = args.calib_seqlen
                    self.w_bits = 8
                    self.a_bits = 8

            model_args = ModelArgs()

            # Determine model type
            if use_mamba:
                model_type = 'mamba'  # Original Mamba
            else:
                model_type = model_name.split('-')[0]  # Quamba

            model, tokenizer, is_quamba = build_mamba_and_tokenizer(model_args, model_type)
            model.config.use_cache = False

            # Quantize if needed
            if model_args.quantize and not is_quamba:
                print("Quantizing model...")
                model = quantize_model_mamba(model, model_type, tokenizer, "cuda", model_args)

            # Save first layer I/O
            save_first_layer_io(mode, model, tokenizer, results, seq_len=args.seq_len)

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ✗ Error processing mode {mode}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print_summary(results)

    # Save results
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)

    # Prepare data for saving
    save_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': args.model,
        'pretrained_dir': args.pretrained_dir,
        'seq_len': args.seq_len
    }

    # Save full numpy arrays
    for mode, data in results.items():
        save_data[f'mode_{mode}_input'] = data['input']['data']
        save_data[f'mode_{mode}_output'] = data['output']['data']

    # Save as npz (compressed numpy format)
    np.savez_compressed(args.output_file, **save_data)
    print(f"\n✓ Full data saved to: {args.output_file}")

    # Save statistics as JSON
    stats_file = args.output_file.replace('.npz', '_stats.json')
    stats = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': args.model,
        'pretrained_dir': args.pretrained_dir,
        'seq_len': args.seq_len,
        'modes': {}
    }

    for mode, data in results.items():
        stats['modes'][mode] = {
            'input': {k: v for k, v in data['input'].items() if k != 'data'},
            'output': {k: v for k, v in data['output'].items() if k != 'data'}
        }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Statistics saved to: {stats_file}")
    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    main()
