#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Verification script to confirm that the Qwen2.5-7B model supports tensor parallelism
across all required mesh shapes: 2x4, 1x8, 1x32, 8x4.
This is a requirement for the bounty.
"""

import os
# Force JAX to use simulated devices for tensor parallelism testing
# We set this here, but it can be overridden by the environment
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"

import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from typing import List, Dict, Optional, Tuple, Set
import re
from tqdm import tqdm

# Import model components
from tensor_parallel import (
    TensorParallelQwen2ForCausalLM,
    create_device_mesh,
)
from config import load_qwen_config, get_qwen2_7b_config
from weight_loading import init_model_from_weights
from verify_gsm8k_scores import setup_tokenizer


def verify_mesh_shape(model_path: str, mesh_shape: Tuple[int, int]):
    """
    Verify that the model works with the given mesh shape.
    
    Args:
        model_path: Path to the model
        mesh_shape: Shape of the device mesh to test
        
    Returns:
        bool: True if verification succeeds, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Verifying mesh shape: {mesh_shape}")
    print(f"{'='*80}")
    
    try:
        # Check if we have enough devices (or if simulation is active)
        required_devices = mesh_shape[0] * mesh_shape[1]
        available_devices = len(jax.devices())
        
        if available_devices < required_devices and "XLA_FLAGS" not in os.environ:
            print(f"⚠️ Not enough devices for mesh shape {mesh_shape} ({available_devices}/{required_devices})")
            print(f"⚠️ Set XLA_FLAGS='--xla_force_host_platform_device_count={required_devices}' to simulate")
            return False
        
        # Create the mesh
        print(f"Creating device mesh with shape {mesh_shape}...")
        mesh = create_device_mesh(mesh_shape)
        print(f"✅ Mesh created successfully")
        
        # Load the configuration
        print(f"Loading model configuration from {model_path}...")
        if os.path.exists(os.path.join(model_path, "config.json")):
            config = load_qwen_config(model_path)
        else:
            config = get_qwen2_7b_config()
        print(f"✅ Configuration loaded")
        
        # Display model information
        print(f"\nModel details:")
        print(f"- Hidden size: {config['hidden_size']}")
        print(f"- Layers: {config['num_hidden_layers']}")
        print(f"- Attention heads: {config['num_attention_heads']}")
        print(f"- KV heads: {config['num_key_value_heads']}")
        print(f"- Vocab size: {config['vocab_size']}")
        
        # Initialize the model
        print(f"\nInitializing tensor-parallel model...")
        start_time = time.time()
        
        try:
            # Load model and weights
            model, params = init_model_from_weights(
                model_class=TensorParallelQwen2ForCausalLM,
                model_path=model_path,
                config=config,
                mesh=mesh,
                param_dtype=jnp.bfloat16
            )
            
            print(f"✅ Model initialized in {time.time() - start_time:.2f} seconds")
            
            # Setup tokenizer
            tokenizer = setup_tokenizer(model_path)
            if tokenizer is None:
                print("❌ Failed to load tokenizer")
                return False
            
            # Run simple inference test
            test_prompt = "What is the capital of France?"
            formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{test_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            input_ids = tokenizer.encode(formatted_prompt, return_tensors="np")
            input_ids = jnp.array(input_ids)
            
            # Ensure input is properly shaped for batch dimension
            if input_ids.shape[0] != mesh_shape[0] and mesh_shape[0] > 1:
                input_ids = jnp.repeat(input_ids, mesh_shape[0], axis=0)
            
            # Create input sharding
            input_sharding = jax.sharding.NamedSharding(mesh, P('batch', None))
            sharded_input = jax.device_put(input_ids, input_sharding)
            
            print(f"\nRunning inference test with prompt: '{test_prompt}'")
            
            # Run forward pass
            with mesh:
                start_time = time.time()
                outputs = model.apply(params, sharded_input)
                elapsed = time.time() - start_time
            
            logits = outputs[0]
            print(f"✅ Forward pass completed in {elapsed:.2f} seconds")
            print(f"✅ Output logits shape: {logits.shape}")
            
            # Get the top predicted token
            last_token_logits = logits[0, -1, :]
            top_tokens = jnp.argsort(last_token_logits, axis=-1)[-5:][::-1]
            
            print(f"\nTop predicted tokens:")
            for i, token_id in enumerate(top_tokens):
                token = tokenizer.decode([token_id])
                print(f"{i+1}. Token {token_id}: '{token}' ({last_token_logits[token_id]:.2f})")
            
            print(f"\n✅ Mesh shape {mesh_shape} verified successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing model with mesh shape {mesh_shape}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"❌ Error testing mesh shape {mesh_shape}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to verify tensor parallel implementation for all mesh shapes."""
    parser = argparse.ArgumentParser(
        description="Verify tensor parallel implementation for Qwen2.5-7B"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model weights (default: ../../../../qwen2.5-7b)"
    )
    
    parser.add_argument(
        "--mesh_shapes",
        type=str,
        default="2x4,1x8,1x32,8x4",
        help="Comma-separated list of mesh shapes to test (e.g., '2x4,1x8')"
    )
    
    args = parser.parse_args()
    
    # Resolve model path
    model_path = args.model_path
    if model_path is None:
        # Look for model in default location (relative to this script)
        default_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "../../../../qwen2.5-7b"
        ))
        
        if os.path.exists(default_path):
            model_path = default_path
            print(f"Using default model path: {model_path}")
        else:
            print(f"Error: Default model path not found: {default_path}")
            print("Please specify a model path with --model_path")
            return
    
    # Verify model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist")
        return
    
    # Parse mesh shapes
    mesh_shapes = []
    for shape_str in args.mesh_shapes.split(","):
        try:
            x, y = map(int, shape_str.split("x"))
            mesh_shapes.append((x, y))
        except ValueError:
            print(f"Error: Invalid mesh shape format: {shape_str}")
            print("Mesh shapes should be in format AxB (e.g., '2x4')")
            return
    
    # Display information
    print(f"Verifying tensor parallel implementation for Qwen2.5-7B")
    print(f"Model path: {model_path}")
    print(f"Mesh shapes to test: {mesh_shapes}")
    print(f"JAX devices available: {len(jax.devices())}")
    
    # If not enough devices and XLA_FLAGS not set, print warning
    max_devices = max([x*y for x, y in mesh_shapes])
    if len(jax.devices()) < max_devices and "XLA_FLAGS" not in os.environ:
        print(f"\n⚠️ Warning: Not enough devices for all mesh shapes")
        print(f"Largest mesh requires {max_devices} devices, but only {len(jax.devices())} available")
        print(f"Set XLA_FLAGS='--xla_force_host_platform_device_count={max_devices}' to simulate more devices")
    
    # Verify each mesh shape
    results = {}
    for mesh_shape in mesh_shapes:
        success = verify_mesh_shape(model_path, mesh_shape)
        results[f"{mesh_shape[0]}x{mesh_shape[1]}"] = success
    
    # Print summary
    print("\n" + "="*80)
    print("Tensor Parallel Verification Results:")
    print("="*80)
    
    all_passed = True
    for shape, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"Mesh shape {shape}: {status}")
        all_passed = all_passed and success
    
    if all_passed:
        print("\n✅ All mesh shapes verified successfully.")
        print("This model implementation meets the tensor parallelism requirements for the bounty.")
    else:
        print("\n❌ Some mesh shapes failed verification.")
        print("Please fix the issues before submitting for the bounty.")


if __name__ == "__main__":
    main() 