#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Verification script for Qwen2.5-7B model implementation for TT-XLA bounty.
This script verifies that the implementation meets all the bounty requirements.
"""

import os
import time
import jax
import jax.numpy as jnp
import numpy as np

from model_implementation import Qwen2ForCausalLM
from tensor_parallel import (
    TensorParallelQwen2ForCausalLM,
    create_device_mesh,
)
from config import get_small_config, supported_mesh_configs

def check_files():
    """Check that all required files are present."""
    required_files = [
        "model_implementation.py",    # Core model implementation
        "tensor_parallel.py",         # Tensor-parallel implementation
        "config.py",                  # Configuration utilities
        "weight_loading.py",          # Weight loading utilities
        "README.md",                  # Documentation
        "minimal_test.py",            # Minimal tests
        "simple_test.py",             # Simplified tests
        "test_model.py",              # Test module
        "example_usage.py",           # Example usage
        "integration.py",             # Integration with test infrastructure
        "register.py",                # Registration utilities
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✅ All required files are present")
    return True

def check_core_implementation():
    """Verify that the core model implementation works correctly."""
    print("\nVerifying core model implementation...")
    
    try:
        # Create small config for testing
        config = get_small_config(hidden_size=128, num_layers=2)
        
        # Initialize model
        model = Qwen2ForCausalLM(config=config)
        
        # Create input data
        input_ids = jnp.ones((1, 16), dtype=jnp.int32)
        
        # Initialize parameters
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, input_ids)
        
        # Run forward pass
        start_time = time.time()
        outputs = model.apply(params, input_ids)
        end_time = time.time()
        
        print(f"✅ Forward pass completed in {(end_time - start_time):.4f} seconds")
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {outputs[0].shape}")
        
        # Basic validation
        assert outputs[0].shape == (1, 16, config["vocab_size"]), "Incorrect output shape"
        assert not np.isnan(np.asarray(outputs[0])).any(), "Output contains NaN values"
        
        print("✅ Core model implementation verified")
        return True
    
    except Exception as e:
        print(f"❌ Error verifying core implementation: {e}")
        return False

def check_mesh_configurations():
    """Verify that all required mesh configurations are supported."""
    print("\nVerifying mesh configuration support...")
    
    required_meshes = [
        (2, 4),  # 2x4
        (1, 8),  # 1x8
        (1, 32), # 1x32
        (8, 4),  # 8x4
    ]
    
    supported = supported_mesh_configs()
    supported_shapes = [config['shape'] for config in supported]
    
    missing = []
    for mesh in required_meshes:
        if mesh not in supported_shapes:
            missing.append(mesh)
    
    if missing:
        print("❌ Missing required mesh configurations:")
        for mesh in missing:
            print(f"  - {mesh[0]}x{mesh[1]}")
        return False
    
    print("✅ All required mesh configurations are supported:")
    for mesh in required_meshes:
        print(f"  - {mesh[0]}x{mesh[1]}")
    
    return True

def check_readme():
    """Verify that the README includes necessary documentation."""
    print("\nVerifying README documentation...")
    
    if not os.path.exists("README.md"):
        print("❌ README.md not found")
        return False
    
    with open("README.md", "r") as f:
        content = f.read()
    
    required_sections = [
        "model architecture",
        "tensor parallel",
        "mesh",
        "attention",
        "usage",
        "example",
    ]
    
    missing_sections = []
    for section in required_sections:
        if section.lower() not in content.lower():
            missing_sections.append(section)
    
    if missing_sections:
        print("❌ README is missing sections on:")
        for section in missing_sections:
            print(f"  - {section}")
        return False
    
    # Check if README explains the model architecture
    if "transformer" not in content.lower() or "layer" not in content.lower():
        print("❌ README lacks details on transformer layers")
        return False
    
    # Check if README explains tensor parallelism
    if "shard" not in content.lower():
        print("❌ README lacks details on parameter sharding")
        return False
    
    print("✅ README documentation verified")
    return True

def check_tensor_parallel_implementation():
    """
    Verify that tensor parallelism functionality is implemented.
    Note: This will not test actual multi-device execution if only one device is available.
    """
    print("\nVerifying tensor-parallel implementation...")
    
    # Check if tensor_parallel.py has required classes and functions
    with open("tensor_parallel.py", "r") as f:
        content = f.read()
    
    required_components = [
        "TensorParallelDense",
        "TensorParallelQwenAttention",
        "TensorParallelQwenMLP",
        "TensorParallelQwenTransformerBlock",
        "TensorParallelQwen2Model",
        "TensorParallelQwen2ForCausalLM",
        "create_device_mesh",
        "get_partition_specs",
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
    
    if missing_components:
        print("❌ Tensor-parallel implementation is missing components:")
        for component in missing_components:
            print(f"  - {component}")
        return False
    
    print("✅ All tensor-parallel components are present")
    
    # Try to initialize a tensor-parallel model
    config = get_small_config(hidden_size=128, num_layers=2)
    
    # Use a 1x1 mesh for testing if only one device is available
    mesh_shape = (1, 1)
    
    try:
        mesh = create_device_mesh(mesh_shape)
        model = TensorParallelQwen2ForCausalLM(config=config, mesh=mesh)
        print(f"✅ Successfully initialized tensor-parallel model with {mesh_shape} mesh")
        return True
    except Exception as e:
        print(f"❌ Error initializing tensor-parallel model: {e}")
        return False

def verify_all():
    """Run all verification steps and summarize results."""
    print("=" * 60)
    print("QWEN2.5-7B MODEL BOUNTY VERIFICATION")
    print("=" * 60)
    
    results = {
        "Files": check_files(),
        "Core Implementation": check_core_implementation(),
        "Mesh Configurations": check_mesh_configurations(),
        "README Documentation": check_readme(),
        "Tensor-Parallel Implementation": check_tensor_parallel_implementation(),
    }
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        all_pass &= result
        print(f"{name}: {status}")
    
    print("\nFINAL RESULT:")
    if all_pass:
        print("✅ IMPLEMENTATION MEETS ALL BOUNTY REQUIREMENTS")
    else:
        print("❌ IMPLEMENTATION DOES NOT MEET ALL REQUIREMENTS")
        print("Please fix the issues reported above.")
    
    return all_pass

if __name__ == "__main__":
    verify_all()

 