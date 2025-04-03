# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for loading Qwen2.5-7B weights from HuggingFace safetensors format.
"""

import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from typing import Dict, List, Optional, Tuple, Any, Union

from tensor_parallel import get_partition_specs

def load_safetensors_index(model_path: str) -> Dict[str, str]:
    """
    Load the safetensors index file for a model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary mapping weight names to their safetensors file
    """
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Safetensors index file not found at {index_file}")
        
    with open(index_file, "r") as f:
        index_data = json.load(f)
        
    # Create mapping from parameter to filename
    param_to_file = {}
    for filename, file_info in index_data["weight_map"].items():
        for param_name in file_info:
            param_to_file[param_name] = os.path.join(model_path, filename)
            
    return param_to_file

def convert_weight_name_to_flax(name: str) -> str:
    """
    Convert transformer weight names to Flax format.
    
    Args:
        name: Parameter name in HuggingFace format
        
    Returns:
        Parameter name in Flax format
    """
    # Remove model prefix if present
    name = name.replace("model.", "")
    
    # Replace "." with "/" for Flax nested dict format
    name = name.replace(".", "/")
    
    # Handle self attention naming
    name = name.replace("self_attn/", "self_attn/")
    
    # Handle layer indices
    if "layers/" in name:
        layer_idx = name.split("layers/")[1].split("/")[0]
        name = name.replace(f"layers/{layer_idx}", f"layers_{layer_idx}")
    
    # Handle special embeddings case
    if "embedding" in name and "token" in name:
        name = name.replace("token_embedding/embedding", "embed_tokens/embedding")
    
    # Handle specific cases for final norm and LM head
    if "norm/" in name:
        name = name.replace("norm/", "norm/")
    
    # Handle the language model head weights
    if "lm_head/" in name:
        name = name.replace("lm_head/", "lm_head/")
        
    return name

def load_qwen_weights(
    model_path: str,
    config: Dict[str, Any],
    mesh: Optional[jax.sharding.Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
) -> Dict:
    """
    Load weights from safetensors files with tensor parallelism.
    
    Args:
        model_path: Path to model safetensors files
        config: Model configuration dictionary
        mesh: Optional JAX mesh for tensor parallelism
        param_dtype: Data type for parameters
        
    Returns:
        Dictionary of model parameters
    """
    # Load safetensors index mapping
    param_file_map = load_safetensors_index(model_path)
    
    # Get the list of all parameter names
    param_names = list(param_file_map.keys())
    
    # Create partition specs for tensor parallelism
    if mesh is not None:
        partition_specs = get_partition_specs(config)
    else:
        partition_specs = None
    
    # Initialize the parameter dictionary
    params = {}
    
    # Map of tensors to load directly from files
    file_handles = {}
    
    # Track which parameters we've loaded
    loaded_params = set()
    
    # Load weights from each file
    for name in param_names:
        file_path = param_file_map[name]
        
        # Create file handle if we don't have one yet
        if file_path not in file_handles:
            file_handles[file_path] = safe_open(file_path, framework="numpy")
            
        # Get the tensor from the file
        try:
            tensor = file_handles[file_path].get_tensor(name)
        except:
            print(f"Error loading tensor {name} from {file_path}")
            continue
            
        # Convert to the right dtype
        tensor = tensor.astype(np.dtype(param_dtype))
        
        # Convert the weight name to Flax format
        flax_name = convert_weight_name_to_flax(name)
        
        # Get the right partition spec for this parameter
        if mesh is not None:
            # Create a path through the partition specs
            flat_spec = flatten_dict(partition_specs)
            
            # Try to find a matching spec
            part_spec = None
            
            for spec_key, spec_value in flat_spec.items():
                # Convert tuple key to string path for matching
                spec_path = "/".join(spec_key)
                
                # Check if this spec matches our parameter
                if spec_path in flax_name:
                    part_spec = spec_value
                    break
                    
            # Shard the tensor if we have a partition spec
            if part_spec is not None:
                tensor = jax.device_put(tensor, jax.sharding.NamedSharding(mesh, part_spec))
            else:
                tensor = jax.device_put(tensor)
        else:
            # Just convert to JAX array
            tensor = jnp.array(tensor)
            
        # Add to the parameter dictionary
        nested_keys = flax_name.split("/")
        
        # Build the nested dictionary structure
        current_dict = params
        for key in nested_keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        
        # Set the tensor in the dictionary
        current_dict[nested_keys[-1]] = tensor
        loaded_params.add(name)
        
    # Close file handles
    for handle in file_handles.values():
        handle.close()
        
    # Check if we're missing any parameters
    missing_params = set(param_names) - loaded_params
    if missing_params:
        print(f"Warning: {len(missing_params)} parameters were not loaded:")
        for name in sorted(list(missing_params)[:10]):
            print(f"  - {name}")
        if len(missing_params) > 10:
            print(f"  ... and {len(missing_params) - 10} more")
            
    return params

def init_model_from_weights(
    model_class,
    model_path: str,
    config: Dict[str, Any],
    mesh: Optional[jax.sharding.Mesh] = None,
    param_dtype: jnp.dtype = jnp.bfloat16,
    input_ids_shape: Tuple[int, int] = (1, 8),
):
    """
    Initialize a model and load weights from safetensors.
    
    Args:
        model_class: Flax module class (TensorParallelQwen2ForCausalLM)
        model_path: Path to model safetensors files
        config: Model configuration dictionary
        mesh: Optional JAX mesh for tensor parallelism
        param_dtype: Data type for parameters
        input_ids_shape: Shape for initializing model parameters
        
    Returns:
        Initialized model parameters with pretrained weights
    """
    # Create partition specs for tensor parallelism
    if mesh is not None:
        ps = get_partition_specs(config)
        
        # Initialize the model with the mesh
        model = model_class(config=config, mesh=mesh, dtype=param_dtype, param_dtype=param_dtype)
        
        # Generate initialization inputs
        # This is just for parameter shape initialization, not dummy weights
        batch_size = input_ids_shape[0]
        
        # Adjust batch size for mesh shape if needed
        if mesh is not None and hasattr(mesh, 'devices'):
            try:
                # Use the mesh device shape which is more reliable
                mesh_shape = mesh.devices.shape
                if len(mesh_shape) >= 2:
                    # Assuming first dimension is batch dimension in the device mesh
                    batch_size = max(batch_size, mesh_shape[0])
                    print(f"Adjusted batch size to {batch_size} based on mesh device shape")
            except (IndexError, TypeError, AttributeError) as e:
                print(f"Warning: Could not adjust batch size based on mesh: {e}")
        
        # Create input tensor for initialization
        input_ids = jnp.zeros((batch_size, input_ids_shape[1]), dtype=jnp.int32)
        print(f"Input shape for initialization: {input_ids.shape}")
        
        # Initialize parameters with the mesh
        with mesh:
            rng = jax.random.PRNGKey(0)
            print(f"Initializing model parameters with input shape {input_ids.shape}...")
            params = jax.jit(model.init)(rng, input_ids)
            print("✅ Model parameters initialized successfully")
    else:
        # Initialize without tensor parallelism
        model = model_class(config=config, dtype=param_dtype, param_dtype=param_dtype)
        
        # Generate initialization inputs
        input_ids = jnp.zeros(input_ids_shape, dtype=jnp.int32)
        
        # Initialize parameters
        rng = jax.random.PRNGKey(0)
        print(f"Initializing model parameters with input shape {input_ids.shape}...")
        params = jax.jit(model.init)(rng, input_ids)
        print("✅ Model parameters initialized successfully")
    
    # Load pretrained weights
    print(f"Loading pretrained weights from {model_path}...")
    try:
        pretrained_params = load_qwen_weights(
            model_path=model_path,
            config=config,
            mesh=mesh,
            param_dtype=param_dtype,
        )
        print("✅ Pretrained weights loaded successfully")
        
        # We've loaded the parameters into the exact same structure our model expects
        # return the loaded parameters
        return model, pretrained_params
    except Exception as e:
        print(f"❌ Error loading pretrained weights: {e}")
        raise 