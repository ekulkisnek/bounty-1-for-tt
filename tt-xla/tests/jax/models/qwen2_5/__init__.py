# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
JAX implementation of the Qwen2.5-7B model with tensor parallelism.
"""

from .model_implementation import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    QwenAttention,
    QwenMLP,
    QwenTransformerBlock,
    Qwen2Model,
    Qwen2ForCausalLM,
)

from .tensor_parallel import (
    TensorParallelQwenAttention,
    TensorParallelQwenMLP,
    TensorParallelQwenTransformerBlock,
    TensorParallelQwen2Model,
    TensorParallelQwen2ForCausalLM,
)

from .config import (
    load_qwen_config,
    get_mesh_config,
    create_partition_specs,
    create_device_mesh,
    supported_mesh_configs,
)

from .weight_loading import (
    load_safetensors_index,
    convert_weight_name_to_flax,
    load_qwen_weights,
    init_model_from_weights,
)

from .integration import (
    MODEL_NAME,
    get_supported_mesh_configs,
    get_tensor_parallel_test_configs,
    get_model_test_runner,
    run_tensor_parallel_tests,
    load_and_run_inference,
)

from .register import (
    register_tests,
    get_model_metadata,
    register_model_factory,
    get_model_example,
)

__all__ = [
    # Model implementation
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "QwenAttention",
    "QwenMLP",
    "QwenTransformerBlock",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    
    # Tensor parallel implementation
    "TensorParallelQwenAttention",
    "TensorParallelQwenMLP",
    "TensorParallelQwenTransformerBlock",
    "TensorParallelQwen2Model",
    "TensorParallelQwen2ForCausalLM",
    
    # Configuration
    "load_qwen_config",
    "get_mesh_config",
    "create_partition_specs",
    "create_device_mesh",
    "supported_mesh_configs",
    
    # Weight loading
    "load_safetensors_index",
    "convert_weight_name_to_flax",
    "load_qwen_weights",
    "init_model_from_weights",
    
    # Integration
    "MODEL_NAME",
    "get_supported_mesh_configs",
    "get_tensor_parallel_test_configs",
    "get_model_test_runner",
    "run_tensor_parallel_tests",
    "load_and_run_inference",
    
    # Registration
    "register_tests",
    "get_model_metadata",
    "register_model_factory",
    "get_model_example",
]
