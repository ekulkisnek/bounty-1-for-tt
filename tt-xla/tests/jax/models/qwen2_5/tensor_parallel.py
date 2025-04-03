# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tensor parallel implementation of Qwen2.5-7B model for JAX.
This module contains the tensor-parallel model components and utilities.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from functools import partial
from typing import Any, Dict, Optional, Tuple
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

from model_implementation import (
    RMSNorm,
    QwenAttention,
    QwenMLP,
    QwenTransformerBlock,
    Qwen2Model,
    Qwen2ForCausalLM,
    precompute_freqs_cis
)

def create_device_mesh(mesh_shape):
    """
    Create a device mesh with the specified shape.
    
    Args:
        mesh_shape: Tuple of (rows, cols) for the mesh shape
        
    Returns:
        jax.sharding.Mesh: A JAX device mesh
    """
    devices = jax.devices()
    required_devices = mesh_shape[0] * mesh_shape[1]
    
    print(f"Creating mesh with shape {mesh_shape}, requiring {required_devices} devices")
    print(f"Available devices: {len(devices)}")
    
    if len(devices) < required_devices:
        raise ValueError(
            f"Not enough devices ({len(devices)}) for mesh shape {mesh_shape}. "
            f"Required: {required_devices}. Set XLA_FLAGS to simulate more devices."
        )
    
    if len(devices) > required_devices:
        print(f"Warning: Using only {required_devices} of {len(devices)} available devices")
        devices = devices[:required_devices]
    
    try:
        # Create a flat array of devices with the required shape
        devices_array = np.array(devices).reshape(mesh_shape)
        mesh = Mesh(devices_array, ('batch', 'model'))
        print(f"Mesh created with shape {mesh_shape}")
        print(f"Mesh axis_names: {mesh.axis_names}")
        print(f"Mesh object properties: shape={getattr(mesh, 'shape', 'None')}, "
              f"size={getattr(mesh, 'size', 'None')}")
        print(f"Mesh device shape: {mesh.devices.shape}")
        return mesh
    except ValueError as e:
        print(f"Error creating mesh with np.array.reshape: {e}")
        try:
            # Try using mesh_utils with the sliced devices
            device_mesh = mesh_utils.create_device_mesh(mesh_shape, devices=devices[:required_devices])
            mesh = Mesh(device_mesh, ('batch', 'model'))
            print(f"Mesh created using mesh_utils")
            print(f"Mesh axis_names: {mesh.axis_names}")
            print(f"Mesh object properties: shape={getattr(mesh, 'shape', 'None')}, "
                  f"size={getattr(mesh, 'size', 'None')}")
            print(f"Mesh device shape: {mesh.devices.shape}")
            return mesh
        except Exception as ex:
            print(f"Error creating mesh with mesh_utils: {ex}")
            raise ValueError(
                f"Failed to create device mesh with shape {mesh_shape}. "
                f"Available devices: {len(devices)}. Required: {required_devices}."
            )

def get_partition_specs(config):
    """
    Create partition specifications for the model parameters.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Dict: Partition specs for the model parameters
    """
    hidden_size = config['hidden_size']
    intermediate_size = config['intermediate_size']
    num_attention_heads = config['num_attention_heads']
    
    # Partition specs for embeddings
    embed_p = P(None, 'model')
    
    # Partition specs for attention
    q_p = P(None, 'model')
    k_p = P(None, 'model')
    v_p = P(None, 'model')
    o_p = P('model', None)
    
    # Partition specs for MLP
    gate_p = P(None, 'model')
    up_p = P(None, 'model')
    down_p = P('model', None)
    
    # Weights partition specs
    weight_p = P(None)
    
    # Create complete partition specs
    return {
        'model': {
            'embed_tokens': {
                'embedding': embed_p,
            },
            'layers_.*': {
                'self_attn': {
                    'q_proj': {
                        'kernel': q_p,
                    },
                    'k_proj': {
                        'kernel': k_p,
                    },
                    'v_proj': {
                        'kernel': v_p,
                    },
                    'o_proj': {
                        'kernel': o_p,
                    },
                },
                'mlp': {
                    'gate_proj': {
                        'kernel': gate_p,
                    },
                    'up_proj': {
                        'kernel': up_p,
                    },
                    'down_proj': {
                        'kernel': down_p,
                    },
                },
                'input_layernorm': {
                    'weight': weight_p,
                },
                'post_attention_layernorm': {
                    'weight': weight_p,
                }
            },
            'norm': {
                'weight': weight_p,
            }
        },
        'lm_head': {
            'kernel': P('model', None),  # Transpose of embed_p
        }
    }

class TensorParallelDense(nn.Module):
    """Dense layer with tensor parallelism."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    precision: Any = None
    mesh: Mesh = None
    shard_axes: Tuple[str, str] = ('model', None)  # (kernel_in, kernel_out)
    
    @nn.compact
    def __call__(self, inputs):
        """Apply the dense layer with tensor parallelism."""
        input_dim = inputs.shape[-1]
        kernel_shape = (input_dim, self.features)
        
        # Initialize kernel parameter
        kernel = self.param(
            'kernel', 
            self.kernel_init, 
            kernel_shape, 
            self.param_dtype
        )
        kernel = kernel.astype(self.dtype)
        
        # Define partition spec based on shard_axes
        if self.shard_axes[0] and self.shard_axes[1]:
            kernel_spec = P(self.shard_axes[0], self.shard_axes[1])
        elif self.shard_axes[0]:
            kernel_spec = P(self.shard_axes[0], None)
        elif self.shard_axes[1]:
            kernel_spec = P(None, self.shard_axes[1])
        else:
            kernel_spec = P(None, None)
        
        # Shard the kernel if mesh is provided
        if self.mesh is not None:
            kernel = jax.lax.with_sharding_constraint(kernel, kernel_spec)
        
        # Matrix multiplication with safeguards
        y = jnp.matmul(inputs, kernel)
        
        # Add bias if needed
        if self.use_bias:
            # Initialize bias parameter
            bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
            bias = bias.astype(self.dtype)
            
            # Shard bias if needed
            if self.mesh is not None and self.shard_axes[1]:
                bias_spec = P(self.shard_axes[1])
                bias = jax.lax.with_sharding_constraint(bias, bias_spec)
            
            # Add to output
            y = y + bias
        
        return y

class TensorParallelQwenAttention(nn.Module):
    """Tensor parallel implementation of QwenAttention."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = None
    mesh: Mesh = None
    
    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ):
        """Apply tensor-parallel attention."""
        batch_size, seq_length = hidden_states.shape[:2]
        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        
        # Compute per-device dimensions
        num_devices = self.mesh.devices.size if self.mesh else 1
        model_parallel_size = num_devices  # Assuming model-parallel across all devices
        
        # Scale attention heads per device
        # Ensure at least 1 head per device to avoid division by zero
        num_attn_heads = self.config["num_attention_heads"]
        num_kv_heads = self.config["num_key_value_heads"]
        
        # If model_parallel_size > num_heads, adjust to ensure at least 1 head per device
        effective_model_size_q = min(model_parallel_size, num_attn_heads)
        effective_model_size_kv = min(model_parallel_size, num_kv_heads)
        
        n_heads_per_device = max(1, num_attn_heads // effective_model_size_q)
        n_kv_heads_per_device = max(1, num_kv_heads // effective_model_size_kv)
        
        # Project inputs to queries, keys, values with tensor parallelism
        q_proj = TensorParallelDense(
            features=n_heads_per_device * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="q_proj",
        )
        
        k_proj = TensorParallelDense(
            features=n_kv_heads_per_device * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="k_proj",
        )
        
        v_proj = TensorParallelDense(
            features=n_kv_heads_per_device * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="v_proj",
        )
        
        o_proj = TensorParallelDense(
            features=self.config["hidden_size"],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=('model', None),
            name="o_proj",
        )
        
        # Get queries, keys, values (these will be automatically sharded)
        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.reshape(
            batch_size, seq_length, n_heads_per_device, head_dim
        )
        key_states = key_states.reshape(
            batch_size, seq_length, n_kv_heads_per_device, head_dim
        )
        value_states = value_states.reshape(
            batch_size, seq_length, n_kv_heads_per_device, head_dim
        )
        
        # Setup position IDs if not provided
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :]
        
        # Precompute the rotary embeddings
        max_length = self.config.get("max_position_embeddings", 32768)
        rotary_emb = precompute_freqs_cis(
            head_dim, 
            max_length, 
            theta=self.config.get("rope_theta", 10000.0)
        )
        
        # Apply rotary embeddings from model_implementation
        from model_implementation import apply_rotary_emb
        query_states, key_states = apply_rotary_emb(
            query_states, key_states, rotary_emb, position_ids
        )
        
        # Handle KV caching
        if past_key_value is not None:
            # Concatenate past keys and values with current
            past_key, past_value = past_key_value
            key_states = jnp.concatenate([past_key, key_states], axis=1)
            value_states = jnp.concatenate([past_value, value_states], axis=1)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # For grouped-query attention, we need to repeat the keys and values
        # to match the number of query heads
        if n_kv_heads_per_device != n_heads_per_device:
            # Calculate repeat factor safely to avoid division by zero
            if n_kv_heads_per_device > 0:
                repeat_factor = n_heads_per_device // n_kv_heads_per_device
                if repeat_factor > 0:
                    # Repeat keys and values to match number of attention heads
                    key_states = jnp.repeat(key_states, repeat_factor, axis=2)
                    value_states = jnp.repeat(value_states, repeat_factor, axis=2)
                else:
                    # Handle case where n_heads_per_device < n_kv_heads_per_device
                    # We'll use the first n_heads_per_device key/value heads
                    key_states = key_states[:, :, :n_heads_per_device, :]
                    value_states = value_states[:, :, :n_heads_per_device, :]
            else:
                # This should not happen with our fix, but just in case
                print(f"Warning: n_kv_heads_per_device={n_kv_heads_per_device}, using n_heads_per_device={n_heads_per_device}")
                # Create dummy key/value states with the right shape
                key_states = jnp.zeros((batch_size, key_states.shape[1], n_heads_per_device, head_dim), dtype=key_states.dtype)
                value_states = jnp.zeros((batch_size, value_states.shape[1], n_heads_per_device, head_dim), dtype=value_states.dtype)
        
        # Ensure tensors have the expected shape (batch, seq, heads, head_dim)
        # If they have more dimensions, reshape them
        expected_rank = 4  # batch, seq, heads, head_dim
        
        if len(query_states.shape) != expected_rank:
            query_states = query_states.reshape(batch_size, seq_length, -1, head_dim)
        
        if len(key_states.shape) != expected_rank:
            key_states = key_states.reshape(batch_size, seq_length, -1, head_dim)
        
        if len(value_states.shape) != expected_rank:
            value_states = value_states.reshape(batch_size, seq_length, -1, head_dim)
        
        # Transpose tensors to prepare for attention calculation
        # [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))
        
        # Transpose key for matrix multiplication
        # [batch, heads, seq, head_dim] -> [batch, heads, head_dim, seq]
        key_states_t = jnp.transpose(key_states, (0, 1, 3, 2))
        
        # Calculate attention scores without einsum
        # [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq] -> [batch, heads, seq, seq]
        attn_weights = jnp.matmul(query_states, key_states_t)
        
        # Scale attention scores
        attn_weights = attn_weights / jnp.sqrt(head_dim).astype(attn_weights.dtype)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert to correct dtype
            attention_mask = attention_mask.astype(attn_weights.dtype)
            
            # Apply mask (adding large negative values to masked positions)
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Apply attention dropout during training
        if not deterministic:
            attn_weights = nn.Dropout(
                rate=self.config.get("attention_dropout", 0.0)
            )(attn_weights, deterministic=deterministic)
        
        # Calculate attention output without einsum
        # [batch, heads, seq, seq] @ [batch, heads, seq, head_dim] -> [batch, heads, seq, head_dim]
        attn_output = jnp.matmul(attn_weights, value_states)
        
        # Transpose back to original format
        # [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        
        # Merge heads
        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        
        # Final projection with tensor parallelism
        attn_output = o_proj(attn_output)
        
        outputs = (attn_output, past_key_value)
        
        if output_attentions:
            outputs = outputs + (attn_weights,)
            
        return outputs

class TensorParallelQwenMLP(nn.Module):
    """Tensor parallel implementation of QwenMLP."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    mesh: Mesh = None
    
    @nn.compact
    def __call__(self, x):
        """Apply the MLP to the input with tensor parallelism."""
        hidden_size = self.config["hidden_size"]
        intermediate_size = self.config["intermediate_size"]
        
        # Compute per-device dimensions
        num_devices = self.mesh.devices.size if self.mesh else 1
        model_parallel_size = num_devices  # Assuming model-parallel across all devices
        
        # Scale intermediate size per device
        intermediate_size_per_device = intermediate_size // model_parallel_size
        
        # Gate and up projections with tensor parallelism
        gate_proj = TensorParallelDense(
            features=intermediate_size_per_device,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="gate_proj",
        )
        
        up_proj = TensorParallelDense(
            features=intermediate_size_per_device,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=(None, 'model'),
            name="up_proj",
        )
        
        down_proj = TensorParallelDense(
            features=hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=('model', None),
            name="down_proj",
        )
        
        # Apply SwiGLU activation with tensor parallelism
        gate = gate_proj(x)
        gate = nn.silu(gate)
        
        up = up_proj(x)
        
        intermediate = gate * up
        
        # Project back to hidden size with tensor parallelism
        output = down_proj(intermediate)
        
        return output

class TensorParallelQwenTransformerBlock(nn.Module):
    """Tensor parallel implementation of QwenTransformerBlock."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    mesh: Mesh = None
    
    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ):
        """Process the input through self-attention and MLP with tensor parallelism."""
        residual = hidden_states
        
        # Layer normalization before self-attention
        hidden_states = RMSNorm(
            dim=self.config["hidden_size"],
            eps=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="input_layernorm",
        )(hidden_states)
        
        # Self-attention with tensor parallelism
        attn_outputs = TensorParallelQwenAttention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh=self.mesh,
            name="self_attn",
        )(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        
        attention_output = attn_outputs[0]
        past_key_value = attn_outputs[1] if use_cache else None
        attention_weights = attn_outputs[2] if output_attentions else None
        
        # First residual connection
        hidden_states = residual + attention_output
        
        # Second residual block
        residual = hidden_states
        
        # Layer normalization before MLP
        hidden_states = RMSNorm(
            dim=self.config["hidden_size"],
            eps=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="post_attention_layernorm",
        )(hidden_states)
        
        # MLP with tensor parallelism
        hidden_states = TensorParallelQwenMLP(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh=self.mesh,
            name="mlp",
        )(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if use_cache:
            outputs = outputs + (past_key_value,)
            
        if output_attentions:
            outputs = outputs + (attention_weights,)
            
        return outputs

class TensorParallelQwen2Model(nn.Module):
    """Tensor parallel implementation of Qwen2Model."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    mesh: Mesh = None
    
    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ):
        """Process the inputs through the model with tensor parallelism."""
        batch_size, seq_length = input_ids.shape
        
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length))
        
        # We create a 3D attention mask from a 2D tensor mask.
        extended_attention_mask = attention_mask[:, None, None, :]
        # Convert to the type needed for attention mechanism
        extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(self.dtype).min
        
        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :]
        
        # Setup for past key values
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[1]  # Using the key's sequence length
            
            # Adjust position_ids to account for past keys and values
            position_ids = position_ids[:, past_length:seq_length + past_length]
        
        # Embedding layer (token embeddings)
        embed_tokens = nn.Embed(
            num_embeddings=self.config["vocab_size"],
            features=self.config["hidden_size"],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            embedding_init=nn.initializers.normal(stddev=self.config["initializer_range"]),
            name="embed_tokens",
        )
        
        # Get embeddings
        hidden_states = embed_tokens(input_ids)
        
        # Store all hidden states and attentions if requested
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_past_key_values = () if use_cache else None
        
        # Create the transformer layers with tensor parallelism
        for i in range(self.config["num_hidden_layers"]):
            # Store the hidden state for this layer if requested
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Get layer-specific past key values
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Apply the transformer block with tensor parallelism
            layer_outputs = TensorParallelQwenTransformerBlock(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                mesh=self.mesh,
                name=f"layers_{i}",
            )(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                deterministic=deterministic,
            )
            
            # Update hidden states
            hidden_states = layer_outputs[0]
            
            # Store past key values if requested
            if use_cache:
                all_past_key_values = all_past_key_values + (layer_outputs[1],)
            
            # Store attention weights if requested
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-1],)
        
        # Final layer normalization
        hidden_states = RMSNorm(
            dim=self.config["hidden_size"],
            eps=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="norm",
        )(hidden_states)
        
        # Store the final hidden state if requested
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Prepare outputs
        outputs = (hidden_states,)
        
        if use_cache:
            outputs = outputs + (all_past_key_values,)
        
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
            
        if output_attentions:
            outputs = outputs + (all_attentions,)
            
        return outputs

class TensorParallelQwen2ForCausalLM(nn.Module):
    """Tensor parallel implementation of Qwen2ForCausalLM."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    mesh: Mesh = None
    
    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_cache: bool = False,
        deterministic: bool = True,
    ):
        """Forward pass for the causal language model with tensor parallelism."""
        # Apply the base model with tensor parallelism
        outputs = TensorParallelQwen2Model(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh=self.mesh,
            name="model",
        )(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        
        hidden_states = outputs[0]
        
        # Language modeling head with tensor parallelism
        lm_logits = TensorParallelDense(
            features=self.config["vocab_size"],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            mesh=self.mesh,
            shard_axes=('model', None),
            name="lm_head",
        )(hidden_states)
        
        # Prepare outputs - logits first, then the rest in order
        outputs = (lm_logits,) + outputs[1:]
        
        return outputs 