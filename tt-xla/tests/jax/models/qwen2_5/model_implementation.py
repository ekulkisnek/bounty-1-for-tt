# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization as used in Qwen2."""
    dim: int
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        """Apply normalization."""
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * (1.0 / jnp.sqrt(variance + self.eps))
        
        weight = self.param(
            'weight',
            nn.initializers.ones,
            (self.dim,),
            self.param_dtype,
        )
        weight = weight.astype(self.dtype)
        
        return x * weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jnp.ndarray:
    """Precompute the frequency tensor for complex exponentials with given dimension."""
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(end)  # type: ignore
    freqs = jnp.outer(t, freqs)  # [end, dim // 2]
    
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    
    return jnp.concatenate([cos, sin], axis=-1)

def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
    position_ids: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary embeddings to the query and key tensors."""
    # xq, xk: [batch, seq_len, n_heads, head_dim]
    # freqs_cis: [seq_len, dim//2]
    # position_ids: [batch_size, seq_len]
    
    batch_size, seq_len = position_ids.shape
    head_dim = xq.shape[-1]
    
    # Get the position embeddings for the positions we're interested in
    freqs_cis = jnp.take(freqs_cis, position_ids.reshape(-1), axis=0)
    freqs_cis = freqs_cis.reshape(batch_size, seq_len, head_dim // 2, 2)
    
    # Reshape query and key for interleaved complex multiplication
    xq_r = xq.reshape(batch_size, seq_len, -1, head_dim // 2, 2)
    xk_r = xk.reshape(batch_size, seq_len, -1, head_dim // 2, 2)
    
    # Extract real and imaginary parts
    xq_real, xq_imag = xq_r[..., 0], xq_r[..., 1]
    xk_real, xk_imag = xk_r[..., 0], xk_r[..., 1]
    
    # Extract cos and sin components
    freqs_cos = freqs_cis[..., 0]  # [batch, seq, dim//2]
    freqs_sin = freqs_cis[..., 1]  # [batch, seq, dim//2]
    
    # Reshape for broadcasting
    freqs_cos = freqs_cos[:, :, None, :]  # [batch, seq, 1, dim//2]
    freqs_sin = freqs_sin[:, :, None, :]  # [batch, seq, 1, dim//2]
    
    # Complex multiplication
    # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    out_q_real = xq_real * freqs_cos - xq_imag * freqs_sin
    out_q_imag = xq_real * freqs_sin + xq_imag * freqs_cos
    out_k_real = xk_real * freqs_cos - xk_imag * freqs_sin
    out_k_imag = xk_real * freqs_sin + xk_imag * freqs_cos
    
    # Stack real and imaginary parts
    out_q = jnp.stack([out_q_real, out_q_imag], axis=-1)
    out_k = jnp.stack([out_k_real, out_k_imag], axis=-1)
    
    # Reshape back to original shapes
    out_q = out_q.reshape(batch_size, seq_len, -1, head_dim)
    out_k = out_k.reshape(batch_size, seq_len, -1, head_dim)
    
    return out_q, out_k

class QwenAttention(nn.Module):
    """Attention module for Qwen2.5 with Grouped Query Attention."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Optional[jax.lax.Precision] = None
    
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
        """
        Applies grouped query attention.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask of shape [batch_size, 1, 1, seq_len]
            position_ids: Optional position indices of shape [batch_size, seq_len]
            past_key_value: Optional cached KV states
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cached KV states
            deterministic: Whether to use deterministic operations (no dropout)
        
        Returns:
            Output tensor and optionally cached KV states and attention weights
        """
        batch_size, seq_length = hidden_states.shape[:2]
        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        
        # Project inputs to queries, keys, and values
        q_proj = nn.Dense(
            features=self.config["num_attention_heads"] * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            name="q_proj",
        )
        
        k_proj = nn.Dense(
            features=self.config["num_key_value_heads"] * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            name="k_proj",
        )
        
        v_proj = nn.Dense(
            features=self.config["num_key_value_heads"] * head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            name="v_proj",
        )
        
        o_proj = nn.Dense(
            features=self.config["hidden_size"],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            name="o_proj",
        )
        
        # Get queries, keys, values
        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.reshape(
            batch_size, seq_length, self.config["num_attention_heads"], head_dim
        )
        key_states = key_states.reshape(
            batch_size, seq_length, self.config["num_key_value_heads"], head_dim
        )
        value_states = value_states.reshape(
            batch_size, seq_length, self.config["num_key_value_heads"], head_dim
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
        
        # Apply rotary embeddings
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
        if self.config["num_key_value_heads"] != self.config["num_attention_heads"]:
            # Repeat keys and values to match number of attention heads
            key_states = jnp.repeat(
                key_states, 
                self.config["num_attention_heads"] // self.config["num_key_value_heads"], 
                axis=2
            )
            value_states = jnp.repeat(
                value_states, 
                self.config["num_attention_heads"] // self.config["num_key_value_heads"], 
                axis=2
            )
        
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
        
        # Final projection
        attn_output = o_proj(attn_output)
        
        outputs = (attn_output, past_key_value)
        
        if output_attentions:
            outputs = outputs + (attn_weights,)
            
        return outputs

class QwenMLP(nn.Module):
    """
    MLP for Qwen2.5 with SwiGLU activation function.
    """
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    
    @nn.compact
    def __call__(self, x):
        """Apply the MLP to the input."""
        hidden_size = self.config["hidden_size"]
        intermediate_size = self.config["intermediate_size"]
        
        # Gate and up projections
        gate_proj = nn.Dense(
            features=intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            name="gate_proj",
        )
        
        up_proj = nn.Dense(
            features=intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            name="up_proj",
        )
        
        down_proj = nn.Dense(
            features=hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            name="down_proj",
        )
        
        # Apply SwiGLU activation
        gate = gate_proj(x)
        gate = nn.silu(gate)
        
        up = up_proj(x)
        
        intermediate = gate * up
        
        # Project back to hidden size
        output = down_proj(intermediate)
        
        return output

class QwenTransformerBlock(nn.Module):
    """Transformer block for Qwen2.5."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    
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
        """Process the input through self-attention and MLP."""
        residual = hidden_states
        
        # Layer normalization before self-attention
        hidden_states = RMSNorm(
            dim=self.config["hidden_size"],
            eps=self.config.get("rms_norm_eps", 1e-6),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="input_layernorm",
        )(hidden_states)
        
        # Self-attention
        attn_outputs = QwenAttention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
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
        
        # MLP
        hidden_states = QwenMLP(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
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

class Qwen2Model(nn.Module):
    """Complete Qwen2.5 model implementation."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    
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
        """
        Process the inputs through the model.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            attention_mask: Optional mask of shape [batch_size, seq_len]
            position_ids: Optional position indices
            past_key_values: Optional cached key values from previous forward passes
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            use_cache: Whether to use cached KV states
            deterministic: Whether to use deterministic operations
            
        Returns:
            Output tensor and optionally past_key_values, hidden_states, and attentions
        """
        batch_size, seq_length = input_ids.shape
        
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length))
        
        # We create a 3D attention mask from a 2D tensor mask.
        # Let's create our attention mask like in the transformers library
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
        
        # Create the transformer layers
        for i in range(self.config["num_hidden_layers"]):
            # Store the hidden state for this layer if requested
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Get layer-specific past key values
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Apply the transformer block
            layer_outputs = QwenTransformerBlock(
                config=self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
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

class Qwen2ForCausalLM(nn.Module):
    """Qwen2.5 model with a language modeling head."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    
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
        """
        Forward pass for the causal language model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: Optional cached KV states
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            use_cache: Whether to use KV caching
            deterministic: Whether to use deterministic operations
            
        Returns:
            Logits and optionally past_key_values, hidden_states, and attentions
        """
        # Apply the base model
        outputs = Qwen2Model(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
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
        
        # Language modeling head
        lm_logits = nn.Dense(
            features=self.config["vocab_size"],
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(self.config["initializer_range"]),
            name="lm_head",
        )(hidden_states)
        
        # Prepare outputs - logits first, then the rest in order
        outputs = (lm_logits,) + outputs[1:]
        
        return outputs 