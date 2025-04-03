#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Verification script to confirm that tensor parallel implementation of Qwen2.5-7B
produces the same GSM8k score as the single-device implementation.
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
from typing import List, Dict, Optional, Tuple
import re
from tqdm import tqdm

# Import model components
from model_implementation import Qwen2ForCausalLM
from tensor_parallel import (
    TensorParallelQwen2ForCausalLM,
    create_device_mesh,
)
from config import get_qwen2_7b_config
from weight_loading import init_model_from_weights


def setup_tokenizer(tokenizer_path=None):
    """Set up the tokenizer."""
    try:
        from transformers import AutoTokenizer
        
        if tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        
        print(f"✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("Please install transformers: pip install transformers")
        return None


def load_standard_model(model_path: str):
    """Load the standard (non-tensor-parallel) model."""
    print("\nLoading standard Qwen2.5-7B model...")
    
    # Get model configuration
    config = get_qwen2_7b_config()
    
    # Initialize model
    model = Qwen2ForCausalLM(config=config)
    
    # Initialize and load weights
    params = init_model_from_weights(
        model_class=model.__class__,
        model_path=model_path,
        config=config,
        mesh=None,
        param_dtype=jnp.bfloat16
    )
    
    print(f"✅ Standard model loaded successfully")
    return model, params


def load_tensor_parallel_model(model_path: str, mesh_shape=(1, 8)):
    """Load the tensor-parallel model."""
    print(f"\nLoading tensor-parallel Qwen2.5-7B model with mesh shape {mesh_shape}...")
    
    # Create device mesh
    mesh = create_device_mesh(mesh_shape)
    print(f"✅ Device mesh created: {mesh_shape}")
    
    # Get model configuration
    config = get_qwen2_7b_config()
    
    # Initialize model
    model = TensorParallelQwen2ForCausalLM(config=config, mesh=mesh)
    
    # Initialize and load weights
    params = init_model_from_weights(
        model_class=model.__class__,
        model_path=model_path,
        config=config,
        mesh=mesh,
        param_dtype=jnp.bfloat16
    )
    
    print(f"✅ Tensor-parallel model loaded successfully")
    return model, params, mesh


def extract_answer(text):
    """Extract the final numerical answer from a solution."""
    # Look for common answer patterns
    patterns = [
        r"(?:answer|result)(?:\s+is)?\s*(?:=|:)?\s*(-?\d+(?:\.\d+)?)",
        r"(?:=|:)\s*(-?\d+(?:\.\d+)?)\s*$",
        r"(-?\d+(?:\.\d+)?)\s*$",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1]
    
    # If no match found, try to find the last number in the text
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if numbers:
        return numbers[-1]
    
    return None


def format_prompt(prompt, use_template=True):
    """Format the input prompt using the Qwen2.5 chat template."""
    if not use_template:
        return prompt
        
    # Qwen2.5 chat template
    template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return template.format(prompt=prompt)


def generate_text_standard(
    model,
    params,
    input_ids,
    tokenizer,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
):
    """Generate text using the standard model."""
    # Track generated tokens
    generated_ids = input_ids.copy()
    
    # Generate tokens autoregressively
    for _ in range(max_length):
        # Run the model on the current input
        outputs = model.apply(params, generated_ids)
        
        # Get logits for the last token
        logits = outputs[0][0, -1, :]
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-p sampling
        sorted_indices = jnp.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)
        mask = cumulative_probs < top_p
        
        # Ensure at least one token is selected
        mask = jnp.concatenate([jnp.ones_like(mask[:1]), mask[1:]], axis=0)
        
        # Get the allowed logits and indices
        allowed_logits = jnp.where(mask, sorted_logits, -float('inf'))
        allowed_indices = sorted_indices[mask]
        
        # Check if we have any allowed indices
        if allowed_indices.size == 0:
            # Fallback to top-1 sampling
            next_token = jnp.argmax(logits)
        else:
            # Sample from the allowed tokens
            probs = jax.nn.softmax(allowed_logits)
            
            # Fix potential shape mismatch by normalizing probabilities
            probs = probs / jnp.sum(probs)
            
            try:
                next_token = jax.random.choice(
                    jax.random.PRNGKey(int(time.time() * 1000000)), 
                    allowed_indices, 
                    p=probs
                )
            except ValueError:
                # Fallback to using the most likely token
                next_token = allowed_indices[0]
        
        # Add the new token to the generated sequence
        generated_ids = jnp.concatenate([generated_ids, next_token.reshape(1, 1)], axis=1)
        
        # Check for EOS tokens
        eos_tokens = [tokenizer.eos_token_id]
        if jnp.any(jnp.isin(generated_ids[0, -1:], jnp.array(eos_tokens))):
            break
    
    # Decode the generated tokens
    decoded_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    return decoded_text


def generate_text_tensor_parallel(
    model,
    params,
    mesh,
    input_ids,
    tokenizer,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
):
    """Generate text using the tensor-parallel model."""
    # Create input sharding
    input_sharding = jax.sharding.NamedSharding(mesh, P('batch', None))
    
    # Track generated tokens
    generated_ids = input_ids.copy()
    
    # Generate tokens autoregressively
    for _ in range(max_length):
        # Shard the input
        sharded_input = jax.device_put(generated_ids, input_sharding)
        
        # Run the model on the current input
        with mesh:
            outputs = model.apply(params, sharded_input)
        
        # Get logits for the last token
        logits = outputs[0][0, -1, :]
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-p sampling
        sorted_indices = jnp.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)
        mask = cumulative_probs < top_p
        
        # Ensure at least one token is selected
        mask = jnp.concatenate([jnp.ones_like(mask[:1]), mask[1:]], axis=0)
        
        # Get the allowed logits and indices
        allowed_logits = jnp.where(mask, sorted_logits, -float('inf'))
        allowed_indices = sorted_indices[mask]
        
        # Check if we have any allowed indices
        if allowed_indices.size == 0:
            # Fallback to top-1 sampling
            next_token = jnp.argmax(logits)
        else:
            # Sample from the allowed tokens
            probs = jax.nn.softmax(allowed_logits)
            
            # Fix potential shape mismatch by normalizing probabilities
            probs = probs / jnp.sum(probs)
            
            try:
                next_token = jax.random.choice(
                    jax.random.PRNGKey(int(time.time() * 1000000)), 
                    allowed_indices, 
                    p=probs
                )
            except ValueError:
                # Fallback to using the most likely token
                next_token = allowed_indices[0]
        
        # Add the new token to the generated sequence
        generated_ids = jnp.concatenate([generated_ids, next_token.reshape(1, 1)], axis=1)
        
        # Check for EOS tokens
        eos_tokens = [tokenizer.eos_token_id]
        if jnp.any(jnp.isin(generated_ids[0, -1:], jnp.array(eos_tokens))):
            break
    
    # Decode the generated tokens
    decoded_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    return decoded_text


def evaluate_gsm8k(
    standard_model=None,
    standard_params=None,
    tp_model=None,
    tp_params=None,
    tp_mesh=None,
    tokenizer=None,
    num_samples=10,
):
    """Evaluate GSM8K problems on both standard and tensor-parallel models."""
    print("\nEvaluating GSM8K problems...")
    
    # Load GSM8K problems
    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main")
        test_problems = dataset["test"]
        print(f"✅ Loaded {len(test_problems)} GSM8K problems")
    except Exception as e:
        print(f"❌ Error loading GSM8K dataset: {e}")
        print("Please install datasets: pip install datasets")
        return
    
    # Track scores
    standard_correct = 0
    tp_correct = 0
    total = 0
    
    # Evaluate on a subset of problems
    for i in tqdm(range(min(num_samples, len(test_problems)))):
        problem = test_problems[i]
        question = problem["question"]
        answer = problem["answer"]
        
        # Format the prompt
        prompt = format_prompt(question)
        input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
        
        # Generate solutions
        if standard_model is not None:
            standard_solution = generate_text_standard(
                standard_model,
                standard_params,
                input_ids,
                tokenizer,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
            )
            standard_answer = extract_answer(standard_solution)
        
        if tp_model is not None:
            tp_solution = generate_text_tensor_parallel(
                tp_model,
                tp_params,
                tp_mesh,
                input_ids,
                tokenizer,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
            )
            tp_answer = extract_answer(tp_solution)
        
        # Compare answers
        if standard_model is not None and standard_answer is not None:
            if float(standard_answer) == float(answer):
                standard_correct += 1
        
        if tp_model is not None and tp_answer is not None:
            if float(tp_answer) == float(answer):
                tp_correct += 1
        
        total += 1
        
        # Print progress
        if (i + 1) % 5 == 0:
            if standard_model is not None:
                print(f"\nStandard model accuracy: {standard_correct/total:.2%}")
            if tp_model is not None:
                print(f"Tensor-parallel model accuracy: {tp_correct/total:.2%}")
    
    # Print final results
    print("\nFinal Results:")
    if standard_model is not None:
        print(f"Standard model accuracy: {standard_correct/total:.2%}")
    if tp_model is not None:
        print(f"Tensor-parallel model accuracy: {tp_correct/total:.2%}")
    
    return standard_correct/total if standard_model is not None else None, \
           tp_correct/total if tp_model is not None else None


def main():
    parser = argparse.ArgumentParser(description="Verify GSM8K scores for Qwen2.5-7B")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen2.5-7B model weights")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of GSM8K problems to evaluate")
    parser.add_argument("--mesh_shapes", type=str, nargs="+", default=["1x8", "2x4", "1x32", "8x4"],
                      help="Mesh shapes to test (e.g., 1x8 2x4)")
    args = parser.parse_args()
    
    # Set up tokenizer
    tokenizer = setup_tokenizer()
    if tokenizer is None:
        return
    
    # Load standard model
    standard_model, standard_params = load_standard_model(args.model_path)
    
    # Test each mesh shape
    for mesh_shape in args.mesh_shapes:
        print(f"\nTesting mesh shape: {mesh_shape}")
        
        # Parse mesh shape
        rows, cols = map(int, mesh_shape.split("x"))
        
        # Load tensor-parallel model
        tp_model, tp_params, tp_mesh = load_tensor_parallel_model(args.model_path, (rows, cols))
        
        # Evaluate GSM8K
        standard_score, tp_score = evaluate_gsm8k(
            standard_model=standard_model,
            standard_params=standard_params,
            tp_model=tp_model,
            tp_params=tp_params,
            tp_mesh=tp_mesh,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
        )
        
        # Compare scores
        if standard_score is not None and tp_score is not None:
            print(f"\nScore comparison for mesh shape {mesh_shape}:")
            print(f"Standard model: {standard_score:.2%}")
            print(f"Tensor-parallel model: {tp_score:.2%}")
            print(f"Difference: {abs(standard_score - tp_score):.4%}")
            
            # Verify scores match within tolerance
            if abs(standard_score - tp_score) > 0.01:  # 1% tolerance
                print("❌ Scores do not match within tolerance!")
            else:
                print("✅ Scores match within tolerance!")


if __name__ == "__main__":
    main() 