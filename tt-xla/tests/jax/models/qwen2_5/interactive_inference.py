#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Interactive inference script for Qwen2.5-7B model with different mesh configurations.
This script allows testing the model interactively with different tensor parallelism
configurations and evaluates GSM8k performance.
"""

import os
# Force JAX to use simulated devices for tensor parallelism testing
# Only set if not already set by the user
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"

import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from typing import List, Dict, Tuple, Optional, Union, Any
import json
from tqdm import tqdm

# Import model components
from model_implementation import Qwen2ForCausalLM
from tensor_parallel import (
    TensorParallelQwen2ForCausalLM,
    create_device_mesh,
)
from config import load_qwen_config, get_qwen2_7b_config
from weight_loading import init_model_from_weights


def setup_tokenizer(tokenizer_path: Optional[str] = None):
    """
    Set up the tokenizer for the model.
    
    Args:
        tokenizer_path: Path to the tokenizer files
        
    Returns:
        tokenizer: Huggingface tokenizer
    """
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
        print("Please make sure you have the 'transformers' library installed.")
        print("Run: pip install transformers")
        return None


def load_model(
    model_path: Optional[str] = None,
    mesh_shape: Tuple[int, int] = (1, 8),
    use_demo_model: bool = False,
):
    """
    Load the Qwen2.5-7B model with the specified mesh configuration.
    
    Args:
        model_path: Path to the model weights
        mesh_shape: Shape of the device mesh for tensor parallelism (batch, model)
        use_demo_model: If True, use a small demo model instead of full model
        
    Returns:
        model: The loaded model
        params: The model parameters
        mesh: The device mesh
    """
    print(f"\nLoading model with mesh shape: {mesh_shape}")
    
    # Check available devices
    devices = jax.devices()
    print(f"Available devices: {len(devices)}")
    
    if len(devices) < mesh_shape[0] * mesh_shape[1]:
        print(f"❌ Not enough devices for mesh shape {mesh_shape}")
        print(f"Required: {mesh_shape[0] * mesh_shape[1]}, Available: {len(devices)}")
        print("Using simulated devices via XLA_FLAGS")
    
    # Create the device mesh
    mesh = create_device_mesh(mesh_shape)
    print(f"✅ Device mesh created: {mesh_shape}")
    
    # Visualize the mesh
    print("Device mesh visualization:")
    jax.debug.visualize_array_sharding(
        jnp.zeros(mesh_shape, dtype=jnp.int32).reshape(mesh_shape)
    )
    
    if use_demo_model:
        # Use a small model for demonstration
        from config import get_small_config
        config = get_small_config(hidden_size=128, num_layers=2)
        
        # Create the model
        model = TensorParallelQwen2ForCausalLM(config=config, mesh=mesh)
        
        # Initialize parameters
        rng = jax.random.PRNGKey(0)
        batch_size = mesh_shape[0]
        
        # Create a sample input
        input_ids = jnp.ones((batch_size, 16), dtype=jnp.int32)
        
        # Shard the input
        input_sharding = jax.sharding.NamedSharding(mesh, P('batch', None))
        sharded_input = jax.device_put(input_ids, input_sharding)
        
        # Initialize parameters
        with mesh:
            params = model.init(rng, sharded_input)
        
        print(f"✅ Demo model initialized")
        return model, params, mesh
        
    else:
        # Check if model path exists
        if model_path is None:
            print("❌ Model path not provided")
            print("Falling back to demo model")
            return load_model(
                model_path=model_path,
                mesh_shape=mesh_shape,
                use_demo_model=True
            )
            
        if not os.path.exists(model_path):
            print(f"❌ Model path {model_path} does not exist")
            print("Falling back to demo model")
            return load_model(
                model_path=model_path,
                mesh_shape=mesh_shape,
                use_demo_model=True
            )
            
        # Check if model path contains required files
        if not os.path.exists(os.path.join(model_path, "config.json")):
            print(f"❌ Model path {model_path} does not contain config.json")
            print("Falling back to demo model")
            return load_model(
                model_path=model_path,
                mesh_shape=mesh_shape,
                use_demo_model=True
            )
            
        if not os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
            print(f"❌ Model path {model_path} does not contain model.safetensors.index.json")
            print("Falling back to demo model")
            return load_model(
                model_path=model_path,
                mesh_shape=mesh_shape,
                use_demo_model=True
            )
        
        # Load the full model
        # Load configuration from the specified path
        config = load_qwen_config(model_path)
        
        print("Model configuration:")
        print(f"Hidden size: {config['hidden_size']}")
        print(f"Attention heads: {config['num_attention_heads']}")
        print(f"Key/Value heads: {config['num_key_value_heads']}")
        print(f"Layers: {config['num_hidden_layers']}")
        print(f"Vocabulary size: {config['vocab_size']}")
        
        try:
            # Initialize and load model weights
            print("Initializing and loading model weights...")
            start_time = time.time()
            
            # Initialize and load weights using our helper function
            model, params = init_model_from_weights(
                model_class=TensorParallelQwen2ForCausalLM,
                model_path=model_path,
                config=config,
                mesh=mesh,
                param_dtype=jnp.bfloat16,
                input_ids_shape=(mesh_shape[0], 16)  # Use appropriate batch size
            )
            
            print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds")
            return model, params, mesh
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Falling back to demo model...")
            return load_model(
                model_path=model_path,
                mesh_shape=mesh_shape,
                use_demo_model=True
            )


def generate_text(
    model,
    params,
    mesh,
    input_ids,
    tokenizer,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Generate text from the model.
    
    Args:
        model: The Qwen2.5 model
        params: The model parameters
        mesh: The device mesh
        input_ids: Input token IDs
        tokenizer: The tokenizer
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Generated text
    """
    # Create input sharding
    input_sharding = jax.sharding.NamedSharding(mesh, P('batch', None))
    sharded_input = jax.device_put(input_ids, input_sharding)
    
    # Get the batch size from the input
    batch_size = input_ids.shape[0]
    
    # Track generated tokens
    generated_ids = input_ids.copy()
    
    # Define the autoregressive generation function
    def sample_token(logits, temperature=temperature, top_p=top_p):
        """Sample next token from logits using temperature and top-p sampling."""
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
            return jnp.argmax(logits)
        
        # Sample from the allowed tokens
        probs = jax.nn.softmax(allowed_logits)
        
        # Fix potential shape mismatch by normalizing probabilities
        probs = probs / jnp.sum(probs)
        
        # Use JAX's safe random choice approach
        try:
            sample_idx = jax.random.choice(
                jax.random.PRNGKey(int(time.time() * 1000000)), 
                allowed_indices, 
                p=probs
            )
        except ValueError:
            # Fallback in case of shape mismatch
            print(f"Warning: Shape mismatch in sampling. Using argmax instead.")
            sample_idx = allowed_indices[0]  # Just take the highest probability token
        
        return sample_idx
    
    # Generate tokens autoregressively
    start_time = time.time()
    print("Generating text...", end="", flush=True)
    
    for _ in range(max_length):
        # Run the model on the current input
        with mesh:
            outputs = model.apply(params, sharded_input)
        
        # Get the logits for the last token of each batch item
        last_token_logits = outputs[0][:, -1, :]
        
        # Sample the next token for each batch item
        next_tokens = []
        for i in range(batch_size):
            next_token = sample_token(last_token_logits[i])
            next_tokens.append(next_token)
        
        next_tokens = jnp.array(next_tokens)
        
        # Add the new token to the generated sequence
        generated_ids = jnp.concatenate([generated_ids, next_tokens.reshape(batch_size, 1)], axis=1)
        
        # Create a new input for the next iteration
        sharded_input = jax.device_put(generated_ids, input_sharding)
        
        # Print progress
        print(".", end="", flush=True)
        
        # Check for EOS tokens
        eos_tokens = [tokenizer.eos_token_id]
        if all(jnp.any(jnp.isin(generated_ids[:, -1], jnp.array(eos_tokens))) for _ in range(batch_size)):
            break
    
    print(f" Done in {time.time() - start_time:.2f} seconds")
    
    # Decode the generated tokens
    generated_text = []
    for i in range(batch_size):
        # Convert to list for easier processing
        ids = generated_ids[i].tolist()
        decoded_text = tokenizer.decode(ids, skip_special_tokens=True)
        generated_text.append(decoded_text)
    
    return generated_text


def format_prompt(prompt, use_template=True):
    """
    Format the input prompt using the Qwen2.5 chat template.
    
    Args:
        prompt: The raw input prompt
        use_template: Whether to use the Qwen chat template
        
    Returns:
        The formatted prompt
    """
    if not use_template:
        return prompt
        
    # Qwen2.5 chat template
    template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return template.format(prompt=prompt)


def format_follow_up(conversation, follow_up, use_template=True):
    """
    Format a follow-up prompt based on existing conversation.
    
    Args:
        conversation: The existing conversation
        follow_up: The follow-up prompt
        use_template: Whether to use the Qwen chat template
        
    Returns:
        The formatted follow-up prompt
    """
    if not use_template:
        return conversation + "\n" + follow_up
    
    # Append the follow-up to the existing conversation
    template = "{conversation}<|im_end|>\n<|im_start|>user\n{follow_up}<|im_end|>\n<|im_start|>assistant\n"
    
    return template.format(conversation=conversation, follow_up=follow_up)


def benchmark_gsm8k(model, params, mesh, tokenizer, num_samples=20):
    """
    Benchmark the model on GSM8k dataset.
    
    Args:
        model: The model
        params: The model parameters
        mesh: The device mesh
        tokenizer: The tokenizer
        num_samples: Number of GSM8k samples to evaluate
        
    Returns:
        Accuracy score as a percentage
    """
    try:
        from datasets import load_dataset
        
        print(f"\nRunning GSM8k benchmark (sampling {num_samples} examples)...")
        
        # Load GSM8k test dataset
        dataset = load_dataset("gsm8k", "main", split="test")
        
        # Limit to specified number of samples
        if num_samples > 0:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        correct = 0
        total = len(dataset)
        
        # Template for GSM8k problems
        template = "Solve the following math problem step by step:\n{question}"
        
        # Process each problem
        for i, item in enumerate(tqdm(dataset)):
            # Format the problem
            question = item["question"]
            prompt = format_prompt(template.format(question=question))
            
            # Encode the prompt
            input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int32)
            
            # Generate a response
            response = generate_text(
                model=model,
                params=params,
                mesh=mesh,
                input_ids=input_ids,
                tokenizer=tokenizer,
                max_length=200,
                temperature=0.2,  # Lower temperature for math problems
            )[0]
            
            # Extract the model's answer
            try:
                # Look for the solution in the model's response
                model_answer = extract_answer(response)
                
                # Get the ground truth answer
                ground_truth = extract_answer(item["answer"])
                
                # Check if the model's answer matches the ground truth
                if model_answer is not None and ground_truth is not None:
                    if abs(float(model_answer) - float(ground_truth)) < 1e-6:
                        correct += 1
                
                print(f"Problem {i+1}: {'✓' if correct > i else '✗'}")
                print(f"Question: {question}")
                print(f"Model response: {response}")
                print(f"Model answer: {model_answer}, Ground truth: {ground_truth}")
                print("-" * 40)
                
            except Exception as e:
                print(f"Error processing problem {i+1}: {e}")
        
        # Calculate accuracy
        accuracy = 100 * correct / total
        print(f"\nGSM8k accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        return accuracy
    
    except Exception as e:
        print(f"❌ Error running GSM8k benchmark: {e}")
        print("Please make sure you have the 'datasets' library installed.")
        print("Run: pip install datasets")
        return None


def extract_answer(text):
    """
    Extract the final numerical answer from a solution.
    
    Args:
        text: The solution text
        
    Returns:
        The extracted numerical answer or None if not found
    """
    # Look for common answer patterns
    import re
    
    # Try to find answers in format "The answer is X" or similar
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


def interactive_test(
    model_path=None,
    use_demo_model=False,
    use_template=True,
    run_gsm8k=False,
    mesh_shape=(1, 8)
):
    """
    Run interactive testing for the tensor-parallel Qwen2.5 model.
    
    Args:
        model_path: Path to model weights
        use_demo_model: Whether to use a demo model (smaller)
        use_template: Whether to use the chat template
        run_gsm8k: Whether to run GSM8K benchmarking
        mesh_shape: Shape of the device mesh for tensor parallelism
    """
    # Resolve the model path
    if model_path is None:
        # Try to find the model in the default location (relative to this script)
        default_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../../qwen2.5-7b"
        ))
        
        if os.path.exists(default_path):
            model_path = default_path
            print(f"Using default model path: {model_path}")
        else:
            print(f"Default model path not found: {default_path}")
            print("Please specify a model path with --model_path")
            return
    
    # Verify model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist")
        return
        
    # Check if model directory has required files
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Error: Model path {model_path} does not contain config.json")
        return
        
    print(f"Using model path: {model_path}")
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(model_path)
    if tokenizer is None:
        print("Error: Failed to load tokenizer")
        return
    
    # Load model with tensor parallelism
    model, params, mesh = load_model(
        model_path=model_path,
        mesh_shape=mesh_shape,
        use_demo_model=use_demo_model
    )
    
    if model is None:
        print("Error: Failed to load model")
        return
    
    # Run GSM8K benchmark if requested
    if run_gsm8k:
        print("\nRunning GSM8K benchmark...")
        benchmark_gsm8k(model, params, mesh, tokenizer)
        return
    
    # Interactive chat loop
    print("\n" + "=" * 50)
    print("Welcome to the Qwen2.5-7B Interactive Chat")
    print("Type 'exit' to quit")
    print("=" * 50 + "\n")
    
    # Track conversation history
    conversation = []
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        # Format the prompt
        if not conversation:
            # First message
            prompt = format_prompt(user_input, use_template)
        else:
            # Follow-up message
            prompt = format_follow_up(conversation, user_input, use_template)
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        input_ids = jnp.array(input_ids)
        
        # Generate response
        print("\nAssistant: ", end="", flush=True)
        
        start_time = time.time()
        output_text = generate_text(
            model=model,
            params=params,
            mesh=mesh,
            input_ids=input_ids,
            tokenizer=tokenizer,
            max_length=256,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Extract just the assistant's reply from the output
        response = output_text.split("<|im_start|>assistant\n")[-1].strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0].strip()
        
        print(response)
        
        generation_time = time.time() - start_time
        print(f"\n[Generated in {generation_time:.2f} seconds]")
        
        # Update conversation history
        conversation.append((user_input, response))
        

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive Qwen2.5-7B testing with different mesh configurations")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model weights (default: use the base configuration)"
    )
    
    parser.add_argument(
        "--use_demo_model",
        action="store_true",
        help="Use a small demo model instead of the full Qwen2.5-7B model"
    )
    
    parser.add_argument(
        "--use_template",
        action="store_true",
        default=True,
        help="Use the Qwen chat template (default: True)"
    )
    
    parser.add_argument(
        "--run_gsm8k",
        action="store_true",
        help="Run GSM8k benchmark"
    )
    
    parser.add_argument(
        "--mesh_shape",
        type=str,
        default="1x8",
        help="Shape of the device mesh for tensor parallelism (format AxB, e.g., '1x8')"
    )
    
    return parser.parse_args()


def main():
    """Main function to parse arguments and run the interactive test."""
    args = parse_arguments()
    
    # Parse mesh shape
    try:
        mesh_shape = tuple(map(int, args.mesh_shape.split('x')))
        assert len(mesh_shape) == 2, "Mesh shape must be in format AxB (e.g., '1x8')"
    except Exception as e:
        print(f"Error parsing mesh shape: {e}")
        print("Using default mesh shape: 1x8")
        mesh_shape = (1, 8)
    
    interactive_test(
        model_path=args.model_path,
        use_demo_model=args.use_demo_model,
        use_template=args.use_template,
        run_gsm8k=args.run_gsm8k,
        mesh_shape=mesh_shape
    )


if __name__ == "__main__":
    main() 