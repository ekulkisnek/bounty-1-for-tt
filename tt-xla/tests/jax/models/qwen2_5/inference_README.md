# Qwen2.5-7B Interactive Inference with Different Mesh Configurations

This document provides instructions on how to use the interactive inference script to test the Qwen2.5-7B model with different tensor parallelism configurations.

## Overview

The `interactive_inference.py` script allows you to:

1. Test inference on different mesh configurations (2x4, 1x8, 1x32, 8x4)
2. Send prompts to the model and get responses from each configuration
3. Continue conversations with follow-up prompts
4. Evaluate GSM8k benchmark performance

## Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install transformers
pip install jax jaxlib
pip install flax
pip install tqdm
pip install datasets  # optional, for GSM8k benchmark
```

## Running the Script

### Basic Usage

To run the script with default settings (using a demo model):

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=8"
python tt-xla/tests/jax/models/qwen2_5/interactive_inference.py --use_demo_model
```

### Using a Full Model

To run with a downloaded Qwen2.5-7B model:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=8"
python tt-xla/tests/jax/models/qwen2_5/interactive_inference.py --model_path /path/to/qwen2.5-7b
```

### Testing More Mesh Configurations

To test larger mesh configurations like 1x32 and 8x4, increase the number of simulated devices:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=32"
python tt-xla/tests/jax/models/qwen2_5/interactive_inference.py --use_demo_model
```

### Running GSM8k Benchmark

To evaluate GSM8k benchmark performance:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=8"
python tt-xla/tests/jax/models/qwen2_5/interactive_inference.py --use_demo_model --run_gsm8k
```

## Command-Line Arguments

The script supports the following arguments:

- `--model_path`: Path to the model weights (default: use base configuration)
- `--use_demo_model`: Use a small demo model instead of full Qwen2.5-7B model
- `--use_template`: Use the Qwen chat template (default: True)
- `--run_gsm8k`: Run GSM8k benchmark

## Interactive Mode

Once the script is running, it enters interactive mode where you can:

1. Type prompts and see responses from all mesh configurations
2. Continue the conversation with follow-up questions
3. Type 'exit' or 'quit' to end the session

Example interaction:

```
Enter your prompt: What is the capital of France?

--------------------------------------------------
Mesh 2x4 response:
The capital of France is Paris.

--------------------------------------------------
Mesh 1x8 response:
The capital of France is Paris.

Enter your prompt: Tell me more about its history.

--------------------------------------------------
Mesh 2x4 response:
Paris has a rich history dating back to Roman times...

--------------------------------------------------
Mesh 1x8 response:
Paris has a rich history dating back to Roman times...
```

## GSM8k Benchmark

The GSM8k benchmark evaluates the model's mathematical reasoning capabilities. When running with `--run_gsm8k`, the script:

1. Loads a subset of the GSM8k dataset
2. Tests the model on math word problems
3. Extracts numerical answers and compares with ground truth
4. Reports accuracy for each mesh configuration

The bounty requirement specifies that tensor parallel implementation should produce the same GSM8k score as single-device implementation, which can be verified with this benchmark.

## Understanding Mesh Configurations

The script supports the following mesh configurations:

- **2x4**: 2 devices for batch parallelism, 4 devices for model parallelism
- **1x8**: 8 devices for model parallelism
- **1x32**: 32 devices for model parallelism
- **8x4**: 8 devices for batch parallelism, 4 devices for model parallelism

Different configurations have different trade-offs:
- More devices in the model dimension = more model parallelism (good for larger models)
- More devices in the batch dimension = more data parallelism (good for throughput) 