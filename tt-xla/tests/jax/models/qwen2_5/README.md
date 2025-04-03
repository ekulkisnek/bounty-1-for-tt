# Qwen2.5-7B Tensor-Parallel JAX Implementation

This directory contains a tensor-parallel implementation of the Qwen2.5-7B model using JAX and Flax. The implementation is designed to run efficiently on multiple devices using JAX's tensor parallelism features.

## Model Architecture

Qwen2.5-7B is a transformer-based language model with the following specifications:

- 7 billion parameters
- 28 transformer layers
- 3584 hidden size
- 18944 intermediate size (MLP)
- 28 attention heads
- 4 key-value heads (Grouped Query Attention)
- RMSNorm for layer normalization
- SwiGLU activation in the MLP layers
- Rotary position embeddings (RoPE)

This implementation follows the architecture of the original Qwen2.5-7B model, with adaptations for tensor parallelism in JAX.

## Tensor Parallelism

The model supports four different tensor-parallel mesh configurations:

- 2x4: 2 devices for batch parallelism, 4 devices for model parallelism
- 1x8: 8 devices for model parallelism
- 1x32: 32 devices for model parallelism
- 8x4: 8 devices for batch parallelism, 4 devices for model parallelism

Tensor parallelism is implemented using JAX's device mesh and sharding primitives. The model parameters are sharded across devices according to the following pattern:

- **Attention Projections**: Query, Key, and Value projections are sharded across the model dimension, allowing parallel computation of attention heads.
- **Attention Output Projection**: Sharded to collect attention heads computed on different devices.
- **MLP Projections**: Gate and Up projections are sharded across the model dimension, while Down projection is sharded in the opposite direction.
- **Embeddings**: The embedding layer is optionally sharded for improved memory efficiency.
- **Layer Norms**: Not sharded as they are computationally inexpensive.

## Directory Structure

```
qwen2_5/
├── config.py                # Model configuration utilities
├── model_implementation.py  # Core model implementation
├── tensor_parallel.py       # Tensor-parallel model components
├── weight_loading.py        # Utilities for loading pretrained weights
├── verify_gsm8k_scores.py   # Verification of GSM8K scores
├── verify_tensor_parallel_bounty.py  # Verification of bounty requirements
├── interactive_inference.py # Interactive chat with the model
├── run_inference_test.sh    # Convenience script for running tests
└── README.md                # This documentation
```

## Setup

1. Install dependencies:
```bash
pip install jax jaxlib flax transformers datasets safetensors tqdm
```

2. Set up JAX for tensor parallelism:
```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=32"
```

3. Ensure the Qwen2.5-7B model weights are available at the correct location:
```
tt-bounty-1/
├── tt-xla/
│   └── tests/
│       └── jax/
│           └── models/
│               └── qwen2_5/
└── qwen2.5-7b/
    ├── config.json
    ├── tokenizer.json
    ├── model.safetensors.index.json
    ├── model-00001-of-00004.safetensors
    ├── model-00002-of-00004.safetensors
    ├── model-00003-of-00004.safetensors
    └── model-00004-of-00004.safetensors
```

Note: The model weights should be located at the path `../../../../qwen2.5-7b` relative to this directory.

## Usage

### Running Tests with the Convenience Script

The `run_inference_test.sh` script provides an easy way to run tests with various mesh configurations:

```bash
# Run interactive chat with default 1x8 mesh
./run_inference_test.sh

# Run with a different mesh shape
./run_inference_test.sh -s 2x4

# Run GSM8K benchmark
./run_inference_test.sh -g

# Run with custom model path
./run_inference_test.sh -m /path/to/qwen2.5-7b
```

For more options:
```bash
./run_inference_test.sh -h
```

### Interactive Inference

You can run the interactive inference script directly:

```bash
# Use default model path and 1x8 mesh shape
python interactive_inference.py

# Specify model path and mesh shape
python interactive_inference.py --model_path ../../../../qwen2.5-7b --mesh_shape 2x4
```

### Verify Tensor Parallelism

To verify that tensor parallelism works correctly with all required mesh shapes:

```bash
python verify_tensor_parallel_bounty.py
```

This will test each of the required mesh shapes (2x4, 1x8, 1x32, 8x4) and verify that the model can be initialized and produce reasonable outputs.

### Verify GSM8K Scores

To verify GSM8K scores match between standard and tensor-parallel implementations:

```bash
python verify_gsm8k_scores.py --model_path ../../../../qwen2.5-7b --mesh_shapes 1x8
```

## Implementation Details

### Weight Loading

This implementation loads the actual Qwen2.5-7B weights from the safetensors files. The weights are automatically sharded according to the device mesh configuration:

1. **Setup**:
   - The `load_qwen_weights` function in `weight_loading.py` loads weights from safetensors files
   - It uses the `model.safetensors.index.json` file to locate parameters
   - No dummy weights are used - all weights come from the real model

2. **Loading Process**:
   - Weights are loaded directly from the safetensors files
   - Names are converted from HuggingFace format to Flax format
   - Weights are sharded according to their partition specs

3. **Verification**:
   - The `verify_tensor_parallel_bounty.py` script verifies that the model works with all required mesh shapes
   - It tests loading the weights and running basic inference
   - The `verify_gsm8k_scores.py` script verifies accuracy against GSM8K problems

### Tensor Parallelism

The tensor parallelism implementation follows these core principles:

1. **Device Mesh**:
   - A 2D mesh with 'batch' and 'model' dimensions is created
   - The 'model' dimension is used for model parallelism
   - The 'batch' dimension is used for data parallelism

2. **Parameter Sharding**:
   - The `create_partition_specs` function in `config.py` defines how parameters are sharded
   - It creates PartitionSpec objects for each parameter based on its role

3. **Sharded Computation**:
   - The tensor-parallel model classes in `tensor_parallel.py` implement sharded computation
   - They utilize `with_sharding_constraint` to ensure proper sharding
   - Communication between devices is handled automatically by JAX

## Supported Mesh Shapes

All the following mesh shapes are fully supported and tested:

| Mesh Shape | Total Devices | Description |
|------------|---------------|-------------|
| 2x4        | 8             | 2 batch x 4 model parallel |
| 1x8        | 8             | Pure model parallelism with 8 devices |
| 1x32       | 32            | Pure model parallelism with 32 devices |
| 8x4        | 32            | 8 batch x 4 model parallel |

## Troubleshooting

If you encounter issues, here are some common solutions:

1. **Not enough devices error**:
   - Make sure you've set the XLA_FLAGS environment variable to simulate the required number of devices:
   ```bash
   export XLA_FLAGS="--xla_force_host_platform_device_count=32"
   ```

2. **Model weights not found**:
   - Check that the Qwen2.5-7B weights are in the correct location (../../../../qwen2.5-7b)
   - Alternatively, specify the model path explicitly with `--model_path`

3. **Memory issues**:
   - For large mesh shapes like 1x32 or 8x4, you may need more RAM
   - Try reducing the batch size by editing the script

## References

- [JAX Device Mesh Documentation](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [Qwen2.5 Model Details](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [TT-Forge Documentation](https://github.com/tenstorrent/tt-forge)
- [TT-xla Repository](https://github.com/tenstorrent/tt-xla)