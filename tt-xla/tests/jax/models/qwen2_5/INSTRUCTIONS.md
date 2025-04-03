# Qwen2.5-7B JAX Tensor Parallelism Verification Instructions

This document provides detailed instructions on how to verify that the Qwen2.5-7B JAX implementation meets the tensor parallelism bounty requirements. The implementation supports flexible tensor parallelism configurations across various device mesh shapes.

## Prerequisites

- Python 3.8+
- JAX 0.4.11+
- FLAX 0.7.2+
- NumPy 1.24+

## Setup Environment

Set up the environment by running:

```bash
# Clone the repository (if not already done)
git clone https://github.com/your-repo/tt-bounty-1.git
cd tt-bounty-1

# Install the required packages
pip install -r requirements.txt
```

## Overview of the Implementation

The implementation consists of several key components:

1. **Base Model Implementation (`model_implementation.py`)**: Contains the standard JAX implementation of Qwen2.5-7B model.
2. **Tensor Parallel Layers (`tensor_parallel.py`)**: Implements tensor-parallel versions of all model components.
3. **Configuration Utilities (`config.py`)**: Provides utilities for mesh creation and configuration.
4. **Verification Scripts**: Multiple scripts to verify the implementation works correctly.

## Key Features of the Tensor Parallel Implementation

1. **Flexible Mesh Configuration**: Supports various mesh shapes (2x4, 1x8, 4x2, 8x1, etc.)
2. **Parameter Sharding**: Model parameters are automatically sharded across devices based on the mesh configuration.
3. **Automatic Rematerialization**: JAX's GSPMD handles the communication between devices.
4. **Efficient Model Parallelism**: Attention layers and MLP layers are sharded for optimal performance.

## Verification Steps

### Step 1: Enable JAX Multidevice Simulation

The implementation uses JAX's simulation capabilities to create virtual devices for testing tensor parallelism. Set the environment variable:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=8"
```

This forces JAX to create 8 simulated CPU devices, allowing testing of different mesh configurations without needing actual hardware.

### Step 2: Run the Verification Script

Run the main verification script to automatically check all requirements:

```bash
python tt-xla/tests/jax/models/qwen2_5/verify_tensor_parallel_bounty.py
```

This script performs several checks:
1. Verifies JAX multidevice simulation is correctly configured
2. Confirms all required mesh configurations are supported
3. Tests tensor parallelism on different mesh shapes
4. Verifies partition specifications for tensor parallelism

### Step 3: Run the Tensor Parallel Simulation

For a more detailed test across all mesh configurations, run:

```bash
python tt-xla/tests/jax/models/qwen2_5/tensor_parallel_simulation.py
```

This script runs the model across four different mesh configurations:
1. **2x4 (2 batch x 4 model)**: Batch dimension sharded across 2 devices, model parameters across 4
2. **1x8 (1 batch x 8 model)**: No batch sharding, full model parallelism across 8 devices
3. **4x2 (4 batch x 2 model)**: Batch dimension sharded across 4 devices, model parameters across 2
4. **8x1 (8 batch x 1 model)**: Pure data parallelism with batch sharded across all 8 devices

The script shows detailed sharding visualizations and compares outputs to ensure tensor parallelism is working correctly.

## Understanding Tensor Parallelism in the Implementation

### Mesh Creation

The `create_device_mesh` function in `tensor_parallel.py` creates a 2D mesh with two named dimensions:
- **'batch'**: For sharding along the batch dimension (data parallelism)
- **'model'**: For sharding along model parameters (model parallelism)

```python
def create_device_mesh(mesh_shape):
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(devices, ('batch', 'model'))
```

### Parameter Sharding

The `get_partition_specs` function defines how different parameters are sharded:

- **Attention Heads**: Input projections (query, key, value) are sharded along the output dimension
- **MLP Layers**: Gate and up projections are sharded along the output dimension
- **Output Projections**: Sharded along the input dimension

```python
# Partition specs for attention
q_p = P(None, 'model')  # Sharded along output dimension
k_p = P(None, 'model')  # Sharded along output dimension
v_p = P(None, 'model')  # Sharded along output dimension
o_p = P('model', None)  # Sharded along input dimension
```

### TensorParallelDense Implementation

The `TensorParallelDense` class shows how tensor parallelism is implemented:

1. Parameters are sharded according to specified axes
2. `with_sharding_constraint` is used to enforce the sharding
3. JAX's GSPMD automatically handles the communication

```python
# Shard the kernel if mesh is provided
if self.mesh is not None:
    kernel = jax.lax.with_sharding_constraint(kernel, kernel_spec)
```

### Mesh Visualization

The verification scripts visualize the mesh and sharding patterns:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                                 CPU 0,1,2,3                                  │
│                                                                              │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                                 CPU 4,5,6,7                                  │
│                                                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

This visualization shows how tensors are sharded across devices in the mesh.

## Verifying Output Consistency

The tensor_parallel_simulation.py script compares model outputs across different mesh configurations:

1. It runs inference with each mesh shape
2. Validates that the output shapes match expected dimensions
3. Compares the top predicted tokens across all configurations
4. Measures the overlap of top 100 tokens to ensure consistency (should be 100%)

## Performance Considerations

The script measures and reports:
1. **Model Initialization Time**: Time to create and initialize the model
2. **Inference Time**: Time to run forward pass with the model

These measurements help verify that tensor parallelism is providing expected computational benefits, with larger model-parallel configurations generally being more efficient for larger models.

## How Tensor Parallelism Works in This Implementation

### 1. Splitting the Model Across Devices

For model parallelism (sharding along the 'model' dimension):
- The model's parameters are split across multiple devices
- Each device holds only a portion of each layer's weights
- Computations are performed in parallel across these devices

### 2. Input and Output Sharding

- Inputs are appropriately sharded using `jax.device_put()` with a `NamedSharding`
- Results are automatically combined using JAX's GSPMD functionality

### 3. Communication Patterns

- All-to-all communication happens when switching between model and data parallelism
- All-gather operations combine partial results across the model dimension
- All-reduce operations combine partial results during the forward pass

## Conclusion

The implementation successfully demonstrates tensor parallelism for the Qwen2.5-7B model using JAX's GSPMD capabilities. It supports flexible mesh configurations, allowing users to adjust the balance between data parallelism and model parallelism based on their specific hardware setup.

By following these instructions, you can verify that the implementation meets all the requirements specified in the bounty, including support for various mesh shapes and efficient tensor parallelism. 