#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Script to run inference tests on Qwen2.5-7B model with different mesh shapes
# This script sets up the environment and runs the interactive inference script

# Default values
MODEL_PATH="../../../../qwen2.5-7b"
MESH_SHAPE="1x8"
RUN_GSM8K=false

# Function to display usage
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -m, --model-path PATH   Path to model weights (default: ../../../../qwen2.5-7b)"
  echo "  -s, --mesh-shape SHAPE  Mesh shape in format AxB (default: 1x8)"
  echo "                          Supported shapes: 2x4, 1x8, 1x32, 8x4"
  echo "  -g, --gsm8k             Run GSM8K benchmark instead of interactive chat"
  echo "  -h, --help              Display this help message"
  echo ""
  echo "Examples:"
  echo "  $0                      Run with default settings (1x8 mesh)"
  echo "  $0 -s 2x4               Run with 2x4 mesh shape"
  echo "  $0 -g -s 1x32           Run GSM8K benchmark with 1x32 mesh shape"
  echo "  $0 -m /path/to/model    Use custom model path"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    -s|--mesh-shape)
      MESH_SHAPE="$2"
      shift 2
      ;;
    -g|--gsm8k)
      RUN_GSM8K=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model path $MODEL_PATH does not exist"
  exit 1
fi

# Parse mesh shape to determine number of devices needed
read -r ROWS COLS <<< $(echo $MESH_SHAPE | tr 'x' ' ')
NUM_DEVICES=$((ROWS * COLS))

# Setup environment for device simulation
echo "Setting up environment for $MESH_SHAPE mesh shape ($NUM_DEVICES devices)"
export XLA_FLAGS="--xla_force_host_platform_device_count=$NUM_DEVICES"

# Print current settings
echo "Running with the following settings:"
echo "- Model path: $MODEL_PATH"
echo "- Mesh shape: $MESH_SHAPE ($ROWS×$COLS)"
echo "- GSM8K benchmark: $RUN_GSM8K"
echo "- XLA_FLAGS: $XLA_FLAGS"
echo ""

# Run the appropriate script
if [ "$RUN_GSM8K" = true ]; then
  # Run GSM8K benchmark
  echo "Running GSM8K benchmark..."
  python verify_gsm8k_scores.py --model_path "$MODEL_PATH" --mesh_shape "$MESH_SHAPE" --num_samples 20
else
  # Run interactive inference
  echo "Running interactive inference..."
  python interactive_inference.py --model_path "$MODEL_PATH" --mesh_shape "$MESH_SHAPE"
fi 