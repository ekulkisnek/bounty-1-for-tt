#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Script to run tests for the Qwen2.5-7B model implementation

set -e  # Exit on error

# Change to the root directory of the repo
cd "$(dirname "$0")/../../../.."

echo "Running Qwen2.5-7B model tests..."

# Run basic model tests
echo "Running basic model tests..."
python -m pytest -xvs tests/jax/models/qwen2_5/test_model.py::test_model_loading_and_forward_pass

# Run tensor parallel tests
echo "Running tensor parallel tests..."
python -m pytest -xvs tests/jax/models/qwen2_5/test_model.py::test_tensor_parallel_model

# Run infrastructure integration tests
echo "Running infrastructure integration tests..."
python -m pytest -xvs tests/jax/models/qwen2_5/test_model.py::test_tensor_parallel_with_infra

echo "All tests completed!"

# Run example script with demo model
echo "Running example usage script with demo model..."
python tests/jax/models/qwen2_5/example_usage.py --use_demo_model --benchmark

echo "Done!" 