#!/bin/bash

# Jetson-specific dependency activation script
# This script only sets environment variables to use system-installed CUDA and TensorRT
# It does NOT download or install anything

function activate_deps()
{
    # Use system CUDA installation
    CUDA_HOME_JETSON="/usr/local/cuda-12.6"

    # Use system TensorRT installation
    TENSORRT_HOME_JETSON="/usr"

    # Backup current environment
    export PRE_JETSON_PATH="$PATH"
    export PRE_JETSON_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    export PRE_JETSON_PYTHONPATH="$PYTHONPATH"

    # Set CUDA environment variables
    export CUDA_HOME="$CUDA_HOME_JETSON"
    export CUDA_ROOT="$CUDA_HOME_JETSON/bin"
    export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME_JETSON"

    # Set TensorRT environment variables
    export TENSORRT_HOME="$TENSORRT_HOME_JETSON"

    # Update PATH
    export PATH="$CUDA_HOME_JETSON/bin:$PATH"

    # Update LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$CUDA_HOME_JETSON/lib64:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"

    # Add system Python packages to PYTHONPATH for TensorRT access
    # 放在末尾，避免覆盖 conda 环境中的同名包（如 transformations、trimesh 等）
    export PYTHONPATH="$PYTHONPATH:/usr/lib/python3.10/dist-packages"

    echo "✓ Activated Jetson system CUDA 12.6 and TensorRT 10.3.0"
    echo "  CUDA_HOME: $CUDA_HOME"
    echo "  TENSORRT_HOME: $TENSORRT_HOME"
}

function deactivate_deps()
{
    export PATH="$PRE_JETSON_PATH"
    export LD_LIBRARY_PATH="$PRE_JETSON_LD_LIBRARY_PATH"
    export PYTHONPATH="$PRE_JETSON_PYTHONPATH"

    unset PRE_JETSON_PATH
    unset PRE_JETSON_LD_LIBRARY_PATH
    unset PRE_JETSON_PYTHONPATH
    unset CUDA_HOME
    unset CUDA_ROOT
    unset CUDA_TOOLKIT_ROOT_DIR
    unset TENSORRT_HOME

    echo "✓ Deactivated Jetson dependencies"
}
