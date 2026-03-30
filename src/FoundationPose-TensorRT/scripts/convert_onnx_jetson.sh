#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate Jetson dependencies
if ! command -v trtexec &> /dev/null; then
    source "${SCRIPT_DIR}/deps_jetson.sh"
    activate_deps
fi

# Set the desired maximum batch size
# For Jetson, you may want to use a smaller value (e.g., 64, 128) to reduce VRAM usage
chunk_size=128

MODEL_FOLDER_PATH="$SCRIPT_DIR/../weights"
ONNX_DIR="${MODEL_FOLDER_PATH}/onnx"
TRT_PLAN_DIR="${MODEL_FOLDER_PATH}/tensorrt"
REFINER_ONNX_MODEL="${ONNX_DIR}/refine_model.onnx"
SCORER_ONNX_MODEL="${ONNX_DIR}/score_model.onnx"

mkdir -p "${ONNX_DIR}"

if [ ! -f "${REFINER_ONNX_MODEL}" ]; then
   cd "${ONNX_DIR}" && \
   wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/refine_model.onnx' -O refine_model.onnx && \
   wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/score_model.onnx' -O score_model.onnx \
   && cd -
fi

echo "Using models from: ${MODEL_FOLDER_PATH}"
echo "Target platform: Jetson (ARM64) with TensorRT 10.3.0"

mkdir -p "${TRT_PLAN_DIR}"

# Convert models using Jetson's TensorRT
echo "Converting the FoundationPose refine model with max batch size: ${chunk_size}"
trtexec --onnx="${REFINER_ONNX_MODEL}" \
        --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
        --optShapes=input1:${chunk_size}x160x160x6,input2:${chunk_size}x160x160x6 \
        --maxShapes=input1:${chunk_size}x160x160x6,input2:${chunk_size}x160x160x6 \
        --saveEngine="${TRT_PLAN_DIR}/refiner_cs${chunk_size}.plan" \
        --fp16

echo "Converting the FoundationPose score model with max batch size: ${chunk_size}"
trtexec --onnx="${SCORER_ONNX_MODEL}" \
        --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
        --optShapes=input1:${chunk_size}x160x160x6,input2:${chunk_size}x160x160x6 \
        --maxShapes=input1:${chunk_size}x160x160x6,input2:${chunk_size}x160x160x6 \
        --saveEngine="${TRT_PLAN_DIR}/scorer_cs${chunk_size}.plan" \
        --fp16

echo ""
echo "Conversion complete. Engines saved in ${TRT_PLAN_DIR}"
echo ""
echo "Note: Added --fp16 flag for better performance on Jetson"
