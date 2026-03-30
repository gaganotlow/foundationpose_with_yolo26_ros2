#!/bin/bash
# 一键构建 ros2_ws 所有自定义包（接口、相机驱动、YOLO、FoundationPose）
# 用法：bash build_all.sh

set -e
WS=/media/rykj/nvme/jetson/ga/code/test_ws
CONDA=/media/rykj/nvme/jetson/miniconda3

source /opt/ros/humble/setup.bash
source "$CONDA/etc/profile.d/conda.sh"

cd "$WS"

echo "========== [1/3] 构建接口包 + 相机驱动（系统 Python）=========="
conda deactivate 2>/dev/null || true
colcon build --symlink-install --packages-select yolo_interfaces fp_interfaces orbbec_camera_msgs orbbec_camera orbbec_description

echo "========== [2/3] 构建 yolo26_seg（yolo 环境）=========="
conda activate yolo
colcon build --symlink-install --packages-select yolo26_seg
conda deactivate

echo "========== [3/3] 构建 foundationpose（foundationpose_ga 环境）=========="
conda activate foundationpose_ga
colcon build --symlink-install --packages-select foundationpose
conda deactivate

echo "========== [4/4] 构建 foundationpose_tensorrt（激活 CUDA/TensorRT）=========="

conda activate foundationpose_ga

# 通过 deps_jetson.sh 激活 CUDA/TensorRT 环境（系统路径放末尾，不覆盖 conda 包）
FP_TRT_SRC="$WS/src/FoundationPose-TensorRT"
source "$FP_TRT_SRC/scripts/deps_jetson.sh" && activate_deps

colcon build --symlink-install --packages-select foundationpose_tensorrt

echo "========== 修正 foundationpose_tensorrt shebang =========="
CONDA_PYTHON="$CONDA/envs/foundationpose_ga/bin/python3"
FP_TRT_ENTRY="$WS/install/foundationpose_tensorrt/lib/foundationpose_tensorrt/fp_service"
sed -i "1s|^#\!.*|#!${CONDA_PYTHON}|" "$FP_TRT_ENTRY"
echo "✓ shebang 已修正为: $(head -1 $FP_TRT_ENTRY)"

echo "========== source install/setup.bash =========="
source "$WS/install/setup.bash"
echo "全部构建完成！"
