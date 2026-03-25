#!/bin/bash
# 一键构建 ros2_ws 所有自定义包（接口、相机驱动、YOLO、FoundationPose）
# 用法：bash build_all.sh

set -e
WS=/media/rykj/nvme/jetson/ga/code/ros2_ws
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

echo "========== source install/setup.bash =========="
source "$WS/install/setup.bash"
echo "全部构建完成！"
