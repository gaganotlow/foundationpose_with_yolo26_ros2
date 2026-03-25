#!/usr/bin/env bash
# 仅负责 ROS 侧：用系统 Python 调用 yolo_detect + CameraInfo，写出 fp_input.npz
set -euo pipefail
export PATH="/usr/bin:/usr/local/bin:${PATH}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
set +u
source /opt/ros/humble/setup.bash
source "${JETSON_ROOT:-/media/rykj/nvme/jetson}/ga/code/ros2_ws/install/setup.bash"
set -u
exec /usr/bin/python3 "${ROOT}/ros_dump_yolo_for_foundationpose.py" "$@"
