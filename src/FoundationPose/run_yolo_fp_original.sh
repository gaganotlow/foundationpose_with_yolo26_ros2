#!/usr/bin/env bash
# 串起来：① ROS 调 yolo_detect + CameraInfo → npz  ② conda 里原版 FoundationPose register → 位姿文件
#
# 数据流说明：
#   - 相机内参 K：来自持续发布的 CameraInfo（仅用于标定矩阵）。
#   - RGB、Depth、Mask：只有在 yolo_detect 按类别请求成功之后，才出现在该次 Service 的
#     Response 里；本脚本第 1 步正是等 YOLO 返回后再解码并写入 npz。
#   - 默认 mesh：类别 k2c → ${JETSON_ROOT}/K2.obj（与本机 /media/rykj/nvme/jetson/K2.obj 一致）。
#
# 用法示例：
#   bash run_yolo_fp_original.sh --class-name k2c
#   bash run_yolo_fp_original.sh --class-name k2c --mesh /media/rykj/nvme/jetson/K2.obj
#   bash run_yolo_fp_original.sh --class-name k2c -- --camera-info-topic /right_camera/color/camera_info
#
# 环境变量（可选）：
#   JETSON_ROOT   默认 /media/rykj/nvme/jetson
#   CONDA_ENV     默认 foundationpose（原版 FP 的 conda 环境名）
#   WAIT_YOLO_SERVICE_SEC   等待 yolo_detect 服务（秒），默认 600
#   DETECT_RETRY_INTERVAL   未检出时重试间隔（秒），默认 2
#   DETECT_RETRY_FOR_SEC    检测重试总时长，0=不限制
#   AUTO_INSTALL_DEPS=1     第2步缺依赖时自动 pip 安装（默认不自动）
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
JETSON_ROOT="${JETSON_ROOT:-/media/rykj/nvme/jetson}"
CONDA_ENV="${CONDA_ENV:-foundationpose}"

CLASS_NAME="k2c"
MESH=""
MESH_EXPLICIT=0
NPZ="/tmp/fp_input.npz"
POSE_OUT="/tmp/ob_in_cam.txt"
EST_REFINE_ITER="5"
WAIT_YOLO_SERVICE_SEC="${WAIT_YOLO_SERVICE_SEC:-600}"
DETECT_RETRY_INTERVAL="${DETECT_RETRY_INTERVAL:-2}"
DETECT_RETRY_FOR_SEC="${DETECT_RETRY_FOR_SEC:-0}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-0}"
CONDA_BIN=""

DUMP_EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --class-name)
      CLASS_NAME="$2"
      shift 2
      ;;
    --mesh)
      MESH="$2"
      MESH_EXPLICIT=1
      shift 2
      ;;
    --npz)
      NPZ="$2"
      shift 2
      ;;
    --pose-out)
      POSE_OUT="$2"
      shift 2
      ;;
    --est-refine-iter)
      EST_REFINE_ITER="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --wait-yolo-service-sec)
      WAIT_YOLO_SERVICE_SEC="$2"
      shift 2
      ;;
    --detect-retry-interval)
      DETECT_RETRY_INTERVAL="$2"
      shift 2
      ;;
    --detect-retry-for-sec)
      DETECT_RETRY_FOR_SEC="$2"
      shift 2
      ;;
    --)
      shift
      DUMP_EXTRA=("$@")
      break
      ;;
    -h|--help)
      sed -n '1,25p' "$0"
      exit 0
      ;;
    *)
      echo "未知参数: $1  （可用 -- 把剩余参数传给 ros_dump，如 -- --camera-info-topic ...）" >&2
      exit 1
      ;;
  esac
done

# 未显式传 --mesh 时：按 YOLO 类别选默认 obj（k2c → K2.obj，与 ${JETSON_ROOT}/K2.obj 对应）
if [[ "${MESH_EXPLICIT}" -eq 0 ]]; then
  _cls_lc="$(echo "${CLASS_NAME}" | tr '[:upper:]' '[:lower:]')"
  case "${_cls_lc}" in
    k2c)
      MESH="${JETSON_ROOT}/K2.obj"
      ;;
    j2)
      MESH="${JETSON_ROOT}/J2.obj"
      ;;
    *)
      MESH="${JETSON_ROOT}/K2.obj"
      echo "提示: 未指定 --mesh，类别 '${CLASS_NAME}' 无内置映射，已默认使用 ${MESH}" >&2
      ;;
  esac
fi

if [[ ! -f "${MESH}" ]]; then
  echo "错误: mesh 文件不存在: ${MESH}" >&2
  exit 1
fi

init_conda() {
  if [[ -x "/media/rykj/nvme/jetson/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/media/rykj/nvme/jetson/miniconda3/bin/conda"
    return 0
  fi
  if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    CONDA_BIN="${HOME}/miniconda3/bin/conda"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
    return 0
  fi
  local _candidates=(
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
    "/media/rykj/nvme/jetson/miniconda3/etc/profile.d/conda.sh"
  )
  for f in "${_candidates[@]}"; do
    if [[ -f "$f" ]]; then
      set +u
      # shellcheck source=/dev/null
      source "$f"
      set -u
      if command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
        return 0
      fi
      return 0
    fi
  done
  echo "未找到 conda，请先安装或把 conda 加入 PATH" >&2
  return 1
}

echo ""
echo "------------------------------------------------------------------"
echo " 步骤 1/2：等待 CameraInfo、YOLO 服务与成功检测（可在另一终端启动 YOLO）"
echo "------------------------------------------------------------------"
# setup.bash 会读可能未设置的变量；set -u 下会报错，source 时临时关闭 nounset
set +u
source /opt/ros/humble/setup.bash
source "${JETSON_ROOT}/ga/code/ros2_ws/install/setup.bash"
set -u
/usr/bin/python3 "${ROOT}/ros_dump_yolo_for_foundationpose.py" \
  --class-name "${CLASS_NAME}" \
  --out "${NPZ}" \
  --wait-yolo-service-sec "${WAIT_YOLO_SERVICE_SEC}" \
  --detect-retry-interval "${DETECT_RETRY_INTERVAL}" \
  --detect-retry-for-sec "${DETECT_RETRY_FOR_SEC}" \
  "${DUMP_EXTRA[@]}"

echo ""
echo "------------------------------------------------------------------"
echo " 步骤 2/2：已收到 RGB-D 与 mask → 正在计算 6D 位姿（FoundationPose）"
echo "------------------------------------------------------------------"
init_conda

echo "Conda env: ${CONDA_ENV}"
echo "Conda bin: ${CONDA_BIN}"
ENV_PREFIX="$("${CONDA_BIN}" env list | awk -v e="${CONDA_ENV}" '$1==e {print $NF; exit}')"
if [[ -z "${ENV_PREFIX}" ]]; then
  echo "错误: conda 环境不存在: ${CONDA_ENV}" >&2
  exit 4
fi

# 避免 conda run 在已激活其它 conda 环境 + ROS 变量下出现前缀污染，改为显式 activate
set +u
source "$(dirname "${CONDA_BIN}")/../etc/profile.d/conda.sh"
conda deactivate >/dev/null 2>&1 || true
conda activate "${CONDA_ENV}"
set -u

PY_INFO="$(python -c "import sys; print(sys.prefix); print(sys.executable)")"
PY_PREFIX="$(echo "${PY_INFO}" | sed -n '1p')"
PY_EXE="$(echo "${PY_INFO}" | sed -n '2p')"
echo "Python prefix: ${PY_PREFIX}"
echo "Python exe:    ${PY_EXE}"
if [[ "${PY_PREFIX}" != "${ENV_PREFIX}" ]]; then
  echo "错误: 当前并未真正进入 conda 环境 '${CONDA_ENV}'。" >&2
  echo "期望 sys.prefix = ${ENV_PREFIX}" >&2
  exit 4
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误: conda activate 后未设置 CONDA_PREFIX" >&2
  exit 4
fi
CONDA_PY="${CONDA_PREFIX}/bin/python"
if [[ ! -x "${CONDA_PY}" ]]; then
  echo "错误: 找不到 conda python: ${CONDA_PY}" >&2
  exit 4
fi
echo "Conda python:  ${CONDA_PY}"

if ! "${CONDA_PY}" -c "import trimesh" >/dev/null 2>&1; then
  if [[ "${AUTO_INSTALL_DEPS}" == "1" ]]; then
    echo "缺少 trimesh，正在自动安装..."
    "${CONDA_PY}" -m pip install trimesh
  else
    echo "错误: 环境 '${CONDA_ENV}' 缺少 trimesh。" >&2
    echo "请执行: conda run -n ${CONDA_ENV} --no-capture-output pip install trimesh" >&2
    echo "或设置 AUTO_INSTALL_DEPS=1 后重试。" >&2
    exit 2
  fi
fi

if ! "${CONDA_PY}" -c "import torch" >/dev/null 2>&1; then
  echo "错误: 环境 '${CONDA_ENV}' 缺少 torch（FoundationPose 必需）。" >&2
  echo "请先在该环境安装 FoundationPose 依赖（含 torch/cuda）。" >&2
  echo "提示: 你当前第2步使用的是错误 Python 时最常见，先确认上面 Python 路径应为 miniconda/envs/${CONDA_ENV}/bin/python。" >&2
  exit 3
fi

"${CONDA_PY}" "${ROOT}/run_register_from_npz.py" \
  --mesh "${MESH}" \
  --npz "${NPZ}" \
  --pose-out "${POSE_OUT}" \
  --est-refine-iter "${EST_REFINE_ITER}"

echo ""
echo "完成。位姿文件: ${POSE_OUT}"
