# FoundationPose-TensorRT ROS2 服务集成

完全按照原版 FoundationPose 的 ROS2 服务架构实现的 TensorRT 加速版本。

## 快速开始

### 1. 构建
```bash
cd /media/rykj/nvme/jetson/ga/code/test_ws

# 构建包
colcon build --symlink-install --packages-select foundationpose_tensorrt

# Source 环境
source install/setup.bash
```

### 2. 启动服务
```bash
# 使用 launch 文件启动
ros2 launch foundationpose_tensorrt service.launch.py \
  mesh_file:=/path/to/your/mesh.obj \
  camera_info_topic:=/right_camera/color/camera_info
```

### 3. 调用服务
```bash
# 命令行调用
ros2 service call /fp_detect fp_interfaces/srv/FpDetect \
  "{class_name: 'k2c', est_refine_iter: 0}"
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mesh_file` | "" | 物体 mesh 文件路径 |
| `yolo_service` | "yolo_detect" | YOLO 服务名称 |
| `camera_info_topic` | "/right_camera/color/camera_info" | 相机内参话题 |
| `est_refine_iter` | 3 | 位姿估计精化迭代次数 |
| `downsample_width` | 256 | 图像下采样宽度 |
| `chunk_size` | 128 | TensorRT chunk 大小 |
| `debug` | 0 | 调试级别 |
| `service_name` | "fp_detect" | 服务名称 |

## 服务接口

**Request:**
- `string class_name` - 目标类别（传给YOLO）
- `int32 est_refine_iter` - 精化迭代次数（0=使用默认）

**Response:**
- `bool success` - 是否成功
- `string message` - 消息
- `geometry_msgs/PoseStamped pose` - 6D位姿

## 工作流程

1. 客户端调用 `/fp_detect` 服务
2. 节点调用 `/yolo_detect` 获取 RGB、Depth、Mask
3. 执行 `reset_scene()` + `add_object()` 进行位姿估计
4. 返回 6D 位姿（PoseStamped）

每次调用都是独立的 register，不保持跟踪状态。

## 与原版对应

| 原版 | TensorRT 版本 |
|------|--------------|
| PyTorch 推理 | TensorRT 推理 |
| conda 环境 | 标准环境 |
| 相同的服务接口 | 相同的服务接口 |
| 相同的工作流程 | 相同的工作流程 |
