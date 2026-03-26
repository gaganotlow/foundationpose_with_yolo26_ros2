# FoundationPose with YOLO26 ROS2

一个把 YOLO26 和 FoundationPose 结合起来的 ROS2 项目，用来做物体的 6D 位姿估计。

## 这个项目是干什么的？

简单来说，就是让机器人能够"看懂"物体在哪里、怎么摆放的。先用 YOLO26 找到物体并分割出来，然后用 FoundationPose 算出物体的精确位置和姿态（6 个自由度：xyz 位置 + 旋转角度）。

整个系统跑在 ROS2 上，可以实时处理相机数据。我在开发的时候主要用 Orbbec 深度相机测试的。

## 项目结构

代码主要分成这几块：

- **yolo26_seg** - YOLO26 检测节点，负责找物体和分割。用了 TensorRT 加速，速度还不错
- **foundationpose** - FoundationPose 位姿估计服务，这是核心部分，算物体的 6D 姿态
- **yolo_interfaces** 和 **fp_interfaces** - 自定义的 ROS2 消息接口，让各个节点能互相通信
- **OrbbecSDK_ROS2** - Orbbec 相机的驱动，我用的是他们的深度相机

## 需要准备什么

开发环境：
- Ubuntu 22.04（其他版本没试过，不保证能跑）
- ROS2 Humble
- Python 3.8 或更高
- CUDA 11.x+（没 GPU 的话 FoundationPose 会很慢）
- Conda（用来管理不同的 Python 环境）

这个项目用了两个独立的 Conda 环境：
- `yolo` - 跑 YOLO26 检测
- `foundationpose_ga` - 跑 FoundationPose 位姿估计

为什么要分两个环境？因为这两个算法的依赖有冲突，放一起会出问题。

## 怎么安装

### 1. 先把代码拉下来

```bash
git clone https://github.com/gaganotlow/foundationpose_with_yolo26_ros2.git
cd foundationpose_with_yolo26_ros2
```

### 2. 装 ROS2 的依赖

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
```

### 3. 创建 Conda 环境

先装 YOLO 的环境：
```bash
conda create -n yolo python=3.8
conda activate yolo
pip install ultralytics tensorrt opencv-python
```

再装 FoundationPose 的环境：
```bash
conda create -n foundationpose_ga python=3.8
conda activate foundationpose_ga
# 这里需要装 FoundationPose 的依赖，具体看 FoundationPose 官方文档
# 主要是 PyTorch、CUDA 相关的库
```

### 4. 编译项目

最简单的方法是用我写的一键脚本：

```bash
bash build_all.sh
```

如果想自己一步步来，可以这样：

```bash
# 先编译接口包（这个不需要特定环境）
colcon build --symlink-install --packages-select yolo_interfaces fp_interfaces

# 编译 YOLO 节点（记得切换到 yolo 环境）
conda activate yolo
colcon build --symlink-install --packages-select yolo26_seg

# 编译 FoundationPose 节点（切换到 foundationpose_ga 环境）
conda activate foundationpose_ga
colcon build --symlink-install --packages-select foundationpose
```

## 怎么用

启动的时候需要开三个终端（或者用 tmux），因为每个节点要在不同的环境里跑。

**终端 1 - 启动 FoundationPose 服务：**
```bash
source install/setup.bash
conda activate foundationpose_ga
ros2 launch foundationpose service.launch.py
```

**终端 2 - 启动 YOLO 检测：**
```bash
source install/setup.bash
conda activate yolo
ros2 run yolo26_seg yolo_node
```

**终端 3 - 启动相机：**
```bash
source install/setup.bash
ros2 launch orbbec_camera camera.launch.py
```

启动顺序随意，但建议先把相机开起来，这样其他节点能直接收到图像数据。

## 目录结构

```
foundationpose_with_yolo26_ros2/
├── src/
│   ├── FoundationPose/          # FoundationPose 算法核心
│   │   ├── foundationpose_ros/  # ROS2 封装代码
│   │   ├── launch/              # 启动文件
│   │   └── ...
│   ├── yolo26_seg/              # YOLO26 检测节点
│   ├── yolo_interfaces/         # YOLO 消息接口定义
│   ├── fp_interfaces/           # FoundationPose 消息接口定义
│   └── OrbbecSDK_ROS2/          # Orbbec 相机驱动
├── build_all.sh                 # 一键编译脚本（推荐用这个）
├── create_ros2_pkg.sh           # 创建新 ROS2 包的脚本
└── README.md                    # 就是你现在看的这个文件
```

## 主要功能

- 实时物体检测和分割（基于 YOLOv26）
- 6D 位姿估计（xyz 位置 + 旋转，精度还挺高的）
- TensorRT 加速（不然 YOLO 会比较慢）
- 支持深度相机（我用的 Orbbec，其他深度相机理论上也行）
- ROS2 服务接口（方便和其他 ROS2 节点集成）
- 多环境管理（虽然麻烦点，但能避免依赖冲突）

## 一些坑和注意事项

1. **环境切换**：每次启动节点前记得切换到对应的 Conda 环境，不然会报各种奇怪的错
2. **GPU 内存**：FoundationPose 比较吃显存，如果你的 GPU 显存小于 8GB 可能会有问题
3. **TensorRT 版本**：确保 TensorRT 版本和 CUDA 版本匹配，不然可能编译不过
4. **相机标定**：如果换了相机，记得重新标定，不然位姿估计会不准

## 许可证

Apache License 2.0

## 联系方式

有问题可以提 issue

## 参考资料

- [FoundationPose](https://github.com/NVlabs/FoundationPose) - NVIDIA 的 6D 位姿估计算法
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO 系列的官方实现
- [ROS2 Humble 文档](https://docs.ros.org/en/humble/) - ROS2 的官方文档

---

如果这个项目对你有帮助，欢迎 star ⭐
