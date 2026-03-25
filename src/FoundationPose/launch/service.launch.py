"""
FoundationPose 服务节点 Launch 文件

使用方式：
  ros2 launch foundationpose service.launch.py

可选参数：
  mesh_file           - 物体 .obj 文件路径（默认 demo_data/k2c/mesh/03.obj）
  yolo_service        - YOLO 服务名（默认 yolo_detect）
  camera_info_topic   - 相机内参话题（默认 /right_camera/color/camera_info）
  est_refine_iter     - 精化迭代次数（默认 5）
  debug               - 调试级别 0/1/2（默认 0）
  debug_dir           - 调试输出目录（默认 /tmp/fp_debug）
  conda_python        - conda 环境 Python 路径（默认 foundationpose_ga）

环境要求：
  FoundationPose 依赖（nvdiffrast、trimesh 等）必须在 conda foundationpose_ga 环境中。
  launch 文件通过 prefix 参数直接指定 conda Python，无需手动 conda activate。
  构建命令：
    cd <ros2_ws>
    colcon build --symlink-install --packages-select fp_interfaces foundationpose
    source install/setup.bash
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fp_src_dir = "/media/rykj/nvme/jetson/ga/code/ros2_ws/src/FoundationPose"

    return LaunchDescription([
        # ---- 环境变量：保证 colcon install 后也能找到 FoundationPose 源码 ----
        SetEnvironmentVariable("FOUNDATIONPOSE_DIR", fp_src_dir),

        # ---- Launch 参数声明 ----
        DeclareLaunchArgument(
            "mesh_file",
            default_value=f"{fp_src_dir}/demo_data/k2c/mesh/03.obj",
            description="物体 mesh 文件路径（.obj）",
        ),
        DeclareLaunchArgument(
            "yolo_service",
            default_value="yolo_detect",
            description="YOLO 服务名称",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/right_camera/color/camera_info",
            description="相机内参话题",
        ),
        DeclareLaunchArgument(
            "est_refine_iter",
            default_value="5",
            description="FoundationPose register 精化迭代次数",
        ),
        DeclareLaunchArgument(
            "debug",
            default_value="0",
            description="调试级别：0=关闭 / 1=显示可视化 / 2=保存图像",
        ),
        DeclareLaunchArgument(
            "debug_dir",
            default_value="/media/rykj/nvme/jetson/ga/code/ros2_ws/src/FoundationPose/fp_debug",
            description="调试文件输出目录",
        ),
        DeclareLaunchArgument(
            "service_name",
            default_value="fp_detect",
            description="对外暴露的服务名称",
        ),
        DeclareLaunchArgument(
            "conda_python",
            default_value="/media/rykj/nvme/jetson/miniconda3/envs/foundationpose_ga/bin/python3",
            description="conda 环境 Python 解释器路径（含 FoundationPose 全部依赖）",
        ),

        # ---- FoundationPose 服务节点 ----
        # prefix 指定 conda Python，解决 ros2 launch 子进程使用系统 Python 的问题
        Node(
            package="foundationpose",
            executable="fp_service",
            name="foundationpose_service_node",
            prefix=LaunchConfiguration("conda_python"),
            parameters=[{
                "mesh_file":          LaunchConfiguration("mesh_file"),
                "yolo_service":       LaunchConfiguration("yolo_service"),
                "camera_info_topic":  LaunchConfiguration("camera_info_topic"),
                "est_refine_iter":    LaunchConfiguration("est_refine_iter"),
                "debug":              LaunchConfiguration("debug"),
                "debug_dir":          LaunchConfiguration("debug_dir"),
                "service_name":       LaunchConfiguration("service_name"),
            }],
            output="screen",
            emulate_tty=True,
        ),
    ])
