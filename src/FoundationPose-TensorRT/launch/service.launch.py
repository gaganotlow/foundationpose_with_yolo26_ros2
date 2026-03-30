"""
FoundationPose-TensorRT 服务节点 Launch 文件

使用方式：
  ros2 launch foundationpose_tensorrt service.launch.py

可选参数：
  mesh_file           - 物体 .obj 文件路径
  yolo_service        - YOLO 服务名（默认 yolo_detect）
  camera_info_topic   - 相机内参话题（默认 /right_camera/color/camera_info）
  est_refine_iter     - 精化迭代次数（默认 3）
  downsample_width    - 下采样宽度（默认 256）
  chunk_size          - TensorRT chunk 大小（默认 128）
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    # 获取包的安装路径
    pkg_dir = os.path.dirname(os.path.realpath(__file__))
    fp_src_dir = os.path.join(pkg_dir, "..")  # 回到包根目录

    return LaunchDescription([
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
            default_value="/right_camera/right_camera/color/camera_info",
            description="相机内参话题",
        ),
        DeclareLaunchArgument(
            "est_refine_iter",
            default_value="3",
            description="FoundationPose register 精化迭代次数",
        ),
        DeclareLaunchArgument(
            "track_refine_iter",
            default_value="2",
            description="FoundationPose track 精化迭代次数",
        ),
        DeclareLaunchArgument(
            "downsample_width",
            default_value="256",
            description="图像下采样宽度（0=不下采样）",
        ),
        DeclareLaunchArgument(
            "chunk_size",
            default_value="128",
            description="TensorRT 引擎 chunk 大小",
        ),
        DeclareLaunchArgument(
            "debug",
            default_value="0",
            description="调试级别：0=关闭 / 1=显示可视化 / 2=保存图像",
        ),
        DeclareLaunchArgument(
            "debug_dir",
            default_value="/tmp/fp_tensorrt_debug",
            description="调试文件输出目录",
        ),
        DeclareLaunchArgument(
            "service_name",
            default_value="fp_detect",
            description="对外暴露的服务名称",
        ),

        # ---- FoundationPose-TensorRT 服务节点 ----
        Node(
            package="foundationpose_tensorrt",
            executable="fp_service",
            name="foundationpose_tensorrt_service_node",
            parameters=[{
                "mesh_file":          LaunchConfiguration("mesh_file"),
                "yolo_service":       LaunchConfiguration("yolo_service"),
                "camera_info_topic":  LaunchConfiguration("camera_info_topic"),
                "est_refine_iter":    LaunchConfiguration("est_refine_iter"),
                "track_refine_iter":  LaunchConfiguration("track_refine_iter"),
                "downsample_width":   LaunchConfiguration("downsample_width"),
                "chunk_size":         LaunchConfiguration("chunk_size"),
                "debug":              LaunchConfiguration("debug"),
                "debug_dir":          LaunchConfiguration("debug_dir"),
                "service_name":       LaunchConfiguration("service_name"),
            }],
            output="screen",
            emulate_tty=True,
        ),
    ])
