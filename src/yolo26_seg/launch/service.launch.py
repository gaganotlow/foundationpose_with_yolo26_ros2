from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    engine_file_arg = DeclareLaunchArgument(
        'engine_file',
        default_value='/media/rykj/nvme/jetson/ga/code/niusuo_perception/models/seg26_s_640_table_20260317.engine',
        description='TensorRT引擎文件路径'
    )
    conf_thresh_arg = DeclareLaunchArgument(
        'conf_thresh',
        default_value='0.5',
        description='置信度阈值'
    )
    service_name_arg = DeclareLaunchArgument(
        'service_name',
        default_value='yolo_detect',
        description='ROS2服务名称'
    )
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/right_camera/color/image_raw',
        description='RGB图像话题'
    )
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/right_camera/depth/image_raw',
        description='深度图像话题'
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/right_camera/color/camera_info',
        description='相机信息话题'
    )

    service_node = Node(
        package='yolo26_seg',
        executable='yolo_service',
        name='yolo26_service_node',
        output='screen',
        parameters=[{
            'engine_file':         LaunchConfiguration('engine_file'),
            'conf_thresh':         LaunchConfiguration('conf_thresh'),
            'service_name':        LaunchConfiguration('service_name'),
            'input_topic':         LaunchConfiguration('input_topic'),
            'depth_topic':         LaunchConfiguration('depth_topic'),
            'camera_info_topic':   LaunchConfiguration('camera_info_topic'),
        }]
    )

    return LaunchDescription([
        engine_file_arg,
        conf_thresh_arg,
        service_name_arg,
        input_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        service_node,
    ])
