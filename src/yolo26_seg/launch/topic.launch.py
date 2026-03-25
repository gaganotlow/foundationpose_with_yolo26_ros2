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
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/right_camera/color/image_raw',
        description='输入图像话题'
    )
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/yolo26/result_image',
        description='输出结果图像话题'
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/right_camera/color/camera_info',
        description='相机内参话题'
    )
    conf_thresh_arg = DeclareLaunchArgument(
        'conf_thresh',
        default_value='0.3',
        description='置信度阈值'
    )
    save_results_arg = DeclareLaunchArgument(
        'save_results',
        default_value='False',
        description='是否保存推理结果到磁盘'
    )
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='/media/rykj/nvme/jetson/ga/code/yolo26_seg/output_ros',
        description='结果保存目录'
    )

    yolo_node = Node(
        package='yolo26_seg',
        executable='yolo_node',
        name='yolo26_seg_node',
        output='screen',
        parameters=[{
            'engine_file':        LaunchConfiguration('engine_file'),
            'input_topic':        LaunchConfiguration('input_topic'),
            'output_topic':       LaunchConfiguration('output_topic'),
            'camera_info_topic':  LaunchConfiguration('camera_info_topic'),
            'conf_thresh':        LaunchConfiguration('conf_thresh'),
            'save_results':       LaunchConfiguration('save_results'),
            'output_dir':         LaunchConfiguration('output_dir'),
        }]
    )

    return LaunchDescription([
        engine_file_arg,
        input_topic_arg,
        output_topic_arg,
        camera_info_topic_arg,
        conf_thresh_arg,
        save_results_arg,
        output_dir_arg,
        yolo_node,
    ])
