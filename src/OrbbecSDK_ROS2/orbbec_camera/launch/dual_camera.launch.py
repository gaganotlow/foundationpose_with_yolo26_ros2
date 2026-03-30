import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_dir = get_package_share_directory('orbbec_camera')
    launch_file_dir = os.path.join(package_dir, 'launch')

    # Left camera configuration (serial: CP32942000JZ)
    left_camera = GroupAction([
        PushRosNamespace('left_camera'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'start.launch.py')
            ),
            launch_arguments={
                'camera_name': 'left_camera',
                'serial_number': 'CP32942000JZ',
                'usb_port': '2-3.1',
                'enable_color': 'true',
                'enable_depth': 'true',
                'enable_point_cloud': 'true',
                'color_width': '640',
                'color_height': '480',
                'depth_width': '640',
                'depth_height': '480',
            }.items()
        )
    ])

    # Right camera configuration (serial: CP2C55300098)
    right_camera = GroupAction([
        PushRosNamespace('right_camera'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'start.launch.py')
            ),
            launch_arguments={
                'camera_name': 'right_camera',
                'serial_number': 'CP2C55300098',
                'usb_port': '2-3.2',
                'enable_color': 'true',
                'enable_depth': 'true',
                'enable_point_cloud': 'true',
                'color_width': '640',
                'color_height': '480',
                'depth_width': '640',
                'depth_height': '480',
            }.items()
        )
    ])

    return LaunchDescription([
        left_camera,
        right_camera,
    ])
