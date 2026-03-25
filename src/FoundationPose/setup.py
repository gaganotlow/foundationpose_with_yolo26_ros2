from setuptools import setup
from glob import glob
import sys

package_name = 'foundationpose'

setup(
    name=package_name,
    version='0.0.0',
    packages=['foundationpose_ros'],
    options={
        'build_scripts': {
            'executable': sys.executable,
        },
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RYKJ',
    maintainer_email='rykj@example.com',
    description='FoundationPose 6D pose estimation as a ROS2 service node',
    license='NVIDIA Proprietary',
    entry_points={
        'console_scripts': [
            'fp_service = foundationpose_ros.fp_service_node:main',
        ],
    },
)
