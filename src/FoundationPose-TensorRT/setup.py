from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'foundationpose_tensorrt'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='FoundationPose-TensorRT ROS2 integration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fp_service = foundationpose_tensorrt.fp_service_node:main',
        ],
    },
)
