from setuptools import setup
from glob import glob
import os
import sys

package_name = 'yolo26_seg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
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
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='RYKJ',
    maintainer_email='rykj@example.com',
    description='YOLO26 Segmentation Detection Node with TensorRT',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'yolo_node = yolo26_seg.yolo_node:main',
            'yolo_offline = yolo26_seg.yolo_node:main_offline',
            'yolo_service = yolo26_seg.yolo_service:main',
        ],
    },
)
