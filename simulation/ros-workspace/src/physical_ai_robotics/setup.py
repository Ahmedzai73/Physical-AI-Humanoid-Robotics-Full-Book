from setuptools import setup
import os
from glob import glob

package_name = 'physical_ai_robotics'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.*')),
        # Include all URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.*')),
        # Include all world files
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Physical AI Team',
    maintainer_email='maintainer@physical-ai-robotics.org',
    description='Physical AI & Humanoid Robotics Examples and Tutorials',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'minimal_publisher = physical_ai_robotics.minimal_publisher:main',
            'minimal_subscriber = physical_ai_robotics.minimal_subscriber:main',
            'robot_controller = physical_ai_robotics.robot_controller:main',
            'sensor_fusion_node = physical_ai_robotics.sensor_fusion_node:main',
            'navigation_node = physical_ai_robotics.navigation_node:main',
        ],
    },
)