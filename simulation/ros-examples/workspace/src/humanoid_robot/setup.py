from setuptools import setup
import os
from glob import glob

package_name = 'humanoid_robot'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        # Include all URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        # Include all worlds
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Textbook Maintainer',
    maintainer_email='example@textbook.com',
    description='Example humanoid robot package for Physical AI & Humanoid Robotics textbook',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'humanoid_controller = humanoid_robot.humanoid_controller:main',
            'humanoid_sensor_processor = humanoid_robot.sensor_processor:main',
        ],
    },
)