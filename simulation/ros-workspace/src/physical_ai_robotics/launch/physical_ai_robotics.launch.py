from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),

        # Minimal publisher node
        Node(
            package='physical_ai_robotics',
            executable='minimal_publisher',
            name='minimal_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        ),

        # Minimal subscriber node
        Node(
            package='physical_ai_robotics',
            executable='minimal_subscriber',
            name='minimal_subscriber',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        ),

        # Robot controller node
        Node(
            package='physical_ai_robotics',
            executable='robot_controller',
            name='robot_controller',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        ),

        # Sensor fusion node
        Node(
            package='physical_ai_robotics',
            executable='sensor_fusion_node',
            name='sensor_fusion_node',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        ),

        # Navigation node
        Node(
            package='physical_ai_robotics',
            executable='navigation_node',
            name='navigation_node',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        ),
    ])