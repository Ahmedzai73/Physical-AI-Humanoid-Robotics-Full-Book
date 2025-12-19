from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Safety parameters
    declare_min_obstacle_distance_cmd = DeclareLaunchArgument(
        'min_obstacle_distance',
        default_value='0.5',
        description='Minimum safe distance to obstacles'
    )

    declare_max_linear_speed_cmd = DeclareLaunchArgument(
        'max_linear_speed',
        default_value='0.5',
        description='Maximum allowed linear speed'
    )

    declare_max_angular_speed_cmd = DeclareLaunchArgument(
        'max_angular_speed',
        default_value='1.0',
        description='Maximum allowed angular speed'
    )

    # Safety guardrails node
    safety_guardrails_node = Node(
        package='physical_ai_robotics',
        executable='safety_guardrails',
        name='safety_guardrails_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'min_obstacle_distance': LaunchConfiguration('min_obstacle_distance')},
            {'max_linear_speed': LaunchConfiguration('max_linear_speed')},
            {'max_angular_speed': LaunchConfiguration('max_angular_speed')},
            {'collision_threshold': 0.3},
            {'emergency_stop_distance': 0.2}
        ],
        remappings=[
            ('/unsafe_cmd_vel', '/vla/unsafe_cmd_vel'),
            ('/safe_cmd_vel', '/cmd_vel'),
            ('/scan', '/scan'),
            ('/odom', '/odom'),
        ],
        output='screen'
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_min_obstacle_distance_cmd)
    ld.add_action(declare_max_linear_speed_cmd)
    ld.add_action(declare_max_angular_speed_cmd)

    # Add nodes
    ld.add_action(safety_guardrails_node)

    return ld