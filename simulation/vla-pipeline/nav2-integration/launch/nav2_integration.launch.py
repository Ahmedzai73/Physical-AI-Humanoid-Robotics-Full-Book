from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Navigation parameters
    declare_min_waypoint_distance_cmd = DeclareLaunchArgument(
        'min_waypoint_distance',
        default_value='0.5',
        description='Minimum distance between consecutive waypoints'
    )

    declare_max_route_length_cmd = DeclareLaunchArgument(
        'max_route_length',
        default_value='20',
        description='Maximum number of waypoints in a route'
    )

    declare_safe_navigation_cmd = DeclareLaunchArgument(
        'safe_navigation',
        default_value='true',
        description='Enable safety checks during navigation'
    )

    # Nav2 integration node
    nav2_integration_node = Node(
        package='physical_ai_robotics',
        executable='nav2_integration',
        name='nav2_integration_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'min_waypoint_distance': LaunchConfiguration('min_waypoint_distance')},
            {'max_route_length': LaunchConfiguration('max_route_length')},
            {'safe_navigation': LaunchConfiguration('safe_navigation')}
        ],
        remappings=[
            ('/llm_navigation_route', '/vla/llm_navigation_route'),
            ('/emergency_stop', '/emergency_stop'),
            ('/navigation_status', '/nav2_integration/status'),
            ('/navigation_feedback', '/nav2_integration/feedback'),
        ],
        output='screen'
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_min_waypoint_distance_cmd)
    ld.add_action(declare_max_route_length_cmd)
    ld.add_action(declare_safe_navigation_cmd)

    # Add nodes
    ld.add_action(nav2_integration_node)

    return ld