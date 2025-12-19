from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Perception parameters
    declare_object_detection_threshold_cmd = DeclareLaunchArgument(
        'object_detection_threshold',
        default_value='0.5',
        description='Minimum confidence threshold for object detection'
    )

    declare_distance_threshold_cmd = DeclareLaunchArgument(
        'distance_threshold',
        default_value='2.0',
        description='Maximum distance for obstacle detection (meters)'
    )

    declare_tracking_enabled_cmd = DeclareLaunchArgument(
        'tracking_enabled',
        default_value='true',
        description='Enable object tracking'
    )

    # Perception-Action Integration node
    perception_action_node = Node(
        package='physical_ai_robotics',
        executable='perception_action_integration',
        name='perception_action_integration_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'object_detection_threshold': LaunchConfiguration('object_detection_threshold')},
            {'distance_threshold': LaunchConfiguration('distance_threshold')},
            {'tracking_enabled': LaunchConfiguration('tracking_enabled')}
        ],
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
            ('/scan', '/scan'),
            ('/pointcloud', '/pointcloud'),
            ('/vla/llm_perception_request', '/vla/llm_perception_request'),
            ('/vla/action_command', '/vla/action_command'),
            ('/perception_action/status', '/perception_action/status'),
            ('/perception_action/feedback', '/perception_action/feedback'),
        ],
        output='screen'
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_object_detection_threshold_cmd)
    ld.add_action(declare_distance_threshold_cmd)
    ld.add_action(declare_tracking_enabled_cmd)

    # Add nodes
    ld.add_action(perception_action_node)

    return ld