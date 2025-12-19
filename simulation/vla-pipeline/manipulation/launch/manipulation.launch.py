from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Manipulation parameters
    declare_pick_approach_distance_cmd = DeclareLaunchArgument(
        'pick_approach_distance',
        default_value='0.1',
        description='Distance to approach before picking (meters)'
    )

    declare_place_approach_distance_cmd = DeclareLaunchArgument(
        'place_approach_distance',
        default_value='0.1',
        description='Distance to approach before placing (meters)'
    )

    declare_alignment_tolerance_cmd = DeclareLaunchArgument(
        'alignment_tolerance',
        default_value='0.01',
        description='Tolerance for alignment operations (meters)'
    )

    # Manipulation node
    manipulation_node = Node(
        package='physical_ai_robotics',
        executable='manipulation_node',
        name='manipulation_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'pick_approach_distance': LaunchConfiguration('pick_approach_distance')},
            {'place_approach_distance': LaunchConfiguration('place_approach_distance')},
            {'alignment_tolerance': LaunchConfiguration('alignment_tolerance')}
        ],
        remappings=[
            ('/vla/llm_manipulation_command', '/vla/llm_manipulation_command'),
            ('/manipulation/status', '/manipulation/status'),
            ('/manipulation/feedback', '/manipulation/feedback'),
            ('/joint_states', '/joint_states'),
        ],
        output='screen'
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_pick_approach_distance_cmd)
    ld.add_action(declare_place_approach_distance_cmd)
    ld.add_action(declare_alignment_tolerance_cmd)

    # Add nodes
    ld.add_action(manipulation_node)

    return ld