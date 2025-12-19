from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Pipeline parameters
    declare_max_pipeline_retries_cmd = DeclareLaunchArgument(
        'max_pipeline_retries',
        default_value='3',
        description='Maximum number of retries for pipeline steps'
    )

    declare_pipeline_timeout_cmd = DeclareLaunchArgument(
        'pipeline_timeout',
        default_value='30.0',
        description='Timeout for pipeline steps in seconds'
    )

    declare_enable_feedback_loop_cmd = DeclareLaunchArgument(
        'enable_feedback_loop',
        default_value='true',
        description='Enable feedback loop for pipeline adjustments'
    )

    # Complete VLA Pipeline node
    complete_vla_pipeline_node = Node(
        package='physical_ai_robotics',
        executable='complete_vla_pipeline',
        name='complete_vla_pipeline_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'max_pipeline_retries': LaunchConfiguration('max_pipeline_retries')},
            {'pipeline_timeout': LaunchConfiguration('pipeline_timeout')},
            {'enable_feedback_loop': LaunchConfiguration('enable_feedback_loop')}
        ],
        remappings=[
            ('/vla/voice_input', '/vla/voice_input'),
            ('/vla/voice_command', '/vla/voice_command'),
            ('/vla/llm_navigation_route', '/vla/llm_navigation_route'),
            ('/vla/llm_perception_request', '/vla/llm_perception_request'),
            ('/vla/llm_manipulation_command', '/vla/llm_manipulation_command'),
            ('/nav2_integration/status', '/nav2_integration/status'),
            ('/perception_action/feedback', '/perception_action/feedback'),
            ('/manipulation/status', '/manipulation/status'),
            ('/vla/pipeline/status', '/vla/pipeline/status'),
            ('/vla/pipeline/feedback', '/vla/pipeline/feedback'),
        ],
        output='screen'
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_max_pipeline_retries_cmd)
    ld.add_action(declare_pipeline_timeout_cmd)
    ld.add_action(declare_enable_feedback_loop_cmd)

    # Add nodes
    ld.add_action(complete_vla_pipeline_node)

    return ld