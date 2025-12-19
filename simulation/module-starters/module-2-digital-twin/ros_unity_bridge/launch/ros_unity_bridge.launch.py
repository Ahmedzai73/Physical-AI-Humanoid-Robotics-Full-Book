from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    config_file = LaunchConfiguration('config_file', default='bridge_config.yaml')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file',
        default_value='bridge_config.yaml',
        description='Path to the bridge configuration file'
    )

    # ROS-Unity bridge node
    ros_unity_bridge_cmd = Node(
        package='rosbridge_suite',
        executable='rosbridge_websocket',
        name='ros_unity_bridge',
        parameters=[
            {'use_sim_time': use_sim_time},
            PathJoinSubstitution([
                FindPackageShare('module_starters'),
                'ros_unity_bridge',
                config_file
            ])
        ],
        output='screen'
    )

    # Unity service bridge node (if available)
    unity_service_bridge_cmd = Node(
        package='unity_ros_tcp_connector',  # This would be a custom package
        executable='unity_service_bridge',
        name='unity_service_bridge',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'unity_host': 'localhost'},
            {'unity_port': 5555}
        ],
        output='screen',
        condition=launch.conditions.IfCondition('unity_service_bridge_enabled')  # This is just an example
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add the launch arguments and nodes to the launch description
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(ros_unity_bridge_cmd)

    return ld