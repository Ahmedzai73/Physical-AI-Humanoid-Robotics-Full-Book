from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the package share directory
    pkg_share = FindPackageShare('physical_ai_robotics').find('physical_ai_robotics')

    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description':
                f'<?xml version="1.0" ?><robot name="simple_robot">{open(get_package_share_directory("physical_ai_robotics") + "/urdf/simple_robot.urdf").read()}</robot>'
            }
        ],
        output='screen'
    )

    # Joint state publisher node
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # RViz node
    rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'simple_robot.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Example publisher node
    publisher_node = Node(
        package='physical_ai_robotics',
        executable='minimal_publisher',
        name='publisher_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Example subscriber node
    subscriber_node = Node(
        package='physical_ai_robotics',
        executable='minimal_subscriber',
        name='subscriber_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Return the launch description
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        robot_state_publisher,
        joint_state_publisher,
        publisher_node,
        subscriber_node,
        # RViz is optional - uncomment to include
        # rviz_node,
    ])