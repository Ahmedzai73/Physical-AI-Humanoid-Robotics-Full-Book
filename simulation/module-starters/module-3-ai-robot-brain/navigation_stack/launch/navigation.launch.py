from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    use_composition = LaunchConfiguration('use_composition')
    container_name = LaunchConfiguration('container_name')
    container_name_full = (container_name, '_container')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value='nav2_config.yaml',
        description='Full path to the ROS2 parameters file to use for all launched nodes'
    )

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack'
    )

    declare_use_composition_cmd = DeclareLaunchArgument(
        'use_composition',
        default_value='False',
        description='Whether to use composed bringup'
    )

    declare_container_name_cmd = DeclareLaunchArgument(
        'container_name',
        default_value='nav2_container',
        description='the name of conatiner that nodes will load in if use composition'
    )

    # Make re-written yaml
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'autostart': autostart
    }

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites=param_substitutions,
        convert_types=True
    )

    # Nodes
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[configured_params],
    )

    start_localization_cmd = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[configured_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    start_world_model_cmd = Node(
        package='nav2_world_model',
        executable='world_model',
        output='screen',
        parameters=[configured_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    start_global_costmap_cmd = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='global_costmap',
        output='screen',
        parameters=[configured_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    start_local_costmap_cmd = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='local_costmap',
        output='screen',
        parameters=[configured_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    start_planner_cmd = Node(
        package='nav2_navfn_planner',
        executable='navfn_planner',
        name='navfn_planner',
        output='screen',
        parameters=[configured_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    start_controller_cmd = Node(
        package='nav2_regulated_pure_pursuit_controller',
        executable='regulated_pure_pursuit_controller',
        name='regulated_pure_pursuit_controller',
        output='screen',
        parameters=[configured_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    start_recovery_cmd = Node(
        package='nav2_recoveries',
        executable='recoveries_server',
        name='recoveries_server',
        output='screen',
        parameters=[configured_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    start_bt_navigator_cmd = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[configured_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    start_waypoint_follower_cmd = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[configured_params],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add the launch arguments and nodes to the launch description
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_params_file_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_composition_cmd)
    ld.add_action(declare_container_name_cmd)

    # Add nodes
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_localization_cmd)
    ld.add_action(start_world_model_cmd)
    ld.add_action(start_global_costmap_cmd)
    ld.add_action(start_local_costmap_cmd)
    ld.add_action(start_planner_cmd)
    ld.add_action(start_controller_cmd)
    ld.add_action(start_recovery_cmd)
    ld.add_action(start_bt_navigator_cmd)
    ld.add_action(start_waypoint_follower_cmd)

    return ld