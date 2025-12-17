---
title: Launch Files for Humanoid Robots
sidebar_position: 7
description: Understanding ROS 2 launch files for managing complex robotic systems and coordinated node startup
---

# Launch Files: Managing Complex Robotic Systems

## Introduction

In the previous chapter, we explored parameters for configuring individual nodes at runtime. Now we'll dive deeper into **launch files**, which are essential for managing complex robotic systems like humanoid robots. Launch files allow you to start multiple nodes with specific configurations in a coordinated manner, making it possible to launch an entire robot system with a single command.

For humanoid robots with dozens of nodes for control, perception, planning, and other functions, launch files are crucial for:
- Coordinated system startup
- Managing complex parameter configurations
- Organizing nodes into functional groups
- Enabling different operational modes

## Understanding Launch Files

Launch files are Python scripts that define how to start a collection of nodes and other processes. They provide a declarative way to specify:
- Which nodes to run
- What parameters to set
- What launch arguments to accept
- How to organize nodes into namespaces
- What conditions apply to node execution

### Basic Launch File Structure

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    declare_example_arg = DeclareLaunchArgument(
        'example_arg',
        default_value='default_value',
        description='Example launch argument'
    )

    # Define nodes
    example_node = Node(
        package='example_package',
        executable='example_node',
        name='example_node',
        parameters=[
            {'param1': LaunchConfiguration('example_arg')}
        ]
    )

    # Return launch description
    return LaunchDescription([
        declare_example_arg,
        example_node
    ])
```

## Advanced Launch File Concepts

### 1. Launch Arguments and Substitutions

Launch arguments allow you to customize launch behavior without modifying the launch file:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare multiple launch arguments
    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid_robot',
        description='Name of the robot'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        choices=['true', 'false'],
        description='Use simulation time'
    )

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_config'),
            'config',
            'default.yaml'
        ]),
        description='Path to configuration file'
    )

    # Use launch configurations in node definitions
    robot_node = Node(
        package='humanoid_control',
        executable='robot_controller',
        name=[LaunchConfiguration('robot_name'), '_controller'],
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'robot_name': LaunchConfiguration('robot_name')}
        ],
        output='screen'
    )

    return LaunchDescription([
        declare_robot_name,
        declare_use_sim_time,
        declare_config_file,
        robot_node
    ])
```

### 2. Namespacing and Remappings

Namespacing helps organize nodes and topics in complex systems:

```python
from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction

def generate_launch_description():
    # Group nodes with a namespace
    left_arm_group = GroupAction(
        actions=[
            PushRosNamespace('left_arm'),
            Node(
                package='humanoid_arm',
                executable='arm_controller',
                name='arm_controller',
                parameters=[{'arm_type': 'left'}]
            ),
            Node(
                package='humanoid_arm',
                executable='gripper_controller',
                name='gripper_controller'
            )
        ]
    )

    # Individual nodes with remappings
    right_arm_node = Node(
        package='humanoid_arm',
        executable='arm_controller',
        name='right_arm_controller',
        namespace='right_arm',
        parameters=[{'arm_type': 'right'}],
        remappings=[
            ('joint_states', 'right_arm/joint_states'),
            ('joint_commands', 'right_arm/joint_commands'),
            ('gripper_command', 'right_arm/gripper_command')
        ]
    )

    return LaunchDescription([
        left_arm_group,
        right_arm_node
    ])
```

## Conditional Launch Files

Launch files can include conditional logic for different configurations:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    declare_enable_perception = DeclareLaunchArgument(
        'enable_perception',
        default_value='true',
        description='Enable perception nodes'
    )

    declare_enable_simulation = DeclareLaunchArgument(
        'enable_simulation',
        default_value='false',
        description='Enable simulation mode'
    )

    # Conditional nodes
    perception_node = Node(
        condition=IfCondition(LaunchConfiguration('enable_perception')),
        package='humanoid_perception',
        executable='perception_pipeline',
        name='perception_pipeline'
    )

    simulation_node = Node(
        condition=IfCondition(LaunchConfiguration('enable_simulation')),
        package='gazebo_ros',
        executable='gazebo',
        name='simulator'
    )

    # Conditional launch file inclusion
    hardware_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_hardware'),
                'launch',
                'hardware.launch.py'
            ])
        ]),
        condition=UnlessCondition(LaunchConfiguration('enable_simulation'))
    )

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_gazebo'),
                'launch',
                'simulation.launch.py'
            ])
        ]),
        condition=IfCondition(LaunchConfiguration('enable_simulation'))
    )

    return LaunchDescription([
        declare_enable_perception,
        declare_enable_simulation,
        perception_node,
        simulation_node,
        hardware_launch,
        sim_launch
    ])
```

## Complex Launch File Example: Humanoid Robot System

Here's a comprehensive example showing how to structure launch files for a humanoid robot system:

```python
# launch/humanoid_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
    TimerAction
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import PushRosNamespace

def generate_launch_description():
    # Declare launch arguments
    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid_robot',
        description='Name of the robot'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        choices=['true', 'false'],
        description='Use simulation time'
    )

    declare_enable_perception = DeclareLaunchArgument(
        'enable_perception',
        default_value='true',
        description='Enable perception nodes'
    )

    declare_enable_navigation = DeclareLaunchArgument(
        'enable_navigation',
        default_value='true',
        description='Enable navigation stack'
    )

    # Robot description (loads URDF, joint state publisher, etc.)
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_description'),
                'launch',
                'robot_description.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
    )

    # Joint controllers group
    joint_control_group = GroupAction(
        actions=[
            PushRosNamespace('joint_control'),
            Node(
                package='humanoid_control',
                executable='joint_manager',
                name='joint_manager',
                parameters=[
                    PathJoinSubstitution([
                        FindPackageShare('humanoid_config'),
                        'config',
                        'joint_manager.yaml'
                    ]),
                    {'use_sim_time': LaunchConfiguration('use_sim_time')}
                ],
                output='screen'
            ),
            Node(
                package='humanoid_control',
                executable='torque_limiter',
                name='torque_limiter',
                parameters=[
                    {'use_sim_time': LaunchConfiguration('use_sim_time')}
                ]
            )
        ]
    )

    # Perception pipeline (conditional)
    perception_group = GroupAction(
        condition=IfCondition(LaunchConfiguration('enable_perception')),
        actions=[
            PushRosNamespace('perception'),
            Node(
                package='humanoid_perception',
                executable='camera_pipeline',
                name='camera_pipeline',
                parameters=[
                    PathJoinSubstitution([
                        FindPackageShare('humanoid_config'),
                        'config',
                        'camera.yaml'
                    ]),
                    {'use_sim_time': LaunchConfiguration('use_sim_time')}
                ]
            ),
            Node(
                package='humanoid_perception',
                executable='lidar_processing',
                name='lidar_processing',
                parameters=[
                    {'use_sim_time': LaunchConfiguration('use_sim_time')}
                ]
            )
        ]
    )

    # Navigation stack (conditional)
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_navigation'),
                'launch',
                'navigation.launch.py'
            ])
        ]),
        condition=IfCondition(LaunchConfiguration('enable_navigation')),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_base_frame': 'base_link'
        }.items()
    )

    # Robot manager (core system management)
    robot_manager = Node(
        package='humanoid_system',
        executable='robot_manager',
        name='robot_manager',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_config'),
                'config',
                'robot_manager.yaml'
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'robot_name': LaunchConfiguration('robot_name')}
        ],
        output='screen'
    )

    # Diagnostic aggregator
    diagnostics = Node(
        package='diagnostic_aggregator',
        executable='aggregator_node',
        name='diagnostic_aggregator',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_config'),
                'config',
                'diagnostics.yaml'
            ])
        ]
    )

    # Return the complete launch description
    return LaunchDescription([
        declare_robot_name,
        declare_use_sim_time,
        declare_enable_perception,
        declare_enable_navigation,

        robot_description_launch,
        joint_control_group,
        perception_group,
        navigation_launch,
        robot_manager,
        diagnostics,
    ])
```

## Parameter Loading in Launch Files

Launch files can load parameters from multiple sources:

```python
# launch/parameter_loading.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Declare arguments
    declare_config_dir = DeclareLaunchArgument(
        'config_dir',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_config'),
            'config'
        ]),
        description='Directory containing configuration files'
    )

    # Node with multiple parameter sources
    controller_node = Node(
        package='humanoid_control',
        executable='advanced_controller',
        name='advanced_controller',
        parameters=[
            # 1. Load from YAML file
            PathJoinSubstitution([
                LaunchConfiguration('config_dir'),
                'controller.yaml'
            ]),

            # 2. Load from another YAML file
            PathJoinSubstitution([
                LaunchConfiguration('config_dir'),
                'safety.yaml'
            ]),

            # 3. Inline parameters
            {
                'use_sim_time': False,
                'robot_name': 'humanoid_v2',
                'control_loop_frequency': 200
            },

            # 4. Launch configuration parameters
            {
                'robot_type': LaunchConfiguration('robot_type'),
                'debug_mode': LaunchConfiguration('debug_mode')
            }
        ],
        output='screen'
    )

    return LaunchDescription([
        declare_config_dir,
        controller_node
    ])
```

## Launch File Best Practices for Humanoid Robots

### 1. Hierarchical Organization

Organize launch files in a hierarchy that matches your system architecture:

```
launch/
├── humanoid_system.launch.py          # Main system launch
├── robot_description.launch.py        # URDF and robot state
├── joint_control.launch.py            # Joint controllers
├── perception.launch.py               # Perception pipeline
├── navigation.launch.py               # Navigation stack
├── simulation/
│   ├── gazebo_simulation.launch.py
│   └── rviz.launch.py
└── hardware/
    ├── real_hardware.launch.py
    └── mock_hardware.launch.py
```

### 2. Configuration Separation

Separate configuration into multiple files:

```python
# launch/modular_launch.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Include different subsystems
    return LaunchDescription([
        # Robot description
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('humanoid_description'),
                    'launch',
                    'robot_description.launch.py'
                ])
            ])
        ),

        # Control system
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('humanoid_control'),
                    'launch',
                    'control_system.launch.py'
                ])
            ])
        ),

        # Perception system
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('humanoid_perception'),
                    'launch',
                    'perception_system.launch.py'
                ])
            ])
        )
    ])
```

### 3. Error Handling and Validation

Add validation to your launch files:

```python
# launch/validated_launch.launch.py
from launch import LaunchDescription, LaunchContext
from launch.actions import OpaqueFunction
from launch_ros.actions import Node
import os

def validate_launch(context: LaunchContext):
    """Validate launch conditions before starting nodes"""
    # Check if required files exist
    config_file = context.launch_configurations.get('config_file', '')
    if config_file and not os.path.exists(config_file):
        raise RuntimeError(f'Config file does not exist: {config_file}')

    # Validate parameter values
    robot_type = context.launch_configurations.get('robot_type', 'unknown')
    if robot_type not in ['humanoid_v1', 'humanoid_v2', 'humanoid_v3']:
        raise RuntimeError(f'Invalid robot type: {robot_type}')

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=validate_launch),
        # Add nodes here after validation
    ])
```

## Launch File Tools and Commands

ROS 2 provides several tools for working with launch files:

```bash
# List all launch files in a package
ros2 launch --list-package humanoid_control

# Show launch file arguments
ros2 launch --show-args humanoid_control robot.launch.py

# Dry run to check launch file syntax
ros2 launch --dry-run humanoid_control robot.launch.py

# Launch with specific arguments
ros2 launch humanoid_control robot.launch.py robot_name:=my_robot use_sim_time:=true

# Launch with multiple arguments
ros2 launch humanoid_control robot.launch.py \
  robot_name:=my_robot \
  use_sim_time:=true \
  enable_perception:=false
```

## Advanced Launch Features

### 1. Event Handling

Launch files can respond to events:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.actions import LogInfo

def generate_launch_description():
    # Node to monitor
    robot_controller = Node(
        package='humanoid_control',
        executable='robot_controller',
        name='robot_controller'
    )

    # Event handler
    event_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=robot_controller,
            on_start=[
                LogInfo(msg='Robot controller started successfully'),
                # Additional actions when node starts
            ]
        )
    )

    # Another event handler for process exit
    exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=robot_controller,
            on_exit=[
                LogInfo(msg='Robot controller exited'),
                # Cleanup actions
            ]
        )
    )

    return LaunchDescription([
        robot_controller,
        event_handler,
        exit_handler
    ])
```

### 2. Timed Launch Actions

Start nodes with delays:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    # Start robot description first
    robot_description = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher'
    )

    # Start controller after a delay
    controller = TimerAction(
        period=2.0,  # Wait 2 seconds
        actions=[
            Node(
                package='humanoid_control',
                executable='joint_controller',
                name='joint_controller'
            )
        ]
    )

    return LaunchDescription([
        robot_description,
        controller
    ])
```

## Launch File Performance Considerations

### 1. Startup Sequence
- Start critical nodes first (robot state publisher, etc.)
- Use timers to stagger non-critical nodes
- Consider lazy loading for optional components

### 2. Resource Management
- Monitor memory usage of launch files
- Use appropriate process groups for related nodes
- Implement proper cleanup on shutdown

### 3. Scalability
- Use modular launch file design
- Consider separate launch files for different robot parts
- Implement conditional inclusion for optional components

## Troubleshooting Launch Files

### Common Issues and Solutions

1. **Parameter Loading Issues**
```bash
# Check if parameter file exists and is valid
ros2 param dump /node_name

# Verify parameter file path in launch file
ls -la /path/to/parameter/file.yaml
```

2. **Node Startup Failures**
```bash
# Check if node executable exists
ros2 run package_name executable_name --ros-args --help

# Verify node name and package in launch file
ros2 node list
```

3. **Namespace Issues**
```bash
# List all topics to check for namespace problems
ros2 topic list

# Use ros2 run to test node independently
ros2 run package_name executable_name --ros-args -r __ns:=/namespace
```

## Summary

Launch files are essential for managing complex robotic systems like humanoid robots. They provide a declarative way to:
- Start multiple nodes with coordinated configurations
- Handle different operational modes (simulation vs. real hardware)
- Organize nodes into functional groups with appropriate namespacing
- Load parameters from multiple sources
- Implement conditional logic for optional components

Proper use of launch files enables reliable system startup and management, which is crucial for humanoid robots with many interconnected components.

## Exercises

1. Create a launch file hierarchy for a humanoid robot that includes:
   - Robot description loading
   - Joint controllers for different body parts
   - Perception nodes
   - Navigation stack
   - Diagnostic nodes

2. Implement conditional launch logic for different robot configurations (e.g., different arm lengths, sensor configurations).

3. Design a launch file that starts nodes in the correct order with appropriate delays for dependencies.

## Next Steps

In the next chapter, we'll explore the Python client library (rclpy) in detail and learn how to integrate AI agents with ROS 2 controllers. This is crucial for humanoid robots that need to connect high-level AI reasoning with low-level control systems.

Continue to Chapter 8: [rclpy: Bridging Python Agents to ROS Controllers](./rclpy-integration.md) to learn about connecting AI algorithms with ROS 2.