---
title: Parameters & Configuration Management
sidebar_position: 6
description: Understanding ROS 2 parameters for runtime configuration and launch files for system management
---

# Parameters & Configuration Management

## Introduction

In the previous chapters, we explored nodes, topics, services, and actions - the primary communication mechanisms in ROS 2. Now we'll focus on **parameters** and **launch files**, which are essential for configuring and managing robotic systems. For humanoid robots with complex configurations and multiple subsystems, proper parameter and launch file management is crucial for reliable operation.

Parameters allow you to configure nodes at runtime without recompiling, while launch files enable you to start multiple nodes with specific configurations in a coordinated manner.

## Understanding Parameters

Parameters are named values that can be set at runtime to configure node behavior. They provide a way to customize node behavior without hardcoding values.

### Parameter Types

ROS 2 supports several parameter types:
- `bool`: True/False values
- `int`: Integer values
- `double`: Floating-point values
- `string`: Text values
- `list`: Arrays of any type (integer_array, double_array, string_array, bool_array)

## Working with Parameters in Nodes

### 1. Declaring and Using Parameters

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.srv import SetParameters, GetParameters, ListParameters

class ParameterExampleNode(Node):
    def __init__(self):
        super().__init__('parameter_example_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter(
            'robot_name',
            'humanoid_robot',
            ParameterDescriptor(
                description='Name of the robot',
                type=ParameterType.PARAMETER_STRING
            )
        )

        self.declare_parameter(
            'max_velocity',
            1.0,
            ParameterDescriptor(
                description='Maximum joint velocity in rad/s',
                type=ParameterType.PARAMETER_DOUBLE,
                additional_constraints='Must be positive'
            )
        )

        self.declare_parameter(
            'joints_to_control',
            ['left_hip', 'left_knee', 'right_hip', 'right_knee'],
            ParameterDescriptor(
                description='List of joints to control',
                type=ParameterType.PARAMETER_STRING_ARRAY
            )
        )

        self.declare_parameter(
            'enable_safety_limits',
            True,
            ParameterDescriptor(
                description='Enable safety limit checking',
                type=ParameterType.PARAMETER_BOOL
            )
        )

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.joints_to_control = self.get_parameter('joints_to_control').value
        self.enable_safety_limits = self.get_parameter('enable_safety_limits').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Joints to control: {self.joints_to_control}')
        self.get_logger().info(f'Safety limits enabled: {self.enable_safety_limits}')

        # Set up parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """Handle parameter changes"""
        for param in params:
            if param.name == 'max_velocity':
                if param.value <= 0:
                    return SetParameters.Result(
                        successful=False,
                        reason='Max velocity must be positive'
                    )
                self.max_velocity = param.value
                self.get_logger().info(f'Max velocity updated to: {self.max_velocity}')
            elif param.name == 'robot_name':
                self.robot_name = param.value
                self.get_logger().info(f'Robot name updated to: {self.robot_name}')

        return SetParameters.Result(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterExampleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

### 2. Parameter Validation and Constraints

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange, FloatingPointRange
from rcl_interfaces.msg import ParameterType

class ParameterValidationNode(Node):
    def __init__(self):
        super().__init__('parameter_validation_node')

        # Declare parameter with integer range
        self.declare_parameter(
            'control_frequency',
            100,
            ParameterDescriptor(
                description='Control loop frequency in Hz',
                type=ParameterType.PARAMETER_INTEGER,
                integer_range=[IntegerRange(from_value=10, to_value=1000, step=1)]
            )
        )

        # Declare parameter with floating point range
        self.declare_parameter(
            'safety_threshold',
            0.5,
            ParameterDescriptor(
                description='Safety threshold value',
                type=ParameterType.PARAMETER_DOUBLE,
                floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)]
            )
        )

        # Get validated parameters
        self.control_frequency = self.get_parameter('control_frequency').value
        self.safety_threshold = self.get_parameter('safety_threshold').value

        self.get_logger().info(f'Control frequency: {self.control_frequency} Hz')
        self.get_logger().info(f'Safety threshold: {self.safety_threshold}')

    def parameter_callback(self, params):
        """Validate parameter changes"""
        for param in params:
            if param.name == 'control_frequency':
                if param.value < 10 or param.value > 1000:
                    return SetParameters.Result(
                        successful=False,
                        reason='Control frequency must be between 10 and 1000 Hz'
                    )
            elif param.name == 'safety_threshold':
                if param.value < 0.0 or param.value > 1.0:
                    return SetParameters.Result(
                        successful=False,
                        reason='Safety threshold must be between 0.0 and 1.0'
                    )

        return SetParameters.Result(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterValidationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

## Parameter Management

### 1. Parameter Files

Parameters can be loaded from YAML files:

```yaml
# robot_config.yaml
parameter_example_node:
  ros__parameters:
    robot_name: "advanced_humanoid"
    max_velocity: 2.0
    joints_to_control: ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]
    enable_safety_limits: true

parameter_validation_node:
  ros__parameters:
    control_frequency: 200
    safety_threshold: 0.7
```

### 2. Loading Parameters in Code

```python
import rclpy
from rclpy.node import Node
import yaml

class ParameterLoaderNode(Node):
    def __init__(self):
        super().__init__('parameter_loader_node')

        # Load parameters from YAML file
        self.load_parameters_from_file('config/robot_config.yaml')

        # Use loaded parameters
        self.robot_name = self.get_parameter('robot_name').value
        self.get_logger().info(f'Loaded robot name: {self.robot_name}')

    def load_parameters_from_file(self, file_path):
        """Load parameters from a YAML file"""
        try:
            with open(file_path, 'r') as file:
                param_data = yaml.safe_load(file)

            # Set parameters from file
            for node_name, node_params in param_data.items():
                if node_name == self.get_name():
                    for param_name, param_value in node_params['ros__parameters'].items():
                        self.declare_parameter(param_name, param_value)
        except FileNotFoundError:
            self.get_logger().warn(f'Parameter file {file_path} not found')
        except yaml.YAMLError as e:
            self.get_logger().error(f'Error parsing parameter file: {e}')
```

## Launch Files

Launch files provide a way to start multiple nodes with specific configurations in a coordinated manner. This is especially important for humanoid robots that have many interconnected nodes.

### 1. Basic Launch File

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'robot_name',
            default_value='humanoid_robot',
            description='Name of the robot'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),

        # Joint controller node
        Node(
            package='humanoid_control',
            executable='joint_controller',
            name='joint_controller',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('humanoid_config'),
                    'config',
                    'joint_controller.yaml'
                ]),
                {'robot_name': LaunchConfiguration('robot_name')},
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            remappings=[
                ('joint_states', 'filtered_joint_states'),
                ('joint_commands', 'smooth_joint_commands')
            ],
            output='screen'
        ),

        # Perception node
        Node(
            package='humanoid_perception',
            executable='camera_processor',
            name='camera_processor',
            parameters=[
                {'camera_topic': '/camera/rgb/image_raw'},
                {'detection_model': 'yolov5'}
            ],
            output='screen'
        )
    ])
```

### 2. Advanced Launch File with Conditions

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    declare_robot_type = DeclareLaunchArgument(
        'robot_type',
        default_value='humanoid_v1',
        choices=['humanoid_v1', 'humanoid_v2', 'humanoid_v3'],
        description='Type of humanoid robot'
    )

    declare_enable_perception = DeclareLaunchArgument(
        'enable_perception',
        default_value='true',
        description='Enable perception nodes'
    )

    # Conditional nodes based on arguments
    perception_nodes = Node(
        condition=IfCondition(LaunchConfiguration('enable_perception')),
        package='humanoid_perception',
        executable='perception_pipeline',
        name='perception_pipeline',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Robot-specific launch files
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_description'),
                'launch',
                LaunchConfiguration('robot_type'),
                'robot.launch.py'
            ])
        ])
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_robot_type,
        declare_enable_perception,
        robot_launch,
        perception_nodes,
    ])
```

### 3. Parameter Files with Launch Files

Create a parameter file for the joint controller:

```yaml
# config/joint_controller.yaml
joint_controller:
  ros__parameters:
    # Control parameters
    control_frequency: 200
    max_joint_velocity: 3.0
    position_tolerance: 0.01
    velocity_tolerance: 0.05

    # Joint configuration
    joint_names: ["left_hip", "left_knee", "left_ankle",
                  "right_hip", "right_knee", "right_ankle",
                  "left_shoulder", "left_elbow", "left_wrist",
                  "right_shoulder", "right_elbow", "right_wrist"]

    # PID gains for each joint type
    hip_pid:
      p: 10.0
      i: 0.1
      d: 0.01
    knee_pid:
      p: 8.0
      i: 0.05
      d: 0.005
    ankle_pid:
      p: 5.0
      i: 0.02
      d: 0.002

    # Safety limits
    enable_safety_limits: true
    max_effort: 100.0
    soft_limits:
      enabled: true
      position_margin: 0.1
      velocity_scale: 0.8
```

## Practical Example: Humanoid Robot Configuration System

Let's create a comprehensive example showing how parameters and launch files work together in a humanoid robot system:

```python
# humanoid_robot_manager.py
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter
from rcl_interfaces.srv import SetParameters, GetParameters
import json
import os

class HumanoidRobotManager(Node):
    def __init__(self):
        super().__init__('humanoid_robot_manager')

        # Declare configuration parameters
        self.declare_parameter('robot_type', 'humanoid_v1')
        self.declare_parameter('control_mode', 'position')  # position, velocity, effort
        self.declare_parameter('enable_logging', True)
        self.declare_parameter('log_level', 'INFO')
        self.declare_parameter('max_operational_time', 3600)  # seconds
        self.declare_parameter('safety_zones', [1.0, 2.0, 3.0])  # distances in meters

        # Initialize robot state
        self.robot_state = {
            'initialized': False,
            'calibrated': False,
            'motors_enabled': False,
            'operational_time': 0.0
        }

        # Set up parameter callback
        self.add_on_set_parameters_callback(self.parameter_change_callback)

        self.get_logger().info('Humanoid Robot Manager initialized')

    def parameter_change_callback(self, params):
        """Handle parameter changes with validation"""
        for param in params:
            if param.name == 'control_mode':
                if param.value not in ['position', 'velocity', 'effort']:
                    return SetParameters.Result(
                        successful=False,
                        reason='Control mode must be position, velocity, or effort'
                    )
            elif param.name == 'log_level':
                if param.value not in ['DEBUG', 'INFO', 'WARN', 'ERROR']:
                    return SetParameters.Result(
                        successful=False,
                        reason='Invalid log level'
                    )

        return SetParameters.Result(successful=True)

    def save_configuration(self, config_path):
        """Save current configuration to file"""
        config = {}
        for param_name in ['robot_type', 'control_mode', 'enable_logging',
                          'log_level', 'max_operational_time', 'safety_zones']:
            param_value = self.get_parameter(param_name).value
            config[param_name] = param_value

        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.get_logger().info(f'Configuration saved to {config_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save configuration: {e}')

    def load_configuration(self, config_path):
        """Load configuration from file"""
        if not os.path.exists(config_path):
            self.get_logger().warn(f'Configuration file {config_path} does not exist')
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Set parameters from file
            for param_name, param_value in config.items():
                if self.has_parameter(param_name):
                    self.set_parameters([Parameter(name=param_name, value=param_value)])
                else:
                    # Declare new parameter if it doesn't exist
                    self.declare_parameter(param_name, param_value)

            self.get_logger().info(f'Configuration loaded from {config_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load configuration: {e}')

def main(args=None):
    rclpy.init(args=args)
    manager = HumanoidRobotManager()

    # Example: Load configuration from file
    config_path = 'config/current_robot_config.json'
    manager.load_configuration(config_path)

    try:
        rclpy.spin(manager)
    except KeyboardInterrupt:
        # Save configuration before shutdown
        manager.save_configuration(config_path)
    finally:
        manager.destroy_node()
        rclpy.shutdown()
```

## Launch File for Humanoid Robot System

```python
# launch/humanoid_robot_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

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
        description='Use simulation time'
    )

    declare_enable_perception = DeclareLaunchArgument(
        'enable_perception',
        default_value='true',
        description='Enable perception nodes'
    )

    declare_control_mode = DeclareLaunchArgument(
        'control_mode',
        default_value='position',
        choices=['position', 'velocity', 'effort'],
        description='Control mode for joints'
    )

    # Robot manager node
    robot_manager = Node(
        package='humanoid_system',
        executable='humanoid_robot_manager',
        name='robot_manager',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_config'),
                'config',
                'robot_manager.yaml'
            ]),
            {'robot_type': LaunchConfiguration('robot_name')},
            {'control_mode': LaunchConfiguration('control_mode')},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Joint controller
    joint_controller = Node(
        package='humanoid_control',
        executable='joint_controller',
        name='joint_controller',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_config'),
                'config',
                'joint_controller.yaml'
            ]),
            {'robot_name': LaunchConfiguration('robot_name')},
            {'control_mode': LaunchConfiguration('control_mode')},
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Perception pipeline (conditional)
    perception_pipeline = Node(
        condition=IfCondition(LaunchConfiguration('enable_perception')),
        package='humanoid_perception',
        executable='perception_pipeline',
        name='perception_pipeline',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_config'),
                'config',
                'perception.yaml'
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Navigation stack
    navigation = Node(
        package='humanoid_navigation',
        executable='navigation_stack',
        name='navigation_stack',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_config'),
                'config',
                'navigation.yaml'
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Launch description
    ld = LaunchDescription()

    # Add arguments
    ld.add_action(declare_robot_name)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_enable_perception)
    ld.add_action(declare_control_mode)

    # Add nodes
    ld.add_action(robot_manager)
    ld.add_action(joint_controller)
    ld.add_action(perception_pipeline)
    ld.add_action(navigation)

    return ld
```

## Parameter Tools and Commands

ROS 2 provides several command-line tools for working with parameters:

```bash
# List parameters of a node
ros2 param list

# List parameters of a specific node
ros2 param list /joint_controller

# Get parameter value
ros2 param get /joint_controller max_velocity

# Set parameter value
ros2 param set /joint_controller max_velocity 2.5

# Dump all parameters to a file
ros2 param dump /joint_controller

# Load parameters from a file
ros2 param load /joint_controller config/joint_controller.yaml

# Describe parameters of a node
ros2 param describe /joint_controller
```

## Best Practices for Humanoid Robots

### 1. Parameter Organization
```python
# Good: Organized parameter structure
'control.gains.position.p': 10.0
'control.gains.position.i': 0.1
'control.gains.position.d': 0.01
'safety.limits.max_velocity': 3.0
'safety.limits.max_effort': 100.0
'joints.left_leg.hip.pid.p': 8.0
'joints.left_leg.knee.pid.p': 6.0
```

### 2. Parameter Validation
```python
def validate_control_parameters(self):
    """Validate that control parameters are within safe ranges"""
    max_vel = self.get_parameter('max_velocity').value
    if max_vel <= 0 or max_vel > 10.0:  # Check against physical limits
        self.get_logger().error(f'Invalid max_velocity: {max_vel}')
        return False
    return True
```

### 3. Configuration Management
```python
class ConfigurationManager:
    """Centralized configuration management"""
    def __init__(self, node):
        self.node = node
        self.config_history = []

    def save_current_config(self, name):
        """Save current parameter configuration"""
        config = {}
        for param_name in self.node.get_parameter_names():
            config[param_name] = self.node.get_parameter(param_name).value

        self.config_history.append({
            'name': name,
            'timestamp': self.node.get_clock().now().nanoseconds,
            'config': config
        })

    def restore_config(self, name):
        """Restore a saved configuration"""
        for saved_config in self.config_history:
            if saved_config['name'] == name:
                params = []
                for param_name, param_value in saved_config['config'].items():
                    params.append(Parameter(name=param_name, value=param_value))

                self.node.set_parameters(params)
                return True
        return False
```

### 4. Launch File Best Practices
- Use descriptive names for launch arguments
- Group related nodes in the same launch file
- Use conditional inclusion for optional components
- Provide default values for all arguments
- Document the purpose of each launch file

## Performance Considerations

### 1. Parameter Updates
- Avoid frequent parameter updates in time-critical loops
- Batch parameter updates when possible
- Use appropriate callback groups for parameter handling

### 2. Launch File Loading
- Organize launch files hierarchically for complex systems
- Use relative paths in launch files for portability
- Consider lazy loading for non-critical nodes

### 3. Memory Usage
- Be mindful of array parameters that can grow large
- Use parameter files instead of hardcoding large configurations
- Implement parameter validation to prevent invalid configurations

## Troubleshooting Common Issues

### 1. Parameter Not Found
```bash
# Check if parameter exists
ros2 param list /your_node_name

# Check parameter description
ros2 param describe /your_node_name parameter_name
```

### 2. Parameter Not Updating
```python
# Make sure parameter is declared before use
if not self.has_parameter('param_name'):
    self.declare_parameter('param_name', default_value)

# Use get_parameter() to retrieve value
param_value = self.get_parameter('param_name').value
```

### 3. Launch File Issues
```bash
# Check launch file syntax
python3 -m launch.config_check path/to/launch/file.py

# Debug launch file
ros2 launch --dry-run package_name launch_file.py
```

## Summary

Parameters and launch files are essential tools for configuring and managing ROS 2 systems, especially complex humanoid robots with many interconnected components. Parameters provide runtime configuration without recompilation, while launch files enable coordinated startup of multiple nodes with specific configurations.

For humanoid robots, proper use of parameters and launch files enables:
- Safe configuration of control parameters
- Coordinated startup of complex subsystems
- Runtime adjustment of robot behavior
- Consistent configuration across different robot instances

## Exercises

1. Create a parameter file for a humanoid robot that includes:
   - Joint limits for each joint group
   - PID controller gains
   - Safety thresholds
   - Operational parameters

2. Design a launch file structure for a humanoid robot system that includes:
   - Robot description loading
   - Joint controllers
   - Perception nodes
   - Navigation stack
   - Optional debugging tools

3. Implement a parameter validation system that checks for safe ranges when parameters are updated, particularly for control gains and safety limits.

## Next Steps

In the next chapter, we'll explore launch files in more detail and learn how to create complex launch file structures for managing entire robotic systems. We'll cover advanced launch file features, parameter loading, and system composition patterns.

Continue to Chapter 7: [Launch Files for Humanoid Robots](./launch-files.md) to master system startup and management in ROS 2.