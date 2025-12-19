# ROS 2 Parameters and Launch Files Simulation Steps

This guide provides step-by-step instructions for simulating ROS 2 parameters and launch files concepts covered in Module 1 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to configure ROS 2 nodes using parameters and how to launch multiple nodes simultaneously using launch files. These are essential for managing complex robotics systems.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- Completed the ROS 2 services and actions simulation steps
- Understanding of node creation and management

## Simulation Environment Setup

1. Open a terminal and navigate to the ROS workspace:
   ```bash
   cd simulation/ros-workspace
   ```

2. Source the ROS 2 installation and workspace:
   ```bash
   source /opt/ros/humble/setup.bash  # Adjust for your ROS 2 distribution
   source install/setup.bash
   ```

## Exercise 1: Basic Parameter Usage

1. Launch a node with command-line parameters:
   ```bash
   ros2 run physical_ai_robotics minimal_publisher --ros-args -p param_name:=value
   ```

2. Check the parameters of a running node:
   ```bash
   ros2 param list
   ros2 param get /minimal_publisher param_name
   ```

3. Change a parameter value at runtime:
   ```bash
   ros2 param set /minimal_publisher new_param 42
   ```

## Exercise 2: Parameter Declaration and Validation

1. Create a node that declares parameters with types and constraints:
   ```python
   # Example node that declares parameters
   # Follow textbook examples for parameter declaration
   ```

2. Test parameter validation by setting invalid values:
   ```bash
   ros2 param set /your_node_name param_name invalid_value
   ```

## Exercise 3: YAML Parameter Files

1. Create a YAML parameter file (config/robot_params.yaml):
   ```yaml
   /**:
     ros__parameters:
       robot_name: "my_robot"
       max_velocity: 1.0
       sensor_range: 10.0
   ```

2. Launch a node with parameters from a YAML file:
   ```bash
   ros2 run physical_ai_robotics robot_controller --ros-args --params-file config/robot_params.yaml
   ```

## Exercise 4: Launch File Basics

1. Launch nodes using the main launch file:
   ```bash
   ros2 launch physical_ai_robotics physical_ai_robotics.launch.py
   ```

2. Observe how multiple nodes start simultaneously.

## Exercise 5: Launch File Configuration

1. Create custom launch files with different configurations:
   ```python
   # Create launch files with different parameter sets
   # Example: simulation.launch.py vs real_robot.launch.py
   ```

2. Launch with specific launch arguments:
   ```bash
   ros2 launch physical_ai_robotics physical_ai_robotics.launch.py use_sim_time:=true
   ```

## Exercise 6: Conditional Launch and Node Management

1. Test conditional node launching based on arguments:
   ```bash
   # Launch with different argument values to enable/disable nodes
   ros2 launch your_package your_launch_file.launch.py enable_debug:=true
   ```

2. Use launch conditions to control node startup:
   ```python
   # Example: only start visualization nodes when enable_viz is true
   ```

## Exercise 7: Launch File Composition

1. Create launch files that include other launch files:
   ```python
   # Create modular launch files that can be combined
   # Example: base_launch.py includes sensor_launch.py and controller_launch.py
   ```

2. Test launch file composition:
   ```bash
   ros2 launch your_package combined_launch.py
   ```

## Exercise 8: Parameter and Launch Integration

1. Combine parameters and launch files effectively:
   ```bash
   # Use launch files to set parameters for multiple nodes
   # Example: set the same robot_name parameter for all nodes in the launch
   ```

2. Test different parameter configurations with the same launch file:
   ```bash
   ros2 launch your_package robot_launch.py --params-file config/config1.yaml
   ros2 launch your_package robot_launch.py --params-file config/config2.yaml
   ```

## Exercise 9: Dynamic Parameter Reconfiguration

1. Use rqt_reconfigure to change parameters at runtime:
   ```bash
   rqt_reconfigure
   ```

2. Observe how parameter changes affect node behavior in real-time.

## Exercise 10: Launch File Best Practices

1. Implement launch file best practices:
   - Proper argument definitions
   - Clear documentation
   - Error handling
   - Modular design

2. Test launch file with different robot configurations.

## Verification Steps

1. Confirm that parameters are properly set and accessible
2. Verify that launch files start all intended nodes
3. Check that parameter files load correctly
4. Ensure that launch arguments work as expected

## Expected Outcomes

- Understanding of ROS 2 parameter system and configuration
- Knowledge of launch file creation and usage
- Experience with YAML parameter files
- Ability to create modular and reusable launch configurations

## Troubleshooting

- If parameters don't load, check YAML syntax and file paths
- If launch files fail, verify all required nodes and dependencies exist
- If nodes don't respond to parameter changes, ensure parameters are properly declared

## Next Steps

After completing these exercises, proceed to the launch files simulation exercises to understand advanced launch configurations for humanoid robots.