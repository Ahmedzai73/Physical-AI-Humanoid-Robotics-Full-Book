# Physical AI & Humanoid Robotics ROS 2 Workspace

This ROS 2 workspace contains example packages and nodes for the Physical AI & Humanoid Robotics textbook. It demonstrates key concepts from the textbook including ROS 2 fundamentals, robot control, sensor fusion, and navigation.

## Package Structure

- `physical_ai_robotics`: Main package containing example nodes and launch files

## Nodes

1. **minimal_publisher**: Basic publisher node demonstrating ROS 2 topics
2. **minimal_subscriber**: Basic subscriber node demonstrating ROS 2 topics
3. **robot_controller**: Advanced robot controller demonstrating joint control and velocity commands
4. **sensor_fusion_node**: Demonstrates sensor fusion concepts from the textbook
5. **navigation_node**: Implements navigation concepts using Nav2

## Launch Files

- `physical_ai_robotics.launch.py`: Launches all example nodes

## Getting Started

1. Source your ROS 2 installation:
   ```bash
   source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
   ```

2. Build the workspace:
   ```bash
   cd simulation/ros-workspace
   colcon build --packages-select physical_ai_robotics
   ```

3. Source the workspace:
   ```bash
   source install/setup.bash
   ```

4. Launch the example nodes:
   ```bash
   ros2 launch physical_ai_robotics physical_ai_robotics.launch.py
   ```

## Dependencies

This workspace requires:
- ROS 2 (Humble Hawksbill or later recommended)
- Standard ROS 2 packages: rclpy, std_msgs, sensor_msgs, geometry_msgs, nav_msgs, tf2_ros, tf2_geometry_msgs

## Usage in Textbook

This workspace provides practical examples for the concepts covered in Module 1: The Robotic Nervous System (ROS 2) of the Physical AI & Humanoid Robotics textbook.