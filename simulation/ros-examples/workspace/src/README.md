# ROS 2 Workspace Template for Physical AI & Humanoid Robotics

This workspace template provides a starting point for ROS 2 development in the Physical AI & Humanoid Robotics textbook.

## Workspace Structure

```
workspace/
├── src/                 # Source packages
│   ├── humanoid_robot/  # Example humanoid robot package
│   ├── perception/      # Perception package
│   └── navigation/      # Navigation package
```

## Getting Started

1. Source your ROS 2 installation:
   ```bash
   source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
   ```

2. Navigate to this workspace directory and build:
   ```bash
   colcon build
   ```

3. Source the workspace:
   ```bash
   source install/setup.bash
   ```

## Example Packages

This template includes example packages to demonstrate common ROS 2 patterns used in humanoid robotics:

- `humanoid_robot`: Basic robot interface and control
- `perception`: Perception nodes for sensor data processing
- `navigation`: Navigation stack integration