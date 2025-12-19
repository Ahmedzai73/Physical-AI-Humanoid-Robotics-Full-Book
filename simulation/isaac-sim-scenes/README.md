# Isaac Sim Sample Scene for Physical AI & Humanoid Robotics

This sample scene provides a foundation for Isaac Sim examples in the Physical AI & Humanoid Robotics textbook. It demonstrates key concepts from Module 3: The AI-Robot Brain (NVIDIA Isaac™).

## Features

- Basic environment with ground plane and lighting
- Differential drive robot with physics properties
- Sample obstacles for navigation examples
- Sensor configurations (camera, LIDAR, IMU)
- ROS bridge setup for textbook examples
- Visual SLAM and navigation ready environment

## Configuration Files

- `scene_config.toml`: Scene configuration with physics, environment, and robot settings
- `setup_scene.py`: Python script to programmatically create the scene in Isaac Sim

## Scene Components

### Environment
- Ground plane (10m x 10m)
- Dome lighting for realistic illumination
- Default camera positioned for visualization

### Robot
- Differential drive robot model
- Physical properties and collision geometry
- Wheels with proper positioning

### Obstacles
- 3 sample obstacles for navigation examples
- Positioned to create interesting navigation challenges

### Sensors
- RGB camera for visual perception
- LIDAR for environment mapping
- IMU for robot state estimation
- Depth sensor for 3D perception

## Usage

1. Open Isaac Sim
2. Use the `setup_scene.py` script to create the scene programmatically
3. Or load the scene configuration from `scene_config.toml`
4. Connect to the ROS bridge endpoints as configured in the setup

## ROS Bridge Endpoints

The scene is configured with the following ROS topics:
- `/cmd_vel` - Robot velocity commands
- `/odom` - Robot odometry
- `/scan` - LIDAR scan data
- `/camera/color/image_raw` - RGB camera images

## Integration with Textbook

This scene provides the foundation for examples in Module 3: The AI-Robot Brain (NVIDIA Isaac™) of the Physical AI & Humanoid Robotics textbook. It demonstrates:
- Isaac Sim environment creation
- Robot physics and control
- Sensor simulation
- ROS integration
- Visual SLAM and navigation

## Requirements

- NVIDIA Isaac Sim 2023.1 or later
- Compatible NVIDIA GPU
- Python 3.8+ for scripting