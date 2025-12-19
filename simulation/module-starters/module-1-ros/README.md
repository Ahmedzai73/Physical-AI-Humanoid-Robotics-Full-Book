# Module 1: The Robotic Nervous System (ROS 2) - Simulation Starter

This directory contains starter files for ROS 2 simulations as covered in Module 1 of the Physical AI & Humanoid Robotics textbook.

## Directory Structure

```
module-1-ros/
├── basic_robot/
│   ├── urdf/
│   │   └── simple_robot.urdf
│   ├── launch/
│   │   └── basic_robot.launch.py
│   ├── config/
│   │   └── robot_params.yaml
│   └── nodes/
│       ├── publisher_node.py
│       └── subscriber_node.py
├── navigation_tutorial/
│   ├── maps/
│   │   └── simple_map.yaml
│   ├── launch/
│   │   └── nav_tutorial.launch.py
│   └── config/
│       └── nav2_params.yaml
└── README.md
```

## Getting Started

1. Navigate to the basic_robot directory
2. Launch the basic robot simulation:
   ```bash
   cd basic_robot
   ros2 launch launch/basic_robot.launch.py
   ```

## Key Concepts Demonstrated

- ROS 2 node creation and communication
- URDF robot modeling
- Parameter configuration
- Topic-based communication
- Basic navigation concepts

For detailed explanations, refer to Module 1: The Robotic Nervous System (ROS 2) in the Physical AI & Humanoid Robotics textbook.