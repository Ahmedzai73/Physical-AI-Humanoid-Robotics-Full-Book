# Module 2: The Digital Twin (Gazebo & Unity) - Simulation Starter

This directory contains starter files for digital twin simulations as covered in Module 2 of the Physical AI & Humanoid Robotics textbook.

## Directory Structure

```
module-2-digital-twin/
├── gazebo_worlds/
│   ├── basic_room.world
│   ├── robot_model/
│   │   ├── model.sdf
│   │   └── model.config
│   └── launch/
│       └── basic_world.launch.py
├── unity_scenes/
│   ├── BasicRobotEnvironment.unity
│   └── Config/
│       └── robot_config.json
├── ros_unity_bridge/
│   ├── bridge_config.yaml
│   └── launch/
│       └── ros_unity_bridge.launch.py
└── README.md
```

## Getting Started

1. For Gazebo simulation:
   ```bash
   cd gazebo_worlds
   ros2 launch launch/basic_world.launch.py
   ```

2. For Unity simulation, open the project in Unity and load the BasicRobotEnvironment scene

3. To connect ROS with Unity:
   ```bash
   ros2 launch ros_unity_bridge/launch/ros_unity_bridge.launch.py
   ```

## Key Concepts Demonstrated

- Gazebo physics simulation
- URDF to SDF conversion
- Unity HDRP environment setup
- ROS-Unity bridge communication
- Digital twin synchronization

For detailed explanations, refer to Module 2: The Digital Twin (Gazebo & Unity) in the Physical AI & Humanoid Robotics textbook.