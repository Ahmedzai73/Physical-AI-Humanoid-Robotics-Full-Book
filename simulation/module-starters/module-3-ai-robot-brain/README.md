# Module 3: The AI-Robot Brain (NVIDIA Isaac™) - Simulation Starter

This directory contains starter files for AI robot brain simulations as covered in Module 3 of the Physical AI & Humanoid Robotics textbook.

## Directory Structure

```
module-3-ai-robot-brain/
├── isaac_sim_scenes/
│   ├── basic_scene.py
│   └── scene_config.toml
├── perception_nodes/
│   ├── vision_node.py
│   ├── lidar_processing.py
│   └── sensor_fusion.py
├── navigation_stack/
│   ├── nav2_config.yaml
│   └── launch/
│       └── navigation.launch.py
├── ai_brain/
│   ├── behavior_tree.xml
│   ├── neural_network.py
│   └── decision_making.py
└── README.md
```

## Getting Started

1. For Isaac Sim scene:
   ```bash
   # Run within Isaac Sim environment
   python basic_scene.py
   ```

2. For perception nodes:
   ```bash
   ros2 run perception_nodes vision_node
   ros2 run perception_nodes lidar_processing
   ```

3. For navigation:
   ```bash
   ros2 launch navigation_stack/launch/navigation.launch.py
   ```

## Key Concepts Demonstrated

- Isaac Sim photorealistic simulation
- GPU-accelerated perception
- Visual SLAM
- Isaac ROS integration
- AI-driven decision making
- Behavior trees for robot control

For detailed explanations, refer to Module 3: The AI-Robot Brain (NVIDIA Isaac™) in the Physical AI & Humanoid Robotics textbook.