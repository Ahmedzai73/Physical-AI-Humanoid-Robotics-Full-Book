# Manipulation Tasks for LLM-Controlled Robots

This module implements manipulation tasks (pick, place, align) for the Vision-Language-Action (VLA) system in the Physical AI & Humanoid Robotics textbook. The manipulation system allows the AI to perform complex manipulation tasks based on natural language commands.

## Overview

The manipulation system processes LLM-generated manipulation commands and executes them safely using robotic manipulators. It handles pick, place, and alignment operations with proper approach trajectories and gripper control.

### Key Features

1. **Pick Operations**: Safely pick up objects with approach trajectories and gripper control
2. **Place Operations**: Accurately place objects at specified locations with proper approach
3. **Alignment Operations**: Position objects relative to each other based on specified constraints
4. **Custom Manipulation Sequences**: Execute complex multi-step manipulation tasks
5. **Safety Checks**: Validate object and target positions before manipulation
6. **Feedback System**: Provide status updates during manipulation operations

## Architecture

```
LLM/VLA Command → Manipulation Node → Robot Manipulator
      ↑                      ↓
   Manipulation Cmd        Joint Trajectories
      ↓                      ↑
   Sensor Data ←——————— Joint States
```

The system sits between the VLA command generator and the robot's manipulator control system, processing manipulation commands and executing them safely.

## Components

### 1. Manipulation Node (`manipulation_node.py`)

The main manipulation node that processes incoming manipulation commands and executes them on the robot.

**Topics Subscribed:**
- `/vla/llm_manipulation_command` - Incoming manipulation commands from VLA system
- `/joint_states` - Robot joint states for feedback

**Topics Published:**
- `/manipulation/status` - Manipulation system status information
- `/manipulation/feedback` - Detailed feedback during manipulation operations

**Services Used:**
- `/compute_ik` - Inverse kinematics service for calculating joint positions
- `/compute_fk` - Forward kinematics service for position verification

**Action Clients:**
- `/arm_controller/follow_joint_trajectory` - For arm movement
- `/gripper_controller/follow_joint_trajectory` - For gripper control

### 2. Launch File (`launch/manipulation.launch.py`)

Launch file to start the manipulation system with configurable parameters.

## Configuration Parameters

- `pick_approach_distance`: Distance to approach before picking (default: 0.1m)
- `place_approach_distance`: Distance to approach before placing (default: 0.1m)
- `alignment_tolerance`: Tolerance for alignment operations (default: 0.01m)
- `orientation_tolerance`: Tolerance for orientation alignment (default: 0.1 rad)

## Usage

### Running the Manipulation System

```bash
# Launch the manipulation system
ros2 launch physical_ai_robotics manipulation.launch.py
```

### Integration with VLA Pipeline

The manipulation system should be integrated into the VLA pipeline as follows:

1. VLA system generates manipulation commands and publishes to `/vla/llm_manipulation_command`
2. Manipulation node processes the commands and executes them safely
3. Status and feedback are published to `/manipulation/status` and `/manipulation/feedback`
4. Robot executes the resulting joint trajectories

## Command Types

The system supports multiple types of manipulation commands:

### 1. Pick Operation
```json
{
  "type": "pick",
  "object": {
    "name": "red_cube",
    "x": 0.5,
    "y": 0.2,
    "z": 0.1,
    "orientation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "w": 1.0
    }
  }
}
```

### 2. Place Operation
```json
{
  "type": "place",
  "target": {
    "name": "destination_table",
    "x": 0.8,
    "y": 0.3,
    "z": 0.1,
    "orientation": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "w": 1.0
    }
  }
}
```

### 3. Pick-Place Operation
```json
{
  "type": "pick_place",
  "object": {
    "name": "blue_cylinder",
    "x": 0.4,
    "y": 0.1,
    "z": 0.1
  },
  "target": {
    "name": "storage_bin",
    "x": 0.9,
    "y": 0.4,
    "z": 0.2
  }
}
```

### 4. Alignment Operation
```json
{
  "type": "align",
  "description": "Align object with reference",
  "type": "align_xy",
  "reference_object": {
    "x": 0.5,
    "y": 0.5,
    "z": 0.1
  },
  "target_object": {
    "x": 0.6,
    "y": 0.6,
    "z": 0.1
  },
  "orientation": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "w": 1.0
  }
}
```

### 5. Custom Manipulation Sequence
```json
{
  "type": "custom_manipulation",
  "steps": [
    {
      "type": "move",
      "x": 0.5,
      "y": 0.2,
      "z": 0.3,
      "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    },
    {
      "type": "gripper",
      "command": "close"
    },
    {
      "type": "move",
      "x": 0.8,
      "y": 0.3,
      "z": 0.3
    },
    {
      "type": "gripper",
      "command": "open"
    }
  ]
}
```

## Safety Validation

The system performs the following safety checks:

1. **Object Position Validation**: Ensures object coordinates are valid and reachable
2. **Target Position Validation**: Verifies target coordinates are valid and reachable
3. **Collision Prevention**: Checks for potential collisions during movement
4. **Gripper Control**: Properly controls gripper for secure grasp and release
5. **Approach Trajectories**: Uses safe approach trajectories to avoid collisions

## Performance Metrics

The manipulation system tracks and reports the following metrics:

- Total manipulation commands processed
- Success rate for pick operations
- Success rate for place operations
- Average execution time for manipulation tasks
- Number of safety interventions

## Example Integration

```python
# In your VLA system, publish to the manipulation system
import rclpy
from std_msgs.msg import String
from rclpy.node import Node
import json

class VLAExampleNode(Node):
    def __init__(self):
        super().__init__('vla_example_node')

        # Publish to manipulation system
        self.manipulation_pub = self.create_publisher(
            String,
            '/vla/llm_manipulation_command',
            10
        )

    def send_pick_command(self, object_info):
        command = {
            "type": "pick",
            "object": object_info
        }

        msg = String()
        msg.data = json.dumps(command)

        # This command will be processed by the manipulation system
        self.manipulation_pub.publish(msg)
```

## Testing

The manipulation system includes various test scenarios to validate its functionality:

1. **Pick Operation Test**: Verify safe object pickup with approach trajectories
2. **Place Operation Test**: Test accurate object placement at target locations
3. **Alignment Test**: Validate object alignment relative to references
4. **Custom Sequence Test**: Test complex multi-step manipulation tasks
5. **Safety Test**: Verify safety checks and collision prevention

## Troubleshooting

### Commands not executing
- Check that the manipulation node is running
- Verify topic remappings are correct
- Ensure joint states and IK services are available

### Failed to reach positions
- Verify robot kinematics and joint limits
- Check that target positions are within reachable workspace
- Ensure sufficient clearance for approach trajectories

### Gripper not responding
- Verify gripper controller is available
- Check joint names match robot configuration
- Confirm gripper limits and control parameters

## Integration with Textbook

This manipulation system is designed to work with the Vision-Language-Action (VLA) pipeline described in Module 4 of the Physical AI & Humanoid Robotics textbook. It demonstrates how to implement manipulation capabilities when integrating LLM-generated commands with robotic systems.