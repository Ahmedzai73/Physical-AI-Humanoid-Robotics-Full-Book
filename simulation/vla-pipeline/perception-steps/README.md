# Perception-Action Integration for LLM-Controlled Robots

This module implements perception-action integration for the Vision-Language-Action (VLA) system in the Physical AI & Humanoid Robotics textbook. The perception-action system allows the AI to perceive the environment and take appropriate actions based on the perceived information.

## Overview

The perception-action integration system processes sensor data (camera, LIDAR, point cloud) to understand the environment and generates appropriate actions based on LLM commands. It handles object detection, obstacle detection, environment mapping, and action planning.

### Key Features

1. **Object Detection**: Detect objects in camera images using computer vision techniques
2. **Obstacle Detection**: Identify obstacles using LIDAR data
3. **Environment Mapping**: Create environment maps using multiple sensors
4. **Object Tracking**: Track objects across multiple frames
5. **Action Planning**: Plan robot actions based on perception results
6. **Multi-Sensor Fusion**: Integrate data from multiple sensors for robust perception
7. **Feedback System**: Provide detailed feedback during perception and action operations

## Architecture

```
Sensors (Camera, LIDAR, PointCloud) → Perception System → Action Planning → Robot Actions
                    ↑                          ↓                    ↓
            LLM Perception Requests ←————— Perception Results —————→ Action Commands
```

The system integrates multiple sensors to create a comprehensive understanding of the environment and translates this understanding into appropriate robot actions.

## Components

### 1. Perception-Action Integration Node (`perception_action_integration.py`)

The main node that processes sensor data and generates appropriate actions based on LLM commands.

**Topics Subscribed:**
- `/camera/image_raw` - Camera images for visual perception
- `/scan` - LIDAR scan data for distance perception
- `/pointcloud` - Point cloud data for 3D perception
- `/vla/llm_perception_request` - Perception requests from VLA system

**Topics Published:**
- `/vla/action_command` - Action commands for robot execution
- `/perception_action/status` - Perception system status information
- `/perception_action/feedback` - Detailed feedback during perception operations

### 2. Launch File (`perception_action_integration.launch.py`)

Launch file to start the perception-action integration system with configurable parameters.

## Configuration Parameters

- `object_detection_threshold`: Minimum confidence threshold for object detection (default: 0.5)
- `distance_threshold`: Maximum distance for obstacle detection (default: 2.0m)
- `tracking_enabled`: Enable object tracking (default: true)

## Usage

### Running the Perception-Action System

```bash
# Launch the perception-action integration system
ros2 launch physical_ai_robotics perception_action_integration.launch.py
```

### Integration with VLA Pipeline

The perception-action system should be integrated into the VLA pipeline as follows:

1. Sensor data (camera, LIDAR, point cloud) is fed into the perception system
2. LLM generates perception requests and publishes to `/vla/llm_perception_request`
3. Perception system processes sensor data and generates perception results
4. Action planning module determines appropriate actions based on perception
5. Action commands are published to `/vla/action_command` for robot execution

## Request Types

The system supports multiple types of perception requests:

### 1. Object Detection Request
```json
{
  "type": "object_detection",
  "target": "red_cube"
}
```

### 2. Obstacle Detection Request
```json
{
  "type": "obstacle_detection"
}
```

### 3. Environment Mapping Request
```json
{
  "type": "environment_mapping"
}
```

### 4. Object Tracking Request
```json
{
  "type": "object_tracking",
  "target": "person"
}
```

### 5. Action Planning Request
```json
{
  "type": "action_planning",
  "target_object": "blue_cylinder",
  "action_type": "navigate_to_object"
}
```

## Perception Capabilities

### Object Detection
The system performs object detection using computer vision techniques:
- Color-based detection as a simple example
- Extensible to deep learning-based detection
- Returns object positions, sizes, and confidence scores

### Obstacle Detection
The system detects obstacles using LIDAR data:
- Range-based obstacle detection
- Angular position and distance information
- Obstacle avoidance planning

### Environment Mapping
The system creates environment maps using multiple sensors:
- Combines visual, distance, and 3D data
- Creates comprehensive environment representation
- Supports navigation and planning

## Action Planning

Based on perception results, the system can plan various actions:

### Navigate to Object
- Move towards detected objects
- Center the object in the camera view
- Approach when close enough

### Avoid Obstacles
- Detect obstacles in the robot's path
- Plan alternative routes around obstacles
- Stop when obstacles are too close

### Follow Object
- Track moving objects
- Maintain appropriate distance
- Adjust orientation to follow the object

## Performance Metrics

The perception-action system tracks and reports the following metrics:

- Object detection accuracy
- Obstacle detection rate
- Action planning success rate
- Perception processing time
- Sensor fusion effectiveness

## Example Integration

```python
# In your VLA system, request perception services
import rclpy
from std_msgs.msg import String
from rclpy.node import Node
import json

class VLAExampleNode(Node):
    def __init__(self):
        super().__init__('vla_example_node')

        # Publish perception requests
        self.perception_request_pub = self.create_publisher(
            String,
            '/vla/llm_perception_request',
            10
        )

        # Subscribe to perception feedback
        self.perception_feedback_sub = self.create_subscription(
            String,
            '/perception_action/feedback',
            self.perception_feedback_callback,
            10
        )

    def request_object_detection(self, target_object):
        request = {
            "type": "object_detection",
            "target": target_object
        }

        msg = String()
        msg.data = json.dumps(request)

        # Request object detection from the perception system
        self.perception_request_pub.publish(msg)

    def perception_feedback_callback(self, msg):
        try:
            feedback = json.loads(msg.data)
            self.get_logger().info(f'Perception feedback: {feedback}')
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON in perception feedback: {msg.data}')
```

## Testing

The perception-action system includes various test scenarios to validate its functionality:

1. **Object Detection Test**: Verify object detection in camera images
2. **Obstacle Detection Test**: Test obstacle detection using LIDAR data
3. **Environment Mapping Test**: Validate comprehensive environment mapping
4. **Action Planning Test**: Test action generation based on perception results
5. **Sensor Fusion Test**: Verify integration of multiple sensor modalities

## Troubleshooting

### No objects detected
- Check camera calibration and lighting conditions
- Verify sensor data is being received
- Adjust detection thresholds if needed

### Poor obstacle detection
- Verify LIDAR sensor is functioning properly
- Check for sensor calibration issues
- Adjust distance thresholds as needed

### Delayed responses
- Check system performance and CPU usage
- Verify sensor data rates are appropriate
- Consider optimizing perception algorithms

## Integration with Textbook

This perception-action integration system is designed to work with the Vision-Language-Action (VLA) pipeline described in Module 4 of the Physical AI & Humanoid Robotics textbook. It demonstrates how to implement perception-action loops when integrating LLM-generated commands with robotic systems.