# Complete Voice → Plan → Navigate → Perceive → Manipulate Pipeline

This module implements the complete Vision-Language-Action (VLA) pipeline for the Physical AI & Humanoid Robotics textbook. The complete pipeline integrates voice commands, cognitive planning, navigation, perception, and manipulation in a unified system.

## Overview

The complete VLA pipeline processes natural language commands through a series of integrated steps: Voice → Plan → Navigate → Perceive → Manipulate. This creates a seamless flow from human intent to robotic action execution.

### Key Features

1. **Voice Command Processing**: Natural language understanding and command parsing
2. **Cognitive Planning**: High-level task planning and decomposition
3. **Integrated Navigation**: Coordinated movement through environments
4. **Real-time Perception**: Continuous environmental awareness
5. **Robotic Manipulation**: Precise object interaction capabilities
6. **Pipeline Orchestration**: Coordinated execution of all pipeline stages
7. **Feedback Integration**: Adaptive responses based on sensor feedback
8. **Error Handling**: Robust error detection and recovery mechanisms

## Architecture

```
Voice Command → Cognitive Planning → Navigation → Perception → Manipulation → Robot Action
      ↑              ↓                ↓           ↓            ↓              ↑
   Natural      Task Decomposition  Movement   Environment  Interaction  Physical
   Language        Planning        Planning   Understanding  Planning    Execution
```

The system orchestrates all VLA components to execute complex tasks based on natural language commands.

## Components

### 1. Complete VLA Pipeline Node (`complete_vla_pipeline.py`)

The main node that orchestrates the complete VLA pipeline from voice input to robotic action.

**Topics Subscribed:**
- `/vla/voice_input` - Voice commands from user
- `/nav2_integration/status` - Navigation system status
- `/perception_action/feedback` - Perception system feedback
- `/manipulation/status` - Manipulation system status

**Topics Published:**
- `/vla/voice_command` - Processed voice commands
- `/vla/llm_navigation_route` - Navigation commands for Nav2
- `/vla/llm_perception_request` - Perception requests
- `/vla/llm_manipulation_command` - Manipulation commands
- `/vla/pipeline/status` - Pipeline execution status
- `/vla/pipeline/feedback` - Detailed pipeline feedback

### 2. Launch File (`complete_vla_pipeline.launch.py`)

Launch file to start the complete VLA pipeline system with configurable parameters.

## Configuration Parameters

- `max_pipeline_retries`: Maximum number of retries for pipeline steps (default: 3)
- `pipeline_timeout`: Timeout for pipeline steps in seconds (default: 30.0s)
- `enable_feedback_loop`: Enable feedback loop for pipeline adjustments (default: true)

## Usage

### Running the Complete VLA Pipeline

```bash
# Launch the complete VLA pipeline system
ros2 launch physical_ai_robotics complete_vla_pipeline.launch.py
```

### Integration with VLA Pipeline

The complete pipeline system should be integrated as follows:

1. Voice commands are published to `/vla/voice_input`
2. The pipeline node parses and decomposes the command into subtasks
3. Each subtask is routed to the appropriate subsystem (navigation, perception, manipulation)
4. Results from each subsystem are coordinated and fed back into the pipeline
5. Final actions are executed by the robot

## Pipeline Stages

### 1. Voice Processing
- Natural language command input
- Semantic parsing and intent recognition
- Command decomposition into executable tasks

### 2. Cognitive Planning
- Task scheduling and prioritization
- Resource allocation and coordination
- Constraint checking and validation

### 3. Navigation Planning
- Path planning and obstacle avoidance
- Waypoint generation and route optimization
- Dynamic replanning based on environment

### 4. Perception Processing
- Object detection and recognition
- Environment mapping and localization
- State estimation and tracking

### 5. Manipulation Planning
- Grasp planning and trajectory generation
- Force control and compliance planning
- Safety validation and execution

## Command Types

The system supports various types of voice commands:

### Pick and Place Commands
```
"Pick up the red cube and place it on the table"
"Grab the blue bottle and put it in the bin"
```

### Navigation Commands
```
"Go to the kitchen"
"Navigate to the charging station"
"Move to the living room"
```

### Perception Commands
```
"Find the person in the room"
"Locate the blue ball"
"Look for obstacles ahead"
```

### Complex Multi-Step Commands
```
"Go to the kitchen, find the red apple, and bring it to me"
"Move to the office, locate the documents, and place them on the desk"
```

## Pipeline Execution Flow

The pipeline executes commands through the following stages:

1. **Voice Input**: Receive and parse voice command
2. **Task Decomposition**: Break down command into subtasks
3. **Stage Planning**: Plan each pipeline stage (Perceive → Plan → Navigate → Perceive → Manipulate)
4. **Execution**: Execute each stage in sequence
5. **Monitoring**: Monitor execution and handle feedback
6. **Completion**: Report results and handle errors

## Performance Metrics

The complete VLA pipeline tracks and reports the following metrics:

- Voice command understanding accuracy
- Pipeline execution success rate
- Average pipeline execution time
- Stage completion rates
- Error recovery success rate

## Example Integration

```python
# In your application, send voice commands to the pipeline
import rclpy
from std_msgs.msg import String
from rclpy.node import Node
import json

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')

        # Publish voice commands to the pipeline
        self.voice_pub = self.create_publisher(
            String,
            '/vla/voice_input',
            10
        )

        # Subscribe to pipeline feedback
        self.pipeline_feedback_sub = self.create_subscription(
            String,
            '/vla/pipeline/feedback',
            self.pipeline_feedback_callback,
            10
        )

    def send_voice_command(self, command: str):
        msg = String()
        msg.data = command

        # Send voice command to the complete VLA pipeline
        self.voice_pub.publish(msg)

    def pipeline_feedback_callback(self, msg):
        try:
            feedback = json.loads(msg.data)
            self.get_logger().info(f'Pipeline feedback: {feedback["status"]}')
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON in pipeline feedback: {msg.data}')

# Example usage
def main(args=None):
    rclpy.init(args=args)

    voice_node = VoiceCommandNode()

    # Send a command to pick up an object and place it somewhere
    voice_node.send_voice_command("Pick up the red cube and place it on the table")

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        pass
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()
```

## Testing

The complete VLA pipeline includes various test scenarios to validate its functionality:

1. **Simple Command Test**: Verify basic voice command processing
2. **Navigation Test**: Test navigation-only commands
3. **Manipulation Test**: Test manipulation-only commands
4. **Multi-Step Test**: Test complex multi-step commands
5. **Error Recovery Test**: Verify error handling and recovery
6. **Performance Test**: Measure pipeline execution times

## Troubleshooting

### Voice commands not processed
- Check that the voice input topic is correctly connected
- Verify that the pipeline node is running
- Ensure proper audio input configuration

### Pipeline execution fails
- Check that all required subsystems (navigation, perception, manipulation) are running
- Verify topic remappings are correct
- Check for sensor data availability

### Unexpected behavior
- Review pipeline feedback messages for detailed information
- Check individual subsystem logs for specific errors
- Verify robot configuration matches pipeline expectations

## Integration with Textbook

This complete VLA pipeline system is designed to work with the Vision-Language-Action (VLA) pipeline described in Module 4 of the Physical AI & Humanoid Robotics textbook. It demonstrates how to integrate all components of the VLA system to execute complex tasks based on natural language commands.