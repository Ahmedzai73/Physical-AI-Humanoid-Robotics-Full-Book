---
title: Topics - Publish-Subscribe Messaging in ROS 2
sidebar_position: 4
description: Understanding ROS 2 topics for asynchronous data exchange between nodes
---

# Topics: Pub–Sub Messaging + Practical Examples

## Introduction

In the previous chapter, we explored ROS 2 nodes - the computational units of our robotic systems. Now we'll dive into the primary communication mechanism between nodes: **topics**. Topics implement the publish-subscribe pattern, which is fundamental to ROS 2 and especially important for humanoid robots with many sensors and actuators that need to communicate asynchronously.

## Understanding the Publish-Subscribe Pattern

The publish-subscribe (pub-sub) pattern is a messaging paradigm where senders (publishers) broadcast messages without knowing who will receive them, and receivers (subscribers) express interest in specific types of messages without knowing who sent them.

### Key Concepts

- **Publisher**: A node that sends messages to a topic
- **Subscriber**: A node that receives messages from a topic
- **Topic**: A named channel for message exchange
- **Message**: The data structure sent between nodes
- **Message Type**: Defines the structure and content of messages

## Topic Communication in Humanoid Robots

For humanoid robots, topics are essential for:

- **Sensor Data Distribution**: Camera images, LiDAR scans, IMU data flowing from sensors to processing nodes
- **Control Commands**: Joint position/velocity commands flowing from controllers to actuators
- **State Information**: Robot pose, joint positions, battery status shared between nodes
- **Event Notifications**: Collision warnings, goal completions, system status changes

## Creating Publishers

Let's start by creating a publisher node that could be used in a humanoid robot system:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
import math
import random

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Create publisher for joint states
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)

        # Create timer to publish at regular intervals
        timer_period = 0.05  # 20 Hz (50ms)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize joint information
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint'
        ]

        self.get_logger().info('Joint State Publisher initialized')

    def timer_callback(self):
        msg = JointState()

        # Set timestamp
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Set joint names
        msg.name = self.joint_names

        # Generate simulated joint positions (in radians)
        # In a real robot, these would come from encoders
        msg.position = [math.sin(self.get_clock().now().nanoseconds * 1e-9 + i)
                       for i in range(len(self.joint_names))]

        # Generate simulated velocities
        msg.velocity = [math.cos(self.get_clock().now().nanoseconds * 1e-9 + i)
                       for i in range(len(self.joint_names))]

        # Generate simulated efforts
        msg.effort = [random.uniform(-1.0, 1.0) for _ in range(len(self.joint_names))]

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published joint states: {len(msg.name)} joints')

def main(args=None):
    rclpy.init(args=args)
    joint_state_publisher = JointStatePublisher()

    try:
        rclpy.spin(joint_state_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        joint_state_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Subscribers

Now let's create a subscriber that receives and processes the joint state messages:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

class JointStateProcessor(Node):
    def __init__(self):
        super().__init__('joint_state_processor')

        # Create subscriber for joint states
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10)  # QoS depth

        # Store previous state for velocity calculation
        self.previous_positions = {}
        self.previous_time = None

        self.get_logger().info('Joint State Processor initialized')

    def joint_state_callback(self, msg):
        self.get_logger().info(f'Received joint states from {len(msg.name)} joints')

        # Process each joint
        for i, joint_name in enumerate(msg.name):
            if i < len(msg.position):
                position = msg.position[i]
                self.get_logger().debug(f'{joint_name}: pos={position:.3f}')

                # Store for velocity calculation
                if self.previous_time is not None:
                    dt = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) - \
                         (self.previous_time.sec + self.previous_time.nanosec * 1e-9)

                    if dt > 0 and joint_name in self.previous_positions:
                        velocity = (position - self.previous_positions[joint_name]) / dt
                        self.get_logger().info(f'{joint_name} velocity: {velocity:.3f} rad/s')

        # Update previous state
        self.previous_time = msg.header.stamp
        for i, joint_name in enumerate(msg.name):
            if i < len(msg.position):
                self.previous_positions[joint_name] = msg.position[i]

def main(args=None):
    rclpy.init(args=args)
    joint_state_processor = JointStateProcessor()

    try:
        rclpy.spin(joint_state_processor)
    except KeyboardInterrupt:
        pass
    finally:
        joint_state_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service (QoS) for Topics

For humanoid robots, choosing the right QoS settings is crucial for performance and reliability:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class QoSDemonstration(Node):
    def __init__(self):
        super().__init__('qos_demonstration')

        # QoS for sensor data (camera images) - best effort, volatile
        sensor_qos = QoSProfile(
            depth=5,  # Keep only recent messages
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Allow message drops for low latency
            durability=DurabilityPolicy.VOLATILE,  # Don't keep messages for late joiners
            history=HistoryPolicy.KEEP_LAST  # Keep only the last N messages
        )

        # QoS for critical commands - reliable, with history
        command_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,  # Ensure delivery
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # QoS for configuration parameters - transient local
        config_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # Deliver to late joiners
            history=HistoryPolicy.KEEP_LAST
        )

        # Create publishers with different QoS
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', sensor_qos)
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', command_qos)

        # Create subscribers with matching QoS
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, sensor_qos)
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, command_qos)

    def image_callback(self, msg):
        self.get_logger().info(f'Received image: {msg.width}x{msg.height}')

    def joint_callback(self, msg):
        self.get_logger().info(f'Received joint state for {len(msg.name)} joints')

def main(args=None):
    rclpy.init(args=args)
    qos_demo = QoSDemonstration()

    try:
        rclpy.spin(qos_demo)
    except KeyboardInterrupt:
        pass
    finally:
        qos_demo.destroy_node()
        rclpy.shutdown()
```

## Message Types and Custom Messages

ROS 2 provides many standard message types, but you can also create custom ones. Here's how to work with different message types:

### Standard Message Types
```python
from std_msgs.msg import String, Int32, Float64
from sensor_msgs.msg import JointState, Image, LaserScan
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Odometry
```

### Creating Custom Messages
First, create a message definition file (e.g., `HumanoidStatus.msg`):

```
# Custom message for humanoid robot status
string robot_name
float64[] joint_temperatures
float64 battery_level
bool[] joint_faults
builtin_interfaces/Time timestamp
```

Then use it in your code:
```python
from your_package_name.msg import HumanoidStatus

class StatusPublisher(Node):
    def __init__(self):
        super().__init__('status_publisher')
        self.publisher_ = self.create_publisher(HumanoidStatus, 'robot_status', 10)
        # ... rest of implementation
```

## Advanced Topic Patterns

### 1. Multiple Publishers, Multiple Subscribers
```python
# One publisher, multiple subscribers pattern
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribe to multiple sensor topics
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        # Publish fused data
        self.fused_pub = self.create_publisher(
            Odometry, 'fused_odom', 10)

    def imu_callback(self, msg):
        # Process IMU data
        pass

    def odom_callback(self, msg):
        # Process odometry data
        pass

    def laser_callback(self, msg):
        # Process laser data
        pass
```

### 2. Topic Remapping
```python
# Remap topics for different robot instances
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='your_package',
            executable='your_node',
            name='node_with_remap',
            remappings=[
                ('input_topic', 'robot1_input'),
                ('output_topic', 'robot1_output')
            ]
        )
    ])
```

## Practical Example: Humanoid Perception Pipeline

Let's create a practical example showing how topics work in a humanoid robot perception system:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
import cv2
from cv_bridge import CvBridge
import numpy as np

class HumanoidPerceptionNode(Node):
    def __init__(self):
        super().__init__('humanoid_perception')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Subscribe to camera topics
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10)

        # Publishers for processed data
        self.depth_pub = self.create_publisher(
            Image, 'camera/depth_processed', 10)
        self.marker_pub = self.create_publisher(
            Marker, 'visualization_marker', 10)
        self.object_pub = self.create_publisher(
            PointStamped, 'detected_object', 10)

        # Store camera info
        self.camera_info = None

        self.get_logger().info('Humanoid Perception Node initialized')

    def camera_info_callback(self, msg):
        """Store camera calibration information"""
        self.camera_info = msg

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform object detection (simplified example)
            detected_objects = self.detect_objects(cv_image)

            # Publish detected objects
            for obj in detected_objects:
                point_msg = PointStamped()
                point_msg.header = msg.header
                point_msg.point.x = obj['x']
                point_msg.point.y = obj['y']
                point_msg.point.z = obj['z']  # Estimated depth

                self.object_pub.publish(point_msg)

                # Publish visualization marker
                marker = self.create_marker(obj, msg.header)
                self.marker_pub.publish(marker)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_objects(self, cv_image):
        """Simple object detection (placeholder for real algorithm)"""
        # This would typically use a deep learning model
        # For this example, we'll simulate detection
        height, width = cv_image.shape[:2]

        # Simulate detecting a person in the center
        objects = [{
            'x': width / 2,
            'y': height / 2,
            'width': 100,
            'height': 200,
            'class': 'person',
            'confidence': 0.95
        }]

        return objects

    def create_marker(self, obj, header):
        """Create a visualization marker for detected objects"""
        marker = Marker()
        marker.header = header
        marker.ns = "detection"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Position in image coordinates (simplified)
        marker.pose.position.x = obj['x']
        marker.pose.position.y = obj['y']
        marker.pose.position.z = 1.0  # Fixed depth for visualization

        marker.scale.x = obj['width'] / 100.0
        marker.scale.y = obj['height'] / 100.0
        marker.scale.z = 0.1

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        return marker

def main(args=None):
    rclpy.init(args=args)
    perception_node = HumanoidPerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        pass
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()
```

## Topic Tools and Debugging

ROS 2 provides several command-line tools for working with topics:

```bash
# List all topics
ros2 topic list

# Show topic information
ros2 topic info /joint_states

# Echo messages from a topic
ros2 topic echo /joint_states

# Echo with field filtering
ros2 topic echo /joint_states name --field position

# Show topic type
ros2 topic type /camera/image_raw

# Publish to a topic (for testing)
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}'

# Monitor topic rate
ros2 topic hz /camera/image_raw
```

## Performance Considerations for Humanoid Robots

### 1. Message Frequency
- High-frequency topics (e.g., joint states): 100-1000 Hz
- Medium-frequency topics (e.g., camera images): 10-30 Hz
- Low-frequency topics (e.g., configuration): 1-5 Hz

### 2. Message Size
- Keep messages small for high-frequency topics
- Use compression for large data (images, point clouds)
- Consider subsampling when appropriate

### 3. QoS Configuration
- Use appropriate reliability for each data type
- Configure history depth based on application needs
- Match publisher and subscriber QoS settings

## Best Practices

### 1. Naming Conventions
```python
# Good naming
'/arm/joint_states'
'/camera/rgb/image_raw'
'/mobile_base/cmd_vel'
'/tf'  # Standard transform topic

# Avoid
'/data'  # Too generic
'/robot1_data'  # Instance-specific names
```

### 2. Topic Organization
```python
# Organize by function
'/sensors/imu/data'
'/sensors/camera/rgb/image_raw'
'/actuators/joint_commands'
'/planning/trajectory'

# Or by robot part
'/head/camera/image_raw'
'/left_arm/joint_states'
'/right_leg/force_torque'
```

### 3. Error Handling
```python
def image_callback(self, msg):
    try:
        # Process message
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # ... further processing
    except Exception as e:
        self.get_logger().error(f'Failed to process image: {e}')
        # Don't crash the node, continue processing other messages
```

## Summary

Topics provide the asynchronous communication backbone for ROS 2 systems. For humanoid robots, understanding topics is crucial because:

- They enable efficient distribution of sensor data from multiple sources
- They allow for decoupled, modular system design
- They support different QoS requirements for various data types
- They scale well for complex systems with many interacting components

The publish-subscribe pattern allows different parts of a humanoid robot to communicate without tight coupling, making systems more robust and maintainable.

## Exercises

1. Create a publisher-subscriber pair that demonstrates different QoS profiles and observe the differences in message delivery.

2. Design a topic architecture for a humanoid robot with:
   - 20 joints with position, velocity, and effort feedback
   - 2 cameras (front and head)
   - IMU, force-torque sensors, and LiDAR
   - Planning and control modules

3. Implement a simple message filter that processes incoming sensor data and publishes processed results, using appropriate QoS settings for your use case.

## Next Steps

In the next chapter, we'll explore services and actions, which provide synchronous and long-running communication patterns that complement the asynchronous nature of topics. These are essential for humanoid robots when you need guaranteed responses or need to execute complex behaviors with feedback.

Continue to Chapter 5: [Services & Actions: Request/Response + Long-running Tasks](./services-actions.md) to learn about synchronous communication in ROS 2.