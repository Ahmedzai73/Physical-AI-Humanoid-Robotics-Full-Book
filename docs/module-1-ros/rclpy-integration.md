---
title: rclpy - Bridging Python Agents to ROS Controllers
sidebar_position: 8
description: Understanding the Python client library for ROS 2 and integrating AI agents with robotic controllers
---

# rclpy: Bridging Python Agents to ROS Controllers

## Introduction

In previous chapters, we've explored the core concepts of ROS 2: nodes, topics, services, actions, and launch files. Now we'll focus on **rclpy**, the Python client library for ROS 2. This is particularly important for humanoid robots as Python is widely used in AI and machine learning applications that need to interface with robotic systems.

rclpy provides Python APIs for all ROS 2 concepts and enables seamless integration between high-level AI algorithms and low-level robotic controllers. For humanoid robots that often use AI for perception, planning, and decision-making, mastering rclpy is essential.

## Understanding rclpy Architecture

rclpy is built on top of the ROS Client Library (rcl) and provides Python-specific abstractions:

```mermaid
graph TD
    A[Python AI Agent] --> B[rclpy API]
    B --> C[rcl - ROS Client Library]
    C --> D[ROS Middleware Interface (RMW)]
    D --> E[DDS Implementation]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
```

### Key Components of rclpy

- `Node`: The base class for creating ROS 2 nodes
- `Publisher`: For publishing messages to topics
- `Subscriber`: For subscribing to topics
- `Client`: For calling services
- `Service`: For creating service servers
- `ActionClient`: For sending action goals
- `ActionServer`: For executing action goals
- `Parameter`: For managing node parameters

## Basic rclpy Concepts

### 1. Node Creation and Management

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class AIControllerNode(Node):
    def __init__(self):
        # Initialize the node with a name
        super().__init__('ai_controller_node')

        # Create a publisher
        self.publisher_ = self.create_publisher(String, 'ai_commands', 10)

        # Create a subscriber
        self.subscription = self.create_subscription(
            String,
            'sensor_data',
            self.sensor_callback,
            10
        )

        # Create a timer for periodic execution
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('AI Controller Node initialized')

    def sensor_callback(self, msg):
        """Handle incoming sensor data"""
        self.get_logger().info(f'Received sensor data: {msg.data}')
        # Process sensor data and generate commands
        self.process_sensor_data(msg.data)

    def control_loop(self):
        """Main control loop"""
        # This runs every 0.1 seconds
        self.get_logger().debug('Control loop executing')

    def process_sensor_data(self, data):
        """Process sensor data and generate commands"""
        # AI processing logic would go here
        command = f'Processed: {data}'

        # Publish the command
        msg = String()
        msg.data = command
        self.publisher_.publish(msg)

def main(args=None):
    # Initialize rclpy
    rclpy.init(args=args)

    # Create the node
    ai_controller = AIControllerNode()

    try:
        # Spin the node to process callbacks
        rclpy.spin(ai_controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        ai_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Advanced Node Features

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import numpy as np

class AdvancedAIController(Node):
    def __init__(self):
        super().__init__('advanced_ai_controller')

        # Define QoS profiles for different data types
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        control_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publishers with custom QoS
        self.joint_command_pub = self.create_publisher(
            JointState, 'joint_commands', control_qos)
        self.status_pub = self.create_publisher(
            String, 'ai_status', 10)

        # Subscribers with custom QoS
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, sensor_qos)

        # Store robot state
        self.current_joint_positions = {}
        self.imu_data = None

        # Timer with custom period
        self.control_timer = self.create_timer(
            0.05,  # 20 Hz control loop
            self.control_callback
        )

        self.get_logger().info('Advanced AI Controller initialized')

    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def control_callback(self):
        """Main control loop with AI integration"""
        # Check if we have necessary data
        if not self.current_joint_positions or not self.imu_data:
            return

        # AI-based control logic
        commands = self.compute_control_commands()

        # Publish joint commands
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = list(commands.keys())
        joint_msg.position = list(commands.values())

        self.joint_command_pub.publish(joint_msg)

    def compute_control_commands(self):
        """AI-based control computation"""
        # This is where machine learning models would be integrated
        commands = {}

        # Example: Simple balance control based on IMU data
        if self.imu_data:
            # Simplified balance control
            roll, pitch = self.get_orientation_angles(self.imu_data['orientation'])

            # Adjust joint positions based on balance
            for joint_name in self.current_joint_positions:
                if 'hip' in joint_name:
                    # Compensate for balance
                    compensation = -pitch * 0.1
                    current_pos = self.current_joint_positions[joint_name]
                    commands[joint_name] = current_pos + compensation
                else:
                    commands[joint_name] = self.current_joint_positions[joint_name]

        return commands

    def get_orientation_angles(self, orientation_quat):
        """Convert quaternion to roll/pitch/yaw"""
        # Simplified conversion (in practice, use tf2 for robust conversion)
        x, y, z, w = orientation_quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        return roll, pitch
```

## Integrating AI Agents with ROS 2

### 1. Basic AI Integration Pattern

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
import numpy as np
import tensorflow as tf  # Example ML library

class AIBasedNavigator(Node):
    def __init__(self):
        super().__init__('ai_navigator')

        # Subscribe to sensor data
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)

        # Publisher for movement commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Publisher for AI decisions
        self.decision_pub = self.create_publisher(String, 'ai_decision', 10)

        # Initialize AI model
        self.ai_model = self.initialize_model()

        # Store sensor data
        self.latest_image = None
        self.latest_laser = None

        # Control timer
        self.nav_timer = self.create_timer(0.1, self.navigation_callback)

        self.get_logger().info('AI Navigator initialized')

    def initialize_model(self):
        """Initialize or load AI model"""
        # In a real implementation, you would load a pre-trained model
        # For this example, we'll create a simple placeholder
        self.get_logger().info('Loading AI navigation model...')

        # Example: Load a TensorFlow model
        # model = tf.saved_model.load('path/to/model')
        # return model

        # For now, return a placeholder
        return {'initialized': True}

    def image_callback(self, msg):
        """Process camera images for AI analysis"""
        # Convert ROS image to format suitable for AI model
        image_data = self.convert_image_msg_to_array(msg)
        self.latest_image = image_data

    def laser_callback(self, msg):
        """Process laser scan for AI analysis"""
        # Convert laser scan to format suitable for AI model
        laser_data = np.array(msg.ranges)
        self.latest_laser = laser_data

    def convert_image_msg_to_array(self, img_msg):
        """Convert ROS Image message to numpy array"""
        # This is a simplified conversion
        # In practice, use cv_bridge for proper conversion
        import cv2
        from cv_bridge import CvBridge

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        return cv_image

    def navigation_callback(self):
        """Main navigation logic using AI"""
        if self.latest_image is not None and self.latest_laser is not None:
            # Run AI inference
            navigation_command = self.run_ai_navigation(
                self.latest_image,
                self.latest_laser
            )

            # Publish the command
            self.publish_navigation_command(navigation_command)

    def run_ai_navigation(self, image, laser_scan):
        """Run AI-based navigation inference"""
        # This is where the actual AI processing happens
        # For this example, we'll use a simple heuristic

        # Example: Simple obstacle avoidance based on laser scan
        if len(laser_scan) > 0:
            min_distance = min([d for d in laser_scan if not np.isnan(d) and d > 0])

            if min_distance < 0.5:  # Obstacle too close
                return {'linear': 0.0, 'angular': 0.5}  # Turn
            else:
                return {'linear': 0.5, 'angular': 0.0}  # Go forward

        return {'linear': 0.0, 'angular': 0.0}  # Stop

    def publish_navigation_command(self, command):
        """Publish navigation command to robot"""
        twist_msg = Twist()
        twist_msg.linear.x = command['linear']
        twist_msg.angular.z = command['angular']

        self.cmd_vel_pub.publish(twist_msg)

        # Log the decision
        decision_msg = String()
        decision_msg.data = f"Linear: {command['linear']}, Angular: {command['angular']}"
        self.decision_pub.publish(decision_msg)
```

### 2. Advanced AI Integration with Model Serving

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Float32MultiArray
import requests
import json
import threading
from queue import Queue

class ModelServingAIController(Node):
    def __init__(self):
        super().__init__('model_serving_ai_controller')

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback, 10)
        self.control_pub = self.create_publisher(
            JointState, 'ai_joint_commands', 10)

        # Internal state
        self.current_state = {
            'joint_positions': {},
            'image_data': None,
            'command_queue': Queue()
        }

        # Model server configuration
        self.model_server_url = 'http://localhost:8501/v1/models/robot_control:predict'

        # Asynchronous processing
        self.processing_thread = threading.Thread(target=self.process_commands)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Control timer
        self.control_timer = self.create_timer(0.05, self.control_callback)

        self.get_logger().info('Model Serving AI Controller initialized')

    def joint_state_callback(self, msg):
        """Update joint state"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_state['joint_positions'][name] = msg.position[i]

    def camera_callback(self, msg):
        """Process camera data"""
        # Convert and store image data for AI processing
        image_array = self.convert_image_msg_to_array(msg)
        self.current_state['image_data'] = image_array

    def control_callback(self):
        """Main control callback"""
        if self.current_state['joint_positions'] and self.current_state['image_data']:
            # Prepare input for AI model
            model_input = self.prepare_model_input()

            # Send to model server asynchronously
            self.send_to_model_server_async(model_input)

    def prepare_model_input(self):
        """Prepare input data for AI model"""
        # Extract relevant features from current state
        joint_positions = list(self.current_state['joint_positions'].values())

        # Prepare model input (simplified)
        input_data = {
            'instances': [{
                'joint_positions': joint_positions,
                'image_features': self.extract_image_features(self.current_state['image_data'])
            }]
        }

        return input_data

    def extract_image_features(self, image_data):
        """Extract features from image data"""
        # In a real implementation, this would run a feature extraction model
        # For now, return a simplified representation
        return [0.0] * 10  # Placeholder

    def send_to_model_server_async(self, input_data):
        """Send data to model server asynchronously"""
        def send_request():
            try:
                response = requests.post(
                    self.model_server_url,
                    data=json.dumps(input_data),
                    headers={'Content-Type': 'application/json'}
                )

                if response.status_code == 200:
                    result = response.json()
                    # Process the AI model result
                    self.process_model_result(result)
                else:
                    self.get_logger().error(f'Model server error: {response.status_code}')

            except Exception as e:
                self.get_logger().error(f'Error sending to model server: {e}')

        # Run in separate thread to avoid blocking ROS callbacks
        thread = threading.Thread(target=send_request)
        thread.daemon = True
        thread.start()

    def process_model_result(self, result):
        """Process AI model results and queue commands"""
        # Extract joint commands from model result
        if 'predictions' in result and len(result['predictions']) > 0:
            prediction = result['predictions'][0]

            # Create joint command
            joint_command = JointState()
            joint_command.header.stamp = self.get_clock().now().to_msg()
            joint_command.name = list(self.current_state['joint_positions'].keys())
            joint_command.position = prediction.get('joint_commands', [0.0] * len(joint_command.name))

            # Add to command queue
            self.current_state['command_queue'].put(joint_command)

    def process_commands(self):
        """Process commands from queue in separate thread"""
        while rclpy.ok():
            try:
                # Get command from queue (non-blocking)
                if not self.current_state['command_queue'].empty():
                    command = self.current_state['command_queue'].get_nowait()
                    self.control_pub.publish(command)
            except:
                pass  # Queue was empty

            # Sleep briefly to prevent busy waiting
            time.sleep(0.01)
```

## Real-time Performance Considerations

When integrating AI agents with ROS 2 controllers, real-time performance is critical:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Header
import time
import threading
from collections import deque

class RealTimeAIController(Node):
    def __init__(self):
        super().__init__('real_time_ai_controller')

        # High-frequency control loop
        self.control_timer = self.create_timer(
            0.005,  # 200 Hz for real-time control
            self.high_freq_control,
            clock=self.get_clock()
        )

        # Publishers and subscribers
        self.joint_state_pub = self.create_publisher(JointState, 'ai_joint_commands', 1)
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 1)

        # Real-time state buffer
        self.state_buffer = deque(maxlen=10)  # Keep last 10 states
        self.command_buffer = deque(maxlen=5)  # Keep last 5 commands

        # Performance monitoring
        self.last_callback_time = time.time()
        self.callback_interval = 0.005  # Expected 200Hz

        self.get_logger().info('Real-time AI Controller initialized')

    def joint_state_callback(self, msg):
        """High-priority state callback"""
        # Add timestamp for real-time analysis
        state_entry = {
            'timestamp': self.get_clock().now(),
            'positions': dict(zip(msg.name, msg.position)),
            'velocities': dict(zip(msg.name, msg.velocity)) if msg.velocity else {},
            'efforts': dict(zip(msg.name, msg.effort)) if msg.effort else {}
        }

        self.state_buffer.append(state_entry)

    def high_freq_control(self):
        """High-frequency control loop"""
        current_time = time.time()
        actual_interval = current_time - self.last_callback_time
        self.last_callback_time = current_time

        # Check for timing violations
        if actual_interval > self.callback_interval * 2:
            self.get_logger().warn(f'Timing violation: {actual_interval:.4f}s vs expected {self.callback_interval:.4f}s')

        # Get current state
        if self.state_buffer:
            current_state = self.state_buffer[-1]  # Latest state

            # Compute real-time control command
            control_command = self.compute_real_time_control(current_state)

            # Publish command
            if control_command:
                self.joint_state_pub.publish(control_command)

                # Add to command buffer for monitoring
                self.command_buffer.append({
                    'timestamp': self.get_clock().now(),
                    'command': control_command
                })

    def compute_real_time_control(self, state):
        """Real-time control computation (must be fast!)"""
        # CRITICAL: This function must execute quickly (<1ms)
        # No complex AI models or blocking operations here

        # Simple PD controller example
        try:
            command = JointState()
            command.header.stamp = self.get_clock().now().to_msg()
            command.name = list(state['positions'].keys())

            # Simple position control (in real application, this would use pre-computed AI commands)
            command.position = [pos * 0.99 for pos in state['positions'].values()]  # Dampening
            command.velocity = [0.0] * len(command.name)
            command.effort = [0.0] * len(command.name)

            return command
        except Exception as e:
            self.get_logger().error(f'Control computation error: {e}')
            return None

    def get_performance_stats(self):
        """Get real-time performance statistics"""
        if len(self.command_buffer) >= 2:
            # Calculate actual control frequency
            first_time = self.command_buffer[0]['timestamp']
            last_time = self.command_buffer[-1]['timestamp']
            time_diff = (last_time.nanoseconds - first_time.nanoseconds) * 1e-9
            actual_freq = len(self.command_buffer) / time_diff if time_diff > 0 else 0

            return {
                'target_frequency': 200.0,
                'actual_frequency': actual_freq,
                'buffer_size': len(self.command_buffer)
            }

        return {'target_frequency': 200.0, 'actual_frequency': 0.0, 'buffer_size': 0}
```

## Integration Patterns for Humanoid Robots

### 1. Perception-Action Loop

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import Twist, PoseStamped
from humanoid_msgs.msg import HumanoidJointCommand, HumanoidState
from std_msgs.msg import String
import numpy as np

class HumanoidPerceptionActionLoop(Node):
    def __init__(self):
        super().__init__('humanoid_perception_action_loop')

        # Perception publishers/subscribers
        self.camera_sub = self.create_subscription(Image, 'camera/rgb/image_raw', self.camera_callback, 10)
        self.depth_sub = self.create_subscription(Image, 'camera/depth/image_raw', self.depth_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.joint_state_sub = self.create_subscription(HumanoidState, 'humanoid_state', self.state_callback, 10)

        # Action publishers
        self.joint_cmd_pub = self.create_publisher(HumanoidJointCommand, 'humanoid_joint_commands', 10)
        self.base_cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # AI state
        self.perception_data = {
            'rgb_image': None,
            'depth_image': None,
            'imu_data': None,
            'current_state': None
        }

        # Control timer
        self.control_timer = self.create_timer(0.033, self.perception_action_loop)  # ~30 Hz

        self.get_logger().info('Humanoid Perception-Action Loop initialized')

    def camera_callback(self, msg):
        """Handle RGB camera data"""
        self.perception_data['rgb_image'] = self.convert_image_msg_to_array(msg)

    def depth_callback(self, msg):
        """Handle depth camera data"""
        self.perception_data['depth_image'] = self.convert_image_msg_to_array(msg)

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.perception_data['imu_data'] = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def state_callback(self, msg):
        """Handle humanoid state"""
        self.perception_data['current_state'] = {
            'joint_positions': dict(zip(msg.joint_names, msg.joint_positions)),
            'joint_velocities': dict(zip(msg.joint_names, msg.joint_velocities)),
            'base_pose': msg.base_pose,
            'base_twist': msg.base_twist
        }

    def perception_action_loop(self):
        """Main perception-action loop"""
        if all(self.perception_data.values()):  # All data available
            # Process perception data
            perception_result = self.process_perception_data()

            # Make decisions based on perception
            action_command = self.decide_action(perception_result)

            # Execute action
            self.execute_action(action_command)

    def process_perception_data(self):
        """Process all sensor data to extract meaningful information"""
        result = {}

        # Object detection in RGB image
        if self.perception_data['rgb_image'] is not None:
            result['detected_objects'] = self.detect_objects(self.perception_data['rgb_image'])

        # Obstacle detection from depth
        if self.perception_data['depth_image'] is not None:
            result['obstacles'] = self.detect_obstacles(self.perception_data['depth_image'])

        # Balance state from IMU
        if self.perception_data['imu_data'] is not None:
            result['balance_state'] = self.estimate_balance(self.perception_data['imu_data'])

        # Current configuration
        if self.perception_data['current_state'] is not None:
            result['current_config'] = self.perception_data['current_state']

        return result

    def detect_objects(self, image):
        """Detect objects in image (placeholder for real detection)"""
        # In real implementation, this would use a deep learning model
        return [{'class': 'person', 'confidence': 0.9, 'position': [1.0, 2.0, 0.0]}]

    def detect_obstacles(self, depth_image):
        """Detect obstacles from depth data (placeholder)"""
        # Simple threshold-based obstacle detection
        obstacles = []
        # In real implementation, process depth image to find obstacles
        return obstacles

    def estimate_balance(self, imu_data):
        """Estimate robot balance from IMU data"""
        orientation = imu_data['orientation']
        # Convert quaternion to roll/pitch
        import tf_transformations
        euler = tf_transformations.euler_from_quaternion(orientation)
        roll, pitch, yaw = euler

        return {
            'roll': roll,
            'pitch': pitch,
            'is_balanced': abs(roll) < 0.2 and abs(pitch) < 0.2
        }

    def decide_action(self, perception_result):
        """Make action decision based on perception"""
        action = {
            'joint_commands': {},
            'base_velocity': {'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                             'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}},
            'action_type': 'idle'
        }

        # Balance correction
        if 'balance_state' in perception_result:
            balance = perception_result['balance_state']
            if not balance['is_balanced']:
                action['action_type'] = 'balance_correction'
                action['joint_commands'] = self.compute_balance_correction(balance)

        # Navigation if objects detected
        if 'detected_objects' in perception_result and perception_result['detected_objects']:
            action['action_type'] = 'navigate_to_object'
            action['base_velocity'] = self.compute_navigation_command(perception_result['detected_objects'][0])

        return action

    def compute_balance_correction(self, balance_state):
        """Compute joint commands for balance correction"""
        # Simple balance correction based on IMU data
        corrections = {}

        # Adjust hip joints based on pitch
        corrections['left_hip_joint'] = -balance_state['pitch'] * 0.5
        corrections['right_hip_joint'] = -balance_state['pitch'] * 0.5

        # Adjust ankle joints based on roll
        corrections['left_ankle_joint'] = balance_state['roll'] * 0.3
        corrections['right_ankle_joint'] = balance_state['roll'] * 0.3

        return corrections

    def compute_navigation_command(self, detected_object):
        """Compute base velocity command to navigate to object"""
        # Simple navigation command
        return {
            'linear': {'x': 0.2, 'y': 0.0, 'z': 0.0},  # Move forward
            'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}  # No rotation
        }

    def execute_action(self, action_command):
        """Execute the computed action"""
        if action_command['joint_commands']:
            # Publish joint commands
            joint_cmd_msg = HumanoidJointCommand()
            joint_cmd_msg.header.stamp = self.get_clock().now().to_msg()
            joint_cmd_msg.joint_names = list(action_command['joint_commands'].keys())
            joint_cmd_msg.joint_commands = list(action_command['joint_commands'].values())

            self.joint_cmd_pub.publish(joint_cmd_msg)

        if action_command['base_velocity']:
            # Publish base velocity command
            twist_msg = Twist()
            twist_msg.linear.x = action_command['base_velocity']['linear']['x']
            twist_msg.linear.y = action_command['base_velocity']['linear']['y']
            twist_msg.linear.z = action_command['base_velocity']['linear']['z']
            twist_msg.angular.x = action_command['base_velocity']['angular']['x']
            twist_msg.angular.y = action_command['base_velocity']['angular']['y']
            twist_msg.angular.z = action_command['base_velocity']['angular']['z']

            self.base_cmd_pub.publish(twist_msg)
```

## Performance Optimization

### 1. Memory Management

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
import numpy as np
import gc

class OptimizedAIController(Node):
    def __init__(self):
        super().__init__('optimized_ai_controller')

        # Pre-allocate message objects to reduce memory allocation
        self.image_msg_buffer = None
        self.joint_cmd_buffer = None

        # Reuse numpy arrays
        self.image_array = None
        self.processed_array = None

        # Subscriptions
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.optimized_image_callback, 10)

        # Control timer
        self.process_timer = self.create_timer(0.033, self.process_callback)

        self.get_logger().info('Optimized AI Controller initialized')

    def optimized_image_callback(self, msg):
        """Optimized image callback with minimal memory allocation"""
        # Convert image message to numpy array efficiently
        if self.image_array is None or self.image_array.shape != (msg.height, msg.width, 3):
            # Only reallocate if size changes
            self.image_array = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)

        # Fill the pre-allocated array
        self.image_array.data = msg.data

        # Process image
        self.processed_array = self.process_image_efficiently(self.image_array)

    def process_image_efficiently(self, image_array):
        """Process image with minimal memory allocation"""
        # Use in-place operations where possible
        if self.processed_array is None or self.processed_array.shape != image_array.shape:
            self.processed_array = np.empty_like(image_array)

        # Example: Simple color space conversion (in-place)
        self.processed_array[:] = image_array  # Copy data

        # Perform operations that modify the array in place
        # Instead of creating new arrays, modify existing ones

        return self.processed_array

    def process_callback(self):
        """Process data with memory optimization"""
        if self.processed_array is not None:
            # AI processing that reuses arrays
            result = self.ai_process_reuse_arrays(self.processed_array)

            # Publish result
            self.publish_result(result)

    def ai_process_reuse_arrays(self, input_array):
        """AI processing that reuses arrays"""
        # Example: Feature extraction that reuses arrays
        # Instead of creating new arrays, use existing ones
        return input_array  # Placeholder

    def publish_result(self, result):
        """Publish result efficiently"""
        # Reuse message objects where possible
        pass
```

## Error Handling and Robustness

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from builtin_interfaces.msg import Time
import traceback
import signal
import sys

class RobustAIController(Node):
    def __init__(self):
        super().__init__('robust_ai_controller')

        # Initialize components safely
        self.ai_model = None
        self.safety_enabled = True
        self.emergency_stop = False

        # Publishers
        self.status_pub = self.create_publisher(String, 'ai_status', 10)
        self.error_pub = self.create_publisher(String, 'ai_errors', 10)

        # Initialize AI model with error handling
        self.initialize_ai_model()

        # Control timer
        self.control_timer = self.create_timer(0.1, self.safe_control_loop)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.get_logger().info('Robust AI Controller initialized')

    def initialize_ai_model(self):
        """Initialize AI model with error handling"""
        try:
            # Initialize AI model
            self.ai_model = self.load_ai_model_safely()
            self.publish_status('AI model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load AI model: {e}')
            self.publish_error(f'AI model initialization failed: {e}')
            self.ai_model = None  # Fallback to safe mode

    def load_ai_model_safely(self):
        """Load AI model with safety checks"""
        # Add safety checks and validation here
        return {'model_loaded': True}

    def safe_control_loop(self):
        """Control loop with comprehensive error handling"""
        try:
            # Check for emergency conditions
            if self.emergency_stop:
                self.execute_emergency_stop()
                return

            # Check if AI model is available
            if self.ai_model is None:
                self.fallback_control()
                return

            # Normal AI-based control
            self.normal_control()

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            self.get_logger().info('Interrupt received, shutting down...')
            self.shutdown_procedure()
        except Exception as e:
            # Log the error
            error_msg = f'Control loop error: {e}\n{traceback.format_exc()}'
            self.get_logger().error(error_msg)
            self.publish_error(error_msg)

            # Fallback to safe behavior
            self.fallback_control()

    def normal_control(self):
        """Normal AI-based control"""
        # AI processing logic
        pass

    def fallback_control(self):
        """Fallback control when AI is unavailable"""
        self.get_logger().warn('Using fallback control mode')
        # Implement safe, simple control behavior
        pass

    def execute_emergency_stop(self):
        """Execute emergency stop procedure"""
        self.get_logger().error('EMERGENCY STOP EXECUTED')
        # Stop all robot movement
        pass

    def publish_status(self, message):
        """Safely publish status message"""
        try:
            status_msg = String()
            status_msg.data = message
            self.status_pub.publish(status_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish status: {e}')

    def publish_error(self, error_message):
        """Safely publish error message"""
        try:
            error_msg = String()
            error_msg.data = error_message
            self.error_pub.publish(error_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish error: {e}')

    def signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.get_logger().info(f'Received signal {signum}, initiating shutdown...')
        self.shutdown_procedure()
        sys.exit(0)

    def shutdown_procedure(self):
        """Graceful shutdown procedure"""
        self.get_logger().info('Executing shutdown procedure...')

        # Stop all robot motion
        self.execute_emergency_stop()

        # Clean up resources
        if self.ai_model:
            # Save model state if needed
            pass

        # Destroy node
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    try:
        ai_controller = RobustAIController()

        # Use SingleThreadedExecutor for simpler error handling
        executor = SingleThreadedExecutor()
        executor.add_node(ai_controller)

        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            ai_controller.destroy_node()

    except Exception as e:
        print(f'Fatal error in AI controller: {e}')
        traceback.print_exc()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Integrating AI agents with ROS 2 controllers using rclpy requires careful consideration of:
- Real-time performance requirements
- Memory management and optimization
- Error handling and robustness
- Proper threading models
- Safety and emergency procedures

For humanoid robots, this integration enables sophisticated behaviors that combine AI perception and reasoning with precise robotic control, making them capable of complex autonomous tasks.

## Exercises

1. Create an AI controller that integrates a simple machine learning model (using scikit-learn or TensorFlow) with ROS 2 for basic object recognition and navigation.

2. Implement a perception-action loop for a humanoid robot that processes camera and IMU data to maintain balance while walking.

3. Design a robust error handling system for an AI controller that includes fallback behaviors and graceful degradation.

## Next Steps

In the next chapter, we'll explore URDF (Unified Robot Description Format) - the standard way to describe robot models in ROS. This is crucial for humanoid robots as it defines their physical structure, kinematics, and visual properties.

Continue to Chapter 9: [URDF Basics for Humanoid Robots](./urdf-fundamentals.md) to learn about robot description and modeling.