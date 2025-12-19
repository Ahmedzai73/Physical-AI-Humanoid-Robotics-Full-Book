# Perception-Action Integration Simulation Steps

This guide provides step-by-step instructions for integrating perception models with manipulation actions in the Vision-Language-Action (VLA) system as covered in Module 4 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to connect perception models (object detection, segmentation) with manipulation actions for picking target objects, creating a complete perception-action pipeline for the VLA system.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later)
- Completed voice command and cognitive planning exercises
- Computer vision libraries (OpenCV, etc.)
- Understanding of robot manipulation concepts

## Simulation Environment Setup

1. Install perception and computer vision packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge
   pip3 install opencv-python numpy
   ```

2. Install deep learning packages if using neural networks:
   ```bash
   pip3 install torch torchvision
   ```

## Exercise 1: Create Object Detection Node

1. Create object detection node (`src/physical_ai_robotics/object_detector.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, CameraInfo
   from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
   from cv_bridge import CvBridge
   import cv2
   import numpy as np
   from std_msgs.msg import String

   class ObjectDetector(Node):
       def __init__(self):
           super().__init__('object_detector')

           # Initialize OpenCV bridge
           self.bridge = CvBridge()

           # Subscribers
           self.image_subscriber = self.create_subscription(
               Image, '/camera/color/image_raw', self.image_callback, 10
           )
           self.camera_info_subscriber = self.create_subscription(
               CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10
           )

           # Publishers
           self.detection_publisher = self.create_publisher(
               Detection2DArray, '/object_detections', 10
           )
           self.status_publisher = self.create_publisher(String, '/detection_status', 10)

           # Camera parameters (will be updated from camera_info)
           self.camera_matrix = None
           self.distortion_coeffs = None

           # Known objects to detect
           self.known_objects = [
               'cup', 'bottle', 'book', 'phone', 'keys', 'apple', 'banana'
           ]

           # Initialize detector (using a simple color-based approach for simulation)
           self.get_logger().info('Object Detector initialized')

       def camera_info_callback(self, msg):
           """Update camera parameters from camera info"""
           self.camera_matrix = np.array(msg.k).reshape(3, 3)
           self.distortion_coeffs = np.array(msg.d)

       def image_callback(self, msg):
           """Process incoming image and detect objects"""
           try:
               # Convert ROS Image to OpenCV image
               cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

               # Perform object detection
               detections = self.detect_objects(cv_image)

               # Create Detection2DArray message
               detection_array = Detection2DArray()
               detection_array.header = msg.header
               detection_array.detections = detections

               # Publish detections
               self.detection_publisher.publish(detection_array)

               # Publish status
               status_msg = String()
               status_msg.data = f'Detected {len(detections)} objects'
               self.status_publisher.publish(status_msg)

           except Exception as e:
               self.get_logger().error(f'Error processing image: {e}')

       def detect_objects(self, cv_image):
           """Detect objects in the image"""
           detections = []

           # For simulation purposes, we'll use a simple color-based detection
           # In a real system, you'd use a trained neural network

           # Convert BGR to HSV for better color detection
           hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

           # Define color ranges for different objects (for simulation)
           color_ranges = {
               'red_cup': ([0, 50, 50], [10, 255, 255]),
               'blue_bottle': ([100, 50, 50], [130, 255, 255]),
               'yellow_banana': ([20, 50, 50], [30, 255, 255]),
           }

           for obj_name, (lower, upper) in color_ranges.items():
               # Create mask for the color range
               mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

               # Find contours
               contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

               for contour in contours:
                   # Filter by size to avoid noise
                   area = cv2.contourArea(contour)
                   if area > 500:  # Minimum area threshold
                       # Get bounding box
                       x, y, w, h = cv2.boundingRect(contour)

                       # Create detection
                       detection = Detection2D()
                       detection.header.stamp = self.get_clock().now().to_msg()
                       detection.header.frame_id = 'camera_link'

                       # Set bounding box
                       detection.bbox.center.x = x + w / 2
                       detection.bbox.center.y = y + h / 2
                       detection.bbox.size_x = w
                       detection.bbox.size_y = h

                       # Set object hypothesis
                       hypothesis = ObjectHypothesisWithPose()
                       hypothesis.id = obj_name
                       hypothesis.score = 0.8  # Confidence score
                       detection.results.append(hypothesis)

                       detections.append(detection)

           return detections

   def main(args=None):
       rclpy.init(args=args)
       detector = ObjectDetector()
       rclpy.spin(detector)
       detector.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 2: Create 3D Object Pose Estimation

1. Create 3D pose estimation node (`src/physical_ai_robotics/pose_estimator.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, CameraInfo
   from vision_msgs.msg import Detection2DArray
   from geometry_msgs.msg import Point, PoseStamped
   from cv_bridge import CvBridge
   import numpy as np
   import cv2
   from std_msgs.msg import String

   class PoseEstimator(Node):
       def __init__(self):
           super().__init__('pose_estimator')

           # Initialize OpenCV bridge
           self.bridge = CvBridge()

           # Subscribers
           self.detection_subscriber = self.create_subscription(
               Detection2DArray, '/object_detections', self.detection_callback, 10
           )
           self.depth_subscriber = self.create_subscription(
               Image, '/camera/depth/image_raw', self.depth_callback, 10
           )
           self.camera_info_subscriber = self.create_subscription(
               CameraInfo, '/camera/depth/camera_info', self.camera_info_callback, 10
           )

           # Publishers
           self.pose_publisher = self.create_publisher(
               PoseStamped, '/object_pose', 10
           )
           self.status_publisher = self.create_publisher(String, '/pose_status', 10)

           # Camera parameters
           self.camera_matrix = None
           self.depth_image = None

           self.get_logger().info('Pose Estimator initialized')

       def camera_info_callback(self, msg):
           """Update camera parameters"""
           self.camera_matrix = np.array(msg.k).reshape(3, 3)

       def depth_callback(self, msg):
           """Store depth image"""
           try:
               self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
           except Exception as e:
               self.get_logger().error(f'Error converting depth image: {e}')

       def detection_callback(self, msg):
           """Process object detections and estimate 3D poses"""
           if self.depth_image is None or self.camera_matrix is None:
               return

           for detection in msg.detections:
               # Get object center in image coordinates
               center_x = int(detection.bbox.center.x)
               center_y = int(detection.bbox.center.y)

               # Get depth at object center
               if center_y < self.depth_image.shape[0] and center_x < self.depth_image.shape[1]:
                   depth = self.depth_image[center_y, center_x]

                   if depth > 0:  # Valid depth
                       # Convert pixel coordinates to 3D world coordinates
                       x = (center_x - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
                       y = (center_y - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
                       z = depth

                       # Create pose message
                       pose_msg = PoseStamped()
                       pose_msg.header.stamp = self.get_clock().now().to_msg()
                       pose_msg.header.frame_id = 'camera_link'
                       pose_msg.pose.position.x = x
                       pose_msg.pose.position.y = y
                       pose_msg.pose.position.z = z
                       pose_msg.pose.orientation.w = 1.0  # No rotation for now

                       # Publish pose
                       self.pose_publisher.publish(pose_msg)

                       # Publish status
                       status_msg = String()
                       status_msg.data = f'Estimated pose for {detection.results[0].id} at ({x:.2f}, {y:.2f}, {z:.2f})'
                       self.status_publisher.publish(status_msg)

   def main(args=None):
       rclpy.init(args=args)
       estimator = PoseEstimator()
       rclpy.spin(estimator)
       estimator.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 3: Create Manipulation Controller

1. Create manipulation controller node (`src/physical_ai_robotics/manipulation_controller.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped, Point
   from std_msgs.msg import String, Bool
   from sensor_msgs.msg import JointState
   import math

   class ManipulationController(Node):
       def __init__(self):
           super().__init__('manipulation_controller')

           # Subscribers
           self.object_pose_subscriber = self.create_subscription(
               PoseStamped, '/object_pose', self.object_pose_callback, 10
           )
           self.manipulation_command_subscriber = self.create_subscription(
               String, '/manipulation_command', self.manipulation_command_callback, 10
           )
           self.joint_state_subscriber = self.create_subscription(
               JointState, '/joint_states', self.joint_state_callback, 10
           )

           # Publishers
           self.target_pose_publisher = self.create_publisher(
               PoseStamped, '/manipulation_target', 10
           )
           self.status_publisher = self.create_publisher(String, '/manipulation_status', 10)
           self.success_publisher = self.create_publisher(Bool, '/manipulation_success', 10)

           # Robot arm parameters (for simulation)
           self.arm_base_position = Point(x=0.0, y=0.0, z=0.5)  # Base of the arm
           self.arm_length = 0.5  # Length of the arm (for simple IK)

           # Current state
           self.current_object_pose = None
           self.current_joint_states = JointState()

           self.get_logger().info('Manipulation Controller initialized')

       def joint_state_callback(self, msg):
           """Update current joint states"""
           self.current_joint_states = msg

       def object_pose_callback(self, msg):
           """Process object pose for manipulation"""
           self.current_object_pose = msg
           self.get_logger().info(f'Received object pose: ({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z})')

       def manipulation_command_callback(self, msg):
           """Process manipulation command"""
           command = msg.data.lower()
           self.get_logger().info(f'Received manipulation command: {command}')

           if command.startswith('pick') and self.current_object_pose:
               self.execute_pick_operation(self.current_object_pose)
           elif command.startswith('place') and self.current_object_pose:
               self.execute_place_operation(self.current_object_pose)
           elif command.startswith('move_to') and self.current_object_pose:
               self.execute_move_to_operation(self.current_object_pose)
           else:
               status_msg = String()
               status_msg.data = f'Unknown manipulation command: {command}'
               self.status_publisher.publish(status_msg)

       def execute_pick_operation(self, object_pose):
           """Execute pick operation"""
           self.get_logger().info('Executing pick operation')

           # Calculate approach pose (slightly above the object)
           approach_pose = PoseStamped()
           approach_pose.header = object_pose.header
           approach_pose.pose.position.x = object_pose.pose.position.x
           approach_pose.pose.position.y = object_pose.pose.position.y
           approach_pose.pose.position.z = object_pose.pose.position.z + 0.1  # 10cm above
           approach_pose.pose.orientation.w = 1.0

           # Move to approach position
           self.move_to_pose(approach_pose, 'approach')

           # Descend to object
           grasp_pose = PoseStamped()
           grasp_pose.header = object_pose.header
           grasp_pose.pose.position.x = object_pose.pose.position.x
           grasp_pose.pose.position.y = object_pose.pose.position.y
           grasp_pose.pose.position.z = object_pose.pose.position.z + 0.05  # 5cm above grasp point
           grasp_pose.pose.orientation.w = 1.0

           self.move_to_pose(grasp_pose, 'descend')

           # Simulate grasp
           self.simulate_grasp()

           # Lift object
           lift_pose = PoseStamped()
           lift_pose.header = object_pose.header
           lift_pose.pose.position.x = object_pose.pose.position.x
           lift_pose.pose.position.y = object_pose.pose.position.y
           lift_pose.pose.position.z = object_pose.pose.position.z + 0.2  # Lift 20cm
           lift_pose.pose.orientation.w = 1.0

           self.move_to_pose(lift_pose, 'lift')

           # Publish success
           success_msg = Bool()
           success_msg.data = True
           self.success_publisher.publish(success_msg)

           status_msg = String()
           status_msg.data = f'Pick operation completed for object at ({object_pose.pose.position.x:.2f}, {object_pose.pose.position.y:.2f}, {object_pose.pose.position.z:.2f})'
           self.status_publisher.publish(status_msg)

       def execute_place_operation(self, object_pose):
           """Execute place operation"""
           self.get_logger().info('Executing place operation')

           # For now, just move to the location and release
           place_pose = PoseStamped()
           place_pose.header = object_pose.header
           place_pose.pose.position.x = object_pose.pose.position.x
           place_pose.pose.position.y = object_pose.pose.position.y
           place_pose.pose.position.z = object_pose.pose.position.z + 0.1  # Place 10cm above surface
           place_pose.pose.orientation.w = 1.0

           self.move_to_pose(place_pose, 'place')

           # Simulate release
           self.simulate_release()

           # Publish success
           success_msg = Bool()
           success_msg.data = True
           self.success_publisher.publish(success_msg)

           status_msg = String()
           status_msg.data = f'Place operation completed'
           self.status_publisher.publish(status_msg)

       def execute_move_to_operation(self, target_pose):
           """Execute move to target position"""
           self.get_logger().info('Executing move to operation')

           # Move to target pose
           self.move_to_pose(target_pose, 'move_to')

           # Publish success
           success_msg = Bool()
           success_msg.data = True
           self.success_publisher.publish(success_msg)

           status_msg = String()
           status_msg.data = f'Move to operation completed'
           self.status_publisher.publish(status_msg)

       def move_to_pose(self, target_pose, operation_type):
           """Move arm to target pose (simulation)"""
           self.get_logger().info(f'Moving to {operation_type} pose: ({target_pose.pose.position.x:.2f}, {target_pose.pose.position.y:.2f}, {target_pose.pose.position.z:.2f})')

           # In a real system, this would call an IK solver and send joint commands
           # For simulation, we'll just publish the target
           self.target_pose_publisher.publish(target_pose)

           # Simulate movement time
           import time
           time.sleep(2)  # Simulate 2 seconds for movement

       def simulate_grasp(self):
           """Simulate grasp action"""
           self.get_logger().info('Simulating grasp action')
           # In a real system, this would close the gripper
           import time
           time.sleep(1)  # Simulate grasp time

       def simulate_release(self):
           """Simulate release action"""
           self.get_logger().info('Simulating release action')
           # In a real system, this would open the gripper
           import time
           time.sleep(1)  # Simulate release time

   def main(args=None):
       rclpy.init(args=args)
       controller = ManipulationController()
       rclpy.spin(controller)
       controller.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 4: Create Perception-Action Integration Node

1. Create the main integration node (`src/physical_ai_robotics/perception_action_integrator.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from vision_msgs.msg import Detection2DArray
   from geometry_msgs.msg import PoseStamped
   from std_msgs.msg import String
   import json

   class PerceptionActionIntegrator(Node):
       def __init__(self):
           super().__init__('perception_action_integrator')

           # Subscribers
           self.detection_subscriber = self.create_subscription(
               Detection2DArray, '/object_detections', self.detection_callback, 10
           )
           self.cognitive_plan_subscriber = self.create_subscription(
               String, '/cognitive_plan', self.cognitive_plan_callback, 10
           )

           # Publishers
           self.manipulation_command_publisher = self.create_publisher(
               String, '/manipulation_command', 10
           )
           self.navigation_goal_publisher = self.create_publisher(
               PoseStamped, '/navigation_goal', 10
           )
           self.status_publisher = self.create_publisher(String, '/integration_status', 10)

           # Current state
           self.current_detections = Detection2DArray()
           self.current_plan = None

           self.get_logger().info('Perception-Action Integrator initialized')

       def detection_callback(self, msg):
           """Update current detections"""
           self.current_detections = msg
           self.get_logger().debug(f'Updated detections: {len(msg.detections)} objects')

       def cognitive_plan_callback(self, msg):
           """Process cognitive plan and integrate with perception"""
           try:
               plan_data = json.loads(msg.data)
               self.current_plan = plan_data

               self.get_logger().info(f'Received cognitive plan: {plan_data}')

               # Integrate plan with current perception
               self.integrate_plan_with_perception(plan_data)

           except json.JSONDecodeError:
               self.get_logger().error(f'Invalid JSON in cognitive plan: {msg.data}')

       def integrate_plan_with_perception(self, plan):
           """Integrate cognitive plan with current perception data"""
           for action in plan.get('actions', []):
               action_type = action.get('action_type', '')
               target_object = action.get('target_object', '')
               target_location = action.get('target_location', '')

               if action_type == 'perception' and target_object:
                   self.execute_perception_action(target_object)
               elif action_type == 'manipulation' and target_object:
                   self.execute_manipulation_action(target_object, action.get('action', ''))
               elif action_type == 'navigation' and target_location:
                   self.execute_navigation_action(target_location)

       def execute_perception_action(self, target_object):
           """Execute perception action to find specific object"""
           self.get_logger().info(f'Executing perception action for: {target_object}')

           # Check if target object is currently detected
           target_detected = False
           for detection in self.current_detections.detections:
               if target_object.lower() in detection.results[0].id.lower():
                   target_detected = True
                   self.get_logger().info(f'Found {target_object} at position in camera frame')

                   # Publish object pose for manipulation
                   pose_msg = PoseStamped()
                   pose_msg.header = self.current_detections.header
                   pose_msg.pose.position.x = detection.bbox.center.x
                   pose_msg.pose.position.y = detection.bbox.center.y
                   # Depth would come from depth estimation node
                   pose_msg.pose.position.z = 1.0  # Placeholder depth
                   pose_msg.pose.orientation.w = 1.0

                   # In a real system, we'd use the 3D pose from pose estimator

                   break

           if not target_detected:
               status_msg = String()
               status_msg.data = f'Could not find {target_object} in current view. Need to navigate or search.'
               self.status_publisher.publish(status_msg)

       def execute_manipulation_action(self, target_object, manipulation_type):
           """Execute manipulation action with perception feedback"""
           self.get_logger().info(f'Executing {manipulation_type} action for: {target_object}')

           # First, ensure the object is detected
           object_pose = self.get_object_pose(target_object)

           if object_pose:
               # Publish manipulation command
               command_msg = String()
               command_msg.data = f'{manipulation_type} {target_object}'
               self.manipulation_command_publisher.publish(command_msg)

               status_msg = String()
               status_msg.data = f'Sent {manipulation_type} command for {target_object}'
               self.status_publisher.publish(status_msg)
           else:
               status_msg = String()
               status_msg.data = f'Cannot {manipulation_type} {target_object} - not detected'
               self.status_publisher.publish(status_msg)

       def execute_navigation_action(self, target_location):
           """Execute navigation action based on location"""
           self.get_logger().info(f'Executing navigation to: {target_location}')

           # In a real system, this would use a map to get the actual coordinates
           # For simulation, we'll use predefined locations

           # Define location coordinates (in simulation map frame)
           location_coords = {
               'kitchen': {'x': 2.0, 'y': 1.0, 'theta': 0.0},
               'bedroom': {'x': -1.0, 'y': 3.0, 'theta': 1.57},
               'living_room': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
               'office': {'x': 3.0, 'y': -2.0, 'theta': 3.14},
               'dining_room': {'x': 1.5, 'y': 2.5, 'theta': -1.57}
           }

           if target_location in location_coords:
               coords = location_coords[target_location]
               goal_pose = PoseStamped()
               goal_pose.header.frame_id = 'map'
               goal_pose.header.stamp = self.get_clock().now().to_msg()
               goal_pose.pose.position.x = coords['x']
               goal_pose.pose.position.y = coords['y']
               goal_pose.pose.position.z = 0.0

               # Convert theta to quaternion
               import math
               theta = coords['theta']
               goal_pose.pose.orientation.z = math.sin(theta / 2.0)
               goal_pose.pose.orientation.w = math.cos(theta / 2.0)

               # Publish navigation goal
               self.navigation_goal_publisher.publish(goal_pose)

               status_msg = String()
               status_msg.data = f'Navigating to {target_location} at ({coords["x"]}, {coords["y"]})'
               self.status_publisher.publish(status_msg)
           else:
               status_msg = String()
               status_msg.data = f'Unknown location: {target_location}'
               self.status_publisher.publish(status_msg)

       def get_object_pose(self, target_object):
           """Get 3D pose of target object from detections"""
           for detection in self.current_detections.detections:
               if target_object.lower() in detection.results[0].id.lower():
                   # Return the detection info (in a real system, we'd use the 3D pose)
                   return detection
           return None

   def main(args=None):
       rclpy.init(args=args)
       integrator = PerceptionActionIntegrator()
       rclpy.spin(integrator)
       integrator.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 5: Create Integration Launch File

1. Create launch file for perception-action integration (`launch/perception_action_integration.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='false')

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='false',
               description='Use simulation clock if true'
           ),

           # Object detector
           Node(
               package='physical_ai_robotics',
               executable='object_detector',
               name='object_detector',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Pose estimator
           Node(
               package='physical_ai_robotics',
               executable='pose_estimator',
               name='pose_estimator',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Manipulation controller
           Node(
               package='physical_ai_robotics',
               executable='manipulation_controller',
               name='manipulation_controller',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Perception-action integrator
           Node(
               package='physical_ai_robotics',
               executable='perception_action_integrator',
               name='perception_action_integrator',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           )
       ])
   ```

## Exercise 6: Test Perception-Action Pipeline

1. Launch the perception-action integration system:
   ```bash
   ros2 launch physical_ai_robotics perception_action_integration.launch.py
   ```

2. Test with various commands:
   ```bash
   # Test object detection
   ros2 topic echo /object_detections

   # Test manipulation
   ros2 topic pub /manipulation_command std_msgs/String "data: 'pick cup'"

   # Test integration
   ros2 topic pub /cognitive_plan std_msgs/String "data: '{\"actions\": [{\"action_type\": \"perception\", \"target_object\": \"cup\"}, {\"action_type\": \"manipulation\", \"target_object\": \"cup\", \"action\": \"pick\"}]}"
   ```

3. Monitor the system:
   ```bash
   # Monitor integration status
   ros2 topic echo /integration_status

   # Monitor detection status
   ros2 topic echo /detection_status

   # Monitor manipulation status
   ros2 topic echo /manipulation_status
   ```

## Exercise 7: Implement Object Recognition with Deep Learning

1. Create a more advanced object recognition node using neural networks:
   ```python
   # Add to object_detector.py or create new file
   import torch
   import torchvision.transforms as transforms
   from torchvision.models import detection

   class DeepObjectDetector(Node):
       def __init__(self):
           super().__init__('deep_object_detector')

           # Load pre-trained model (e.g., MobileNet SSD)
           self.model = detection.retinanet_resnet50_fpn(pretrained=True)
           self.model.eval()

           # Other initialization code...
   ```

## Exercise 8: Add Grasp Planning

1. Create grasp planning node:
   ```python
   class GraspPlanner(Node):
       def __init__(self):
           super().__init__('grasp_planner')

           # Subscribe to object poses
           # Plan optimal grasp points
           # Publish grasp poses
           pass
   ```

## Exercise 9: Implement Visual Servoing

1. Add visual servoing capabilities for precise manipulation:
   ```python
   # Add to manipulation controller
   def execute_visual_servoing(self, target_object):
       """Execute visual servoing to precisely approach target"""
       # Use visual feedback to adjust approach
       # Continue until object is in optimal position for grasp
       pass
   ```

## Exercise 10: Add Multi-Object Handling

1. Implement multi-object scene understanding:
   ```python
   def process_multi_object_scene(self, detections):
       """Process scene with multiple objects"""
       # Determine object relationships
       # Plan manipulation sequence
       # Handle object interactions
       pass
   ```

## Exercise 11: Integrate with Navigation

1. Connect perception-action with navigation system:
   ```python
   # Add navigation integration to perception_action_integrator
   def execute_search_pattern(self, target_object):
       """Execute search pattern to find target object"""
       # Navigate through predefined locations
       # Use perception to detect object
       # Return location when found
       pass
   ```

## Exercise 12: Performance Optimization

1. Optimize perception pipeline for real-time performance:
   ```python
   # Add performance monitoring
   def monitor_performance(self):
       """Monitor perception pipeline performance"""
       # Track detection rates
       # Monitor processing time
       # Adjust parameters for optimal performance
       pass
   ```

## Verification Steps

1. Confirm object detection works with camera input
2. Verify 3D pose estimation from depth data
3. Check manipulation commands are executed properly
4. Validate perception-action integration pipeline
5. Ensure system operates in real-time

## Expected Outcomes

- Understanding of perception-action pipeline integration
- Knowledge of object detection and pose estimation
- Experience with manipulation planning
- Ability to create complete perception-action systems

## Troubleshooting

- If object detection fails, check camera calibration and lighting
- If poses are inaccurate, verify depth data and camera parameters
- If manipulation doesn't work, check robot arm calibration

## Next Steps

After completing these exercises, proceed to the complete VLA integration to build a full Voice → Plan → Navigate → Perceive → Act pipeline.