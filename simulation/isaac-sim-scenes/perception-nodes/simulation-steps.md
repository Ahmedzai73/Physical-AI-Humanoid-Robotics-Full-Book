# Isaac ROS Perception Nodes Simulation Steps

This guide provides step-by-step instructions for implementing Isaac ROS perception nodes including AprilTags, stereo vision, and depth processing as covered in Module 3 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to configure and use Isaac ROS perception nodes for various computer vision tasks including AprilTag detection, stereo vision processing, and depth-based perception in robotics applications.

## Prerequisites

- Isaac Sim with robot and sensors configured
- Isaac ROS packages installed
- Completed VSLAM pipeline exercises
- RGB cameras configured in Isaac Sim

## Simulation Environment Setup

1. Ensure Isaac ROS perception packages are installed:
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/isaac_ros_ws/install/setup.bash
   ```

## Exercise 1: Install Isaac ROS Perception Packages

1. Install required perception packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-isaac-ros-apriltag
   sudo apt install ros-humble-isaac-ros-stereo-image-pipeline
   sudo apt install ros-humble-isaac-ros-rectify
   sudo apt install ros-humble-isaac-ros-stereo-disparity
   sudo apt install ros-humble-isaac-ros-dnn-image-encoder
   sudo apt install ros-humble-isaac-ros-dnn-ros
   ```

2. Or build from source:
   ```bash
   cd ~/isaac_ros_ws/src
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_stereo_image_pipeline.git
   git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git
   ```

3. Build perception packages:
   ```bash
   cd ~/isaac_ros_ws
   colcon build --packages-select isaac_ros_apriltag isaac_ros_stereo_image_pipeline isaac_ros_dnn_inference
   source install/setup.bash
   ```

## Exercise 2: Configure AprilTag Detection

1. Create AprilTag detection launch file (`launch/apriltag_detection.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import ComposableNodeContainer
   from launch_ros.descriptions import ComposableNode

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')

       apriltag_node = ComposableNodeContainer(
           name='apriltag_container',
           namespace='',
           package='rclcpp_components',
           executable='component_container_mt',
           composable_node_descriptions=[
               ComposableNode(
                   package='isaac_ros_apriltag',
                   plugin='nvidia::isaac_ros::apriltag::AprilTagNode',
                   name='apriltag',
                   parameters=[{
                       'use_sim_time': use_sim_time,
                       'family': 'tag36h11',
                       'size': 0.1,  # Tag size in meters
                       'max_tags': 10,
                       'tag_layout': 'grid',
                       'tag_rows': 2,
                       'tag_cols': 2,
                       'tag_spacing': 0.1,  # Spacing between tags (meters)
                   }],
                   remappings=[
                       ('image', '/camera/color/image_raw'),
                       ('camera_info', '/camera/color/camera_info'),
                       ('detections', '/apriltag_detections'),
                   ],
               )
           ],
           output='screen'
       )

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),
           apriltag_node,
       ])
   ```

2. Launch AprilTag detection:
   ```bash
   ros2 launch your_package apriltag_detection.launch.py
   ```

3. Test AprilTag detection:
   ```bash
   # Monitor AprilTag detections
   ros2 topic echo /apriltag_detections
   ```

## Exercise 3: Set Up Stereo Disparity Processing

1. Configure stereo image rectification:
   ```bash
   # Launch stereo rectification if using stereo cameras
   ros2 launch isaac_ros_stereo_image_pipeline stereo_rectify.launch.py
   ```

2. Set up disparity processing pipeline:
   ```python
   # Create launch file for stereo disparity
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import ComposableNodeContainer
   from launch_ros.descriptions import ComposableNode

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')

       stereo_node = ComposableNodeContainer(
           name='stereo_container',
           namespace='',
           package='rclcpp_components',
           executable='component_container_mt',
           composable_node_descriptions=[
               ComposableNode(
                   package='isaac_ros_stereo_image_proc',
                   plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
                   name='disparity',
                   parameters=[{
                       'use_sim_time': use_sim_time,
                       'min_disparity': 0.0,
                       'max_disparity': 64.0,
                       'delta_d': 1.0,
                   }],
                   remappings=[
                       ('left/image_rect', '/stereo_left/image_rect_color'),
                       ('left/camera_info', '/stereo_left/camera_info'),
                       ('right/image_rect', '/stereo_right/image_rect'),
                       ('right/camera_info', '/stereo_right/camera_info'),
                       ('disparity', '/stereo/disparity'),
                   ],
               )
           ],
           output='screen'
       )

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),
           stereo_node,
       ])
   ```

3. Test stereo disparity:
   ```bash
   # Monitor disparity output
   ros2 topic echo /stereo/disparity
   ```

## Exercise 4: Configure Depth-based Perception

1. Set up depth processing pipeline:
   ```python
   # Create depth processing launch file
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import ComposableNodeContainer
   from launch_ros.descriptions import ComposableNode

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')

       depth_node = ComposableNodeContainer(
           name='depth_container',
           namespace='',
           package='rclcpp_components',
           executable='component_container_mt',
           composable_node_descriptions=[
               ComposableNode(
                   package='isaac_ros_depth_image_proc',
                   plugin='nvidia::isaac_ros::depth_image_proc::PointCloudNode',
                   name='pointcloud',
                   parameters=[{
                       'use_sim_time': use_sim_time,
                       'input_encoding': '16UC1',  # or '32FC1' depending on depth format
                   }],
                   remappings=[
                       ('image', '/camera/depth/image_raw'),
                       ('camera_info', '/camera/depth/camera_info'),
                       ('points', '/camera/depth/points'),
                   ],
               )
           ],
           output='screen'
       )

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),
           depth_node,
       ])
   ```

2. Launch depth processing:
   ```bash
   ros2 launch your_package depth_processing.launch.py
   ```

3. Test point cloud generation:
   ```bash
   # Monitor point cloud output
   ros2 topic echo /camera/depth/points
   ```

## Exercise 5: Integrate with Isaac Sim Sensors

1. Configure Isaac Sim to publish appropriate sensor data:
   - Ensure RGB camera publishes to `/camera/color/image_raw`
   - Verify depth camera publishes to `/camera/depth/image_raw`
   - Check camera info topics are available

2. Create Isaac Sim sensor configuration:
   ```python
   # Example script to configure Isaac Sim sensors via Python API
   import omni
   from pxr import Gf

   # This would be run within Isaac Sim Python console
   # Configure camera properties for optimal perception
   ```

3. Validate sensor data quality:
   - Check image resolution and format
   - Verify depth accuracy
   - Ensure proper calibration

## Exercise 6: Set Up DNN-based Object Detection

1. Configure Isaac ROS DNN inference:
   ```python
   # Create DNN object detection launch file
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import ComposableNodeContainer
   from launch_ros.descriptions import ComposableNode

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')

       dnn_container = ComposableNodeContainer(
           name='dnn_container',
           namespace='',
           package='rclcpp_components',
           executable='component_container_mt',
           composable_node_descriptions=[
               ComposableNode(
                   package='isaac_ros_dnn_image_encoder',
                   plugin='nvidia::isaac_ros::dnn_inference::ImageEncoderNode',
                   name='image_encoder',
                   parameters=[{
                       'use_sim_time': use_sim_time,
                       'input_tensor_names': ['input_tensor'],
                       'output_tensor_names': ['output_tensor'],
                       'network_output_type': 'detection',
                       'tensorrt_engine_file': '/path/to/tensorrt/engine.plan',
                       'model_input_width': 640,
                       'model_input_height': 480,
                       'max_batch_size': 1,
                       'input_tensor_layout': 'NHWC',
                       'mean': [0.0, 0.0, 0.0],
                       'stddev': [1.0, 1.0, 1.0],
                   }],
                   remappings=[
                       ('image', '/camera/color/image_raw'),
                       ('tensor', '/tensor_sub'),
                   ],
               )
           ],
           output='screen'
       )

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),
           dnn_container,
       ])
   ```

2. Launch DNN inference:
   ```bash
   ros2 launch your_package dnn_detection.launch.py
   ```

## Exercise 7: Create Multi-Modal Perception Pipeline

1. Combine multiple perception nodes:
   ```python
   # Create comprehensive perception pipeline launch file
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import ComposableNodeContainer
   from launch_ros.descriptions import ComposableNode

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')

       perception_container = ComposableNodeContainer(
           name='perception_container',
           namespace='',
           package='rclcpp_components',
           executable='component_container_mt',
           composable_node_descriptions=[
               # AprilTag detection
               ComposableNode(
                   package='isaac_ros_apriltag',
                   plugin='nvidia::isaac_ros::apriltag::AprilTagNode',
                   name='apriltag',
                   parameters=[{'use_sim_time': use_sim_time, 'size': 0.1}],
                   remappings=[
                       ('image', '/camera/color/image_raw'),
                       ('camera_info', '/camera/color/camera_info'),
                   ],
               ),
               # Depth processing
               ComposableNode(
                   package='isaac_ros_depth_image_proc',
                   plugin='nvidia::isaac_ros::depth_image_proc::PointCloudNode',
                   name='pointcloud',
                   parameters=[{'use_sim_time': use_sim_time}],
                   remappings=[
                       ('image', '/camera/depth/image_raw'),
                       ('camera_info', '/camera/depth/camera_info'),
                   ],
               ),
           ],
           output='screen'
       )

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),
           perception_container,
       ])
   ```

## Exercise 8: Test Perception in Isaac Sim Environment

1. Create Isaac Sim scene with perception targets:
   - Place AprilTags in the environment
   - Set up objects for detection
   - Configure lighting for optimal perception

2. Move robot through environment:
   - Navigate to different areas
   - Test perception at various distances
   - Validate detection accuracy

3. Monitor perception performance:
   ```bash
   # Monitor all perception outputs
   ros2 topic list | grep -E "(apriltag|detection|pointcloud)"
   ```

## Exercise 9: Configure Perception Parameters

1. Tune AprilTag parameters:
   - Adjust tag size for your specific tags
   - Configure detection thresholds
   - Set up multiple tag families if needed

2. Optimize depth processing:
   - Configure depth range for your application
   - Adjust point cloud density
   - Set up coordinate frame transformations

3. Validate parameter effectiveness:
   - Test with various object distances
   - Check detection rates
   - Monitor computational performance

## Exercise 10: Integrate with Robot Control

1. Create perception-to-control pipeline:
   ```python
   # Example perception-based control node
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, PointCloud2
   from geometry_msgs.msg import PoseStamped
   from vision_msgs.msg import Detection2DArray

   class PerceptionController(Node):
       def __init__(self):
           super().__init__('perception_controller')

           # Subscribers for perception outputs
           self.detection_subscriber = self.create_subscription(
               Detection2DArray, '/apriltag_detections', self.detection_callback, 10
           )
           self.pointcloud_subscriber = self.create_subscription(
               PointCloud2, '/camera/depth/points', self.pointcloud_callback, 10
           )

           # Publisher for robot commands
           self.cmd_publisher = self.create_publisher(PoseStamped, '/navigation_goal', 10)

           self.get_logger().info('Perception Controller initialized')

       def detection_callback(self, msg):
           """Process AprilTag detections"""
           if len(msg.detections) > 0:
               # Process the first detected tag
               detection = msg.detections[0]
               # Convert to navigation goal
               goal = PoseStamped()
               # ... process detection to navigation command
               self.cmd_publisher.publish(goal)

       def pointcloud_callback(self, msg):
           """Process point cloud data"""
           # Process for obstacle detection, etc.
           pass

   def main(args=None):
       rclpy.init(args=args)
       controller = PerceptionController()
       rclpy.spin(controller)
       controller.destroy_node()
       rclpy.shutdown()
   ```

## Exercise 11: Performance Optimization

1. Optimize perception pipeline:
   - Adjust processing frequencies
   - Configure resource allocation
   - Set up multi-threading appropriately

2. Monitor computational requirements:
   - Track CPU/GPU utilization
   - Monitor memory usage
   - Measure processing latency

3. Balance accuracy vs. performance:
   - Adjust detection thresholds
   - Configure processing quality settings
   - Optimize for target platform

## Exercise 12: Validation and Testing

1. Test perception accuracy:
   - Validate detection rates
   - Check measurement accuracy
   - Verify robustness to environmental changes

2. Performance benchmarking:
   - Measure frames per second
   - Validate real-time performance
   - Test under various conditions

## Verification Steps

1. Confirm all perception nodes launch correctly
2. Verify sensor data flows to perception nodes
3. Check that perception outputs are generated
4. Validate detection accuracy and reliability
5. Ensure computational performance meets requirements

## Expected Outcomes

- Understanding of Isaac ROS perception nodes
- Knowledge of multi-modal perception integration
- Experience with DNN-based detection
- Ability to create perception pipelines

## Troubleshooting

- If AprilTags aren't detected, check tag size and lighting
- If depth processing fails, verify depth format and calibration
- If DNN inference is slow, check TensorRT engine and hardware

## Next Steps

After completing these exercises, proceed to Nav2 integration to connect perception with navigation capabilities.