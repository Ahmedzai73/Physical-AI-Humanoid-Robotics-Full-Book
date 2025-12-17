---
title: Visualizing Humanoid Robots in RViz
sidebar_position: 10
description: Understanding RViz for visualizing robot models, sensors, and states in ROS 2
---

# Building a Full Humanoid URDF + Visualization in RViz

## Introduction

In the previous chapter, we learned about URDF (Unified Robot Description Format) and how to create robot models. Now we'll explore **RViz**, the 3D visualization tool for ROS, which is essential for visualizing our humanoid robot models and understanding their behavior. RViz allows us to see the robot's structure, sensor data, planned paths, and system states in an intuitive 3D environment.

For humanoid robots with complex kinematic structures and numerous sensors, RViz is invaluable for:
- Debugging robot models and kinematic chains
- Visualizing sensor data (cameras, LiDAR, IMU)
- Monitoring robot state and control performance
- Planning and navigation visualization
- Debugging complex multi-joint systems

## Understanding RViz Architecture

RViz is a visualization tool that displays information about a robot and its environment. It works by:

1. **Subscribing to ROS topics** to get data (sensor readings, transforms, etc.)
2. **Using TF transforms** to properly position visualized elements
3. **Rendering 3D graphics** to display the robot and environment
4. **Providing interactive tools** for user interaction with the visualization

### Key RViz Components

- **Displays**: Visual representations of ROS data (robot model, sensor data, paths)
- **Tools**: Interactive tools (select, move camera, publish points)
- **Views**: Different camera perspectives
- **Panels**: Property editors and other UI elements
- **TF Tree**: Coordinate frame relationships

## Setting up RViz for Humanoid Robots

### 1. Basic RViz Configuration

First, let's create a basic RViz configuration file for our humanoid robot:

```yaml
# config/humanoid.rviz
Panels:
  - Class: rviz_common/Displays
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
        - /TF1
        - /LaserScan1
        - /PointCloud21
      Splitter Ratio: 0.5
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        {}
      Update Interval: 0
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
      Line color: 128; 128; 0
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 10
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.5
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.5
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002f4fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000002f4000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000002f4fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073000000003d000002f4000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1200
  X: 72
  Y: 60
```

### 2. Launching RViz with Robot Model

To properly visualize your humanoid robot in RViz, you need to launch several nodes:

```python
# launch/rviz_visualization.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    declare_model_path = DeclareLaunchArgument(
        'model',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_description'),
            'urdf',
            'humanoid.urdf.xacro'
        ]),
        description='Path to robot URDF file'
    )

    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_description'),
            'rviz',
            'humanoid.rviz'
        ]),
        description='Path to RViz configuration file'
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['xacro ', LaunchConfiguration('model')])
        }],
        output='screen'
    )

    # Joint State Publisher (for visualization)
    joint_state_publisher = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )

    # RViz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen'
    )

    return LaunchDescription([
        declare_model_path,
        declare_rviz_config,
        robot_state_publisher,
        joint_state_publisher,
        rviz
    ])
```

## RobotModel Display

The RobotModel display is the primary way to visualize your URDF in RViz:

### 1. Basic RobotModel Configuration

```yaml
- Alpha: 1
  Class: rviz_default_plugins/RobotModel
  Collision Enabled: false  # Show collision geometry instead of visual
  Description File: ""      # Path to URDF file (if not using topic)
  Description Source: Topic  # How to get robot description
  Description Topic: /robot_description  # Topic to get URDF from
  Enabled: true
  Links:
    All Links Enabled: true
    Expand Joint Details: false
    Expand Link Details: false
    Expand Tree: false
    Link Tree Style: Links in Alphabetic Order
  Name: RobotModel
  TF Prefix: ""  # Prefix for TF frames
  Update Interval: 0  # 0 = continuous updates
  Value: true
  Visual Enabled: true  # Show visual geometry
```

### 2. Customizing RobotModel Appearance

You can customize the appearance of different links:

```python
# rviz_custom_model.py
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class RobotModelCustomizer(Node):
    def __init__(self):
        super().__init__('robot_model_customizer')

        # Publisher for custom visualization markers
        self.marker_pub = self.create_publisher(Marker, 'custom_markers', 10)

        # Timer to publish markers
        self.timer = self.create_timer(0.1, self.publish_custom_markers)

        self.get_logger().info('Robot Model Customizer initialized')

    def publish_custom_markers(self):
        """Publish custom visualization markers"""
        # Example: Highlight important joints
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "joints"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD

        # Set scale
        marker.scale.x = 0.05  # Sphere diameter
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        # Set color (red)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Alpha (opacity)

        # Add points for important joints
        # This is an example - you'd get actual joint positions from TF
        point = Point()
        point.x = 0.0
        point.y = 0.1  # Left hip position
        point.z = 0.0
        marker.points.append(point)

        point2 = Point()
        point2.x = 0.0
        point2.y = -0.1  # Right hip position
        point2.z = 0.0
        marker.points.append(point2)

        self.marker_pub.publish(marker)
```

## TF (Transforms) Visualization

TF trees are crucial for understanding the spatial relationships between robot parts:

### 1. Understanding TF Frames

For a humanoid robot, typical TF frames include:
- `base_link`: Robot's main reference frame
- `left_leg/hip`: Left leg hip joint frame
- `left_leg/knee`: Left leg knee joint frame
- `left_leg/foot`: Left leg foot frame
- `right_leg/hip`: Right leg hip joint frame
- `torso`: Torso frame
- `head`: Head frame
- `left_arm/shoulder`: Left arm shoulder frame
- `left_arm/elbow`: Left arm elbow frame
- `left_arm/wrist`: Left arm wrist frame

### 2. TF Tree Analysis

```python
# tf_analyzer.py
import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
import tf_transformations

class TFRvizAnalyzer(Node):
    def __init__(self):
        super().__init__('tf_rviz_analyzer')

        # Create TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically analyze TF tree
        self.timer = self.create_timer(1.0, self.analyze_tf_tree)

        self.get_logger().info('TF Analyzer initialized')

    def analyze_tf_tree(self):
        """Analyze the current TF tree structure"""
        try:
            # Get all available frames
            frame_names = self.tf_buffer.all_frames_as_string()
            self.get_logger().info(f'Available frames:\n{frame_names}')

            # Try to get a specific transform
            transform = self.tf_buffer.lookup_transform(
                'base_link',  # Target frame
                'head',       # Source frame
                rclpy.time.Time(),  # Time (0 = latest)
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Log transform information
            t = transform.transform.translation
            r = transform.transform.rotation

            self.get_logger().info(
                f'Transform from base_link to head: '
                f'({t.x:.3f}, {t.y:.3f}, {t.z:.3f}), '
                f'Rotation: ({r.x:.3f}, {r.y:.3f}, {r.z:.3f}, {r.w:.3f})'
            )

        except TransformException as ex:
            self.get_logger().warn(f'Could not transform: {ex}')

def main(args=None):
    rclpy.init(args=args)
    analyzer = TFRvizAnalyzer()

    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()
```

## Sensor Data Visualization

Humanoid robots have many sensors that need visualization in RViz:

### 1. LaserScan Visualization

```yaml
- Angle Max: 3.141599864959717
  Angle Min: -3.141599864959717
  Class: rviz_default_plugins/LaserScan
  Color: 255; 255; 255
  Color Transformer: Intensity
  Decay Time: 0
  Enabled: true
  Invert Rainbow: false
  Max Color: 255; 255; 255
  Max Intensity: 4096
  Min Color: 0; 0; 0
  Min Intensity: 0
  Name: LaserScan
  Position Transformer: XYZ
  Queue Size: 10
  Selectable: true
  Size (Pixels): 3
  Size (m): 0.009999999776482582
  Style: Flat Squares
  Topic:
    Depth: 5
    Durability Policy: Volatile
    Filter size: 10
    History Policy: Keep Last
    Reliability Policy: Best Effort
    Value: /scan
  Use rainbow: true
  Value: true
```

### 2. PointCloud2 Visualization

```yaml
- Alpha: 1
  Autocompute Intensity Bounds: true
  Autocompute Value Bounds:
    Max Value: 10
    Min Value: -10
    Value: true
  Axis: Z
  Channel Name: intensity
  Class: rviz_default_plugins/PointCloud2
  Color: 255; 255; 255
  Color Transformer: RGB8
  Decay Time: 0
  Enabled: true
  Invert Rainbow: false
  Max Color: 255; 255; 255
  Max Intensity: 4096
  Min Color: 0; 0; 0
  Min Intensity: 0
  Name: PointCloud2
  Position Transformer: XYZ
  Queue Size: 10
  Selectable: true
  Size (Pixels): 3
  Size (m): 0.009999999776482582
  Style: Flat Squares
  Topic:
    Depth: 5
    Durability Policy: Volatile
    Filter size: 10
    History Policy: Keep Last
    Reliability Policy: Best Effort
    Value: /camera/depth/color/points
  Use rainbow: true
  Value: true
```

### 3. Camera Image Visualization

```yaml
- Class: rviz_default_plugins/Image
  Enabled: true
  Max Value: 1
  Min Value: 0
  Name: Image
  Normalize Range: true
  Topic:
    Depth: 5
    Durability Policy: Volatile
    History Policy: Keep Last
    Reliability Policy: Best Effort
    Value: /camera/image_raw
  Value: true
```

## Advanced RViz Displays for Humanoid Robots

### 1. Robot Path Planning Visualization

```python
# path_visualizer.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
import math

class PathVisualizer(Node):
    def __init__(self):
        super().__init__('path_visualizer')

        # Publisher for path
        self.path_pub = self.create_publisher(Path, 'robot_path', 10)

        # Publisher for path markers
        self.marker_pub = self.create_publisher(Marker, 'path_markers', 10)

        # Timer to publish example path
        self.timer = self.create_timer(2.0, self.publish_example_path)

        self.get_logger().info('Path Visualizer initialized')

    def publish_example_path(self):
        """Publish an example path for visualization"""
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        # Create a circular path
        for i in range(20):
            angle = i * 2 * math.pi / 20
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = 2.0 * math.cos(angle)
            pose.pose.position.y = 2.0 * math.sin(angle)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

        # Also publish as a marker for more control
        self.publish_path_marker(path_msg)

    def publish_path_marker(self, path_msg):
        """Publish path as a marker for custom visualization"""
        marker = Marker()
        marker.header = path_msg.header
        marker.ns = "robot_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Set scale (line width)
        marker.scale.x = 0.05

        # Set color (blue)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Add points from path
        for pose_stamped in path_msg.poses:
            marker.points.append(pose_stamped.pose.position)

        self.marker_pub.publish(marker)
```

### 2. Joint Trajectory Visualization

```python
# trajectory_visualizer.py
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

class TrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer')

        # Publisher for joint trajectories
        self.traj_pub = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)

        # Timer to publish example trajectory
        self.timer = self.create_timer(5.0, self.publish_example_trajectory)

        self.get_logger().info('Trajectory Visualizer initialized')

    def publish_example_trajectory(self):
        """Publish an example joint trajectory"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        # Create trajectory points
        for i in range(10):
            point = JointTrajectoryPoint()

            # Create oscillating joint positions
            time_from_start = Duration()
            time_from_start.sec = i
            time_from_start.nanosec = 0
            point.time_from_start = time_from_start

            positions = []
            for j, joint_name in enumerate(traj_msg.joint_names):
                # Create different oscillation patterns for each joint
                pos = math.sin(i * 0.5 + j) * 0.5
                positions.append(pos)

            point.positions = positions
            traj_msg.points.append(point)

        self.traj_pub.publish(traj_msg)
        self.get_logger().info(f'Published trajectory with {len(traj_msg.points)} points')
```

## Interactive Tools in RViz

RViz provides several interactive tools for working with humanoid robots:

### 1. 2D Pose Estimate
Used to set the robot's initial position in localization systems.

### 2. 2D Nav Goal
Used to send navigation goals to the robot.

### 3. Publish Point
Used to publish points in the environment (for mapping).

### 4. Interactive Markers
For direct manipulation of robot poses:

```python
# interactive_marker_server.py
import rclpy
from rclpy.node import Node
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import ColorRGBA

class InteractiveMarkerController(Node):
    def __init__(self):
        super().__init__('interactive_marker_controller')

        # Create interactive marker server
        self.server = InteractiveMarkerServer(self, 'humanoid_control')
        self.menu_handler = MenuHandler()

        # Create menu options
        self.menu_handler.insert("Reset to Zero", callback=self.reset_cb)
        self.menu_handler.insert("Move to Home", callback=self.home_cb)

        # Create interactive marker
        self.create_interactive_marker()

        self.get_logger().info('Interactive Marker Controller initialized')

    def create_interactive_marker(self):
        """Create an interactive marker for the robot"""
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base_link"
        int_marker.name = "humanoid_control"
        int_marker.description = "Humanoid Robot Control"
        int_marker.pose.position.x = 0.0
        int_marker.pose.position.y = 0.0
        int_marker.pose.position.z = 1.0
        int_marker.pose.orientation.w = 1.0

        # Create control for moving in plane
        control = InteractiveMarkerControl()
        control.name = "move_xy"
        control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        control.orientation.w = 1.0
        control.orientation.x = 0.0
        control.orientation.y = 1.0
        control.orientation.z = 0.0

        # Add marker to control
        marker = Marker()
        marker.type = Marker.CYLINDER
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.1
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 0.5

        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)

        # Add menu control
        menu_control = InteractiveMarkerControl()
        menu_control.interaction_mode = InteractiveMarkerControl.MENU
        menu_control.description = "Right Click for Options"
        menu_control.name = "menu_only_control"
        int_marker.controls.append(menu_control)

        # Insert the marker
        self.server.insert(int_marker, feedback_callback=self.process_feedback)
        self.menu_handler.apply(self.server, int_marker.name)
        self.server.applyChanges()

    def process_feedback(self, feedback):
        """Process feedback from interactive marker"""
        self.get_logger().info(f'Feedback received: {feedback.event_type}')

    def reset_cb(self, feedback):
        """Reset callback"""
        self.get_logger().info('Reset command received')

    def home_cb(self, feedback):
        """Home command callback"""
        self.get_logger().info('Move to home command received')

def main(args=None):
    rclpy.init(args=args)
    controller = InteractiveMarkerController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()
```

## RViz Plugins for Humanoid Robots

### 1. Creating Custom Displays

You can create custom RViz displays for humanoid-specific data:

```python
# Custom display for humanoid balance visualization
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker
import math

class BalanceVisualizer(Node):
    def __init__(self):
        super().__init__('balance_visualizer')

        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Publisher for balance visualization
        self.balance_pub = self.create_publisher(Marker, 'balance_indicator', 10)

        self.get_logger().info('Balance Visualizer initialized')

    def imu_callback(self, msg):
        """Process IMU data for balance visualization"""
        # Extract orientation from IMU
        orientation = msg.orientation

        # Convert quaternion to roll/pitch (simplified)
        sinr_cosp = 2 * (orientation.w * orientation.x + orientation.y * orientation.z)
        cosr_cosp = 1 - 2 * (orientation.x * orientation.x + orientation.y * orientation.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (orientation.w * orientation.y - orientation.z * orientation.x)
        pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi/2, sinp)

        # Create balance indicator marker
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "balance"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Set scale
        marker.scale.x = 0.5  # Shaft length
        marker.scale.y = 0.1  # Head width
        marker.scale.z = 0.1  # Head height

        # Color based on balance (green = balanced, red = unbalanced)
        if abs(roll) < 0.2 and abs(pitch) < 0.2:
            marker.color.r = 0.0  # Green
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0  # Red
            marker.color.g = 0.0
            marker.color.b = 0.0
        marker.color.a = 1.0

        # Set direction based on tilt
        marker.points = []
        start_point = Vector3()
        start_point.x = 0.0
        start_point.y = 0.0
        start_point.z = 0.0
        marker.points.append(start_point)

        end_point = Vector3()
        end_point.x = -pitch * 0.5  # Tilt direction
        end_point.y = -roll * 0.5
        end_point.z = 0.0
        marker.points.append(end_point)

        self.balance_pub.publish(marker)
```

## Performance Optimization for Complex Humanoid Models

### 1. LOD (Level of Detail) Techniques

For complex humanoid models, consider using simplified versions for real-time visualization:

```xml
<!-- Use different meshes based on distance -->
<link name="head">
  <visual>
    <geometry>
      <!-- Use detailed mesh when close -->
      <mesh filename="package://humanoid_description/meshes/head_detailed.stl"/>
    </geometry>
  </visual>
  <collision>
    <!-- Use simple shape for collision -->
    <sphere radius="0.08"/>
  </collision>
</link>
```

### 2. Efficient Joint State Publishing

```python
# Efficient joint state publisher for many joints
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time

class EfficientJointStatePublisher(Node):
    def __init__(self):
        super().__init__('efficient_joint_state_publisher')

        # Publisher with appropriate QoS for visualization
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 1)

        # Use timer with appropriate frequency
        self.timer = self.create_timer(0.033, self.publish_joint_states)  # ~30 Hz

        # Pre-allocate joint names to avoid repeated allocations
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'neck_joint', 'waist_joint'
        ]

        # Pre-allocate position array
        self.positions = [0.0] * len(self.joint_names)

        self.get_logger().info('Efficient Joint State Publisher initialized')

    def publish_joint_states(self):
        """Efficiently publish joint states"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Update positions efficiently
        current_time = time.time()
        for i, _ in enumerate(self.joint_names):
            # Use pre-calculated positions based on time
            self.positions[i] = math.sin(current_time + i) * 0.1
        msg.position = self.positions

        # Publish with minimal overhead
        self.joint_pub.publish(msg)
```

## Debugging URDF in RViz

### 1. Common URDF Issues and Visualization

When your robot doesn't display correctly in RViz, check:

1. **Joint connections**: Are all links properly connected?
2. **Frame names**: Do TF frame names match your URDF?
3. **Origin definitions**: Are joint origins correctly defined?
4. **Mesh paths**: Are mesh files accessible?

### 2. TF Tree Validation

```bash
# Check TF tree
ros2 run tf2_tools view_frames

# View TF transforms
ros2 run rqt_tf_tree rqt_tf_tree

# Echo TF transforms
ros2 run tf2_ros tf2_echo base_link head
```

### 3. URDF Validation

```bash
# Check URDF syntax
check_urdf /path/to/your/robot.urdf

# Visualize kinematic tree
urdf_to_graphiz /path/to/your/robot.urdf
```

## Best Practices for Humanoid Robot Visualization

### 1. Color Coding
Use consistent colors for different robot parts:
- Blue: Legs
- Green: Arms
- Orange: Torso
- Grey: Feet/hands

### 2. Appropriate Scaling
Ensure visualization scales are appropriate for your robot size.

### 3. Frame Naming Convention
Use consistent naming:
- `base_link` for main robot frame
- `{side}_{part}_{joint}` for joints (e.g., `left_leg_hip`)
- `world` or `map` for global frames

### 4. Visualization Optimization
- Use simplified meshes for real-time visualization
- Reduce polygon count where possible
- Use appropriate update rates for different displays

## Launch Files for RViz Visualization

Here's a complete launch file for humanoid robot visualization:

```python
# launch/humanoid_rviz.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    declare_model_path = DeclareLaunchArgument(
        'model',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_description'),
            'urdf',
            'humanoid.urdf.xacro'
        ]),
        description='Path to robot URDF file'
    )

    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_description'),
            'rviz',
            'humanoid.rviz'
        ]),
        description='Path to RViz configuration file'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'robot_description': Command([
                'xacro ',
                LaunchConfiguration('model')
            ])}
        ],
        output='screen'
    )

    # Joint State Publisher (GUI for manual joint control)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # RViz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    return LaunchDescription([
        declare_model_path,
        declare_rviz_config,
        declare_use_sim_time,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz
    ])
```

## Summary

RViz is an essential tool for visualizing and debugging humanoid robots in ROS 2. It allows you to:

- See your robot model in 3D space
- Monitor sensor data and robot states
- Debug kinematic chains and joint configurations
- Visualize planning and navigation data
- Interact with your robot through various tools

Proper RViz configuration is crucial for developing and debugging complex humanoid robots with many joints and sensors.

## Exercises

1. Create a URDF for a simple humanoid robot with at least 12 joints and visualize it in RViz.

2. Add sensor plugins to your RViz configuration to visualize IMU data, camera feeds, and LiDAR scans from your humanoid robot.

3. Implement a custom RViz display that visualizes the robot's center of mass and balance indicators.

## Next Steps

In the next chapter, we'll create a practical mini-project where we control a humanoid arm using the concepts we've learned. We'll integrate all the components: nodes, topics, services, URDF, and visualization.

Continue to Chapter 11: [Mini Project: Controlling a Humanoid Arm via Python + ROS 2](./mini-project.md) to apply all learned concepts in a practical project.