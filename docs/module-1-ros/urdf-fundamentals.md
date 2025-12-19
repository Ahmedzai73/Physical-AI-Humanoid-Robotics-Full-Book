---
title: URDF Basics - Robot Description for Humanoid Robots
sidebar_position: 9
description: Understanding Unified Robot Description Format for modeling humanoid robots in ROS
---

# URDF Basics for Humanoid Robots

## Introduction

In the previous chapters, we've learned about ROS 2's communication mechanisms and how to integrate AI agents with robotic controllers. Now we'll explore **URDF (Unified Robot Description Format)**, which is essential for describing robot models in ROS. URDF is an XML-based format that defines the physical and kinematic properties of robots, making it possible to simulate, visualize, and control them in ROS environments.

For humanoid robots with complex kinematic chains and many degrees of freedom, understanding URDF is crucial. It defines everything from the robot's physical structure to its visual appearance, collision properties, and joint constraints.

## Understanding URDF Concepts

URDF describes a robot as a collection of rigid bodies (links) connected by joints. This creates a kinematic tree structure that represents the robot's physical makeup.

### Key URDF Elements

- **Links**: Rigid bodies that represent robot parts (limbs, torso, head, etc.)
- **Joints**: Connections between links that allow relative motion
- **Materials**: Visual properties for rendering
- **Transmissions**: Mapping between joints and actuators
- **Gazebo tags**: Simulation-specific properties

### URDF Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_leg"/>
    <origin xyz="0 -0.1 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_leg">
    <!-- Link definition -->
  </link>
</robot>
```

## Creating Links

Links represent the rigid bodies of the robot. Each link has three main components:

### 1. Visual Properties
Defines how the link appears in visualization tools like RViz and simulation environments.

```xml
<link name="link_name">
  <visual>
    <!-- Origin: position and orientation relative to joint -->
    <origin xyz="0 0 0.1" rpy="0 0 0"/>

    <!-- Geometry: shape and dimensions -->
    <geometry>
      <!-- Box geometry -->
      <box size="0.1 0.1 0.2"/>

      <!-- Sphere geometry -->
      <!-- <sphere radius="0.05"/> -->

      <!-- Cylinder geometry -->
      <!-- <cylinder radius="0.05" length="0.1"/> -->

      <!-- Mesh geometry -->
      <!-- <mesh filename="package://robot_description/meshes/link.stl"/> -->
    </geometry>

    <!-- Material -->
    <material name="red">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>
</link>
```

### 2. Collision Properties
Defines how the link interacts in collision detection and physics simulation.

```xml
<collision>
  <!-- Usually simpler geometry than visual for performance -->
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <geometry>
    <box size="0.1 0.1 0.2"/>
  </geometry>
</collision>
```

### 3. Inertial Properties
Defines mass and inertial tensor for physics simulation.

```xml
<inertial>
  <!-- Mass of the link -->
  <mass value="0.5"/>

  <!-- Inertia matrix (in link's frame) -->
  <inertia
    ixx="0.01" ixy="0.0" ixz="0.0"
    iyy="0.01" iyz="0.0"
    izz="0.01"/>
</inertial>
```

## Creating Joints

Joints define how links connect and move relative to each other. For humanoid robots, the joint types are particularly important for realistic movement.

### Joint Types

1. **Revolute**: Rotational joint with limited range
2. **Continuous**: Rotational joint without limits
3. **Prismatic**: Linear sliding joint
4. **Fixed**: No movement (rigid connection)
5. **Floating**: 6DOF (rarely used)
6. **Planar**: Motion constrained to a plane (rarely used)

### Joint Definition

```xml
<joint name="joint_name" type="revolute">
  <!-- Parent link (closer to base) -->
  <parent link="parent_link_name"/>

  <!-- Child link (further from base) -->
  <child link="child_link_name"/>

  <!-- Position and orientation of joint frame -->
  <origin xyz="0.0 0.0 0.1" rpy="0 0 0"/>

  <!-- Rotation axis -->
  <axis xyz="0 0 1"/>

  <!-- Joint limits -->
  <limit
    lower="-1.57"      <!-- Lower limit in radians -->
    upper="1.57"       <!-- Upper limit in radians -->
    effort="100"       <!-- Max torque (Nm) -->
    velocity="2"/>     <!-- Max velocity (rad/s) -->

  <!-- Joint dynamics -->
  <dynamics damping="0.1" friction="0.01"/>
</joint>
```

## Complete Humanoid Robot URDF Example

Here's a simplified example of a humanoid robot URDF:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.2 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link (pelvis) -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.25 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Spine/torso -->
  <joint name="torso_joint" type="revolute">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>

  <link name="torso">
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.2 0.4"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.2 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="8.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Hip -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="0 -0.1 -0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="150" velocity="1"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Knee -->
  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.5" effort="150" velocity="1"/>
  </joint>

  <link name="left_shin">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Ankle -->
  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1"/>
  </joint>

  <link name="left_foot">
    <visual>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right Leg (similar to left leg) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="0 0.1 -0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="150" velocity="1"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.05" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.5" effort="150" velocity="1"/>
  </joint>

  <link name="right_shin">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1"/>
  </joint>

  <link name="right_foot">
    <visual>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Left Shoulder -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 -0.12 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.15"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Left Elbow -->
  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_forearm"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="0.5" effort="80" velocity="1"/>
  </joint>

  <link name="left_forearm">
    <visual>
      <origin xyz="0 0 -0.08" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.03" length="0.1"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.08" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.03" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right Arm (similar to left arm) -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0 0.12 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.15"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.04" length="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_forearm"/>
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="0.5" effort="80" velocity="1"/>
  </joint>

  <link name="right_forearm">
    <visual>
      <origin xyz="0 0 -0.08" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.03" length="0.1"/>
      </geometry>
      <material name="green"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.08" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.03" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Transmissions for control -->
  <transmission name="left_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_knee_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_knee_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_ankle_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_ankle_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_ankle_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Additional transmissions for other joints would follow the same pattern -->

</robot>
```

## Xacro for Complex URDFs

For complex humanoid robots, using Xacro (XML Macros) can greatly simplify the URDF definition:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Properties -->
  <xacro:property name="PI" value="3.1415926535897931"/>

  <!-- Define a macro for a leg -->
  <xacro:macro name="leg" params="prefix reflect">
    <joint name="${prefix}_hip_joint" type="revolute">
      <parent link="base_link"/>
      <child link="${prefix}_thigh"/>
      <origin xyz="0 ${reflect*0.1} -0.05" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="-${PI/2}" upper="${PI/2}" effort="150" velocity="1"/>
    </joint>

    <link name="${prefix}_thigh">
      <visual>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <capsule radius="0.05" length="0.2"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <capsule radius="0.05" length="0.2"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="3.0"/>
        <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>

    <joint name="${prefix}_knee_joint" type="revolute">
      <parent link="${prefix}_thigh"/>
      <child link="${prefix}_shin"/>
      <origin xyz="0 0 -0.3" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="0" upper="${PI*0.8}" effort="150" velocity="1"/>
    </joint>

    <link name="${prefix}_shin">
      <visual>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <capsule radius="0.04" length="0.2"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <capsule radius="0.04" length="0.2"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="2.0"/>
        <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the macro to create both legs -->
  <xacro:leg prefix="left" reflect="1"/>
  <xacro:leg prefix="right" reflect="-1"/>

</robot>
```

## Working with URDF in ROS 2

### 1. Loading URDF in Launch Files

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare arguments
    declare_model_path = DeclareLaunchArgument(
        'model',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_description'),
            'urdf',
            'humanoid.urdf.xacro'
        ]),
        description='Path to robot URDF file'
    )

    # Launch robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': Command(['xacro ', LaunchConfiguration('model')])
        }],
        output='screen'
    )

    # Launch joint state publisher (for visualization)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen'
    )

    return LaunchDescription([
        declare_model_path,
        robot_state_publisher,
        joint_state_publisher
    ])
```

### 2. Validating URDF

Before using a URDF, it's important to validate it:

```bash
# Check if URDF is syntactically correct
check_urdf /path/to/robot.urdf

# Or with xacro
xacro --inorder robot.urdf.xacro | check_urdf /dev/stdin

# Visualize the kinematic tree
urdf_to_graphiz /path/to/robot.urdf
```

### 3. Using Robot State Publisher

The robot_state_publisher node broadcasts the robot's joint states as transforms:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Timer for publishing joint states
        self.timer = self.create_timer(0.1, self.publish_joint_states)

        # Initial joint positions
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

        self.joint_positions = [0.0] * len(self.joint_names)

        self.get_logger().info('Joint State Publisher initialized')

    def publish_joint_states(self):
        """Publish joint states to robot_state_publisher"""
        # Create joint state message
        msg = JointState()
        msg.name = self.joint_names

        # Update positions (in a real system, these would come from encoders)
        for i in range(len(self.joint_positions)):
            self.joint_positions[i] = math.sin(
                self.get_clock().now().nanoseconds * 1e-9 + i
            ) * 0.5  # Oscillating joints for visualization

        msg.position = self.joint_positions
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Publish the message
        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    publisher = JointStatePublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Visualization with RViz

To visualize your URDF in RViz:

1. Launch your robot:
```bash
ros2 launch robot_description display.launch.py
```

2. In RViz, add a RobotModel display and set the TF frame to your robot's base frame.

3. You should see your robot model with all links and joints properly displayed.

## Common URDF Issues and Solutions

### 1. Floating Point Precision
Use consistent precision in URDF files:
```xml
<!-- Good -->
<origin xyz="0.0 0.0 0.1" rpy="0.0 0.0 0.0"/>

<!-- Avoid -->
<origin xyz="0 0 0.1" rpy="0 0 0"/>
```

### 2. Joint Direction
Make sure joint axes are correctly oriented:
```xml
<!-- For a hip joint that moves the leg side-to-side -->
<axis xyz="0 0 1"/>  <!-- Z-axis for rotation -->
```

### 3. Inertial Properties
Always provide realistic inertial properties for physics simulation:
```xml
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
</inertial>
```

### 4. Connected Links
Ensure all links are properly connected through joints:
```xml
<!-- Every child link should have a corresponding joint -->
<joint name="connection" type="fixed">
  <parent link="link_a"/>
  <child link="link_b"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

## Advanced URDF Features

### 1. Gazebo-Specific Tags

For simulation in Gazebo:

```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>

<!-- Gazebo plugins -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/humanoid</robotNamespace>
  </plugin>
</gazebo>
```

### 2. Safety Controllers

For real robots, add safety controllers:

```xml
<safety_controller k_velocity="10" soft_lower_limit="-1.5" soft_upper_limit="1.5"/>
```

## Best Practices for Humanoid Robots

### 1. Consistent Naming Convention
```xml
<!-- Good naming convention -->
<joint name="left_leg_hip_yaw" type="revolute"/>
<joint name="left_leg_hip_roll" type="revolute"/>
<joint name="left_leg_hip_pitch" type="revolute"/>
```

### 2. Proper Frame Definitions
Always define frames consistently:
- Base frame at robot's center of mass or pelvis
- X-forward, Y-left, Z-up coordinate system
- Joint origins at physical joint locations

### 3. Realistic Limits
Set realistic joint limits based on human anatomy or mechanical constraints:
```xml
<!-- Human-like joint limits -->
<limit lower="-2.0" upper="1.0" effort="100" velocity="2"/>
```

### 4. Separate Files
Organize large URDFs into multiple files:
```
robot.urdf.xacro
├── base.urdf.xacro
├── leg.urdf.xacro
├── arm.urdf.xacro
└── head.urdf.xacro
```

## Tools for Working with URDF

### 1. Command Line Tools
```bash
# View robot model
ros2 run rviz2 rviz2

# Check URDF
check_urdf robot.urdf

# Convert URDF to DOT graph
urdf_to_graphiz robot.urdf
```

### 2. Visualization
```bash
# Launch robot in RViz
ros2 launch robot_description display.launch.py model:=path/to/robot.urdf

# With joint state publisher for interactive viewing
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

## Summary

URDF is fundamental to representing robots in ROS 2, especially for complex humanoid robots with many degrees of freedom. Understanding URDF concepts is crucial for:
- Robot simulation and visualization
- Kinematic calculations
- Physics simulation
- Motion planning
- Control system development

Proper URDF modeling ensures that your humanoid robot behaves correctly in both simulation and real-world applications.

## Exercises

1. Create a simple 2-link manipulator URDF with proper inertial properties and visualize it in RViz.

2. Design a URDF for a simple humanoid robot with at least 12 joints (6 DOF per leg, 6 DOF for arms) using Xacro macros.

3. Implement a joint state publisher that animates your humanoid robot model with realistic movement patterns.

## Next Steps

In the next chapter, we'll explore how to build a complete humanoid robot URDF model and visualize it in RViz. We'll also learn how to control the joints using ROS 2 interfaces.

Continue to Chapter 10: [Building a Full Humanoid URDF + Visualization in RViz](./rviz-visualization.md) to learn about complete robot modeling and visualization.