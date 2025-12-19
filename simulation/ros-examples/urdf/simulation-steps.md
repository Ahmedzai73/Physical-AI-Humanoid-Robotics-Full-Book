# URDF Fundamentals and Humanoid Modeling Simulation Steps

This guide provides step-by-step instructions for simulating URDF modeling concepts covered in Module 1 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to create URDF (Unified Robot Description Format) models for humanoid robots, including proper joint configurations, kinematic chains, and visualization in RViz.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- RViz2 installed
- Basic understanding of 3D geometry and kinematics
- Completed the rclpy integration simulation steps

## Simulation Environment Setup

1. Open a terminal and navigate to the ROS workspace:
   ```bash
   cd simulation/ros-workspace
   ```

2. Source the ROS 2 installation and workspace:
   ```bash
   source /opt/ros/humble/setup.bash  # Adjust for your ROS 2 distribution
   source install/setup.bash
   ```

## Exercise 1: Basic URDF Structure

1. Examine the simple robot URDF created earlier:
   ```bash
   cat src/physical_ai_robotics/urdf/simple_robot.urdf
   ```

2. Visualize the robot in RViz:
   ```bash
   # Terminal 1: Start robot state publisher
   ros2 launch physical_ai_robotics basic_robot.launch.py
   ```

3. In RViz, add a RobotModel display and set the fixed frame to 'base_link'.

## Exercise 2: Creating a Humanoid Skeleton URDF

1. Create a basic humanoid URDF file (urdf/humanoid.urdf):
   ```xml
   <?xml version="1.0"?>
   <robot name="humanoid_robot">
     <!-- Pelvis (base link) -->
     <link name="pelvis">
       <visual>
         <geometry>
           <box size="0.3 0.2 0.1"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.3 0.2 0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="5.0"/>
         <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Torso -->
     <link name="torso">
       <visual>
         <geometry>
           <box size="0.2 0.1 0.4"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.2 0.1 0.4"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="3.0"/>
         <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
       </inertial>
     </link>

     <!-- Joints connecting links -->
     <joint name="pelvis_torso_joint" type="fixed">
       <parent link="pelvis"/>
       <child link="torso"/>
       <origin xyz="0 0 0.25"/>
     </joint>

     <!-- Add more links and joints for arms, legs, head -->
     <!-- Following the pattern above -->
   </robot>
   ```

2. Save the file as `src/physical_ai_robotics/urdf/humanoid.urdf`.

## Exercise 3: Adding Arms to Humanoid Model

1. Extend the humanoid URDF with arms:
   ```xml
   <!-- Left upper arm -->
   <link name="left_upper_arm">
     <visual>
       <geometry>
         <cylinder length="0.3" radius="0.05"/>
       </geometry>
       <material name="green">
         <color rgba="0 1 0 0.8"/>
       </material>
     </visual>
     <collision>
       <geometry>
         <cylinder length="0.3" radius="0.05"/>
       </geometry>
     </collision>
     <inertial>
       <mass value="1.0"/>
       <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
     </inertial>
   </link>

   <!-- Left shoulder joint -->
   <joint name="left_shoulder_joint" type="revolute">
     <parent link="torso"/>
     <child link="left_upper_arm"/>
     <origin xyz="0.05 0.15 0.1"/>
     <axis xyz="0 1 0"/>
     <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
   </joint>
   ```

2. Add right arm following the same pattern.

## Exercise 4: Adding Legs to Humanoid Model

1. Extend the humanoid URDF with legs:
   ```xml
   <!-- Left thigh -->
   <link name="left_thigh">
     <visual>
       <geometry>
         <cylinder length="0.4" radius="0.06"/>
       </geometry>
       <material name="yellow">
         <color rgba="1 1 0 0.8"/>
       </material>
     </visual>
     <collision>
       <geometry>
         <cylinder length="0.4" radius="0.06"/>
       </geometry>
     </collision>
     <inertial>
       <mass value="1.5"/>
       <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
     </inertial>
   </link>

   <!-- Left hip joint -->
   <joint name="left_hip_joint" type="revolute">
     <parent link="pelvis"/>
     <child link="left_thigh"/>
     <origin xyz="0.05 -0.1 -0.05"/>
     <axis xyz="0 1 0"/>
     <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
   </joint>
   ```

2. Add right leg following the same pattern.

## Exercise 5: Visualizing Humanoid Model in RViz

1. Create a launch file for the humanoid robot:
   ```xml
   # Create launch/humanoid.launch.py
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
               description='Use simulation (Gazebo) clock if true'
           ),
           # Robot state publisher
           Node(
               package='robot_state_publisher',
               executable='robot_state_publisher',
               name='robot_state_publisher',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),
           # Joint state publisher (for visualization)
           Node(
               package='joint_state_publisher',
               executable='joint_state_publisher',
               name='joint_state_publisher',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           )
       ])
   ```

2. Launch the humanoid model:
   ```bash
   ros2 launch physical_ai_robotics humanoid.launch.py
   ```

3. Visualize in RViz with the RobotModel display.

## Exercise 6: Testing Kinematic Chains

1. Use joint state publisher GUI to move joints:
   ```bash
   ros2 run joint_state_publisher_gui joint_state_publisher_gui
   ```

2. Observe how joint movements affect the kinematic chain.

## Exercise 7: Adding Gazebo-Specific Elements

1. Create a Gazebo-compatible URDF with physics properties:
   ```xml
   <!-- Add Gazebo-specific elements to your URDF -->
   <gazebo reference="base_link">
     <material>Gazebo/Blue</material>
   </gazebo>

   <!-- Add transmission elements for actuator control -->
   <transmission name="left_shoulder_trans">
     <type>transmission_interface/SimpleTransmission</type>
     <joint name="left_shoulder_joint">
       <hardwareInterface>PositionJointInterface</hardwareInterface>
     </joint>
     <actuator name="left_shoulder_motor">
       <hardwareInterface>PositionJointInterface</hardwareInterface>
       <mechanicalReduction>1</mechanicalReduction>
     </actuator>
   </transmission>
   ```

## Exercise 8: Validating URDF Model

1. Check URDF syntax:
   ```bash
   # Install urdfdom package if not already installed
   check_urdf src/physical_ai_robotics/urdf/humanoid.urdf
   ```

2. Visualize the kinematic tree:
   ```bash
   # Generate a PDF of the kinematic tree
   urdf_to_graphiz src/physical_ai_robotics/urdf/humanoid.urdf
   ```

## Exercise 9: Creating Robot Description Launch

1. Set up robot description for the system:
   ```bash
   # Create a launch file that properly sets robot_description
   # This makes the URDF available to other ROS nodes
   ```

2. Test that other nodes can access the robot description:
   ```bash
   ros2 run tf2_tools view_frames
   ```

## Exercise 10: Integration with Navigation

1. Prepare URDF for navigation stack:
   ```xml
   <!-- Add necessary elements for navigation -->
   <!-- Laser scanner mount points -->
   <!-- IMU mounting -->
   <!-- Camera mounting points -->
   ```

2. Verify the URDF is ready for simulation in Gazebo or Isaac Sim.

## Verification Steps

1. Confirm that the URDF loads without errors
2. Verify that all joints are properly connected
3. Check that the robot model displays correctly in RViz
4. Ensure kinematic chains work as expected

## Expected Outcomes

- Understanding of URDF structure and elements
- Knowledge of proper joint configurations for humanoid robots
- Experience with visualizing robot models in RViz
- Ability to create complete robot descriptions

## Troubleshooting

- If URDF fails to load, check for XML syntax errors
- If joints don't appear in RViz, verify joint connections
- If robot looks deformed, check origin and axis definitions

## Next Steps

After completing these exercises, proceed to the RViz visualization simulation exercises to understand robot visualization in robotics applications.