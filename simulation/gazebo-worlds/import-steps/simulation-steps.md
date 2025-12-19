# Gazebo Import Simulation Steps

This guide provides step-by-step instructions for importing URDF models into Gazebo and configuring physics properties as covered in Module 2 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to import a URDF robot model into Gazebo, configure physics properties, and set up the simulation environment for realistic robot behavior.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- Gazebo Garden or Fortress installed
- Completed Module 1 simulation exercises
- Basic understanding of URDF robot models

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

## Exercise 1: Install Gazebo ROS2 Bridge

1. Install the Gazebo ROS2 bridge package:
   ```bash
   sudo apt update
   sudo apt install ros-humble-gazebo-ros ros-humble-gazebo-plugins ros-humble-gazebo-dev
   ```

2. Verify Gazebo installation:
   ```bash
   gazebo --version
   ```

## Exercise 2: Create Gazebo-Compatible URDF

1. Modify your URDF to include Gazebo-specific elements:
   ```xml
   <?xml version="1.0"?>
   <robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
     <!-- Include Gazebo plugins -->
     <gazebo>
       <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
         <parameters>$(find physical_ai_robotics)/config/humanoid_control.yaml</parameters>
       </plugin>
     </gazebo>

     <!-- Link with Gazebo properties -->
     <link name="base_link">
       <visual>
         <geometry>
           <box size="0.5 0.3 0.2"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.5 0.3 0.2"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Add Gazebo material -->
     <gazebo reference="base_link">
       <material>Gazebo/Blue</material>
       <mu1>0.2</mu1>
       <mu2>0.2</mu2>
       <kp>1000000.0</kp>
       <kd>1.0</kd>
     </gazebo>
   </robot>
   ```

2. Save as `src/physical_ai_robotics/urdf/humanoid_gazebo.urdf`.

## Exercise 3: Create Robot Spawn Launch File

1. Create a launch file to spawn the robot in Gazebo (`launch/spawn_robot.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, ExecuteProcess
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare

   def generate_launch_description():
       # Launch configuration
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')
       robot_name = LaunchConfiguration('robot_name', default='humanoid_robot')
       world_name = LaunchConfiguration('world_name', default='empty.sdf')

       # Paths
       pkg_share = FindPackageShare('physical_ai_robotics').find('physical_ai_robotics')
       urdf_path = PathJoinSubstitution([pkg_share, 'urdf', 'humanoid_gazebo.urdf'])

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),
           DeclareLaunchArgument(
               'robot_name',
               default_value='humanoid_robot',
               description='Name of the robot'
           ),
           DeclareLaunchArgument(
               'world_name',
               default_value='empty.sdf',
               description='Name of the world file'
           ),

           # Start Gazebo
           ExecuteProcess(
               cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so',
                    '-s', 'libgazebo_ros_init.so'],
               output='screen'
           ),

           # Robot state publisher
           Node(
               package='robot_state_publisher',
               executable='robot_state_publisher',
               name='robot_state_publisher',
               parameters=[{'use_sim_time': use_sim_time,
                           'robot_description':
                               open(urdf_path.get_path()).read()}],
               output='screen'
           ),

           # Spawn robot in Gazebo
           Node(
               package='gazebo_ros',
               executable='spawn_entity.py',
               arguments=['-topic', 'robot_description',
                         '-entity', robot_name,
                         '-x', '0', '-y', '0', '-z', '0.5'],
               output='screen'
           )
       ])
   ```

## Exercise 4: Launch Robot in Gazebo

1. Launch the robot in Gazebo:
   ```bash
   ros2 launch physical_ai_robotics spawn_robot.launch.py
   ```

2. Observe the robot spawning in the Gazebo environment.

## Exercise 5: Configure Physics Properties

1. Create a physics configuration file (`config/physics_properties.yaml`):
   ```yaml
   physics:
     type: ode
     update_rate: 1000
     max_step_size: 0.001
     real_time_factor: 1
     real_time_update_rate: 1000
     gravity: [0.0, 0.0, -9.8]
     ode_config:
       solver_type: quick
       iters: 10
       sor: 1.3
       contact_surface_layer: 0.001
       contact_max_correcting_vel: 100.0
       cfm: 0.0
       erp: 0.2
       max_contacts: 20
   ```

2. Apply physics configuration in Gazebo launch.

## Exercise 6: Add Joint Control Plugins

1. Create a controller configuration file (`config/humanoid_control.yaml`):
   ```yaml
   controller_manager:
     ros__parameters:
       update_rate: 100
       use_sim_time: true

   joint_state_broadcaster:
     type: joint_state_broadcaster/JointStateBroadcaster

   # Example position controllers for humanoid joints
   shoulder_controller:
     type: position_controllers/JointPositionController
     joint: shoulder_yaw
     interface_name: position

   elbow_controller:
     type: position_controllers/JointPositionController
     joint: elbow_pitch
     interface_name: position

   wrist_controller:
     type: position_controllers/JointPositionController
     joint: wrist_roll
     interface_name: position
   ```

2. Add controller spawners to your launch file.

## Exercise 7: Test Joint Control in Simulation

1. Launch the robot with controllers:
   ```bash
   ros2 launch physical_ai_robotics spawn_robot.launch.py
   ```

2. In another terminal, send commands to the joints:
   ```bash
   ros2 topic pub /shoulder_controller/commands std_msgs/msg/Float64MultiArray '{data: [0.5]}'
   ```

3. Observe the robot responding in Gazebo.

## Exercise 8: Configure Collision Properties

1. Modify URDF to include proper collision properties:
   ```xml
   <collision>
     <geometry>
       <box size="0.5 0.3 0.2"/>
     </geometry>
     <surface>
       <friction>
         <ode>
           <mu>1.0</mu>
           <mu2>1.0</mu2>
           <fdir1>0 0 0</fdir1>
           <slip1>0</slip1>
           <slip2>0</slip2>
         </ode>
       </friction>
       <bounce>
         <restitution_coefficient>0.1</restitution_coefficient>
         <threshold>100000</threshold>
       </bounce>
       <contact>
         <ode>
           <soft_cfm>0</soft_cfm>
           <soft_erp>0.2</soft_erp>
           <kp>1e+13</kp>
           <kd>1</kd>
           <max_vel>0.01</max_vel>
           <min_depth>0</min_depth>
         </ode>
       </contact>
     </surface>
   </collision>
   ```

## Exercise 9: Create Custom Gazebo World

1. Create a custom world file (`worlds/humanoid_world.sdf`):
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="humanoid_world">
       <light name="sun" type="directional">
         <cast_shadows>true</cast_shadows>
         <pose>0 0 10 0 0 0</pose>
         <diffuse>0.8 0.8 0.8 1</diffuse>
         <specular>0.2 0.2 0.2 1</specular>
         <attenuation>
           <range>1000</range>
           <constant>0.9</constant>
           <linear>0.01</linear>
           <quadratic>0.001</quadratic>
         </attenuation>
         <direction>-0.6 0.4 -0.8</direction>
       </light>

       <model name="ground_plane">
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>100 100</size>
               </plane>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>100 100</size>
               </plane>
             </geometry>
             <material>
               <ambient>0.8 0.8 0.8 1</ambient>
               <diffuse>0.8 0.8 0.8 1</diffuse>
               <specular>0.8 0.8 0.8 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <!-- Add obstacles and environment elements -->
       <model name="table">
         <pose>2 0 0 0 0 0</pose>
         <link name="table_link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>1 0.8 0.8</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>1 0.8 0.8</size>
               </box>
             </geometry>
             <material>
               <ambient>0.5 0.3 0.1 1</ambient>
               <diffuse>0.5 0.3 0.1 1</diffuse>
             </material>
           </visual>
           <inertial>
             <mass>10</mass>
             <inertia>
               <ixx>1</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>1</iyy>
               <iyz>0</iyz>
               <izz>1</izz>
             </inertia>
           </inertial>
         </link>
       </model>
     </world>
   </sdf>
   ```

## Exercise 10: Test Robot in Custom World

1. Launch robot in custom world:
   ```bash
   ros2 launch physical_ai_robotics spawn_robot.launch.py world_name:=humanoid_world.sdf
   ```

2. Test robot movement and interaction with environment objects.

## Verification Steps

1. Confirm that robot spawns correctly in Gazebo
2. Verify that joint controllers work properly
3. Check that physics properties behave as expected
4. Ensure collision detection works correctly

## Expected Outcomes

- Understanding of URDF to Gazebo import process
- Knowledge of Gazebo-specific URDF elements
- Experience with physics configuration
- Ability to create custom simulation environments

## Troubleshooting

- If robot doesn't spawn, check URDF syntax and Gazebo plugin configuration
- If joints don't respond, verify controller configuration and topic names
- If physics behave unexpectedly, adjust collision and friction parameters

## Next Steps

After completing these exercises, proceed to the sensor simulation exercises to understand how to integrate perception sensors in Gazebo.