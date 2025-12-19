# Sensor Simulation in Gazebo Simulation Steps

This guide provides step-by-step instructions for simulating perception sensors (LiDAR, Depth Camera, IMU) in Gazebo and connecting them to ROS 2 nodes as covered in Module 2 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to add various sensors to your robot model in Gazebo and connect them to ROS 2 for realistic sensor data simulation.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- Gazebo Garden or Fortress installed
- Completed Gazebo import simulation exercises
- Basic understanding of ROS 2 sensor message types

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

## Exercise 1: Add LiDAR Sensor to Robot URDF

1. Modify your URDF to include a LiDAR sensor:
   ```xml
   <?xml version="1.0"?>
   <robot name="humanoid_robot_with_sensors">
     <!-- Robot links and joints from previous exercises -->

     <!-- LiDAR sensor link -->
     <link name="lidar_link">
       <visual>
         <geometry>
           <cylinder radius="0.05" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.05" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.1"/>
         <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
       </inertial>
     </link>

     <!-- Joint to attach LiDAR to robot -->
     <joint name="lidar_joint" type="fixed">
       <parent link="base_link"/>
       <child link="lidar_link"/>
       <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
     </joint>

     <!-- Gazebo plugin for LiDAR -->
     <gazebo reference="lidar_link">
       <sensor name="lidar_sensor" type="ray">
         <pose>0 0 0 0 0 0</pose>
         <visualize>true</visualize>
         <update_rate>10</update_rate>
         <ray>
           <scan>
             <horizontal>
               <samples>360</samples>
               <resolution>1.0</resolution>
               <min_angle>-3.14159</min_angle>
               <max_angle>3.14159</max_angle>
             </horizontal>
           </scan>
           <range>
             <min>0.1</min>
             <max>30.0</max>
             <resolution>0.01</resolution>
           </range>
         </ray>
         <plugin name="gazebo_ros_lidar" filename="libgazebo_ros_ray_sensor.so">
           <ros>
             <namespace>/robot</namespace>
             <remapping>~/out:=scan</remapping>
           </ros>
           <output_type>sensor_msgs/LaserScan</output_type>
         </plugin>
       </sensor>
     </gazebo>
   </robot>
   ```

2. Save as `src/physical_ai_robotics/urdf/humanoid_with_lidar.urdf`.

## Exercise 2: Add Depth Camera Sensor

1. Add a depth camera to your URDF:
   ```xml
   <!-- Depth camera link -->
   <link name="camera_link">
     <visual>
       <geometry>
         <box size="0.05 0.05 0.05"/>
       </geometry>
       <material name="red">
         <color rgba="1 0 0 1"/>
       </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.05 0.05 0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.05"/>
         <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
       </inertial>
   </link>

   <!-- Joint to attach camera to robot -->
   <joint name="camera_joint" type="fixed">
     <parent link="base_link"/>
     <child link="camera_link"/>
     <origin xyz="0.25 0 0.05" rpy="0 0 0"/>
   </joint>

   <!-- Gazebo plugin for depth camera -->
   <gazebo reference="camera_link">
     <sensor name="depth_camera" type="depth">
       <visualize>true</visualize>
       <update_rate>30</update_rate>
       <camera name="head">
         <horizontal_fov>1.047</horizontal_fov>
         <image>
           <width>640</width>
           <height>480</height>
           <format>R8G8B8</format>
         </image>
         <clip>
           <near>0.1</near>
           <far>10</far>
         </clip>
       </camera>
       <plugin name="gazebo_ros_depth_camera" filename="libgazebo_ros_depth_camera.so">
         <ros>
           <namespace>/robot</namespace>
           <remapping>image_raw:=/camera/image_raw</remapping>
           <remapping>camera_info:=/camera/camera_info</remapping>
           <remapping>depth/image_raw:=/camera/depth/image_raw</remapping>
         </ros>
         <output_type>sensor_msgs/Image</output_type>
       </plugin>
     </sensor>
   </gazebo>
   ```

## Exercise 3: Add IMU Sensor

1. Add an IMU sensor to your URDF:
   ```xml
   <!-- IMU link -->
   <link name="imu_link">
     <inertial>
       <mass value="0.01"/>
       <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
     </inertial>
   </link>

   <!-- Joint to attach IMU to robot -->
   <joint name="imu_joint" type="fixed">
     <parent link="base_link"/>
     <child link="imu_link"/>
     <origin xyz="0 0 0.1" rpy="0 0 0"/>
   </joint>

   <!-- Gazebo plugin for IMU -->
   <gazebo reference="imu_link">
     <sensor name="imu_sensor" type="imu">
       <always_on>true</always_on>
       <update_rate>100</update_rate>
       <visualize>false</visualize>
       <imu>
         <angular_velocity>
           <x>
             <noise type="gaussian">
               <mean>0.0</mean>
               <stddev>2e-4</stddev>
             </noise>
           </x>
           <y>
             <noise type="gaussian">
               <mean>0.0</mean>
               <stddev>2e-4</stddev>
             </noise>
           </y>
           <z>
             <noise type="gaussian">
               <mean>0.0</mean>
               <stddev>2e-4</stddev>
             </noise>
           </z>
         </angular_velocity>
         <linear_acceleration>
           <x>
             <noise type="gaussian">
               <mean>0.0</mean>
               <stddev>1.7e-2</stddev>
             </noise>
           </x>
           <y>
             <noise type="gaussian">
               <mean>0.0</mean>
               <stddev>1.7e-2</stddev>
             </noise>
           </y>
           <z>
             <noise type="gaussian">
               <mean>0.0</mean>
               <stddev>1.7e-2</stddev>
             </noise>
           </z>
         </linear_acceleration>
       </imu>
       <plugin name="gazebo_ros_imu" filename="libgazebo_ros_imu.so">
         <ros>
           <namespace>/robot</namespace>
           <remapping>~/out:=imu</remapping>
         </ros>
         <initial_orientation_as_reference>false</initial_orientation_as_reference>
       </plugin>
     </sensor>
   </gazebo>
   ```

## Exercise 4: Create Complete Sensor URDF

1. Combine all sensors in a complete URDF file:
   ```xml
   <?xml version="1.0"?>
   <robot name="humanoid_robot_with_all_sensors">
     <!-- Base link -->
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

     <!-- Include all sensor definitions from above -->
     <!-- LiDAR, Camera, IMU, etc. -->

     <!-- Gazebo plugin for ROS control -->
     <gazebo>
       <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
         <parameters>$(find physical_ai_robotics)/config/humanoid_control.yaml</parameters>
       </plugin>
     </gazebo>
   </robot>
   ```

2. Save as `src/physical_ai_robotics/urdf/humanoid_with_sensors.urdf`.

## Exercise 5: Create Sensor Launch File

1. Create a launch file for the robot with sensors (`launch/sensor_robot.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, ExecuteProcess
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')
       robot_name = LaunchConfiguration('robot_name', default='sensor_robot')

       pkg_share = FindPackageShare('physical_ai_robotics').find('physical_ai_robotics')
       urdf_path = PathJoinSubstitution([pkg_share, 'urdf', 'humanoid_with_sensors.urdf'])

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),
           DeclareLaunchArgument(
               'robot_name',
               default_value='sensor_robot',
               description='Name of the robot'
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

## Exercise 6: Launch Robot with Sensors

1. Launch the robot with all sensors:
   ```bash
   ros2 launch physical_ai_robotics sensor_robot.launch.py
   ```

2. Verify that the robot appears with all sensors in Gazebo.

## Exercise 7: Subscribe to Sensor Data

1. In another terminal, check available topics:
   ```bash
   ros2 topic list | grep robot
   ```

2. Subscribe to LiDAR data:
   ```bash
   ros2 topic echo /robot/scan sensor_msgs/msg/LaserScan
   ```

3. Subscribe to camera data:
   ```bash
   ros2 topic echo /robot/camera/image_raw sensor_msgs/msg/Image
   ```

4. Subscribe to IMU data:
   ```bash
   ros2 topic echo /robot/imu sensor_msgs/msg/Imu
   ```

## Exercise 8: Visualize Sensor Data in RViz

1. Start RViz in a new terminal:
   ```bash
   rviz2
   ```

2. Add displays for sensor data:
   - "LaserScan" display for LiDAR data
   - "Image" display for camera data
   - "RobotModel" display for robot visualization
   - "TF" display for coordinate frames

3. Set the appropriate topics for each display.

## Exercise 9: Create Sensor Processing Node

1. Create a sensor fusion node (`src/physical_ai_robotics/sensor_processor.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import LaserScan, Image, Imu
   from geometry_msgs.msg import Twist
   import numpy as np

   class SensorProcessor(Node):
       def __init__(self):
           super().__init__('sensor_processor')

           # Subscribers for all sensor data
           self.lidar_subscriber = self.create_subscription(
               LaserScan, '/robot/scan', self.lidar_callback, 10
           )
           self.camera_subscriber = self.create_subscription(
               Image, '/robot/camera/image_raw', self.camera_callback, 10
           )
           self.imu_subscriber = self.create_subscription(
               Imu, '/robot/imu', self.imu_callback, 10
           )

           # Publisher for robot commands
           self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

           self.get_logger().info('Sensor Processor initialized')

           # Store latest sensor data
           self.latest_lidar = None
           self.latest_image = None
           self.latest_imu = None

       def lidar_callback(self, msg):
           """Process LiDAR data"""
           self.latest_lidar = msg
           self.get_logger().debug(f'Received LiDAR data with {len(msg.ranges)} ranges')

           # Simple obstacle detection
           if self.is_obstacle_ahead(msg):
               self.stop_robot()

       def camera_callback(self, msg):
           """Process camera data"""
           self.latest_image = msg
           self.get_logger().debug(f'Received camera image: {msg.width}x{msg.height}')

       def imu_callback(self, msg):
           """Process IMU data"""
           self.latest_imu = msg
           self.get_logger().debug(f'Received IMU data')

       def is_obstacle_ahead(self, scan_msg):
           """Check if there's an obstacle directly ahead"""
           # Check the middle range values (front of robot)
           front_ranges = scan_msg.ranges[len(scan_msg.ranges)//2-10:len(scan_msg.ranges)//2+10]
           front_ranges = [r for r in front_ranges if not np.isnan(r) and r != float('inf')]

           if front_ranges:
               min_distance = min(front_ranges)
               return min_distance < 1.0  # Obstacle within 1 meter
           return False

       def stop_robot(self):
           """Send stop command to robot"""
           cmd = Twist()
           cmd.linear.x = 0.0
           cmd.angular.z = 0.0
           self.cmd_publisher.publish(cmd)
           self.get_logger().info('Obstacle detected! Stopping robot.')

   def main(args=None):
       rclpy.init(args=args)
       processor = SensorProcessor()
       rclpy.spin(processor)
       processor.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 10: Test Sensor Integration

1. Run the sensor processor node:
   ```bash
   ros2 run physical_ai_robotics sensor_processor
   ```

2. Move the robot in Gazebo and observe how it processes sensor data.

## Exercise 11: Sensor Calibration and Validation

1. Create a calibration node to validate sensor data:
   ```python
   # Create a node that validates sensor readings
   # Check for expected ranges, data validity, etc.
   ```

2. Test sensors in various environments and conditions.

## Verification Steps

1. Confirm that all sensors publish data on appropriate topics
2. Verify that sensor data is realistic and accurate
3. Check that sensor processing nodes receive and handle data correctly
4. Ensure that visualization tools display sensor data properly

## Expected Outcomes

- Understanding of how to add sensors to URDF models
- Knowledge of Gazebo sensor plugins and configuration
- Experience with sensor data processing in ROS 2
- Ability to integrate multiple sensors into a robot system

## Troubleshooting

- If sensors don't publish data, check Gazebo plugin configuration
- If data seems incorrect, verify sensor parameters in URDF
- If nodes can't subscribe to sensor topics, check topic names and permissions

## Next Steps

After completing these exercises, proceed to the Unity environment simulation exercises to understand high-fidelity visualization in robotics applications.