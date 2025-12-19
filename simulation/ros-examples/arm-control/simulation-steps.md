# Mini-Project: Controlling Humanoid Arm Simulation Steps

This guide provides step-by-step instructions for simulating the humanoid arm control mini-project covered in Module 1 of the Physical AI & Humanoid Robotics textbook.

## Overview

This mini-project integrates all concepts learned in Module 1 to control a humanoid arm using ROS 2. You'll create a complete system that includes URDF modeling, joint control, sensor feedback, and visualization.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- Completed all previous simulation exercises (ROS 2 basics, nodes, topics, services, URDF, RViz)
- Basic understanding of inverse kinematics (optional but helpful)

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

## Exercise 1: Complete Humanoid Arm URDF

1. Create a complete humanoid arm URDF with all necessary joints:
   ```xml
   <?xml version="1.0"?>
   <robot name="humanoid_arm">
     <!-- Shoulder base -->
     <link name="shoulder_base">
       <visual>
         <geometry>
           <cylinder length="0.1" radius="0.05"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.1" radius="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.5"/>
         <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
       </inertial>
     </link>

     <!-- Shoulder joint -->
     <joint name="shoulder_yaw" type="revolute">
       <parent link="shoulder_base"/>
       <child link="upper_arm"/>
       <origin xyz="0 0 0.05"/>
       <axis xyz="0 0 1"/>
       <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
     </joint>

     <!-- Upper arm -->
     <link name="upper_arm">
       <visual>
         <geometry>
           <cylinder length="0.3" radius="0.04"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.3" radius="0.04"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <!-- Elbow joint -->
     <joint name="elbow_pitch" type="revolute">
       <parent link="upper_arm"/>
       <child link="forearm"/>
       <origin xyz="0 0 -0.3"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
     </joint>

     <!-- Forearm -->
     <link name="forearm">
       <visual>
         <geometry>
           <cylinder length="0.25" radius="0.035"/>
         </geometry>
         <material name="green">
           <color rgba="0 1 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.25" radius="0.035"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.8"/>
         <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
       </inertial>
     </link>

     <!-- Wrist joint -->
     <joint name="wrist_roll" type="revolute">
       <parent link="forearm"/>
       <child link="hand"/>
       <origin xyz="0 0 -0.25"/>
       <axis xyz="0 0 1"/>
       <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
     </joint>

     <!-- Hand -->
     <link name="hand">
       <visual>
         <geometry>
           <box size="0.1 0.08 0.05"/>
         </geometry>
         <material name="yellow">
           <color rgba="1 1 0 0.8"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.1 0.08 0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.3"/>
         <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
       </inertial>
     </link>
   </robot>
   ```

2. Save as `src/physical_ai_robotics/urdf/humanoid_arm.urdf`.

## Exercise 2: Create Joint Controller Node

1. Create a joint controller node (`src/physical_ai_robotics/arm_controller.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import Float64MultiArray
   from sensor_msgs.msg import JointState
   import math

   class ArmController(Node):
       def __init__(self):
           super().__init__('arm_controller')

           # Publisher for joint commands
           self.joint_cmd_publisher = self.create_publisher(
               Float64MultiArray, '/joint_commands', 10
           )

           # Subscriber for joint states
           self.joint_state_subscriber = self.create_subscription(
               JointState, '/joint_states', self.joint_state_callback, 10
           )

           # Timer for control loop
           self.timer = self.create_timer(0.1, self.control_loop)

           # Target positions for joints [shoulder_yaw, elbow_pitch, wrist_roll]
           self.target_positions = [0.0, 0.0, 0.0]
           self.current_positions = [0.0, 0.0, 0.0]

           self.get_logger().info('Humanoid Arm Controller initialized')

       def joint_state_callback(self, msg):
           """Callback for joint state messages"""
           for i, name in enumerate(msg.name):
               if name == 'shoulder_yaw':
                   self.current_positions[0] = msg.position[i]
               elif name == 'elbow_pitch':
                   self.current_positions[1] = msg.position[i]
               elif name == 'wrist_roll':
                   self.current_positions[2] = msg.position[i]

       def control_loop(self):
           """Main control loop - send joint commands"""
           cmd_msg = Float64MultiArray()

           # Simple trajectory: move to target positions
           cmd_msg.data = self.target_positions
           self.joint_cmd_publisher.publish(cmd_msg)

       def set_target_positions(self, shoulder_yaw, elbow_pitch, wrist_roll):
           """Set target joint positions"""
           self.target_positions = [shoulder_yaw, elbow_pitch, wrist_roll]
           self.get_logger().info(f'Set targets: {self.target_positions}')

   def main(args=None):
       rclpy.init(args=args)
       controller = ArmController()

       # Example: Set a target position
       controller.set_target_positions(0.5, -0.3, 0.2)

       rclpy.spin(controller)
       controller.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. Update the setup.py file to include this new executable.

## Exercise 3: Create Trajectory Generator

1. Create a trajectory generator node (`src/physical_ai_robotics/trajectory_generator.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import Float64MultiArray
   import math
   import time

   class TrajectoryGenerator(Node):
       def __init__(self):
           super().__init__('trajectory_generator')

           self.cmd_publisher = self.create_publisher(
               Float64MultiArray, '/joint_commands', 10
           )

           # Timer for trajectory generation
           self.timer = self.create_timer(0.05, self.generate_trajectory)

           self.time_counter = 0.0
           self.get_logger().info('Trajectory Generator initialized')

       def generate_trajectory(self):
           """Generate smooth trajectories for arm joints"""
           t = self.time_counter

           # Create interesting movement patterns
           shoulder_pos = 0.5 * math.sin(0.5 * t)
           elbow_pos = 0.3 * math.sin(0.7 * t + math.pi/4)
           wrist_pos = 0.4 * math.sin(0.9 * t + math.pi/2)

           cmd_msg = Float64MultiArray()
           cmd_msg.data = [shoulder_pos, elbow_pos, wrist_pos]

           self.cmd_publisher.publish(cmd_msg)
           self.time_counter += 0.05

   def main(args=None):
       rclpy.init(args=args)
       generator = TrajectoryGenerator()
       rclpy.spin(generator)
       generator.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 4: Launch Complete System

1. Create a launch file for the complete arm system (`launch/arm_system.launch.py`):
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

           # Joint state publisher
           Node(
               package='joint_state_publisher',
               executable='joint_state_publisher',
               name='joint_state_publisher',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Arm controller
           Node(
               package='physical_ai_robotics',
               executable='arm_controller',
               name='arm_controller',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Trajectory generator
           Node(
               package='physical_ai_robotics',
               executable='trajectory_generator',
               name='trajectory_generator',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           )
       ])
   ```

## Exercise 5: Visualize in RViz

1. Launch the complete arm system:
   ```bash
   ros2 launch physical_ai_robotics arm_system.launch.py
   ```

2. In another terminal, start RViz:
   ```bash
   rviz2
   ```

3. Add RobotModel display and set robot description
4. Add TF display to visualize coordinate frames
5. Observe the arm moving according to the generated trajectories

## Exercise 6: Add Sensor Feedback

1. Create a sensor feedback node that publishes end-effector position:
   ```python
   # Create a node that calculates and publishes end-effector position
   # based on current joint angles using forward kinematics
   ```

2. Add this information to RViz visualization.

## Exercise 7: Implement Simple Inverse Kinematics (Optional)

1. Create a simple inverse kinematics solver:
   ```python
   # Create a node that takes desired end-effector position
   # and calculates required joint angles
   ```

2. Test the IK solver with various target positions.

## Exercise 8: Add User Interface

1. Create a simple GUI or command-line interface to control the arm:
   ```python
   # Create a node that accepts user input to set target positions
   # Use ROS services or parameters to send commands
   ```

## Exercise 9: Implement Safety Features

1. Add joint limit checking and safety constraints:
   ```python
   # Create a safety node that monitors joint positions
   # and prevents dangerous movements
   ```

## Exercise 10: Integration Testing

1. Test the complete system:
   - Verify all joints move as expected
   - Check that visualization updates correctly
   - Ensure safety limits are respected
   - Validate sensor feedback accuracy

2. Create a comprehensive test script that exercises all functionality.

## Verification Steps

1. Confirm that the arm model displays correctly in RViz
2. Verify that joint controllers respond to commands
3. Check that trajectories execute smoothly
4. Ensure sensor feedback works properly

## Expected Outcomes

- Complete integration of URDF, control, and visualization
- Understanding of practical robot control challenges
- Experience with real-time control systems
- Ability to debug complex ROS 2 systems

## Troubleshooting

- If joints don't move, check joint state publisher and controllers
- If visualization is incorrect, verify URDF joint definitions
- If trajectories are jerky, adjust control loop timing

## Next Steps

After completing this mini-project, you have successfully integrated all Module 1 concepts. Proceed to Module 2 to learn about digital twin creation with Gazebo and Unity.