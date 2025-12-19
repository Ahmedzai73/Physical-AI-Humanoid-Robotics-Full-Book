# ROS 2 Architecture Simulation Steps

This guide provides step-by-step instructions for simulating the ROS 2 architecture concepts covered in Module 1 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates the core ROS 2 architecture concepts:
- Nodes and their communication
- DDS (Data Distribution Service) middleware
- Topics, services, and actions
- Parameter server functionality

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- Basic understanding of ROS 2 concepts
- Docusaurus server running (for reference to textbook content)

## Simulation Environment Setup

1. Open a terminal and navigate to the ROS workspace:
   ```bash
   cd simulation/ros-workspace
   ```

2. Source the ROS 2 installation:
   ```bash
   source /opt/ros/humble/setup.bash  # Adjust for your ROS 2 distribution
   ```

3. Source the workspace:
   ```bash
   colcon build --packages-select physical_ai_robotics
   source install/setup.bash
   ```

## Exercise 1: Basic Node Communication

1. Launch the basic publisher and subscriber nodes:
   ```bash
   ros2 run physical_ai_robotics minimal_publisher
   ```
   In another terminal:
   ```bash
   ros2 run physical_ai_robotics minimal_subscriber
   ```

2. Observe the communication pattern between nodes.

3. Use the following commands to inspect the system:
   ```bash
   # List active nodes
   ros2 node list

   # List topics
   ros2 topic list

   # Echo the topic to see messages
   ros2 topic echo /physical_ai_robotics/topic std_msgs/msg/String
   ```

## Exercise 2: Understanding DDS Middleware

1. Launch the robot controller node:
   ```bash
   ros2 run physical_ai_robotics robot_controller
   ```

2. Monitor the DDS communication:
   ```bash
   # List all active topics and their types
   ros2 topic list -t

   # Check the quality of service settings
   ros2 topic info /cmd_vel --verbose
   ```

3. Observe how DDS handles message delivery between nodes.

## Exercise 3: Services and Actions

1. Create a simple service client and server (you can create these following the textbook examples):
   ```bash
   # In the physical_ai_robotics package, create service files and nodes
   # This demonstrates request-response communication
   ```

2. Test the service communication:
   ```bash
   # Call the service from command line
   ros2 service call /your_service_name your_package/srv/YourServiceType
   ```

## Exercise 4: Parameters and Launch Files

1. Create a launch file that demonstrates parameter usage:
   ```bash
   ros2 launch physical_ai_robotics physical_ai_robotics.launch.py
   ```

2. Interact with parameters:
   ```bash
   # List parameters of a node
   ros2 param list

   # Get a specific parameter
   ros2 param get /minimal_publisher use_sim_time

   # Set a parameter
   ros2 param set /minimal_publisher new_param 42
   ```

## Verification Steps

1. Confirm that all nodes are communicating properly
2. Verify that topics are publishing and subscribing correctly
3. Check that services are responding to requests
4. Ensure parameters are being shared between nodes

## Expected Outcomes

- Understanding of ROS 2 node communication patterns
- Knowledge of how DDS middleware facilitates communication
- Experience with topics, services, and actions
- Familiarity with parameter management

## Troubleshooting

- If nodes fail to communicate, check that they are on the same ROS domain
- Ensure that topic names match exactly between publishers and subscribers
- Verify that message types are compatible between nodes

## Next Steps

After completing these exercises, proceed to the next module on Python agent integration with ROS 2.