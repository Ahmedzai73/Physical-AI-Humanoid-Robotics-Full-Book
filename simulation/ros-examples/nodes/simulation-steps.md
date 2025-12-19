# ROS 2 Node Simulation Steps

This guide provides step-by-step instructions for simulating ROS 2 node concepts covered in Module 1 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to create, run, and manage ROS 2 nodes with practical examples using Python and rclpy.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- Basic understanding of Python programming
- Completed the ROS 2 architecture simulation steps

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

## Exercise 1: Creating and Running Basic Nodes

1. Run the minimal publisher node:
   ```bash
   ros2 run physical_ai_robotics minimal_publisher
   ```

2. In a separate terminal, run the minimal subscriber node:
   ```bash
   ros2 run physical_ai_robotics minimal_subscriber
   ```

3. Observe the communication between nodes in the terminal output.

## Exercise 2: Node Lifecycle Management

1. Create a node with custom parameters:
   ```bash
   # Modify parameters via command line
   ros2 run physical_ai_robotics minimal_publisher --ros-args -p param_name:=value
   ```

2. Use the node management tools:
   ```bash
   # List all running nodes
   ros2 node list

   # Get information about a specific node
   ros2 node info /minimal_publisher
   ```

## Exercise 3: Node Composition

1. Create a composite node that combines publisher and subscriber functionality:
   ```python
   # Example of a node that both publishes and subscribes
   # This can be implemented following the textbook examples
   ```

2. Run the composite node and observe its behavior.

## Exercise 4: Node Quality of Service (QoS)

1. Test different QoS profiles:
   ```bash
   # Launch nodes with specific QoS settings
   ros2 run physical_ai_robotics minimal_publisher --ros-args -p qos_profile:=reliable
   ```

2. Observe how QoS settings affect communication reliability and performance.

## Exercise 5: Node Monitoring and Debugging

1. Monitor node performance:
   ```bash
   # Check the rate of topic publishing
   ros2 topic hz /physical_ai_robotics/topic

   # Monitor message contents
   ros2 topic echo /physical_ai_robotics/topic
   ```

2. Use ROS 2 tools for debugging:
   ```bash
   # Get detailed information about topics
   ros2 topic info /physical_ai_robotics/topic --verbose

   # Visualize the node graph
   rqt_graph
   ```

## Exercise 6: Advanced Node Features

1. Implement a node with services:
   ```bash
   # Create and run a service server node
   ros2 run physical_ai_robotics your_service_server
   ```

2. Test the service from a client:
   ```bash
   # Call the service
   ros2 service call /your_service_name your_package/srv/YourServiceType
   ```

## Verification Steps

1. Confirm that all nodes start without errors
2. Verify that nodes can communicate with each other
3. Check that node parameters are correctly applied
4. Ensure that node lifecycle management works properly

## Expected Outcomes

- Ability to create ROS 2 nodes using Python and rclpy
- Understanding of node lifecycle and management
- Experience with node composition and communication
- Knowledge of QoS profiles and their impact

## Troubleshooting

- If nodes fail to start, check for import errors or missing dependencies
- If nodes can't communicate, verify topic names and message types match
- If parameters don't work, ensure they're declared properly in the node

## Next Steps

After completing these exercises, proceed to the topics simulation exercises to understand pub-sub communication patterns.