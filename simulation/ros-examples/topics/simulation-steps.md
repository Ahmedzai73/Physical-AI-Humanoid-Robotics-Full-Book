# ROS 2 Topics Simulation Steps

This guide provides step-by-step instructions for simulating ROS 2 topic concepts covered in Module 1 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates the publisher-subscriber communication pattern using ROS 2 topics, including message types, Quality of Service (QoS) settings, and practical examples.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- Completed the ROS 2 nodes simulation steps
- Basic understanding of message types and communication patterns

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

## Exercise 1: Basic Topic Communication

1. Launch the minimal publisher node in one terminal:
   ```bash
   ros2 run physical_ai_robotics minimal_publisher
   ```

2. Launch the minimal subscriber node in another terminal:
   ```bash
   ros2 run physical_ai_robotics minimal_subscriber
   ```

3. Observe the publisher-subscriber communication pattern in the terminal outputs.

## Exercise 2: Topic Inspection and Monitoring

1. List all active topics:
   ```bash
   ros2 topic list
   ```

2. Get detailed information about a specific topic:
   ```bash
   ros2 topic info /physical_ai_robotics/topic
   ```

3. Monitor the message rate on the topic:
   ```bash
   ros2 topic hz /physical_ai_robotics/topic
   ```

4. View the actual messages being published:
   ```bash
   ros2 topic echo /physical_ai_robotics/topic
   ```

## Exercise 3: Different Message Types

1. Explore different standard message types:
   ```bash
   # List available message types
   ros2 interface list | grep std_msgs

   # Get information about a specific message type
   ros2 interface show std_msgs/msg/String
   ```

2. Create nodes that use different message types (geometry_msgs, sensor_msgs, etc.) following textbook examples.

## Exercise 4: Quality of Service (QoS) Settings

1. Test different QoS profiles:
   ```bash
   # Run publisher with reliable delivery
   ros2 run physical_ai_robotics minimal_publisher --ros-args -p qos_reliability:=reliable

   # Run subscriber with matching QoS
   ros2 run physical_ai_robotics minimal_subscriber --ros-args -p qos_reliability:=reliable
   ```

2. Observe how different QoS settings affect communication.

## Exercise 5: Topic Remapping

1. Use topic remapping to change topic names:
   ```bash
   ros2 run physical_ai_robotics minimal_publisher --ros-args -r __ns:=/robot1
   ros2 run physical_ai_robotics minimal_subscriber --ros-args -r __ns:=/robot1
   ```

2. Test communication with remapped topics.

## Exercise 6: Multiple Publishers and Subscribers

1. Launch multiple publisher nodes:
   ```bash
   # Terminal 1
   ros2 run physical_ai_robotics minimal_publisher

   # Terminal 2
   ros2 run physical_ai_robotics minimal_publisher
   ```

2. Launch a single subscriber to receive from multiple publishers:
   ```bash
   # Terminal 3
   ros2 run physical_ai_robotics minimal_subscriber
   ```

3. Observe how the subscriber receives messages from all publishers.

## Exercise 7: Custom Message Types

1. Create and use custom message types following the textbook examples:
   ```bash
   # Define custom message in msg/ directory
   # Build the package with colcon
   # Use the custom message in nodes
   ```

## Exercise 8: Topic Performance Analysis

1. Analyze topic performance using ROS 2 tools:
   ```bash
   # Monitor topic latency
   ros2 topic delay /physical_ai_robotics/topic

   # Monitor topic bandwidth
   ros2 topic bw /physical_ai_robotics/topic
   ```

2. Record topic data for analysis:
   ```bash
   # Start recording
   ros2 bag record /physical_ai_robotics/topic

   # Play back recorded data
   ros2 bag play your_recording_folder
   ```

## Verification Steps

1. Confirm that publishers and subscribers can communicate
2. Verify that different QoS settings work as expected
3. Check that topic remapping functions properly
4. Ensure that multiple publishers can send to one subscriber

## Expected Outcomes

- Understanding of the publisher-subscriber communication pattern
- Knowledge of Quality of Service settings and their impact
- Experience with topic monitoring and analysis tools
- Ability to work with different message types

## Troubleshooting

- If topics don't connect, check that message types match between publisher and subscriber
- If communication is unreliable, try adjusting QoS settings
- If topics don't appear in list, ensure nodes are running and connected to the same ROS domain

## Next Steps

After completing these exercises, proceed to the services and actions simulation exercises to understand request-response communication patterns.