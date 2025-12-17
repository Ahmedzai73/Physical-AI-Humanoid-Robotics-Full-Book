---
title: Module 1 Summary - MCQs and Exercises
sidebar_position: 12
description: Comprehensive summary of ROS 2 fundamentals with review questions and exercises
---

# Module 1 Summary + MCQs + Exercises

## Introduction

Congratulations! You've completed Module 1: The Robotic Nervous System (ROS 2). In this module, you've learned the fundamental concepts of ROS 2, which serves as the middleware nervous system of humanoid robots. You've explored nodes, topics, services, actions, parameters, launch files, URDF, and RViz - all essential components for building complex robotic systems.

This chapter provides a comprehensive summary of the key concepts, multiple-choice questions (MCQs) to test your understanding, hands-on exercises to reinforce your skills, and a bridge to the next module.

## Key Concepts Review

### 1. ROS 2 Architecture & DDS
- **Middleware**: ROS 2 uses DDS (Data Distribution Service) as its communication middleware
- **Nodes**: The basic computational units of a ROS 2 system
- **Graph**: The network of nodes and their connections
- **RMW**: ROS Middleware Interface abstracts different DDS implementations

### 2. Nodes
- **Definition**: Basic execution units that perform computation
- **Lifecycle**: Unconfigured → Inactive → Active → Finalized
- **Composition**: Multiple nodes can run in the same process
- **Parameters**: Runtime configuration without recompilation

### 3. Topics (Publish-Subscribe)
- **Asynchronous**: Publishers and subscribers operate independently
- **Decoupled**: Publishers don't know who subscribes, subscribers don't know who publishes
- **QoS**: Quality of Service settings control reliability, durability, and performance
- **Message Types**: Standardized formats for data exchange

### 4. Services & Actions
- **Services**: Synchronous request-response communication
- **Actions**: Long-running tasks with feedback and cancellation
- **Use Cases**: Services for simple queries, Actions for complex behaviors

### 5. Parameters
- **Runtime Configuration**: Change behavior without restarting nodes
- **Types**: bool, int, double, string, arrays
- **Declaration**: Done in nodes using `declare_parameter()`
- **Management**: Can be loaded from YAML files

### 6. Launch Files
- **System Management**: Start multiple nodes with coordinated configurations
- **Arguments**: Customize behavior at launch time
- **Conditions**: Enable/disable components based on conditions
- **Organization**: Group related functionality

### 7. URDF (Unified Robot Description Format)
- **XML Format**: Describes robot structure and properties
- **Links**: Rigid bodies representing robot parts
- **Joints**: Connections between links with defined motion
- **Visual/Collision**: Define appearance and collision properties
- **Inertial**: Physical properties for simulation

### 8. RViz Visualization
- **3D Visualization**: Shows robot models, sensors, and states
- **Displays**: Various types for different data visualization
- **TF**: Visualizes coordinate frame relationships
- **Interactive Markers**: Direct manipulation of robot poses

## Multiple Choice Questions (MCQs)

### 1. What does DDS stand for in the context of ROS 2?
A) Distributed Data System
B) Data Distribution Service
C) Dynamic Discovery System
D) Device Driver Service

**Answer**: B) Data Distribution Service

### 2. Which of the following is NOT a valid QoS reliability policy?
A) RELIABLE
B) BEST_EFFORT
C) GUARANTEED
D) All of the above are valid

**Answer**: C) GUARANTEED

### 3. What is the main difference between a Service and an Action in ROS 2?
A) Services are asynchronous, Actions are synchronous
B) Actions are for long-running tasks with feedback, Services are for immediate responses
C) Services can be cancelled, Actions cannot
D) There is no difference

**Answer**: B) Actions are for long-running tasks with feedback, Services are for immediate responses

### 4. Which URDF element defines the physical properties for simulation?
A) `<visual>`
B) `<collision>`
C) `<inertial>`
D) `<geometry>`

**Answer**: C) `<inertial>`

### 5. What does the `robot_state_publisher` node do?
A) Publishes joint states from encoders
B) Publishes TF transforms from URDF and joint states
C) Controls robot joints
D) Visualizes the robot in RViz

**Answer**: B) Publishes TF transforms from URDF and joint states

### 6. In ROS 2, what is the purpose of launch files?
A) Compile ROS packages
B) Start multiple nodes with coordinated configurations
C) Visualize robot models
D) Store parameter values

**Answer**: B) Start multiple nodes with coordinated configurations

### 7. Which command-line tool is used to visualize the TF tree?
A) `ros2 run tf2_tools view_frames`
B) `ros2 topic echo /tf`
C) `ros2 run rqt_tf_tree rqt_tf_tree`
D) Both A and C

**Answer**: D) Both A and C

### 8. What is the default QoS history policy for publishers?
A) KEEP_ALL
B) KEEP_LAST
C) VOLATILE
D) TRANSIENT_LOCAL

**Answer**: B) KEEP_LAST

### 9. Which of these is NOT a valid joint type in URDF?
A) revolute
B) continuous
C) prismatic
D) rotational

**Answer**: D) rotational

### 10. What is the purpose of the `joint_state_publisher`?
A) Publish commands to move joints
B) Publish current joint positions and velocities
C) Calibrate joint encoders
D) Plan joint trajectories

**Answer**: B) Publish current joint positions and velocities

## Hands-on Exercises

### Exercise 1: Create a Simple Publisher-Subscriber System

Create two nodes:
1. A publisher that publishes "Hello, ROS 2!" every 2 seconds
2. A subscriber that receives the message and prints it to the console

**Requirements:**
- Use the `std_msgs/String` message type
- Name the topic `/greetings`
- Include proper error handling
- Add a parameter to customize the message content

### Exercise 2: Implement a Service Server

Create a service that calculates the factorial of a number:
- Service type: `std_srvs/srv/SetInt` (input: an integer, output: success and message)
- The service should calculate the factorial of the input number
- Return an error if the number is negative or too large (>20)
- Create a client node that calls the service with different numbers

### Exercise 3: Build a Simple Robot Model

Create a URDF for a simple wheeled robot:
- A cylindrical base (wheelbase)
- Two cylindrical wheels
- Use appropriate materials and colors
- Add proper inertial properties
- Validate the URDF and visualize in RViz

### Exercise 4: Create a Parameter-Configured Node

Create a node that:
- Accepts at least 3 different parameters (e.g., robot_name, max_speed, operation_mode)
- Validates parameter values
- Changes behavior based on parameters
- Logs parameter values at startup
- Handles parameter changes at runtime

### Exercise 5: Design a Launch File System

Create a launch file that:
- Starts 3 different nodes with specific configurations
- Uses launch arguments to customize behavior
- Conditionally starts one node based on an argument
- Loads parameters from a YAML file
- Sets up proper namespaces for the nodes

### Exercise 6: Action Server Implementation

Implement an action that:
- Moves a simulated robot to a target position (x, y)
- Provides feedback on progress (distance to goal, estimated time)
- Supports cancellation
- Reports success/failure in the result

### Exercise 7: URDF with Xacro

Create a URDF using Xacro that:
- Defines macros for repeated elements (e.g., wheels, sensors)
- Uses properties for dimensions and materials
- Includes a robot with at least 6 joints
- Generates the URDF from the Xacro file

### Exercise 8: RViz Configuration

Create an RViz configuration file that displays:
- Robot model (RobotModel display)
- Grid for reference
- TF frames with arrows
- Custom markers for goals or waypoints
- Set appropriate fixed frame and view angles

## Programming Challenges

### Challenge 1: Robot Arm Controller
Design and implement a complete system for controlling a 3-DOF robot arm:
- URDF model of the arm
- Joint trajectory publisher
- Inverse kinematics calculator (simple geometric solution)
- RViz visualization
- Keyboard/command interface

### Challenge 2: Multi-Robot Communication
Create a system with 2 robot nodes that:
- Share sensor data via topics
- Coordinate using services
- Use parameters to identify each robot
- Include a coordinator node that manages both robots
- Visualize both robots in RViz

### Challenge 3: Adaptive Parameter Tuning
Create a system that:
- Uses parameters for control gains
- Monitors system performance
- Adjusts parameters based on performance metrics
- Includes a learning algorithm to optimize parameters
- Logs parameter changes and performance over time

## Review Questions

### Conceptual Questions
1. Explain the publish-subscribe pattern and its advantages in robotic systems.
2. Compare and contrast services, actions, and topics. When would you use each?
3. Describe the ROS 2 node lifecycle and why it's important for robot systems.
4. What are QoS policies and why are they crucial for robotic applications?
5. Explain the relationship between URDF, TF, and robot visualization.

### Application Questions
1. Design a ROS 2 architecture for a mobile robot with camera, LiDAR, and wheel encoders.
2. How would you organize nodes for a humanoid robot with 20+ joints?
3. What QoS policies would you use for different types of robot data (e.g., sensor data, control commands)?
4. Design a parameter system for configuring different robot types from a single codebase.
5. How would you structure launch files for a complex robotic system with multiple operational modes?

### Troubleshooting Questions
1. How would you debug a situation where a subscriber is not receiving messages?
2. What steps would you take if a robot model appears incorrectly in RViz?
3. How would you investigate why a service call is timing out?
4. What tools would you use to find why a node is not starting properly?
5. How would you identify and fix TF tree issues in a multi-robot system?

## Module 1 Achievement Check

By completing this module, you should be able to:
- [ ] Create and run basic ROS 2 nodes in Python
- [ ] Implement publish-subscribe communication patterns
- [ ] Use services and actions appropriately
- [ ] Configure nodes using parameters
- [ ] Create and use launch files for system management
- [ ] Define robot models using URDF
- [ ] Visualize robots and data in RViz
- [ ] Design modular, maintainable ROS 2 systems
- [ ] Apply best practices for robotic software development

## Bridge to Module 2

Module 1 has established the foundation for robotic software development with ROS 2. Now that you understand the "nervous system" of robots, Module 2 will explore how to create digital twins using simulation environments like Gazebo and Unity.

In Module 2, you'll learn to:
- Import your URDF models into physics simulation
- Simulate sensors and perception systems
- Create realistic environments for testing
- Connect simulation to your ROS 2 nodes
- Build interactive 3D visualizations

The concepts from Module 1 (nodes, topics, services, parameters, URDF) will be essential as you create digital twins that mirror the behavior of real robots in simulation environments.

## Resources for Further Learning

### Official Documentation
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)
- [RViz User Guide](http://wiki.ros.org/rviz/UserGuide)

### Recommended Reading
- "Programming Robots with ROS" by Morgan Quigley
- "Effective Robotics Programming with ROS" by Anil Mahtani
- "Mastering ROS for Robotics Programming" by Josep Casas

### Practice Projects
1. Build a teleoperation system for a simulated robot
2. Implement a simple navigation stack
3. Create a robot with multiple sensors and fusion
4. Develop a multi-robot coordination system
5. Build a complete robot application with GUI

## Summary

Module 1 has equipped you with the essential skills for ROS 2 development:
- Understanding the architecture and communication patterns
- Creating modular, maintainable robotic systems
- Modeling robots with URDF
- Visualizing and debugging with RViz
- Managing complex systems with launch files

These skills form the foundation for all advanced robotic development. Take time to practice these concepts with the exercises provided, as they will be essential as you advance to more complex topics in Modules 2-4.

Remember: Robotics software development is an iterative process. Start simple, test frequently, and gradually add complexity. The modular nature of ROS 2 allows you to develop and test components independently before integrating them into complete systems.

## Next Steps

Proceed to Module 2: [The Digital Twin (Gazebo & Unity)](../module-2-digital-twin/intro.md) where you'll learn to create physics simulations and digital twins for your robotic systems. The URDF models you've learned to create will come alive in simulation environments!

---

*This concludes Module 1: The Robotic Nervous System (ROS 2). Congratulations on completing the foundational module of the Physical AI & Humanoid Robotics series!*