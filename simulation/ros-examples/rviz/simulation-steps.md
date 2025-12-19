# RViz Visualization Tutorial Simulation Steps

This guide provides step-by-step instructions for simulating RViz visualization concepts covered in Module 1 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to use RViz2 (Robot Visualizer) to visualize robot models, sensor data, and navigation information. RViz is essential for debugging and understanding robot behavior in ROS 2 systems.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- RViz2 installed
- Completed the URDF fundamentals simulation steps
- Basic understanding of robot sensors and coordinate frames

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

## Exercise 1: Launching RViz with Robot Model

1. Start the robot state publisher with your humanoid model:
   ```bash
   # Terminal 1: Launch robot state publisher
   ros2 launch physical_ai_robotics humanoid.launch.py
   ```

2. In another terminal, start RViz:
   ```bash
   rviz2
   ```

3. In RViz, add a RobotModel display:
   - Click "Add" in the Displays panel
   - Select "RobotModel" under "Robot Models"
   - Set the "Robot Description" to "/robot_description"

## Exercise 2: Setting Up Coordinate Frames

1. In RViz, set the "Fixed Frame" to "base_link" or "map":
   - In the "Global Options" panel, set "Fixed Frame" to "base_link"

2. Add a TF (Transform) display to visualize coordinate frames:
   - Click "Add" in the Displays panel
   - Select "TF" under "Robot Elements"
   - This shows all coordinate frames and their relationships

3. Observe how the TF tree represents your robot's kinematic structure.

## Exercise 3: Visualizing Sensor Data

1. Create a node that publishes sensor data (e.g., laser scan):
   ```python
   # Create a simple laser scan publisher following textbook examples
   # This simulates data from a LIDAR sensor
   ```

2. Add a LaserScan display in RViz:
   - Click "Add" in the Displays panel
   - Select "LaserScan" under "Sensors"
   - Set "Topic" to "/scan" (or your laser scan topic)

3. Visualize the laser scan data in the 3D view.

## Exercise 4: Adding Point Cloud Visualization

1. Create or simulate point cloud data:
   ```python
   # Create a node that publishes PointCloud2 messages
   # This simulates data from a depth camera or 3D sensor
   ```

2. Add a PointCloud2 display in RViz:
   - Click "Add" in the Displays panel
   - Select "PointCloud2" under "Sensors"
   - Set "Topic" to your point cloud topic

3. Adjust visualization properties like size and coloring.

## Exercise 5: Navigation Visualization

1. Add displays for navigation stack:
   - "Map" display for occupancy grid
   - "Path" display for planned paths
   - "Pose" display for robot pose
   - "Marker" display for custom visualization

2. Set up the "Global Options" with appropriate coordinate frames.

## Exercise 6: Custom Visualization with Markers

1. Create a node that publishes visualization markers:
   ```python
   # Create a node that publishes visualization_msgs/Marker
   # This allows custom shapes, text, and interactive elements
   ```

2. Add a Marker display in RViz:
   - Click "Add" in the Displays panel
   - Select "Marker" under "Visualization"
   - Set "Namespace" to match your marker namespace

## Exercise 7: Camera Image Integration

1. Publish camera images from your robot:
   ```python
   # Create a node that publishes sensor_msgs/Image
   # This simulates data from a robot's camera
   ```

2. Add an Image display in RViz:
   - Click "Add" in the Displays panel
   - Select "Image" under "Image"
   - Set "Image Topic" to your camera image topic

## Exercise 8: Interactive Markers

1. Create interactive markers for robot control:
   ```python
   # Implement interactive markers that allow clicking/dragging
   # in RViz to control the robot
   ```

2. Add an InteractiveMarkers display in RViz:
   - Click "Add" in the Displays panel
   - Select "InteractiveMarkers" under "Interactive"
   - Set "Update Topic" to your interactive marker topic

## Exercise 9: Saving and Loading RViz Configurations

1. Configure your RViz layout with all desired displays:
   - Position displays in the interface
   - Set appropriate parameters for each display
   - Adjust visualization colors and properties

2. Save the configuration:
   - Go to "File" → "Save Config As..."
   - Save as "humanoid_robot.rviz"

3. Load the configuration later:
   - Go to "File" → "Open Config"
   - Select your saved .rviz file

## Exercise 10: Advanced Visualization Techniques

1. Use multiple viewports for different perspectives:
   - Add multiple "Displays" panels
   - Create different camera views (top, side, perspective)

2. Implement custom plugins if needed for specialized visualization.

## Exercise 11: Performance Optimization

1. Optimize visualization for large datasets:
   - Use decimation for large point clouds
   - Limit update rates for high-frequency topics
   - Use appropriate buffer sizes

2. Monitor RViz performance and adjust settings accordingly.

## Verification Steps

1. Confirm that the robot model displays correctly
2. Verify that sensor data visualizes properly
3. Check that coordinate frames are properly aligned
4. Ensure that navigation displays work as expected

## Expected Outcomes

- Understanding of RViz interface and display types
- Knowledge of how to visualize different sensor data
- Experience with coordinate frame management
- Ability to create custom visualizations with markers

## Troubleshooting

- If robot model doesn't appear, check robot_description parameter
- If TF frames don't show, verify transform publisher is running
- If sensor data doesn't display, check topic names and message types

## Next Steps

After completing these exercises, proceed to the mini-project simulation exercises to apply all learned concepts in a practical humanoid arm control application.