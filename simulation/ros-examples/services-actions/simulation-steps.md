# ROS 2 Services and Actions Simulation Steps

This guide provides step-by-step instructions for simulating ROS 2 services and actions concepts covered in Module 1 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates the request-response communication pattern using ROS 2 services and the goal-result-feedback pattern using ROS 2 actions. These are essential for robotics applications requiring guaranteed communication and long-running tasks.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- Completed the ROS 2 topics simulation steps
- Understanding of message types and communication patterns

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

## Exercise 1: Basic Service Communication

1. Create a simple service server and client (following textbook examples):
   ```bash
   # Terminal 1 - Start a service server
   ros2 run your_package your_service_server
   ```

2. In another terminal, call the service:
   ```bash
   # Terminal 2 - Call the service
   ros2 service call /your_service_name your_package/srv/YourServiceType
   ```

3. Observe the request-response communication pattern.

## Exercise 2: Service Inspection and Monitoring

1. List all available services:
   ```bash
   ros2 service list
   ```

2. Get detailed information about a specific service:
   ```bash
   ros2 service info /your_service_name
   ```

3. Get information about the service type:
   ```bash
   ros2 interface show your_package/srv/YourServiceType
   ```

## Exercise 3: Creating Custom Services

1. Define a custom service type in your package:
   ```bash
   # Create a srv/YourService.srv file with request and response definitions
   # Example:
   # string request_message
   # ---
   # string response_message
   # int32 result_code
   ```

2. Build the package with the new service:
   ```bash
   colcon build --packages-select your_package
   source install/setup.bash
   ```

3. Implement the service server and client using the custom service type.

## Exercise 4: Basic Action Communication

1. Create an action server and client (following textbook examples):
   ```bash
   # Terminal 1 - Start an action server
   ros2 run your_package your_action_server
   ```

2. In another terminal, send an action goal:
   ```bash
   # Terminal 2 - Send an action goal
   ros2 action send_goal /your_action_name your_package/action/YourActionType
   ```

3. Observe the goal-result-feedback communication pattern.

## Exercise 5: Action Monitoring and Feedback

1. Monitor action status:
   ```bash
   # List active actions
   ros2 action list

   # Get information about a specific action
   ros2 action info /your_action_name
   ```

2. Monitor action feedback:
   ```bash
   # Monitor feedback during action execution
   ros2 action send_goal /your_action_name your_package/action/YourActionType '{goal_field: value}' --feedback
   ```

## Exercise 6: Complex Action with Feedback

1. Implement an action that provides continuous feedback:
   ```bash
   # Create an action that reports progress during execution
   # Example: navigation action that reports current position
   ```

2. Test the action with feedback monitoring:
   ```bash
   ros2 action send_goal /navigation_action your_package/action/NavigateToPose '{target_pose: ...}' --feedback
   ```

## Exercise 7: Service vs Action Comparison

1. Compare service and action behavior:
   ```bash
   # Test a service for quick request-response
   time ros2 service call /quick_service_name your_package/srv/QuickServiceType

   # Test an action for long-running task
   ros2 action send_goal /long_action_name your_package/action/LongActionType
   ```

2. Observe the differences in communication patterns and use cases.

## Exercise 8: Error Handling

1. Test error handling in services:
   ```bash
   # Create service calls that trigger error conditions
   # Observe how errors are handled and reported
   ```

2. Test error handling in actions:
   ```bash
   # Send action goals that should fail
   # Cancel running actions to test cancellation
   ros2 action send_goal /test_action your_package/action/TestActionType '{goal_field: value}'
   # In another terminal: ros2 action cancel /test_action
   ```

## Exercise 9: Integration with Robot Control

1. Use services and actions for robot control:
   ```bash
   # Service for immediate robot commands (e.g., emergency stop)
   ros2 service call /emergency_stop std_srvs/srv/Trigger

   # Action for long-running robot tasks (e.g., navigation)
   ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose '{...}'
   ```

## Verification Steps

1. Confirm that services respond to requests properly
2. Verify that actions execute goals and provide feedback
3. Check that error handling works for both services and actions
4. Ensure that action cancellation functions correctly

## Expected Outcomes

- Understanding of request-response communication with services
- Knowledge of goal-result-feedback pattern with actions
- Experience with custom service and action types
- Ability to choose appropriate communication pattern for different use cases

## Troubleshooting

- If services don't respond, check that service types match between client and server
- If actions don't start, ensure action types are properly defined and built
- If feedback isn't received, verify that the action server is publishing feedback

## Next Steps

After completing these exercises, proceed to the parameters and launch files simulation exercises to understand configuration management in ROS 2.