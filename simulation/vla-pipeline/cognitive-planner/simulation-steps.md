# LLM-Based Cognitive Planning Simulation Steps

This guide provides step-by-step instructions for implementing an LLM-based cognitive planner that outputs ROS 2 action sequences as covered in Module 4 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to build a reasoning pipeline where an LLM outputs ROS 2 action sequences, mapping natural language commands to specific robotic actions using cognitive planning with safety guardrails.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later)
- OpenAI API access or local LLM (e.g., Ollama)
- Completed voice command processing exercises
- Understanding of ROS 2 action and service concepts

## Simulation Environment Setup

1. Install LLM integration packages:
   ```bash
   pip3 install openai langchain langchain-community
   # Or for local models:
   pip3 install ollama transformers torch
   ```

2. Set up API keys or local model servers as needed

## Exercise 1: Create Cognitive Planning Node

1. Create the cognitive planning node (`src/physical_ai_robotics/cognitive_planner.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from geometry_msgs.msg import PoseStamped
   from nav2_msgs.action import NavigateToPose
   from rclpy.action import ActionClient
   from rclpy.qos import QoSProfile
   import json
   import re
   from typing import Dict, List, Any

   class CognitivePlanner(Node):
       def __init__(self):
           super().__init__('cognitive_planner')

           # Subscriber for natural language commands
           self.command_subscriber = self.create_subscription(
               String, '/natural_language_command', self.command_callback, 10
           )

           # Publisher for plan status
           self.status_publisher = self.create_publisher(String, '/plan_status', 10)

           # Action client for navigation
           self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

           # Define robot capabilities
           self.robot_capabilities = {
               'navigation': ['go to', 'move to', 'navigate to', 'walk to'],
               'object_interaction': ['pick up', 'grasp', 'take', 'get', 'place', 'put'],
               'manipulation': ['move arm', 'lift', 'lower', 'rotate'],
               'perception': ['look at', 'find', 'detect', 'search for'],
               'communication': ['say', 'speak', 'tell', 'announce']
           }

           # Define known locations
           self.known_locations = {
               'kitchen': {'x': 2.0, 'y': 1.0, 'theta': 0.0},
               'bedroom': {'x': -1.0, 'y': 3.0, 'theta': 1.57},
               'living_room': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
               'office': {'x': 3.0, 'y': -2.0, 'theta': 3.14},
               'dining_room': {'x': 1.5, 'y': 2.5, 'theta': -1.57}
           }

           # Define known objects
           self.known_objects = [
               'cup', 'bottle', 'book', 'phone', 'keys', 'apple',
               'banana', 'box', 'toy', 'pen', 'notebook'
           ]

           self.get_logger().info('Cognitive Planner initialized')

       def command_callback(self, msg):
           """Process natural language command and generate action plan"""
           command_text = msg.data
           self.get_logger().info(f'Received command: "{command_text}"')

           # Publish processing status
           status_msg = String()
           status_msg.data = f'Processing command: {command_text}'
           self.status_publisher.publish(status_msg)

           # Parse and plan the command
           plan = self.parse_and_plan_command(command_text)

           if plan:
               self.get_logger().info(f'Generated plan: {plan}')
               status_msg.data = f'Plan generated with {len(plan)} steps'
               self.status_publisher.publish(status_msg)

               # Execute the plan
               self.execute_plan(plan)
           else:
               status_msg.data = f'Could not understand command: {command_text}'
               self.status_publisher.publish(status_msg)

       def parse_and_plan_command(self, command: str) -> List[Dict]:
           """Parse natural language command and generate action plan"""
           command_lower = command.lower()

           # Identify the main action type
           action_type = self.identify_action_type(command_lower)

           if action_type == 'navigation':
               return self.create_navigation_plan(command_lower)
           elif action_type == 'object_interaction':
               return self.create_object_interaction_plan(command_lower)
           elif action_type == 'complex':
               return self.create_complex_plan(command_lower)
           else:
               return self.create_simple_plan(command_lower)

       def identify_action_type(self, command: str) -> str:
           """Identify the type of action from the command"""
           for action_type, keywords in self.robot_capabilities.items():
               for keyword in keywords:
                   if keyword in command:
                       return action_type

           # Check for complex commands that combine multiple actions
           if any(word in command for word in ['then', 'and', 'after', 'before']):
               return 'complex'

           return 'unknown'

       def create_navigation_plan(self, command: str) -> List[Dict]:
           """Create a navigation plan based on the command"""
           plan = []

           # Extract location from command
           target_location = None
           for location in self.known_locations.keys():
               if location in command:
                   target_location = location
                   break

           if target_location:
               # Add navigation action
               nav_action = {
                   'action_type': 'navigation',
                   'target_location': target_location,
                   'x': self.known_locations[target_location]['x'],
                   'y': self.known_locations[target_location]['y'],
                   'theta': self.known_locations[target_location]['theta']
               }
               plan.append(nav_action)
           else:
               # If no known location, try to extract coordinates or use default
               self.get_logger().warn(f'Unknown location in command: {command}')
               # Default to living room
               nav_action = {
                   'action_type': 'navigation',
                   'target_location': 'living_room',
                   'x': self.known_locations['living_room']['x'],
                   'y': self.known_locations['living_room']['y'],
                   'theta': self.known_locations['living_room']['theta']
               }
               plan.append(nav_action)

           return plan

       def create_object_interaction_plan(self, command: str) -> List[Dict]:
           """Create an object interaction plan based on the command"""
           plan = []

           # Extract object from command
           target_object = None
           for obj in self.known_objects:
               if obj in command:
                   target_object = obj
                   break

           if target_object:
               # Add perception action first
               perception_action = {
                   'action_type': 'perception',
                   'action': 'find_object',
                   'target_object': target_object
               }
               plan.append(perception_action)

               # Add navigation to object
               nav_action = {
                   'action_type': 'navigation',
                   'action': 'navigate_to_object',
                   'target_object': target_object
               }
               plan.append(nav_action)

               # Add interaction action
               if 'pick' in command or 'grasp' in command or 'take' in command:
                   interaction_action = {
                       'action_type': 'manipulation',
                       'action': 'pick_object',
                       'target_object': target_object
                   }
                   plan.append(interaction_action)
               elif 'place' in command or 'put' in command:
                   interaction_action = {
                       'action_type': 'manipulation',
                       'action': 'place_object',
                       'target_object': target_object
                   }
                   plan.append(interaction_action)
           else:
               self.get_logger().warn(f'Unknown object in command: {command}')

           return plan

       def create_complex_plan(self, command: str) -> List[Dict]:
           """Create a complex plan with multiple sequential actions"""
           plan = []

           # For complex commands, we'll use a simple approach
           # In a real system, this would involve more sophisticated NLP

           if 'go to kitchen and pick up cup' in command:
               # Navigate to kitchen
               plan.append({
                   'action_type': 'navigation',
                   'target_location': 'kitchen',
                   'x': self.known_locations['kitchen']['x'],
                   'y': self.known_locations['kitchen']['y'],
                   'theta': self.known_locations['kitchen']['theta']
               })

               # Find and pick up cup
               plan.append({
                   'action_type': 'perception',
                   'action': 'find_object',
                   'target_object': 'cup'
               })

               plan.append({
                   'action_type': 'manipulation',
                   'action': 'pick_object',
                   'target_object': 'cup'
               })

           elif 'navigate to bedroom and find keys' in command:
               # Navigate to bedroom
               plan.append({
                   'action_type': 'navigation',
                   'target_location': 'bedroom',
                   'x': self.known_locations['bedroom']['x'],
                   'y': self.known_locations['bedroom']['y'],
                   'theta': self.known_locations['bedroom']['theta']
               })

               # Find keys
               plan.append({
                   'action_type': 'perception',
                   'action': 'find_object',
                   'target_object': 'keys'
               })

           else:
               # Default to simple plan
               return self.create_simple_plan(command)

           return plan

       def create_simple_plan(self, command: str) -> List[Dict]:
           """Create a simple plan for unrecognized commands"""
           plan = []

           # Default action - navigate to living room
           plan.append({
               'action_type': 'navigation',
               'target_location': 'living_room',
               'x': self.known_locations['living_room']['x'],
               'y': self.known_locations['living_room']['y'],
               'theta': self.known_locations['living_room']['theta']
           })

           return plan

       def execute_plan(self, plan: List[Dict]):
           """Execute the generated action plan"""
           self.get_logger().info(f'Executing plan with {len(plan)} steps')

           for i, action in enumerate(plan):
               self.get_logger().info(f'Executing step {i+1}/{len(plan)}: {action["action_type"]}')

               if action['action_type'] == 'navigation':
                   self.execute_navigation(action)
               elif action['action_type'] == 'perception':
                   self.execute_perception(action)
               elif action['action_type'] == 'manipulation':
                   self.execute_manipulation(action)

       def execute_navigation(self, action: Dict):
           """Execute navigation action"""
           if not self.nav_client.wait_for_server(timeout_sec=5.0):
               self.get_logger().error('Navigation action server not available')
               return

           goal_msg = NavigateToPose.Goal()
           goal_msg.pose.header.frame_id = 'map'
           goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
           goal_msg.pose.pose.position.x = action['x']
           goal_msg.pose.pose.position.y = action['y']
           goal_msg.pose.pose.position.z = 0.0

           # Convert theta to quaternion
           import math
           theta = action['theta']
           goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
           goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

           # Send goal
           send_goal_future = self.nav_client.send_goal_async(goal_msg)
           send_goal_future.add_done_callback(self.navigation_goal_response_callback)

       def navigation_goal_response_callback(self, future):
           """Handle navigation goal response"""
           goal_handle = future.result()
           if not goal_handle.accepted:
               self.get_logger().info('Navigation goal rejected')
               return

           self.get_logger().info('Navigation goal accepted, getting result...')
           get_result_future = goal_handle.get_result_async()
           get_result_future.add_done_callback(self.navigation_result_callback)

       def navigation_result_callback(self, future):
           """Handle navigation result"""
           result = future.result().result
           if result:
               self.get_logger().info('Navigation completed successfully')
           else:
               self.get_logger().error('Navigation failed')

       def execute_perception(self, action: Dict):
           """Execute perception action (placeholder)"""
           self.get_logger().info(f'Perception action: {action["action"]}')
           # In a real system, this would call perception services
           # For now, just log the action

       def execute_manipulation(self, action: Dict):
           """Execute manipulation action (placeholder)"""
           self.get_logger().info(f'Manipulation action: {action["action"]}')
           # In a real system, this would call manipulation services
           # For now, just log the action

   def main(args=None):
       rclpy.init(args=args)
       planner = CognitivePlanner()

       try:
           rclpy.spin(planner)
       except KeyboardInterrupt:
           planner.get_logger().info('Cognitive planner stopped by user')
       finally:
           planner.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 2: Integrate with LLM for Advanced Planning

1. Create LLM-enhanced cognitive planner (`src/physical_ai_robotics/llm_cognitive_planner.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from geometry_msgs.msg import PoseStamped
   from nav2_msgs.action import NavigateToPose
   from rclpy.action import ActionClient
   import json
   import openai
   from typing import Dict, List, Any

   class LLMCognitivePlanner(Node):
       def __init__(self):
           super().__init__('llm_cognitive_planner')

           # Subscriber for natural language commands
           self.command_subscriber = self.create_subscription(
               String, '/natural_language_command', self.command_callback, 10
           )

           # Publisher for plan status
           self.status_publisher = self.create_publisher(String, '/plan_status', 10)

           # Action client for navigation
           self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

           # Set up OpenAI client (or use local LLM)
           # openai.api_key = "your-api-key-here"  # Set in environment variable instead
           self.client = openai.OpenAI()

           # Robot context for LLM
           self.robot_context = {
               "capabilities": [
                   "navigation to known locations",
                   "object detection and recognition",
                   "object manipulation (pick/place)",
                   "basic communication"
               ],
               "known_locations": [
                   "kitchen", "bedroom", "living_room", "office", "dining_room"
               ],
               "known_objects": [
                   "cup", "bottle", "book", "phone", "keys", "apple", "banana",
                   "box", "toy", "pen", "notebook"
               ],
               "current_location": "living_room"
           }

           self.get_logger().info('LLM Cognitive Planner initialized')

       def command_callback(self, msg):
           """Process natural language command using LLM"""
           command_text = msg.data
           self.get_logger().info(f'Received command: "{command_text}"')

           # Publish processing status
           status_msg = String()
           status_msg.data = f'Processing with LLM: {command_text}'
           self.status_publisher.publish(status_msg)

           try:
               # Generate plan using LLM
               plan = self.generate_plan_with_llm(command_text)

               if plan:
                   self.get_logger().info(f'LLM generated plan: {plan}')
                   status_msg.data = f'LLM plan generated with {len(plan)} steps'
                   self.status_publisher.publish(status_msg)

                   # Execute the plan
                   self.execute_plan(plan)
               else:
                   status_msg.data = f'LLM could not generate plan for: {command_text}'
                   self.status_publisher.publish(status_msg)

           except Exception as e:
               self.get_logger().error(f'Error generating plan with LLM: {e}')
               status_msg.data = f'LLM error: {str(e)}'
               self.status_publisher.publish(status_msg)

       def generate_plan_with_llm(self, command: str) -> List[Dict]:
           """Generate action plan using LLM"""
           system_prompt = f"""
           You are a cognitive planner for a humanoid robot. Your task is to convert natural language commands into a sequence of executable robot actions.

           Robot capabilities:
           - Navigation: move to specific locations
           - Object interaction: detect, pick up, and place objects
           - Manipulation: basic arm movements
           - Perception: detect objects in the environment

           Known locations: {self.robot_context['known_locations']}
           Known objects: {self.robot_context['known_objects']}

           Output format: Return a JSON list of action objects with the following structure:
           [
             {{
               "action_type": "navigation|perception|manipulation|communication",
               "action": "specific_action_name",
               "target_location": "location_name",  // for navigation
               "target_object": "object_name",     // for object actions
               "x": number, "y": number, "theta": number  // for navigation
             }}
           ]

           Be specific and use only the known locations and objects. If the command is unclear or impossible, return an empty list.
           """

           user_prompt = f"Command: {command}"

           try:
               response = self.client.chat.completions.create(
                   model="gpt-3.5-turbo",  # or gpt-4 for better performance
                   messages=[
                       {"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_prompt}
                   ],
                   temperature=0.1,
                   max_tokens=500,
                   response_format={"type": "json_object"}
               )

               # Extract and parse the response
               response_text = response.choices[0].message.content
               self.get_logger().debug(f'LLM response: {response_text}')

               # Parse the JSON response
               plan_data = json.loads(response_text)
               plan = plan_data.get('plan', [])

               return plan

           except json.JSONDecodeError as e:
               self.get_logger().error(f'Error parsing LLM response: {e}')
               return []
           except Exception as e:
               self.get_logger().error(f'Error calling LLM: {e}')
               return []

       def execute_plan(self, plan: List[Dict]):
           """Execute the generated action plan"""
           self.get_logger().info(f'Executing LLM-generated plan with {len(plan)} steps')

           for i, action in enumerate(plan):
               self.get_logger().info(f'Executing step {i+1}/{len(plan)}: {action}')

               action_type = action.get('action_type', '')
               if action_type == 'navigation':
                   self.execute_navigation(action)
               elif action_type in ['perception', 'object_interaction', 'manipulation']:
                   self.execute_action(action)
               else:
                   self.get_logger().warn(f'Unknown action type: {action_type}')

       def execute_navigation(self, action: Dict):
           """Execute navigation action"""
           if not self.nav_client.wait_for_server(timeout_sec=5.0):
               self.get_logger().error('Navigation action server not available')
               return

           # Use provided coordinates or look up location
           x = action.get('x')
           y = action.get('y')
           theta = action.get('theta')

           if x is None or y is None:
               # Look up coordinates by location name
               location_name = action.get('target_location')
               if location_name in self.robot_context['known_locations']:
                   # In a real system, you'd have actual coordinates
                   # For this example, we'll use dummy coordinates
                   x, y, theta = 0.0, 0.0, 0.0
               else:
                   self.get_logger().error(f'Unknown location: {location_name}')
                   return

           goal_msg = NavigateToPose.Goal()
           goal_msg.pose.header.frame_id = 'map'
           goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
           goal_msg.pose.pose.position.x = x
           goal_msg.pose.pose.position.y = y
           goal_msg.pose.pose.position.z = 0.0

           # Convert theta to quaternion
           import math
           goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
           goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

           # Send goal
           send_goal_future = self.nav_client.send_goal_async(goal_msg)
           send_goal_future.add_done_callback(self.navigation_goal_response_callback)

       def navigation_goal_response_callback(self, future):
           """Handle navigation goal response"""
           goal_handle = future.result()
           if not goal_handle.accepted:
               self.get_logger().info('Navigation goal rejected')
               return

           self.get_logger().info('Navigation goal accepted, getting result...')
           get_result_future = goal_handle.get_result_async()
           get_result_future.add_done_callback(self.navigation_result_callback)

       def navigation_result_callback(self, future):
           """Handle navigation result"""
           result = future.result().result
           if result:
               self.get_logger().info('Navigation completed successfully')
           else:
               self.get_logger().error('Navigation failed')

       def execute_action(self, action: Dict):
           """Execute non-navigation actions (placeholder)"""
           action_name = action.get('action', 'unknown')
           self.get_logger().info(f'Executing action: {action_name}')
           # In a real system, this would call appropriate services

   def main(args=None):
       rclpy.init(args=args)
       planner = LLMCognitivePlanner()

       try:
           rclpy.spin(planner)
       except KeyboardInterrupt:
           planner.get_logger().info('LLM Cognitive planner stopped by user')
       finally:
           planner.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 3: Create Safety Guardrails Node

1. Create safety guardrails node (`src/physical_ai_robotics/safety_guardrails.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from geometry_msgs.msg import Twist
   from sensor_msgs.msg import LaserScan
   from typing import Dict, Any

   class SafetyGuardrails(Node):
       def __init__(self):
           super().__init__('safety_guardrails')

           # Subscriber for planned actions
           self.plan_subscriber = self.create_subscription(
               String, '/planned_action', self.plan_callback, 10
           )

           # Subscriber for sensor data
           self.laser_subscriber = self.create_subscription(
               LaserScan, '/scan', self.laser_callback, 10
           )

           # Publisher for safety-critical commands
           self.safety_publisher = self.create_publisher(Twist, '/safety_cmd', 10)

           # Publisher for safety status
           self.status_publisher = self.create_publisher(String, '/safety_status', 10)

           # Safety parameters
           self.safety_distance = 0.5  # meters
           self.emergency_stop_active = False
           self.obstacle_detected = False

           # Known dangerous commands
           self.dangerous_commands = [
               'self-destruct', 'harm', 'damage', 'break', 'destroy',
               'jump', 'fly', 'go fast', 'collide', 'crash'
           ]

           self.get_logger().info('Safety Guardrails initialized')

       def plan_callback(self, msg):
           """Check planned actions for safety"""
           try:
               plan_data = eval(msg.data)  # In real system, use proper JSON parsing
               action = plan_data if isinstance(plan_data, dict) else {}

               # Check if action is dangerous
               is_dangerous = self.check_dangerous_action(action)

               if is_dangerous:
                   self.get_logger().warn(f'Dangerous action detected: {action}')
                   self.publish_safety_status(f'Dangerous action blocked: {action}')
                   return  # Don't execute dangerous action

               # Check if environment is safe for action
               is_safe = self.check_environment_safety(action)

               if not is_safe:
                   self.get_logger().warn(f'Environment unsafe for action: {action}')
                   self.publish_safety_status(f'Environment unsafe for action: {action}')
                   return  # Don't execute in unsafe environment

               self.publish_safety_status('Action approved')

           except Exception as e:
               self.get_logger().error(f'Error checking action safety: {e}')

       def check_dangerous_action(self, action: Dict) -> bool:
           """Check if action is inherently dangerous"""
           action_text = str(action).lower()

           for dangerous_cmd in self.dangerous_commands:
               if dangerous_cmd in action_text:
                   return True

           # Check for dangerous parameters
           if isinstance(action, dict):
               # Check for unsafe speeds
               linear_speed = action.get('linear_speed', 0)
               angular_speed = action.get('angular_speed', 0)

               if abs(linear_speed) > 2.0 or abs(angular_speed) > 1.5:  # Unsafe speeds
                   return True

           return False

       def check_environment_safety(self, action: Dict) -> bool:
           """Check if environment is safe for the action"""
           if not self.obstacle_detected:
               return True

           # For navigation actions, check path safety
           if action.get('action_type') == 'navigation':
               # In a real system, check the planned path for obstacles
               # For now, return False if obstacles are detected
               return not self.obstacle_detected

           return True

       def laser_callback(self, msg):
           """Process laser scan for obstacle detection"""
           # Check for obstacles in front of robot
           front_ranges = msg.ranges[len(msg.ranges)//2-30:len(msg.ranges)//2+30]
           front_ranges = [r for r in front_ranges if not r != float('inf') and r is not None]

           if front_ranges:
               min_distance = min(front_ranges) if front_ranges else float('inf')
               self.obstacle_detected = min_distance < self.safety_distance
           else:
               self.obstacle_detected = False

       def publish_safety_status(self, status: str):
           """Publish safety status"""
           status_msg = String()
           status_msg.data = status
           self.status_publisher.publish(status_msg)

   def main(args=None):
       rclpy.init(args=args)
       guardrails = SafetyGuardrails()
       rclpy.spin(guardrails)
       guardrails.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 4: Create Cognitive Planning Integration Launch File

1. Create launch file for cognitive planning system (`launch/cognitive_planning_system.launch.py`):
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
               description='Use simulation clock if true'
           ),

           # LLM Cognitive Planner
           Node(
               package='physical_ai_robotics',
               executable='llm_cognitive_planner',
               name='llm_cognitive_planner',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen',
               # Set environment variables if needed
               # arguments=['--ros-args', '-p', 'openai_api_key:=your_key_here']
           ),

           # Safety Guardrails
           Node(
               package='physical_ai_robotics',
               executable='safety_guardrails',
               name='safety_guardrails',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           )
       ])
   ```

## Exercise 5: Test Cognitive Planning System

1. Launch the cognitive planning system:
   ```bash
   ros2 launch physical_ai_robotics cognitive_planning_system.launch.py
   ```

2. Test with various natural language commands:
   ```bash
   # Test navigation command
   ros2 topic pub /natural_language_command std_msgs/String "data: 'Go to the kitchen'"

   # Test object interaction
   ros2 topic pub /natural_language_command std_msgs/String "data: 'Find the cup and pick it up'"

   # Test complex command
   ros2 topic pub /natural_language_command std_msgs/String "data: 'Navigate to bedroom and find the keys'"
   ```

3. Monitor the system:
   ```bash
   # Monitor plan status
   ros2 topic echo /plan_status

   # Monitor safety status
   ros2 topic echo /safety_status
   ```

## Exercise 6: Enhance with Context Awareness

1. Add context awareness to the planner:
   ```python
   # Add to LLMCognitivePlanner class
   def update_robot_context(self):
       """Update robot context with current state"""
       # This would integrate with other ROS nodes to get current state
       # such as current location, battery level, etc.
       pass
   ```

## Exercise 7: Implement Plan Validation

1. Add plan validation before execution:
   ```python
   def validate_plan(self, plan: List[Dict]) -> bool:
       """Validate plan before execution"""
       if not plan:
           return False

       # Check for valid action types
       valid_action_types = ['navigation', 'perception', 'manipulation', 'communication']
       for action in plan:
           if action.get('action_type') not in valid_action_types:
               return False

       # Check for reasonable parameters
       for action in plan:
           if action.get('action_type') == 'navigation':
               x, y = action.get('x'), action.get('y')
               if x is None or y is None:
                   return False

       return True
   ```

## Exercise 8: Add Learning Capabilities

1. Implement plan learning and improvement:
   ```python
   # Add to LLMCognitivePlanner class
   def learn_from_execution(self, plan: List[Dict], success: bool):
       """Learn from plan execution results"""
       # Store successful plans for future reference
       # Adjust LLM prompts based on execution results
       pass
   ```

## Exercise 9: Integrate with Perception System

1. Connect cognitive planner with perception capabilities:
   ```python
   # Add perception service client to LLMCognitivePlanner
   def __init__(self):
       # ... existing initialization ...

       # Create client for object detection service
       self.object_detection_client = self.create_client(
           YourObjectDetectionSrv, 'detect_objects'
       )
   ```

## Exercise 10: Performance Optimization

1. Optimize LLM calls for real-time performance:
   ```python
   # Use caching for common commands
   from functools import lru_cache

   @lru_cache(maxsize=128)
   def get_cached_plan(self, command: str) -> str:
       """Get cached plan for common commands"""
       # This would call the LLM and cache the result
       pass
   ```

## Exercise 11: Error Handling and Recovery

1. Implement robust error handling:
   ```python
   def handle_execution_error(self, action: Dict, error: Exception):
       """Handle errors during action execution"""
       self.get_logger().error(f'Error executing action {action}: {error}')

       # Implement recovery strategies
       # Log error for learning
       # Possibly modify plan and retry
   ```

## Exercise 12: Integration Testing

1. Create comprehensive tests for the cognitive planning system:
   - Test various natural language commands
   - Validate safety guardrails
   - Test plan execution success rates
   - Verify system behavior under different conditions

## Verification Steps

1. Confirm LLM can generate action plans from natural language
2. Verify safety guardrails block dangerous commands
3. Check that plans are executed correctly
4. Validate system handles various command types
5. Ensure error handling works properly

## Expected Outcomes

- Understanding of LLM integration for robotic planning
- Knowledge of cognitive planning pipeline
- Experience with safety guardrails implementation
- Ability to create natural language to action mapping

## Troubleshooting

- If LLM calls fail, check API key and network connectivity
- If plans aren't executing, verify action server availability
- If safety guardrails are too restrictive, adjust parameters

## Next Steps

After completing these exercises, proceed to perception-action integration to connect cognitive planning with real-world perception and manipulation capabilities.