# Complete VLA Agent Integration Simulation Steps

This guide provides step-by-step instructions for building the full VLA agent loop that integrates all components into a complete autonomous system as covered in Module 4 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to integrate all VLA components (voice, language, reasoning, action) into a complete autonomous system that can execute the full "Voice → Plan → Navigate → Perceive → Act" pipeline, creating an autonomous humanoid that responds to spoken commands.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later)
- Completed all previous VLA simulation exercises
- All component systems (voice, planning, perception, manipulation, navigation)
- Understanding of complete robotics system integration

## Simulation Environment Setup

1. Ensure all VLA components are installed and tested:
   ```bash
   # Verify all required packages are available
   ros2 pkg list | grep physical_ai_robotics
   ```

2. Set up environment for full system integration

## Exercise 1: Create VLA Agent Orchestrator

1. Create the main VLA orchestrator node (`src/physical_ai_robotics/vla_agent.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String, Bool
   from geometry_msgs.msg import Twist, PoseStamped
   from sensor_msgs.msg import Image, LaserScan
   from nav2_msgs.action import NavigateToPose
   from rclpy.action import ActionClient
   import json
   import threading
   import time
   from typing import Dict, Any, Optional

   class VLAAgent(Node):
       def __init__(self):
           super().__init__('vla_agent')

           # State management
           self.current_state = 'IDLE'  # IDLE, LISTENING, PROCESSING, EXECUTING
           self.current_command = ''
           self.current_plan = []
           self.execution_step = 0

           # Subscribers
           self.voice_command_subscriber = self.create_subscription(
               String, '/recognized_speech', self.voice_command_callback, 10
           )
           self.plan_status_subscriber = self.create_subscription(
               String, '/plan_status', self.plan_status_callback, 10
           )
           self.execution_status_subscriber = self.create_subscription(
               Bool, '/execution_success', self.execution_status_callback, 10
           )

           # Publishers
           self.system_status_publisher = self.create_publisher(
               String, '/vla_system_status', 10
           )
           self.cognitive_plan_publisher = self.create_publisher(
               String, '/cognitive_plan', 10
           )
           self.manipulation_command_publisher = self.create_publisher(
               String, '/manipulation_command', 10
           )
           self.navigation_goal_publisher = self.create_publisher(
               PoseStamped, '/navigation_goal', 10
           )

           # Action clients
           self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

           # Timers
           self.state_machine_timer = self.create_timer(0.1, self.state_machine_update)

           # Command queue for handling multiple commands
           self.command_queue = []
           self.command_queue_lock = threading.Lock()

           self.get_logger().info('VLA Agent initialized and ready to receive commands')

       def voice_command_callback(self, msg):
           """Handle incoming voice commands"""
           command = msg.data.strip()
           if command:
               self.get_logger().info(f'Received voice command: "{command}"')

               # Add to command queue
               with self.command_queue_lock:
                   self.command_queue.append(command)

               # Update system status
               status_msg = String()
               status_msg.data = f'VLA_AGENT: Received command - {command}'
               self.system_status_publisher.publish(status_msg)

       def plan_status_callback(self, msg):
           """Handle plan status updates"""
           self.get_logger().info(f'Plan status: {msg.data}')

       def execution_status_callback(self, msg):
           """Handle execution status updates"""
           success = msg.data
           if success:
               self.get_logger().info('Execution step completed successfully')
               self.mark_execution_step_complete()
           else:
               self.get_logger().error('Execution step failed')
               self.handle_execution_failure()

       def state_machine_update(self):
           """Main state machine update loop"""
           if self.current_state == 'IDLE':
               # Check for new commands in queue
               with self.command_queue_lock:
                   if self.command_queue:
                       command = self.command_queue.pop(0)
                       self.current_command = command
                       self.current_state = 'PROCESSING'
                       self.get_logger().info(f'Processing command: {command}')

                       # Send command to cognitive planner
                       plan_msg = String()
                       plan_msg.data = command
                       self.cognitive_plan_publisher.publish(plan_msg)

           elif self.current_state == 'PROCESSING':
               # Wait for plan to be generated (handled by other nodes)
               pass

           elif self.current_state == 'EXECUTING':
               # Execute current plan step (handled by other nodes)
               pass

       def receive_cognitive_plan(self, plan_data: str):
           """Receive and process cognitive plan from planner"""
           try:
               plan = json.loads(plan_data)
               self.current_plan = plan.get('actions', [])
               self.execution_step = 0

               self.get_logger().info(f'Received plan with {len(self.current_plan)} steps')
               self.current_state = 'EXECUTING'

               # Begin execution of first step
               if self.current_plan:
                   self.execute_next_step()

           except json.JSONDecodeError as e:
               self.get_logger().error(f'Error parsing cognitive plan: {e}')

       def execute_next_step(self):
           """Execute the next step in the current plan"""
           if self.execution_step < len(self.current_plan):
               step = self.current_plan[self.execution_step]
               self.get_logger().info(f'Executing step {self.execution_step + 1}: {step}')

               # Execute based on action type
               action_type = step.get('action_type', '')
               if action_type == 'navigation':
                   self.execute_navigation_step(step)
               elif action_type == 'manipulation':
                   self.execute_manipulation_step(step)
               elif action_type == 'perception':
                   self.execute_perception_step(step)
               elif action_type == 'communication':
                   self.execute_communication_step(step)
               else:
                   self.get_logger().warn(f'Unknown action type: {action_type}')
                   self.mark_execution_step_complete()
           else:
               # All steps completed
               self.get_logger().info('Plan execution completed successfully')
               self.current_state = 'IDLE'
               self.publish_completion_status()

       def execute_navigation_step(self, step):
           """Execute navigation step"""
           target_location = step.get('target_location')
           x = step.get('x', 0.0)
           y = step.get('y', 0.0)
           theta = step.get('theta', 0.0)

           if target_location:
               self.get_logger().info(f'Navigating to {target_location} at ({x}, {y})')
           else:
               self.get_logger().info(f'Navigating to ({x}, {y})')

           # Create and send navigation goal
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
           if self.nav_client.wait_for_server(timeout_sec=5.0):
               send_goal_future = self.nav_client.send_goal_async(goal_msg)
               send_goal_future.add_done_callback(self.navigation_goal_response_callback)
           else:
               self.get_logger().error('Navigation server not available')
               self.mark_execution_step_complete()

       def navigation_goal_response_callback(self, future):
           """Handle navigation goal response"""
           goal_handle = future.result()
           if not goal_handle.accepted:
               self.get_logger().info('Navigation goal rejected')
               self.mark_execution_step_complete()
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
           self.mark_execution_step_complete()

       def execute_manipulation_step(self, step):
           """Execute manipulation step"""
           action = step.get('action', '')
           target_object = step.get('target_object', '')

           command = f'{action} {target_object}'.strip()
           self.get_logger().info(f'Executing manipulation: {command}')

           # Send manipulation command
           cmd_msg = String()
           cmd_msg.data = command
           self.manipulation_command_publisher.publish(cmd_msg)

       def execute_perception_step(self, step):
           """Execute perception step"""
           action = step.get('action', '')
           target_object = step.get('target_object', '')

           self.get_logger().info(f'Executing perception: {action} for {target_object}')

           # In a real system, this would trigger perception nodes
           # For now, simulate completion
           self.mark_execution_step_complete()

       def execute_communication_step(self, step):
           """Execute communication step"""
           message = step.get('message', 'Hello')

           self.get_logger().info(f'Executing communication: {message}')

           # In a real system, this would trigger text-to-speech
           # For now, simulate completion
           self.mark_execution_step_complete()

       def mark_execution_step_complete(self):
           """Mark current execution step as complete"""
           self.execution_step += 1
           self.get_logger().info(f'Completed step {self.execution_step - 1}, total: {len(self.current_plan)}')

           # Execute next step if available
           if self.execution_step < len(self.current_plan):
               self.execute_next_step()
           else:
               # All steps completed
               self.get_logger().info('All plan steps completed')
               self.current_state = 'IDLE'
               self.publish_completion_status()

       def handle_execution_failure(self):
           """Handle execution failure"""
           self.get_logger().error('Execution failed, returning to IDLE state')
           self.current_state = 'IDLE'
           self.publish_error_status()

       def publish_completion_status(self):
           """Publish completion status"""
           status_msg = String()
           status_msg.data = f'VLA_AGENT: Command completed - {self.current_command}'
           self.system_status_publisher.publish(status_msg)

       def publish_error_status(self):
           """Publish error status"""
           status_msg = String()
           status_msg.data = f'VLA_AGENT: Command failed - {self.current_command}'
           self.system_status_publisher.publish(status_msg)

   def main(args=None):
       rclpy.init(args=args)
       agent = VLAAgent()

       try:
           rclpy.spin(agent)
       except KeyboardInterrupt:
           agent.get_logger().info('VLA Agent stopped by user')
       finally:
           agent.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 2: Create VLA Agent with LLM Integration

1. Create enhanced VLA agent with LLM integration (`src/physical_ai_robotics/llm_vla_agent.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String, Bool
   from geometry_msgs.msg import Twist, PoseStamped
   from nav2_msgs.action import NavigateToPose
   from rclpy.action import ActionClient
   import json
   import openai
   import threading
   from typing import Dict, Any, Optional

   class LLMVLAAgent(Node):
       def __init__(self):
           super().__init__('llm_vla_agent')

           # State management
           self.current_state = 'IDLE'
           self.current_command = ''
           self.current_plan = []
           self.execution_step = 0

           # LLM client
           self.client = openai.OpenAI()

           # System context for LLM
           self.system_context = {
               "robot_capabilities": [
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
               "environment": "indoor home environment with structured layout"
           }

           # Subscribers
           self.voice_command_subscriber = self.create_subscription(
               String, '/recognized_speech', self.voice_command_callback, 10
           )
           self.execution_status_subscriber = self.create_subscription(
               Bool, '/execution_success', self.execution_status_callback, 10
           )

           # Publishers
           self.system_status_publisher = self.create_publisher(
               String, '/vla_system_status', 10
           )
           self.navigation_goal_publisher = self.create_publisher(
               PoseStamped, '/navigation_goal', 10
           )
           self.manipulation_command_publisher = self.create_publisher(
               String, '/manipulation_command', 10
           )

           # Action clients
           self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

           # Timers
           self.state_machine_timer = self.create_timer(0.1, self.state_machine_update)

           # Command queue
           self.command_queue = []
           self.command_queue_lock = threading.Lock()

           self.get_logger().info('LLM VLA Agent initialized')

       def voice_command_callback(self, msg):
           """Handle incoming voice commands"""
           command = msg.data.strip()
           if command:
               self.get_logger().info(f'Received voice command: "{command}"')

               # Add to command queue
               with self.command_queue_lock:
                   self.command_queue.append(command)

               # Update system status
               status_msg = String()
               status_msg.data = f'LLM_VLA_AGENT: Processing - {command}'
               self.system_status_publisher.publish(status_msg)

       def state_machine_update(self):
           """Main state machine update loop"""
           if self.current_state == 'IDLE':
               # Check for new commands in queue
               with self.command_queue_lock:
                   if self.command_queue:
                       command = self.command_queue.pop(0)
                       self.current_command = command
                       self.current_state = 'PLANNING'
                       self.get_logger().info(f'Planning for command: {command}')

                       # Generate plan using LLM
                       self.generate_plan_with_llm(command)

           elif self.current_state == 'PLANNING':
               # Wait for plan generation (handled asynchronously)
               pass

           elif self.current_state == 'EXECUTING':
               # Execute current plan (handled by other methods)
               pass

       def generate_plan_with_llm(self, command: str):
           """Generate execution plan using LLM"""
           system_prompt = f"""
           You are an intelligent planning system for a humanoid robot. Convert natural language commands into a sequence of executable robot actions.

           Robot capabilities:
           - Navigation: move to specific locations (kitchen, bedroom, living_room, office, dining_room)
           - Object interaction: detect, pick up, and place objects (cup, bottle, book, phone, keys, apple, banana, box, toy, pen, notebook)
           - Manipulation: basic arm movements
           - Perception: detect objects in the environment

           Known locations: {self.system_context['known_locations']}
           Known objects: {self.system_context['known_objects']}

           Output format: Return a JSON object with a "plan" field containing an array of action objects:
           {{
             "plan": [
               {{
                 "action_type": "navigation|perception|manipulation|communication",
                 "action": "specific_action_name",
                 "target_location": "location_name",  // for navigation
                 "target_object": "object_name",     // for object actions
                 "x": number, "y": number, "theta": number  // for navigation
               }}
             ]
           }}

           Be specific and use only the known locations and objects. If the command is complex, break it into multiple simple steps.
           """

           user_prompt = f"Command: {command}\n\nProvide a detailed execution plan for this command."

           # Call LLM in a separate thread to avoid blocking
           thread = threading.Thread(target=self._call_llm_async, args=(system_prompt, user_prompt))
           thread.start()

       def _call_llm_async(self, system_prompt: str, user_prompt: str):
           """Call LLM asynchronously"""
           try:
               response = self.client.chat.completions.create(
                   model="gpt-3.5-turbo",
                   messages=[
                       {"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_prompt}
                   ],
                   temperature=0.1,
                   max_tokens=1000,
                   response_format={"type": "json_object"}
               )

               # Parse the response
               response_text = response.choices[0].message.content
               self.get_logger().debug(f'LLM response: {response_text}')

               # Process the plan
               plan_data = json.loads(response_text)
               self.process_generated_plan(plan_data)

           except json.JSONDecodeError as e:
               self.get_logger().error(f'Error parsing LLM response: {e}')
               self.handle_planning_failure()
           except Exception as e:
               self.get_logger().error(f'Error calling LLM: {e}')
               self.handle_planning_failure()

       def process_generated_plan(self, plan_data: Dict):
           """Process the plan generated by LLM"""
           plan = plan_data.get('plan', [])

           if plan:
               self.current_plan = plan
               self.execution_step = 0
               self.get_logger().info(f'Generated plan with {len(plan)} steps')

               # Update state to executing
               self.current_state = 'EXECUTING'

               # Begin execution of first step
               if self.current_plan:
                   self.execute_next_step()
           else:
               self.get_logger().error('LLM did not generate a valid plan')
               self.handle_planning_failure()

       def handle_planning_failure(self):
           """Handle planning failure"""
           self.get_logger().error('Planning failed, returning to IDLE state')
           self.current_state = 'IDLE'

           status_msg = String()
           status_msg.data = f'LLM_VLA_AGENT: Planning failed for - {self.current_command}'
           self.system_status_publisher.publish(status_msg)

       def execute_next_step(self):
           """Execute the next step in the current plan"""
           if self.execution_step < len(self.current_plan):
               step = self.current_plan[self.execution_step]
               self.get_logger().info(f'Executing step {self.execution_step + 1}: {step}')

               # Execute based on action type
               action_type = step.get('action_type', '')
               if action_type == 'navigation':
                   self.execute_navigation_step(step)
               elif action_type == 'manipulation':
                   self.execute_manipulation_step(step)
               elif action_type == 'perception':
                   self.execute_perception_step(step)
               elif action_type == 'communication':
                   self.execute_communication_step(step)
               else:
                   self.get_logger().warn(f'Unknown action type: {action_type}')
                   self.mark_execution_step_complete()
           else:
               # All steps completed
               self.get_logger().info('Plan execution completed successfully')
               self.current_state = 'IDLE'
               self.publish_completion_status()

       def execute_navigation_step(self, step):
           """Execute navigation step"""
           target_location = step.get('target_location')
           x = step.get('x')
           y = step.get('y')
           theta = step.get('theta')

           if target_location:
               self.get_logger().info(f'Navigating to {target_location}')
           else:
               self.get_logger().info(f'Navigating to ({x}, {y})')

           # Create and send navigation goal
           goal_msg = NavigateToPose.Goal()
           goal_msg.pose.header.frame_id = 'map'
           goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
           goal_msg.pose.pose.position.x = x or 0.0
           goal_msg.pose.pose.position.y = y or 0.0
           goal_msg.pose.pose.position.z = 0.0

           # Convert theta to quaternion
           import math
           theta_val = theta or 0.0
           goal_msg.pose.pose.orientation.z = math.sin(theta_val / 2.0)
           goal_msg.pose.pose.orientation.w = math.cos(theta_val / 2.0)

           # Send goal
           if self.nav_client.wait_for_server(timeout_sec=5.0):
               send_goal_future = self.nav_client.send_goal_async(goal_msg)
               send_goal_future.add_done_callback(self.navigation_goal_response_callback)
           else:
               self.get_logger().error('Navigation server not available')
               self.mark_execution_step_complete()

       def navigation_goal_response_callback(self, future):
           """Handle navigation goal response"""
           goal_handle = future.result()
           if not goal_handle.accepted:
               self.get_logger().info('Navigation goal rejected')
               self.mark_execution_step_complete()
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
           self.mark_execution_step_complete()

       def execute_manipulation_step(self, step):
           """Execute manipulation step"""
           action = step.get('action', '')
           target_object = step.get('target_object', '')

           command = f'{action} {target_object}'.strip()
           self.get_logger().info(f'Executing manipulation: {command}')

           # Send manipulation command
           cmd_msg = String()
           cmd_msg.data = command
           self.manipulation_command_publisher.publish(cmd_msg)

       def execute_perception_step(self, step):
           """Execute perception step"""
           action = step.get('action', '')
           target_object = step.get('target_object', '')

           self.get_logger().info(f'Executing perception: {action} for {target_object}')

           # In a real system, this would trigger perception nodes
           # For now, simulate completion
           self.mark_execution_step_complete()

       def execute_communication_step(self, step):
           """Execute communication step"""
           message = step.get('message', 'Hello')

           self.get_logger().info(f'Executing communication: {message}')

           # In a real system, this would trigger text-to-speech
           # For now, simulate completion
           self.mark_execution_step_complete()

       def execution_status_callback(self, msg):
           """Handle execution status updates"""
           success = msg.data
           if success:
               self.get_logger().info('Execution step completed successfully')
               self.mark_execution_step_complete()
           else:
               self.get_logger().error('Execution step failed')
               self.handle_execution_failure()

       def mark_execution_step_complete(self):
           """Mark current execution step as complete"""
           self.execution_step += 1
           self.get_logger().info(f'Completed step {self.execution_step - 1}')

           # Execute next step if available
           if self.execution_step < len(self.current_plan):
               self.execute_next_step()
           else:
               # All steps completed
               self.get_logger().info('All plan steps completed')
               self.current_state = 'IDLE'
               self.publish_completion_status()

       def handle_execution_failure(self):
           """Handle execution failure"""
           self.get_logger().error('Execution failed, returning to IDLE state')
           self.current_state = 'IDLE'
           self.publish_error_status()

       def publish_completion_status(self):
           """Publish completion status"""
           status_msg = String()
           status_msg.data = f'LLM_VLA_AGENT: Command completed - {self.current_command}'
           self.system_status_publisher.publish(status_msg)

       def publish_error_status(self):
           """Publish error status"""
           status_msg = String()
           status_msg.data = f'LLM_VLA_AGENT: Command failed - {self.current_command}'
           self.system_status_publisher.publish(status_msg)

   def main(args=None):
       rclpy.init(args=args)
       agent = LLMVLAAgent()

       try:
           rclpy.spin(agent)
       except KeyboardInterrupt:
           agent.get_logger().info('LLM VLA Agent stopped by user')
       finally:
           agent.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 3: Create VLA System Integration Launch File

1. Create comprehensive launch file for complete VLA system (`launch/vla_complete_system.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='false')

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='false',
               description='Use simulation clock if true'
           ),

           # Audio input for voice commands
           Node(
               package='physical_ai_robotics',
               executable='audio_input',
               name='audio_input',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Whisper processor for speech-to-text
           Node(
               package='physical_ai_robotics',
               executable='whisper_processor',
               name='whisper_processor',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # LLM-based cognitive planner
           Node(
               package='physical_ai_robotics',
               executable='llm_cognitive_planner',
               name='llm_cognitive_planner',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Object detection
           Node(
               package='physical_ai_robotics',
               executable='object_detector',
               name='object_detector',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Pose estimation
           Node(
               package='physical_ai_robotics',
               executable='pose_estimator',
               name='pose_estimator',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Manipulation controller
           Node(
               package='physical_ai_robotics',
               executable='manipulation_controller',
               name='manipulation_controller',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Perception-action integrator
           Node(
               package='physical_ai_robotics',
               executable='perception_action_integrator',
               name='perception_action_integrator',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Safety guardrails
           Node(
               package='physical_ai_robotics',
               executable='safety_guardrails',
               name='safety_guardrails',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # VLA Agent (main orchestrator)
           Node(
               package='physical_ai_robotics',
               executable='llm_vla_agent',
               name='vla_agent',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           )
       ])
   ```

## Exercise 4: Create Capstone Autonomous Humanoid Demo

1. Create capstone demonstration node (`src/physical_ai_robotics/capstone_demo.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from geometry_msgs.msg import Twist
   import time
   import threading

   class CapstoneDemo(Node):
       def __init__(self):
           super().__init__('capstone_demo')

           # Publishers
           self.voice_command_publisher = self.create_publisher(
               String, '/recognized_speech', 10
           )
           self.system_status_subscriber = self.create_subscription(
               String, '/vla_system_status', self.system_status_callback, 10
           )

           # Demo sequence
           self.demo_commands = [
               "Go to the kitchen",
               "Find the red cup",
               "Pick up the cup",
               "Go to the living room",
               "Place the cup on the table"
           ]

           self.demo_index = 0
           self.demo_active = False

           self.get_logger().info('Capstone Demo initialized')

       def start_demo(self):
           """Start the capstone demonstration"""
           self.get_logger().info('Starting capstone demonstration...')
           self.demo_active = True
           self.demo_index = 0

           # Execute demo commands sequentially
           for i, command in enumerate(self.demo_commands):
               self.get_logger().info(f'Waiting before command {i+1}: {command}')
               time.sleep(5)  # Wait between commands

               if not self.demo_active:
                   break

               self.get_logger().info(f'Issuing command {i+1}: {command}')
               cmd_msg = String()
               cmd_msg.data = command
               self.voice_command_publisher.publish(cmd_msg)

           self.get_logger().info('Capstone demonstration completed')

       def system_status_callback(self, msg):
           """Monitor system status during demo"""
           self.get_logger().info(f'System status: {msg.data}')

   def main(args=None):
       rclpy.init(args=args)
       demo = CapstoneDemo()

       # Start demo in separate thread to allow ROS spinning
       demo_thread = threading.Thread(target=demo.start_demo)
       demo_thread.start()

       try:
           rclpy.spin(demo)
       except KeyboardInterrupt:
           demo.get_logger().info('Capstone demo stopped by user')
           demo.demo_active = False
       finally:
           demo.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 5: Test Complete VLA System

1. Launch the complete VLA system:
   ```bash
   ros2 launch physical_ai_robotics vla_complete_system.launch.py
   ```

2. Test with various voice commands:
   ```bash
   # Test simple navigation
   ros2 topic pub /recognized_speech std_msgs/String "data: 'Go to the kitchen'"

   # Test object interaction
   ros2 topic pub /recognized_speech std_msgs/String "data: 'Find the cup and pick it up'"

   # Test complex command
   ros2 topic pub /recognized_speech std_msgs/String "data: 'Go to bedroom, find the keys, and bring them to me'"
   ```

3. Monitor the complete system:
   ```bash
   # Monitor system status
   ros2 topic echo /vla_system_status

   # Monitor plan execution
   ros2 topic echo /plan_status

   # Monitor all components
   ros2 topic list | grep -E "(vla|cognitive|detection|manipulation|navigation)"
   ```

## Exercise 6: Performance Validation

1. Create performance monitoring node:
   ```python
   # Add to VLA agent or create separate monitoring node
   class VLAMonitor(Node):
       def __init__(self):
           super().__init__('vla_monitor')

           # Monitor all VLA components
           self.command_response_times = []
           self.execution_success_rates = []
           self.system_resource_usage = []

           # Timer for performance monitoring
           self.monitor_timer = self.create_timer(1.0, self.monitor_performance)

       def monitor_performance(self):
           """Monitor VLA system performance"""
           # Track response times
           # Monitor success rates
           # Check resource usage
           pass
   ```

## Exercise 7: Error Handling and Recovery

1. Implement comprehensive error handling:
   ```python
   # Add to VLA agent
   def handle_system_error(self, error_type: str, error_message: str):
       """Handle various system errors"""
       self.get_logger().error(f'System error ({error_type}): {error_message}')

       # Implement recovery strategies
       if error_type == 'navigation_failure':
           self.attempt_navigation_recovery()
       elif error_type == 'manipulation_failure':
           self.attempt_manipulation_recovery()
       elif error_type == 'perception_failure':
           self.attempt_perception_recovery()

       # Return to safe state
       self.current_state = 'IDLE'
   ```

## Exercise 8: Add Learning and Adaptation

1. Implement learning capabilities:
   ```python
   # Add learning to VLA agent
   def learn_from_execution(self, command: str, plan: list, success: bool):
       """Learn from plan execution results"""
       # Store successful command-plan pairs
       # Update LLM prompts based on results
       # Adapt system behavior based on experience
       pass
   ```

## Exercise 9: Multi-Modal Integration

1. Enhance with additional modalities:
   ```python
   # Add touch, force, or other sensory feedback
   def integrate_haptic_feedback(self):
       """Integrate haptic/tactile feedback for better manipulation"""
       # Use force/torque sensors
       # Implement compliant control
       # Enhance manipulation with tactile feedback
       pass
   ```

## Exercise 10: Human-Robot Interaction

1. Implement improved human-robot interaction:
   ```python
   # Add to VLA agent
   def enable_conversation_mode(self):
       """Enable conversational interaction with the robot"""
       # Implement dialogue management
       # Handle follow-up commands
       # Support command clarification
       pass
   ```

## Exercise 11: Safety and Ethics

1. Implement advanced safety measures:
   ```python
   # Add to safety guardrails
   def implement_ethical_constraints(self):
       """Implement ethical and safety constraints"""
       # Ensure robot follows ethical guidelines
       # Implement Asimov-like laws
       # Prevent harmful actions
       pass
   ```

## Exercise 12: System Integration Testing

1. Create comprehensive integration tests:
   - Test full voice-to-action pipeline
   - Validate system response to various commands
   - Test error handling and recovery
   - Verify safety constraints
   - Measure system performance and reliability

## Verification Steps

1. Confirm all VLA components integrate properly
2. Verify end-to-end voice command processing
3. Check that complex multi-step plans execute correctly
4. Validate safety systems function during operation
5. Ensure system handles errors gracefully

## Expected Outcomes

- Complete VLA system integration
- End-to-end voice-to-action pipeline
- Autonomous humanoid behavior
- Understanding of complex system integration

## Troubleshooting

- If system is too slow, optimize LLM calls or use smaller models
- If plans fail frequently, improve perception and manipulation
- If safety systems are too restrictive, adjust parameters

## Next Steps

After completing the VLA system integration, you have finished all modules of the Physical AI & Humanoid Robotics textbook! The complete system now includes:
- Module 1: ROS 2 fundamentals and robot control
- Module 2: Digital twin with Gazebo and Unity
- Module 3: AI-robot brain with Isaac Sim and perception
- Module 4: Vision-Language-Action system for autonomous behavior

The system is now ready for deployment, further enhancement, and real-world testing.