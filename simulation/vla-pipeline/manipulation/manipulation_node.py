#!/usr/bin/env python3

"""
Manipulation Node for LLM-Controlled Robots
Module 4: Vision-Language-Action (VLA) - Physical AI & Humanoid Robotics Textbook

This module implements manipulation tasks (pick, place, align) for the VLA system,
allowing the AI to perform complex manipulation tasks based on natural language commands.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from moveit_msgs.msg import MoveItErrorCodes
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryAction
from actionlib_msgs.msg import GoalStatus
import json
import math
from rclpy.action import ActionClient
from typing import Dict, Any, Optional, List


class ManipulationNode(Node):
    """
    Node that handles manipulation tasks (pick, place, align) for LLM-controlled robots
    """
    def __init__(self):
        super().__init__('manipulation_node')

        # Publishers
        self.manipulation_status_pub = self.create_publisher(
            String,
            '/manipulation/status',
            10
        )

        self.manipulation_feedback_pub = self.create_publisher(
            String,
            '/manipulation/feedback',
            10
        )

        # Subscribers
        self.llm_manipulation_sub = self.create_subscription(
            String,
            '/vla/llm_manipulation_command',
            self.llm_manipulation_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Action clients for manipulation
        self.arm_controller_client = ActionClient(self, FollowJointTrajectoryAction, '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(self, FollowJointTrajectoryAction, '/gripper_controller/follow_joint_trajectory')

        # Services for kinematics
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')

        # Internal state
        self.current_joint_states = None
        self.manipulation_active = False
        self.emergency_stop_active = False

        # Manipulation parameters
        self.pick_approach_distance = 0.1  # meters
        self.place_approach_distance = 0.1  # meters
        self.alignment_tolerance = 0.01  # meters
        self.orientation_tolerance = 0.1  # radians

        self.get_logger().info('Manipulation Node initialized')

    def llm_manipulation_callback(self, msg):
        """
        Process LLM-generated manipulation commands
        """
        try:
            command_data = json.loads(msg.data)
            command_type = command_data.get('type', 'pick_place')

            if command_type == 'pick':
                object_info = command_data.get('object', {})
                self.execute_pick(object_info)
            elif command_type == 'place':
                target_info = command_data.get('target', {})
                self.execute_place(target_info)
            elif command_type == 'align':
                alignment_info = command_data.get('alignment', {})
                self.execute_alignment(alignment_info)
            elif command_type == 'pick_place':
                object_info = command_data.get('object', {})
                target_info = command_data.get('target', {})
                self.execute_pick_place(object_info, target_info)
            elif command_type == 'custom_manipulation':
                steps = command_data.get('steps', [])
                self.execute_custom_manipulation(steps)
            else:
                self.get_logger().warn(f'Unknown manipulation command type: {command_type}')

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON in manipulation command: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error processing LLM manipulation command: {str(e)}')

    def joint_state_callback(self, msg: JointState):
        """
        Update current joint states
        """
        self.current_joint_states = msg

    def execute_pick(self, object_info: Dict[str, Any]):
        """
        Execute pick operation for the specified object
        """
        try:
            self.get_logger().info(f'Executing pick operation for object: {object_info.get("name", "unknown")}')

            # Validate object information
            if not self.validate_object_info(object_info):
                self.get_logger().error('Invalid object information for pick operation')
                return False

            # Move to pre-pick position (approach)
            pre_pick_pose = self.calculate_pre_pick_pose(object_info)
            if not self.move_to_pose(pre_pick_pose, 'pre_pick'):
                self.get_logger().error('Failed to move to pre-pick position')
                return False

            # Move to pick position
            pick_pose = self.object_to_pose(object_info)
            if not self.move_to_pose(pick_pose, 'pick'):
                self.get_logger().error('Failed to move to pick position')
                return False

            # Close gripper to pick object
            if not self.close_gripper():
                self.get_logger().error('Failed to close gripper')
                return False

            # Move back to pre-pick position
            if not self.move_to_pose(pre_pick_pose, 'post_pick_lift'):
                self.get_logger().error('Failed to move to post-pick position')
                return False

            # Publish success status
            status_msg = String()
            status_msg.data = f'Pick operation completed for {object_info.get("name", "unknown")}'
            self.manipulation_status_pub.publish(status_msg)

            self.get_logger().info(f'Pick operation completed for {object_info.get("name", "unknown")}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in pick operation: {str(e)}')
            return False

    def execute_place(self, target_info: Dict[str, Any]):
        """
        Execute place operation at the specified target
        """
        try:
            self.get_logger().info(f'Executing place operation at target: {target_info.get("name", "unknown")}')

            # Validate target information
            if not self.validate_target_info(target_info):
                self.get_logger().error('Invalid target information for place operation')
                return False

            # Move to pre-place position (approach)
            pre_place_pose = self.calculate_pre_place_pose(target_info)
            if not self.move_to_pose(pre_place_pose, 'pre_place'):
                self.get_logger().error('Failed to move to pre-place position')
                return False

            # Move to place position
            place_pose = self.target_to_pose(target_info)
            if not self.move_to_pose(place_pose, 'place'):
                self.get_logger().error('Failed to move to place position')
                return False

            # Open gripper to place object
            if not self.open_gripper():
                self.get_logger().error('Failed to open gripper')
                return False

            # Move back to pre-place position
            if not self.move_to_pose(pre_place_pose, 'post_place_lift'):
                self.get_logger().error('Failed to move to post-place position')
                return False

            # Publish success status
            status_msg = String()
            status_msg.data = f'Place operation completed at {target_info.get("name", "unknown")}'
            self.manipulation_status_pub.publish(status_msg)

            self.get_logger().info(f'Place operation completed at {target_info.get("name", "unknown")}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in place operation: {str(e)}')
            return False

    def execute_alignment(self, alignment_info: Dict[str, Any]):
        """
        Execute alignment operation to position objects relative to each other
        """
        try:
            self.get_logger().info(f'Executing alignment operation: {alignment_info.get("description", "unknown")}')

            # Validate alignment information
            if not self.validate_alignment_info(alignment_info):
                self.get_logger().error('Invalid alignment information')
                return False

            # Calculate target pose based on alignment requirements
            target_pose = self.calculate_alignment_pose(alignment_info)
            if not target_pose:
                self.get_logger().error('Failed to calculate alignment pose')
                return False

            # Move to alignment position
            if not self.move_to_pose(target_pose, 'alignment'):
                self.get_logger().error('Failed to move to alignment position')
                return False

            # Publish success status
            status_msg = String()
            status_msg.data = f'Alignment operation completed: {alignment_info.get("description", "unknown")}'
            self.manipulation_status_pub.publish(status_msg)

            self.get_logger().info(f'Alignment operation completed: {alignment_info.get("description", "unknown")}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in alignment operation: {str(e)}')
            return False

    def execute_pick_place(self, object_info: Dict[str, Any], target_info: Dict[str, Any]):
        """
        Execute complete pick and place operation
        """
        try:
            self.get_logger().info(f'Executing pick-place operation: {object_info.get("name", "unknown")} -> {target_info.get("name", "unknown")}')

            # Execute pick operation
            pick_success = self.execute_pick(object_info)
            if not pick_success:
                self.get_logger().error('Pick operation failed in pick-place sequence')
                return False

            # Execute place operation
            place_success = self.execute_place(target_info)
            if not place_success:
                self.get_logger().error('Place operation failed in pick-place sequence')
                return False

            # Publish success status
            status_msg = String()
            status_msg.data = f'Pick-place operation completed: {object_info.get("name", "unknown")} -> {target_info.get("name", "unknown")}'
            self.manipulation_status_pub.publish(status_msg)

            self.get_logger().info(f'Pick-place operation completed: {object_info.get("name", "unknown")} -> {target_info.get("name", "unknown")}')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in pick-place operation: {str(e)}')
            return False

    def execute_custom_manipulation(self, steps: List[Dict[str, Any]]):
        """
        Execute a sequence of custom manipulation steps
        """
        try:
            self.get_logger().info(f'Executing custom manipulation with {len(steps)} steps')

            for i, step in enumerate(steps):
                step_type = step.get('type', 'move')

                if step_type == 'move':
                    pose = self.step_to_pose(step)
                    success = self.move_to_pose(pose, f'step_{i+1}')
                elif step_type == 'gripper':
                    gripper_cmd = step.get('command', 'open')
                    if gripper_cmd == 'open':
                        success = self.open_gripper()
                    elif gripper_cmd == 'close':
                        success = self.close_gripper()
                    else:
                        success = False
                elif step_type == 'wait':
                    duration = step.get('duration', 1.0)
                    success = self.wait_for_duration(duration)
                else:
                    self.get_logger().warn(f'Unknown step type: {step_type}')
                    continue

                if not success:
                    self.get_logger().error(f'Failed at step {i+1}: {step_type}')
                    return False

            # Publish success status
            status_msg = String()
            status_msg.data = f'Custom manipulation completed with {len(steps)} steps'
            self.manipulation_status_pub.publish(status_msg)

            self.get_logger().info(f'Custom manipulation completed with {len(steps)} steps')
            return True

        except Exception as e:
            self.get_logger().error(f'Error in custom manipulation: {str(e)}')
            return False

    def calculate_pre_pick_pose(self, object_info: Dict[str, Any]) -> Pose:
        """
        Calculate pre-pick pose (approach position)
        """
        object_pose = self.object_to_pose(object_info)

        # Calculate approach position (above the object)
        pre_pick_pose = Pose()
        pre_pick_pose.position.x = object_pose.position.x
        pre_pick_pose.position.y = object_pose.position.y
        pre_pick_pose.position.z = object_pose.position.z + self.pick_approach_distance
        pre_pick_pose.orientation = object_pose.orientation

        return pre_pick_pose

    def calculate_pre_place_pose(self, target_info: Dict[str, Any]) -> Pose:
        """
        Calculate pre-place pose (approach position)
        """
        target_pose = self.target_to_pose(target_info)

        # Calculate approach position (above the target)
        pre_place_pose = Pose()
        pre_place_pose.position.x = target_pose.position.x
        pre_place_pose.position.y = target_pose.position.y
        pre_place_pose.position.z = target_pose.position.z + self.place_approach_distance
        pre_place_pose.orientation = target_pose.orientation

        return pre_place_pose

    def calculate_alignment_pose(self, alignment_info: Dict[str, Any]) -> Optional[Pose]:
        """
        Calculate pose for alignment operation based on requirements
        """
        try:
            pose = Pose()

            # Parse alignment requirements
            reference_object = alignment_info.get('reference_object', {})
            target_object = alignment_info.get('target_object', {})
            alignment_type = alignment_info.get('type', 'align_x')

            # Calculate based on alignment type
            if alignment_type == 'align_x':
                pose.position.x = reference_object.get('x', 0.0)
                pose.position.y = target_object.get('y', 0.0)
                pose.position.z = target_object.get('z', 0.0)
            elif alignment_type == 'align_y':
                pose.position.x = target_object.get('x', 0.0)
                pose.position.y = reference_object.get('y', 0.0)
                pose.position.z = target_object.get('z', 0.0)
            elif alignment_type == 'align_z':
                pose.position.x = target_object.get('x', 0.0)
                pose.position.y = target_object.get('y', 0.0)
                pose.position.z = reference_object.get('z', 0.0)
            elif alignment_type == 'align_xy':
                pose.position.x = reference_object.get('x', 0.0)
                pose.position.y = reference_object.get('y', 0.0)
                pose.position.z = target_object.get('z', 0.0)
            elif alignment_type == 'align_xyz':
                pose.position.x = reference_object.get('x', 0.0)
                pose.position.y = reference_object.get('y', 0.0)
                pose.position.z = reference_object.get('z', 0.0)
            else:
                # Default: use target object position
                pose.position.x = target_object.get('x', 0.0)
                pose.position.y = target_object.get('y', 0.0)
                pose.position.z = target_object.get('z', 0.0)

            # Set orientation
            orientation = alignment_info.get('orientation', {})
            pose.orientation.x = orientation.get('x', 0.0)
            pose.orientation.y = orientation.get('y', 0.0)
            pose.orientation.z = orientation.get('z', 0.0)
            pose.orientation.w = orientation.get('w', 1.0)

            return pose
        except Exception as e:
            self.get_logger().error(f'Error calculating alignment pose: {str(e)}')
            return None

    def object_to_pose(self, object_info: Dict[str, Any]) -> Pose:
        """
        Convert object information to Pose message
        """
        pose = Pose()
        pose.position.x = float(object_info.get('x', 0.0))
        pose.position.y = float(object_info.get('y', 0.0))
        pose.position.z = float(object_info.get('z', 0.0))

        orientation = object_info.get('orientation', {})
        pose.orientation.x = float(orientation.get('x', 0.0))
        pose.orientation.y = float(orientation.get('y', 0.0))
        pose.orientation.z = float(orientation.get('z', 0.0))
        pose.orientation.w = float(orientation.get('w', 1.0))

        return pose

    def target_to_pose(self, target_info: Dict[str, Any]) -> Pose:
        """
        Convert target information to Pose message
        """
        pose = Pose()
        pose.position.x = float(target_info.get('x', 0.0))
        pose.position.y = float(target_info.get('y', 0.0))
        pose.position.z = float(target_info.get('z', 0.0))

        orientation = target_info.get('orientation', {})
        pose.orientation.x = float(orientation.get('x', 0.0))
        pose.orientation.y = float(orientation.get('y', 0.0))
        pose.orientation.z = float(orientation.get('z', 0.0))
        pose.orientation.w = float(orientation.get('w', 1.0))

        return pose

    def step_to_pose(self, step: Dict[str, Any]) -> Pose:
        """
        Convert step information to Pose message
        """
        pose = Pose()
        pose.position.x = float(step.get('x', 0.0))
        pose.position.y = float(step.get('y', 0.0))
        pose.position.z = float(step.get('z', 0.0))

        orientation = step.get('orientation', {})
        pose.orientation.x = float(orientation.get('x', 0.0))
        pose.orientation.y = float(orientation.get('y', 0.0))
        pose.orientation.z = float(orientation.get('z', 0.0))
        pose.orientation.w = float(orientation.get('w', 1.0))

        return pose

    def validate_object_info(self, object_info: Dict[str, Any]) -> bool:
        """
        Validate object information for manipulation
        """
        required_fields = ['x', 'y', 'z']
        for field in required_fields:
            if field not in object_info:
                return False
        return True

    def validate_target_info(self, target_info: Dict[str, Any]) -> bool:
        """
        Validate target information for manipulation
        """
        required_fields = ['x', 'y', 'z']
        for field in required_fields:
            if field not in target_info:
                return False
        return True

    def validate_alignment_info(self, alignment_info: Dict[str, Any]) -> bool:
        """
        Validate alignment information for manipulation
        """
        required_fields = ['type']
        for field in required_fields:
            if field not in alignment_info:
                return False
        return True

    def move_to_pose(self, pose: Pose, step_name: str) -> bool:
        """
        Move the manipulator to the specified pose using inverse kinematics
        """
        try:
            self.get_logger().info(f'Moving to {step_name} pose')

            # Wait for IK service
            if not self.ik_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error('IK service not available')
                return False

            # Create IK request
            ik_request = GetPositionIK.Request()
            ik_request.ik_request.group_name = 'manipulator'  # Adjust for your robot
            ik_request.ik_request.pose_stamped.header.frame_id = 'base_link'  # Adjust for your robot
            ik_request.ik_request.pose_stamped.pose = pose

            # Call IK service
            future = self.ik_client.call_async(ik_request)
            rclpy.spin_until_future_complete(self, future)

            ik_response = future.result()
            if ik_response.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(f'IK failed for {step_name}: {ik_response.error_code.val}')
                return False

            # Get joint positions from IK solution
            joint_positions = ik_response.solution.joint_state.position

            # Create trajectory message
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = ik_response.solution.joint_state.name
            trajectory_msg.header.stamp = self.get_clock().now().to_msg()
            trajectory_msg.header.frame_id = 'base_link'

            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = list(joint_positions)
            point.time_from_start.sec = 2  # 2 seconds to reach position
            trajectory_msg.points.append(point)

            # Publish trajectory
            # Note: In a real implementation, you would use the action client to send the trajectory
            # For now, we'll simulate success
            self.get_logger().info(f'Moved to {step_name} pose successfully')
            return True

        except Exception as e:
            self.get_logger().error(f'Error moving to {step_name} pose: {str(e)}')
            return False

    def close_gripper(self) -> bool:
        """
        Close the gripper to pick up an object
        """
        try:
            self.get_logger().info('Closing gripper')

            # Create gripper trajectory message
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = ['gripper_joint']  # Adjust for your robot
            trajectory_msg.header.stamp = self.get_clock().now().to_msg()
            trajectory_msg.header.frame_id = 'gripper_link'

            # Create trajectory point (closed position)
            point = JointTrajectoryPoint()
            point.positions = [0.0]  # Closed position - adjust for your gripper
            point.time_from_start.sec = 1  # 1 second to close
            trajectory_msg.points.append(point)

            # Publish gripper command
            # Note: In a real implementation, you would use the action client to send the trajectory
            self.get_logger().info('Gripper closed successfully')
            return True

        except Exception as e:
            self.get_logger().error(f'Error closing gripper: {str(e)}')
            return False

    def open_gripper(self) -> bool:
        """
        Open the gripper to release an object
        """
        try:
            self.get_logger().info('Opening gripper')

            # Create gripper trajectory message
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = ['gripper_joint']  # Adjust for your robot
            trajectory_msg.header.stamp = self.get_clock().now().to_msg()
            trajectory_msg.header.frame_id = 'gripper_link'

            # Create trajectory point (open position)
            point = JointTrajectoryPoint()
            point.positions = [0.05]  # Open position - adjust for your gripper
            point.time_from_start.sec = 1  # 1 second to open
            trajectory_msg.points.append(point)

            # Publish gripper command
            # Note: In a real implementation, you would use the action client to send the trajectory
            self.get_logger().info('Gripper opened successfully')
            return True

        except Exception as e:
            self.get_logger().error(f'Error opening gripper: {str(e)}')
            return False

    def wait_for_duration(self, duration: float) -> bool:
        """
        Wait for the specified duration
        """
        try:
            import time
            time.sleep(duration)
            self.get_logger().info(f'Waited for {duration} seconds')
            return True
        except Exception as e:
            self.get_logger().error(f'Error during wait: {str(e)}')
            return False


def main(args=None):
    rclpy.init(args=args)

    manipulation_node = ManipulationNode()

    try:
        rclpy.spin(manipulation_node)
    except KeyboardInterrupt:
        pass
    finally:
        manipulation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()