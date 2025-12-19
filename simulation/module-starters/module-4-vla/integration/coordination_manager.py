#!/usr/bin/env python3

"""
Coordination Manager for Physical AI & Humanoid Robotics Textbook
Module 4: Vision-Language-Action (VLA)

This module manages coordination between VLA components.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
import json
import threading
import time
from typing import Dict, List, Callable, Any
from enum import Enum
import queue


class CoordinationState(Enum):
    IDLE = "idle"
    COORDINATING = "coordinating"
    WAITING_FOR_COMPONENTS = "waiting_for_components"
    SYNCHRONIZING = "synchronizing"
    EXECUTING_PLAN = "executing_plan"
    ERROR = "error"


class ComponentStatus(Enum):
    UNKNOWN = "unknown"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class CoordinationManagerNode(Node):
    def __init__(self):
        super().__init__('coordination_manager_node')

        # Initialize coordination state
        self.coordination_state = CoordinationState.IDLE
        self.component_statuses = {}
        self.pending_requests = queue.Queue()
        self.active_plans = []
        self.synchronization_points = {}

        # Component readiness tracking
        self.component_readiness = {
            'vision': ComponentStatus.UNKNOWN,
            'language': ComponentStatus.UNKNOWN,
            'action': ComponentStatus.UNKNOWN,
            'navigation': ComponentStatus.UNKNOWN
        }

        # Timers
        self.status_check_timer = self.create_timer(1.0, self.check_component_statuses)
        self.coordination_timer = self.create_timer(0.1, self.process_coordination)

        # Publishers
        self.coordination_status_pub = self.create_publisher(
            String,
            '/coordination/status',
            10
        )

        self.coordination_command_pub = self.create_publisher(
            String,
            '/coordination/command',
            10
        )

        self.system_ready_pub = self.create_publisher(
            String,
            '/coordination/system_ready',
            10
        )

        # Subscribers for component status
        self.vision_status_sub = self.create_subscription(
            String,
            '/vision/status',
            lambda msg: self.update_component_status('vision', msg.data),
            10
        )

        self.language_status_sub = self.create_subscription(
            String,
            '/language/status',
            lambda msg: self.update_component_status('language', msg.data),
            10
        )

        self.action_status_sub = self.create_subscription(
            String,
            '/action/status',
            lambda msg: self.update_component_status('action', msg.data),
            10
        )

        self.navigation_status_sub = self.create_subscription(
            String,
            '/navigation/status',
            lambda msg: self.update_component_status('navigation', msg.data),
            10
        )

        # Subscribers for coordination requests
        self.vla_request_sub = self.create_subscription(
            String,
            '/vla/request',
            self.vla_request_callback,
            10
        )

        self.synchronization_sub = self.create_subscription(
            String,
            '/coordination/sync',
            self.synchronization_callback,
            10
        )

        self.get_logger().info('Coordination Manager Node initialized')

        # Lock for thread safety
        self.coordination_lock = threading.Lock()

    def vla_request_callback(self, msg):
        """Handle VLA system requests"""
        try:
            request_data = json.loads(msg.data)
            request_type = request_data.get('type', 'unknown')
            request_id = request_data.get('id', str(time.time()))

            self.get_logger().info(f'Received VLA request: {request_type} (ID: {request_id})')

            # Add request to pending queue
            request = {
                'id': request_id,
                'type': request_type,
                'data': request_data,
                'timestamp': time.time(),
                'status': 'pending'
            }

            self.pending_requests.put(request)

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid VLA request JSON: {e}')

    def synchronization_callback(self, msg):
        """Handle synchronization signals from components"""
        try:
            sync_data = json.loads(msg.data)
            sync_point = sync_data.get('point', 'unknown')
            component = sync_data.get('component', 'unknown')

            self.get_logger().info(f'Synchronization signal from {component} at {sync_point}')

            # Register synchronization
            if sync_point not in self.synchronization_points:
                self.synchronization_points[sync_point] = []
            self.synchronization_points[sync_point].append(component)

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid sync data JSON: {e}')

    def update_component_status(self, component: str, status: str):
        """Update status of a component"""
        with self.coordination_lock:
            # Map status strings to ComponentStatus enum
            status_map = {
                'ready': ComponentStatus.READY,
                'busy': ComponentStatus.BUSY,
                'error': ComponentStatus.ERROR,
                'offline': ComponentStatus.OFFLINE
            }

            self.component_readiness[component] = status_map.get(status.lower(), ComponentStatus.UNKNOWN)

            # Log status change
            self.get_logger().debug(f'Component {component} status: {status}')

    def check_component_statuses(self):
        """Periodically check component statuses"""
        with self.coordination_lock:
            all_ready = all(
                status == ComponentStatus.READY
                for status in self.component_readiness.values()
            )

            if all_ready and self.coordination_state == CoordinationState.WAITING_FOR_COMPONENTS:
                self.coordination_state = CoordinationState.IDLE
                self.publish_system_ready()

    def process_coordination(self):
        """Main coordination processing loop"""
        with self.coordination_lock:
            # Process pending requests
            while not self.pending_requests.empty():
                try:
                    request = self.pending_requests.get_nowait()
                    self.process_request(request)
                except queue.Empty:
                    break

            # Check for synchronization needs
            self.check_synchronization()

            # Publish coordination status
            self.publish_coordination_status()

    def process_request(self, request: Dict):
        """Process a coordination request"""
        request_type = request['type']

        if request_type == 'execute_plan':
            self.execute_plan_request(request)
        elif request_type == 'synchronize':
            self.synchronize_request(request)
        elif request_type == 'status_check':
            self.status_check_request(request)
        else:
            self.get_logger().warn(f'Unknown request type: {request_type}')
            request['status'] = 'unknown_type'

    def execute_plan_request(self, request: Dict):
        """Handle plan execution request"""
        plan_data = request.get('data', {}).get('plan', {})

        with self.coordination_lock:
            self.coordination_state = CoordinationState.EXECUTING_PLAN
            self.active_plans.append(request)

            # Check if all required components are ready
            required_components = plan_data.get('required_components', ['vision', 'language', 'action'])
            all_components_ready = all(
                self.component_readiness[comp] == ComponentStatus.READY
                for comp in required_components
                if comp in self.component_readiness
            )

            if not all_components_ready:
                self.coordination_state = CoordinationState.WAITING_FOR_COMPONENTS
                request['status'] = 'waiting_for_components'
                self.get_logger().info(f'Waiting for components: {required_components}')
            else:
                # Execute the plan by sending commands to components
                self.execute_plan(plan_data)
                request['status'] = 'executing'

    def synchronize_request(self, request: Dict):
        """Handle synchronization request"""
        sync_data = request.get('data', {})
        sync_point = sync_data.get('point', 'default')
        components = sync_data.get('components', [])

        with self.coordination_lock:
            self.coordination_state = CoordinationState.SYNCHRONIZING

            # Send synchronization command to specified components
            sync_cmd = {
                'command': 'synchronize',
                'point': sync_point,
                'components': components,
                'request_id': request['id']
            }

            cmd_msg = String()
            cmd_msg.data = json.dumps(sync_cmd)
            self.coordination_command_pub.publish(cmd_msg)

            request['status'] = 'synchronizing'

    def status_check_request(self, request: Dict):
        """Handle status check request"""
        with self.coordination_lock:
            status_info = {
                'coordination_state': self.coordination_state.value,
                'component_readiness': {
                    comp: status.value for comp, status in self.component_readiness.items()
                },
                'active_plans': len(self.active_plans),
                'pending_requests': self.pending_requests.qsize(),
                'timestamp': time.time()
            }

            # Publish status information
            status_msg = String()
            status_msg.data = json.dumps(status_info)
            self.coordination_status_pub.publish(status_msg)

            request['status'] = 'completed'
            request['result'] = status_info

    def execute_plan(self, plan_data: Dict):
        """Execute a plan by coordinating components"""
        steps = plan_data.get('steps', [])

        for step in steps:
            step_type = step.get('type', 'unknown')
            required_components = step.get('requires', [])

            # Wait for required components to be ready
            if not self.wait_for_components(required_components, timeout=5.0):
                self.get_logger().error(f'Timeout waiting for components: {required_components}')
                break

            # Execute the step
            self.execute_step(step)

    def execute_step(self, step: Dict):
        """Execute a single step in a plan"""
        step_type = step.get('type', 'unknown')
        step_data = step.get('data', {})

        self.get_logger().info(f'Executing step: {step_type}')

        # Send step command to appropriate component
        step_cmd = {
            'command': 'execute',
            'step_type': step_type,
            'step_data': step_data,
            'step_id': step.get('id', str(time.time()))
        }

        cmd_msg = String()
        cmd_msg.data = json.dumps(step_cmd)
        self.coordination_command_pub.publish(cmd_msg)

    def wait_for_components(self, components: List[str], timeout: float = 5.0) -> bool:
        """Wait for specified components to be ready"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_ready = True
            for comp in components:
                if comp in self.component_readiness:
                    if self.component_readiness[comp] != ComponentStatus.READY:
                        all_ready = False
                        break

            if all_ready:
                return True

            time.sleep(0.1)  # Small delay to prevent busy waiting

        return False

    def check_synchronization(self):
        """Check if synchronization points are reached"""
        # Check if all required components have reached sync points
        for sync_point, components in self.synchronization_points.items():
            # This is a simplified check - in practice, you'd have more complex sync logic
            if len(components) >= 2:  # Example: at least 2 components synchronized
                self.get_logger().info(f'Synchronization achieved at {sync_point}: {components}')
                # Clear this synchronization point
                del self.synchronization_points[sync_point]

    def publish_coordination_status(self):
        """Publish current coordination status"""
        status_data = {
            'state': self.coordination_state.value,
            'component_readiness': {
                comp: status.value for comp, status in self.component_readiness.items()
            },
            'active_plans': len(self.active_plans),
            'pending_requests': self.pending_requests.qsize(),
            'synchronization_points': list(self.synchronization_points.keys()),
            'timestamp': time.time()
        }

        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.coordination_status_pub.publish(status_msg)

    def publish_system_ready(self):
        """Publish system ready signal"""
        ready_msg = String()
        ready_msg.data = json.dumps({
            'status': 'ready',
            'components': {
                comp: status.value for comp, status in self.component_readiness.items()
            },
            'timestamp': time.time()
        })
        self.system_ready_pub.publish(ready_msg)

    def get_coordination_state(self) -> Dict[str, Any]:
        """Get current coordination state"""
        return {
            'coordination_state': self.coordination_state.value,
            'component_readiness': {
                comp: status.value for comp, status in self.component_readiness.items()
            },
            'active_plans': len(self.active_plans),
            'pending_requests': self.pending_requests.qsize(),
            'synchronization_points': self.synchronization_points,
            'timestamp': time.time()
        }

    def reset_coordination(self):
        """Reset coordination state"""
        with self.coordination_lock:
            self.coordination_state = CoordinationState.IDLE
            self.active_plans = []
            self.synchronization_points = {}
            self.pending_requests = queue.Queue()

            # Reset component statuses
            for comp in self.component_readiness:
                self.component_readiness[comp] = ComponentStatus.UNKNOWN

            self.publish_coordination_status()
            self.get_logger().info('Coordination system reset')


def main(args=None):
    rclpy.init(args=args)

    coordination_manager = CoordinationManagerNode()

    try:
        # Example: Simulate system startup
        def simulate_startup():
            coordination_manager.get_logger().info('VLA Coordination System starting up...')

            # Simulate component readiness
            def set_component_ready():
                vision_msg = String()
                vision_msg.data = 'ready'
                coordination_manager.update_component_status('vision', 'ready')

                language_msg = String()
                language_msg.data = 'ready'
                coordination_manager.update_component_status('language', 'ready')

                action_msg = String()
                action_msg.data = 'ready'
                coordination_manager.update_component_status('action', 'ready')

                navigation_msg = String()
                navigation_msg.data = 'ready'
                coordination_manager.update_component_status('navigation', 'ready')

            coordination_manager.create_timer(1.0, set_component_ready)

        coordination_manager.create_timer(0.5, simulate_startup)

        # Run the coordination manager
        rclpy.spin(coordination_manager)
    except KeyboardInterrupt:
        pass
    finally:
        coordination_manager.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()