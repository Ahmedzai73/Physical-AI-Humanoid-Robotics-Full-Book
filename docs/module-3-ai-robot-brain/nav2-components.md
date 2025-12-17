# Nav2: Map Server, Planner, Controller, Recovery Server

## Nav2 Architecture Deep Dive

Nav2 is built on a modular architecture where each component has a specific responsibility in the navigation pipeline. Understanding these components is crucial for configuring and optimizing navigation for humanoid robots. Each component can be customized, replaced, or extended to meet specific requirements.

The core Nav2 components include:
1. **Map Server**: Provides static map and costmap functionality
2. **Global Planner**: Computes optimal paths from start to goal
3. **Local Planner/Controller**: Executes path following while avoiding obstacles
4. **Recovery Server**: Handles navigation failures and recovery behaviors
5. **Lifecycle Manager**: Manages the state of navigation components

## Map Server and Costmap Components

### Static Map Server

The map server provides the static map that represents the known environment for navigation:

```python
# map_server_example.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import yaml
import numpy as np

class HumanoidMapServer(Node):
    def __init__(self):
        super().__init__('humanoid_map_server')

        # Publisher for the static map
        self.map_publisher = self.create_publisher(
            OccupancyGrid,
            '/map',
            1
        )

        # Load map from file
        self.map_data = self.load_map_from_file('humanoid_environment.yaml')

        # Timer to publish map periodically
        self.publish_timer = self.create_timer(5.0, self.publish_map)

    def load_map_from_file(self, map_file):
        """
        Load map from YAML file
        """
        with open(map_file, 'r') as f:
            map_config = yaml.safe_load(f)

        # Load the actual map image
        import cv2
        image_path = map_config['image']
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Convert to occupancy grid format
        occupancy_grid = self.image_to_occupancy_grid(image, map_config)

        return occupancy_grid

    def image_to_occupancy_grid(self, image, config):
        """
        Convert image to occupancy grid format
        """
        # Convert image to occupancy values (-1: unknown, 0: free, 100: occupied)
        occupancy_data = np.zeros(image.shape, dtype=np.int8)

        # Threshold the image to create occupancy values
        occupancy_data[image < 50] = 100  # Occupied (dark areas)
        occupancy_data[image > 200] = 0   # Free (light areas)
        occupancy_data[(image >= 50) & (image <= 200)] = -1  # Unknown

        # Create OccupancyGrid message
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = config['resolution']
        map_msg.info.width = image.shape[1]
        map_msg.info.height = image.shape[0]
        map_msg.info.origin.position.x = config['origin'][0]
        map_msg.info.origin.position.y = config['origin'][1]
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # Flatten the occupancy data
        map_msg.data = occupancy_data.flatten().tolist()

        return map_msg

    def publish_map(self):
        """
        Publish the static map
        """
        self.map_publisher.publish(self.map_data)
```

### Costmap Configuration for Humanoids

```yaml
# costmap_config_humanoid.yaml
global_costmap:
  global_costmap:
    ros__parameters:
      # Map settings
      map_topic: map
      track_unknown_space: true
      use_maximum: false

      # Resolution and size
      resolution: 0.05  # Higher resolution for humanoid precision
      width: 40.0
      height: 40.0
      origin_x: -20.0
      origin_y: -20.0

      # Frame settings
      global_frame: map
      robot_base_frame: base_link
      transform_tolerance: 0.5

      # Robot settings (larger for humanoid)
      robot_radius: 0.4  # Humanoid is larger than typical robots

      # Update rates
      update_frequency: 2.0
      publish_frequency: 1.0
      always_send_full_costmap: false

      # Plugins
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0  # Humanoid height consideration
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 5.0
          raytrace_min_range: 0.0
          obstacle_max_range: 4.0
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 5.0  # Higher for humanoid safety
        inflation_radius: 1.0     # Larger for humanoid safety margin

local_costmap:
  local_costmap:
    ros__parameters:
      # Map settings
      global_frame: odom
      robot_base_frame: base_link
      transform_tolerance: 0.5

      # Resolution and size (rolling window for local)
      resolution: 0.05
      width: 8.0   # Larger for humanoid planning horizon
      height: 8.0
      origin_x: -4.0
      origin_y: -4.0

      # Update rates
      update_frequency: 10.0
      publish_frequency: 5.0
      rolling_window: true

      # Robot settings
      robot_radius: 0.4

      # Plugins
      plugins: ["obstacle_layer", "voxel_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 5.0
          raytrace_min_range: 0.0
          obstacle_max_range: 4.0
          obstacle_min_range: 0.0
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 5.0
          raytrace_min_range: 0.0
          obstacle_max_range: 4.0
          obstacle_min_range: 0.0
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 5.0
        inflation_radius: 0.8
```

### Humanoid-Specific Costmap Considerations

```python
# humanoid_costmap.py
class HumanoidCostmap:
    def __init__(self):
        # Humanoid-specific parameters
        self.step_height = 0.15  # Max step height humanoid can handle
        self.step_width = 0.30   # Typical step width
        self.com_height = 0.85   # Center of mass height
        self.foot_size = [0.25, 0.15]  # Foot dimensions

    def update_traversability_costs(self, raw_costmap):
        """
        Update costmap based on humanoid traversability
        """
        # Consider step height constraints
        traversability_map = self.apply_step_height_constraints(raw_costmap)

        # Consider balance constraints
        balance_map = self.apply_balance_constraints(traversability_map)

        # Consider foot placement constraints
        foot_placement_map = self.apply_foot_placement_constraints(balance_map)

        return foot_placement_map

    def apply_step_height_constraints(self, costmap):
        """
        Apply constraints based on humanoid step height capability
        """
        # Identify areas where step height exceeds humanoid capability
        # Mark as high cost or impassable
        pass

    def apply_balance_constraints(self, costmap):
        """
        Apply constraints based on humanoid balance requirements
        """
        # Consider areas that might affect humanoid balance
        # Wider safety margins around obstacles
        pass

    def apply_foot_placement_constraints(self, costmap):
        """
        Apply constraints for safe foot placement
        """
        # Ensure sufficient space for foot placement
        # Consider surface stability
        pass
```

## Global Planner Components

### NavFn Global Planner

The NavFn planner is the default global planner in Nav2, implementing Dijkstra's algorithm:

```python
# navfn_planner_config.yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: False
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5  # Humanoid-specific tolerance
      use_astar: false
      allow_unknown: true
      visualize_potential: false
```

### Custom Global Planner for Humanoids

```python
# humanoid_global_planner.py
import rclpy
from rclpy.node import Node
from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.action import ActionServer
import numpy as np

class HumanoidGlobalPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_global_planner')

        # Action server for path computation
        self._action_server = ActionServer(
            self,
            ComputePathToPose,
            'compute_path_to_pose',
            self.execute_callback
        )

        # Initialize humanoid-specific planning parameters
        self.step_constraints = {
            'max_step_length': 0.4,  # Maximum step length
            'min_step_length': 0.1,  # Minimum step length
            'max_step_height': 0.15, # Maximum step height
            'turn_radius': 0.3       # Minimum turning radius
        }

    def execute_callback(self, goal_handle):
        """
        Execute path planning for humanoid robot
        """
        self.get_logger().info('Received path planning request')

        # Get start and goal poses
        start = goal_handle.request.start
        goal = goal_handle.request.goal

        # Plan path considering humanoid constraints
        path = self.plan_path_with_constraints(start, goal)

        if path is not None:
            goal_handle.succeed()
            result = ComputePathToPose.Result()
            result.path = path
            return result
        else:
            goal_handle.abort()
            result = ComputePathToPose.Result()
            result.path = Path()
            return result

    def plan_path_with_constraints(self, start, goal):
        """
        Plan path considering humanoid kinematic constraints
        """
        # Use a path planning algorithm that considers humanoid constraints
        # such as step length, turning radius, and balance

        # For demonstration, using a modified A* algorithm
        path = self.humanoid_aware_astar(start, goal)
        return path

    def humanoid_aware_astar(self, start, goal):
        """
        A* algorithm adapted for humanoid constraints
        """
        # Implement A* considering humanoid-specific constraints:
        # - Step length limitations
        # - Turning radius
        # - Balance requirements
        # - Traversability for bipedal locomotion

        # This is a simplified version - a real implementation would be more complex
        path = Path()
        path.header.frame_id = 'map'

        # Generate waypoints that respect humanoid constraints
        current_pos = [start.pose.position.x, start.pose.position.y]
        goal_pos = [goal.pose.position.x, goal.pose.position.y]

        # Interpolate path with humanoid-aware waypoints
        waypoints = self.generate_humanoid_waypoints(current_pos, goal_pos)

        for waypoint in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path.poses.append(pose)

        return path

    def generate_humanoid_waypoints(self, start, goal):
        """
        Generate waypoints considering humanoid step constraints
        """
        # Calculate straight-line path
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = np.sqrt(dx*dx + dy*dy)

        # Determine step size based on humanoid constraints
        step_size = min(self.step_constraints['max_step_length'], distance/10)

        # Generate waypoints at appropriate intervals
        num_steps = int(distance / step_size)
        waypoints = []

        for i in range(num_steps + 1):
            t = i / num_steps
            x = start[0] + t * dx
            y = start[1] + t * dy
            waypoints.append([x, y])

        return waypoints
```

### Alternative Global Planners

```yaml
# Alternative planner configuration
planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: False
    planner_plugins: ["GridBased", "RRTPlanner"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
    RRTPlanner:
      plugin: "nav2_rrt_planner::RRTPlanner"  # Example alternative planner
      max_iterations: 1000
      step_size: 0.5
      goal_bias: 0.05
```

## Local Planner and Controller Components

### DWB Controller

The DWB (Dynamic Window Approach) controller is the default local planner in Nav2:

```yaml
# dwb_controller_config.yaml
controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # DWB Controller
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      debug_trajectory_details: True
      min_vel_x: 0.0
      min_vel_y: 0.0
      max_vel_x: 0.5  # Lower for humanoid stability
      max_vel_y: 0.0
      max_vel_theta: 0.6
      min_speed_xy: 0.0
      max_speed_xy: 0.5
      min_speed_theta: 0.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      vx_samples: 20
      vy_samples: 0
      vtheta_samples: 40
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
      transform_tolerance: 0.2
      xy_goal_tolerance: 0.25  # Larger for humanoid
      yaw_goal_tolerance: 0.3
      stateful: True
      short_circuit_trajectory_evaluation: True
      global_plan_overwrite_orientation: True
      prune_plan: True
      prune_distance: 1.0
      oscillation_reset_dist: 0.05
      deviated_path_distance: 1.0
      deviated_path_angle: 1.57
      conservative_reset_dist: 3.0
      cost_scaling_dist: 0.6
      cost_scaling_gain: 1.0
      inflation_cost_scaling_factor: 3.0
      replan_from_global_path: False
      use_dijkstra: True
      use_grid_path: False
      allow_unknown: True
      lethal_cost: 253
      neutral_cost: 50
      publish_cost_grid: False
      tolerance: 0.0

    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5  # Larger for humanoid
      movement_time_allowance: 10.0

    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.3
      stateful: True
```

### MPPI Controller for Humanoids

```yaml
# mppi_controller_config.yaml
controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 10.0  # Lower for humanoid stability
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # MPPI Controller (Model Predictive Path Integral)
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 25
      model_dt: 0.05
      batch_size: 2000
      vx_std: 0.2
      vy_std: 0.1
      wz_std: 0.4
      vx_max: 0.5  # Humanoid speed limit
      vx_min: -0.2
      vy_max: 0.3
      wz_max: 0.8  # Lower for humanoid stability
      xy_goal_tolerance: 0.3
      yaw_goal_tolerance: 0.4
      state_reset_threshold: 0.5
      ctrl_freq: 10.0  # Lower frequency for stability
      prediction_horizon: 1.5  # Longer for humanoid planning
      no_progress_check: 1.5
      goal_angle_tolerance: 0.2
      transform_tolerance: 0.1
      obstacle_cost_mult: 5.0  # Higher for humanoid safety
      goal_cost_mult: 1.0
      reference_cost_mult: 1.0
      curvature_cost_mult: 0.5
      forward_preference_cost_mult: 0.1

    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.4
      movement_time_allowance: 15.0

    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.3
      yaw_goal_tolerance: 0.4
      stateful: True
```

### Custom Humanoid Controller

```python
# humanoid_controller.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import FollowPath
from rclpy.action import ActionServer
from tf2_ros import TransformListener, Buffer
import numpy as np

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Action server for path following
        self._action_server = ActionServer(
            self,
            FollowPath,
            'follow_path',
            self.execute_callback
        )

        # Publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.current_pose_subscriber = self.create_subscription(
            PoseStamped, 'current_pose', self.pose_callback, 10
        )

        # TF listener for pose transformation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Controller parameters for humanoid
        self.controller_params = {
            'linear_kp': 1.0,      # Proportional gain for linear velocity
            'angular_kp': 2.0,     # Proportional gain for angular velocity
            'max_linear_vel': 0.4, # Maximum linear velocity for humanoid
            'max_angular_vel': 0.6, # Maximum angular velocity
            'min_linear_vel': 0.05, # Minimum linear velocity
            'min_angular_vel': 0.05, # Minimum angular velocity
            'path_tolerance': 0.3,  # Tolerance for path following
            'goal_tolerance': 0.25  # Tolerance for goal reaching
        }

        # State variables
        self.current_pose = None
        self.current_path = None
        self.path_index = 0

    def execute_callback(self, goal_handle):
        """
        Execute path following for humanoid robot
        """
        self.get_logger().info('Received path following request')

        # Store the path
        self.current_path = goal_handle.request.path
        self.path_index = 0

        # Follow the path
        success = self.follow_path()

        if success:
            goal_handle.succeed()
            result = FollowPath.Result()
            return result
        else:
            goal_handle.abort()
            result = FollowPath.Result()
            return result

    def follow_path(self):
        """
        Follow the current path with humanoid-specific control
        """
        if not self.current_path or len(self.current_path.poses) == 0:
            return False

        # Control loop
        control_timer = self.create_timer(0.1, self.control_callback)  # 10 Hz

        # Wait until goal is reached or path is completed
        path_completed = False
        while rclpy.ok() and not path_completed:
            # Check if we've reached the goal
            if self.has_reached_goal():
                path_completed = True
                break

            # Check path progress
            if self.path_index >= len(self.current_path.poses) - 1:
                path_completed = True
                break

            # Small delay to allow timer to run
            rclpy.spin_once(self, timeout_sec=0.01)

        control_timer.destroy()
        return path_completed

    def control_callback(self):
        """
        Main control callback for humanoid navigation
        """
        if not self.current_pose or not self.current_path:
            return

        # Get the next waypoint to track
        target_pose = self.get_next_waypoint()

        if target_pose is None:
            return

        # Compute control commands
        cmd_vel = self.compute_control_command(target_pose)

        # Publish command
        self.cmd_vel_publisher.publish(cmd_vel)

        # Check if we've reached the current waypoint
        if self.has_reached_waypoint(target_pose):
            self.path_index += 1

    def compute_control_command(self, target_pose):
        """
        Compute velocity command for humanoid to reach target pose
        """
        # Calculate error between current pose and target pose
        error_x = target_pose.pose.position.x - self.current_pose.pose.position.x
        error_y = target_pose.pose.position.y - self.current_pose.pose.position.y

        # Calculate distance to target
        distance = np.sqrt(error_x**2 + error_y**2)

        # Calculate angle to target
        target_angle = np.arctan2(error_y, error_x)
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.pose.orientation)

        # Calculate angle difference
        angle_diff = target_angle - current_yaw
        # Normalize angle difference to [-π, π]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # Compute linear velocity (proportional to distance, limited)
        linear_vel = min(self.controller_params['max_linear_vel'],
                        self.controller_params['linear_kp'] * distance)

        # Only move forward if we're pointing in roughly the right direction
        if abs(angle_diff) > np.pi / 4:  # 45 degrees
            linear_vel = 0.0

        # Compute angular velocity (proportional to angle error, limited)
        angular_vel = min(self.controller_params['max_angular_vel'],
                         max(-self.controller_params['max_angular_vel'],
                         self.controller_params['angular_kp'] * angle_diff))

        # Create Twist message
        cmd_vel = Twist()
        cmd_vel.linear.x = max(self.controller_params['min_linear_vel'], linear_vel) if linear_vel > 0 else linear_vel
        cmd_vel.angular.z = angular_vel

        return cmd_vel

    def get_next_waypoint(self):
        """
        Get the next waypoint from the path to follow
        """
        if self.path_index < len(self.current_path.poses):
            return self.current_path.poses[self.path_index]
        else:
            # Return the last waypoint if we've reached the end
            return self.current_path.poses[-1]

    def has_reached_waypoint(self, waypoint):
        """
        Check if we've reached the current waypoint
        """
        if not self.current_pose:
            return False

        # Calculate distance to waypoint
        dx = waypoint.pose.position.x - self.current_pose.pose.position.x
        dy = waypoint.pose.position.y - self.current_pose.pose.position.y
        distance = np.sqrt(dx**2 + dy**2)

        return distance < self.controller_params['path_tolerance']

    def has_reached_goal(self):
        """
        Check if we've reached the final goal
        """
        if not self.current_pose or not self.current_path:
            return False

        # Get the final pose in the path
        final_pose = self.current_path.poses[-1]

        # Calculate distance to final pose
        dx = final_pose.pose.position.x - self.current_pose.pose.position.x
        dy = final_pose.pose.position.y - self.current_pose.pose.position.y
        distance = np.sqrt(dx**2 + dy**2)

        # Check both position and orientation
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.pose.orientation)
        goal_yaw = self.get_yaw_from_quaternion(final_pose.pose.orientation)
        angle_diff = abs(current_yaw - goal_yaw)

        return (distance < self.controller_params['goal_tolerance'] and
                angle_diff < self.controller_params['goal_tolerance'])

    def get_yaw_from_quaternion(self, quaternion):
        """
        Extract yaw angle from quaternion
        """
        import math
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def pose_callback(self, msg):
        """
        Update current pose from localization
        """
        self.current_pose = msg
```

## Recovery Server Components

### Recovery Behaviors

The recovery server provides behaviors to handle navigation failures:

```yaml
# recovery_config.yaml
behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait", "humanoid_balance_recovery"]
    spin:
      plugin: "nav2_behaviors::Spin"
      spin_dist: 1.57  # 90 degrees
      time_allowance: 10
    backup:
      plugin: "nav2_behaviors::BackUp"
      backup_dist: 0.3  # Humanoid-specific backup distance
      backup_speed: 0.05  # Slower for humanoid stability
      time_allowance: 10
    wait:
      plugin: "nav2_behaviors::Wait"
      wait_duration: 5.0
    humanoid_balance_recovery:
      plugin: "humanoid_behaviors::BalanceRecovery"  # Custom humanoid behavior
      max_attempts: 3
      recovery_time: 10.0
```

### Custom Recovery Behaviors for Humanoids

```python
# humanoid_recovery_behaviors.py
import rclpy
from rclpy.node import Node
from nav2_msgs.action import Recover
from rclpy.action import ActionServer
from geometry_msgs.msg import Twist
import time

class HumanoidRecoveryBehaviors(Node):
    def __init__(self):
        super().__init__('humanoid_recovery_behaviors')

        # Action server for recovery behaviors
        self._action_server = ActionServer(
            self,
            Recover,
            'recover',
            self.execute_callback
        )

        # Publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

    def execute_callback(self, goal_handle):
        """
        Execute the specified recovery behavior
        """
        behavior = goal_handle.request.behavior

        self.get_logger().info(f'Executing recovery behavior: {behavior}')

        success = False
        if behavior == 'balance_recovery':
            success = self.balance_recovery()
        elif behavior == 'step_over_obstacle':
            success = self.step_over_obstacle()
        elif behavior == 'wait_for_balance':
            success = self.wait_for_balance()
        else:
            self.get_logger().warn(f'Unknown recovery behavior: {behavior}')
            goal_handle.abort()
            result = Recover.Result()
            result.status = Recover.Result.FAILED
            return result

        if success:
            goal_handle.succeed()
            result = Recover.Result()
            result.status = Recover.Result.SUCCEEDED
        else:
            goal_handle.aborted()
            result = Recover.Result()
            result.status = Recover.Result.FAILED

        return result

    def balance_recovery(self):
        """
        Humanoid-specific balance recovery behavior
        """
        self.get_logger().info('Attempting balance recovery')

        # Stop the robot
        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)
        time.sleep(1.0)

        # Attempt to regain balance
        # This would interface with the humanoid's balance controller
        # For simulation, we'll just wait and assume balance is restored
        time.sleep(2.0)

        # Check if balance is restored
        # In a real implementation, this would check sensor data
        balance_restored = True  # Assume successful for this example

        return balance_restored

    def step_over_obstacle(self):
        """
        Humanoid-specific step-over obstacle behavior
        """
        self.get_logger().info('Attempting to step over obstacle')

        # Stop the robot
        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)
        time.sleep(0.5)

        # Execute step-over motion
        # This would interface with the humanoid's step controller
        # For simulation, we'll publish appropriate commands
        step_cmd = Twist()
        step_cmd.linear.x = 0.1  # Small forward step
        self.cmd_vel_publisher.publish(step_cmd)
        time.sleep(1.0)

        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)

        # Assume step-over was successful
        return True

    def wait_for_balance(self):
        """
        Wait for humanoid to achieve stable balance
        """
        self.get_logger().info('Waiting for stable balance')

        # Stop the robot
        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)

        # Wait for a period to allow balance to stabilize
        time.sleep(3.0)

        # In a real implementation, check balance sensors
        # For now, assume balance is achieved
        return True
```

## Lifecycle Management

### Nav2 Lifecycle Manager

Nav2 uses lifecycle nodes to manage the state of navigation components:

```yaml
# lifecycle_manager_config.yaml
lifecycle_manager:
  ros__parameters:
    use_sim_time: False
    autostart: True
    node_names: [
      'map_server',
      'planner_server',
      'controller_server',
      'recoveries_server',
      'bt_navigator',
      'waypoint_follower'
    ]
```

### Custom Lifecycle Node for Humanoid Navigation

```python
# humanoid_lifecycle_node.py
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from rclpy.action import ActionServer
import time

class HumanoidNavigationLifecycle(LifecycleNode):
    def __init__(self):
        super().__init__('humanoid_navigation_lifecycle')

        # Initialize components as inactive
        self.map_server_active = False
        self.planner_active = False
        self.controller_active = False
        self.recovery_active = False

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Configure the navigation components
        """
        self.get_logger().info('Configuring humanoid navigation components')

        # Load configurations
        self.load_configurations()

        # Initialize components in unconfigured state
        self.get_logger().info('Navigation components configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Activate the navigation components
        """
        self.get_logger().info('Activating humanoid navigation components')

        # Activate each component
        self.activate_map_server()
        self.activate_planner()
        self.activate_controller()
        self.activate_recovery()

        self.get_logger().info('Navigation components activated')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Deactivate the navigation components
        """
        self.get_logger().info('Deactivating humanoid navigation components')

        # Deactivate components safely
        self.deactivate_recovery()
        self.deactivate_controller()
        self.deactivate_planner()
        self.deactivate_map_server()

        self.get_logger().info('Navigation components deactivated')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Clean up resources
        """
        self.get_logger().info('Cleaning up navigation components')

        # Clean up resources
        self.cleanup_resources()

        self.get_logger().info('Navigation components cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def load_configurations(self):
        """
        Load navigation configurations for humanoid
        """
        # Load parameters specific to humanoid navigation
        pass

    def activate_map_server(self):
        """
        Activate map server for humanoid
        """
        self.map_server_active = True

    def activate_planner(self):
        """
        Activate global planner for humanoid
        """
        self.planner_active = True

    def activate_controller(self):
        """
        Activate controller for humanoid
        """
        self.controller_active = True

    def activate_recovery(self):
        """
        Activate recovery behaviors for humanoid
        """
        self.recovery_active = True

    def deactivate_map_server(self):
        """
        Deactivate map server
        """
        self.map_server_active = False

    def deactivate_planner(self):
        """
        Deactivate global planner
        """
        self.planner_active = False

    def deactivate_controller(self):
        """
        Deactivate controller
        """
        self.controller_active = False

    def deactivate_recovery(self):
        """
        Deactivate recovery behaviors
        """
        self.recovery_active = False

    def cleanup_resources(self):
        """
        Clean up resources used by navigation components
        """
        pass
```

## Performance Optimization and Monitoring

### Component Performance Monitoring

```python
# performance_monitor.py
class Nav2PerformanceMonitor:
    def __init__(self):
        self.planning_times = []
        self.control_times = []
        self.costmap_update_times = []

    def monitor_planning_performance(self, planning_time):
        """
        Monitor global planning performance
        """
        self.planning_times.append(planning_time)

        # Calculate average planning time
        avg_time = sum(self.planning_times) / len(self.planning_times)

        # Log warning if planning is too slow
        if avg_time > 0.1:  # 100ms threshold
            print(f"WARNING: Average planning time is {avg_time:.3f}s")

    def monitor_control_performance(self, control_time):
        """
        Monitor local control performance
        """
        self.control_times.append(control_time)

        # Calculate average control time
        avg_time = sum(self.control_times) / len(self.control_times)

        # Log warning if control is too slow
        if avg_time > 0.05:  # 50ms threshold for 20Hz control
            print(f"WARNING: Average control time is {avg_time:.3f}s")

    def monitor_costmap_performance(self, update_time):
        """
        Monitor costmap update performance
        """
        self.costmap_update_times.append(update_time)

        # Calculate average update time
        avg_time = sum(self.costmap_update_times) / len(self.costmap_update_times)

        # Log warning if costmap updates are too slow
        if avg_time > 0.1:  # 100ms threshold
            print(f"WARNING: Average costmap update time is {avg_time:.3f}s")
```

## Integration with Isaac ROS

### Perception-Enhanced Navigation

```python
# perception_enhanced_navigation.py
class PerceptionEnhancedNavigation:
    def __init__(self):
        # Integrate with Isaac ROS perception nodes
        self.depth_obstacle_detector = None
        self.semantic_obstacle_detector = None
        self.human_detector = None

    def integrate_perception_data(self, perception_output):
        """
        Integrate Isaac ROS perception data into navigation
        """
        # Update costmaps with 3D obstacle information
        self.update_costmaps_with_3d_data(
            perception_output.get('obstacles_3d', []),
            perception_output.get('free_space', [])
        )

        # Update navigation behavior based on detected humans
        self.update_social_navigation(
            perception_output.get('detected_humans', [])
        )

        # Adjust navigation parameters based on semantic information
        self.adjust_navigation_for_surface_types(
            perception_output.get('surface_types', [])
        )

    def update_costmaps_with_3d_data(self, obstacles_3d, free_space):
        """
        Update costmaps with 3D obstacle information from Isaac ROS
        """
        # Convert 3D obstacle information to costmap representation
        # Consider height constraints for humanoid navigation
        pass

    def update_social_navigation(self, detected_humans):
        """
        Update navigation to consider detected humans
        """
        # Implement social navigation behaviors
        # Maintain appropriate distance from humans
        # Yield to humans in shared spaces
        pass

    def adjust_navigation_for_surface_types(self, surface_types):
        """
        Adjust navigation parameters based on surface types
        """
        # Modify speed and step parameters based on terrain
        # Consider slippery, uneven, or unstable surfaces
        pass
```

## Troubleshooting Nav2 Components

### Common Issues and Solutions

```bash
# Issue: Costmap not updating properly
# Solutions:
# 1. Check sensor data topics and message types
# 2. Verify transform frames are correct
# 3. Check costmap configuration parameters
# 4. Ensure proper QoS settings for sensor data

# Issue: Robot getting stuck in local minima
# Solutions:
# 1. Adjust inflation parameters
# 2. Use different global planner
# 3. Implement better recovery behaviors
# 4. Increase local costmap size

# Issue: Controller oscillation
# Solutions:
# 1. Tune PID parameters
# 2. Adjust controller frequency
# 3. Modify velocity limits
# 4. Check path smoothing

# Issue: Poor path quality
# Solutions:
# 1. Adjust global planner parameters
# 2. Use path smoother
# 3. Increase map resolution
# 4. Check heuristic function
```

## Best Practices for Nav2 Components

### Configuration Guidelines

1. **Costmap Configuration**: Set appropriate resolution and inflation for humanoid safety
2. **Controller Tuning**: Adjust for humanoid dynamics and stability requirements
3. **Planning Parameters**: Configure for humanoid speed and turning capabilities
4. **Recovery Behaviors**: Implement humanoid-appropriate recovery actions
5. **Performance Monitoring**: Continuously monitor component performance
6. **Integration**: Properly integrate with perception and control systems

### Performance Optimization

1. **Component Scheduling**: Optimize update frequencies for each component
2. **Resource Management**: Efficiently use computational resources
3. **Real-time Constraints**: Ensure all components meet timing requirements
4. **Scalability**: Design components to scale with environment complexity

## Summary

Nav2's component-based architecture provides a flexible and powerful navigation system that can be customized for humanoid robots. The key components include:

1. **Map Server**: Provides static environmental information
2. **Global Planner**: Computes long-term navigation paths
3. **Local Controller**: Executes path following with obstacle avoidance
4. **Recovery Server**: Handles navigation failures and recovery
5. **Lifecycle Manager**: Coordinates component states

Each component can be configured and customized to meet the specific requirements of humanoid navigation, considering factors such as bipedal dynamics, balance constraints, and human-scale environments. Proper integration of these components with perception systems like Isaac ROS enables sophisticated autonomous navigation for humanoid robots.

In the next chapter, we'll explore how to integrate all these components together into a complete AI-robot brain system that combines Isaac Sim, Isaac ROS, and Nav2.