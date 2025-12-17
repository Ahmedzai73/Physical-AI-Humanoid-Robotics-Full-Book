# Introduction to Nav2 for Humanoid Navigation

## Understanding Navigation in Robotics

Navigation is the capability that allows robots to move autonomously from one location to another while avoiding obstacles and respecting environmental constraints. For humanoid robots, navigation is particularly challenging due to their complex dynamics, bipedal locomotion, and the need to operate in human environments.

Nav2 (Navigation 2) is ROS 2's state-of-the-art navigation framework that provides a flexible, modular, and extensible system for robot navigation. Unlike its predecessor Nav1, Nav2 is designed from the ground up for ROS 2, leveraging modern software engineering practices and providing better performance, reliability, and maintainability.

## Nav2 Architecture Overview

### The Navigation System Components

Nav2 is composed of several interconnected components that work together to provide complete navigation functionality:

1. **Navigation Server**: The main orchestrator that coordinates all navigation activities
2. **Map Server**: Provides static and costmap functionality
3. **Global Planner**: Computes optimal paths from start to goal
4. **Local Planner**: Executes path following while avoiding obstacles
5. **Controller Server**: Manages local planning and control
6. **Recovery Server**: Handles navigation failures and recovery behaviors
7. **Lifecycle Manager**: Manages the state of navigation components

### Nav2 for Humanoid Robots

While Nav2 was initially designed for wheeled robots, it has been extended and adapted to support humanoid navigation. The key differences for humanoid navigation include:

- **Bipedal Motion Models**: Different kinematic and dynamic constraints
- **Footstep Planning**: Consideration of foot placement and balance
- **Terrain Adaptation**: Different requirements for traversable terrain
- **Human-Scale Navigation**: Navigation in human environments with human-scale obstacles

## Nav2 Core Concepts

### Costmaps

Costmaps are fundamental to Nav2's navigation system. They represent the environment as a 2D grid where each cell contains information about the traversability of that area:

```yaml
# Example costmap configuration
global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      width: 20
      height: 20
      resolution: 0.05  # meters per cell
      origin_x: -10.0
      origin_y: -10.0
      robot_base_frame: base_link
      global_frame: map
      rolling_window: false
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 10.0
      publish_frequency: 5.0
      width: 6
      height: 6
      resolution: 0.05
      robot_base_frame: base_link
      global_frame: odom
      rolling_window: true
      plugins: ["obstacle_layer", "voxel_layer", "inflation_layer"]
```

### Planning Hierarchy

Nav2 implements a hierarchical planning approach:

1. **Global Planning**: Long-term path planning from current position to goal
2. **Local Planning**: Short-term path following and obstacle avoidance
3. **Control**: Low-level velocity commands to achieve planned trajectories

### Action Servers

Nav2 uses ROS 2 action servers for navigation tasks:

- **NavigateToPose**: Navigate to a specific pose (position + orientation)
- **NavigateThroughPoses**: Navigate through a sequence of poses
- **ComputePathToPose**: Compute a path without executing it
- **FollowPath**: Follow a pre-computed path

## Setting Up Nav2 for Humanoid Robots

### Installation and Dependencies

```bash
# Install Nav2 packages
sudo apt update
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
sudo apt install ros-humble-nav2-gui
sudo apt install ros-humble-nav2-common
```

### Basic Launch File Structure

```xml
<!-- humanoid_nav2_launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart', default='true')
    use_composition = LaunchConfiguration('use_composition', default='True')

    # Paths
    pkg_nav2_bringup = FindPackageShare('nav2_bringup')
    pkg_humanoid_nav2_config = FindPackageShare('humanoid_nav2_config')

    # Launch files
    navigation_launch_file = PathJoinSubstitution(
        [pkg_nav2_bringup, 'launch', 'navigation_launch.py'])

    # Navigation launch
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_launch_file),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
            'autostart': autostart,
            'use_composition': use_composition
        }.items()
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [pkg_humanoid_nav2_config, 'config', 'nav2_params.yaml']),
            description='Full path to the ROS2 parameters file to use for all launched nodes'),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Automatically startup the nav2 stack'),
        DeclareLaunchArgument(
            'use_composition',
            default_value='True',
            description='Whether to launch the controller manager with lifecycle nodes'),
        navigation_launch,
    ])
```

### Humanoid-Specific Configuration

```yaml
# humanoid_nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: False
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    set_initial_pose: true
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: False
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Humanoid-specific behavior tree
    default_nav_to_pose_bt_xml: /opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_w_replanning_and_recovery.xml

controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    # Humanoid-specific controllers
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid path follower
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 25
      model_dt: 0.05
      batch_size: 2000
      vx_std: 0.2
      vy_std: 0.1
      wz_std: 0.4
      vx_max: 0.8
      vx_min: -0.3
      vy_max: 0.5
      wz_max: 1.0
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      state_reset_threshold: 0.5
      ctrl_freq: 20.0
      prediction_horizon: 1.0
      no_progress_check: 1.0
      goal_angle_tolerance: 0.1
      transform_tolerance: 0.1
      obstacle_cost_mult: 3.0
      goal_cost_mult: 1.0
      reference_cost_mult: 1.0
      curvature_cost_mult: 0.0
      forward_preference_cost_mult: 0.0

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 10.0
      publish_frequency: 5.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: False
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      # Humanoid-specific parameters
      robot_radius: 0.4  # Larger than typical wheeled robots
      plugins: ["obstacle_layer", "voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.8  # Larger for humanoid safety
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
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 0.5
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: False
      robot_radius: 0.4
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 1.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
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
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: False
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      # Humanoid-specific parameters
      visualize_potential: false

smoother_server:
  ros__parameters:
    use_sim_time: False
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: True

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "drive_on_heading", "wait"]
    spin:
      plugin: "nav2_behaviors::Spin"
      spin_dist: 1.57
    backup:
      plugin: "nav2_behaviors::BackUp"
      backup_dist: 0.15
      backup_speed: 0.025
    drive_on_heading:
      plugin: "nav2_behaviors::DriveOnHeading"
      drive_on_heading_timeout: 10.0
    wait:
      plugin: "nav2_behaviors::Wait"
      wait_duration: 1.0
```

## Nav2 for Bipedal Locomotion

### Humanoid Motion Constraints

Humanoid robots have different motion constraints compared to wheeled robots:

```python
# humanoid_motion_constraints.py
class HumanoidMotionConstraints:
    def __init__(self):
        # Humanoid-specific motion parameters
        self.max_linear_velocity = 0.5  # m/s (slower than wheeled robots)
        self.max_angular_velocity = 0.6  # rad/s
        self.min_turning_radius = 0.3  # m (larger than wheeled robots)
        self.step_height = 0.1  # m (for step-over capability)
        self.step_length = 0.3  # m (typical step length)

    def validate_trajectory(self, trajectory):
        """
        Validate trajectory for humanoid motion constraints
        """
        # Check linear velocity constraints
        for pose in trajectory.poses:
            # Verify that poses are achievable with humanoid kinematics
            if not self.is_pose_achievable(pose):
                return False

        # Check dynamic constraints
        if not self.is_trajectory_dynamically_feasible(trajectory):
            return False

        return True

    def is_pose_achievable(self, pose):
        """
        Check if a pose is achievable by humanoid robot
        """
        # Consider bipedal kinematic constraints
        # Check if pose is within reachable workspace
        # Consider balance constraints
        return True  # Placeholder

    def is_trajectory_dynamically_feasible(self, trajectory):
        """
        Check if trajectory is dynamically feasible for bipedal locomotion
        """
        # Consider center of mass constraints
        # Check for balance during motion
        # Verify step timing is appropriate
        return True  # Placeholder
```

### Footstep Planning Integration

```python
# footstep_planning.py
class FootstepPlannerIntegration:
    def __init__(self):
        self.footstep_planner = self.initialize_footstep_planner()

    def initialize_footstep_planner(self):
        """
        Initialize footstep planner for humanoid navigation
        """
        # This would integrate with footstep planning algorithms
        # that consider bipedal locomotion constraints
        pass

    def integrate_with_nav2(self, global_path):
        """
        Integrate footstep planning with Nav2 global path
        """
        # Convert Nav2 path to footstep plan
        # Consider terrain traversability for bipedal locomotion
        # Generate stable footstep sequence
        footstep_plan = self.plan_footsteps(global_path)
        return footstep_plan

    def plan_footsteps(self, path):
        """
        Plan footstep sequence for humanoid to follow path
        """
        # This would implement footstep planning algorithms
        # considering step constraints, balance, and terrain
        pass
```

## Nav2 Behavior Trees

### Understanding Behavior Trees

Behavior trees are a key component of Nav2 that define the decision-making logic for navigation:

```xml
<!-- Example behavior tree for humanoid navigation -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <PipelineSequence name="NavigateWithReplanning">
            <RateController hz="1.0">
                <RecoveryNode number_of_retries="6">
                    <PipelineSequence name="ComputeAndExecutePath">
                        <RateController hz="1.0">
                            <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
                        </RateController>
                        <FollowPath path="{path}" controller_id="FollowPath"/>
                    </PipelineSequence>
                    <RecoveryNode number_of_retries="2">
                        <BackUp backup_dist="0.15" backup_speed="0.025"/>
                        <Spin spin_dist="1.57"/>
                    </RecoveryNode>
                </RecoveryNode>
            </RateController>
            <ReactiveSequence>
                <GoalUpdated/>
                <ClearEntireCostmap name="ClearGlobalCostmap" service_name="global_costmap/clear_entirely_global_costmap"/>
                <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
            </ReactiveSequence>
        </PipelineSequence>
    </BehaviorTree>
</root>
```

### Custom Behavior Trees for Humanoids

```python
# custom_humanoid_bt.py
class HumanoidBehaviorTree:
    def __init__(self):
        # Define humanoid-specific behavior tree logic
        self.behavior_tree = self.create_humanoid_behavior_tree()

    def create_humanoid_behavior_tree(self):
        """
        Create behavior tree optimized for humanoid navigation
        """
        # Humanoid-specific behaviors might include:
        # - Balance recovery actions
        # - Step-over obstacle behaviors
        # - Human-aware navigation
        # - Social navigation patterns

        bt_xml = """
        <root main_tree_to_execute="MainTree">
            <BehaviorTree ID="MainTree">
                <PipelineSequence name="HumanoidNavigateWithRecovery">
                    <RateController hz="1.0">
                        <RecoveryNode number_of_retries="6">
                            <PipelineSequence name="ComputeAndExecutePath">
                                <RateController hz="1.0">
                                    <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
                                </RateController>
                                <FollowPath path="{path}" controller_id="FollowPath"/>
                            </PipelineSequence>
                            <RecoveryNode number_of_retries="2">
                                <!-- Humanoid-specific recovery behaviors -->
                                <BalanceRecovery/>
                                <StepOverObstacle height="0.2"/>
                                <BackUp backup_dist="0.15" backup_speed="0.025"/>
                                <Spin spin_dist="1.57"/>
                            </RecoveryNode>
                        </RecoveryNode>
                    </RateController>
                    <ReactiveSequence>
                        <GoalUpdated/>
                        <ClearEntireCostmap name="ClearGlobalCostmap" service_name="global_costmap/clear_entirely_global_costmap"/>
                        <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
                    </ReactiveSequence>
                </PipelineSequence>
            </BehaviorTree>
        </root>
        """
        return bt_xml
```

## Nav2 Performance Considerations

### Optimizing for Real-Time Performance

```python
# nav2_optimization.py
class Nav2Optimizer:
    def __init__(self):
        self.costmap_resolution = 0.05  # meters
        self.planning_frequency = 1.0   # Hz
        self.control_frequency = 20.0   # Hz

    def optimize_for_humanoid(self):
        """
        Optimize Nav2 parameters for humanoid performance
        """
        # Adjust costmap resolution for humanoid size
        self.costmap_resolution = 0.05  # Finer resolution for precise navigation

        # Adjust planning frequency for humanoid speed
        self.planning_frequency = 0.5   # Lower frequency due to slower humanoid speed

        # Adjust control frequency for humanoid dynamics
        self.control_frequency = 10.0   # Lower frequency for stable control

    def adaptive_parameter_tuning(self, environment_type):
        """
        Adapt Nav2 parameters based on environment
        """
        if environment_type == "cluttered":
            # More conservative parameters for cluttered environments
            return {
                'inflation_radius': 1.0,
                'min_obstacle_clearance': 0.8,
                'max_linear_velocity': 0.3,
                'max_angular_velocity': 0.4
            }
        elif environment_type == "open":
            # More aggressive parameters for open environments
            return {
                'inflation_radius': 0.5,
                'min_obstacle_clearance': 0.4,
                'max_linear_velocity': 0.6,
                'max_angular_velocity': 0.8
            }
        else:
            # Default parameters
            return {
                'inflation_radius': 0.8,
                'min_obstacle_clearance': 0.6,
                'max_linear_velocity': 0.5,
                'max_angular_velocity': 0.6
            }
```

## Nav2 Integration with Perception Systems

### Sensor Integration

```python
# sensor_integration.py
class Nav2SensorIntegration:
    def __init__(self):
        self.lidar_data = None
        self.camera_data = None
        self.depth_data = None

    def integrate_lidar(self, lidar_msg):
        """
        Integrate LIDAR data into Nav2 costmaps
        """
        # Process LIDAR data for obstacle detection
        # Update costmaps with LIDAR observations
        pass

    def integrate_camera(self, image_msg, depth_msg):
        """
        Integrate camera and depth data for enhanced navigation
        """
        # Use Isaac ROS perception nodes to process camera data
        # Integrate 3D obstacle information into costmaps
        pass

    def create_dynamic_costmap(self):
        """
        Create costmap with dynamic obstacle information
        """
        # Combine static map with dynamic obstacle detection
        # from perception systems
        pass
```

## Troubleshooting Nav2

### Common Issues and Solutions

```bash
# Issue: Robot oscillates or cannot reach goal
# Solutions:
# 1. Adjust controller parameters (inflation radius, tolerances)
# 2. Verify transform tree is correct
# 3. Check odometry quality
# 4. Adjust costmap inflation parameters

# Issue: Local planner fails frequently
# Solutions:
# 1. Increase local costmap size
# 2. Adjust controller frequency
# 3. Verify sensor data quality
# 4. Check robot velocity limits

# Issue: Global planner cannot find path
# Solutions:
# 1. Verify map quality and resolution
# 2. Check inflation parameters
# 3. Verify start and goal poses are valid
# 4. Check map coordinate frame

# Issue: High CPU usage
# Solutions:
# 1. Reduce costmap resolution
# 2. Lower planning frequency
# 3. Optimize transform publishing
# 4. Use efficient costmap layers
```

## Best Practices for Nav2 with Humanoid Robots

### Configuration Guidelines

1. **Costmap Configuration**: Set appropriate robot radius and inflation parameters for humanoid size
2. **Controller Parameters**: Adjust for humanoid dynamics and slower movement speeds
3. **Planning Frequency**: Match planning frequency to humanoid movement capabilities
4. **Safety Margins**: Increase safety margins for stable bipedal locomotion
5. **Recovery Behaviors**: Implement humanoid-appropriate recovery behaviors
6. **Sensor Integration**: Properly integrate 3D sensors for humanoid-aware navigation

### Performance Optimization

1. **Real-time Requirements**: Ensure navigation runs at appropriate frequencies for humanoid dynamics
2. **Stability**: Prioritize stable, predictable navigation over speed for humanoid safety
3. **Human-aware Navigation**: Consider human presence and social navigation norms
4. **Terrain Adaptation**: Account for terrain traversability for bipedal locomotion

## Summary

Nav2 provides a comprehensive navigation framework that can be adapted for humanoid robots. The key aspects include:

1. **Modular Architecture**: Flexible components that can be customized for humanoid needs
2. **Costmap System**: Environment representation for navigation planning
3. **Planning Hierarchy**: Global and local planning for effective navigation
4. **Behavior Trees**: Decision-making logic for navigation behaviors
5. **Recovery Systems**: Handling navigation failures and obstacles
6. **Parameter Configuration**: Tuning for humanoid-specific requirements

For humanoid robotics, Nav2 requires specific adaptations to account for bipedal locomotion, human-scale environments, and the unique dynamics of walking robots. Properly configured, Nav2 enables humanoid robots to navigate autonomously in complex human environments.

In the next chapter, we'll explore the individual Nav2 components in detail, including the map server, planners, controllers, and recovery behaviors.