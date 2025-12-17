# Mini Project: Humanoid Walks Through an Obstacle Course

## Project Overview

In this comprehensive mini-project, we'll implement a complete AI-robot brain system that enables a humanoid robot to autonomously navigate through a complex obstacle course using the integrated Isaac Sim, Isaac ROS, and Nav2 system we've developed throughout this module. This project will demonstrate all the concepts learned and showcase the complete pipeline from simulation to perception to navigation.

### Project Objectives

By completing this project, you will:

1. Implement a complete humanoid navigation system integrating Isaac Sim, Isaac ROS, and Nav2
2. Create a photorealistic obstacle course environment in Isaac Sim
3. Configure Isaac ROS perception nodes for real-time obstacle detection
4. Set up Nav2 for humanoid-specific navigation
5. Implement the integration layer that coordinates all components
6. Test and validate the complete system in simulation

### Learning Outcomes

- Practical experience with the complete AI-robot brain pipeline
- Understanding of system integration challenges and solutions
- Hands-on experience with GPU-accelerated perception
- Knowledge of humanoid-specific navigation considerations
- Experience with real-time system performance optimization

## Project Setup and Prerequisites

### System Requirements

Before starting the project, ensure you have:

- NVIDIA GPU with compute capability 6.0+ (RTX series recommended)
- Ubuntu 20.04 or 22.04 with ROS 2 Humble
- Isaac Sim installed and properly configured
- Isaac ROS packages installed
- Nav2 packages installed
- At least 32GB RAM and 100GB free disk space

### Project Structure

```
humanoid_obstacle_course/
├── isaac_sim_envs/           # Isaac Sim environment files
│   ├── obstacle_course.usd
│   └── humanoid_robot.usd
├── isaac_ros_launch/         # Isaac ROS launch files
│   ├── perception_pipeline.launch.py
│   └── vslam_pipeline.launch.py
├── nav2_config/             # Nav2 configuration
│   ├── nav2_params.yaml
│   ├── costmap_config.yaml
│   └── behavior_trees.xml
├── integration_launch/      # Integration launch files
│   ├── complete_system.launch.py
│   └── integration_manager.launch.py
├── scripts/                 # Helper scripts
│   ├── setup_env.sh
│   ├── run_project.sh
│   └── validate_system.py
└── docs/                    # Project documentation
    └── implementation_guide.md
```

## Phase 1: Environment Setup

### Creating the Obstacle Course in Isaac Sim

```python
# create_obstacle_course.py
import omni
from pxr import Gf, UsdGeom, Sdf, UsdPhysics, PhysicsSchemaTools
import numpy as np

def create_obstacle_course():
    """
    Create a photorealistic obstacle course for humanoid navigation
    """
    stage = omni.usd.get_context().get_stage()

    # Create world origin
    world = UsdGeom.Xform.Define(stage, "/World")

    # Create ground plane
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.GetSizeAttr().Set(100.0)
    ground.CreateScaleAttr().Set(Gf.Vec3f(20.0, 20.0, 0.1))
    ground.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, -0.05))

    # Create walls
    create_walls(stage)

    # Create obstacles of varying types
    create_static_obstacles(stage)
    create_dynamic_obstacles(stage)

    # Create navigation waypoints
    create_navigation_waypoints(stage)

    # Apply materials for photorealistic rendering
    apply_photorealistic_materials(stage)

def create_walls(stage):
    """
    Create walls for the obstacle course
    """
    # Left wall
    left_wall = UsdGeom.Cube.Define(stage, "/World/LeftWall")
    left_wall.GetSizeAttr().Set(1.0)
    left_wall.CreateScaleAttr().Set(Gf.Vec3f(0.2, 20.0, 2.0))
    left_wall.AddTranslateOp().Set(Gf.Vec3f(-10.0, 0.0, 1.0))

    # Right wall
    right_wall = UsdGeom.Cube.Define(stage, "/World/RightWall")
    right_wall.GetSizeAttr().Set(1.0)
    right_wall.CreateScaleAttr().Set(Gf.Vec3f(0.2, 20.0, 2.0))
    right_wall.AddTranslateOp().Set(Gf.Vec3f(10.0, 0.0, 1.0))

    # Back wall
    back_wall = UsdGeom.Cube.Define(stage, "/World/BackWall")
    back_wall.GetSizeAttr().Set(1.0)
    back_wall.CreateScaleAttr().Set(Gf.Vec3f(20.0, 0.2, 2.0))
    back_wall.AddTranslateOp().Set(Gf.Vec3f(0.0, 10.0, 1.0))

def create_static_obstacles(stage):
    """
    Create static obstacles for the course
    """
    obstacles = [
        # Pillars
        {"name": "Pillar1", "pos": [-5.0, 3.0, 1.0], "size": [0.5, 0.5, 2.0]},
        {"name": "Pillar2", "pos": [5.0, -3.0, 1.0], "size": [0.5, 0.5, 2.0]},
        {"name": "Pillar3", "pos": [0.0, 6.0, 1.0], "size": [0.8, 0.8, 2.0]},

        # Boxes
        {"name": "Box1", "pos": [-3.0, -2.0, 0.5], "size": [1.0, 1.0, 1.0]},
        {"name": "Box2", "pos": [4.0, 4.0, 0.5], "size": [1.2, 0.8, 1.0]},
        {"name": "Box3", "pos": [-1.0, -5.0, 0.5], "size": [0.8, 1.2, 1.0]},

        # Narrow passages
        {"name": "NarrowWall1", "pos": [0.0, -1.0, 1.0], "size": [6.0, 0.1, 2.0]},
        {"name": "Gap1", "pos": [0.0, -1.0, 1.0], "size": [1.0, 0.1, 0.5], "is_gap": True},
    ]

    for i, obs in enumerate(obstacles):
        if obs.get("is_gap", False):
            # Create gap (negative space)
            continue
        else:
            obstacle = UsdGeom.Cube.Define(stage, f"/World/Obstacles/Obstacle{i+1}")
            obstacle.GetSizeAttr().Set(1.0)
            obstacle.CreateScaleAttr().Set(Gf.Vec3f(*obs["size"]))
            obstacle.AddTranslateOp().Set(Gf.Vec3f(*obs["pos"]))

def create_dynamic_obstacles(stage):
    """
    Create dynamic obstacles that move during the course
    """
    # Moving platform
    platform = UsdGeom.Cube.Define(stage, "/World/Dynamic/MovingPlatform")
    platform.GetSizeAttr().Set(1.0)
    platform.CreateScaleAttr().Set(Gf.Vec3f(2.0, 0.5, 0.2))
    platform.AddTranslateOp().Set(Gf.Vec3f(2.0, 2.0, 0.2))

    # Oscillating barrier
    barrier = UsdGeom.Cube.Define(stage, "/World/Dynamic/OscillatingBarrier")
    barrier.GetSizeAttr().Set(1.0)
    barrier.CreateScaleAttr().Set(Gf.Vec3f(0.2, 1.5, 1.0))
    barrier.AddTranslateOp().Set(Gf.Vec3f(-2.0, 0.0, 0.5))

def create_navigation_waypoints(stage):
    """
    Create navigation waypoints for the course
    """
    waypoints = [
        {"name": "Start", "pos": [-8.0, -8.0, 0.0]},
        {"name": "Checkpoint1", "pos": [-5.0, -5.0, 0.0]},
        {"name": "Checkpoint2", "pos": [0.0, 0.0, 0.0]},
        {"name": "Checkpoint3", "pos": [5.0, 5.0, 0.0]},
        {"name": "Finish", "pos": [8.0, 8.0, 0.0]},
    ]

    for wp in waypoints:
        waypoint = UsdGeom.Sphere.Define(stage, f"/World/Waypoints/{wp['name']}")
        waypoint.GetRadiusAttr().Set(0.2)
        waypoint.AddTranslateOp().Set(Gf.Vec3f(*wp["pos"]))

def apply_photorealistic_materials(stage):
    """
    Apply photorealistic materials to the environment
    """
    # Create ground material
    ground_material = create_photorealistic_material(stage, "/World/Materials/GroundMaterial", "ground")

    # Create wall material
    wall_material = create_photorealistic_material(stage, "/World/Materials/WallMaterial", "wall")

    # Create obstacle materials
    obstacle_material = create_photorealistic_material(stage, "/World/Materials/ObstacleMaterial", "obstacle")

def create_photorealistic_material(stage, material_path, material_type):
    """
    Create a photorealistic material based on type
    """
    from pxr import UsdShade

    material_prim = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")

    if material_type == "ground":
        shader.CreateIdAttr("OmniPBR")
        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.7, 0.7, 0.7))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    elif material_type == "wall":
        shader.CreateIdAttr("OmniPBR")
        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.8, 0.8, 0.9))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
    else:  # obstacle
        shader.CreateIdAttr("OmniPBR")
        shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.9, 0.6, 0.2))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)

    material_prim.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "out")
    return material_prim

# Run the environment creation
create_obstacle_course()
print("Obstacle course created successfully!")
```

### Setting up the Humanoid Robot

```python
# setup_humanoid_robot.py
import omni
from pxr import Gf, UsdGeom, Sdf, UsdPhysics
from omni.isaac.urdf import _urdf
import carb

def setup_humanoid_robot():
    """
    Setup the humanoid robot in the obstacle course
    """
    stage = omni.usd.get_context().get_stage()

    # Import humanoid URDF (assuming you have a humanoid.urdf file)
    urdf_interface = _urdf.acquire_urdf_interface()

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.make_default_prim = False
    import_config.self_collision = False
    import_config.default_drive_strength = 20000
    import_config.default_position_drive_damping = 200

    # Import the humanoid robot
    # Replace with your actual humanoid URDF path
    humanoid_urdf_path = "/path/to/humanoid.urdf"  # Update this path

    imported_robot_path = urdf_interface.parse_urdf(
        humanoid_urdf_path,
        import_config
    )

    robot_path = urdf_interface.import_robot(
        imported_robot_path,
        "/World/HumanoidRobot",
        import_config
    )

    # Position the robot at the start
    robot_prim = stage.GetPrimAtPath(robot_path)
    if robot_prim:
        # Set initial position at start waypoint
        xform = UsdGeom.Xformable(robot_prim)
        xform.AddTranslateOp().Set(Gf.Vec3f(-8.0, -8.0, 1.0))  # At start position

    print(f"Humanoid robot imported to: {robot_path}")
    return robot_path

# Setup robot cameras for perception
def setup_robot_sensors(robot_path):
    """
    Setup cameras and sensors on the humanoid robot
    """
    stage = omni.usd.get_context().get_stage()

    # Head camera (RGB)
    head_camera = UsdGeom.Cone.Define(stage, f"{robot_path}/HeadCamera")
    head_camera.GetHeightAttr().Set(0.1)
    head_camera.GetRadiusAttr().Set(0.05)
    head_camera.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.1))  # On head

    # Chest camera (optional secondary view)
    chest_camera = UsdGeom.Cone.Define(stage, f"{robot_path}/ChestCamera")
    chest_camera.GetHeightAttr().Set(0.1)
    chest_camera.GetRadiusAttr().Set(0.05)
    chest_camera.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))  # On chest

    print("Robot sensors configured")
```

## Phase 2: Isaac ROS Perception Pipeline

### Perception Pipeline Configuration

```python
# perception_pipeline_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    camera_namespace = LaunchConfiguration('camera_namespace', default='camera')

    # Isaac ROS Visual SLAM
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='visual_slam_node',
        name='visual_slam',
        parameters=[{
            'use_sim_time': use_sim_time,
            'enable_rectified_pose': True,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'publish_odom_tf': True,
            'publish_map_tf': True,
            # GPU acceleration
            'enable_gpu_acceleration': True,
            'gpu_device_id': 0
        }],
        remappings=[
            ('/visual_slam/image', [camera_namespace, '/image_rect_color']),
            ('/visual_slam/camera_info', [camera_namespace, '/camera_info']),
        ],
        output='screen'
    )

    # Isaac ROS Stereo Dense Reconstruction (if using stereo)
    stereo_node = Node(
        package='isaac_ros_stereo_image_proc',
        executable='isaac_ros_stereo_rectify_node',
        name='stereo_rectify',
        parameters=[{
            'use_sim_time': use_sim_time,
            'width': 1280,
            'height': 720,
            'Q_identity': False
        }],
        remappings=[
            ('left/image_raw', [camera_namespace, '/left/image_raw']),
            ('right/image_raw', [camera_namespace, '/right/image_raw']),
            ('left/camera_info', [camera_namespace, '/left/camera_info']),
            ('right/camera_info', [camera_namespace, '/right/camera_info']),
        ]
    )

    # Isaac ROS Bi3D for 3D semantic segmentation
    bi3d_node = Node(
        package='isaac_ros_bi3d',
        executable='bi3d_node',
        name='bi3d_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'featnet_engine_file_path': '/path/to/featnet.plan',  # Update path
            'segnet_engine_file_path': '/path/to/segnet.plan',    # Update path
            'max_distance': 2.0,
            'min_distance': 0.5
        }],
        remappings=[
            ('image', [camera_namespace, '/image_rect_color']),
            ('camera_info', [camera_namespace, '/camera_info']),
        ]
    )

    # Isaac ROS AprilTag for precise localization
    apriltag_node = Node(
        package='isaac_ros_apriltag',
        executable='apriltag_node',
        name='apriltag',
        parameters=[{
            'use_sim_time': use_sim_time,
            'family': 'tag36h11',
            'size': 0.16,
            'max_tags': 10,
            'debug': False
        }],
        remappings=[
            ('image', [camera_namespace, '/image_rect_color']),
            ('camera_info', [camera_namespace, '/camera_info']),
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('camera_namespace', default_value='camera'),
        visual_slam_node,
        stereo_node,
        bi3d_node,
        apriltag_node,
    ])
```

### Perception Configuration Files

```yaml
# perception_config.yaml
perception_pipeline:
  ros__parameters:
    # Processing rates
    image_processing_rate: 30.0  # Hz
    detection_rate: 10.0         # Hz

    # Confidence thresholds
    detection_confidence_threshold: 0.7
    tracking_confidence_threshold: 0.5

    # GPU settings
    gpu_acceleration: true
    gpu_device_id: 0
    gpu_memory_fraction: 0.8

    # Image preprocessing
    image_width: 1280
    image_height: 720
    enable_rectification: true

    # Sensor fusion parameters
    temporal_filtering: true
    spatial_filtering: true
    fusion_confidence_threshold: 0.8
```

## Phase 3: Nav2 Navigation Configuration

### Navigation Parameters

```yaml
# nav2_params_humanoid.yaml
bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_to_pose_bt_xml: /opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_w_replanning_and_recovery.xml

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 30
      model_dt: 0.05
      batch_size: 2000
      vx_std: 0.2
      vy_std: 0.1
      wz_std: 0.4
      vx_max: 0.4  # Humanoid speed limit
      vx_min: -0.2
      vy_max: 0.3
      wz_max: 0.6  # Reduced for stability
      xy_goal_tolerance: 0.3
      yaw_goal_tolerance: 0.4
      state_reset_threshold: 0.5
      ctrl_freq: 20.0
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

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 10.0
      publish_frequency: 5.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 8.0
      height: 8.0
      resolution: 0.05  # High resolution for humanoid precision
      robot_radius: 0.4  # Humanoid size
      plugins: ["obstacle_layer", "voxel_layer", "inflation_layer"]

      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan camera_obstacles
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
        camera_obstacles:
          topic: /perception/obstacles
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "PointCloud2"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 5.0  # Higher for humanoid safety
        inflation_radius: 0.8     # Larger safety margin

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 2.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.4
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]

      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True

      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: perception_obstacles
        perception_obstacles:
          topic: /perception/obstacles_3d
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "PointCloud2"
          raytrace_max_range: 5.0
          raytrace_min_range: 0.0
          obstacle_max_range: 4.0
          obstacle_min_range: 0.0

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 1.0

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      visualize_potential: false

smoother_server:
  ros__parameters:
    use_sim_time: True
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
    behavior_plugins: ["spin", "backup", "drive_on_heading", "wait", "humanoid_balance_recovery"]

    spin:
      plugin: "nav2_behaviors::Spin"
      spin_dist: 1.57
    backup:
      plugin: "nav2_behaviors::BackUp"
      backup_dist: 0.3
      backup_speed: 0.05
    drive_on_heading:
      plugin: "nav2_behaviors::DriveOnHeading"
      drive_on_heading_timeout: 10.0
    wait:
      plugin: "nav2_behaviors::Wait"
      wait_duration: 1.0
    humanoid_balance_recovery:
      plugin: "humanoid_behaviors::BalanceRecovery"
      max_attempts: 3
      recovery_time: 5.0
```

### Custom Behavior Tree for Obstacle Course

```xml
<!-- obstacle_course_bt.xml -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <ReactiveSequence name="NavigateWithObstacleCourseRecovery">
            <!-- Main navigation sequence -->
            <PipelineSequence name="NavigateThroughCourse">
                <RateController hz="1.0">
                    <RecoveryNode number_of_retries="6">
                        <PipelineSequence name="ComputeAndExecutePath">
                            <RateController hz="1.0">
                                <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
                            </RateController>
                            <FollowPath path="{path}" controller_id="FollowPath"/>
                        </PipelineSequence>
                        <!-- Recovery behaviors specific to obstacle course -->
                        <RecoveryNode number_of_retries="2">
                            <HumanoidBalanceRecovery/>
                            <StepOverObstacle height="0.3"/>
                            <BackUp backup_dist="0.4" backup_speed="0.05"/>
                            <Spin spin_dist="1.57"/>
                        </RecoveryNode>
                    </RecoveryNode>
                </RateController>
            </PipelineSequence>

            <!-- Handle dynamic obstacles -->
            <ReactiveFallback name="HandleDynamicObstacles">
                <IsDynamicObstacleNearby/>
                <WaitForObstacleClearance duration="10.0"/>
            </ReactiveFallback>

            <!-- Checkpoint management -->
            <ReactiveSequence>
                <GoalUpdated/>
                <ClearEntireCostmap name="ClearGlobalCostmap" service_name="global_costmap/clear_entirely_global_costmap"/>
                <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
            </ReactiveSequence>
        </ReactiveSequence>
    </BehaviorTree>

    <!-- Custom actions for obstacle course -->
    <BehaviorTree ID="HumanoidBalanceRecovery">
        <Sequence>
            <StopRobot/>
            <Wait wait_duration="2.0"/>
            <CheckBalanceStability timeout="5.0"/>
            <Succeeded/>
        </Sequence>
    </BehaviorTree>

    <BehaviorTree ID="StepOverObstacle">
        <Sequence>
            <StopRobot/>
            <ComputeStepTrajectory height="{height}"/>
            <ExecuteStepMotion/>
            <Succeeded/>
        </Sequence>
    </BehaviorTree>
</root>
```

## Phase 4: Integration Manager Implementation

### Complete Integration Node

```python
# obstacle_course_integration.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from tf2_ros import TransformListener, Buffer
from builtin_interfaces.msg import Duration
import numpy as np
import threading
import time
from collections import deque

class ObstacleCourseIntegration(Node):
    def __init__(self):
        super().__init__('obstacle_course_integration')

        # Navigation state
        self.navigation_state = 'idle'  # idle, navigating, paused, completed
        self.course_waypoints = [
            [-5.0, -5.0, 0.0],  # Checkpoint 1
            [0.0, 0.0, 0.0],    # Checkpoint 2
            [5.0, 5.0, 0.0],    # Checkpoint 3
            [8.0, 8.0, 0.0]     # Finish
        ]
        self.current_waypoint_index = 0

        # Data buffers
        self.perception_buffer = deque(maxlen=5)
        self.odometry_buffer = deque(maxlen=10)
        self.obstacle_buffer = deque(maxlen=10)

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Publishers and subscribers
        self.status_pub = self.create_publisher(String, 'course_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.perception_sub = self.create_subscription(
            PointCloud2, '/perception/obstacles_3d', self.perception_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Integration timer
        self.integration_timer = self.create_timer(0.1, self.integration_callback)

        # Course control
        self.course_active = False
        self.emergency_stop = False

        self.get_logger().info('Obstacle course integration node initialized')

    def start_course(self):
        """
        Start the obstacle course navigation
        """
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation server not available')
            return False

        self.course_active = True
        self.current_waypoint_index = 0
        self.navigation_state = 'navigating'

        self.get_logger().info('Starting obstacle course navigation')
        self.navigate_to_next_waypoint()

        return True

    def navigate_to_next_waypoint(self):
        """
        Navigate to the next waypoint in the course
        """
        if self.current_waypoint_index >= len(self.course_waypoints):
            self.complete_course()
            return

        target = self.course_waypoints[self.current_waypoint_index]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = target[0]
        goal_msg.pose.pose.position.y = target[1]
        goal_msg.pose.pose.position.z = target[2]
        # Simple orientation (facing forward)
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f'Navigating to waypoint {self.current_waypoint_index + 1}: {target}')

        # Send navigation goal
        self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        ).add_done_callback(self.navigation_done_callback)

    def navigation_feedback_callback(self, feedback):
        """
        Handle navigation feedback
        """
        self.get_logger().debug(f'Navigation feedback: {feedback.feedback.current_pose}')

    def navigation_done_callback(self, future):
        """
        Handle completion of navigation goal
        """
        goal_result = future.result()

        if goal_result.status == 4:  # SUCCEEDED
            self.get_logger().info(f'Reached waypoint {self.current_waypoint_index + 1}')
            self.current_waypoint_index += 1

            if self.current_waypoint_index < len(self.course_waypoints):
                # Navigate to next waypoint
                self.navigate_to_next_waypoint()
            else:
                self.complete_course()
        else:
            self.get_logger().error(f'Navigation failed to waypoint {self.current_waypoint_index + 1}')
            self.handle_navigation_failure()

    def complete_course(self):
        """
        Handle course completion
        """
        self.navigation_state = 'completed'
        self.course_active = False

        status_msg = String()
        status_msg.data = 'COURSE_COMPLETED'
        self.status_pub.publish(status_msg)

        self.get_logger().info('Obstacle course completed successfully!')

        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

    def handle_navigation_failure(self):
        """
        Handle navigation failure
        """
        self.get_logger().warn('Navigation failure, attempting recovery')

        # Try to continue with next waypoint or implement recovery
        if self.current_waypoint_index + 1 < len(self.course_waypoints):
            self.current_waypoint_index += 1
            self.navigate_to_next_waypoint()
        else:
            self.emergency_stop = True
            self.stop_navigation()

    def integration_callback(self):
        """
        Main integration callback
        """
        if not self.course_active or self.emergency_stop:
            return

        # Monitor for obstacles and adjust navigation if needed
        self.check_environment_safety()

        # Publish course status
        status_msg = String()
        status_msg.data = f'NAVIGATING_WP_{self.current_waypoint_index + 1}_OF_{len(self.course_waypoints)}'
        self.status_pub.publish(status_msg)

    def check_environment_safety(self):
        """
        Check environment for safety and adjust navigation if needed
        """
        if self.obstacle_buffer:
            latest_obstacles = self.obstacle_buffer[-1]
            if self.detect_hazardous_obstacles(latest_obstacles):
                self.get_logger().warn('Hazardous obstacle detected, pausing navigation')
                # Implement obstacle avoidance or pause navigation
                pass

    def detect_hazardous_obstacles(self, obstacles):
        """
        Detect if there are hazardous obstacles in the path
        """
        # Analyze obstacle data to detect potential hazards
        # This would use Isaac ROS perception data
        return False  # Placeholder

    def perception_callback(self, msg):
        """
        Handle perception data from Isaac ROS
        """
        self.perception_buffer.append(msg)

        # Process perception data for obstacle detection
        obstacles = self.process_perception_data(msg)
        if obstacles:
            self.obstacle_buffer.append(obstacles)

    def process_perception_data(self, perception_msg):
        """
        Process perception data from Isaac ROS
        """
        # Convert perception message to obstacle information
        # This would interface with Isaac ROS perception outputs
        return []  # Placeholder for obstacle list

    def odom_callback(self, msg):
        """
        Handle odometry data
        """
        self.odometry_buffer.append(msg)

    def scan_callback(self, msg):
        """
        Handle laser scan data
        """
        # Process laser scan for immediate obstacle detection
        pass

    def stop_navigation(self):
        """
        Stop all navigation activities
        """
        self.course_active = False
        self.emergency_stop = True

        # Stop the robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        self.get_logger().info('Navigation stopped')

def main(args=None):
    rclpy.init(args=args)

    integration_node = ObstacleCourseIntegration()

    # Start the course after a short delay to ensure all systems are ready
    def start_course_delayed():
        time.sleep(5.0)  # Wait for systems to initialize
        integration_node.start_course()

    # Start course in a separate thread to allow ROS spinning
    start_thread = threading.Thread(target=start_course_delayed)
    start_thread.start()

    try:
        rclpy.spin(integration_node)
    except KeyboardInterrupt:
        integration_node.get_logger().info('Interrupted, stopping course')
        integration_node.stop_navigation()
    finally:
        integration_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Phase 5: Launch and Execution Scripts

### Complete System Launch

```xml
<!-- obstacle_course_complete_launch.py -->
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration('params_file')

    # Package locations
    pkg_nav2_bringup = FindPackageShare('nav2_bringup')
    pkg_isaac_ros_examples = FindPackageShare('isaac_ros_examples')
    pkg_humanoid_course = FindPackageShare('humanoid_obstacle_course')

    # Isaac Sim bridge launch
    isaac_sim_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_isaac_ros_examples, 'launch', 'isaac_ros_bridge.launch.py'])
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Isaac ROS perception pipeline
    perception_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_isaac_ros_examples, 'launch', 'perception_pipeline.launch.py'])
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Nav2 navigation system
    navigation_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_nav2_bringup, 'launch', 'navigation_launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': PathJoinSubstitution([pkg_humanoid_course, 'config', 'nav2_params_humanoid.yaml']),
            'autostart': 'true'
        }.items()
    )

    # Obstacle course integration manager
    integration_manager = Node(
        package='humanoid_obstacle_course',
        executable='obstacle_course_integration',
        name='obstacle_course_integration',
        parameters=[{
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # Delay integration manager start to allow other systems to initialize
    delayed_integration = TimerAction(
        period=10.0,  # Wait 10 seconds
        actions=[integration_manager]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [pkg_humanoid_course, 'config', 'nav2_params_humanoid.yaml'])),

        # Launch systems in order
        isaac_sim_bridge,
        perception_pipeline,
        navigation_system,
        delayed_integration,
    ])
```

### Execution Script

```bash
#!/bin/bash
# run_obstacle_course.sh

echo "Starting Humanoid Obstacle Course Project..."

# Source ROS 2 and Isaac Sim environments
source /opt/ros/humble/setup.bash
source ~/isaac-sim/setup_conda_env.sh  # Adjust path as needed

# Create workspace if it doesn't exist
mkdir -p ~/humanoid_course_ws/src
cd ~/humanoid_course_ws

# Build the workspace
colcon build --symlink-install
source install/setup.bash

# Launch the complete system
echo "Launching obstacle course system..."
ros2 launch humanoid_obstacle_course obstacle_course_complete_launch.py

echo "Obstacle course completed!"
```

### Validation Script

```python
# validate_system.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import time

class SystemValidator(Node):
    def __init__(self):
        super().__init__('system_validator')

        # Track system components
        self.components_status = {
            'perception': False,
            'navigation': False,
            'integration': False,
            'control': False
        }

        # Subscribe to key topics to verify functionality
        self.perception_sub = self.create_subscription(
            PointCloud2, '/perception/obstacles_3d', self.perception_cb, 1)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 1)
        self.status_sub = self.create_subscription(
            String, '/course_status', self.status_cb, 1)

        # Publisher to verify control
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        # Validation timer
        self.validation_timer = self.create_timer(1.0, self.validate_system)

        self.validation_start_time = self.get_clock().now()
        self.validation_results = []

    def perception_cb(self, msg):
        self.components_status['perception'] = True

    def odom_cb(self, msg):
        self.components_status['navigation'] = True

    def status_cb(self, msg):
        self.components_status['integration'] = True

    def validate_system(self):
        current_time = self.get_clock().now()
        elapsed = (current_time - self.validation_start_time).nanoseconds / 1e9

        if elapsed > 30:  # Validate for 30 seconds
            self.perform_final_validation()
            self.validation_timer.destroy()
            return

        # Check if all components are active
        all_active = all(self.components_status.values())

        if all_active:
            self.get_logger().info('✓ All system components are active and communicating')
        else:
            inactive = [comp for comp, active in self.components_status.items() if not active]
            self.get_logger().warn(f'Inactive components: {inactive}')

    def perform_final_validation(self):
        """
        Perform final validation of the complete system
        """
        self.get_logger().info('Performing final system validation...')

        # Check all components are active
        all_active = all(self.components_status.values())

        if all_active:
            self.get_logger().info('✓ System validation PASSED - All components are functional')
            self.validation_results.append('PASSED: All components functional')
        else:
            self.get_logger().error('✗ System validation FAILED - Some components are not responding')
            self.validation_results.append('FAILED: Components not responding')

        # Test basic control
        self.test_control_system()

        # Print validation summary
        self.print_validation_summary()

    def test_control_system(self):
        """
        Test the control system by sending a simple command
        """
        self.get_logger().info('Testing control system...')

        # Send a small forward command
        cmd = Twist()
        cmd.linear.x = 0.1
        cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)
        time.sleep(1.0)  # Let command execute

        # Stop
        cmd.linear.x = 0.0
        self.cmd_pub.publish(cmd)

        self.get_logger().info('✓ Control system test completed')

    def print_validation_summary(self):
        """
        Print a summary of validation results
        """
        self.get_logger().info('=== SYSTEM VALIDATION SUMMARY ===')
        for result in self.validation_results:
            self.get_logger().info(result)
        self.get_logger().info('=================================')

def main(args=None):
    rclpy.init(args=args)
    validator = SystemValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.get_logger().info('Validation interrupted')
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Phase 6: Testing and Validation

### Test Scenarios

1. **Basic Navigation Test**: Robot successfully navigates from start to finish without dynamic obstacles
2. **Obstacle Avoidance Test**: Robot detects and avoids static and dynamic obstacles
3. **Perception Integration Test**: Isaac ROS perception data properly updates Nav2 costmaps
4. **Recovery Behavior Test**: Robot successfully recovers from navigation failures
5. **Performance Test**: System maintains real-time performance throughout the course

### Running the Complete Project

To run the complete obstacle course project:

1. **Setup the environment**:
   ```bash
   cd ~/humanoid_course_ws
   source install/setup.bash
   ```

2. **Launch Isaac Sim** with the obstacle course environment

3. **Run the complete system**:
   ```bash
   ros2 launch humanoid_obstacle_course obstacle_course_complete_launch.py
   ```

4. **Monitor the system** using RViz, Nav2 tools, and Isaac tools

5. **Validate system performance** using the validation script

## Troubleshooting Common Issues

```bash
# Issue: Perception nodes not receiving data
# Solutions:
# 1. Check Isaac Sim ROS bridge is active
# 2. Verify topic remappings are correct
# 3. Check camera calibration parameters
# 4. Ensure Isaac Sim is publishing sensor data

# Issue: Navigation fails to find paths
# Solutions:
# 1. Verify costmap inflation parameters
# 2. Check robot radius settings
# 3. Validate map and localization
# 4. Inspect obstacle detection quality

# Issue: System performance degradation
# Solutions:
# 1. Monitor GPU utilization
# 2. Adjust processing frequencies
# 3. Optimize data buffering
# 4. Check for memory leaks
```

## Project Extensions and Enhancements

Consider these extensions to deepen your understanding:

1. **Dynamic Obstacle Handling**: Implement prediction for moving obstacles
2. **Multi-Modal Perception**: Combine different sensor modalities
3. **Learning-Based Navigation**: Integrate reinforcement learning
4. **Human-Robot Interaction**: Add social navigation capabilities
5. **Real-World Transfer**: Test sim-to-real transfer capabilities

## Summary

This mini-project demonstrates the complete implementation of an AI-robot brain system for humanoid navigation. You've successfully:

1. Created a photorealistic obstacle course in Isaac Sim
2. Configured Isaac ROS perception nodes for real-time processing
3. Set up Nav2 for humanoid-specific navigation
4. Implemented an integration layer that coordinates all components
5. Validated the complete system in simulation

The project showcases how Isaac Sim, Isaac ROS, and Nav2 work together to create a sophisticated AI system capable of autonomous navigation in complex environments. This integrated approach forms the foundation for advanced humanoid robotics applications.

In the next and final chapter of Module 3, we'll summarize the key concepts and provide assessment questions to reinforce your learning.