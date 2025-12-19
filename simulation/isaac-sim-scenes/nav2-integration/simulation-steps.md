# Isaac ROS and Nav2 Integration Simulation Steps

This guide provides step-by-step instructions for connecting Nav2 with Isaac ROS for humanoid navigation in Isaac Sim as covered in Module 3 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to integrate Isaac ROS perception and mapping capabilities with the Nav2 navigation stack for complete autonomous navigation in Isaac Sim environments.

## Prerequisites

- Isaac Sim with robot model configured
- Isaac ROS VSLAM and perception nodes working
- Nav2 packages installed
- Completed perception and mapping exercises

## Simulation Environment Setup

1. Ensure all required packages are installed:
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/isaac_ros_ws/install/setup.bash
   ```

## Exercise 1: Install Nav2 Packages

1. Install Nav2 packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
   sudo apt install ros-humble-nav2-gui ros-humble-nav2-rviz-plugins
   sudo apt install ros-humble-slam-toolbox
   ```

2. Verify Nav2 installation:
   ```bash
   ros2 launch nav2_bringup navigation_launch.py --show-args
   ```

## Exercise 2: Configure Nav2 for Isaac Sim

1. Create Nav2 configuration file (`config/nav2_params.yaml`):
   ```yaml
   amcl:
     ros__parameters:
       use_sim_time: True
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
       sigma_hit: 0.2
       tf_broadcast: true
       transform_tolerance: 1.0
       update_min_a: 0.2
       update_min_d: 0.25
       z_hit: 0.5
       z_max: 0.05
       z_rand: 0.5
       z_short: 0.05
       scan_topic: /scan

   amcl_map_client:
     ros__parameters:
       use_sim_time: True

   amcl_rclcpp_node:
     ros__parameters:
       use_sim_time: True

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
       navigate_through_poses: False
       navigate_with_feedback: False
       action_server_result_timeout: 900.0
       # Specify the path where the BT XML file is located
       default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
       plugin_lib_names:
       - nav2_compute_path_to_pose_action_bt_node
       - nav2_compute_path_through_poses_action_bt_node
       - nav2_follow_path_action_bt_node
       - nav2_back_up_action_bt_node
       - nav2_spin_action_bt_node
       - nav2_wait_action_bt_node
       - nav2_clear_costmap_service_bt_node
       - nav2_is_stuck_condition_bt_node
       - nav2_goal_reached_condition_bt_node
       - nav2_goal_updated_condition_bt_node
       - nav2_initial_pose_received_condition_bt_node
       - nav2_reinitialize_global_localization_service_bt_node
       - nav2_rate_controller_bt_node
       - nav2_distance_controller_bt_node
       - nav2_speed_controller_bt_node
       - nav2_truncate_path_action_bt_node
       - nav2_truncate_path_local_action_bt_node
       - nav2_goal_updater_node_bt_node
       - nav2_recovery_node_bt_node
       - nav2_pipeline_sequence_bt_node
       - nav2_round_robin_node_bt_node
       - nav2_transform_available_condition_bt_node
       - nav2_time_expired_condition_bt_node
       - nav2_path_expiring_timer_condition
       - nav2_distance_traveled_condition_bt_node
       - nav2_single_trigger_bt_node
       - nav2_is_battery_low_condition_bt_node
       - nav2_navigate_through_poses_action_bt_node
       - nav2_navigate_to_pose_action_bt_node
       - nav2_remove_passed_goals_action_bt_node
       - nav2_planner_selector_bt_node
       - nav2_controller_selector_bt_node
       - nav2_goal_checker_selector_bt_node

   bt_navigator_rclcpp_node:
     ros__parameters:
       use_sim_time: True

   controller_server:
     ros__parameters:
       use_sim_time: True
       controller_frequency: 20.0
       min_x_velocity_threshold: 0.001
       min_y_velocity_threshold: 0.5
       min_theta_velocity_threshold: 0.001
       progress_checker_plugin: "progress_checker"
       goal_checker_plugin: "goal_checker"
       controller_plugins: ["FollowPath"]

       # Progress checker parameters
       progress_checker:
         plugin: "nav2_controller::SimpleProgressChecker"
         required_movement_radius: 0.5
         movement_time_allowance: 10.0

       # Goal checker parameters
       goal_checker:
         plugin: "nav2_controller::SimpleGoalChecker"
         xy_goal_tolerance: 0.25
         yaw_goal_tolerance: 0.25
         stateful: True

       # Controller parameters
       FollowPath:
         plugin: "nav2_rotation_shim::RotationShimController"
         progress_checker_plugin: "progress_checker"
         goal_checker_plugin: "goal_checker"
         # The underlying controller
         controller_plugin: "FollowPath"
         rotation_shim:
           plugin: "nav2_controller::SimpleProgressChecker"
           required_movement_radius: 0.5
           movement_time_allowance: 2.0
         FollowPath:
           plugin: "nav2_controllers::DifferentialSpeedController"
           speed_ramp_duration: 2.0
           max_linear_speed: 1.0
           min_linear_speed: 0.1
           max_angular_speed: 1.0
           min_angular_speed: 0.1

   controller_server_rclcpp_node:
     ros__parameters:
       use_sim_time: True

   local_costmap:
     local_costmap:
       ros__parameters:
         update_frequency: 5.0
         publish_frequency: 2.0
         global_frame: odom
         robot_base_frame: base_link
         use_sim_time: True
         rolling_window: true
         width: 3
         height: 3
         resolution: 0.05
         robot_radius: 0.22
         plugins: ["voxel_layer", "inflation_layer"]
         inflation_layer:
           plugin: "nav2_costmap_2d::InflationLayer"
           cost_scaling_factor: 3.0
           inflation_radius: 0.55
         voxel_layer:
           plugin: "nav2_costmap_2d::VoxelLayer"
           enabled: True
           publish_voxel_map: True
           origin_z: 0.0
           z_resolution: 0.05
           z_voxels: 16
           max_obstacle_height: 2.0
           mark_threshold: 0
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
         always_send_full_costmap: True
     local_costmap_client:
       ros__parameters:
         use_sim_time: True
     local_costmap_rclcpp_node:
       ros__parameters:
         use_sim_time: True

   global_costmap:
     global_costmap:
       ros__parameters:
         update_frequency: 1.0
         publish_frequency: 1.0
         global_frame: map
         robot_base_frame: base_link
         use_sim_time: True
         robot_radius: 0.22
         resolution: 0.05
         track_unknown_space: true
         plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
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
         static_layer:
           plugin: "nav2_costmap_2d::StaticLayer"
           map_subscribe_transient_local: True
         inflation_layer:
           plugin: "nav2_costmap_2d::InflationLayer"
           cost_scaling_factor: 3.0
           inflation_radius: 0.55
         always_send_full_costmap: True
     global_costmap_client:
       ros__parameters:
         use_sim_time: True
     global_costmap_rclcpp_node:
       ros__parameters:
         use_sim_time: True

   map_server:
     ros__parameters:
       use_sim_time: True
       yaml_filename: "turtlebot3_world.yaml"

   map_saver:
     ros__parameters:
       use_sim_time: True
       save_map_timeout: 5.0
       free_thresh_default: 0.25
       occupied_thresh_default: 0.65
       map_subscribe_transient_local: True

   planner_server:
     ros__parameters:
       expected_planner_frequency: 20.0
       use_sim_time: True
       planner_plugins: ["GridBased"]
       GridBased:
         plugin: "nav2_navfn_planner/NavfnPlanner"
         tolerance: 0.5
         use_astar: false
         allow_unknown: true

   planner_server_rclcpp_node:
     ros__parameters:
       use_sim_time: True

   recoveries_server:
     ros__parameters:
       costmap_topic: local_costmap/costmap_raw
       footprint_topic: local_costmap/published_footprint
       cycle_frequency: 10.0
       recovery_plugins: ["spin", "backup", "wait"]
       spin:
         plugin: "nav2_recoveries/Spin"
         sim_frequency: 10
         angle_thresh: 0.5
         time_allowance: 15.0
       backup:
         plugin: "nav2_recoveries/BackUp"
         sim_frequency: 10
         backup_dist: -0.15
         backup_speed: 0.025
         time_allowance: 15.0
       wait:
         plugin: "nav2_recoveries/Wait"
         sim_frequency: 10
         wait_duration: 1.0

   recoveries_server_rclcpp_node:
     ros__parameters:
       use_sim_time: True

   wayfinding_server:
     ros__parameters:
       use_sim_time: True

   waypoint_follower:
     ros__parameters:
       use_sim_time: True
       loop_rate: 20
       stop_on_failure: false
       waypoint_task_executor_plugin: "wait_at_waypoint"
       wait_at_waypoint:
         plugin: "nav2_waypoint_follower::WaitAtWaypoint"
         enabled: true
         wait_time: 100
   ```

## Exercise 3: Create Isaac ROS and Nav2 Integration Launch File

1. Create integration launch file (`launch/isaac_nav2_integration.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.conditions import IfCondition
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node, ComposableNodeContainer
   from launch_ros.descriptions import ComposableNode
   from launch_ros.substitutions import FindPackageShare

   def generate_launch_description():
       # Launch configuration
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')
       params_file = LaunchConfiguration('params_file', default='nav2_params.yaml')

       # Isaac ROS Visual SLAM container
       visual_slam_node = ComposableNodeContainer(
           name='visual_slam_container',
           namespace='',
           package='rclcpp_components',
           executable='component_container_mt',
           composable_node_descriptions=[
               ComposableNode(
                   package='isaac_ros_visual_slam',
                   plugin='isaac_ros::slam::VisualSlamNode',
                   name='visual_slam',
                   parameters=[{
                       'use_sim_time': use_sim_time,
                       'enable_occupancy_map': True,
                       'occupancy_map_resolution': 0.05,
                       'occupancy_map_size': 20.0,
                       'enable_slam_visualization': False,
                       'enable_landmarks_view': False,
                       'enable_observations_view': False,
                       'map_frame': 'map',
                       'odom_frame': 'odom',
                       'base_frame': 'base_link',
                       'input_voxel_map_frame': 'map',
                   }],
                   remappings=[
                       ('stereo_camera/left/image', '/camera/color/image_raw'),
                       ('stereo_camera/left/camera_info', '/camera/color/camera_info'),
                       ('stereo_camera/right/image', '/camera/depth/image_raw'),
                       ('stereo_camera/right/camera_info', '/camera/depth/camera_info'),
                   ],
               )
           ],
           output='screen'
       )

       # Nav2 launch
       nav2_launch = IncludeLaunchDescription(
           PythonLaunchDescriptionSource([
               PathJoinSubstitution([
                   FindPackageShare('nav2_bringup'),
                   'launch',
                   'navigation_launch.py'
               ])
           ]),
           launch_arguments={
               'use_sim_time': use_sim_time,
               'params_file': PathJoinSubstitution([
                   FindPackageShare('physical_ai_robotics'),
                   'config',
                   params_file
               ])
           }.items()
       )

       # TF publisher for camera to base_link
       tf_publisher = Node(
           package='tf2_ros',
           executable='static_transform_publisher',
           name='camera_link_broadcaster',
           arguments=['0.1', '0.0', '0.2', '0.0', '0.0', '0.0', 'base_link', 'camera_link'],
           parameters=[{'use_sim_time': use_sim_time}],
           output='screen'
       )

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),
           DeclareLaunchArgument(
               'params_file',
               default_value='nav2_params.yaml',
               description='Full path to the Nav2 parameters file'
           ),
           visual_slam_node,
           nav2_launch,
           tf_publisher,
       ])
   ```

## Exercise 4: Configure Isaac Sim for Navigation

1. Set up Isaac Sim scene for navigation:
   - Create navigable environment with obstacles
   - Configure ground plane for wheeled robot navigation
   - Add visual landmarks for perception

2. Configure robot for navigation in Isaac Sim:
   - Set up differential drive controller
   - Configure collision properties appropriately
   - Add navigation-relevant sensors (LiDAR, cameras)

3. Verify sensor configuration:
   - Ensure LiDAR publishes to `/scan` topic
   - Verify camera topics are available
   - Check odometry publishing

## Exercise 5: Launch Integrated System

1. Launch the complete Isaac ROS + Nav2 system:
   ```bash
   ros2 launch physical_ai_robotics isaac_nav2_integration.launch.py
   ```

2. Verify all components are running:
   ```bash
   # Check all running nodes
   ros2 node list

   # Check all topics
   ros2 topic list
   ```

3. Monitor system status:
   ```bash
   # Monitor TF frames
   ros2 run tf2_tools view_frames

   # Monitor Nav2 status
   ros2 service call /navigate_to_pose nav2_msgs/action/NavigateToPose
   ```

## Exercise 6: Test Mapping with Isaac ROS

1. Move robot through environment:
   - Send velocity commands to explore environment
   - Monitor Isaac ROS mapping progress
   - Verify map building in RViz

2. Validate map quality:
   ```bash
   # Save the map created by Isaac ROS
   ros2 run nav2_map_server map_saver_cli -f ~/isaac_ros_map
   ```

3. Compare with SLAM Toolbox (alternative):
   ```bash
   # Launch SLAM Toolbox for comparison
   ros2 launch slam_toolbox online_async_launch.py
   ```

## Exercise 7: Configure Navigation Goals

1. Set up navigation interface:
   - Use RViz2 to send navigation goals
   - Configure 2D Pose Estimate for localization
   - Test navigation to various locations

2. Test navigation in Isaac Sim:
   - Send goals to different areas of the environment
   - Monitor path planning and execution
   - Validate obstacle avoidance

3. Monitor navigation performance:
   ```bash
   # Monitor navigation status
   ros2 topic echo /navigation/status

   # Monitor path planning
   ros2 topic echo /plan
   ```

## Exercise 8: Integrate Perception with Navigation

1. Use Isaac ROS perception for navigation:
   - Connect AprilTag detections to navigation goals
   - Use object detection for dynamic obstacle avoidance
   - Integrate semantic segmentation for terrain classification

2. Create perception-guided navigation:
   ```python
   # Example node that combines perception and navigation
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped
   from nav2_msgs.action import NavigateToPose
   from rclpy.action import ActionClient
   from vision_msgs.msg import Detection2DArray

   class PerceptionNavigation(Node):
       def __init__(self):
           super().__init__('perception_navigation')

           # Navigation action client
           self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

           # Perception subscriber
           self.detection_subscriber = self.create_subscription(
               Detection2DArray, '/apriltag_detections', self.detection_callback, 10
           )

           self.get_logger().info('Perception Navigation node initialized')

       def detection_callback(self, msg):
           """Process detections and navigate to detected objects"""
           if len(msg.detections) > 0:
               # Navigate to the first detected object
               self.navigate_to_object(msg.detections[0])

       def navigate_to_object(self, detection):
           """Send navigation goal to detected object"""
           # Wait for action server
           self.nav_client.wait_for_server()

           # Create navigation goal
           goal_msg = NavigateToPose.Goal()
           goal_msg.pose.header.frame_id = 'map'
           # Set pose based on detection (simplified)
           goal_msg.pose.pose.position.x = 1.0
           goal_msg.pose.pose.position.y = 1.0
           goal_msg.pose.pose.orientation.w = 1.0

           # Send goal
           self.nav_client.send_goal_async(goal_msg)

   def main(args=None):
       rclpy.init(args=args)
       node = PerceptionNavigation()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 9: Test Autonomous Navigation

1. Create autonomous navigation scenario:
   - Set up waypoints for robot to visit
   - Configure exploration behavior
   - Test navigation in complex environments

2. Validate navigation reliability:
   - Test with various obstacle configurations
   - Validate recovery behaviors
   - Monitor navigation success rates

3. Performance testing:
   - Measure navigation time to goals
   - Validate path optimality
   - Check computational performance

## Exercise 10: Optimize Navigation Parameters

1. Tune Nav2 parameters for Isaac Sim:
   - Adjust costmap resolution and inflation
   - Optimize path planner settings
   - Configure recovery behaviors

2. Optimize Isaac ROS parameters:
   - Adjust mapping parameters for navigation
   - Configure tracking for mobile robot
   - Optimize computational performance

3. Validate parameter effectiveness:
   - Test navigation success rates
   - Measure computational efficiency
   - Validate map quality and consistency

## Exercise 11: Handle Navigation Failures

1. Implement navigation monitoring:
   - Monitor navigation status and progress
   - Implement failure detection and recovery
   - Log navigation performance metrics

2. Test failure scenarios:
   - Blocked paths
   - Localization failures
   - Perception limitations

3. Implement fallback strategies:
   - Alternative navigation approaches
   - Manual intervention capabilities
   - Safe stop procedures

## Exercise 12: Integration Validation

1. Comprehensive system testing:
   - Test end-to-end navigation scenarios
   - Validate mapping and localization
   - Verify perception integration

2. Performance benchmarking:
   - Measure system response times
   - Validate computational requirements
   - Test reliability over extended operation

## Verification Steps

1. Confirm Isaac ROS and Nav2 launch together
2. Verify TF tree connects all frames properly
3. Check that navigation goals are accepted and executed
4. Validate map building from Isaac ROS
5. Ensure obstacle avoidance works properly

## Expected Outcomes

- Understanding of Isaac ROS and Nav2 integration
- Knowledge of mapping and navigation pipeline
- Experience with perception-guided navigation
- Ability to create autonomous navigation systems

## Troubleshooting

- If navigation fails, check TF frames and sensor data
- If mapping is poor, verify Isaac ROS VSLAM configuration
- If obstacle avoidance doesn't work, check costmap configuration

## Next Steps

After completing these exercises, proceed to the Module 3 mini-project to implement a complete humanoid navigation system in Isaac Sim.