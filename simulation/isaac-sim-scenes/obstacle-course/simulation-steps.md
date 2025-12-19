# Mini-Project: Humanoid Navigation Through Obstacle Course Simulation Steps

This guide provides step-by-step instructions for creating a complete humanoid navigation system that walks through an obstacle course using Isaac Sim, Isaac ROS, and Nav2 as covered in Module 3 of the Physical AI & Humanoid Robotics textbook.

## Overview

This mini-project integrates all concepts learned in Module 3 to create a complete autonomous navigation system where a humanoid robot successfully navigates through a complex obstacle course using Isaac Sim for simulation, Isaac ROS for perception and mapping, and Nav2 for path planning and execution.

## Prerequisites

- Isaac Sim with robot model configured
- Isaac ROS VSLAM and perception nodes working
- Nav2 integrated with Isaac ROS
- Completed all previous Module 3 simulation exercises
- Understanding of humanoid robot kinematics and navigation

## Simulation Environment Setup

1. Ensure all required packages are installed and sourced:
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/isaac_ros_ws/install/setup.bash
   ```

## Exercise 1: Design Obstacle Course Environment

1. Create challenging obstacle course in Isaac Sim:
   - Design narrow passages requiring precise navigation
   - Include dynamic obstacles for advanced challenge
   - Add areas with poor texture for perception challenges
   - Create loops and dead ends to test mapping capabilities

2. Set up environment elements:
   - Place walls and barriers
   - Add furniture and static obstacles
   - Configure lighting conditions that vary throughout the course
   - Include visual markers (AprilTags) for localization assistance

3. Validate environment setup:
   - Ensure all obstacles have proper collision properties
   - Verify lighting doesn't cause perception issues
   - Check that navigation space is appropriately defined

## Exercise 2: Configure Humanoid Robot for Navigation

1. Set up humanoid robot configuration:
   - Configure appropriate base controller (differential or omnidirectional)
   - Set up sensor suite (LiDAR, RGB-D camera, IMU)
   - Configure safety limits and joint constraints
   - Verify kinematic properties for stable navigation

2. Configure robot dimensions and properties:
   ```yaml
   # Robot configuration for navigation
   robot_radius: 0.3  # Adjust based on robot size
   max_linear_speed: 0.5  # Conservative speed for humanoid
   max_angular_speed: 0.6
   acceleration_limit: 0.5
   deceleration_limit: 0.8
   ```

3. Set up robot-specific navigation parameters:
   - Configure footprint for humanoid base
   - Set appropriate inflation radius
   - Adjust controller parameters for humanoid dynamics

## Exercise 3: Create Obstacle Course Navigation Launch File

1. Create comprehensive launch file (`launch/obstacle_course_navigation.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
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
                       'occupancy_map_size': 30.0,  # Larger for obstacle course
                       'enable_slam_visualization': True,
                       'enable_landmarks_view': True,
                       'enable_observations_view': True,
                       'map_frame': 'map',
                       'odom_frame': 'odom',
                       'base_frame': 'base_link',
                       'input_voxel_map_frame': 'map',
                       'min_num_landmarks': 100,  # More landmarks for complex env
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

       # Isaac ROS perception nodes container
       perception_container = ComposableNodeContainer(
           name='perception_container',
           namespace='',
           package='rclcpp_components',
           executable='component_container_mt',
           composable_node_descriptions=[
               ComposableNode(
                   package='isaac_ros_apriltag',
                   plugin='nvidia::isaac_ros::apriltag::AprilTagNode',
                   name='apriltag',
                   parameters=[{
                       'use_sim_time': use_sim_time,
                       'family': 'tag36h11',
                       'size': 0.16,  # Larger tags for longer distance detection
                       'max_tags': 20,
                   }],
                   remappings=[
                       ('image', '/camera/color/image_raw'),
                       ('camera_info', '/camera/color/camera_info'),
                   ],
               )
           ],
           output='screen'
       )

       # Nav2 bringup
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

       # TF broadcasters
       tf_publishers = [
           Node(
               package='tf2_ros',
               executable='static_transform_publisher',
               name='camera_link_broadcaster',
               arguments=['0.2', '0.0', '0.3', '0.0', '0.0', '0.0', 'base_link', 'camera_link'],
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),
           Node(
               package='tf2_ros',
               executable='static_transform_publisher',
               name='lidar_link_broadcaster',
               arguments=['0.15', '0.0', '0.2', '0.0', '0.0', '0.0', 'base_link', 'lidar_link'],
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           )
       ]

       # Course navigation controller
       course_controller = Node(
           package='physical_ai_robotics',
           executable='obstacle_course_controller',
           name='obstacle_course_controller',
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
           perception_container,
           nav2_launch,
       ] + tf_publishers + [course_controller])
   ```

## Exercise 4: Implement Course Navigation Controller

1. Create obstacle course navigation controller (`src/physical_ai_robotics/obstacle_course_controller.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped, Twist
   from nav2_msgs.action import NavigateToPose, NavigateThroughPoses
   from rclpy.action import ActionClient
   from sensor_msgs.msg import LaserScan
   from std_msgs.msg import Bool
   import math
   import time

   class ObstacleCourseController(Node):
       def __init__(self):
           super().__init__('obstacle_course_controller')

           # Navigation action client
           self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

           # Publishers and subscribers
           self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
           self.laser_subscriber = self.create_subscription(
               LaserScan, '/scan', self.laser_callback, 10
           )

           # Navigation state
           self.course_waypoints = [
               {'x': 2.0, 'y': 2.0, 'theta': 0.0, 'name': 'Checkpoint 1'},
               {'x': 5.0, 'y': 1.0, 'theta': 1.57, 'name': 'Checkpoint 2'},
               {'x': 7.0, 'y': 4.0, 'theta': 3.14, 'name': 'Checkpoint 3'},
               {'x': 3.0, 'y': 6.0, 'theta': -1.57, 'name': 'Checkpoint 4'},
               {'x': 1.0, 'y': 1.0, 'theta': 0.0, 'name': 'Finish Line'}
           ]

           self.current_waypoint_index = 0
           self.navigation_active = False
           self.obstacle_detected = False

           # Timer for course management
           self.course_timer = self.create_timer(5.0, self.course_management)

           self.get_logger().info('Obstacle Course Controller initialized')
           self.get_logger().info(f'Course has {len(self.course_waypoints)} waypoints')

       def laser_callback(self, msg):
           """Process laser scan for obstacle detection"""
           # Check for obstacles in front of robot
           front_ranges = msg.ranges[len(msg.ranges)//2-30:len(msg.ranges)//2+30]
           front_ranges = [r for r in front_ranges if not math.isnan(r) and r != float('inf')]

           if front_ranges:
               min_distance = min(front_ranges) if front_ranges else float('inf')
               self.obstacle_detected = min_distance < 0.8  # Obstacle within 0.8m
           else:
               self.obstacle_detected = False

       def course_management(self):
           """Manage the obstacle course navigation"""
           if not self.navigation_active:
               if self.current_waypoint_index < len(self.course_waypoints):
                   waypoint = self.course_waypoints[self.current_waypoint_index]
                   self.get_logger().info(f'Navigating to {waypoint["name"]}: ({waypoint["x"]}, {waypoint["y"]})')
                   self.navigate_to_waypoint(waypoint)
                   self.current_waypoint_index += 1
               else:
                   self.get_logger().info('Course completed! All waypoints reached.')
                   self.navigation_active = False
                   self.current_waypoint_index = 0  # Reset for next run

       def navigate_to_waypoint(self, waypoint):
           """Send navigation goal to specified waypoint"""
           # Wait for action server
           if not self.nav_client.wait_for_server(timeout_sec=5.0):
               self.get_logger().error('Navigation action server not available')
               return

           # Create navigation goal
           goal_msg = NavigateToPose.Goal()
           goal_msg.pose.header.frame_id = 'map'
           goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
           goal_msg.pose.pose.position.x = waypoint['x']
           goal_msg.pose.pose.position.y = waypoint['y']
           goal_msg.pose.pose.position.z = 0.0

           # Convert theta to quaternion
           theta = waypoint['theta']
           goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
           goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

           # Send goal
           self.navigation_active = True
           send_goal_future = self.nav_client.send_goal_async(goal_msg)
           send_goal_future.add_done_callback(self.goal_response_callback)

       def goal_response_callback(self, future):
           """Handle navigation goal response"""
           goal_handle = future.result()
           if not goal_handle.accepted:
               self.get_logger().info('Goal rejected')
               self.navigation_active = False
               return

           self.get_logger().info('Goal accepted, getting result...')
           get_result_future = goal_handle.get_result_async()
           get_result_future.add_done_callback(self.get_result_callback)

       def get_result_callback(self, future):
           """Handle navigation result"""
           result = future.result().result
           self.navigation_active = False

           if result:
               self.get_logger().info('Waypoint reached successfully!')
           else:
               self.get_logger().error('Navigation failed to reach waypoint')

       def emergency_stop(self):
           """Emergency stop for safety"""
           cmd = Twist()
           cmd.linear.x = 0.0
           cmd.angular.z = 0.0
           self.cmd_vel_publisher.publish(cmd)

   def main(args=None):
       rclpy.init(args=args)
       controller = ObstacleCourseController()

       try:
           rclpy.spin(controller)
       except KeyboardInterrupt:
           controller.get_logger().info('Navigation interrupted by user')
       finally:
           controller.emergency_stop()
           controller.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 5: Create Advanced Navigation Behaviors

1. Implement adaptive navigation for obstacle course:
   ```python
   # Add to the controller class
   def adaptive_navigation(self):
       """Implement adaptive navigation behaviors for obstacle course"""
       if self.obstacle_detected:
           # Implement reactive obstacle avoidance
           self.execute_obstacle_avoidance()
       else:
           # Continue with planned navigation
           pass

   def execute_obstacle_avoidance(self):
       """Execute obstacle avoidance behavior"""
       # This could include:
       # - Local path replanning
       # - Wall following
       # - Temporary goal adjustment
       pass
   ```

2. Add recovery behaviors for challenging sections:
   ```python
   def add_recovery_behaviors(self):
       """Add specialized recovery behaviors for obstacle course"""
       # Implement behaviors for:
       # - Narrow passages
       # - Tight corners
       # - Areas with poor localization
       # - Dynamic obstacles
       pass
   ```

## Exercise 6: Set Up Course Monitoring and Evaluation

1. Create course evaluation node (`src/physical_ai_robotics/course_evaluator.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped, Point
   from nav_msgs.msg import Path
   from std_msgs.msg import Float32
   import time

   class CourseEvaluator(Node):
       def __init__(self):
           super().__init__('course_evaluator')

           # Subscribers
           self.pose_subscriber = self.create_subscription(
               PoseStamped, '/robot_pose', self.pose_callback, 10
           )
           self.path_subscriber = self.create_subscription(
               Path, '/plan', self.path_callback, 10
           )

           # Publishers
           self.time_publisher = self.create_publisher(Float32, '/course_time', 10)
           self.efficiency_publisher = self.create_publisher(Float32, '/path_efficiency', 10)

           # Course evaluation parameters
           self.start_time = time.time()
           self.total_distance_traveled = 0.0
           self.previous_pose = None
           self.course_completed = False

           # Course checkpoints
           self.checkpoints = [
               Point(x=2.0, y=2.0, z=0.0),
               Point(x=5.0, y=1.0, z=0.0),
               Point(x=7.0, y=4.0, z=0.0),
               Point(x=3.0, y=6.0, z=0.0),
               Point(x=1.0, y=1.0, z=0.0)
           ]

           self.checkpoint_reached = [False] * len(self.checkpoints)
           self.current_checkpoint = 0

           self.get_logger().info('Course Evaluator initialized')

       def pose_callback(self, msg):
           """Track robot position and evaluate course progress"""
           current_time = time.time()
           course_time = current_time - self.start_time

           # Calculate distance traveled
           if self.previous_pose:
               dx = msg.pose.position.x - self.previous_pose.pose.position.x
               dy = msg.pose.position.y - self.previous_pose.pose.position.y
               dz = msg.pose.position.z - self.previous_pose.pose.position.z
               distance_increment = (dx**2 + dy**2 + dz**2)**0.5
               self.total_distance_traveled += distance_increment

           self.previous_pose = msg

           # Check if reached checkpoints
           for i, checkpoint in enumerate(self.checkpoints):
               distance_to_checkpoint = ((msg.pose.position.x - checkpoint.x)**2 +
                                        (msg.pose.position.y - checkpoint.y)**2)**0.5

               if distance_to_checkpoint < 0.5 and not self.checkpoint_reached[i]:  # 0.5m tolerance
                   self.checkpoint_reached[i] = True
                   self.get_logger().info(f'Checkpoint {i+1} reached!')

                   if i == len(self.checkpoints) - 1:  # Last checkpoint = finish
                       self.course_completed = True
                       self.publish_evaluation(course_time)

           # Publish course time
           time_msg = Float32()
           time_msg.data = float(course_time)
           self.time_publisher.publish(time_msg)

       def path_callback(self, msg):
           """Evaluate path efficiency"""
           if len(msg.poses) > 1:
               # Calculate path efficiency as ratio of straight-line distance to actual path length
               start = msg.poses[0].pose.position
               end = msg.poses[-1].pose.position
               straight_line_distance = ((end.x - start.x)**2 + (end.y - start.y)**2)**0.5

               path_length = 0.0
               for i in range(1, len(msg.poses)):
                   p1 = msg.poses[i-1].pose.position
                   p2 = msg.poses[i].pose.position
                   segment_length = ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5
                   path_length += segment_length

               efficiency = straight_line_distance / path_length if path_length > 0 else 0.0
               efficiency_msg = Float32()
               efficiency_msg.data = efficiency
               self.efficiency_publisher.publish(efficiency_msg)

       def publish_evaluation(self, course_time):
           """Publish final course evaluation"""
           self.get_logger().info(f'Course completed in {course_time:.2f} seconds')
           self.get_logger().info(f'Total distance traveled: {self.total_distance_traveled:.2f} meters')

   def main(args=None):
       rclpy.init(args=args)
       evaluator = CourseEvaluator()
       rclpy.spin(evaluator)
       evaluator.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 7: Implement Safety and Monitoring Systems

1. Create safety monitor for obstacle course:
   ```python
   # Add to launch file or create separate safety node
   class SafetyMonitor(Node):
       def __init__(self):
           super().__init__('safety_monitor')

           # Subscribers for critical data
           self.odom_subscriber = self.create_subscription(
               Odometry, '/odom', self.odom_callback, 10
           )
           self.scan_subscriber = self.create_subscription(
               LaserScan, '/scan', self.scan_callback, 10
           )

           # Publisher for safety state
           self.safety_publisher = self.create_publisher(Bool, '/safety_state', 10)

           # Safety parameters
           self.safety_distance = 0.3  # Stop if obstacle closer than 0.3m
           self.max_velocity = 0.5
           self.emergency_stop_active = False

           # Timer for safety checks
           self.safety_timer = self.create_timer(0.1, self.safety_check)

       def scan_callback(self, msg):
           """Monitor for safety-critical obstacles"""
           # Check for very close obstacles
           min_range = min([r for r in msg.ranges if not math.isnan(r) and r != float('inf')], default=float('inf'))
           if min_range < self.safety_distance:
               self.emergency_stop_active = True
               self.get_logger().warn(f'OBSTACLE TOO CLOSE: {min_range:.2f}m, ACTIVATING EMERGENCY STOP')

       def safety_check(self):
           """Publish safety state"""
           safety_msg = Bool()
           safety_msg.data = not self.emergency_stop_active
           self.safety_publisher.publish(safety_msg)
   ```

## Exercise 8: Test Navigation in Various Course Configurations

1. Create multiple course configurations:
   - Easy course (wide paths, few obstacles)
   - Medium course (moderate complexity)
   - Hard course (narrow paths, many obstacles)
   - Dynamic course (moving obstacles)

2. Test navigation robustness:
   - Vary starting positions
   - Test with different lighting conditions
   - Validate performance with sensor noise

3. Evaluate navigation performance:
   - Success rate across different configurations
   - Time to complete course
   - Path efficiency
   - Safety incidents

## Exercise 9: Optimize Navigation for Humanoid Characteristics

1. Tune navigation for humanoid robot:
   - Adjust footprint to match humanoid base
   - Configure appropriate speeds for humanoid dynamics
   - Optimize for humanoid turning radius
   - Consider humanoid stability during navigation

2. Implement humanoid-specific behaviors:
   - More conservative obstacle avoidance
   - Smoother path following for stability
   - Adjusted recovery behaviors for humanoid base

## Exercise 10: Performance Validation

1. Comprehensive testing of obstacle course navigation:
   - Run multiple trials to validate reliability
   - Test edge cases and failure scenarios
   - Validate mapping consistency throughout course
   - Verify localization accuracy

2. Performance metrics collection:
   - Navigation success rate
   - Average completion time
   - Path efficiency metrics
   - Computational resource usage

## Exercise 11: Advanced Course Features

1. Implement dynamic obstacle handling:
   - Detect and track moving obstacles
   - Plan around dynamic obstacles
   - Implement prediction for moving obstacles

2. Add multi-objective navigation:
   - Navigate to multiple goals simultaneously
   - Optimize for multiple criteria (time, energy, safety)
   - Handle priority-based goal selection

## Exercise 12: Final Integration and Demonstration

1. Execute complete obstacle course demonstration:
   - Launch full integrated system
   - Navigate through entire obstacle course
   - Monitor all system components
   - Validate successful completion

2. Comprehensive system validation:
   - Verify all components work together
   - Validate safety systems
   - Confirm performance metrics
   - Document any issues and solutions

## Verification Steps

1. Confirm robot successfully navigates through obstacle course
2. Verify all navigation components work together
3. Check that safety systems function properly
4. Validate mapping and localization throughout course
5. Ensure obstacle avoidance works correctly

## Expected Outcomes

- Complete integration of Isaac Sim, Isaac ROS, and Nav2
- Successful autonomous navigation through complex obstacle course
- Understanding of humanoid navigation challenges
- Experience with complete robotics system integration

## Troubleshooting

- If navigation fails in certain areas, check sensor coverage
- If mapping is inconsistent, verify VSLAM configuration
- If obstacle avoidance doesn't work, check costmap settings

## Next Steps

After completing this obstacle course mini-project, proceed to Module 4 to learn about Vision-Language-Action (VLA) systems that integrate perception, language understanding, and action generation.