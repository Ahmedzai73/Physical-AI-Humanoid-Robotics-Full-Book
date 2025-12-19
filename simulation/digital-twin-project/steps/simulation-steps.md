# Digital Twin Project: Full Humanoid Integration Simulation Steps

This guide provides step-by-step instructions for creating a complete ROS-Unity bridge system for digital twin applications as covered in Module 2 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to connect Unity with ROS 2 using the ROS-Unity bridge, creating an end-to-end pipeline from URDF through Gazebo sensors to Unity visualization. This creates a complete digital twin system.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later recommended)
- Unity 2022.3 LTS or later installed
- Completed Module 1 and Module 2 simulation exercises
- Basic understanding of ROS-Unity bridge concepts
- Completed Gazebo and Unity environment exercises

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

## Exercise 1: Install ROS-Unity Bridge

1. Install the ROS-Unity bridge package:
   ```bash
   # Option 1: Use the official Unity Robotics package
   # Download from Unity Asset Store: ROS-TCP-Connector and ROS-TCP-Endpoint

   # Option 2: Clone from source (if available)
   git clone https://github.com/Unity-Technologies/ROS-TCP-Connector.git
   ```

2. Verify ROS-Unity bridge installation by checking available packages:
   ```bash
   ros2 pkg list | grep unity
   ```

## Exercise 2: Set Up ROS-TCP Connection

1. Create a ROS node to manage the TCP connection (`src/physical_ai_robotics/ros_unity_bridge.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import JointState, LaserScan, Image
   from geometry_msgs.msg import Twist, Pose, Point
   from nav_msgs.msg import Odometry
   import socket
   import json
   import threading
   import time

   class ROSUnityBridge(Node):
       def __init__(self):
           super().__init__('ros_unity_bridge')

           # TCP server setup
           self.tcp_host = 'localhost'
           self.tcp_port = 10000
           self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
           self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
           self.tcp_socket.bind((self.tcp_host, self.tcp_port))
           self.tcp_socket.listen(1)
           self.tcp_socket.settimeout(1.0)  # Non-blocking with timeout

           self.client_socket = None
           self.client_address = None

           # ROS publishers and subscribers
           self.joint_state_publisher = self.create_publisher(JointState, '/unity_joint_states', 10)
           self.odom_publisher = self.create_publisher(Odometry, '/unity_odom', 10)
           self.cmd_vel_subscriber = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)

           # Timer for connection management
           self.connection_timer = self.create_timer(1.0, self.manage_connection)

           # Robot state
           self.robot_pose = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0}
           self.joint_positions = {}

           self.get_logger().info(f'ROS-Unity Bridge initialized, listening on {self.tcp_host}:{self.tcp_port}')

       def manage_connection(self):
           """Manage TCP connection to Unity"""
           if self.client_socket is None:
               try:
                   self.get_logger().info('Waiting for Unity connection...')
                   self.client_socket, self.client_address = self.tcp_socket.accept()
                   self.get_logger().info(f'Unity connected from {self.client_address}')
               except socket.timeout:
                   pass  # Expected when no connection
               except Exception as e:
                   self.get_logger().error(f'Error accepting connection: {e}')
           else:
               # Connection established, periodically send data
               self.send_robot_data_to_unity()

       def cmd_vel_callback(self, msg):
           """Handle velocity commands from ROS"""
           self.get_logger().debug(f'Received cmd_vel: linear={msg.linear.x}, angular={msg.angular.z}')
           # Forward to Unity if connected
           if self.client_socket:
               command = {
                   'type': 'cmd_vel',
                   'linear_x': msg.linear.x,
                   'angular_z': msg.angular.z
               }
               self.send_to_unity(command)

       def send_robot_data_to_unity(self):
           """Send robot state data to Unity"""
           if self.client_socket:
               try:
                   # Send current robot pose
                   pose_data = {
                       'type': 'robot_pose',
                       'pose': self.robot_pose
                   }
                   self.send_to_unity(pose_data)

                   # Send joint states
                   joint_data = {
                       'type': 'joint_states',
                       'positions': self.joint_positions
                   }
                   self.send_to_unity(joint_data)

               except Exception as e:
                   self.get_logger().error(f'Error sending data to Unity: {e}')
                   self.disconnect_unity()

       def send_to_unity(self, data):
           """Send JSON data to Unity"""
           try:
               json_data = json.dumps(data) + '\n'
               self.client_socket.send(json_data.encode('utf-8'))
           except Exception as e:
               self.get_logger().error(f'Error sending to Unity: {e}')
               self.disconnect_unity()

       def receive_from_unity(self):
           """Receive data from Unity (to be called in separate thread)"""
           while self.client_socket:
               try:
                   data = self.client_socket.recv(1024).decode('utf-8')
                   if data:
                       self.process_unity_data(data)
               except Exception as e:
                   self.get_logger().error(f'Error receiving from Unity: {e}')
                   self.disconnect_unity()
                   break

       def process_unity_data(self, data):
           """Process data received from Unity"""
           try:
               msg = json.loads(data)
               msg_type = msg.get('type')

               if msg_type == 'robot_pose':
                   self.robot_pose = msg.get('pose', self.robot_pose)
                   self.publish_odometry()
               elif msg_type == 'joint_states':
                   self.joint_positions = msg.get('positions', {})
                   self.publish_joint_states()
           except Exception as e:
               self.get_logger().error(f'Error processing Unity data: {e}')

       def publish_odometry(self):
           """Publish odometry data to ROS"""
           msg = Odometry()
           msg.header.stamp = self.get_clock().now().to_msg()
           msg.header.frame_id = 'odom'
           msg.child_frame_id = 'base_link'
           msg.pose.pose.position.x = self.robot_pose['x']
           msg.pose.pose.position.y = self.robot_pose['y']
           msg.pose.pose.position.z = self.robot_pose['z']
           msg.pose.pose.orientation.x = self.robot_pose['qx']
           msg.pose.pose.orientation.y = self.robot_pose['qy']
           msg.pose.pose.orientation.z = self.robot_pose['qz']
           msg.pose.pose.orientation.w = self.robot_pose['qw']

           self.odom_publisher.publish(msg)

       def publish_joint_states(self):
           """Publish joint state data to ROS"""
           msg = JointState()
           msg.header.stamp = self.get_clock().now().to_msg()
           msg.name = list(self.joint_positions.keys())
           msg.position = list(self.joint_positions.values())

           self.joint_state_publisher.publish(msg)

       def disconnect_unity(self):
           """Clean up Unity connection"""
           if self.client_socket:
               self.client_socket.close()
               self.client_socket = None
           self.client_address = None

   def main(args=None):
       rclpy.init(args=args)
       bridge = ROSUnityBridge()

       # Start receiving thread
       receive_thread = threading.Thread(target=bridge.receive_from_unity, daemon=True)
       receive_thread.start()

       try:
           rclpy.spin(bridge)
       except KeyboardInterrupt:
           pass
       finally:
           bridge.disconnect_unity()
           bridge.tcp_socket.close()
           bridge.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 3: Create Unity ROS Connector Script

1. In Unity, create a new C# script (`ROSConnector.cs`) in the Assets/Scripts folder:
   ```csharp
   using System.Collections;
   using System.Collections.Generic;
   using UnityEngine;
   using System.Net.Sockets;
   using System.Text;
   using System.Threading;
   using Newtonsoft.Json;

   public class ROSConnector : MonoBehaviour
   {
       [Header("Connection Settings")]
       public string ipAddress = "127.0.0.1";
       public int port = 10000;

       private TcpClient tcpClient;
       private NetworkStream stream;
       private Thread receiveThread;
       private bool isConnected = false;

       [Header("Robot References")]
       public GameObject robotBase;
       public Dictionary<string, Transform> jointMap = new Dictionary<string, Transform>();

       void Start()
       {
           ConnectToROS();
       }

       void ConnectToROS()
       {
           try
           {
               tcpClient = new TcpClient(ipAddress, port);
               stream = tcpClient.GetStream();
               isConnected = true;

               receiveThread = new Thread(ReceiveData);
               receiveThread.IsBackground = true;
               receiveThread.Start();

               Debug.Log("Connected to ROS bridge");
           }
           catch (System.Exception e)
           {
               Debug.LogError("Failed to connect to ROS: " + e.Message);
           }
       }

       void ReceiveData()
       {
           while (isConnected)
           {
               try
               {
                   byte[] buffer = new byte[1024];
                   int bytesRead = stream.Read(buffer, 0, buffer.Length);
                   if (bytesRead > 0)
                   {
                       string data = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                       ProcessReceivedData(data);
                   }
               }
               catch (System.Exception e)
               {
                   Debug.LogError("Error receiving data: " + e.Message);
                   isConnected = false;
                   break;
               }
           }
       }

       void ProcessReceivedData(string data)
       {
           // Split data by newline (in case multiple messages are received)
           string[] messages = data.Split('\n');
           foreach (string message in messages)
           {
               if (string.IsNullOrEmpty(message)) continue;

               try
               {
                   var jsonData = JsonConvert.DeserializeObject<Dictionary<string, object>>(message);
                   string msgType = jsonData["type"].ToString();

                   switch (msgType)
                   {
                       case "robot_pose":
                           UpdateRobotPose(jsonData);
                           break;
                       case "joint_states":
                           UpdateJointStates(jsonData);
                           break;
                       case "cmd_vel":
                           ProcessVelocityCommand(jsonData);
                           break;
                   }
               }
               catch (System.Exception e)
               {
                   Debug.LogError("Error parsing JSON: " + e.Message);
               }
           }
       }

       void UpdateRobotPose(Dictionary<string, object> data)
       {
           var poseData = (Dictionary<string, object>)data["pose"];
           Vector3 position = new Vector3(
               float.Parse(poseData["x"].ToString()),
               float.Parse(poseData["y"].ToString()),
               float.Parse(poseData["z"].ToString())
           );

           Quaternion rotation = new Quaternion(
               float.Parse(poseData["qx"].ToString()),
               float.Parse(poseData["qy"].ToString()),
               float.Parse(poseData["qz"].ToString()),
               float.Parse(poseData["qw"].ToString())
           );

           if (robotBase != null)
           {
               robotBase.transform.position = position;
               robotBase.transform.rotation = rotation;
           }
       }

       void UpdateJointStates(Dictionary<string, object> data)
       {
           var positions = (Dictionary<string, object>)data["positions"];
           foreach (var joint in positions)
           {
               string jointName = joint.Key;
               float position = float.Parse(joint.Value.ToString());

               if (jointMap.ContainsKey(jointName))
               {
                   // Update joint rotation based on position
                   // This is a simplified example - you may need to adjust based on joint type
                   jointMap[jointName].localRotation = Quaternion.Euler(0, position * Mathf.Rad2Deg, 0);
               }
           }
       }

       void ProcessVelocityCommand(Dictionary<string, object> data)
       {
           float linearX = float.Parse(data["linear_x"].ToString());
           float angularZ = float.Parse(data["angular_z"].ToString());

           // Apply velocity to robot (implementation depends on your robot setup)
           if (robotBase != null)
           {
               robotBase.transform.Translate(Vector3.forward * linearX * Time.deltaTime);
               robotBase.transform.Rotate(Vector3.up, angularZ * Mathf.Rad2Deg * Time.deltaTime);
           }
       }

       public void SendToROS(Dictionary<string, object> data)
       {
           if (isConnected && stream != null)
           {
               try
               {
                   string json = JsonConvert.SerializeObject(data) + "\n";
                   byte[] buffer = Encoding.UTF8.GetBytes(json);
                   stream.Write(buffer, 0, buffer.Length);
               }
               catch (System.Exception e)
               {
                   Debug.LogError("Error sending to ROS: " + e.Message);
               }
           }
       }

       void OnApplicationQuit()
       {
           isConnected = false;
           if (receiveThread != null)
           {
               receiveThread.Join(1000); // Wait up to 1 second for thread to finish
           }
           if (tcpClient != null)
           {
               tcpClient.Close();
           }
       }
   }
   ```

## Exercise 4: Set Up Unity Scene with ROS Connector

1. In Unity:
   - Create an empty GameObject named "ROSConnector"
   - Attach the ROSConnector.cs script to it
   - Configure the IP address and port (should match the ROS node settings)

2. Set up robot hierarchy:
   - Create your robot model with appropriate joints
   - Assign joint transforms to the jointMap in the ROSConnector component
   - Set the robotBase reference to the main robot object

## Exercise 5: Create Robot Control Node

1. Create a ROS node to control the robot (`src/physical_ai_robotics/digital_twin_controller.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import Twist
   from sensor_msgs.msg import JointState
   import time
   import math

   class DigitalTwinController(Node):
       def __init__(self):
           super().__init__('digital_twin_controller')

           self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
           self.joint_cmd_publisher = self.create_publisher(JointState, '/joint_commands', 10)

           # Timer for control loop
           self.timer = self.create_timer(0.1, self.control_loop)

           self.time_counter = 0.0
           self.get_logger().info('Digital Twin Controller initialized')

       def control_loop(self):
           """Send commands to robot in both Gazebo and Unity"""
           # Send velocity command for navigation
           cmd_vel = Twist()
           cmd_vel.linear.x = 0.5 * math.sin(self.time_counter * 0.5)  # Forward/backward
           cmd_vel.angular.z = 0.3 * math.sin(self.time_counter * 0.3)  # Turn
           self.cmd_vel_publisher.publish(cmd_vel)

           # Send joint commands for arm movement
           joint_cmd = JointState()
           joint_cmd.header.stamp = self.get_clock().now().to_msg()
           joint_cmd.name = ['shoulder_yaw', 'elbow_pitch', 'wrist_roll']
           joint_cmd.position = [
               0.4 * math.sin(self.time_counter * 0.7),
               0.3 * math.sin(self.time_counter * 0.9),
               0.2 * math.sin(self.time_counter * 1.1)
           ]
           self.joint_cmd_publisher.publish(joint_cmd)

           self.time_counter += 0.1

   def main(args=None):
       rclpy.init(args=args)
       controller = DigitalTwinController()
       rclpy.spin(controller)
       controller.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 6: Launch Complete Digital Twin System

1. Create a launch file for the complete digital twin system (`launch/digital_twin.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
   from launch.event_handlers import OnProcessStart
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='true')

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),

           # Robot state publisher
           Node(
               package='robot_state_publisher',
               executable='robot_state_publisher',
               name='robot_state_publisher',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # ROS-Unity bridge
           Node(
               package='physical_ai_robotics',
               executable='ros_unity_bridge',
               name='ros_unity_bridge',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Digital twin controller
           Node(
               package='physical_ai_robotics',
               executable='digital_twin_controller',
               name='digital_twin_controller',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           )
       ])
   ```

## Exercise 7: Test ROS-Unity Communication

1. Launch the ROS side of the system:
   ```bash
   ros2 launch physical_ai_robotics digital_twin.launch.py
   ```

2. In Unity, press Play to start the simulation
3. Verify that the robot moves in Unity based on ROS commands
4. Check that Unity sends pose and joint state data back to ROS

## Exercise 8: Add Sensor Simulation in Unity

1. Create a Unity script to simulate sensors (`UnitySensorSimulator.cs`):
   ```csharp
   using UnityEngine;
   using System.Collections.Generic;

   public class UnitySensorSimulator : MonoBehaviour
   {
       [Header("Sensor Settings")]
       public float lidarRange = 30.0f;
       public int lidarResolution = 360;
       public LayerMask detectionMask = -1;

       [Header("References")]
       public ROSConnector rosConnector;

       private float[] lidarReadings;

       void Start()
       {
           lidarReadings = new float[lidarResolution];
       }

       void Update()
       {
           if (Time.frameCount % 10 == 0) // Update every 10 frames for performance
           {
               SimulateLidar();
           }
       }

       void SimulateLidar()
       {
           // Simulate 360-degree LiDAR scan
           for (int i = 0; i < lidarResolution; i++)
           {
               float angle = (float)i * 360.0f / lidarResolution;
               Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;

               RaycastHit hit;
               if (Physics.Raycast(transform.position, direction, out hit, lidarRange, detectionMask))
               {
                   lidarReadings[i] = hit.distance;
               }
               else
               {
                   lidarReadings[i] = lidarRange; // No obstacle detected
               }
           }

           // Send sensor data to ROS
           if (rosConnector != null)
           {
               Dictionary<string, object> sensorData = new Dictionary<string, object>();
               sensorData["type"] = "sensor_data";
               sensorData["sensor_type"] = "lidar";
               sensorData["ranges"] = lidarReadings;
               sensorData["angle_min"] = -Mathf.PI;
               sensorData["angle_max"] = Mathf.PI;
               sensorData["angle_increment"] = 2 * Mathf.PI / lidarResolution;

               rosConnector.SendToROS(sensorData);
           }
       }
   }
   ```

2. Attach this script to your robot in Unity
3. Configure the detection mask to detect appropriate objects

## Exercise 9: Visualize Unity Sensor Data in ROS

1. Create a ROS node to process Unity sensor data (`src/physical_ai_robotics/unity_sensor_processor.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import LaserScan
   import json

   class UnitySensorProcessor(Node):
       def __init__(self):
           super().__init__('unity_sensor_processor')

           self.scan_publisher = self.create_publisher(LaserScan, '/unity_scan', 10)
           # Subscribe to Unity sensor data via ROS bridge

           self.get_logger().info('Unity Sensor Processor initialized')

       def process_unity_sensor_data(self, data):
           """Process sensor data received from Unity"""
           if data.get('sensor_type') == 'lidar':
               scan_msg = LaserScan()
               scan_msg.header.stamp = self.get_clock().now().to_msg()
               scan_msg.header.frame_id = 'lidar_link'
               scan_msg.angle_min = data.get('angle_min', -3.14)
               scan_msg.angle_max = data.get('angle_max', 3.14)
               scan_msg.angle_increment = data.get('angle_increment', 0.0174533)  # Default to ~1 degree
               scan_msg.range_min = 0.1
               scan_msg.range_max = 30.0
               scan_msg.ranges = data.get('ranges', [])

               self.scan_publisher.publish(scan_msg)

   def main(args=None):
       rclpy.init(args=args)
       processor = UnitySensorProcessor()
       rclpy.spin(processor)
       processor.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 10: Create Digital Twin Visualization

1. Set up RViz to visualize both Gazebo and Unity data:
   - Add RobotModel displays for both simulators
   - Add LaserScan displays for both sensor sources
   - Add TF displays to visualize coordinate frames

2. Create a visualization that shows the digital twin concept:
   - Side-by-side comparison of Gazebo and Unity
   - Synchronized robot movements
   - Sensor data overlay

## Exercise 11: Implement Synchronization

1. Create a synchronization node to ensure consistency between simulators:
   ```python
   # Create a node that ensures both simulators stay synchronized
   # Monitor state differences and apply corrections if needed
   ```

## Exercise 12: Performance Testing

1. Test the system with various loads:
   - Multiple robots
   - Complex environments
   - High-frequency sensor updates
   - Long-duration runs

2. Monitor performance metrics:
   - Network latency
   - Frame rates in Unity
   - ROS message rates

## Verification Steps

1. Confirm that ROS and Unity communicate bidirectionally
2. Verify that robot movements are synchronized between simulators
3. Check that sensor data flows correctly from Unity to ROS
4. Ensure that control commands are executed in both simulators

## Expected Outcomes

- Understanding of ROS-Unity bridge implementation
- Knowledge of bidirectional communication patterns
- Experience with digital twin synchronization
- Ability to create integrated simulation systems

## Troubleshooting

- If connection fails, check IP addresses and ports
- If data doesn't flow, verify JSON format and message structure
- If performance is poor, reduce update rates or optimize data transmission

## Next Steps

After completing this digital twin project, proceed to Module 3 to learn about Isaac Sim integration and AI-driven robotics.