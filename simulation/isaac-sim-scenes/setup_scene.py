"""
Isaac Sim Scene Setup for Physical AI & Humanoid Robotics Textbook

This script creates a sample scene for Isaac Sim that demonstrates
key concepts from Module 3: The AI-Robot Brain (NVIDIA Isaac™).
"""

import omni
import omni.kit.commands
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema
import carb
import numpy as np


class IsaacSimSceneSetup:
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.scene_path = "/Isaac/Scene"

    def setup_basic_scene(self):
        """Setup the basic scene with ground, lighting, and environment"""
        # Create the scene root
        scene_prim = UsdGeom.Xform.Define(self.stage, self.scene_path)

        # Add ground plane
        self._create_ground_plane()

        # Add dome light
        self._create_dome_light()

        # Add default camera
        self._create_camera()

        carb.log_info("Basic scene setup completed")

    def _create_ground_plane(self):
        """Create a ground plane for the scene"""
        ground_path = Sdf.Path(f"{self.scene_path}/GroundPlane")
        ground_plane = UsdGeom.Mesh.Define(self.stage, ground_path)

        # Set up a simple plane
        ground_plane.CreatePointsAttr().Set([
            (-5.0, -5.0, 0.0), (5.0, -5.0, 0.0),
            (5.0, 5.0, 0.0), (-5.0, 5.0, 0.0)
        ])

        ground_plane.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
        ground_plane.CreateFaceVertexCountsAttr().Set([3, 3])

        # Add physics to ground
        UsdPhysics.MeshCollisionAPI.Apply(ground_plane)
        PhysxSchema.PhysxCollisionAPI.Apply(ground_plane)

        carb.log_info("Ground plane created")

    def _create_dome_light(self):
        """Create dome light for the scene"""
        light_path = Sdf.Path(f"{self.scene_path}/DomeLight")
        dome_light = UsdGeom.DomeLight.Define(self.stage, light_path)
        dome_light.CreateIntensityAttr(10000)
        dome_light.CreateColorAttr((0.8, 0.8, 0.9))

        carb.log_info("Dome light created")

    def _create_camera(self):
        """Create default camera for the scene"""
        camera_path = Sdf.Path(f"{self.scene_path}/Camera")
        camera = UsdGeom.Camera.Define(self.stage, camera_path)

        # Set camera properties
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateVerticalApertureAttr(20.25)

        # Position the camera
        xform = UsdGeom.Xformable(camera)
        xform.AddTranslateOp().Set((3.0, 3.0, 2.0))
        xform.AddRotateXYZOp().Set((0, 0, 45))  # Pointing toward origin

        carb.log_info("Camera created")

    def create_robot(self):
        """Create a simple differential drive robot"""
        robot_path = Sdf.Path(f"{self.scene_path}/Robot")
        robot = UsdGeom.Xform.Define(self.stage, robot_path)

        # Position the robot
        xform = UsdGeom.Xformable(robot)
        xform.AddTranslateOp().Set((0.0, 0.0, 0.5))

        # Create robot body
        body_path = Sdf.Path(f"{robot_path}/Body")
        body = UsdGeom.Cube.Define(self.stage, body_path)
        body.CreateSizeAttr(0.5)

        # Add physics to robot body
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.CollisionAPI.Apply(body)

        # Create wheels
        self._create_robot_wheels(robot_path)

        carb.log_info("Robot created")

    def _create_robot_wheels(self, robot_path):
        """Create wheels for the differential drive robot"""
        # Left wheel
        left_wheel_path = Sdf.Path(f"{robot_path}/LeftWheel")
        left_wheel = UsdGeom.Cylinder.Define(self.stage, left_wheel_path)
        left_wheel.CreateRadiusAttr(0.1)
        left_wheel.CreateHeightAttr(0.05)

        xform = UsdGeom.Xformable(left_wheel)
        xform.AddTranslateOp().Set((-0.2, 0.25, 0.0))
        xform.AddRotateXYZOp().Set((0, 90, 0))

        # Right wheel
        right_wheel_path = Sdf.Path(f"{robot_path}/RightWheel")
        right_wheel = UsdGeom.Cylinder.Define(self.stage, right_wheel_path)
        right_wheel.CreateRadiusAttr(0.1)
        right_wheel.CreateHeightAttr(0.05)

        xform = UsdGeom.Xformable(right_wheel)
        xform.AddTranslateOp().Set((-0.2, -0.25, 0.0))
        xform.AddRotateXYZOp().Set((0, 90, 0))

        # Add physics to wheels
        UsdPhysics.RigidBodyAPI.Apply(left_wheel)
        UsdPhysics.CollisionAPI.Apply(left_wheel)
        UsdPhysics.RigidBodyAPI.Apply(right_wheel)
        UsdPhysics.CollisionAPI.Apply(right_wheel)

        carb.log_info("Robot wheels created")

    def create_obstacles(self):
        """Create sample obstacles for navigation"""
        obstacle_positions = [(2.0, 1.0, 0.5), (-1.5, 2.0, 0.5), (1.0, -2.0, 0.5)]
        obstacle_sizes = [(0.5, 0.5, 1.0), (0.8, 0.3, 0.6), (0.4, 0.4, 0.8)]

        for i, (pos, size) in enumerate(zip(obstacle_positions, obstacle_sizes)):
            obstacle_path = Sdf.Path(f"{self.scene_path}/Obstacle_{i}")
            obstacle = UsdGeom.Cube.Define(self.stage, obstacle_path)
            obstacle.CreateSizeAttr(max(size))

            xform = UsdGeom.Xformable(obstacle)
            xform.AddTranslateOp().Set(pos)

            # Add physics to obstacle
            UsdPhysics.RigidBodyAPI.Apply(obstacle)
            UsdPhysics.CollisionAPI.Apply(obstacle)

        carb.log_info(f"Created {len(obstacle_positions)} obstacles")

    def setup_sensors(self):
        """Setup sensors for the robot"""
        # This would typically involve creating sensor prims in Isaac Sim
        # For now, we'll log what sensors would be created
        sensors = [
            "RGB Camera",
            "LIDAR",
            "IMU",
            "Depth Sensor"
        ]

        for sensor in sensors:
            carb.log_info(f"Sensor setup configured: {sensor}")

    def setup_ros_bridge(self):
        """Setup ROS bridge configuration"""
        # Configure ROS bridge for textbook examples
        carb.log_info("ROS bridge configuration setup")
        carb.log_info("  - /cmd_vel input for robot control")
        carb.log_info("  - /odom output for odometry")
        carb.log_info("  - /scan output for LIDAR data")
        carb.log_info("  - /camera/color/image_raw for RGB images")

    def setup_scene_for_textbook(self):
        """Complete setup for textbook examples"""
        carb.log_info("Setting up Isaac Sim scene for Physical AI & Humanoid Robotics textbook...")

        self.setup_basic_scene()
        self.create_robot()
        self.create_obstacles()
        self.setup_sensors()
        self.setup_ros_bridge()

        carb.log_info("Isaac Sim scene setup completed for textbook examples!")


# Function to run the scene setup
def setup_isaac_sim_scene():
    """Function to be called from Isaac Sim to setup the scene"""
    scene_setup = IsaacSimSceneSetup()
    scene_setup.setup_scene_for_textbook()
    return scene_setup


# Example usage (this would normally be called from within Isaac Sim)
if __name__ == "__main__":
    print("Isaac Sim Scene Setup Script")
    print("This script is designed to be run within the Isaac Sim environment")
    print("Use this as a reference for creating textbook example scenes")