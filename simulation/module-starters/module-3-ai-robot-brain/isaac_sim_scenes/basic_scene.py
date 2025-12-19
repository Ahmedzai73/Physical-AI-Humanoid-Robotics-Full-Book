#!/usr/bin/env python3

"""
Basic Isaac Sim Scene for Physical AI & Humanoid Robotics Textbook
Module 3: The AI-Robot Brain (NVIDIA Isaac™)

This script creates a basic scene in Isaac Sim for AI robot brain examples.
"""

import omni
import omni.kit.commands
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.semantics import add_semantics


class BasicIsaacScene:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.stage = omni.usd.get_context().get_stage()

        carb.log_info("Basic Isaac Sim scene initialized")

    def setup_environment(self):
        """Setup the basic environment with ground, lighting, and obstacles"""
        # Create ground plane
        self.world.scene.add_ground_plane(
            "ground_plane",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.1
        )

        # Add dome light
        dome_light = UsdGeom.DomeLight.Define(self.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(1000)
        dome_light.CreateColorAttr((0.8, 0.8, 0.9))

        # Add a simple room with walls
        self._create_walls()

        # Add some objects for perception
        self._create_objects_for_perception()

        carb.log_info("Environment setup completed")

    def _create_walls(self):
        """Create simple walls for the environment"""
        wall_material = self.world.materials.VisualMaterial(
            prim_path="/World/looks/wall_material",
            diffuse_color=np.array([0.5, 0.5, 0.5]),
            metallic=0.0,
            roughness=0.8
        )

        # Back wall
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/BackWall",
                name="back_wall",
                position=np.array([0, 5, 1]),
                size=np.array([10, 0.2, 2]),
                color=np.array([0.5, 0.5, 0.5])
            )
        )

        # Front wall
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/FrontWall",
                name="front_wall",
                position=np.array([0, -5, 1]),
                size=np.array([10, 0.2, 2]),
                color=np.array([0.5, 0.5, 0.5])
            )
        )

        # Left wall
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/LeftWall",
                name="left_wall",
                position=np.array([-5, 0, 1]),
                size=np.array([0.2, 10, 2]),
                color=np.array([0.4, 0.4, 0.4])
            )
        )

        # Right wall
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/RightWall",
                name="right_wall",
                position=np.array([5, 0, 1]),
                size=np.array([0.2, 10, 2]),
                color=np.array([0.4, 0.4, 0.4])
            )
        )

    def _create_objects_for_perception(self):
        """Create objects that can be used for perception examples"""
        # Add some colored cubes for object detection
        from omni.isaac.core.objects import DynamicCuboid

        # Red cube
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/RedCube",
                name="red_cube",
                position=np.array([2, 2, 0.5]),
                size=0.3,
                color=np.array([0.8, 0.1, 0.1])
            )
        )

        # Green cube
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/GreenCube",
                name="green_cube",
                position=np.array([-2, 1, 0.5]),
                size=0.3,
                color=np.array([0.1, 0.8, 0.1])
            )
        )

        # Blue cube
        self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/BlueCube",
                name="blue_cube",
                position=np.array([0, -2, 0.5]),
                size=0.3,
                color=np.array([0.1, 0.1, 0.8])
            )
        )

        # Add semantics to objects
        add_semantics(self.world.scene.get_object("red_cube").prim, "cube")
        add_semantics(self.world.scene.get_object("green_cube").prim, "cube")
        add_semantics(self.world.scene.get_object("blue_cube").prim, "cube")

    def setup_robot(self):
        """Setup a robot in the scene"""
        # For this example, we'll use a simple robot model
        # In practice, you would load a more complex robot
        try:
            # Try to add a simple wheeled robot
            self.robot = self.world.scene.add(
                Robot(
                    prim_path="/World/Robot",
                    name="basic_robot",
                    usd_path="/Isaac/Robots/TurtleBot3/nav_sensors.usd",  # Example path
                    position=[0, 0, 0.2],
                    orientation=[0, 0, 0, 1]
                )
            )
            carb.log_info("Robot setup completed")
        except Exception as e:
            carb.log_warn(f"Could not add robot from USD: {e}")
            carb.log_info("Using fallback robot setup")

            # Fallback: create a simple robot representation
            from omni.isaac.core.objects import DynamicCuboid
            self.robot = self.world.scene.add(
                DynamicCuboid(
                    prim_path="/World/Robot",
                    name="simple_robot",
                    position=np.array([0, 0, 0.5]),
                    size=np.array([0.5, 0.3, 0.2]),
                    color=np.array([0.0, 0.5, 1.0])
                )
            )

    def setup_sensors(self):
        """Setup sensors for the robot"""
        # This is where you would add camera, LIDAR, and other sensors
        # For this basic example, we'll just log that sensors would be added
        carb.log_info("Sensor setup for robot would be configured here")
        carb.log_info("  - RGB camera")
        carb.log_info("  - Depth sensor")
        carb.log_info("  - IMU")
        carb.log_info("  - LIDAR")

    def setup_scene(self):
        """Complete scene setup"""
        carb.log_info("Setting up Isaac Sim scene for AI robot brain examples...")

        self.setup_environment()
        self.setup_robot()
        self.setup_sensors()

        carb.log_info("Isaac Sim scene setup completed!")

    def run_simulation(self, steps=1000):
        """Run the simulation for a number of steps"""
        carb.log_info(f"Running simulation for {steps} steps...")

        self.world.reset()

        for i in range(steps):
            self.world.step(render=True)

            if i % 100 == 0:
                carb.log_info(f"Simulation step {i}/{steps}")

        carb.log_info("Simulation completed")


# Import required classes for the example
try:
    from omni.isaac.core.objects import FixedCuboid
except ImportError:
    # Create a simple placeholder if not available in this context
    class FixedCuboid:
        def __init__(self, prim_path, name, position, size, color):
            self.prim_path = prim_path
            self.name = name
            self.position = position
            self.size = size
            self.color = color


def main():
    """Main function to run the Isaac Sim scene setup"""
    carb.log_info("Starting Isaac Sim Basic Scene Setup")

    # This would normally be run within the Isaac Sim environment
    # For demonstration purposes, we'll just show what would happen
    scene = BasicIsaacScene()
    scene.setup_scene()

    carb.log_info("Scene setup complete. Run this script within Isaac Sim for full functionality.")


if __name__ == "__main__":
    main()