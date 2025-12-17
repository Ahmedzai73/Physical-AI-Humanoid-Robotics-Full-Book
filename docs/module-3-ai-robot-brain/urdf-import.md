# Importing URDF into Isaac Sim with ROS 2 Bridges

## Understanding URDF to USD Conversion

Universal Robot Description Format (URDF) is the standard format for describing robot models in ROS, while Isaac Sim uses Universal Scene Description (USD) as its native format. The conversion from URDF to USD is a critical step in bringing your humanoid robot into Isaac Sim for simulation and AI development.

The conversion process involves:
- Translating kinematic chains and joint definitions
- Converting visual and collision geometries
- Mapping materials and textures
- Preserving physical properties like mass and inertia
- Setting up ROS 2 bridge connections for real-time control

## Preparing Your URDF Model

Before importing, ensure your URDF model is properly structured for Isaac Sim compatibility:

### URDF Prerequisites
1. **Complete kinematic chain**: Ensure all joints are properly connected
2. **Valid physical properties**: Check mass, inertia, and friction values
3. **Proper mesh references**: Ensure all mesh files are accessible
4. **Joint limits**: Define appropriate limits for humanoid joints
5. **Fixed base**: Decide if your robot has a fixed base or floating base

### Example Humanoid URDF Structure
```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="package://humanoid_description/meshes/base.dae"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="package://humanoid_description/meshes/base_collision.dae"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Hip joint and links -->
  <joint name="hip_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso_link"/>
  </joint>

  <link name="torso_link">
    <!-- Torso definition -->
  </link>

  <!-- Additional joints and links for legs, arms, etc. -->
</robot>
```

## Importing URDF via Isaac Sim GUI

### Method 1: Direct Import
1. **Open Isaac Sim**
   - Launch Isaac Sim from your installation directory
   - Ensure the ROS 2 bridge extension is enabled

2. **Import URDF**
   - Go to `File` > `Import` > `URDF`
   - Navigate to your URDF file location
   - Select the URDF file and click "Open"

3. **Configure Import Settings**
   - Set the import scale if needed (default is usually 1.0)
   - Choose whether to create collision meshes automatically
   - Select import options for visual materials
   - Configure joint properties for simulation

4. **Verify Import**
   - Check that all links and joints appear correctly
   - Verify that the robot maintains proper proportions
   - Test joint movement in the viewport

### Method 2: Using the URDF Importer Extension
1. **Enable URDF Importer Extension**
   - Go to `Window` > `Extensions`
   - Search for "URDF Importer"
   - Enable the extension

2. **Import Process**
   - Use the extension's UI to specify URDF file path
   - Configure import parameters through the extension interface
   - Execute the import operation

## Programmatic URDF Import

For automation and batch processing, you can import URDF programmatically:

```python
# import_urdf_script.py
import omni
from omni.isaac.urdf import _urdf
import carb

def import_urdf_to_isaac(urdf_path, prim_path="/World/HumanoidRobot",
                        import_config=None):
    """
    Import a URDF file into Isaac Sim programmatically

    Args:
        urdf_path: Path to the URDF file
        prim_path: USD prim path where robot will be placed
        import_config: Optional import configuration
    """

    # Initialize URDF importer
    urdf_interface = _urdf.acquire_urdf_interface()

    # Set import configuration
    if import_config is None:
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = False
        import_config.make_default_prim = False
        import_config.self_collision = False
        import_config.import_inertia_tensor = True
        import_config.default_drive_strength = 20000
        import_config.default_position_drive_damping = 200

    # Import the URDF
    imported_robot_path = urdf_interface.parse_urdf(
        urdf_path,
        import_config
    )

    # Create the robot in the stage
    robot_path = urdf_interface.import_robot(
        imported_robot_path,
        prim_path,
        import_config
    )

    carb.log_info(f"Robot imported to: {robot_path}")

    return robot_path

# Example usage
urdf_file_path = "/path/to/your/humanoid.urdf"
robot_prim_path = "/World/HumanoidRobot"
imported_robot = import_urdf_to_isaac(urdf_file_path, robot_prim_path)

print(f"URDF imported successfully to {imported_robot}")
```

## Setting Up ROS 2 Bridges

### Configuring Joint State Bridge
To enable communication between ROS 2 and Isaac Sim:

```python
# setup_joint_state_bridge.py
import omni
from omni.isaac.core.utils.extensions import enable_extension
import omni.isaac.ros_bridge._ros_bridge as ros_bridge

def setup_joint_state_bridge(robot_path, joint_names):
    """
    Set up ROS 2 bridge for joint states
    """
    # Enable ROS bridge extension
    enable_extension("omni.isaac.ros_bridge")

    # Create joint state publisher
    ros_bridge.ROSJointStatePublisher(
        prim_path=robot_path,
        topic_name="joint_states",
        joint_names=joint_names,
        publish_rate=50  # Hz
    )

    print("Joint state bridge configured")

# Example usage
joint_names = [
    "left_hip_joint", "left_knee_joint", "left_ankle_joint",
    "right_hip_joint", "right_knee_joint", "right_ankle_joint",
    # Add other joint names as needed
]

setup_joint_state_bridge("/World/HumanoidRobot", joint_names)
```

### Configuring Joint Command Bridge
For controlling the robot from ROS 2:

```python
# setup_joint_command_bridge.py
def setup_joint_command_bridge(robot_path, joint_names):
    """
    Set up ROS 2 bridge for joint commands
    """
    # Create joint command subscriber
    ros_bridge.ROSJointCommandSubscriber(
        prim_path=robot_path,
        topic_name="joint_commands",
        joint_names=joint_names,
        position_only=True  # Set to False for position, velocity, effort
    )

    print("Joint command bridge configured")
```

## Humanoid-Specific Import Considerations

### Bipedal Locomotion Setup
When importing humanoid robots, special attention is needed for:

1. **Foot Contact Points**
   ```python
   # Add contact sensors to feet for locomotion
   # In Isaac Sim, add contact sensors to foot links
   # Configure appropriate friction coefficients for stable walking
   ```

2. **Center of Mass**
   - Ensure the URDF has accurate center of mass information
   - Verify that the CoM is properly positioned for stable bipedal locomotion
   - Adjust if necessary for simulation stability

3. **Joint Configuration**
   - Hip joints: typically revolute with 3-DOF (roll, pitch, yaw)
   - Knee joints: 1-DOF for forward/backward movement
   - Ankle joints: 2-DOF (pitch, roll) for balance
   - Spine joints: 3-DOF for upper body movement

### Multi-Body Dynamics
For complex humanoid models:

```python
# setup_multibody_dynamics.py
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import get_prim_at_path

def setup_humanoid_articulation(robot_path):
    """
    Configure the imported robot as an articulation for physics simulation
    """
    # Get the robot prim
    robot_prim = get_prim_at_path(robot_path)

    # Create articulation
    humanoid_robot = Articulation(
        prim_path=robot_path,
        name="humanoid_robot",
        position=[0, 0, 1.0],  # Start slightly above ground
        orientation=[0, 0, 0, 1]
    )

    # Initialize the articulation
    world = omni.isaac.core.World()
    world.add_articulation(humanoid_robot)

    print("Humanoid articulation configured")
```

## Troubleshooting Common Import Issues

### Mesh Import Problems
```bash
# Common issues and solutions:

# Issue: Meshes not appearing
# Solution: Check file paths and ensure meshes are in correct format (dae, obj, glb)
# Verify that the package:// URDF paths are correctly resolved

# Issue: Missing textures
# Solution: Ensure texture files are in the correct location
# Check material definitions in URDF

# Issue: Scale problems
# Solution: Verify import scale settings
# Check if URDF uses meters or millimeters as units
```

### Joint Configuration Issues
```python
# Verify joint limits after import
def verify_joint_limits(robot_articulation):
    """Check that imported joints have appropriate limits"""
    joint_names = robot_articulation.dof_names
    for i, joint_name in enumerate(joint_names):
        lower_limit = robot_articulation.get_dof_lower_limits()[i]
        upper_limit = robot_articulation.get_dof_upper_limits()[i]

        print(f"Joint {joint_name}: [{lower_limit}, {upper_limit}]")

        # Check if limits are reasonable for humanoid
        if abs(upper_limit - lower_limit) > 6.28:  # More than 1 revolution
            print(f"Warning: Joint {joint_name} has unusually wide limits")
```

### Physics Stability Issues
- **Increase solver iterations** for better stability
- **Adjust joint damping** values if oscillations occur
- **Verify mass and inertia** properties match real robot
- **Check collision geometry** for proper overlap prevention

## Optimizing Imported Models

### Performance Optimization
1. **Simplify collision geometry** where possible
2. **Reduce mesh complexity** for non-visual elements
3. **Use proxy shapes** for complex geometries
4. **Optimize material definitions** for rendering performance

### Simulation Optimization
1. **Adjust physics substeps** for stable simulation
2. **Configure appropriate solver settings** for humanoid dynamics
3. **Set proper collision filtering** to reduce unnecessary calculations
4. **Use level-of-detail (LOD)** models when appropriate

## Testing the Imported Model

Create a comprehensive test to verify the imported model works correctly:

```python
# test_imported_model.py
import omni
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np

def test_imported_humanoid(robot_path):
    """
    Test the imported humanoid model for basic functionality
    """
    # Create world
    world = World(stage_units_in_meters=1.0)

    # Add robot to world
    robot = world.scene.add(
        Articulation(
            prim_path=robot_path,
            name="humanoid_robot",
            position=[0, 0, 1.0]
        )
    )

    # Reset the world
    world.reset()

    # Test joint access
    joint_names = robot.dof_names
    print(f"Detected {len(joint_names)} joints:")
    for name in joint_names:
        print(f"  - {name}")

    # Test joint position setting
    initial_positions = robot.get_joint_positions()
    print(f"Initial joint positions: {initial_positions[:5]}...")  # Show first 5

    # Apply small joint position change
    new_positions = initial_positions.copy()
    new_positions[0] += 0.1  # Move first joint slightly
    robot.set_joint_positions(new_positions)

    # Step simulation to see if robot responds
    for i in range(10):
        world.step(render=True)

    print("✓ Imported model test completed successfully")

    return robot

# Example usage
# robot = test_imported_humanoid("/World/HumanoidRobot")
```

## Best Practices for URDF Import

### Before Import
- Validate your URDF using `check_urdf` command
- Ensure all mesh files exist and are accessible
- Verify physical properties (mass, inertia) are reasonable
- Test the URDF in RViz to ensure it displays correctly

### After Import
- Verify all joints move as expected
- Check that the robot maintains stability when idle
- Test range of motion for all joints
- Validate that ROS 2 bridges publish/subscribe correctly

### Performance Considerations
- Keep mesh complexity reasonable for real-time simulation
- Use simplified collision meshes where visual fidelity isn't needed
- Consider using proxy shapes for complex geometries
- Balance visual quality with simulation performance

## Summary

Importing URDF models into Isaac Sim is a critical step in developing AI-powered humanoid robots. The process involves:

1. Preparing your URDF model with proper structure and properties
2. Importing via GUI or programmatically
3. Setting up ROS 2 bridges for communication
4. Configuring humanoid-specific parameters for stable locomotion
5. Testing and optimizing the imported model

The successful import of your humanoid robot into Isaac Sim enables the development of sophisticated AI capabilities that will be explored in the remaining chapters of this module. Properly configured, your imported robot will serve as the foundation for perception, navigation, and behavior learning in photorealistic simulation environments.