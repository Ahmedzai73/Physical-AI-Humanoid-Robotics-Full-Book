---
title: Physics Simulation - Gravity, Friction, Collisions, and Inertia
sidebar_position: 4
description: Configuring realistic physics properties for humanoid robot simulation in Gazebo
---

# Physics Simulation: Gravity, Friction, Collisions, and Inertia

## Introduction

Physics simulation is the cornerstone of creating realistic digital twins for humanoid robots. In this chapter, we'll explore how to configure physics properties in Gazebo to ensure your humanoid robot behaves realistically in simulation. Proper physics configuration is essential for humanoid robots because they interact with the environment through many contact points (feet, hands) and must maintain balance under gravitational forces.

For humanoid robots, physics simulation must accurately represent:
- **Gravity effects**: How the robot responds to downward forces
- **Friction properties**: How feet grip the ground, hands grasp objects
- **Collision detection**: How different body parts interact with each other and the environment
- **Inertial properties**: How mass distribution affects movement and stability

## Understanding Gazebo Physics Concepts

### 1. Physics Engines in Gazebo

Gazebo supports multiple physics engines, each with different characteristics:

#### ODE (Open Dynamics Engine)
- **Default engine** for most Gazebo installations
- **Strengths**: Stable, mature, good for ground vehicles and basic robotics
- **Best for**: Basic humanoid locomotion, manipulation tasks
- **Configuration**:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

#### Bullet Physics
- **Alternative engine** with different numerical methods
- **Strengths**: Better contact stability, good for complex interactions
- **Best for**: Complex contact scenarios, detailed manipulation

### 2. Key Physics Parameters for Humanoid Robots

#### Time Step Configuration
For humanoid robots with many joints and contacts:
- **Recommended time step**: 0.001s (1ms) for stable simulation
- **Maximum**: 0.01s (10ms) for performance
- **Minimum**: 0.0001s (0.1ms) for high-precision control

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Critical for humanoid stability -->
  <real_time_update_rate>1000</real_time_update_rate>  <!-- 1000Hz update rate -->
  <real_time_factor>1</real_time_factor>  <!-- Real-time simulation -->
</physics>
```

#### Solver Parameters
- **Iterations**: Higher values improve stability but reduce performance
- **SOR (Successive Over-Relaxation)**: Controls convergence rate (typically 1.2-1.3)

## Gravity Configuration

Gravity is fundamental for humanoid robots as it affects balance, locomotion, and all interactions with the ground.

### 1. Global Gravity Settings

The global gravity setting affects all objects in the simulation world:

```xml
<!-- In world file -->
<world name="humanoid_world">
  <gravity>0 0 -9.8</gravity>  <!-- Earth's gravity: 9.8 m/s² downward -->
</world>
```

### 2. Custom Gravity for Special Scenarios

For simulating different environments (moon, Mars, zero-g):

```xml
<!-- Moon gravity (~1/6 Earth) -->
<gravity>0 0 -1.63</gravity>

<!-- Mars gravity (~0.38 Earth) -->
<gravity>0 0 -3.71</gravity>

<!-- Zero gravity (space robotics) -->
<gravity>0 0 0</gravity>
```

### 3. Gravity Compensation in Control

For humanoid robots, you may want to implement gravity compensation in your controllers:

```python
# Example gravity compensation in a humanoid controller
import numpy as np

class HumanoidGravityCompensation:
    def __init__(self, robot_mass, gravity=9.81):
        self.mass = robot_mass
        self.gravity = gravity
        self.g = gravity  # Acceleration due to gravity

    def calculate_gravity_compensation(self, joint_positions, link_properties):
        """
        Calculate torques needed to compensate for gravity
        This is a simplified example - real implementation would use dynamics models
        """
        # This would typically use robot dynamics libraries like Pinocchio
        gravity_torques = []

        for i, joint_pos in enumerate(joint_positions):
            # Calculate gravity effect on each link
            # This is a simplified calculation
            gravity_effect = link_properties[i]['mass'] * self.g * np.sin(joint_pos)
            gravity_torques.append(gravity_effect)

        return np.array(gravity_torques)
```

## Friction Properties

Friction is critical for humanoid robots as it determines how well they can grip surfaces and maintain balance.

### 1. Friction Coefficients for Different Surfaces

Different surfaces require different friction coefficients:

```xml
<!-- High friction surface (rubber on concrete) -->
<gazebo reference="foot_link">
  <mu1>1.0</mu1>  <!-- Primary friction coefficient -->
  <mu2>1.0</mu2>  <!-- Secondary friction coefficient -->
</gazebo>

<!-- Medium friction surface (shoes on wood) -->
<gazebo reference="foot_link">
  <mu1>0.6</mu1>
  <mu2>0.6</mu2>
</gazebo>

<!-- Low friction surface (ice) -->
<gazebo reference="foot_link">
  <mu1>0.1</mu1>
  <mu2>0.1</mu2>
</gazebo>
```

### 2. Anisotropic Friction

For surfaces with different friction in different directions (like ice skates):

```xml
<gazebo reference="special_surface">
  <mu1>0.1</mu1>    <!-- Low friction in primary direction -->
  <mu2>0.8</mu2>    <!-- High friction in secondary direction -->
  <fdir1>1 0 0</fdir1>  <!-- Direction of mu1 -->
</gazebo>
```

### 3. Friction for Different Robot Parts

Different robot parts need different friction properties:

```xml
<!-- Feet - high friction for stability -->
<gazebo reference="left_foot">
  <mu1>0.9</mu1>
  <mu2>0.9</mu2>
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>100.0</kd>      <!-- Contact damping -->
</gazebo>

<!-- Hands - moderate friction for grasping -->
<gazebo reference="left_hand">
  <mu1>0.7</mu1>
  <mu2>0.7</mu2>
  <kp>500000.0</kp>
  <kd>50.0</kd>
</gazebo>

<!-- Body - low friction to prevent sticking -->
<gazebo reference="torso">
  <mu1>0.3</mu1>
  <mu2>0.3</mu2>
  <kp>100000.0</kp>
  <kd>10.0</kd>
</gazebo>
```

## Collision Detection and Configuration

Collision detection is essential for humanoid robots to prevent self-collisions and environmental collisions.

### 1. Collision Geometries

Choose appropriate collision geometries for different robot parts:

```xml
<!-- Simple shapes for computational efficiency -->
<link name="torso">
  <collision>
    <geometry>
      <box size="0.2 0.2 0.5"/>  <!-- Simple box for torso -->
    </geometry>
  </collision>
</link>

<link name="upper_arm">
  <collision>
    <geometry>
      <capsule radius="0.05" length="0.25"/>  <!-- Capsule for limbs -->
    </geometry>
  </collision>
</link>

<link name="head">
  <collision>
    <geometry>
      <sphere radius="0.1"/>  <!-- Sphere for head -->
    </geometry>
  </collision>
</link>
```

### 2. Self-Collision Avoidance

Configure collision properties to handle self-collisions:

```xml
<!-- Disable collisions between adjacent links to prevent sticking -->
<gazebo reference="left_hip_joint">
  <disable_collisions_between_children>false</disable_collisions_between_children>
</gazebo>

<!-- For specific pairs of links that shouldn't collide -->
<gazebo reference="left_upper_arm">
  <self_collide>false</self_collide>
</gazebo>
```

### 3. Contact Properties

Fine-tune contact behavior for realistic interactions:

```xml
<gazebo reference="foot_link">
  <surface>
    <friction>
      <ode>
        <mu>0.8</mu>
        <mu2>0.8</mu2>
        <fdir1>0 0 1</fdir1>
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.1</restitution_coefficient>  <!-- Low bounce for feet -->
      <threshold>100000</threshold>
    </bounce>
    <contact>
      <ode>
        <soft_cfm>0.0001</soft_cfm>
        <soft_erp>0.2</soft_erp>
        <kp>1000000000000.0</kp>  <!-- Penetration stiffness -->
        <kd>1000000000000.0</kd>  <!-- Damping coefficient -->
        <max_vel>100.0</max_vel>
        <min_depth>0.001</min_depth>  <!-- Minimum penetration depth -->
      </ode>
    </contact>
  </surface>
</gazebo>
```

## Inertial Properties

Inertial properties determine how the robot responds to forces and torques. These are critical for realistic humanoid simulation.

### 1. Understanding Inertial Parameters

For each link, you need to specify:
- **Mass**: The mass of the link
- **Center of Mass**: Location of the center of mass relative to the link frame
- **Inertia Matrix**: How mass is distributed around the center of mass

```xml
<link name="thigh">
  <inertial>
    <mass value="3.0"/>  <!-- 3 kg -->
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>  <!-- COM offset -->
    <inertia
      ixx="0.05" ixy="0.0" ixz="0.0"
      iyy="0.05" iyz="0.0"
      izz="0.01"/>  <!-- Inertia matrix values -->
  </inertial>
</link>
```

### 2. Calculating Inertial Properties

For common shapes, use standard formulas:

#### Cylinder (for limbs):
```xml
<!-- For a cylinder of mass m, radius r, height h -->
<!-- Ixx = Iyy = m*(3*r² + h²)/12, Izz = m*r²/2 -->
<inertial>
  <mass value="2.0"/>
  <inertia
    ixx="0.03125" ixy="0.0" ixz="0.0"
    iyy="0.03125" iyz="0.0"
    izz="0.025"/>  <!-- For r=0.05, h=0.3, m=2.0 -->
</inertial>
```

#### Box (for torso):
```xml
<!-- For a box of mass m, dimensions x, y, z -->
<!-- Ixx = m*(y² + z²)/12, Iyy = m*(x² + z²)/12, Izz = m*(x² + y²)/12 -->
<inertial>
  <mass value="8.0"/>
  <inertia
    ixx="0.1133" ixy="0.0" ixz="0.0"
    iyy="0.1067" iyz="0.0"
    izz="0.0267"/>  <!-- For 0.2×0.25×0.4 m box, m=8.0 -->
</inertial>
```

#### Sphere (for head):
```xml
<!-- For a sphere of mass m, radius r -->
<!-- Ixx = Iyy = Izz = 2*m*r²/5 -->
<inertial>
  <mass value="2.0"/>
  <inertia
    ixx="0.016" ixy="0.0" ixz="0.0"
    iyy="0.016" iyz="0.0"
    izz="0.016"/>  <!-- For r=0.1, m=2.0 -->
</inertial>
```

### 3. Realistic Humanoid Inertial Values

For a realistic humanoid robot, consider these approximate values:

```xml
<!-- Pelvis (hips) -->
<link name="pelvis">
  <inertial>
    <mass value="10.0"/>
    <inertia
      ixx="0.15" ixy="0.0" ixz="0.0"
      iyy="0.15" iyz="0.0"
      izz="0.2"/>  <!-- Larger Izz for rotational stability -->
  </inertial>
</link>

<!-- Thigh (upper leg) -->
<link name="left_thigh">
  <inertial>
    <mass value="5.0"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <inertia
      ixx="0.1" ixy="0.0" ixz="0.0"
      iyy="0.1" iyz="0.0"
      izz="0.015"/>
  </inertial>
</link>

<!-- Shank (lower leg) -->
<link name="left_shank">
  <inertial>
    <mass value="3.0"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <inertia
      ixx="0.05" ixy="0.0" ixz="0.0"
      iyy="0.05" iyz="0.0"
      izz="0.01"/>
  </inertial>
</link>

<!-- Foot -->
<link name="left_foot">
  <inertial>
    <mass value="1.5"/>
    <origin xyz="0.05 0 -0.05" rpy="0 0 0"/>
    <inertia
      ixx="0.005" ixy="0.0" ixz="0.0"
      iyy="0.008" iyz="0.0"
      izz="0.005"/>
  </inertial>
</link>

<!-- Torso -->
<link name="torso">
  <inertial>
    <mass value="15.0"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <inertia
      ixx="0.5" ixy="0.0" ixz="0.0"
      iyy="0.5" iyz="0.0"
      izz="0.8"/>
  </inertial>
</link>
```

## Physics Tuning Strategies

### 1. Stability vs. Performance Trade-offs

When tuning physics for humanoid robots, you need to balance stability and performance:

```xml
<!-- For high stability (slower but more accurate) -->
<physics type="ode">
  <max_step_size>0.0005</max_step_size>  <!-- Smaller steps -->
  <real_time_update_rate>2000</real_time_update_rate>  <!-- Higher update rate -->
  <ode>
    <solver>
      <iters>50</iters>  <!-- More iterations -->
      <sor>1.0</sor>     <!-- Conservative relaxation -->
    </solver>
  </ode>
</physics>

<!-- For better performance (faster but potentially less stable) -->
<physics type="ode">
  <max_step_size>0.002</max_step_size>  <!-- Larger steps -->
  <real_time_update_rate>500</real_time_update_rate>  <!-- Lower update rate -->
  <ode>
    <solver>
      <iters>10</iters>  <!-- Fewer iterations -->
      <sor>1.3</sor>     <!-- Aggressive relaxation -->
    </solver>
  </ode>
</physics>
```

### 2. Humanoid-Specific Tuning

Humanoid robots have specific requirements that need special attention:

#### Balance and Walking
```xml
<!-- For feet links (critical for balance) -->
<gazebo reference="left_foot">
  <mu1>0.9</mu1>        <!-- High friction for grip -->
  <mu2>0.9</mu2>
  <kp>1e+13</kp>        <!-- Very high stiffness for solid contact -->
  <kd>1000</kd>         <!-- Adequate damping -->
  <max_vel>100.0</max_vel>
  <min_depth>0.0001</min_depth>  <!-- Minimal penetration -->
</gazebo>
```

#### Joint Compliance
```xml
<!-- For joints that need to be compliant during contact -->
<joint name="left_ankle_joint" type="revolute">
  <parent link="left_shank"/>
  <child link="left_foot"/>
  <origin xyz="0 0 -0.25" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Ankle pitch axis -->
  <limit lower="-0.5" upper="0.5" effort="100" velocity="2"/>
  <dynamics damping="1.0" friction="0.1"/>  <!-- More damping for compliance -->
</joint>
```

## Advanced Physics Configuration

### 1. Multi-Material Contacts

For more realistic contact behavior with different materials:

```xml
<!-- Define materials with specific properties -->
<gazebo reference="rubber_foot">
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>
        <mu2>0.8</mu2>
      </ode>
    </friction>
    <contact>
      <ode>
        <kp>1e+14</kp>  <!-- Very stiff rubber -->
        <kd>1000</kd>
      </ode>
    </contact>
  </surface>
</gazebo>

<gazebo reference="metallic_body">
  <surface>
    <friction>
      <ode>
        <mu>0.5</mu>
        <mu2>0.5</mu2>
      </ode>
    </friction>
    <contact>
      <ode>
        <kp>1e+15</kp>  <!-- Very stiff metal -->
        <kd>5000</kd>
      </ode>
    </contact>
  </surface>
</gazebo>
```

### 2. Environmental Physics

Configure environmental physics for realistic simulation:

```xml
<!-- Wind effects (for outdoor scenarios) -->
<world name="outdoor_humanoid">
  <gravity>0 0 -9.8</gravity>
  <model name="wind_generator">
    <static>true</static>
    <link name="wind_link">
      <inertial>
        <mass value="0.001"/>
        <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
      </inertial>
    </link>
    <gazebo>
      <plugin name="wind_plugin" filename="libgazebo_wind_plugin.so">
        <wind_direction>1 0 0</wind_direction>
        <wind_force>0.1 0.05 0</wind_force>
        <wind_velocity>2.0</wind_velocity>
      </plugin>
    </gazebo>
  </model>
</world>
```

### 3. Soft Contacts for Humanoid Safety

For more realistic soft contact that mimics human flexibility:

```xml
<!-- Soft contacts for safer humanoid interaction -->
<gazebo reference="humanoid_body">
  <surface>
    <contact>
      <ode>
        <soft_cfm>0.001</soft_cfm>  <!-- Soft constraint force mixing -->
        <soft_erp>0.1</soft_erp>    <!-- Soft error reduction parameter -->
        <kp>1e+10</kp>              <!-- Moderate stiffness -->
        <kd>100</kd>                <!-- Moderate damping -->
        <max_vel>10.0</max_vel>      <!-- Limit max velocity correction -->
        <min_depth>0.005</min_depth> <!-- Slightly softer contact -->
      </ode>
    </contact>
  </surface>
</gazebo>
```

## Debugging Physics Issues

### 1. Common Physics Problems and Solutions

#### Robot Falling Through Floor
- Check collision geometry in URDF
- Verify mass and inertial properties are positive
- Ensure `<static>false</static>` for dynamic models

#### Unstable Joint Behavior
- Increase solver iterations
- Decrease time step size
- Check joint limits and dynamics parameters
- Verify inertial properties are realistic

#### Excessive Bouncing
- Increase damping in joint dynamics
- Adjust ERP (Error Reduction Parameter) and CFM (Constraint Force Mixing)
- Lower restitution coefficients

#### Sliding/Jittering
- Increase friction coefficients
- Improve contact stiffness (kp) and damping (kd)
- Check for proper mass distribution

### 2. Physics Debugging Tools

#### Gazebo GUI Debugging
```bash
# Launch Gazebo with physics debugging enabled
gz sim -r -v 4 --gui-config ~/.gazebo/gui.config.world
```

#### Checking Contact Forces
```python
# Python script to monitor contact forces
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ContactsState

class ContactMonitor(Node):
    def __init__(self):
        super().__init__('contact_monitor')
        self.subscription = self.create_subscription(
            ContactsState,
            '/gazebo/contact',
            self.contact_callback,
            10
        )

    def contact_callback(self, msg):
        for contact in msg.states:
            if contact.collision1_name.startswith('humanoid') or \
               contact.collision2_name.startswith('humanoid'):
                self.get_logger().info(f'Contact: {contact.collision1_name} <-> {contact.collision2_name}')
                for wrench in contact.contact_positions:
                    self.get_logger().info(f'Position: {wrench}')

def main(args=None):
    rclpy.init(args=args)
    contact_monitor = ContactMonitor()

    try:
        rclpy.spin(contact_monitor)
    except KeyboardInterrupt:
        pass
    finally:
        contact_monitor.destroy_node()
        rclpy.shutdown()
```

## Performance Optimization

### 1. Physics Performance Tips

#### Simplify Collision Geometries
```xml
<!-- Good: Simple collision shapes -->
<collision>
  <geometry>
    <box size="0.1 0.1 0.1"/>
  </geometry>
</collision>

<!-- Avoid: Complex mesh collisions (unless necessary) -->
<collision>
  <geometry>
    <mesh filename="complex_shape.stl"/>
  </geometry>
</collision>
```

#### Adjust Update Rates
```xml
<!-- Balance update rates based on requirements -->
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_update_rate>1000</real_time_update_rate>  <!-- For control -->
  <!-- Use lower rates for non-critical components -->
</physics>
```

### 2. Computational Efficiency

#### Parallel Processing
```xml
<!-- Use multiple threads for physics simulation -->
<physics type="ode">
  <ode>
    <thread_position_correction>true</thread_position_correction>
  </ode>
</physics>
```

#### Selective Physics Updates
For humanoid robots, not all joints need the same update rate:
- Critical joints (ankles, hips): 1000Hz
- Less critical joints (elbows, wrists): 100Hz

## Physics Validation and Testing

### 1. Validation Procedures

Create tests to validate physics behavior:

```python
# physics_validation_test.py
import unittest
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
import time

class PhysicsValidationTest(Node):
    def __init__(self):
        super().__init__('physics_validation_test')
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.joint_states = None

    def joint_state_callback(self, msg):
        self.joint_states = msg

    def test_standing_stability(self):
        """Test that humanoid remains stable when standing"""
        time.sleep(5)  # Let robot settle

        if self.joint_states is None:
            self.get_logger().error('No joint states received')
            return False

        # Check that joint positions are within reasonable bounds
        for pos in self.joint_states.position:
            if abs(pos) > 1.57:  # Joint not moving excessively
                return False

        return True

def main():
    rclpy.init()
    validator = PhysicsValidationTest()

    # Run validation tests
    time.sleep(10)  # Give time for simulation to settle
    standing_stable = validator.test_standing_stability()

    if standing_stable:
        print("Physics validation PASSED: Robot maintains stable stance")
    else:
        print("Physics validation FAILED: Robot is unstable")

    validator.destroy_node()
    rclpy.shutdown()
```

### 2. Physics Benchmarking

Compare simulation performance with different physics settings:

```bash
# Benchmark script
#!/bin/bash

echo "Testing physics configurations..."

# Test different time steps
for step in 0.001 0.002 0.005; do
  echo "Testing time step: $step"
  # Launch simulation with specific time step
  # Measure stability and performance
done

# Test different solver iterations
for iter in 10 20 50; do
  echo "Testing solver iterations: $iter"
  # Launch simulation with specific iterations
  # Measure stability and performance
done
```

## Best Practices for Humanoid Physics

### 1. Iterative Tuning Process
1. Start with conservative values
2. Test basic stability
3. Gradually increase performance parameters
4. Validate with complex movements
5. Fine-tune for specific behaviors

### 2. Realistic Parameter Ranges
- Mass: Use realistic human proportions (legs heavier than arms)
- Friction: 0.5-0.9 for rubber/plastic on common surfaces
- Inertia: Calculate from actual geometry when possible
- Time step: 0.001s for humanoid locomotion

### 3. Documentation and Versioning
- Document physics parameters with justification
- Keep physics settings in separate config files
- Track performance metrics for different configurations

## Summary

Physics simulation is crucial for creating realistic digital twins of humanoid robots. The key elements for proper physics configuration are:

1. **Gravity**: Properly configured to match the simulation environment
2. **Friction**: Appropriate coefficients for different surfaces and robot parts
3. **Collisions**: Well-designed collision geometries with proper contact properties
4. **Inertial Properties**: Realistic mass distribution and moments of inertia

For humanoid robots specifically, pay special attention to:
- Foot-ground contact properties for stable locomotion
- Proper mass distribution for balance
- Appropriate joint damping for realistic movement
- Collision avoidance between robot parts

Balancing stability and performance requires iterative tuning and validation with real-world scenarios.

## Exercises

1. Create a simple humanoid model with realistic physics properties and test its stability when standing.

2. Experiment with different friction coefficients for the feet and observe the effect on walking stability.

3. Implement a physics validation test that checks if your humanoid robot maintains balance under small perturbations.

4. Compare the simulation performance of your robot with different time step sizes and solver iterations.

5. Add environmental physics effects (wind, different gravity) and observe how your humanoid adapts.

## Next Steps

In the next chapter, we'll explore how to create and configure Gazebo worlds with different environments and scenarios for testing your humanoid robot. We'll learn to build complex environments with obstacles, varying terrains, and interactive elements.

Continue to Chapter 5: [Environment Building in Gazebo (Worlds, Lights, Materials)](./environment-building.md) to learn about creating realistic simulation environments.