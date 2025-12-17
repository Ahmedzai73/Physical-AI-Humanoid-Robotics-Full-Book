---
title: Environment Building in Gazebo (Worlds, Lights, Materials)
sidebar_position: 5
description: Creating realistic simulation environments with worlds, lighting, and materials for humanoid robot testing
---

# Environment Building in Gazebo (Worlds, Lights, Materials)

## Introduction

Creating realistic environments is essential for testing humanoid robots in simulation. Unlike simple wheeled robots, humanoid robots interact with complex 3D environments that include stairs, furniture, narrow passages, and varied terrain. In this chapter, we'll explore how to create comprehensive Gazebo worlds with appropriate lighting, materials, and interactive elements to thoroughly test your humanoid robot.

The environment plays a crucial role in:
- **Locomotion testing**: Validating walking gaits on different surfaces
- **Navigation validation**: Testing path planning and obstacle avoidance
- **Perception evaluation**: Testing sensors in realistic conditions
- **Interaction scenarios**: Validating manipulation and social behaviors
- **Stress testing**: Pushing robot capabilities in challenging scenarios

## Understanding Gazebo Worlds

### World File Structure

A Gazebo world file is an SDF (Simulation Description Format) file that defines the complete simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_test_environment">
    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Your models -->
    <model name="humanoid_robot">
      <pose>0 0 1 0 0 0</pose>
      <!-- Model definition -->
    </model>

    <!-- Additional environment elements -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <!-- Table model -->
    </model>

  </world>
</sdf>
```

### World Components for Humanoid Robots

For humanoid robot testing, consider these essential environment components:

1. **Ground surfaces** with different friction properties
2. **Obstacles** of various shapes and sizes
3. **Terrain variations** (stairs, slopes, uneven ground)
4. **Furniture** for navigation and interaction testing
5. **Lighting** that affects perception systems
6. **Markers** for localization and navigation

## Creating Complex Environments

### 1. Indoor Environments

Indoor environments are common for humanoid robot testing:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="house_environment">
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
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Room walls -->
    <model name="living_room_walls">
      <static>true</static>
      <link name="north_wall">
        <pose>0 5 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia ixx="100" ixy="0" ixz="0" iyy="100" iyz="0" izz="100"/>
        </inertial>
      </link>

      <link name="south_wall">
        <pose>0 -5 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia ixx="100" ixy="0" ixz="0" iyy="100" iyz="0" izz="100"/>
        </inertial>
      </link>

      <link name="east_wall">
        <pose>5 0 1.5 0 0 1.5708</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia ixx="100" ixy="0" ixz="0" iyy="100" iyz="0" izz="100"/>
        </inertial>
      </link>

      <link name="west_wall">
        <pose>-5 0 1.5 0 0 1.5708</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia ixx="100" ixy="0" ixz="0" iyy="100" iyz="0" izz="100"/>
        </inertial>
      </link>
    </model>

    <!-- Furniture -->
    <model name="dining_table">
      <pose>2 0 0.4 0 0 0</pose>
      <link name="table_top">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.2 0.8 0.05</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.2 0.8 0.05</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>20</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>

      <link name="leg_1">
        <pose>0.5 0.3 -0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>

      <link name="leg_2">
        <pose>-0.5 0.3 -0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>

      <link name="leg_3">
        <pose>0.5 -0.3 -0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>

      <link name="leg_4">
        <pose>-0.5 -0.3 -0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>

      <joint name="top_to_leg1" type="fixed">
        <parent>table_top</parent>
        <child>leg_1</child>
        <pose>0.5 0.3 0 0 0 0</pose>
      </joint>

      <joint name="top_to_leg2" type="fixed">
        <parent>table_top</parent>
        <child>leg_2</child>
        <pose>-0.5 0.3 0 0 0 0</pose>
      </joint>

      <joint name="top_to_leg3" type="fixed">
        <parent>table_top</parent>
        <child>leg_3</child>
        <pose>0.5 -0.3 0 0 0 0</pose>
      </joint>

      <joint name="top_to_leg4" type="fixed">
        <parent>table_top</parent>
        <child>leg_4</child>
        <pose>-0.5 -0.3 0 0 0 0</pose>
      </joint>
    </model>

    <!-- Chair -->
    <model name="office_chair">
      <pose>3 1 0 0 0 0</pose>
      <link name="seat">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.4 0.05</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.4 0.05</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.7 1</ambient>
            <diffuse>0.2 0.2 0.7 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>5</mass>
          <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
      </link>

      <link name="back">
        <pose>0 0.2 0.2 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.05 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.05 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.2 0.7 1</ambient>
            <diffuse>0.2 0.2 0.7 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>3</mass>
          <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
      </link>

      <joint name="seat_to_back" type="fixed">
        <parent>seat</parent>
        <child>back</child>
        <pose>0 0.2 0.2 0 0 0</pose>
      </joint>
    </model>

    <!-- Stairs for locomotion testing -->
    <model name="stairs">
      <pose>-2 0 0 0 0 0</pose>
      <static>true</static>
      <link name="step_1">
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>50</mass>
          <inertia ixx="10" ixy="0" ixz="0" iyy="10" iyz="0" izz="10"/>
        </inertial>
      </link>

      <link name="step_2">
        <pose>0 0 0.2 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>50</mass>
          <inertia ixx="10" ixy="0" ixz="0" iyy="10" iyz="0" izz="10"/>
        </inertial>
      </link>

      <link name="step_3">
        <pose>0 0 0.4 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>2 1 0.2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>2 1 0.2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>50</mass>
          <inertia ixx="10" ixy="0" ixz="0" iyy="10" iyz="0" izz="10"/>
        </inertial>
      </link>

      <joint name="step1_to_step2" type="fixed">
        <parent>step_1</parent>
        <child>step_2</child>
        <pose>0 0 0.2 0 0 0</pose>
      </joint>

      <joint name="step2_to_step3" type="fixed">
        <parent>step_2</parent>
        <child>step_3</child>
        <pose>0 0 0.2 0 0 0</pose>
      </joint>
    </model>

    <!-- Door for navigation testing -->
    <model name="doorway">
      <pose>0 3 0 0 0 0</pose>
      <static>true</static>
      <link name="left_post">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>

      <link name="right_post">
        <pose>1.5 0 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.1 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.1 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>

      <link name="top_beam">
        <pose>0.75 0 1.9 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>1.5 0.1 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.5 0.1 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.2 1</ambient>
            <diffuse>0.8 0.6 0.2 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>5</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>
    </model>

    <!-- Add your humanoid robot here -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>

  </world>
</sdf>
```

### 2. Outdoor Environments

Outdoor environments add additional challenges for humanoid robots:

```xml
<!-- worlds/outdoor_park.world -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="outdoor_park">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Sun with realistic outdoor lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground with realistic outdoor texture -->
    <model name="grass_ground">
      <static>true</static>
      <link name="ground_link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Grass</name>
            </script>
          </material>
        </visual>
        <inertial>
          <mass>1000</mass>
          <inertia ixx="1000" ixy="0" ixz="0" iyy="1000" iyz="0" izz="1000"/>
        </inertial>
      </link>
    </model>

    <!-- Trees for navigation challenges -->
    <model name="tree_1">
      <pose>5 5 0 0 0 0</pose>
      <static>true</static>
      <link name="trunk">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>4</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.2</radius>
              <length>4</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0.4 0.2 0.1 1</ambient>
            <diffuse>0.4 0.2 0.1 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>20</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>
      <link name="leaves">
        <pose>0 0 3.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>1.5</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>1.5</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>0.1 0.6 0.1 1</ambient>
            <diffuse>0.1 0.6 0.1 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>5</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>
      <joint name="trunk_to_leaves" type="fixed">
        <parent>trunk</parent>
        <child>leaves</child>
        <pose>0 0 2 0 0 0</pose>
      </joint>
    </model>

    <!-- Bench for interaction testing -->
    <model name="park_bench">
      <pose>-3 -2 0 0 0 0</pose>
      <static>true</static>
      <link name="seat">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.5 0.4 0.05</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.5 0.4 0.05</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>10</mass>
          <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
      </link>
      <link name="leg_1">
        <pose>0.6 0.15 -0.2 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
      </link>
      <link name="leg_2">
        <pose>-0.6 0.15 -0.2 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
      </link>
      <link name="leg_3">
        <pose>0.6 -0.15 -0.2 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
      </link>
      <link name="leg_4">
        <pose>-0.6 -0.15 -0.2 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.05 0.05 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.05 0.05 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
            <specular>0.2 0.2 0.2 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>2</mass>
          <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
        </inertial>
      </link>
      <joint name="seat_to_leg1" type="fixed">
        <parent>seat</parent>
        <child>leg_1</child>
        <pose>0.6 0.15 -0.25 0 0 0</pose>
      </joint>
      <joint name="seat_to_leg2" type="fixed">
        <parent>seat</parent>
        <child>leg_2</child>
        <pose>-0.6 0.15 -0.25 0 0 0</pose>
      </joint>
      <joint name="seat_to_leg3" type="fixed">
        <parent>seat</parent>
        <child>leg_3</child>
        <pose>0.6 -0.15 -0.25 0 0 0</pose>
      </joint>
      <joint name="seat_to_leg4" type="fixed">
        <parent>seat</parent>
        <child>leg_4</child>
        <pose>-0.6 -0.15 -0.25 0 0 0</pose>
      </joint>
    </model>

    <!-- Sloped terrain for locomotion challenges -->
    <model name="slope">
      <pose>0 5 0 0 0 0.3</pose>
      <static>true</static>
      <link name="slope_link">
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>file://meshes/slope.dae</uri>
            </mesh>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>file://meshes/slope.dae</uri>
            </mesh>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
        <inertial>
          <mass>100</mass>
          <inertia ixx="10" ixy="0" ixz="0" iyy="10" iyz="0" izz="10"/>
        </inertial>
      </link>
    </model>

    <!-- Humanoid robot spawn point -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>

  </world>
</sdf>
```

## Lighting and Materials for Realistic Simulation

### 1. Lighting Configuration

Proper lighting is crucial for perception-based humanoid robots:

```xml
<!-- Directional light (sun) -->
<light name="main_sun" type="directional">
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <attenuation>
    <range>1000</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <direction>-0.5 0.1 -0.9</direction>
  <cast_shadows>true</cast_shadows>
</light>

<!-- Point lights for indoor environments -->
<light name="ceiling_light" type="point">
  <pose>0 0 3 0 0 0</pose>
  <diffuse>1.0 1.0 0.9 1</diffuse>
  <specular>0.5 0.5 0.5 1</specular>
  <attenuation>
    <range>10</range>
    <constant>0.2</constant>
    <linear>0.5</linear>
    <quadratic>0.05</quadratic>
  </attenuation>
  <cast_shadows>true</cast_shadows>
</light>

<!-- Spot lights for focused illumination -->
<light name="spotlight" type="spot">
  <pose>5 5 3 -0.5 -0.5 -1.0</pose>
  <diffuse>1.0 0.9 0.8 1</diffuse>
  <specular>0.5 0.5 0.5 1</specular>
  <attenuation>
    <range>15</range>
    <constant>0.2</constant>
    <linear>0.3</linear>
    <quadratic>0.01</quadratic>
  </attenuation>
  <direction>-0.5 -0.5 -1.0</direction>
  <spot>
    <inner_angle>0.1</inner_angle>
    <outer_angle>0.5</outer_angle>
    <falloff>1.0</falloff>
  </spot>
  <cast_shadows>true</cast_shadows>
</light>
```

### 2. Material Definitions

Create realistic materials for different surfaces:

```xml
<!-- In a separate materials.sdf file or within the world file -->
<world name="humanoid_world">
  <!-- Define custom materials -->
  <gazebo reference="floor_material">
    <material>Gazebo/White</material>
  </gazebo>

  <!-- Floor with specific friction for humanoid locomotion -->
  <model name="floor_with_friction">
    <static>true</static>
    <link name="floor_link">
      <collision name="collision">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>10 10</size>
          </plane>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>10 10</size>
          </plane>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
      <inertial>
        <mass>1000</mass>
        <inertia ixx="1000" ixy="0" ixz="0" iyy="1000" iyz="0" izz="1000"/>
      </inertial>
    </link>
    <gazebo reference="floor_link">
      <mu1>0.8</mu1>
      <mu2>0.8</mu2>
      <kp>1000000.0</kp>
      <kd>100.0</kd>
    </gazebo>
  </model>
</world>
```

## Advanced Environment Features

### 1. Dynamic Environments

Create environments with moving elements for advanced testing:

```xml
<!-- Moving platform for testing dynamic balance -->
<model name="moving_platform">
  <link name="platform">
    <collision name="collision">
      <geometry>
        <box>
          <size>2 2 0.1</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>2 2 0.1</size>
        </box>
      </geometry>
      <material>
        <ambient>0.7 0.7 0.7 1</ambient>
        <diffuse>0.7 0.7 0.7 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>50</mass>
      <inertia ixx="10" ixy="0" ixz="0" iyy="10" iyz="0" izz="10"/>
    </inertial>
  </link>

  <!-- Plugin to make platform move in a circle -->
  <gazebo>
    <plugin name="sine_motion" filename="libgazebo_ros_pubslish_odom.so">
      <ros>
        <namespace>/moving_platform</namespace>
      </ros>
      <update_rate>100</update_rate>
      <x>
        <amplitude>2.0</amplitude>
        <frequency>0.2</frequency>
        <offset>0.0</offset>
      </x>
      <y>
        <amplitude>1.5</amplitude>
        <frequency>0.15</frequency>
        <offset>0.0</offset>
      </y>
    </plugin>
  </gazebo>
</model>
```

### 2. Sensor Testing Environments

Create specialized environments for sensor validation:

```xml
<!-- Calibration checkerboard for camera calibration -->
<model name="checkerboard">
  <pose>2 0 1 0 0 0</pose>
  <static>true</static>
  <link name="checkerboard_link">
    <collision name="collision">
      <geometry>
        <box>
          <size>0.5 0.5 0.01</size>
        </box>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <box>
          <size>0.5 0.5 0.01</size>
        </box>
      </geometry>
      <material>
        <script>
          <uri>file://media/materials/scripts/gazebo.material</uri>
          <name>Gazebo/CheckerBlue</name>
        </script>
      </material>
    </visual>
    <inertial>
      <mass>1</mass>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</model>

<!-- LiDAR testing environment with various shapes -->
<model name="lidar_test_shapes">
  <pose>-2 0 1 0 0 0</pose>
  <static>true</static>
  <link name="sphere_shape">
    <pose>0 0 0.5 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <sphere>
          <radius>0.2</radius>
        </sphere>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <sphere>
          <radius>0.2</radius>
        </sphere>
      </geometry>
      <material>
        <ambient>0.8 0.2 0.2 1</ambient>
        <diffuse>0.8 0.2 0.2 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>1</mass>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <link name="cylinder_shape">
    <pose>0 0.5 0.5 0 0 0</pose>
    <collision name="collision">
      <geometry>
        <cylinder>
          <radius>0.15</radius>
          <length>0.4</length>
        </cylinder>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <cylinder>
          <radius>0.15</radius>
          <length>0.4</length>
        </cylinder>
      </geometry>
      <material>
        <ambient>0.2 0.8 0.2 1</ambient>
        <diffuse>0.2 0.8 0.2 1</diffuse>
      </material>
    </visual>
    <inertial>
      <mass>1</mass>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</model>
```

## Environment Testing and Validation

### 1. Testing Framework

Create tests to validate your environments:

```python
# env_validation_test.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan, Image
import numpy as np

class EnvironmentValidator(Node):
    def __init__(self):
        super().__init__('environment_validator')

        # Subscribers for different sensor data
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        # Publisher for navigation goals
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.environment_tests = {
            'obstacle_detection': self.test_obstacle_detection,
            'navigation_space': self.test_navigation_space,
            'lighting_conditions': self.test_lighting_conditions
        }

        self.get_logger().info('Environment Validator initialized')

    def scan_callback(self, msg):
        """Process laser scan data"""
        # Store scan data for testing
        self.last_scan = msg

    def camera_callback(self, msg):
        """Process camera image data"""
        # Store image data for testing
        self.last_image = msg

    def map_callback(self, msg):
        """Process occupancy grid map"""
        # Store map data for testing
        self.last_map = msg

    def test_obstacle_detection(self):
        """Test that obstacles are properly detected in the environment"""
        if not hasattr(self, 'last_scan'):
            self.get_logger().warn('No scan data available')
            return False

        # Check that obstacles are detected in expected locations
        # Example: Check for obstacles in doorway area
        door_scan_ranges = self.last_scan.ranges[300:420]  # Specific angle range
        min_distance = min(door_scan_ranges) if door_scan_ranges else float('inf')

        if min_distance < 1.0:  # Obstacle detected
            self.get_logger().info('Obstacle detection test PASSED')
            return True
        else:
            self.get_logger().warn('Obstacle detection test FAILED')
            return False

    def test_navigation_space(self):
        """Test that navigation space is properly mapped"""
        if not hasattr(self, 'last_map'):
            self.get_logger().warn('No map data available')
            return False

        # Convert map to numpy array for analysis
        map_data = np.array(self.last_map.data).reshape(
            self.last_map.info.height, self.last_map.info.width
        )

        # Check for navigable space vs obstacles
        free_space_ratio = np.sum(map_data == 0) / map_data.size

        if free_space_ratio > 0.5:  # At least 50% free space
            self.get_logger().info('Navigation space test PASSED')
            return True
        else:
            self.get_logger().warn('Navigation space test FAILED')
            return False

    def test_lighting_conditions(self):
        """Test lighting conditions affect perception appropriately"""
        if not hasattr(self, 'last_image'):
            self.get_logger().warn('No image data available')
            return False

        # This is a simplified test - in reality, you'd check for lighting effects
        # like shadows, brightness levels, etc.
        avg_brightness = np.mean(np.array(self.last_image.data))

        if 0 < avg_brightness < 255:  # Reasonable brightness range
            self.get_logger().info('Lighting conditions test PASSED')
            return True
        else:
            self.get_logger().warn('Lighting conditions test FAILED')
            return False

    def run_all_tests(self):
        """Run all environment validation tests"""
        results = {}
        for test_name, test_func in self.environment_tests.items():
            self.get_logger().info(f'Running {test_name} test...')
            results[test_name] = test_func()

        # Report results
        passed = sum(results.values())
        total = len(results)
        self.get_logger().info(f'Environment validation: {passed}/{total} tests passed')

        for test_name, result in results.items():
            status = 'PASSED' if result else 'FAILED'
            self.get_logger().info(f'  {test_name}: {status}')

        return results

def main(args=None):
    rclpy.init(args=args)
    validator = EnvironmentValidator()

    # Run validation after allowing time for data collection
    import time
    time.sleep(5)  # Allow time for sensor data collection

    results = validator.run_all_tests()

    validator.destroy_node()
    rclpy.shutdown()

    return all(results.values())  # Return True if all tests passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
```

### 2. Performance Testing

Evaluate environment performance:

```python
# env_performance_test.py
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetPhysicsProperties
import time

class EnvironmentPerformanceTester(Node):
    def __init__(self):
        super().__init__('env_performance_tester')

        # Client to get physics properties
        self.physics_client = self.create_client(GetPhysicsProperties, '/gazebo/get_physics_properties')

        # Wait for service
        while not self.physics_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Physics service not available, waiting again...')

        self.get_logger().info('Environment Performance Tester initialized')

    def measure_physics_stability(self, duration=30):
        """Measure physics simulation stability over time"""
        start_time = time.time()
        poses_over_time = []

        while time.time() - start_time < duration:
            # Get physics properties to check for stability
            future = self.physics_client.call_async(GetPhysicsProperties.Request())

            # In a real test, you'd also measure robot pose stability
            time.sleep(0.1)  # Small delay to avoid overwhelming the service

        self.get_logger().info(f'Physics stability measured over {duration} seconds')
        return True

def main(args=None):
    rclpy.init(args=args)
    tester = EnvironmentPerformanceTester()

    try:
        success = tester.measure_physics_stability(duration=10)
        tester.get_logger().info(f'Performance test result: {"PASSED" if success else "FAILED"}')
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()
```

## Best Practices for Environment Design

### 1. Scalability and Modularity

Design environments that can be easily extended and modified:

```xml
<!-- Use includes to modularize environments -->
<world name="modular_house">
  <include>
    <uri>model://rooms/living_room</uri>
    <pose>0 0 0 0 0 0</pose>
  </include>

  <include>
    <uri>model://rooms/kitchen</uri>
    <pose>5 0 0 0 0 0</pose>
  </include>

  <include>
    <uri>model://rooms/bedroom</uri>
    <pose>-5 0 0 0 0 0</pose>
  </include>
</world>
```

### 2. Realistic Physics Properties

Use realistic friction, damping, and material properties:

```xml
<!-- Different surfaces with appropriate properties -->
<gazebo reference="wood_floor">
  <mu1>0.6</mu1>
  <mu2>0.6</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>

<gazebo reference="carpet">
  <mu1>0.8</mu1>
  <mu2>0.8</mu2>
  <kp>500000.0</kp>
  <kd>50.0</kd>
</gazebo>

<gazebo reference="tile">
  <mu1>0.4</mu1>
  <mu2>0.4</mu2>
  <kp>2000000.0</kp>
  <kd>200.0</kd>
</gazebo>
```

### 3. Performance Optimization

Optimize environments for simulation performance:

```xml
<!-- Use simple collision geometries where possible -->
<collision>
  <geometry>
    <box size="0.2 0.2 0.2"/>  <!-- Simple box instead of complex mesh -->
  </geometry>
</collision>

<!-- Reduce visual complexity for distant objects -->
<visual name="distant_building">
  <geometry>
    <box size="10 10 20"/>  <!-- Simple representation -->
  </geometry>
  <material>
    <ambient>0.5 0.5 0.5 1</ambient>
    <diffuse>0.5 0.5 0.5 1</diffuse>
  </material>
</visual>
```

## Troubleshooting Common Environment Issues

### 1. Robot Falling Through Floors
- Check collision geometry in URDF/models
- Verify static property for ground models
- Ensure proper mass and inertial properties

### 2. Poor Performance
- Simplify collision geometries
- Reduce polygon count in visual meshes
- Limit number of complex physics interactions

### 3. Lighting Artifacts
- Check light direction and intensity
- Verify material properties
- Adjust shadow settings for performance

### 4. Sensor Inaccuracies
- Validate sensor placement and parameters
- Check lighting conditions affecting cameras
- Verify noise models match real sensors

## Creating Environment Variants

For comprehensive testing, create multiple environment variants:

```bash
# Create environment variants
mkdir -p worlds/variants
cp worlds/house_environment.world worlds/variants/house_daytime.world
cp worlds/house_environment.world worlds/variants/house_nighttime.world
cp worlds/house_environment.world worlds/variants/house_low_light.world
cp worlds/house_environment.world worlds/variants/house_high_contrast.world
```

Then modify each variant for specific testing scenarios:
- Daytime: Bright lighting, good visibility
- Nighttime: Low lighting, challenging perception
- Low light: Minimal lighting to test perception limits
- High contrast: Strong shadows and highlights

## Integration with Simulation Workflows

Connect your environments to the broader simulation workflow:

```python
# env_integration.py
import subprocess
import os
from pathlib import Path

class EnvironmentManager:
    def __init__(self, workspace_path):
        self.workspace_path = Path(workspace_path)
        self.environments_path = self.workspace_path / "worlds"

    def validate_environment(self, world_file):
        """Validate that a world file is syntactically correct"""
        try:
            result = subprocess.run([
                "gz", "sdf", "-k", str(world_file)
            ], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def list_available_environments(self):
        """List all available world files"""
        world_files = list(self.environments_path.glob("*.world"))
        return [wf.name for wf in world_files]

    def benchmark_environment(self, world_file):
        """Run a quick benchmark of the environment"""
        # This would run the environment for a short time and measure performance
        pass

# Usage example
env_manager = EnvironmentManager("~/ros2_ws/src/humanoid_simulation/")
available_worlds = env_manager.list_available_environments()
print(f"Available environments: {available_worlds}")
```

## Summary

Creating realistic simulation environments is crucial for humanoid robot development. Good environments should:

1. **Match real-world scenarios**: Include relevant obstacles, furniture, and terrain
2. **Support comprehensive testing**: Provide various challenges for different robot capabilities
3. **Be computationally efficient**: Balance realism with simulation performance
4. **Include proper physics**: Use realistic friction, collision, and material properties
5. **Enable sensor validation**: Provide appropriate lighting and textures for perception testing
6. **Be modular and extensible**: Allow for easy modification and extension

For humanoid robots specifically, environments should include:
- Varied terrain (stairs, slopes, uneven ground)
- Navigation challenges (doors, narrow passages)
- Interaction opportunities (furniture, objects to manipulate)
- Different lighting conditions for perception testing
- Realistic physics properties for stable locomotion

## Exercises

1. Create a world file with a complex indoor environment including multiple rooms, furniture, and navigation challenges.

2. Design a testing environment specifically for humanoid locomotion with stairs, ramps, and different surface types.

3. Implement a dynamic environment with moving platforms or objects to test robot adaptation capabilities.

4. Create an outdoor environment with natural obstacles like trees, benches, and varied terrain.

5. Develop a validation framework that automatically tests environment properties like obstacle detection and navigation space availability.

## Next Steps

In the next chapter, we'll explore sensor simulation in Gazebo, learning how to configure realistic sensors for your humanoid robot that provide data equivalent to real-world sensors. We'll cover LiDAR, cameras, IMUs, and other perception systems that are essential for autonomous humanoid operation.

Continue to Chapter 6: [Simulating Sensors: LiDAR, Depth Camera, IMU, RGB Cameras](./sensor-simulation.md) to learn about creating realistic sensor systems for your digital twin.