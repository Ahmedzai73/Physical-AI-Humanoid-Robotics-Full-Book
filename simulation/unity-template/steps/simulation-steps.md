# Unity HDRP Environment Simulation Steps

This guide provides step-by-step instructions for creating high-fidelity environments and interaction scenes in Unity HDRP as covered in Module 2 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to create realistic human-robot interaction scenarios using Unity's High Definition Render Pipeline (HDRP), with focus on creating photorealistic environments for robotics applications.

## Prerequisites

- Unity 2022.3 LTS or later installed
- Completed Module 1 and Module 2 Gazebo simulation exercises
- Basic understanding of Unity interface and HDRP
- Adequate GPU for HDRP rendering

## Simulation Environment Setup

1. Open Unity Hub and create a new project
2. Select "HDRP" template or create a 3D (Built-in Render Pipeline) project and convert to HDRP
3. Import the Unity HDRP template created earlier in the textbook project

## Exercise 1: Setting Up HDRP Pipeline

1. Create a new Unity project or open the existing template:
   - File → New Project
   - Select "3D (HDRP)" template or "3D (Built-in Render Pipeline)" if HDRP is not available by default

2. If starting with Built-in Render Pipeline, convert to HDRP:
   - Go to Edit → Render Pipeline → Convert Project to URP/HDRP
   - Follow the conversion wizard to switch to HDRP

3. Verify HDRP settings:
   - In Project Settings → Graphics, check that HDRP Asset is assigned
   - Verify that the scene uses HDRP settings

## Exercise 2: Creating Basic Environment

1. Create a basic environment scene:
   - Create a ground plane (GameObject → 3D Object → Plane)
   - Scale it to 10x10 units (Scale: 10, 1, 10)
   - Apply a material with realistic ground texture

2. Add basic lighting:
   - Create a Directional Light (GameObject → Light → Directional Light)
   - Adjust rotation to simulate sun (e.g., Rotation: 50, -30, 0)
   - Set Light Type to "Realtime" for dynamic shadows

3. Configure HDRP lighting settings:
   - In the Directional Light component, expand "HDRP Additional Settings"
   - Enable "Support Light Layers" and configure as needed
   - Adjust Intensity and Color Temperature for realistic lighting

## Exercise 3: Adding HDRP Volumes

1. Create a Volume for global effects:
   - GameObject → Volume → Global Volume
   - Add Volume components for visual effects

2. Add common HDRP effects:
   - In the Global Volume, click "Add Override"
   - Add effects like Bloom, Chromatic Aberration, Color Adjustments
   - Adjust settings for realistic appearance

3. Create local volumes for specific areas:
   - Add local volumes with specific lighting conditions
   - Use masks to limit the effect area

## Exercise 4: Creating Robot Model in Unity

1. Import robot model:
   - If you have a 3D model, import it via Assets → Import New Asset
   - If creating from primitives, build the robot using basic shapes
   - Create a hierarchy with appropriate joints (empty GameObjects as joints)

2. Set up robot components:
   - Create parent GameObject for the robot
   - Add child objects for each body part (base, torso, arms, legs)
   - Use appropriate primitive shapes or imported models

3. Configure robot materials:
   - Create materials with metallic/roughness properties
   - Apply to robot parts for realistic appearance
   - Use HDRP-specific material properties for better rendering

## Exercise 5: Implementing HDRP Materials

1. Create realistic materials:
   - Create new Material (Assets → Create → Material)
   - Set Material Type to "Lit"
   - Configure HDRP Lit properties (Albedo, Normal Map, Metallic, Smoothness)

2. Apply advanced material features:
   - Use HDRP-specific features like Subsurface Scattering for skin
   - Apply Clearcoat for shiny surfaces
   - Use Anisotropy for brushed metal effects
   - Add Iridescence for special effects

3. Set up material properties for robot:
   - Metallic surfaces for mechanical parts
   - Plastic/rubber materials for joints
   - Glass materials for sensors/cameras

## Exercise 6: Adding Realistic Lighting

1. Set up multiple light sources:
   - Directional Light for main illumination (sun/sky)
   - Point Lights for local illumination
   - Area Lights for soft lighting effects

2. Configure Light Unit for physical accuracy:
   - Set Directional Light to "Lux" with ~100,000 lux for sunlight
   - Set indoor lights to appropriate lumens
   - Use IES profiles for realistic light distribution

3. Add Volumetric Lighting:
   - Enable volumetric fog in the Volume
   - Configure Scattering and Extinction properties
   - Add fog effects for atmospheric appearance

## Exercise 7: Creating Interactive Environment

1. Add environmental objects:
   - Create furniture, obstacles, or scene elements
   - Use realistic models or primitive shapes
   - Apply appropriate HDRP materials

2. Implement interactivity:
   - Add colliders to objects
   - Create simple scripts for interaction
   - Add UI elements to display information

3. Set up realistic surfaces:
   - Add textures with normal maps for detail
   - Configure surface properties for physical accuracy
   - Use HDRP's surface gradient for complex materials

## Exercise 8: Implementing Reflection Probes

1. Add Reflection Probes for realistic reflections:
   - GameObject → Light → Reflection Probe
   - Set to "Realtime" for dynamic reflections
   - Position strategically around the scene

2. Configure probe settings:
   - Adjust resolution for quality/performance balance
   - Set up probe blending for smooth transitions
   - Use "Box Projection" for better reflection quality

3. Optimize probe placement:
   - Place probes where the robot will be
   - Avoid overlapping probe areas where possible
   - Test reflection quality and performance

## Exercise 9: Adding Post-Processing Effects

1. Add post-processing effects to the camera:
   - Select Main Camera
   - Add "Post-process Layer" component
   - Add "Post-process Volume" to the scene

2. Configure post-processing effects:
   - Add effects like Bloom, Color Grading, Depth of Field
   - Adjust intensity for realistic but visually appealing results
   - Balance effects for performance

3. Set up camera-specific effects:
   - Configure Depth of Field for focus effects
   - Add Motion Blur for realistic movement
   - Implement Screen Space Reflections for shiny surfaces

## Exercise 10: Creating Human-Robot Interaction Scenarios

1. Add human avatars or characters:
   - Import or create simple human models
   - Set up basic animations or poses
   - Position for interaction scenarios

2. Implement interaction points:
   - Create designated areas for interaction
   - Add visual indicators for robot attention points
   - Set up simple interaction scripts

3. Design interaction scenarios:
   - Create scenarios for navigation around humans
   - Set up pick-and-place tasks
   - Design social interaction areas

## Exercise 11: Optimizing for Performance

1. Configure rendering settings:
   - Adjust Quality Settings for target hardware
   - Configure LOD groups for complex objects
   - Set up occlusion culling for large scenes

2. Optimize materials:
   - Use texture atlasing where possible
   - Reduce shader complexity for distant objects
   - Implement Level of Detail for materials

3. Test performance:
   - Monitor frame rate and GPU usage
   - Adjust settings based on performance requirements
   - Balance visual quality with performance needs

## Exercise 12: Setting Up Camera Systems

1. Configure main camera:
   - Set appropriate field of view (e.g., 60 degrees)
   - Configure clipping planes
   - Add camera controller script for movement

2. Add additional cameras:
   - Create robot's eye-level camera
   - Add top-down overview camera
   - Implement camera switching system

3. Implement camera effects:
   - Add camera shake for impact events
   - Implement smooth camera transitions
   - Add depth of field for focus effects

## Verification Steps

1. Confirm that HDRP pipeline is properly configured
2. Verify that lighting appears realistic and physically accurate
3. Check that materials render with appropriate properties
4. Ensure that performance is acceptable for target hardware

## Expected Outcomes

- Understanding of HDRP setup and configuration
- Knowledge of realistic material creation and application
- Experience with advanced lighting techniques
- Ability to create photorealistic environments for robotics

## Troubleshooting

- If rendering appears incorrect, check HDRP asset assignment
- If performance is poor, reduce post-processing effects and material complexity
- If lighting seems wrong, verify light units and HDRP settings

## Next Steps

After completing these exercises, proceed to the ROS-Unity bridge simulation exercises to understand how to connect Unity with ROS 2 for integrated simulation.