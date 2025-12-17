// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics',
      items: [
        'glossary',
        {
          type: 'category',
          label: 'Module 1: The Robotic Nervous System (ROS 2)',
          items: [
            'module-1-ros/intro',
            'module-1-ros/overview',
            'module-1-ros/setup',
            'module-1-ros/nodes',
            'module-1-ros/topics',
            'module-1-ros/services-actions',
            'module-1-ros/parameters-launch',
            'module-1-ros/launch-files',
            'module-1-ros/rclpy-integration',
            'module-1-ros/urdf-fundamentals',
            'module-1-ros/rviz-visualization',
            'module-1-ros/mini-project',
            'module-1-ros/summary'
          ],
        },
        {
          type: 'category',
          label: 'Module 2: The Digital Twin (Gazebo & Unity)',
          items: [
            'module-2-digital-twin/intro',
            'module-2-digital-twin/gazebo-overview',
            'module-2-digital-twin/import-urdf',
            'module-2-digital-twin/physics-simulation',
            'module-2-digital-twin/environment-building',
            'module-2-digital-twin/sensors',
            'module-2-digital-twin/unity-intro',
            'module-2-digital-twin/ros-unity-bridge',
            'module-2-digital-twin/digital-twin-project',
            'module-2-digital-twin/unity-environment',
            'module-2-digital-twin/summary'
          ],
        },
        {
          type: 'category',
          label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
          items: [
            'module-3-ai-robot-brain/intro',
            'module-3-ai-robot-brain/isaac-sim-overview',
            'module-3-ai-robot-brain/isaac-sim-setup',
            'module-3-ai-robot-brain/urdf-import',
            'module-3-ai-robot-brain/photorealistic-rendering',
            'module-3-ai-robot-brain/synthetic-data-generation',
            'module-3-ai-robot-brain/isaac-ros-overview',
            'module-3-ai-robot-brain/isaac-ros-vslam',
            'module-3-ai-robot-brain/isaac-ros-perception-nodes',
            'module-3-ai-robot-brain/introduction-to-nav2',
            'module-3-ai-robot-brain/nav2-components',
            'module-3-ai-robot-brain/integration',
            'module-3-ai-robot-brain/mini-project',
            'module-3-ai-robot-brain/summary'
          ],
        },
        {
          type: 'category',
          label: 'Module 4: Vision-Language-Action (VLA)',
          items: [
            'module-4-vla/intro',
            'module-4-vla/overview',
            'module-4-vla/multimodal-perception',
            'module-4-vla/language-understanding',
            'module-4-vla/action-generation',
            'module-4-vla/vla-integration',
            'module-4-vla/practical-implementation',
            'module-4-vla/summary'
          ],
        },
        {
          type: 'category',
          label: 'Capstone Project',
          items: [
            'capstone/intro',
            'capstone/project-overview',
            'capstone/system-design',
            'capstone/implementation',
            'capstone/testing',
            'capstone/evaluation',
            'capstone/conclusion'
          ],
        }
      ],
    },
  ],
};

export default sidebars;