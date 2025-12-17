# Quickstart Guide: Physical AI & Humanoid Robotics Book Development

**Date**: 2025-12-16
**Feature**: Physical AI & Humanoid Robotics — Full Book Plan

## Overview

This guide provides a quick setup process for developing the Physical AI & Humanoid Robotics book, including all required tools, environments, and initial configuration steps.

## Prerequisites

### System Requirements
- Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- 16GB+ RAM (32GB recommended for Isaac Sim)
- 500GB+ free disk space
- NVIDIA GPU with CUDA support (for Isaac Sim, optional for basic development)

### Software Requirements
- Git
- Node.js 18+ and npm/yarn
- Python 3.10+ with pip
- Docker and Docker Compose
- ROS 2 Humble Hawksbill
- CUDA Toolkit 11.8+ (if using Isaac Sim)

## Initial Setup

### 1. Clone and Initialize Repository
```bash
git clone <repository-url>
cd <repository-name>
git checkout -b book-plan
```

### 2. Set up Docusaurus Environment
```bash
# Install Node.js dependencies
npm install

# Create docs directory structure
mkdir -p docs/{module-1-ros,module-2-digital-twin,module-3-ai-brain,module-4-vla,capstone}

# Start local development server
npm start
```

### 3. ROS 2 Environment Setup
```bash
# Install ROS 2 Humble (Ubuntu)
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep2 python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Source ROS 2 environment
source /opt/ros/humble/setup.bash
```

### 4. Simulation Environment Setup

#### Gazebo Setup
```bash
# Install Gazebo Garden
sudo apt install ros-humble-gazebo-*
sudo apt install ignition-garden

# Verify installation
ign gazebo
```

#### Isaac Sim Setup (Optional)
```bash
# Download Isaac Sim from NVIDIA Developer portal
# Follow NVIDIA's installation guide for your system
# Ensure CUDA and GPU drivers are properly configured
```

## Development Workflow

### Creating New Chapters
1. Create new MDX file in appropriate module directory
2. Follow the chapter template structure:
   - Learning objectives
   - Content with code examples
   - Simulation steps
   - Diagram descriptions
   - Exercises/MCQs

### Running Simulations
1. Navigate to simulation directory
2. Source ROS 2 environment: `source /opt/ros/humble/setup.bash`
3. Run simulation: `ros2 launch <package> <launch_file>.py`

### Testing RAG System
1. Set up environment variables:
   ```bash
   export OPENAI_API_KEY=your_key_here
   export NEON_DATABASE_URL=your_url
   export QDRANT_URL=your_url
   ```
2. Run the RAG API:
   ```bash
   cd rag-system/api
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

## Book Structure

### Module 1: The Robotic Nervous System (ROS 2)
- Directory: `docs/module-1-ros/`
- Focus: ROS 2 fundamentals, nodes, topics, services, URDF
- Target: Students learning ROS 2 basics

### Module 2: The Digital Twin (Gazebo & Unity)
- Directory: `docs/module-2-digital-twin/`
- Focus: Simulation, physics, sensors, Unity integration
- Target: Students learning simulation environments

### Module 3: The AI-Robot Brain (NVIDIA Isaac™)
- Directory: `docs/module-3-ai-brain/`
- Focus: Perception, VSLAM, navigation, Isaac ROS
- Target: Students learning advanced AI robotics

### Module 4: Vision-Language-Action (VLA)
- Directory: `docs/module-4-vla/`
- Focus: LLM integration, voice commands, autonomous agents
- Target: Students learning full autonomous systems

### Capstone Project
- Directory: `docs/capstone/`
- Focus: Complete Voice→Plan→Navigate→Perceive→Act pipeline
- Integration of all previous modules

## Building and Deployment

### Local Build
```bash
# Build static site
npm run build

# Serve locally to test
npm run serve
```

### GitHub Pages Deployment
```bash
# Deploy to GitHub Pages
GIT_USER=<Your GitHub username> CURRENT_BRANCH=main USE_SSH=true npm run deploy
```

## RAG System Setup

### Initial Ingestion
```bash
cd rag-system/ingestion
python parser.py --source-dir ../../docs --output-dir ./chunks
python chunker.py --input-dir ./chunks --output-db ../vector-store/book-vectors
```

### API Server
```bash
cd rag-system/api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Testing the Chat Interface
```bash
# Query the API directly
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain ROS 2 nodes", "context": "module-1-ros"}'
```

## Troubleshooting

### Common ROS 2 Issues
- **"Command 'ros2' not found"**: Ensure ROS 2 environment is sourced
- **"Package not found"**: Check package installation and workspace setup
- **"Permission denied"**: Check file permissions and ROS 2 configuration

### Docusaurus Issues
- **"Module not found"**: Run `npm install` to install dependencies
- **"Port already in use"**: Use `npm start -- --port 3001` for different port
- **"Build fails"**: Check MDX syntax and image references

### Simulation Issues
- **Gazebo crashes**: Check GPU drivers and graphics libraries
- **Isaac Sim won't start**: Verify CUDA installation and GPU compatibility
- **URDF import errors**: Validate URDF syntax and file paths

## Next Steps

1. Begin with Module 1 content creation
2. Set up simulation environments for each module
3. Implement RAG system integration
4. Create capstone project connecting all modules
5. Test complete Voice→Plan→Navigate→Perceive→Act pipeline