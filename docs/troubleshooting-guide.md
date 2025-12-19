# Troubleshooting Guide: Physical AI & Humanoid Robotics Textbook

This guide addresses common issues and problems encountered while working with the Physical AI & Humanoid Robotics textbook modules and simulations.

## Common Installation Issues

### ROS 2 Installation Problems
**Problem**: Cannot source ROS 2 setup.bash
**Solution**:
- Ensure ROS 2 Humble Hawksbill is properly installed
- Check the installation path: `source /opt/ros/humble/setup.bash` (Ubuntu) or `call c:\dev\ros_humble\setup.bat` (Windows)
- Verify your ROS_DISTRO environment variable is set correctly

**Problem**: Colcon build fails with dependency errors
**Solution**:
- Run `rosdep install --from-paths src --ignore-src -r -y` to install missing dependencies
- Ensure all required packages are in your workspace src directory
- Check for missing package.xml files

### Isaac Sim Issues
**Problem**: Isaac Sim fails to start or crashes
**Solution**:
- Ensure you have an NVIDIA GPU with CUDA support
- Update to the latest NVIDIA drivers
- Check that Isaac Sim is properly licensed
- Verify system requirements are met

**Problem**: Isaac Sim ROS bridge not connecting
**Solution**:
- Check that Isaac Sim extension for ROS bridge is enabled
- Verify correct ROS distribution is sourced
- Ensure Isaac Sim and ROS are using the same IP/network

### Unity Integration Problems
**Problem**: Unity ROS-TCP-Connector fails to connect
**Solution**:
- Verify Unity and ROS are on the same network
- Check firewall settings block the connection port (default 10000)
- Ensure ROS TCP Connector package is properly installed in Unity

**Problem**: Scene fails to load in Unity
**Solution**:
- Check Unity version compatibility (2022.3 LTS recommended)
- Verify HDRP project template was used
- Ensure all required packages are imported

## Simulation-Specific Issues

### Gazebo Simulation Problems
**Problem**: URDF model doesn't appear in Gazebo
**Solution**:
- Verify URDF file is valid and properly formatted
- Check that joint and link names are consistent
- Ensure all mesh files are in correct locations

**Problem**: Robot falls through the ground
**Solution**:
- Check collision properties in URDF
- Verify mass and inertia parameters are set
- Ensure physics engine is properly configured

### Navigation Issues
**Problem**: Nav2 fails to plan paths
**Solution**:
- Verify map is properly loaded and published
- Check that costmaps are updating
- Ensure transforms (TF) are properly published
- Verify sensor data is being received

**Problem**: Robot doesn't follow planned path
**Solution**:
- Check controller configuration
- Verify local and global planners are active
- Ensure robot is localized in the map

## VLA System Issues

### Voice Recognition Problems
**Problem**: Whisper model fails to process audio
**Solution**:
- Ensure PyAudio is properly installed
- Check microphone permissions and availability
- Verify audio format compatibility

**Problem**: Voice commands not recognized
**Solution**:
- Check audio input levels
- Ensure quiet environment for better recognition
- Verify Whisper model is loaded correctly

### LLM Integration Issues
**Problem**: LLM responses are slow or fail
**Solution**:
- Check API key validity and rate limits
- Verify network connectivity to LLM provider
- Consider using local LLM if available

**Problem**: LLM generates invalid robot commands
**Solution**:
- Review prompt engineering for better command structure
- Implement additional validation layers
- Check safety guardrails configuration

## Performance Issues

### High CPU/GPU Usage
**Problem**: Simulation runs slowly
**Solution**:
- Reduce simulation frequency in configuration
- Lower visual quality settings temporarily
- Close unnecessary applications to free resources
- Check for infinite loops in custom nodes

### Memory Leaks
**Problem**: Memory usage increases over time
**Solution**:
- Check for proper cleanup of subscribers/publishers
- Verify timers and callbacks are properly managed
- Use memory profiling tools to identify leaks

## Networking and Communication

### ROS 2 Communication Problems
**Problem**: Nodes cannot communicate across machines
**Solution**:
- Verify RMW implementation is consistent
- Check network configuration and firewall settings
- Ensure ROS_DOMAIN_ID is the same on all machines

**Problem**: Topics show no data
**Solution**:
- Check that publishers and subscribers are active
- Verify topic names match exactly
- Check message type compatibility

### Quality of Service (QoS) Issues
**Problem**: Messages are dropped or lost
**Solution**:
- Review QoS profile settings for publisher/subscriber
- Ensure QoS policies are compatible
- Consider reliability and durability settings

## Sensor Simulation Issues

### Camera/LiDAR Data Problems
**Problem**: Sensor data appears corrupted or incorrect
**Solution**:
- Verify sensor configuration in URDF
- Check frame IDs and TF relationships
- Validate sensor noise parameters

**Problem**: Sensor data publishing too slowly
**Solution**:
- Adjust sensor update rates in Gazebo/Isaac Sim
- Check for processing bottlenecks
- Optimize sensor configuration

## RAG System Issues

### Content Indexing Problems
**Problem**: RAG system cannot find relevant content
**Solution**:
- Verify content was properly ingested
- Check embedding quality and chunk sizes
- Adjust similarity threshold settings

**Problem**: RAG responses are slow
**Solution**:
- Optimize vector database queries
- Consider reducing context window size
- Check database indexing and performance

## Development Environment

### Python Virtual Environment
**Problem**: Package conflicts or missing dependencies
**Solution**:
- Create isolated virtual environments for different projects
- Use requirements.txt to manage dependencies
- Ensure correct Python version is used

### Docusaurus Documentation
**Problem**: Local documentation fails to build
**Solution**:
- Run `npm install` to reinstall dependencies
- Clear cache with `npm run clear`
- Check for syntax errors in MDX files

## Hardware-Specific Issues

### GPU Acceleration Problems
**Problem**: GPU acceleration not utilized
**Solution**:
- Verify CUDA is properly installed and configured
- Check that Isaac ROS nodes are GPU-enabled
- Confirm NVIDIA drivers are up to date

**Problem**: GPU memory errors
**Solution**:
- Reduce batch sizes in perception pipelines
- Close other GPU-intensive applications
- Consider using mixed precision where applicable

## Debugging Strategies

### ROS 2 Debugging
- Use `rqt_graph` to visualize node connections
- Monitor topics with `ros2 topic echo`
- Check logs with `ros2 topic list` and `ros2 node list`
- Use `tf2_tools` to debug transforms

### Simulation Debugging
- Enable verbose logging in simulation environments
- Use RViz for visual debugging
- Check simulation time vs real time ratios
- Monitor physics engine performance

## Getting Help

If issues persist:

1. **Check the logs**: Examine console output and log files for error messages
2. **ROS Answers**: Visit answers.ros.org for ROS-specific issues
3. **Isaac Community**: Check developer.nvidia.com/isaac for Isaac Sim issues
4. **Unity Forums**: For Unity-specific problems
5. **Documentation**: Review official documentation for each component
6. **Reproduce in minimal example**: Create a simple test case to isolate the issue

### Creating Effective Bug Reports
When seeking help, include:
- System specifications (OS, GPU, RAM)
- ROS/Isaac/Unity versions
- Exact error messages
- Steps to reproduce
- What you've already tried