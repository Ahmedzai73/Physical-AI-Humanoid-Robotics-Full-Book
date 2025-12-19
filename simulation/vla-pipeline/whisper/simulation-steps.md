# Voice Command Processing with Whisper Simulation Steps

This guide provides step-by-step instructions for integrating Whisper for speech-to-text robot commands in the Vision-Language-Action (VLA) system as covered in Module 4 of the Physical AI & Humanoid Robotics textbook.

## Overview

This simulation demonstrates how to process voice input using Whisper and convert it into structured commands for robot control, establishing the foundational input mechanism for the VLA system.

## Prerequisites

- ROS 2 installed (Humble Hawksbill or later)
- Python 3.8+ with pip
- Completed Module 1-3 simulation exercises
- Basic understanding of speech recognition concepts

## Simulation Environment Setup

1. Install Whisper and related packages:
   ```bash
   pip3 install openai-whisper
   # Or for GPU acceleration:
   # pip3 install --upgrade --force-reinstall openai-whisper[cuda]
   ```

2. Install additional dependencies:
   ```bash
   pip3 install sounddevice numpy pyaudio
   ```

## Exercise 1: Install and Test Whisper

1. Verify Whisper installation:
   ```bash
   python3 -c "import whisper; print(whisper.__version__)"
   ```

2. Test Whisper with a sample audio file:
   ```bash
   # Download a sample audio file or use your own
   # Test with Whisper
   whisper sample_audio.wav --model tiny
   ```

3. Verify available models:
   ```python
   import whisper
   print(whisper.available_models())
   ```

## Exercise 2: Create Basic Whisper ROS Node

1. Create Whisper integration node (`src/physical_ai_robotics/whisper_processor.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   import whisper
   import numpy as np
   import sounddevice as sd
   import queue
   import threading
   import tempfile
   import wave
   import io
   from std_msgs.msg import String
   from sensor_msgs.msg import AudioData

   class WhisperProcessor(Node):
       def __init__(self):
           super().__init__('whisper_processor')

           # Publisher for recognized text
           self.text_publisher = self.create_publisher(String, '/recognized_speech', 10)

           # Subscriber for audio data
           self.audio_subscriber = self.create_subscription(
               AudioData, '/audio_input', self.audio_callback, 10
           )

           # Load Whisper model
           self.get_logger().info('Loading Whisper model...')
           self.model = whisper.load_model("tiny")  # Use "base" or "small" for better accuracy
           self.get_logger().info('Whisper model loaded successfully')

           # Audio processing parameters
           self.sample_rate = 16000  # Whisper expects 16kHz
           self.audio_buffer = []
           self.buffer_size = 16000  # 1 second of audio at 16kHz
           self.recording = False

           # Timer for processing audio
           self.process_timer = self.create_timer(2.0, self.process_audio_buffer)

           self.get_logger().info('Whisper Processor initialized')

       def audio_callback(self, msg):
           """Process incoming audio data"""
           # Convert audio data to numpy array
           audio_data = np.frombuffer(msg.data, dtype=np.int16)
           audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

           # Add to buffer
           self.audio_buffer.extend(audio_data)

           # Keep only the most recent buffer_size samples
           if len(self.audio_buffer) > self.buffer_size:
               self.audio_buffer = self.audio_buffer[-self.buffer_size:]

       def process_audio_buffer(self):
           """Process accumulated audio buffer with Whisper"""
           if len(self.audio_buffer) < 0.5 * self.sample_rate:  # At least 0.5 seconds
               return

           # Convert buffer to numpy array
           audio_array = np.array(self.audio_buffer)

           # Process with Whisper
           try:
               # Transcribe the audio
               result = self.model.transcribe(audio_array, fp16=False)  # Use fp16=False for CPU
               recognized_text = result['text'].strip()

               if recognized_text:  # Only publish if there's text
                   self.get_logger().info(f'Recognized: "{recognized_text}"')

                   # Publish recognized text
                   text_msg = String()
                   text_msg.data = recognized_text
                   self.text_publisher.publish(text_msg)

                   # Clear buffer after successful recognition
                   self.audio_buffer = []

           except Exception as e:
               self.get_logger().error(f'Error processing audio: {e}')

   def main(args=None):
       rclpy.init(args=args)
       processor = WhisperProcessor()

       try:
           rclpy.spin(processor)
       except KeyboardInterrupt:
           processor.get_logger().info('Whisper processor stopped by user')
       finally:
           processor.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 3: Create Audio Input Node

1. Create audio input node to capture microphone input (`src/physical_ai_robotics/audio_input.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   import pyaudio
   import numpy as np
   from sensor_msgs.msg import AudioData
   import struct

   class AudioInput(Node):
       def __init__(self):
           super().__init__('audio_input')

           # Publisher for audio data
           self.audio_publisher = self.create_publisher(AudioData, '/audio_input', 10)

           # Audio parameters
           self.rate = 16000  # Whisper expects 16kHz
           self.chunk = 1024
           self.format = pyaudio.paInt16
           self.channels = 1

           # Initialize PyAudio
           self.p = pyaudio.PyAudio()

           # Open audio stream
           self.stream = self.p.open(
               format=self.format,
               channels=self.channels,
               rate=self.rate,
               input=True,
               frames_per_buffer=self.chunk
           )

           # Timer for audio capture
           self.audio_timer = self.create_timer(0.1, self.capture_audio)

           self.get_logger().info('Audio input node initialized')

       def capture_audio(self):
           """Capture audio from microphone"""
           try:
               # Read audio data
               data = self.stream.read(self.chunk, exception_on_overflow=False)

               # Create AudioData message
               audio_msg = AudioData()
               audio_msg.data = data
               audio_msg.layout.data_offset = 0

               # Publish audio data
               self.audio_publisher.publish(audio_msg)

           except Exception as e:
               self.get_logger().error(f'Error capturing audio: {e}')

       def destroy_node(self):
           """Clean up audio resources"""
           if self.stream:
               self.stream.stop_stream()
               self.stream.close()
           if self.p:
               self.p.terminate()
           super().destroy_node()

   def main(args=None):
       rclpy.init(args=args)
       audio_input = AudioInput()

       try:
           rclpy.spin(audio_input)
       except KeyboardInterrupt:
           audio_input.get_logger().info('Audio input stopped by user')
       finally:
           audio_input.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 4: Create Voice Command Parser

1. Create voice command parser node (`src/physical_ai_robotics/voice_command_parser.py`):
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from geometry_msgs.msg import Twist
   from std_msgs.msg import Bool
   import re

   class VoiceCommandParser(Node):
       def __init__(self):
           super().__init__('voice_command_parser')

           # Subscriber for recognized speech
           self.speech_subscriber = self.create_subscription(
               String, '/recognized_speech', self.speech_callback, 10
           )

           # Publisher for robot commands
           self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

           # Publisher for command status
           self.status_publisher = self.create_publisher(String, '/command_status', 10)

           # Define command patterns
           self.command_patterns = {
               'move_forward': [
                   r'move forward',
                   r'go forward',
                   r'forward',
                   r'straight',
                   r'go straight'
               ],
               'move_backward': [
                   r'move backward',
                   r'go backward',
                   r'backward',
                   r'reverse',
                   r'back'
               ],
               'turn_left': [
                   r'turn left',
                   r'left',
                   r'go left',
                   r'rotate left'
               ],
               'turn_right': [
                   r'turn right',
                   r'right',
                   r'go right',
                   r'rotate right'
               ],
               'stop': [
                   r'stop',
                   r'pause',
                   r'hold',
                   r'freeze'
               ],
               'move_to_location': [
                   r'go to (\w+)',
                   r'move to (\w+)',
                   r'go to the (\w+)',
                   r'go to (kitchen|bedroom|living room|office|dining room)'
               ]
           }

           self.get_logger().info('Voice Command Parser initialized')

       def speech_callback(self, msg):
           """Process recognized speech and convert to robot commands"""
           recognized_text = msg.data.lower().strip()
           self.get_logger().info(f'Processing voice command: "{recognized_text}"')

           # Parse the command
           command = self.parse_command(recognized_text)

           if command:
               # Publish command status
               status_msg = String()
               status_msg.data = f'Executing command: {command["type"]}'
               self.status_publisher.publish(status_msg)

               # Execute the command
               self.execute_command(command)
           else:
               # Unknown command
               status_msg = String()
               status_msg.data = f'Unknown command: {recognized_text}'
               self.status_publisher.publish(status_msg)

       def parse_command(self, text):
           """Parse text command and return structured command"""
           for cmd_type, patterns in self.command_patterns.items():
               for pattern in patterns:
                   match = re.search(pattern, text)
                   if match:
                       if cmd_type == 'move_to_location':
                           location = match.group(1) if match.groups() else 'unknown'
                           return {
                               'type': cmd_type,
                               'location': location
                           }
                       else:
                           return {
                               'type': cmd_type
                           }

           return None

       def execute_command(self, command):
           """Execute parsed command"""
           cmd_vel = Twist()

           if command['type'] == 'move_forward':
               cmd_vel.linear.x = 0.3  # Forward at 0.3 m/s
           elif command['type'] == 'move_backward':
               cmd_vel.linear.x = -0.3  # Backward at 0.3 m/s
           elif command['type'] == 'turn_left':
               cmd_vel.angular.z = 0.5  # Turn left at 0.5 rad/s
           elif command['type'] == 'turn_right':
               cmd_vel.angular.z = -0.5  # Turn right at 0.5 rad/s
           elif command['type'] == 'stop':
               cmd_vel.linear.x = 0.0
               cmd_vel.angular.z = 0.0
           elif command['type'] == 'move_to_location':
               # For location commands, we would need a navigation system
               # This is a simplified version that just moves forward
               cmd_vel.linear.x = 0.2
               cmd_vel.angular.z = 0.0

           # Publish command
           self.cmd_publisher.publish(cmd_vel)
           self.get_logger().info(f'Published command: {command["type"]}')

   def main(args=None):
       rclpy.init(args=args)
       parser = VoiceCommandParser()
       rclpy.spin(parser)
       parser.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## Exercise 5: Create Voice Command Integration Launch File

1. Create launch file for voice command system (`launch/voice_command_system.launch.py`):
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node

   def generate_launch_description():
       use_sim_time = LaunchConfiguration('use_sim_time', default='false')

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='false',
               description='Use simulation clock if true'
           ),

           # Audio input node
           Node(
               package='physical_ai_robotics',
               executable='audio_input',
               name='audio_input',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Whisper processor node
           Node(
               package='physical_ai_robotics',
               executable='whisper_processor',
               name='whisper_processor',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Voice command parser
           Node(
               package='physical_ai_robotics',
               executable='voice_command_parser',
               name='voice_command_parser',
               parameters=[{'use_sim_time': use_sim_time}],
               output='screen'
           )
       ])
   ```

## Exercise 6: Test Voice Command System

1. Launch the voice command system:
   ```bash
   ros2 launch physical_ai_robotics voice_command_system.launch.py
   ```

2. Test with simple voice commands:
   - "Move forward"
   - "Turn left"
   - "Stop"
   - "Go backward"

3. Monitor the system:
   ```bash
   # Monitor recognized speech
   ros2 topic echo /recognized_speech

   # Monitor command status
   ros2 topic echo /command_status

   # Monitor robot commands
   ros2 topic echo /cmd_vel
   ```

## Exercise 7: Improve Voice Recognition Accuracy

1. Add voice activity detection (VAD) to improve recognition:
   ```python
   # Add to WhisperProcessor class
   def detect_voice_activity(self, audio_data, threshold=0.01):
       """Simple VAD based on audio energy"""
       energy = np.mean(np.abs(audio_data))
       return energy > threshold
   ```

2. Implement confidence scoring for recognition:
   ```python
   # Add confidence scoring to transcriptions
   def transcribe_with_confidence(self, audio_array):
       """Transcribe audio and estimate confidence"""
       result = self.model.transcribe(audio_array, fp16=False)
       # In a real implementation, you would calculate confidence
       # For now, return result with placeholder confidence
       return result, 0.9  # Placeholder confidence
   ```

## Exercise 8: Add Command Confirmation

1. Create command confirmation system:
   ```python
   # Add to VoiceCommandParser class
   def confirm_command(self, command_text, command_parsed):
       """Confirm command with user"""
       confirmation_msg = String()
       confirmation_msg.data = f'Did you say: {command_text}? Confirming action: {command_parsed["type"]}'
       self.status_publisher.publish(confirmation_msg)
   ```

## Exercise 9: Handle Noisy Environments

1. Add noise reduction capabilities:
   ```python
   from scipy import signal

   # Add noise reduction to WhisperProcessor
   def reduce_noise(self, audio_data, noise_reduction_factor=0.1):
       """Simple noise reduction"""
       # Apply a simple high-pass filter to remove low-frequency noise
       b, a = signal.butter(3, 0.1, btype='high', fs=self.sample_rate)
       filtered_audio = signal.filtfilt(b, a, audio_data)
       return filtered_audio
   ```

## Exercise 10: Integrate with Robot Control

1. Connect voice commands to actual robot control:
   ```python
   # Update voice command parser to work with navigation
   def execute_navigation_command(self, location):
       """Execute navigation to specific location"""
       # This would integrate with Nav2 for actual navigation
       # For simulation, we'll send basic navigation commands
       pass
   ```

## Exercise 11: Add Voice Feedback

1. Create text-to-speech feedback:
   ```python
   import pyttsx3

   class VoiceFeedback(Node):
       def __init__(self):
           super().__init__('voice_feedback')

           # Subscriber for status updates
           self.status_subscriber = self.create_subscription(
               String, '/command_status', self.status_callback, 10
           )

           # Initialize text-to-speech engine
           self.tts_engine = pyttsx3.init()
           self.tts_engine.setProperty('rate', 150)  # Speed of speech
           self.tts_engine.setProperty('volume', 0.9)  # Volume level

       def status_callback(self, msg):
           """Provide voice feedback for system status"""
           self.get_logger().info(f'Voice feedback: {msg.data}')
           self.tts_engine.say(msg.data)
           self.tts_engine.runAndWait()
   ```

## Exercise 12: Performance Optimization

1. Optimize Whisper for real-time processing:
   ```python
   # Use smaller models for real-time processing
   # Or implement streaming transcription
   class OptimizedWhisperProcessor(Node):
       def __init__(self):
           super().__init__('optimized_whisper_processor')
           # Load smaller model for faster processing
           self.model = whisper.load_model("tiny.en")  # English-only model is faster
   ```

## Verification Steps

1. Confirm Whisper model loads successfully
2. Verify audio input is captured correctly
3. Check that voice commands are recognized and parsed
4. Validate that robot commands are published based on voice input
5. Ensure system operates in real-time

## Expected Outcomes

- Understanding of Whisper integration with ROS
- Knowledge of voice command processing pipeline
- Experience with speech-to-text conversion
- Ability to create voice-controlled robot systems

## Troubleshooting

- If audio input fails, check microphone permissions and drivers
- If Whisper doesn't recognize speech, try different models or improve audio quality
- If commands aren't parsed, verify command patterns and audio quality

## Next Steps

After completing these exercises, proceed to the LLM-based cognitive planning system to understand how to map natural language commands to specific robotic actions using cognitive planning.