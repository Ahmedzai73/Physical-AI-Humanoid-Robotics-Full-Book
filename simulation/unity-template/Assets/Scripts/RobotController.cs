using UnityEngine;

namespace PhysicalAI
{
    /// <summary>
    /// Basic Robot Controller for the Physical AI & Humanoid Robotics textbook
    /// This script demonstrates how to control a robot in Unity for visualization purposes
    /// </summary>
    public class RobotController : MonoBehaviour
    {
        [Header("Movement Settings")]
        public float moveSpeed = 2.0f;
        public float turnSpeed = 100.0f;

        [Header("Joint Control")]
        public Transform[] joints;

        [Header("Visualization")]
        public bool showTrajectory = true;
        public LineRenderer trajectoryLine;

        private Vector3 lastPosition;
        private float jointAngle = 0.0f;

        void Start()
        {
            lastPosition = transform.position;

            if (showTrajectory && trajectoryLine == null)
            {
                trajectoryLine = gameObject.AddComponent<LineRenderer>();
                trajectoryLine.material = new Material(Shader.Find("Sprites/Default"));
                trajectoryLine.widthMultiplier = 0.1f;
                trajectoryLine.positionCount = 1;
                trajectoryLine.SetPosition(0, transform.position);
            }
        }

        void Update()
        {
            // Basic movement controls for demonstration
            // In a real implementation, this would be driven by ROS messages
            HandleMovement();

            // Animate joints for visualization
            AnimateJoints();

            // Update trajectory visualization
            UpdateTrajectory();
        }

        private void HandleMovement()
        {
            // Example movement - in practice, this would be controlled by ROS
            float horizontal = Input.GetAxis("Horizontal");
            float vertical = Input.GetAxis("Vertical");

            Vector3 forward = transform.forward * vertical * moveSpeed * Time.deltaTime;
            Vector3 right = transform.right * horizontal * moveSpeed * Time.deltaTime;

            transform.position += forward + right;
            transform.Rotate(Vector3.up, horizontal * turnSpeed * Time.deltaTime);
        }

        private void AnimateJoints()
        {
            // Simple oscillating animation for demonstration
            jointAngle += Time.deltaTime * 2.0f;

            if (joints != null)
            {
                for (int i = 0; i < joints.Length; i++)
                {
                    // Apply a simple oscillating rotation to each joint
                    float jointRotation = Mathf.Sin(jointAngle + i) * 10.0f;
                    joints[i].Rotate(Vector3.right, jointRotation * Time.deltaTime);
                }
            }
        }

        private void UpdateTrajectory()
        {
            if (showTrajectory && trajectoryLine != null)
            {
                // Add new position to trajectory if moved significantly
                if (Vector3.Distance(transform.position, lastPosition) > 0.1f)
                {
                    int currentPoints = trajectoryLine.positionCount;
                    trajectoryLine.positionCount = currentPoints + 1;
                    trajectoryLine.SetPosition(currentPoints, transform.position);
                    lastPosition = transform.position;
                }
            }
        }

        /// <summary>
        /// Method to receive position updates from ROS (simulated)
        /// </summary>
        public void SetPosition(Vector3 newPosition)
        {
            transform.position = newPosition;
        }

        /// <summary>
        /// Method to receive rotation updates from ROS (simulated)
        /// </summary>
        public void SetRotation(Quaternion newRotation)
        {
            transform.rotation = newRotation;
        }

        /// <summary>
        /// Method to update joint angles from ROS (simulated)
        /// </summary>
        public void SetJointAngles(float[] angles)
        {
            if (joints != null && angles != null)
            {
                for (int i = 0; i < Mathf.Min(joints.Length, angles.Length); i++)
                {
                    joints[i].localRotation = Quaternion.Euler(angles[i], 0, 0);
                }
            }
        }
    }
}