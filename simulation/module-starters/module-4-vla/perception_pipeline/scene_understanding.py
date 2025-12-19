#!/usr/bin/env python3

"""
Scene Understanding Module for Physical AI & Humanoid Robotics Textbook
Module 4: Vision-Language-Action (VLA)

This module demonstrates scene understanding and spatial reasoning for the VLA system.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from geometry_msgs.msg import Point, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from collections import defaultdict
import math


class SceneUnderstandingNode(Node):
    def __init__(self):
        super().__init__('scene_understanding_node')

        # Create CvBridge for image conversion
        self.bridge = CvBridge()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detection/detections',
            self.detection_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publishers
        self.spatial_relationships_pub = self.create_publisher(
            String,
            '/scene_understanding/spatial_relationships',
            10
        )

        self.scene_graph_pub = self.create_publisher(
            String,
            '/scene_understanding/scene_graph',
            10
        )

        self.spatial_query_pub = self.create_publisher(
            String,
            '/scene_understanding/query_response',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/scene_understanding/status',
            10
        )

        self.get_logger().info('Scene Understanding Node initialized')

        # Store data
        self.latest_detections = None
        self.latest_scan = None
        self.pending_query = None

        # Spatial relationship thresholds
        self.near_threshold = 1.0  # meters
        self.far_threshold = 3.0   # meters

        # Scene representation
        self.scene_objects = []
        self.spatial_relations = []

    def image_callback(self, msg):
        """Process incoming image (for future use)"""
        pass

    def detection_callback(self, msg):
        """Process incoming detections for scene understanding"""
        self.latest_detections = msg
        self.update_scene_representation()

        # Publish spatial relationships
        relationships = self.compute_spatial_relationships()
        relationships_msg = String()
        relationships_msg.data = json.dumps(relationships)
        self.spatial_relationships_pub.publish(relationships_msg)

        # Publish scene graph
        scene_graph = self.build_scene_graph()
        graph_msg = String()
        graph_msg.data = json.dumps(scene_graph)
        self.scene_graph_pub.publish(graph_msg)

        # Process pending query if available
        if self.pending_query:
            response = self.answer_spatial_query(self.pending_query, relationships)
            response_msg = String()
            response_msg.data = response
            self.spatial_query_pub.publish(response_msg)
            self.pending_query = None

        # Publish status
        status_msg = String()
        status_msg.data = f'Updated scene with {len(self.scene_objects)} objects'
        self.status_pub.publish(status_msg)

        self.get_logger().debug(f'Scene understanding updated - {len(self.scene_objects)} objects')

    def scan_callback(self, msg):
        """Process LIDAR scan for spatial context"""
        self.latest_scan = msg

    def update_scene_representation(self):
        """Update internal scene representation from detections"""
        if not self.latest_detections:
            return

        # Clear old objects
        self.scene_objects = []

        # Process each detection
        for detection in self.latest_detections.detections:
            if detection.results:
                obj_class = detection.results[0].hypothesis.class_id
                confidence = detection.results[0].hypothesis.score

                # Create object representation
                obj = {
                    'class': obj_class,
                    'confidence': confidence,
                    'position': {
                        'x': detection.bbox.center.x,
                        'y': detection.bbox.center.y,
                        'size_x': detection.bbox.size_x,
                        'size_y': detection.bbox.size_y
                    }
                }

                self.scene_objects.append(obj)

    def compute_spatial_relationships(self):
        """Compute spatial relationships between objects"""
        relationships = []

        if len(self.scene_objects) < 2:
            return relationships

        for i, obj1 in enumerate(self.scene_objects):
            for j, obj2 in enumerate(self.scene_objects):
                if i != j:
                    # Calculate distance between objects (simplified 2D distance)
                    dx = obj1['position']['x'] - obj2['position']['x']
                    dy = obj1['position']['y'] - obj2['position']['y']
                    distance = math.sqrt(dx*dx + dy*dy)

                    # Determine spatial relationship
                    if distance < 50:  # pixels, for image-based detection
                        relation = {
                            'subject': obj1['class'],
                            'relation': 'near',
                            'object': obj2['class'],
                            'distance': distance,
                            'confidence': min(obj1['confidence'], obj2['confidence'])
                        }
                        relationships.append(relation)

        return relationships

    def build_scene_graph(self):
        """Build a scene graph representation"""
        scene_graph = {
            'objects': [],
            'relationships': []
        }

        # Add objects
        for obj in self.scene_objects:
            scene_graph['objects'].append({
                'class': obj['class'],
                'confidence': obj['confidence'],
                'position': obj['position']
            })

        # Add relationships
        relationships = self.compute_spatial_relationships()
        scene_graph['relationships'] = relationships

        return scene_graph

    def answer_spatial_query(self, query, relationships):
        """Answer spatial queries about the scene"""
        query_lower = query.lower()

        # Check for spatial relationship queries
        if "near" in query_lower or "close to" in query_lower:
            target_objects = self.extract_objects_from_query(query_lower)
            if target_objects:
                return self.find_objects_near(target_objects[0], relationships)

        elif "left of" in query_lower or "right of" in query_lower:
            target_objects = self.extract_objects_from_query(query_lower)
            if target_objects:
                return self.find_objects_directional(target_objects[0])

        elif "between" in query_lower:
            target_objects = self.extract_objects_from_query(query_lower)
            if len(target_objects) >= 2:
                return self.find_objects_between(target_objects[0], target_objects[1])

        # Default response
        return f"I understand you're asking about spatial relationships, but I need more specific information. The scene contains: {', '.join([obj['class'] for obj in self.scene_objects])}"

    def extract_objects_from_query(self, query):
        """Extract object names from a query"""
        # Simple keyword matching - in practice, use NLP
        color_keywords = ["red", "green", "blue", "yellow", "orange", "purple", "cyan"]
        shape_keywords = ["object", "item", "thing", "box", "cylinder"]

        found_objects = []
        for color in color_keywords:
            if color in query:
                found_objects.append(color)

        return found_objects

    def find_objects_near(self, target_object, relationships):
        """Find objects near a target object"""
        nearby_objects = []
        for rel in relationships:
            if target_object in rel['subject'] or target_object in rel['object']:
                if rel['relation'] == 'near':
                    other_obj = rel['object'] if target_object in rel['subject'] else rel['subject']
                    nearby_objects.append(other_obj)

        if nearby_objects:
            return f"The {target_object} object is near: {', '.join(nearby_objects)}"
        else:
            return f"I couldn't find any objects near the {target_object} object."

    def find_objects_directional(self, target_object):
        """Find objects in a directional relationship"""
        # For image-based detection, we can determine relative positions
        target_pos = None
        other_objects = []

        for obj in self.scene_objects:
            if target_object in obj['class']:
                target_pos = obj['position']
            else:
                other_objects.append(obj)

        if not target_pos:
            return f"I couldn't find the {target_object} object."

        left_objects = []
        right_objects = []

        for obj in other_objects:
            if obj['position']['x'] < target_pos['x']:
                left_objects.append(obj['class'])
            else:
                right_objects.append(obj['class'])

        response = f"Objects to the left of {target_object}: {', '.join(left_objects) if left_objects else 'none'}"
        response += f"; Objects to the right of {target_object}: {', '.join(right_objects) if right_objects else 'none'}"
        return response

    def find_objects_between(self, obj1, obj2):
        """Find objects between two specified objects"""
        # Find positions of the two objects
        pos1 = None
        pos2 = None

        for obj in self.scene_objects:
            if obj1 in obj['class']:
                pos1 = obj['position']
            elif obj2 in obj['class']:
                pos2 = obj['class']

        if not pos1 or not pos2:
            return f"Could not locate both {obj1} and {obj2} objects."

        # For now, return all other objects
        between_objects = []
        for obj in self.scene_objects:
            if obj1 not in obj['class'] and obj2 not in obj['class']:
                between_objects.append(obj['class'])

        if between_objects:
            return f"Objects between {obj1} and {obj2}: {', '.join(between_objects)}"
        else:
            return f"There are no other objects between the {obj1} and {obj2} objects."

    def set_query(self, query):
        """Set a query to be processed with next detection"""
        self.pending_query = query
        self.get_logger().info(f'Spatial query set: {query}')


def main(args=None):
    rclpy.init(args=args)

    scene_node = SceneUnderstandingNode()

    try:
        # Example: Set a query after a delay
        def set_example_query():
            scene_node.set_query("What is near the red object?")

        # Set query after 3 seconds
        scene_node.create_timer(3.0, set_example_query)

        rclpy.spin(scene_node)
    except KeyboardInterrupt:
        pass
    finally:
        scene_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()