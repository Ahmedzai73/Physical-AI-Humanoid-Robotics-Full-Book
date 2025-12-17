# Synthetic Data Generation: RGB, Depth, Bounding Boxes, Segmentation

## The Role of Synthetic Data in AI Robotics

Synthetic data generation is a cornerstone of modern AI robotics development, particularly for humanoid robots that must operate in complex, diverse environments. Unlike traditional approaches that rely on expensive and time-consuming real-world data collection, synthetic data generation allows us to create vast, diverse, and perfectly labeled datasets in simulation environments.

For humanoid robots, synthetic data generation serves several critical purposes:

1. **Perception Training**: Training computer vision models to recognize objects, people, and environments
2. **Sensor Simulation**: Creating realistic sensor data for algorithm development and testing
3. **Edge Case Generation**: Creating rare or dangerous scenarios that would be difficult to collect in reality
4. **Domain Randomization**: Improving model robustness by varying lighting, materials, and environments
5. **Cost Reduction**: Eliminating the need for expensive data collection campaigns

## Isaac Sim's Synthetic Data Generation Capabilities

Isaac Sim provides comprehensive synthetic data generation tools that can produce:

- **RGB Images**: Photorealistic color images with proper lighting and materials
- **Depth Maps**: Accurate depth information for 3D understanding
- **Semantic Segmentation**: Pixel-perfect labeling of different objects and surfaces
- **Instance Segmentation**: Individual object identification and separation
- **Bounding Boxes**: 2D and 3D bounding boxes for object detection
- **Pose Data**: Accurate 6D pose information for objects and robots

## Setting Up Synthetic Data Generation

### Configuring Sensors for Data Capture

```python
# sensor_setup.py
from omni.isaac.sensor import Camera
from omni.isaac.synthetic_utils import SyntheticDataHelper
import numpy as np
import carb

def setup_synthetic_data_cameras(robot_path):
    """
    Set up cameras for comprehensive synthetic data generation
    """
    # RGB Camera
    rgb_camera = Camera(
        prim_path=f"{robot_path}/rgb_camera",
        name="rgb_camera",
        translation=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )

    rgb_camera_resolution = (1280, 720)  # 720p for efficient processing
    rgb_camera.set_resolution(rgb_camera_resolution)

    # Depth Camera
    depth_camera = Camera(
        prim_path=f"{robot_path}/depth_camera",
        name="depth_camera",
        translation=np.array([0.05, 0.0, 0.0]),  # Slightly offset for stereo
        orientation=np.array([0.0, 0.0, 0.0, 1.0])
    )

    depth_camera.set_resolution(rgb_camera_resolution)

    print("Synthetic data cameras configured")
    return rgb_camera, depth_camera

def enable_synthetic_data_sensors():
    """
    Enable Isaac Sim's synthetic data sensors and extensions
    """
    # Enable synthetic data extension
    carb.settings.get_settings().set("/isaaclab/tasks/enable_synthetic_data", True)

    # Configure synthetic data settings
    carb.settings.get_settings().set("/app/renderer/enableGPUDrivenRender", True)
    carb.settings.get_settings().set("/app/renderer/renderTargetWidth", 1280)
    carb.settings.get_settings().set("/app/renderer/renderTargetHeight", 720)

    print("Synthetic data sensors enabled")
```

### Creating Ground Truth Annotation Tools

```python
# annotation_setup.py
from omni.isaac.synthetic_utils import plot
import omni.kit
from pxr import UsdGeom

def setup_annotation_channels(camera):
    """
    Set up annotation channels for different types of ground truth data
    """
    # Enable semantic segmentation
    camera.add_semantic_segmentation_to_frame()

    # Enable instance segmentation
    camera.add_instance_segmentation_to_frame()

    # Enable bounding box annotations
    camera.add_bounding_box_2d_tight_to_frame()
    camera.add_bounding_box_2d_loose_to_frame()
    camera.add_bounding_box_3d_to_frame()

    # Enable depth information
    camera.add_distance_to_image_plane_to_frame()

    # Enable normal information
    camera.add_normals_to_frame()

    print("Annotation channels configured for camera")

def assign_semantic_labels(stage):
    """
    Assign semantic labels to objects in the scene
    """
    # Example: Label different objects in the scene
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName()

        if prim_type == "Cube" and "floor" in prim.GetName().lower():
            # Assign floor label
            plot.add_label_to_prim(prim, "floor")
        elif prim_type == "Cube" and "wall" in prim.GetName().lower():
            # Assign wall label
            plot.add_label_to_prim(prim, "wall")
        elif prim_type == "Cylinder" and "chair" in prim.GetName().lower():
            # Assign chair label
            plot.add_label_to_prim(prim, "furniture")
        elif "robot" in prim.GetName().lower():
            # Assign robot label
            plot.add_label_to_prim(prim, "robot")

    print("Semantic labels assigned to scene objects")
```

## Generating RGB Images

### Basic RGB Data Generation

```python
# rgb_generation.py
import cv2
import numpy as np
from PIL import Image
import os

def capture_rgb_frames(camera, output_dir, num_frames=100):
    """
    Capture RGB frames from the camera
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_frames):
        # Render frame
        world.step(render=True)

        # Get RGB data
        rgb_data = camera.get_rgb()

        # Convert to image format
        rgb_image = Image.fromarray(rgb_data, mode="RGBA")
        rgb_image = rgb_image.convert("RGB")  # Convert to RGB

        # Save image
        image_path = os.path.join(output_dir, f"rgb_{i:06d}.jpg")
        rgb_image.save(image_path)

        print(f"Saved RGB frame {i+1}/{num_frames}: {image_path}")

def generate_varied_rgb_data(robot_path, camera, output_dir):
    """
    Generate RGB data with varied conditions
    """
    # Move robot to different positions
    robot = world.scene.get_object("humanoid_robot")

    # Capture from multiple viewpoints
    viewpoints = [
        ([0, 0, 1.6], [0, 0, 0, 1]),  # Head level, forward
        ([0, -1, 1.6], [0.1, 0, 0, 0.99]),  # Head level, looking slightly down
        ([1, 0, 1.6], [-0.1, 0, 0, 0.99]),  # Head level, looking left
        ([0, 0, 1.0], [0, 0.3, 0, 0.95]),  # Chest level, looking forward
    ]

    for idx, (pos, rot) in enumerate(viewpoints):
        # Move camera to new position
        camera.set_translation(np.array(pos))
        camera.set_orientation(np.array(rot))

        # Capture frames at this viewpoint
        for frame in range(25):  # 25 frames per viewpoint
            world.step(render=True)

            rgb_data = camera.get_rgb()
            rgb_image = Image.fromarray(rgb_data, mode="RGBA")
            rgb_image = rgb_image.convert("RGB")

            image_path = os.path.join(output_dir, f"rgb_view_{idx}_frame_{frame:04d}.jpg")
            rgb_image.save(image_path)

    print(f"Generated varied RGB data with {len(viewpoints) * 25} frames")
```

## Generating Depth Maps

### Depth Data Generation and Processing

```python
# depth_generation.py
import numpy as np
import cv2
from scipy import ndimage

def capture_depth_frames(camera, output_dir, num_frames=100):
    """
    Capture depth frames from the camera
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_frames):
        # Render frame
        world.step(render=True)

        # Get depth data
        depth_data = camera.get_depth()

        # Process depth data (convert from linear to proper depth values)
        # Isaac Sim provides linear depth, convert as needed
        processed_depth = process_depth_data(depth_data)

        # Save depth map as 16-bit PNG for precision
        depth_path = os.path.join(output_dir, f"depth_{i:06d}.png")
        cv2.imwrite(depth_path, (processed_depth * 1000).astype(np.uint16))  # Scale for 16-bit storage

        print(f"Saved depth frame {i+1}/{num_frames}: {depth_path}")

def process_depth_data(depth_data):
    """
    Process raw depth data from Isaac Sim
    """
    # Isaac Sim provides linear depth in meters
    # The raw data might need transformation depending on the camera setup
    processed = depth_data.copy()

    # Apply any necessary transformations
    # This could include filtering, scaling, or coordinate system conversions
    return processed

def generate_depth_with_annotations(camera, output_dir, annotation_dir):
    """
    Generate depth data with corresponding annotations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)

    # Get semantic segmentation alongside depth
    for i in range(100):
        world.step(render=True)

        # Capture depth
        depth_data = camera.get_depth()
        depth_path = os.path.join(output_dir, f"depth_{i:06d}.png")
        cv2.imwrite(depth_path, (depth_data * 1000).astype(np.uint16))

        # Capture semantic segmentation
        semantic_data = camera.get_semantic_segmentation()
        semantic_path = os.path.join(annotation_dir, f"semantic_{i:06d}.png")
        cv2.imwrite(semantic_path, semantic_data.astype(np.uint16))

    print(f"Generated {i+1} depth frames with semantic annotations")
```

## Generating Semantic and Instance Segmentation

### Segmentation Data Pipeline

```python
# segmentation_generation.py
def capture_segmentation_data(camera, output_dir, num_frames=100):
    """
    Capture semantic and instance segmentation data
    """
    semantic_dir = os.path.join(output_dir, "semantic")
    instance_dir = os.path.join(output_dir, "instance")

    if not os.path.exists(semantic_dir):
        os.makedirs(semantic_dir)
    if not os.path.exists(instance_dir):
        os.makedirs(instance_dir)

    for i in range(num_frames):
        world.step(render=True)

        # Get semantic segmentation
        semantic_data = camera.get_semantic_segmentation()
        semantic_path = os.path.join(semantic_dir, f"semantic_{i:06d}.png")
        cv2.imwrite(semantic_path, semantic_data.astype(np.uint16))

        # Get instance segmentation
        instance_data = camera.get_instance_segmentation()
        instance_path = os.path.join(instance_dir, f"instance_{i:06d}.png")
        cv2.imwrite(instance_path, instance_data.astype(np.uint16))

        print(f"Saved segmentation frame {i+1}/{num_frames}")

def create_segmentation_label_mapping(stage):
    """
    Create mapping between semantic labels and class IDs
    """
    label_mapping = {}
    next_id = 1  # 0 is typically reserved for background

    # Traverse the stage to find all labeled objects
    for prim in stage.Traverse():
        # Get semantic label if it exists
        semantic_label = get_semantic_label(prim)
        if semantic_label and semantic_label not in label_mapping:
            label_mapping[semantic_label] = next_id
            next_id += 1

    # Add background
    label_mapping["background"] = 0

    return label_mapping

def get_semantic_label(prim):
    """
    Helper function to get semantic label from a prim
    """
    # Implementation depends on how labels are assigned
    # This is a simplified example
    return prim.GetName()  # In practice, would access semantic label data
```

## Generating Bounding Box Annotations

### 2D and 3D Bounding Box Generation

```python
# bounding_box_generation.py
def capture_bounding_boxes(camera, output_dir, num_frames=100):
    """
    Capture 2D and 3D bounding box annotations
    """
    bbox_dir = os.path.join(output_dir, "bounding_boxes")
    if not os.path.exists(bbox_dir):
        os.makedirs(bbox_dir)

    for i in range(num_frames):
        world.step(render=True)

        # Get 2D tight bounding boxes
        bbox_2d_tight = camera.get_bounding_box_2d_tight()

        # Get 3D bounding boxes
        bbox_3d = camera.get_bounding_box_3d()

        # Save bounding box data
        bbox_data = {
            "frame": i,
            "bbox_2d_tight": bbox_2d_tight,
            "bbox_3d": bbox_3d
        }

        bbox_path = os.path.join(bbox_dir, f"bbox_{i:06d}.json")
        import json
        with open(bbox_path, 'w') as f:
            json.dump(bbox_data, f, indent=2)

        print(f"Saved bounding box data {i+1}/{num_frames}")

def format_bounding_box_annotations(bbox_data, image_width=1280, image_height=720):
    """
    Format bounding box data in standard formats (e.g., COCO, YOLO)
    """
    formatted_annotations = []

    for obj_id, bbox in enumerate(bbox_data.get("bbox_2d_tight", [])):
        # Convert to normalized coordinates (0-1 range)
        x_min = bbox[0] / image_width
        y_min = bbox[1] / image_height
        x_max = bbox[2] / image_width
        y_max = bbox[3] / image_height

        # Calculate width and height
        width = x_max - x_min
        height = y_max - y_min

        # Calculate center coordinates
        center_x = x_min + width / 2
        center_y = y_min + height / 2

        # COCO format
        coco_annotation = {
            "id": obj_id,
            "image_id": bbox_data["frame"],
            "category_id": 1,  # Placeholder, should come from semantic labels
            "bbox": [x_min * image_width, y_min * image_height, width * image_width, height * image_height],
            "area": width * height * image_width * image_height,
            "iscrowd": 0
        }

        # YOLO format
        yolo_annotation = f"{1} {center_x} {center_y} {width} {height}"  # Class 1, normalized coords

        formatted_annotations.append({
            "coco": coco_annotation,
            "yolo": yolo_annotation
        })

    return formatted_annotations
```

## Advanced Data Generation Techniques

### Domain Randomization

```python
# domain_randomization.py
import random
from pxr import Gf, UsdShade
import omni

def apply_domain_randomization(stage, frame_interval=10):
    """
    Apply domain randomization to increase dataset diversity
    """
    # Randomize lighting conditions
    randomize_lighting(stage)

    # Randomize material properties
    randomize_materials(stage)

    # Randomize camera parameters
    randomize_camera_parameters()

    # Randomize object positions
    randomize_object_positions(stage)

def randomize_lighting(stage):
    """
    Randomize lighting conditions for domain randomization
    """
    lights = []
    for prim in stage.Traverse():
        if prim.GetTypeName() in ["DistantLight", "DomeLight", "SphereLight"]:
            lights.append(prim)

    if lights:
        # Randomize one light per frame
        light = random.choice(lights)
        light_prim = UsdGeom.Xform(light)

        # Randomize intensity
        intensity_attr = light.GetAttribute("intensity")
        if intensity_attr:
            new_intensity = random.uniform(200, 1500)  # Random intensity
            intensity_attr.Set(new_intensity)

        # Randomize color temperature
        color_attr = light.GetAttribute("color")
        if color_attr:
            # Random color temperature (from warm to cool)
            color_temp = random.choice([
                Gf.Vec3f(1.0, 0.8, 0.6),  # Warm
                Gf.Vec3f(1.0, 0.9, 0.8),  # Slightly warm
                Gf.Vec3f(1.0, 1.0, 1.0),  # Neutral
                Gf.Vec3f(0.9, 0.95, 1.0), # Slightly cool
                Gf.Vec3f(0.8, 0.9, 1.0)   # Cool
            ])
            color_attr.Set(color_temp)

def randomize_materials(stage):
    """
    Randomize material properties for domain randomization
    """
    for prim in stage.Traverse():
        # Find materials in the scene
        if "Material" in prim.GetTypeName():
            material = UsdShade.Material(prim)

            # Find shaders connected to this material
            surface_output = material.GetSurfaceOutput()
            if surface_output:
                connected_shader = surface_output.GetConnectedSource()
                if connected_shader and connected_shader[0]:
                    shader = connected_shader[0]

                    # Randomize material properties
                    diffuse_input = shader.GetInput("diffuse_tint")
                    if diffuse_input:
                        # Randomize color
                        random_color = Gf.Vec3f(
                            random.uniform(0.2, 1.0),
                            random.uniform(0.2, 1.0),
                            random.uniform(0.2, 1.0)
                        )
                        diffuse_input.Set(random_color)

                    roughness_input = shader.GetInput("roughness")
                    if roughness_input:
                        # Randomize roughness
                        random_roughness = random.uniform(0.1, 0.9)
                        roughness_input.Set(random_roughness)

def randomize_object_positions(stage):
    """
    Randomize positions of objects in the scene
    """
    for prim in stage.Traverse():
        if prim.GetTypeName() in ["Cube", "Sphere", "Cylinder", "Mesh"]:
            # Skip the robot and floor
            prim_name = prim.GetName().lower()
            if "robot" in prim_name or "floor" in prim_name or "world" in prim_name:
                continue

            # Randomize position within a bounded area
            x = random.uniform(-3.0, 3.0)
            y = random.uniform(-3.0, 3.0)
            z = random.uniform(0.1, 2.0)  # Above floor

            # Apply new position
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(x, y, z))
```

### Data Augmentation in Simulation

```python
# data_augmentation.py
def apply_simulation_augmentations(robot, environment):
    """
    Apply augmentations directly in simulation for more realistic data
    """
    # Add sensor noise
    add_sensor_noise()

    # Simulate different weather conditions
    simulate_weather_conditions()

    # Add motion blur for moving objects
    enable_motion_blur()

    # Simulate lens effects
    simulate_lens_effects()

def add_sensor_noise():
    """
    Add realistic sensor noise to camera data
    """
    # This would involve configuring camera properties
    # to include realistic noise models
    carb.settings.get_settings().set("/camera/enable_noise", True)
    carb.settings.get_settings().set("/camera/noise_level", 0.01)  # Low noise level

def simulate_weather_conditions():
    """
    Simulate different weather conditions
    """
    # This could involve changing atmospheric properties
    # or adding effects like rain, fog, etc.
    # Implementation depends on available extensions
    pass

def enable_motion_blur():
    """
    Enable motion blur for more realistic camera effects
    """
    carb.settings.get_settings().set("/rtx/motionBlur/enabled", True)
    carb.settings.get_settings().set("/rtx/motionBlur/quality", 1)
```

## Organizing and Managing Synthetic Datasets

### Dataset Structure and Management

```python
# dataset_management.py
import json
import os
from datetime import datetime

class SyntheticDatasetManager:
    def __init__(self, base_path):
        self.base_path = base_path
        self.dataset_info = {
            "created": datetime.now().isoformat(),
            "format_version": "1.0",
            "data_types": [],
            "frame_count": 0,
            "annotations": {}
        }

    def create_dataset_structure(self, dataset_name):
        """
        Create standard directory structure for synthetic dataset
        """
        dataset_path = os.path.join(self.base_path, dataset_name)

        # Create main directories
        os.makedirs(os.path.join(dataset_path, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "depth"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "annotations", "semantic"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "annotations", "instance"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "annotations", "bounding_boxes"), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "metadata"), exist_ok=True)

        return dataset_path

    def update_dataset_info(self, data_type, count):
        """
        Update dataset information as data is generated
        """
        if data_type not in self.dataset_info["data_types"]:
            self.dataset_info["data_types"].append(data_type)

        self.dataset_info["frame_count"] += count

        # Save updated info
        info_path = os.path.join(self.base_path, "dataset_info.json")
        with open(info_path, 'w') as f:
            json.dump(self.dataset_info, f, indent=2)

def generate_complete_dataset(robot_path, camera, dataset_name="humanoid_perception_dataset"):
    """
    Generate a complete synthetic dataset with all data types
    """
    manager = SyntheticDatasetManager("./datasets")
    dataset_path = manager.create_dataset_structure(dataset_name)

    # Generate RGB data
    rgb_path = os.path.join(dataset_path, "rgb")
    capture_rgb_frames(camera, rgb_path, num_frames=1000)
    manager.update_dataset_info("rgb", 1000)

    # Generate depth data
    depth_path = os.path.join(dataset_path, "depth")
    generate_depth_with_annotations(camera, depth_path,
                                  os.path.join(dataset_path, "annotations", "semantic"))
    manager.update_dataset_info("depth", 1000)

    # Generate segmentation data
    seg_path = os.path.join(dataset_path, "annotations", "instance")
    capture_segmentation_data(camera, dataset_path, num_frames=1000)
    manager.update_dataset_info("segmentation", 1000)

    # Generate bounding box data
    bbox_path = os.path.join(dataset_path, "annotations", "bounding_boxes")
    capture_bounding_boxes(camera, dataset_path, num_frames=1000)
    manager.update_dataset_info("bounding_boxes", 1000)

    print(f"Complete synthetic dataset generated at: {dataset_path}")
    return dataset_path
```

## Quality Assurance and Validation

### Data Quality Checks

```python
# quality_assurance.py
def validate_synthetic_data(dataset_path):
    """
    Validate the quality and consistency of generated synthetic data
    """
    validation_results = {
        "rgb_quality": check_rgb_quality(dataset_path),
        "depth_consistency": check_depth_consistency(dataset_path),
        "annotation_accuracy": check_annotation_accuracy(dataset_path),
        "data_completeness": check_data_completeness(dataset_path)
    }

    return validation_results

def check_rgb_quality(dataset_path):
    """
    Check RGB image quality metrics
    """
    rgb_dir = os.path.join(dataset_path, "rgb")
    if not os.path.exists(rgb_dir):
        return {"status": "error", "message": "RGB directory not found"}

    # Check image properties
    import cv2
    import numpy as np

    sample_images = [f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))][:10]

    if not sample_images:
        return {"status": "error", "message": "No RGB images found"}

    quality_metrics = {
        "avg_brightness": 0,
        "avg_contrast": 0,
        "image_count": len(sample_images)
    }

    total_brightness = 0
    total_contrast = 0

    for img_file in sample_images:
        img_path = os.path.join(rgb_dir, img_file)
        img = cv2.imread(img_path)

        # Calculate brightness (mean of grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        total_brightness += brightness

        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        total_contrast += contrast

    quality_metrics["avg_brightness"] = total_brightness / len(sample_images)
    quality_metrics["avg_contrast"] = total_contrast / len(sample_images)

    return {"status": "success", "metrics": quality_metrics}

def check_depth_consistency(dataset_path):
    """
    Check depth data consistency
    """
    depth_dir = os.path.join(dataset_path, "depth")
    if not os.path.exists(depth_dir):
        return {"status": "error", "message": "Depth directory not found"}

    # Check for valid depth ranges and consistency
    sample_depths = [f for f in os.listdir(depth_dir) if f.endswith('.png')][:10]

    if not sample_depths:
        return {"status": "error", "message": "No depth images found"}

    consistency_metrics = {
        "valid_depth_ratio": 0,
        "avg_depth_range": 0,
        "outlier_ratio": 0
    }

    return {"status": "success", "metrics": consistency_metrics}
```

## Performance Optimization for Large-Scale Generation

### Efficient Data Generation Pipeline

```python
# efficient_generation.py
import concurrent.futures
import threading
from queue import Queue

class EfficientDataGenerator:
    def __init__(self, camera, world):
        self.camera = camera
        self.world = world
        self.data_queue = Queue()
        self.is_generating = False

    def start_parallel_generation(self, num_frames=1000, batch_size=10):
        """
        Start parallel data generation for improved performance
        """
        self.is_generating = True

        # Create threads for different data types
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit tasks for different data types
            rgb_future = executor.submit(self._generate_rgb_batch, num_frames, batch_size)
            depth_future = executor.submit(self._generate_depth_batch, num_frames, batch_size)
            seg_future = executor.submit(self._generate_segmentation_batch, num_frames, batch_size)

            # Wait for completion
            rgb_result = rgb_future.result()
            depth_result = depth_future.result()
            seg_result = seg_future.result()

        self.is_generating = False
        return rgb_result, depth_result, seg_result

    def _generate_rgb_batch(self, num_frames, batch_size):
        """
        Generate RGB data in batches for efficiency
        """
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)

            for i in range(batch_start, batch_end):
                if not self.is_generating:
                    break

                self.world.step(render=True)
                rgb_data = self.camera.get_rgb()

                # Process and save RGB data
                self._save_rgb_frame(rgb_data, i)

        return f"Generated {num_frames} RGB frames"

    def _save_rgb_frame(self, rgb_data, frame_id):
        """
        Save RGB frame efficiently
        """
        from PIL import Image
        import os

        output_dir = "./datasets/current/rgb"
        os.makedirs(output_dir, exist_ok=True)

        rgb_image = Image.fromarray(rgb_data, mode="RGBA")
        rgb_image = rgb_image.convert("RGB")

        image_path = os.path.join(output_dir, f"rgb_{frame_id:06d}.jpg")
        rgb_image.save(image_path, quality=95)

def setup_gpu_accelerated_generation():
    """
    Configure settings for GPU-accelerated data generation
    """
    # Enable GPU-driven rendering
    carb.settings.get_settings().set("/app/renderer/enableGPUDrivenRender", True)

    # Optimize memory usage
    carb.settings.get_settings().set("/persistent/image/cacheSize", 1024)  # MB

    # Enable multi-resolution shading for performance
    carb.settings.get_settings().set("/app/renderer/multiResolutionShading/enabled", True)

    print("GPU-accelerated generation configured")
```

## Troubleshooting Common Issues

### Data Generation Problems and Solutions

```bash
# Issue: Slow data generation speed
# Solutions:
# 1. Reduce rendering quality settings during generation
# 2. Use smaller image resolutions for initial testing
# 3. Implement batch processing and parallel generation
# 4. Optimize scene complexity

# Issue: Memory overflow during generation
# Solutions:
# 1. Process and save data in smaller batches
# 2. Clear memory between batches
# 3. Use memory mapping for large datasets
# 4. Implement data streaming

# Issue: Inconsistent annotations
# Solutions:
# 1. Verify semantic labeling is correctly applied
# 2. Check camera calibration parameters
# 3. Validate coordinate system transformations
# 4. Ensure frame synchronization between sensors

# Issue: Poor quality synthetic data
# Solutions:
# 1. Adjust lighting conditions
# 2. Verify material properties
# 3. Check camera parameters (focus, exposure)
# 4. Improve scene complexity and diversity
```

## Best Practices for Synthetic Data Generation

### For Humanoid Robotics Applications

1. **Environment Diversity**: Create varied environments similar to where the humanoid will operate
2. **Human Interaction Scenarios**: Include scenes with humans for social robotics applications
3. **Dynamic Elements**: Add moving objects and changing conditions
4. **Sensor Fusion**: Generate data from multiple sensors simultaneously
5. **Realistic Physics**: Ensure objects behave according to physical laws
6. **Edge Cases**: Include rare but important scenarios for safety
7. **Annotation Quality**: Maintain high-quality, consistent annotations
8. **Validation**: Continuously validate synthetic data against real-world data when possible

### Performance Considerations

1. **Batch Processing**: Process data in batches to optimize I/O operations
2. **Resolution Trade-offs**: Balance image quality with generation speed
3. **Storage Management**: Implement efficient storage strategies for large datasets
4. **Parallel Generation**: Use multi-threading for different data types
5. **Quality Control**: Implement validation checks to ensure data quality

## Summary

Synthetic data generation in Isaac Sim provides the foundation for training AI systems in humanoid robotics. The key components include:

1. **Sensor Configuration**: Setting up cameras and sensors for data capture
2. **Data Types**: Generating RGB, depth, segmentation, and bounding box data
3. **Annotation Systems**: Creating accurate ground truth labels
4. **Domain Randomization**: Increasing dataset diversity for robust training
5. **Quality Assurance**: Validating data quality and consistency
6. **Performance Optimization**: Efficient generation of large-scale datasets

The synthetic data pipeline enables the creation of diverse, perfectly labeled datasets that can be used to train perception systems for humanoid robots. Properly implemented, this pipeline significantly reduces the time and cost of developing AI capabilities while improving the robustness of learned behaviors.

In the next chapter, we'll explore Isaac ROS, NVIDIA's hardware-accelerated perception and navigation stack that works seamlessly with the synthetic data generation pipeline.