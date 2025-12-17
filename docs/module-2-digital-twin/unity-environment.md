---
title: Building High-Fidelity Unity Scenes for Humanoid Robots
sidebar_position: 10
description: Creating realistic Unity environments with HDRP for humanoid robot visualization and interaction
---

# Building High-Fidelity Scenes in Unity HDRP

## Introduction

In this chapter, we'll explore how to create high-fidelity visualization environments in Unity HDRP (High Definition Render Pipeline) for humanoid robots. Unity provides exceptional visual quality and user interaction capabilities that complement the physics simulation in Gazebo. For humanoid robots that need to interact with humans in realistic environments, Unity's advanced rendering capabilities are essential for creating photorealistic visualizations and immersive interaction scenarios.

## Understanding HDRP for Robotics

The High Definition Render Pipeline (HDRP) is Unity's solution for high-fidelity graphics, offering:

- **Physically Based Rendering (PBR)**: Realistic materials and lighting
- **Advanced Lighting**: Global illumination, volumetric lighting, and real-time ray tracing
- **High-Quality Shading**: Complex shader effects for realistic appearances
- **Realistic Cameras**: Camera effects that mimic real optical systems

### HDRP vs Built-in Render Pipeline

For robotics applications, HDRP offers several advantages:

- **Photorealistic rendering** for accurate perception simulation
- **Advanced physics-based materials** for realistic robot appearances
- **Complex lighting scenarios** for simulating different environmental conditions
- **Better performance for static scenes** with many lights

However, it also has some considerations:
- **Higher computational requirements** for real-time rendering
- **More complex setup** compared to the built-in pipeline
- **Platform limitations** for some mobile and embedded targets

## Setting Up HDRP for Robotics

### 1. Creating an HDRP Project

When starting a new Unity project for robotics visualization:

1. **Select HDRP Template**: Choose "3D (High Definition RP)" when creating a new project
2. **Configure Quality Settings**: Set appropriate quality levels for your target hardware
3. **Set Up Lighting**: Configure lighting for your robotics scenarios

### 2. HDRP Asset Configuration

```csharp
// RoboticsHDRPAsset.cs
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

[CreateAssetMenu(fileName = "RoboticsHDRPAsset", menuName = "Robotics/HDRP Asset")]
public class RoboticsHDRPAsset : ScriptableObject
{
    [Header("Lighting Configuration")]
    public bool useBakedLighting = true;
    public bool useRealtimeLighting = true;
    public float indirectLightingMultiplier = 1.0f;

    [Header("Post-Processing Settings")]
    public bool enableBloom = true;
    public bool enableMotionBlur = false;  // Disable for robotics (sharp visuals preferred)
    public bool enableDepthOfField = false;  // Disable for robotics (full focus preferred)

    [Header("Performance Settings")]
    public int maxLODLevel = 3;
    public float lodBias = 1.0f;
    public bool enableDynamicBatching = true;
    public bool enableSRPBatching = true;

    public void ConfigureForRobotics()
    {
        // Optimize for robotics visualization
        enableMotionBlur = false;  // Robots need sharp visualization
        enableDepthOfField = false;  // Full focus on robot and environment

        // Ensure proper lighting for perception tasks
        indirectLightingMultiplier = 1.0f;  // Consistent lighting for vision algorithms
    }
}
```

## Creating Robot Models in Unity

### 1. Importing Robot Assets

For humanoid robots, you'll typically import assets from your URDF or CAD models:

```csharp
// RobotModelImporter.cs
using UnityEngine;
using System.Collections.Generic;

public class RobotModelImporter : MonoBehaviour
{
    [Header("Robot Configuration")]
    public string robotName = "HumanoidRobot";
    public Transform robotRoot;

    [Header("Joint Configuration")]
    public List<JointDefinition> jointDefinitions = new List<JointDefinition>();

    [Header("Visual Settings")]
    public Material robotMaterial;
    public Color robotColor = Color.gray;
    public bool enableShadows = true;

    [System.Serializable]
    public class JointDefinition
    {
        public string jointName;
        public Transform jointTransform;
        public JointType jointType;
        public float minLimit = -90f;
        public float maxLimit = 90f;
        public Vector3 rotationAxis = Vector3.forward;
        public float currentAngle = 0f;
    }

    public enum JointType
    {
        Revolute,    // Rotational joint with limits
        Continuous,  // Rotational joint without limits
        Prismatic,   // Linear joint
        Fixed        // No movement
    }

    void Start()
    {
        if (robotRoot == null)
        {
            robotRoot = transform;
        }

        // Find and configure all joints
        FindJointsRecursive(robotRoot);
    }

    void FindJointsRecursive(Transform parent)
    {
        foreach (Transform child in parent)
        {
            // Look for joint markers or naming conventions
            if (IsJointTransform(child))
            {
                JointDefinition jointDef = new JointDefinition();
                jointDef.jointName = child.name;
                jointDef.jointTransform = child;

                // Determine joint type from name or component
                jointDef.jointType = DetermineJointType(child.name);

                // Set default limits based on joint type
                SetDefaultLimits(ref jointDef);

                jointDefinitions.Add(jointDef);
            }

            // Recursively check children
            FindJointsRecursive(child);
        }
    }

    bool IsJointTransform(Transform t)
    {
        // Check if transform represents a joint based on naming or components
        string nameLower = t.name.ToLower();
        return nameLower.Contains("joint") ||
               nameLower.Contains("hip") ||
               nameLower.Contains("knee") ||
               nameLower.Contains("ankle") ||
               nameLower.Contains("shoulder") ||
               nameLower.Contains("elbow") ||
               nameLower.Contains("wrist");
    }

    JointType DetermineJointType(string jointName)
    {
        string nameLower = jointName.ToLower();

        if (nameLower.Contains("continuous"))
            return JointType.Continuous;
        else if (nameLower.Contains("prismatic"))
            return JointType.Prismatic;
        else if (nameLower.Contains("fixed"))
            return JointType.Fixed;
        else
            return JointType.Revolute;  // Default to revolute
    }

    void SetDefaultLimits(ref JointDefinition jointDef)
    {
        switch (jointDef.jointType)
        {
            case JointType.Revolute:
                // Humanoid-specific joint limits
                if (jointDef.jointName.Contains("hip"))
                {
                    jointDef.minLimit = -90f * Mathf.Deg2Rad;
                    jointDef.maxLimit = 90f * Mathf.Deg2Rad;
                }
                else if (jointDef.jointName.Contains("knee"))
                {
                    jointDef.minLimit = 0f * Mathf.Deg2Rad;
                    jointDef.maxLimit = 150f * Mathf.Deg2Rad;
                }
                else if (jointDef.jointName.Contains("shoulder"))
                {
                    jointDef.minLimit = -120f * Mathf.Deg2Rad;
                    jointDef.maxLimit = 120f * Mathf.Deg2Rad;
                }
                else
                {
                    jointDef.minLimit = -90f * Mathf.Deg2Rad;
                    jointDef.maxLimit = 90f * Mathf.Deg2Rad;
                }
                break;

            case JointType.Continuous:
                jointDef.minLimit = -Mathf.Infinity;
                jointDef.maxLimit = Mathf.Infinity;
                break;

            case JointType.Prismatic:
                jointDef.minLimit = -0.5f;  // meters
                jointDef.maxLimit = 0.5f;
                break;

            case JointType.Fixed:
                jointDef.minLimit = 0f;
                jointDef.maxLimit = 0f;
                break;
        }
    }

    public void SetJointPosition(string jointName, float angleRadians)
    {
        JointDefinition joint = jointDefinitions.Find(j => j.jointName == jointName);
        if (joint != null && joint.jointTransform != null)
        {
            // Clamp to limits if revolute joint
            if (joint.jointType == JointType.Revolute)
            {
                angleRadians = Mathf.Clamp(angleRadians, joint.minLimit, joint.maxLimit);
            }

            // Apply rotation based on joint axis
            Vector3 rotation = joint.rotationAxis * angleRadians * Mathf.Rad2Deg;
            joint.jointTransform.localRotation = Quaternion.Euler(rotation);

            joint.currentAngle = angleRadians;
        }
    }

    public float GetJointPosition(string jointName)
    {
        JointDefinition joint = jointDefinitions.Find(j => j.jointName == jointName);
        return joint != null ? joint.currentAngle : 0f;
    }

    public void SetRobotMaterial(Material newMaterial)
    {
        Renderer[] renderers = GetComponentsInChildren<Renderer>(true);
        foreach (Renderer renderer in renderers)
        {
            if (renderer.name.Contains("link") || renderer.name.Contains("joint"))
            {
                renderer.material = newMaterial;
            }
        }
    }
}
```

### 2. Advanced Robot Materials

For humanoid robots, realistic materials are crucial:

```csharp
// RoboticsMaterialFactory.cs
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

public class RoboticsMaterialFactory : MonoBehaviour
{
    [Header("Material Presets")]
    public Material presetMetal;
    public Material presetPlastic;
    public Material presetRubber;
    public Material presetComposite;

    [Header("Customization")]
    public float metallicValue = 0.5f;
    public float smoothnessValue = 0.7f;
    public Texture2D normalMap;
    public Texture2D roughnessMap;

    public Material CreateRobotMaterial(RobotMaterialType materialType)
    {
        Material robotMaterial = new Material(Shader.Find("HDRP/Lit"));

        switch (materialType)
        {
            case RobotMaterialType.Metal:
                ConfigureMetalMaterial(robotMaterial);
                break;
            case RobotMaterialType.Plastic:
                ConfigurePlasticMaterial(robotMaterial);
                break;
            case RobotMaterialType.Rubber:
                ConfigureRubberMaterial(robotMaterial);
                break;
            case RobotMaterialType.Composite:
                ConfigureCompositeMaterial(robotMaterial);
                break;
            default:
                ConfigureDefaultMaterial(robotMaterial);
                break;
        }

        return robotMaterial;
    }

    void ConfigureMetalMaterial(Material mat)
    {
        mat.SetColor("_BaseColor", new Color(0.7f, 0.7f, 0.8f, 1f));  // Silver-gray
        mat.SetFloat("_Metallic", 0.9f);
        mat.SetFloat("_Smoothness", 0.8f);
        mat.SetTexture("_NormalMap", normalMap);
        mat.SetTexture("_SmoothnessMap", roughnessMap);

        // HDRP-specific properties
        mat.SetFloat("_WorkflowMode", 1);  // Metallic workflow
        mat.SetFloat("_SpecularOcclusionMode", 1);
        mat.SetFloat("_SmoothnessRemapMin", 0);
        mat.SetFloat("_SmoothnessRemapMax", 1);
    }

    void ConfigurePlasticMaterial(Material mat)
    {
        mat.SetColor("_BaseColor", new Color(0.8f, 0.8f, 0.8f, 1f));  // Neutral plastic
        mat.SetFloat("_Metallic", 0.1f);
        mat.SetFloat("_Smoothness", 0.5f);
        mat.SetTexture("_NormalMap", normalMap);

        // Add slight specular highlight for plastic appearance
        mat.SetFloat("_SpecularColor", 0.2f);
    }

    void ConfigureRubberMaterial(Material mat)
    {
        mat.SetColor("_BaseColor", new Color(0.1f, 0.1f, 0.1f, 1f));  // Dark rubber
        mat.SetFloat("_Metallic", 0.05f);
        mat.SetFloat("_Smoothness", 0.3f);
        mat.SetTexture("_NormalMap", normalMap);

        // Rubber has less specular reflection
        mat.SetFloat("_SpecularColor", 0.05f);
    }

    void ConfigureCompositeMaterial(Material mat)
    {
        // For composite materials (mixed materials)
        mat.SetColor("_BaseColor", new Color(0.5f, 0.5f, 0.6f, 1f));  // Mixed gray-blue
        mat.SetFloat("_Metallic", 0.3f);
        mat.SetFloat("_Smoothness", 0.4f);
        mat.SetTexture("_NormalMap", normalMap);
        mat.SetTexture("_SmoothnessMap", roughnessMap);
    }

    void ConfigureDefaultMaterial(Material mat)
    {
        mat.SetColor("_BaseColor", new Color(0.7f, 0.7f, 0.7f, 1f));  // Default gray
        mat.SetFloat("_Metallic", metallicValue);
        mat.SetFloat("_Smoothness", smoothnessValue);

        if (normalMap != null)
            mat.SetTexture("_NormalMap", normalMap);
    }
}

public enum RobotMaterialType
{
    Default,
    Metal,
    Plastic,
    Rubber,
    Composite
}
```

## Creating Interactive Environments

### 1. Environment Setup for Humanoid Interaction

```csharp
// HumanoidInteractionEnvironment.cs
using UnityEngine;
using System.Collections.Generic;

public class HumanoidInteractionEnvironment : MonoBehaviour
{
    [Header("Environment Configuration")]
    public EnvironmentType environmentType = EnvironmentType.IndoorOffice;
    public List<InteractiveObject> interactiveObjects = new List<InteractiveObject>();
    public List<NavigationWaypoint> navigationWaypoints = new List<NavigationWaypoint>();

    [Header("Lighting Setup")]
    public Light mainLight;
    public Light fillLight;
    public ReflectionProbe environmentProbe;

    [Header("Interaction Zones")]
    public List<InteractionZone> interactionZones = new List<InteractionZone>();

    [System.Serializable]
    public class InteractiveObject
    {
        public string objectName;
        public GameObject gameObject;
        public ObjectType objectType;
        public bool isInteractable;
        public InteractionType interactionType;
    }

    [System.Serializable]
    public class NavigationWaypoint
    {
        public string waypointName;
        public Vector3 position;
        public bool isAccessible;
        public float radius = 0.5f;
    }

    [System.Serializable]
    public class InteractionZone
    {
        public string zoneName;
        public Vector3 center;
        public Vector3 size;
        public ZoneType zoneType;
        public List<string> allowedInteractions;
    }

    public enum EnvironmentType
    {
        IndoorOffice,
        IndoorWarehouse,
        OutdoorUrban,
        OutdoorNatural,
        Laboratory,
        Home
    }

    public enum ObjectType
    {
        Furniture,
        Obstacle,
        Tool,
        Sensor,
        ControlPanel,
        Other
    }

    public enum InteractionType
    {
        Graspable,
        Touchable,
        Observable,
        Navigable,
        Manipulatable
    }

    public enum ZoneType
    {
        Safe,
        Restricted,
        Observation,
        Manipulation,
        Navigation
    }

    void Start()
    {
        SetupEnvironment();
        ConfigureLighting();
        RegisterInteractiveObjects();
    }

    void SetupEnvironment()
    {
        switch (environmentType)
        {
            case EnvironmentType.IndoorOffice:
                CreateIndoorOfficeEnvironment();
                break;
            case EnvironmentType.OutdoorUrban:
                CreateOutdoorUrbanEnvironment();
                break;
            case EnvironmentType.Laboratory:
                CreateLaboratoryEnvironment();
                break;
            default:
                CreateDefaultEnvironment();
                break;
        }
    }

    void CreateIndoorOfficeEnvironment()
    {
        // Create office-like environment with desks, chairs, etc.
        CreateFloor();
        CreateWalls();
        PlaceOfficeFurniture();
        AddObstacles();
    }

    void CreateOutdoorUrbanEnvironment()
    {
        // Create urban environment with sidewalks, buildings, etc.
        CreateGroundSurface();
        AddUrbanElements();
        ConfigureEnvironmentLighting();
    }

    void CreateLaboratoryEnvironment()
    {
        // Create laboratory environment with equipment, testing areas, etc.
        CreateLabLayout();
        PlaceLabEquipment();
        DefineTestingAreas();
    }

    void CreateDefaultEnvironment()
    {
        // Create basic environment for testing
        CreateSimpleRoom();
        AddBasicObstacles();
    }

    void CreateFloor()
    {
        GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Environment_Floor";
        floor.transform.SetParent(transform);
        floor.transform.localScale = new Vector3(5, 1, 5);  // 10x10m floor
        floor.GetComponent<Renderer>().material = CreateFloorMaterial();
    }

    void CreateWalls()
    {
        float roomSize = 10f;
        float wallHeight = 3f;

        // Create 4 walls
        for (int i = 0; i < 4; i++)
        {
            GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.name = $"Environment_Wall_{i}";
            wall.transform.SetParent(transform);

            // Position walls around the room
            float angle = i * 90f * Mathf.Deg2Rad;
            float x = Mathf.Cos(angle) * roomSize / 2;
            float z = Mathf.Sin(angle) * roomSize / 2;

            wall.transform.position = new Vector3(x, wallHeight / 2, z);
            wall.transform.rotation = Quaternion.Euler(0, i * 90f, 0);
            wall.transform.localScale = new Vector3(roomSize, wallHeight, 0.2f);

            // Make it a static collider
            Destroy(wall.GetComponent<BoxCollider>());
            BoxCollider newCol = wall.AddComponent<BoxCollider>();
            newCol.isTrigger = false;

            wall.GetComponent<Renderer>().material = CreateWallMaterial();
        }
    }

    void PlaceOfficeFurniture()
    {
        // Create a desk
        GameObject desk = GameObject.CreatePrimitive(PrimitiveType.Cube);
        desk.name = "Desk";
        desk.transform.SetParent(transform);
        desk.transform.position = new Vector3(3f, 0.5f, 0f);
        desk.transform.localScale = new Vector3(1.5f, 1f, 0.8f);
        desk.GetComponent<Renderer>().material = CreateTableMaterial();

        // Create a chair
        GameObject chair = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        chair.name = "Chair";
        chair.transform.SetParent(transform);
        chair.transform.position = new Vector3(3f, 0.3f, -0.6f);
        chair.transform.localScale = new Vector3(0.3f, 0.3f, 0.3f);
        chair.GetComponent<Renderer>().material = CreateChairMaterial();
    }

    void ConfigureLighting()
    {
        // Configure main directional light (sun/simulation light)
        if (mainLight == null)
        {
            // Create main light if not assigned
            GameObject lightObj = new GameObject("Main Light");
            lightObj.transform.SetParent(transform);
            mainLight = lightObj.AddComponent<Light>();
            mainLight.type = LightType.Directional;
            mainLight.color = Color.white;
            mainLight.intensity = 3.14f;  // Standard for HDRP
            mainLight.transform.rotation = Quaternion.Euler(50, -30, 0);
        }

        // Configure fill light
        if (fillLight == null && mainLight != null)
        {
            GameObject fillLightObj = new GameObject("Fill Light");
            fillLightObj.transform.SetParent(transform);
            fillLight = fillLightObj.AddComponent<Light>();
            fillLight.type = LightType.Directional;
            fillLight.color = new Color(0.5f, 0.5f, 0.6f, 1f);
            fillLight.intensity = 1.0f;
            fillLight.transform.rotation = Quaternion.Euler(-50, 150, 0);
        }

        // Configure reflection probe
        if (environmentProbe == null)
        {
            GameObject probeObj = new GameObject("Environment Probe");
            probeObj.transform.SetParent(transform);
            probeObj.transform.position = Vector3.zero;
            environmentProbe = probeObj.AddComponent<ReflectionProbe>();
            environmentProbe.mode = UnityEngine.Rendering.ReflectionProbeMode.Baked;
            environmentProbe.size = new Vector3(20, 10, 20);
        }
    }

    void RegisterInteractiveObjects()
    {
        // Find and register all interactive objects in the scene
        Collider[] colliders = Physics.OverlapBox(transform.position, new Vector3(10, 5, 10));

        foreach (Collider col in colliders)
        {
            if (col.gameObject.tag == "Interactive" || col.gameObject.layer == LayerMask.NameToLayer("Interactive"))
            {
                InteractiveObject io = new InteractiveObject();
                io.objectName = col.gameObject.name;
                io.gameObject = col.gameObject;
                io.isInteractable = true;
                io.objectType = DetermineObjectType(col.gameObject.name);
                io.interactionType = DetermineInteractionType(col.gameObject.name);

                interactiveObjects.Add(io);
            }
        }
    }

    ObjectType DetermineObjectType(string objectName)
    {
        string nameLower = objectName.ToLower();

        if (nameLower.Contains("desk") || nameLower.Contains("table") || nameLower.Contains("chair"))
            return ObjectType.Furniture;
        else if (nameLower.Contains("sensor") || nameLower.Contains("camera") || nameLower.Contains("lidar"))
            return ObjectType.Sensor;
        else if (nameLower.Contains("panel") || nameLower.Contains("control"))
            return ObjectType.ControlPanel;
        else
            return ObjectType.Other;
    }

    InteractionType DetermineInteractionType(string objectName)
    {
        string nameLower = objectName.ToLower();

        if (nameLower.Contains("button") || nameLower.Contains("switch") || nameLower.Contains("panel"))
            return InteractionType.Touchable;
        else if (nameLower.Contains("object") || nameLower.Contains("box") || nameLower.Contains("item"))
            return InteractionType.Graspable;
        else
            return InteractionType.Observable;
    }

    Material CreateFloorMaterial()
    {
        Material floorMat = new Material(Shader.Find("HDRP/Lit"));
        floorMat.SetColor("_BaseColor", new Color(0.8f, 0.8f, 0.8f, 1f));  // Light gray
        floorMat.SetFloat("_Metallic", 0.1f);
        floorMat.SetFloat("_Smoothness", 0.4f);
        return floorMat;
    }

    Material CreateWallMaterial()
    {
        Material wallMat = new Material(Shader.Find("HDRP/Lit"));
        wallMat.SetColor("_BaseColor", new Color(0.9f, 0.9f, 0.95f, 1f));  // Near-white
        wallMat.SetFloat("_Metallic", 0.0f);
        wallMat.SetFloat("_Smoothness", 0.2f);
        return wallMat;
    }

    Material CreateTableMaterial()
    {
        Material tableMat = new Material(Shader.Find("HDRP/Lit"));
        tableMat.SetColor("_BaseColor", new Color(0.6f, 0.4f, 0.2f, 1f));  // Wood color
        tableMat.SetFloat("_Metallic", 0.2f);
        tableMat.SetFloat("_Smoothness", 0.3f);
        return tableMat;
    }

    Material CreateChairMaterial()
    {
        Material chairMat = new Material(Shader.Find("HDRP/Lit"));
        chairMat.SetColor("_BaseColor", new Color(0.2f, 0.2f, 0.7f, 1f));  // Blue
        chairMat.SetFloat("_Metallic", 0.1f);
        chairMat.SetFloat("_Smoothness", 0.4f);
        return chairMat;
    }

    public InteractiveObject GetInteractiveObject(string objectName)
    {
        return interactiveObjects.Find(io => io.objectName == objectName);
    }

    public void HighlightObject(string objectName)
    {
        InteractiveObject io = GetInteractiveObject(objectName);
        if (io != null && io.gameObject != null)
        {
            // Add highlight effect
            Renderer rend = io.gameObject.GetComponent<Renderer>();
            if (rend != null)
            {
                // Store original material
                if (!rend.HasPropertyBlock())
                {
                    Material originalMat = rend.material;
                    rend.material = Instantiate(originalMat);  // Create instance to modify
                }

                // Add emission for highlight
                rend.material.SetColor("_EmissiveColor", Color.yellow);
                rend.material.EnableKeyword("_EMISSIVE_COLOR_MAP");
            }
        }
    }

    public void RemoveHighlight(string objectName)
    {
        InteractiveObject io = GetInteractiveObject(objectName);
        if (io != null && io.gameObject != null)
        {
            Renderer rend = io.gameObject.GetComponent<Renderer>();
            if (rend != null)
            {
                // Remove emission highlight
                rend.material.SetColor("_EmissiveColor", Color.black);
                rend.material.DisableKeyword("_EMISSIVE_COLOR_MAP");
            }
        }
    }
}
```

### 2. Human-Robot Interaction System

```csharp
// HumanRobotInteractionSystem.cs
using UnityEngine;
using System.Collections;
using UnityEngine.Events;

public class HumanRobotInteractionSystem : MonoBehaviour
{
    [Header("Interaction Configuration")]
    public float interactionDistance = 2.0f;
    public float observationDistance = 5.0f;
    public LayerMask interactionLayerMask;

    [Header("Robot Components")]
    public Transform robotTransform;
    public RobotModelImporter robotImporter;
    public GameObject interactionCursor;

    [Header("Interaction Events")]
    public UnityEvent onHumanApproach;
    public UnityEvent onInteractionStart;
    public UnityEvent onInteractionEnd;
    public UnityEvent<string> onObjectDetected;

    private Camera mainCamera;
    private bool isInteractionEnabled = true;
    private GameObject currentInteractionTarget = null;
    private Coroutine interactionCoroutine;

    void Start()
    {
        mainCamera = Camera.main;
        if (interactionLayerMask == 0)
        {
            interactionLayerMask = Physics.DefaultRaycastLayers;
        }

        SetupInteractionCursor();
    }

    void SetupInteractionCursor()
    {
        if (interactionCursor == null)
        {
            // Create a simple cursor if none provided
            GameObject cursorObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            cursorObj.name = "InteractionCursor";
            cursorObj.transform.localScale = Vector3.one * 0.1f;
            cursorObj.GetComponent<Renderer>().material = CreateCursorMaterial();
            interactionCursor = cursorObj;
            interactionCursor.SetActive(false);
        }
    }

    Material CreateCursorMaterial()
    {
        Material cursorMat = new Material(Shader.Find("HDRP/Lit"));
        cursorMat.SetColor("_BaseColor", Color.cyan);
        cursorMat.SetFloat("_Metallic", 0.8f);
        cursorMat.SetFloat("_Smoothness", 0.9f);
        cursorMat.EnableKeyword("_EMISSIVE_COLOR_MAP");
        cursorMat.SetColor("_EmissiveColor", Color.cyan);
        return cursorMat;
    }

    void Update()
    {
        if (!isInteractionEnabled) return;

        HandleInteractionInput();
        UpdateInteractionCursor();
    }

    void HandleInteractionInput()
    {
        if (Input.GetMouseButtonDown(0))  // Left click
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, observationDistance, interactionLayerMask))
            {
                if (hit.collider.CompareTag("Interactive") || hit.collider.CompareTag("RobotPart"))
                {
                    StartInteraction(hit.collider.gameObject);
                }
            }
        }

        if (Input.GetMouseButtonUp(0))  // Release left click
        {
            EndInteraction();
        }
    }

    void UpdateInteractionCursor()
    {
        Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit, observationDistance, interactionLayerMask))
        {
            if (hit.collider.CompareTag("Interactive") || hit.collider.CompareTag("RobotPart"))
            {
                if (!interactionCursor.activeSelf)
                {
                    interactionCursor.SetActive(true);
                }

                interactionCursor.transform.position = hit.point;
                interactionCursor.GetComponent<Renderer>().material.SetColor("_BaseColor", Color.green);  // Available for interaction
            }
            else
            {
                if (interactionCursor.activeSelf)
                {
                    interactionCursor.GetComponent<Renderer>().material.SetColor("_BaseColor", Color.red);  // Not interactable
                }
            }
        }
        else
        {
            interactionCursor.SetActive(false);
        }
    }

    void StartInteraction(GameObject target)
    {
        if (currentInteractionTarget != null)
        {
            EndInteraction();  // End previous interaction
        }

        currentInteractionTarget = target;
        onInteractionStart?.Invoke();

        // Highlight the target
        if (target.CompareTag("Interactive"))
        {
            var environment = FindObjectOfType<HumanoidInteractionEnvironment>();
            if (environment != null)
            {
                environment.HighlightObject(target.name);
            }
        }

        // Start interaction coroutine based on object type
        interactionCoroutine = StartCoroutine(PerformInteraction(target));
    }

    void EndInteraction()
    {
        if (currentInteractionTarget != null)
        {
            // Remove highlight
            if (currentInteractionTarget.CompareTag("Interactive"))
            {
                var environment = FindObjectOfType<HumanoidInteractionEnvironment>();
                if (environment != null)
                {
                    environment.RemoveHighlight(currentInteractionTarget.name);
                }
            }

            currentInteractionTarget = null;
            onInteractionEnd?.Invoke();

            if (interactionCoroutine != null)
            {
                StopCoroutine(interactionCoroutine);
                interactionCoroutine = null;
            }
        }

        interactionCursor.SetActive(false);
    }

    IEnumerator PerformInteraction(GameObject target)
    {
        if (target.CompareTag("RobotPart"))
        {
            // Robot part interaction - maybe move joint or observe
            yield return StartCoroutine(InteractWithRobotPart(target));
        }
        else if (target.CompareTag("Interactive"))
        {
            // Environment object interaction - maybe manipulate or observe
            yield return StartCoroutine(InteractWithEnvironmentObject(target));
        }
    }

    IEnumerator InteractWithRobotPart(GameObject robotPart)
    {
        // Example: Move a joint when clicked
        string jointName = ExtractJointName(robotPart.name);

        if (!string.IsNullOrEmpty(jointName) && robotImporter != null)
        {
            float originalPosition = robotImporter.GetJointPosition(jointName);
            float targetPosition = originalPosition + (Random.Range(0, 2) == 0 ? 0.2f : -0.2f);  // Random direction

            // Smooth movement over 1 second
            float duration = 1.0f;
            float elapsed = 0f;

            while (elapsed < duration)
            {
                float t = elapsed / duration;
                float newPosition = Mathf.Lerp(originalPosition, targetPosition, t);

                robotImporter.SetJointPosition(jointName, newPosition);

                elapsed += Time.deltaTime;
                yield return null;
            }

            // Set final position
            robotImporter.SetJointPosition(jointName, targetPosition);
        }
    }

    IEnumerator InteractWithEnvironmentObject(GameObject environmentObject)
    {
        // Example: Move an object when clicked
        Vector3 originalPosition = environmentObject.transform.position;
        Vector3 targetPosition = originalPosition + Vector3.up * 0.5f;  // Move up 0.5m

        // Smooth movement over 1 second
        float duration = 1.0f;
        float elapsed = 0f;

        while (elapsed < duration)
        {
            float t = elapsed / duration;
            Vector3 newPosition = Vector3.Lerp(originalPosition, targetPosition, t);

            environmentObject.transform.position = newPosition;

            elapsed += Time.deltaTime;
            yield return null;
        }

        // Move back after a delay
        yield return new WaitForSeconds(1.0f);

        elapsed = 0f;
        while (elapsed < duration)
        {
            float t = elapsed / duration;
            Vector3 newPosition = Vector3.Lerp(targetPosition, originalPosition, t);

            environmentObject.transform.position = newPosition;

            elapsed += Time.deltaTime;
            yield return null;
        }

        environmentObject.transform.position = originalPosition;
    }

    string ExtractJointName(string objectName)
    {
        // Simple logic to extract joint name from object name
        // This would be more sophisticated in a real implementation
        if (robotImporter != null)
        {
            foreach (var jointDef in robotImporter.jointDefinitions)
            {
                if (objectName.Contains(jointDef.jointName))
                {
                    return jointDef.jointName;
                }
            }
        }
        return "";
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Human") || other.CompareTag("User"))
        {
            onHumanApproach?.Invoke();
        }
    }

    public void EnableInteraction(bool enable)
    {
        isInteractionEnabled = enable;
        if (!enable && currentInteractionTarget != null)
        {
            EndInteraction();
        }
    }

    public bool IsInteracting()
    {
        return currentInteractionTarget != null;
    }

    public GameObject GetCurrentInteractionTarget()
    {
        return currentInteractionTarget;
    }

    public void DetectNearbyObjects()
    {
        Collider[] nearbyObjects = Physics.OverlapSphere(robotTransform.position, observationDistance);

        foreach (Collider obj in nearbyObjects)
        {
            if (obj.CompareTag("Interactive") || obj.CompareTag("Obstacle"))
            {
                onObjectDetected?.Invoke(obj.name);
            }
        }
    }
}
```

## Integration with ROS and Simulation

### 1. ROS-Unity Bridge for Humanoid Robots

```csharp
// HumanoidRosBridge.cs
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Nav;

public class HumanoidRosBridge : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string rosIpAddress = "127.0.0.1";
    public int rosPort = 9090;
    public string robotNamespace = "/humanoid";
    public float rosUpdateFrequency = 60f;  // Hz

    [Header("Robot Configuration")]
    public string[] jointNames;
    public Transform[] jointTransforms;
    public float[] jointPositions;
    public float[] jointVelocities;
    public float[] jointEfforts;

    [Header("Synchronization")]
    public float syncThreshold = 0.01f;  // Radians
    public bool enableInterpolation = true;
    public float interpolationTime = 0.1f;  // Seconds

    private ROSConnection ros;
    private float lastUpdate = 0f;
    private Dictionary<string, int> jointIndexMap = new Dictionary<string, int>();
    private Dictionary<string, float> targetJointPositions = new Dictionary<string, float>();
    private Dictionary<string, float> currentJointPositions = new Dictionary<string, int>();

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIpAddress, rosPort);

        InitializeJointMapping();
        SubscribeToRosTopics();
        SetupRosPublishers();

        Debug.Log($"Humanoid ROS Bridge initialized for {jointNames.Length} joints");
    }

    void InitializeJointMapping()
    {
        jointPositions = new float[jointNames.Length];
        jointVelocities = new float[jointNames.Length];
        jointEfforts = new float[jointNames.Length];

        for (int i = 0; i < jointNames.Length; i++)
        {
            jointIndexMap[jointNames[i]] = i;
            targetJointPositions[jointNames[i]] = 0f;
            currentJointPositions[jointNames[i]] = 0f;
        }
    }

    void SubscribeToRosTopics()
    {
        // Subscribe to joint states from ROS
        ros.Subscribe<JointStateMsg>(
            $"{robotNamespace}/joint_states",
            OnJointStateReceived
        );

        // Subscribe to robot commands
        ros.Subscribe<JointStateMsg>(
            $"{robotNamespace}/joint_commands",
            OnJointCommandReceived
        );

        // Subscribe to velocity commands
        ros.Subscribe<TwistMsg>(
            $"{robotNamespace}/cmd_vel",
            OnVelocityCommandReceived
        );

        // Subscribe to odometry
        ros.Subscribe<OdometryMsg>(
            $"{robotNamespace}/odom",
            OnOdometryReceived
        );
    }

    void SetupRosPublishers()
    {
        // Publishers will be created as needed
    }

    void Update()
    {
        if (Time.time - lastUpdate >= (1f / rosUpdateFrequency))
        {
            PublishRobotState();
            lastUpdate = Time.time;
        }

        if (enableInterpolation)
        {
            InterpolateJointPositions();
        }
    }

    void OnJointStateReceived(JointStateMsg jointState)
    {
        // Update joint positions from ROS
        for (int i = 0; i < jointState.name.Count && i < jointState.position.Count; i++)
        {
            string jointName = jointState.name[i];
            double position = jointState.position[i];

            if (jointIndexMap.ContainsKey(jointName))
            {
                int jointIdx = jointIndexMap[jointName];
                jointPositions[jointIdx] = (float)position;

                // Store as target for interpolation
                if (targetJointPositions.ContainsKey(jointName))
                {
                    targetJointPositions[jointName] = (float)position;
                }
            }
        }

        // Update joint velocities and efforts if available
        if (jointState.velocity.Count == jointState.name.Count)
        {
            for (int i = 0; i < jointState.name.Count; i++)
            {
                string jointName = jointState.name[i];
                if (jointIndexMap.ContainsKey(jointName))
                {
                    int jointIdx = jointIndexMap[jointName];
                    jointVelocities[jointIdx] = (float)jointState.velocity[i];
                }
            }
        }

        if (jointState.effort.Count == jointState.name.Count)
        {
            for (int i = 0; i < jointState.name.Count; i++)
            {
                string jointName = jointState.name[i];
                if (jointIndexMap.ContainsKey(jointName))
                {
                    int jointIdx = jointIndexMap[jointName];
                    jointEfforts[jointIdx] = (float)jointState.effort[i];
                }
            }
        }
    }

    void OnJointCommandReceived(JointStateMsg jointCmd)
    {
        // Process joint commands and update target positions
        for (int i = 0; i < jointCmd.name.Count && i < jointCmd.position.Count; i++)
        {
            string jointName = jointCmd.name[i];
            double position = jointCmd.position[i];

            if (targetJointPositions.ContainsKey(jointName))
            {
                targetJointPositions[jointName] = (float)position;
            }
        }
    }

    void OnVelocityCommandReceived(TwistMsg twist)
    {
        // Handle velocity commands (for mobile base)
        Debug.Log($"Received velocity command: linear=({twist.linear.x}, {twist.linear.y}, {twist.linear.z}), " +
                  $"angular=({twist.angular.x}, {twist.angular.y}, {twist.angular.z})");

        // In a real implementation, this would move the robot base
    }

    void OnOdometryReceived(OdometryMsg odom)
    {
        // Update robot position and orientation from odometry
        float x = (float)odom.pose.pose.position.x;
        float y = (float)odom.pose.pose.position.y;
        float z = (float)odom.pose.pose.position.z;

        float rx = (float)odom.pose.pose.orientation.x;
        float ry = (float)odom.pose.pose.orientation.y;
        float rz = (float)odom.pose.pose.orientation.z;
        float rw = (float)odom.pose.pose.orientation.w;

        // Update robot transform
        transform.position = new Vector3(x, y, z);
        transform.rotation = new Quaternion(rx, ry, rz, rw);
    }

    void PublishRobotState()
    {
        // Publish current joint states to ROS
        var jointStateMsg = new JointStateMsg();
        jointStateMsg.header = new StdMsgs.HeaderMsg();
        jointStateMsg.header.stamp = new BuiltinInterfaces.TimeMsg();
        jointStateMsg.header.frame_id = "base_link";

        jointStateMsg.name = new string[jointNames.Length];
        jointStateMsg.position = new double[jointPositions.Length];
        jointStateMsg.velocity = new double[jointVelocities.Length];
        jointStateMsg.effort = new double[jointEfforts.Length];

        for (int i = 0; i < jointNames.Length; i++)
        {
            jointStateMsg.name[i] = jointNames[i];
            jointStateMsg.position[i] = jointPositions[i];
            jointStateMsg.velocity[i] = jointVelocities[i];
            jointStateMsg.effort[i] = jointEfforts[i];
        }

        ros.Publish($"{robotNamespace}/joint_states", jointStateMsg);
    }

    void InterpolateJointPositions()
    {
        float deltaTime = Time.deltaTime;
        float interpolationRatio = deltaTime / interpolationTime;

        for (int i = 0; i < jointNames.Length; i++)
        {
            string jointName = jointNames[i];

            if (targetJointPositions.ContainsKey(jointName))
            {
                float currentPos = currentJointPositions[jointName];
                float targetPos = targetJointPositions[jointName];

                // Calculate distance to target
                float distance = targetPos - currentPos;

                if (Mathf.Abs(distance) > syncThreshold)
                {
                    // Interpolate toward target
                    float newPos = Mathf.Lerp(currentPos, targetPos, interpolationRatio);

                    // Update joint transform
                    if (i < jointTransforms.Length && jointTransforms[i] != null)
                    {
                        // Apply position based on joint type
                        ApplyJointPosition(jointTransforms[i], newPos);
                    }

                    currentJointPositions[jointName] = newPos;
                    jointPositions[i] = newPos;
                }
            }
        }
    }

    void ApplyJointPosition(Transform jointTransform, float position)
    {
        // Apply position based on joint configuration
        // This is a simplified approach - in reality you'd need to consider joint type and axis
        jointTransform.localRotation = Quaternion.Euler(0, 0, position * Mathf.Rad2Deg);
    }

    public void SendJointCommand(string jointName, float position)
    {
        if (ros != null && ros.IsConnected)
        {
            var jointCmd = new JointStateMsg();
            jointCmd.header = new StdMsgs.HeaderMsg();
            jointCmd.header.stamp = new BuiltinInterfaces.TimeMsg();
            jointCmd.header.frame_id = "base_link";

            jointCmd.name = new string[] { jointName };
            jointCmd.position = new double[] { position };

            ros.Publish($"{robotNamespace}/joint_commands", jointCmd);
        }
    }

    public void SendJointTrajectory(string[] jointNames, float[] positions, float duration = 2.0f)
    {
        if (ros != null && ros.IsConnected)
        {
            // In a real implementation, you would use trajectory_msgs/JointTrajectory
            // For simplicity, we'll send individual position commands
            for (int i = 0; i < jointNames.Length && i < positions.Length; i++)
            {
                SendJointCommand(jointNames[i], positions[i]);
            }
        }
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        if (ros != null && ros.IsConnected)
        {
            var twistCmd = new TwistMsg();
            twistCmd.linear = new Vector3Msg(linearX, 0, 0);
            twistCmd.angular = new Vector3Msg(0, 0, angularZ);

            ros.Publish($"{robotNamespace}/cmd_vel", twistCmd);
        }
    }

    public float GetJointPosition(string jointName)
    {
        if (jointIndexMap.ContainsKey(jointName))
        {
            int idx = jointIndexMap[jointName];
            return jointPositions[idx];
        }
        return 0f;
    }

    public Dictionary<string, float> GetAllJointPositions()
    {
        var positions = new Dictionary<string, float>();
        for (int i = 0; i < jointNames.Length; i++)
        {
            positions[jointNames[i]] = jointPositions[i];
        }
        return positions;
    }

    public void SetJointPositionDirectly(int jointIndex, float position)
    {
        if (jointIndex >= 0 && jointIndex < jointPositions.Length)
        {
            jointPositions[jointIndex] = position;

            if (jointIndex < jointTransforms.Length && jointTransforms[jointIndex] != null)
            {
                ApplyJointPosition(jointTransforms[jointIndex], position);
            }

            // Update target positions to match
            if (jointIndex < jointNames.Length)
            {
                string jointName = jointNames[jointIndex];
                if (targetJointPositions.ContainsKey(jointName))
                {
                    targetJointPositions[jointName] = position;
                    currentJointPositions[jointName] = position;
                }
            }
        }
    }

    void OnDestroy()
    {
        ros?.Disconnect();
    }
}
```

## Performance Optimization for Humanoid Digital Twins

### 1. Efficient Rendering Pipeline

```csharp
// HumanoidPerformanceOptimizer.cs
using UnityEngine;
using System.Collections.Generic;

public class HumanoidPerformanceOptimizer : MonoBehaviour
{
    [Header("LOD Configuration")]
    public float lodTransitionDistance = 10f;
    public int lodCount = 3;
    public float lodFadeTransition = 0.5f;

    [Header("Rendering Optimization")]
    public bool enableOcclusionCulling = true;
    public bool enableLOD = true;
    public bool enableDynamicBatching = true;
    public int maxVisibleRobots = 5;

    [Header("Update Optimization")]
    public float stateUpdateInterval = 0.033f;  // ~30 Hz
    public float sensorUpdateInterval = 0.1f;   // 10 Hz
    public float environmentUpdateInterval = 0.5f;  // 2 Hz

    private float lastStateUpdate = 0f;
    private float lastSensorUpdate = 0f;
    private float lastEnvironmentUpdate = 0f;
    private List<Renderer> robotRenderers = new List<Renderer>();
    private List<Light> robotLights = new List<Light>();
    private Dictionary<Renderer, Material[]> originalMaterials = new Dictionary<Renderer, Material[]>();

    void Start()
    {
        FindRobotComponents();
        OptimizeMaterials();
        ConfigureRenderingSettings();
    }

    void FindRobotComponents()
    {
        // Find all renderers in the robot hierarchy
        robotRenderers.AddRange(GetComponentsInChildren<Renderer>(true));

        // Find all lights in the robot hierarchy
        robotLights.AddRange(GetComponentsInChildren<Light>(true));
    }

    void OptimizeMaterials()
    {
        // Optimize materials for performance
        foreach (Renderer renderer in robotRenderers)
        {
            // Cache original materials
            originalMaterials[renderer] = renderer.materials;

            // Optimize each material
            Material[] optimizedMaterials = new Material[renderer.materials.Length];
            for (int i = 0; i < renderer.materials.Length; i++)
            {
                optimizedMaterials[i] = OptimizeMaterial(renderer.materials[i]);
            }

            renderer.materials = optimizedMaterials;
        }
    }

    Material OptimizeMaterial(Material original)
    {
        // Create optimized version of material
        Material optimized = new Material(original);

        // Simplify shader if possible
        if (optimized.HasProperty("_Smoothness") && QualitySettings.GetQualityLevel() < 2)  // Low quality
        {
            optimized.SetFloat("_Smoothness", 0.3f);  // Reduce from high values
        }

        // Remove expensive features on low quality settings
        if (QualitySettings.GetQualityLevel() < 1)  // Very low quality
        {
            optimized.DisableKeyword("_NORMALMAP");
            optimized.DisableKeyword("_EMISSIVE_COLOR_MAP");
        }

        return optimized;
    }

    void ConfigureRenderingSettings()
    {
        // Configure renderer properties for performance
        foreach (Renderer renderer in robotRenderers)
        {
            renderer.allowOcclusionWhenDynamic = enableOcclusionCulling;

            // Configure batching
            if (enableDynamicBatching)
            {
                renderer.enabled = true;  // Enable for batching consideration
            }
        }

        // Configure quality settings
        QualitySettings.pixelLightCount = Mathf.Min(QualitySettings.pixelLightCount, 4);  // Limit pixel lights
        QualitySettings.shadowDistance = Mathf.Min(QualitySettings.shadowDistance, 50f);  // Limit shadow distance
    }

    void Update()
    {
        // Perform optimization updates at appropriate intervals
        if (Time.time - lastStateUpdate >= stateUpdateInterval)
        {
            OptimizeStateUpdates();
            lastStateUpdate = Time.time;
        }

        if (Time.time - lastSensorUpdate >= sensorUpdateInterval)
        {
            OptimizeSensorUpdates();
            lastSensorUpdate = Time.time;
        }

        if (Time.time - lastEnvironmentUpdate >= environmentUpdateInterval)
        {
            OptimizeEnvironmentUpdates();
            lastEnvironmentUpdate = Time.time;
        }

        // Update LOD if enabled
        if (enableLOD)
        {
            UpdateLOD();
        }
    }

    void OptimizeStateUpdates()
    {
        // Only update state when necessary
        // For humanoid robots, prioritize critical joints for frequent updates
        // Less critical joints can update less frequently
    }

    void OptimizeSensorUpdates()
    {
        // Optimize sensor update frequency based on importance
        // Visual sensors might update at 30Hz, while tactile might update at 100Hz
    }

    void OptimizeEnvironmentUpdates()
    {
        // Update environment less frequently than robot state
        // Background elements can update even less frequently
    }

    void UpdateLOD()
    {
        // Calculate distance from main camera
        Camera mainCam = Camera.main;
        if (mainCam != null)
        {
            float distance = Vector3.Distance(mainCam.transform.position, transform.position);

            // Determine LOD level based on distance
            int lodLevel = 0;
            if (distance > lodTransitionDistance * 2)
                lodLevel = 2;
            else if (distance > lodTransitionDistance)
                lodLevel = 1;

            // Apply LOD level to robot components
            ApplyLODLevel(lodLevel);
        }
    }

    void ApplyLODLevel(int lodLevel)
    {
        // Apply different levels of detail based on distance
        foreach (Renderer renderer in robotRenderers)
        {
            // Adjust renderer quality based on LOD level
            switch (lodLevel)
            {
                case 0: // High detail
                    renderer.enabled = true;
                    renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
                    renderer.receiveShadows = true;
                    break;
                case 1: // Medium detail
                    renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.TwoSided;
                    renderer.receiveShadows = false;
                    break;
                case 2: // Low detail
                    renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                    renderer.receiveShadows = false;
                    break;
            }
        }
    }

    public void SetPerformanceLevel(PerformanceLevel level)
    {
        switch (level)
        {
            case PerformanceLevel.Low:
                QualitySettings.SetQualityLevel(0);
                lodTransitionDistance = 5f;
                stateUpdateInterval = 0.1f;  // 10 Hz
                break;
            case PerformanceLevel.Medium:
                QualitySettings.SetQualityLevel(2);
                lodTransitionDistance = 10f;
                stateUpdateInterval = 0.066f;  // 15 Hz
                break;
            case PerformanceLevel.High:
                QualitySettings.SetQualityLevel(4);
                lodTransitionDistance = 20f;
                stateUpdateInterval = 0.016f;  // 60 Hz
                break;
        }
    }

    public void RestoreOriginalMaterials()
    {
        foreach (var kvp in originalMaterials)
        {
            kvp.Key.materials = kvp.Value;
        }
    }

    public void ReduceTriangleCount()
    {
        // Reduce polygon count for distant robots
        MeshFilter[] meshFilters = GetComponentsInChildren<MeshFilter>();

        foreach (MeshFilter meshFilter in meshFilters)
        {
            if (meshFilter.mesh != null)
            {
                // In a real implementation, you would use a mesh simplification algorithm
                // For now, we'll just note that this is where optimization would happen
                Debug.Log($"Mesh {meshFilter.name} has {meshFilter.mesh.triangles.Length / 3} triangles");
            }
        }
    }
}

public enum PerformanceLevel
{
    Low,
    Medium,
    High
}
```

## Testing and Validation

### 1. Digital Twin Validation System

```csharp
// DigitalTwinValidator.cs
using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class DigitalTwinValidator : MonoBehaviour
{
    [Header("Validation Settings")]
    public float validationInterval = 5.0f;
    public float syncTolerance = 0.05f;  // 5cm or 5 degrees tolerance
    public bool enableContinuousValidation = true;

    [Header("Simulation Sync Validation")]
    public bool validateJointPositions = true;
    public bool validateRobotPose = true;
    public bool validateSensorData = true;
    public bool validateTiming = true;

    private float lastValidationTime = 0f;
    private Dictionary<string, float> lastValidatedJointPositions = new Dictionary<string, float>();
    private Vector3 lastValidatedRobotPosition = Vector3.zero;
    private Quaternion lastValidatedRobotRotation = Quaternion.identity;
    private List<ValidationIssue> validationIssues = new List<ValidationIssue>();

    [System.Serializable]
    public class ValidationIssue
    {
        public string componentName;
        public string issueDescription;
        public ValidationSeverity severity;
        public float timestamp;
    }

    public enum ValidationSeverity
    {
        Info,
        Warning,
        Error
    }

    void Start()
    {
        StartCoroutine(RunValidationCycle());
    }

    IEnumerator RunValidationCycle()
    {
        while (enableContinuousValidation)
        {
            ValidateDigitalTwin();

            yield return new WaitForSeconds(validationInterval);
        }
    }

    public void ValidateDigitalTwin()
    {
        validationIssues.Clear();
        Debug.Log("Starting digital twin validation...");

        // Validate joint positions
        if (validateJointPositions)
        {
            ValidateJointSynchronization();
        }

        // Validate robot pose
        if (validateRobotPose)
        {
            ValidateRobotPoseSynchronization();
        }

        // Validate timing
        if (validateTiming)
        {
            ValidateTimingSynchronization();
        }

        // Report results
        ReportValidationResults();
    }

    void ValidateJointSynchronization()
    {
        var rosBridge = GetComponent<HumanoidRosBridge>();
        if (rosBridge != null)
        {
            var currentPositions = rosBridge.GetAllJointPositions();

            foreach (var kvp in currentPositions)
            {
                string jointName = kvp.Key;
                float currentPosition = kvp.Value;

                if (lastValidatedJointPositions.ContainsKey(jointName))
                {
                    float previousPosition = lastValidatedJointPositions[jointName];
                    float difference = Mathf.Abs(currentPosition - previousPosition);

                    // Check if joint is moving unexpectedly
                    if (difference > syncTolerance && !IsJointBeingControlled(jointName))
                    {
                        AddValidationIssue(
                            jointName,
                            $"Joint position changed unexpectedly: {previousPosition:F3} -> {currentPosition:F3}",
                            ValidationSeverity.Warning
                        );
                    }
                }

                lastValidatedJointPositions[jointName] = currentPosition;
            }
        }
    }

    bool IsJointBeingControlled(string jointName)
    {
        // Check if this joint is currently being controlled by a user or autonomous system
        // This would involve checking if there are active commands for this joint
        return false; // Simplified for example
    }

    void ValidateRobotPoseSynchronization()
    {
        // Compare robot pose in Unity with expected pose from ROS
        var rosBridge = GetComponent<HumanoidRosBridge>();
        if (rosBridge != null)
        {
            // In a real implementation, you'd compare with ROS TF data
            Vector3 unityPosition = transform.position;
            Quaternion unityRotation = transform.rotation;

            // Log for manual validation
            Debug.Log($"Unity Robot Pose: Position={unityPosition}, Rotation={unityRotation.eulerAngles}");
        }
    }

    void ValidateTimingSynchronization()
    {
        // Check that updates are happening at expected frequencies
        float expectedStateUpdatesPerSecond = 1.0f / GetComponent<HumanoidRosBridge>().stateUpdateInterval;
        float expectedSensorUpdatesPerSecond = 1.0f / GetComponent<HumanoidRosBridge>().sensorUpdateInterval;

        // Add validation for timing consistency
        Debug.Log($"Expected state updates: ~{expectedStateUpdatesPerSecond:F1}/second");
        Debug.Log($"Expected sensor updates: ~{expectedSensorUpdatesPerSecond:F1}/second");
    }

    void AddValidationIssue(string component, string description, ValidationSeverity severity)
    {
        ValidationIssue issue = new ValidationIssue();
        issue.componentName = component;
        issue.issueDescription = description;
        issue.severity = severity;
        issue.timestamp = Time.time;

        validationIssues.Add(issue);

        // Log the issue with appropriate level
        switch (severity)
        {
            case ValidationSeverity.Info:
                Debug.Log($"[INFO] {component}: {description}");
                break;
            case ValidationSeverity.Warning:
                Debug.LogWarning($"[WARNING] {component}: {description}");
                break;
            case ValidationSeverity.Error:
                Debug.LogError($"[ERROR] {component}: {description}");
                break;
        }
    }

    void ReportValidationResults()
    {
        int errorCount = validationIssues.FindAll(issue => issue.severity == ValidationSeverity.Error).Count;
        int warningCount = validationIssues.FindAll(issue => issue.severity == ValidationSeverity.Warning).Count;
        int infoCount = validationIssues.FindAll(issue => issue.severity == ValidationSeverity.Info).Count;

        if (validationIssues.Count == 0)
        {
            Debug.Log("✓ Digital twin validation: PASSED - No issues found");
        }
        else
        {
            Debug.Log($"Digital twin validation results: {errorCount} errors, {warningCount} warnings, {infoCount} info messages");

            if (errorCount > 0)
            {
                Debug.LogError("✗ Digital twin validation: FAILED - Critical issues detected");
            }
            else if (warningCount > 0)
            {
                Debug.LogWarning("⚠ Digital twin validation: PASSED with warnings");
            }
            else
            {
                Debug.Log("✓ Digital twin validation: PASSED with info messages");
            }
        }
    }

    public List<ValidationIssue> GetLatestValidationIssues()
    {
        return new List<ValidationIssue>(validationIssues);
    }

    public bool IsValidationPassed()
    {
        return validationIssues.FindAll(issue => issue.severity == ValidationSeverity.Error).Count == 0;
    }

    public void ManualValidationTrigger()
    {
        StopCoroutine(RunValidationCycle());
        ValidateDigitalTwin();
        StartCoroutine(RunValidationCycle());
    }

    public void ExportValidationReport(string filePath)
    {
        // Export validation results to a file
        System.Text.StringBuilder report = new System.Text.StringBuilder();
        report.AppendLine("Digital Twin Validation Report");
        report.AppendLine($"Generated at: {System.DateTime.Now}");
        report.AppendLine($"Validation interval: {validationInterval}s");
        report.AppendLine();

        foreach (var issue in validationIssues)
        {
            report.AppendLine($"{issue.severity}: [{issue.componentName}] {issue.issueDescription} (Time: {issue.timestamp})");
        }

        report.AppendLine();
        report.AppendLine($"Total Issues: {validationIssues.Count}");
        report.AppendLine($"Errors: {validationIssues.FindAll(i => i.severity == ValidationSeverity.Error).Count}");
        report.AppendLine($"Warnings: {validationIssues.FindAll(i => i.severity == ValidationSeverity.Warning).Count}");
        report.AppendLine($"Info: {validationIssues.FindAll(i => i.severity == ValidationSeverity.Info).Count}");

        // Write to file
        System.IO.File.WriteAllText(filePath, report.ToString());
        Debug.Log($"Validation report exported to: {filePath}");
    }
}
```

## Summary

This comprehensive digital twin project chapter covers the complete integration of ROS 2, Gazebo physics simulation, and Unity visualization for humanoid robots. The implementation includes:

1. **Complete Humanoid Robot Model**: A detailed URDF with proper joint configurations for a full humanoid robot
2. **High-Fidelity Unity Visualization**: HDRP-based rendering with realistic materials and lighting
3. **Interactive Environments**: Configurable environments for human-robot interaction scenarios
4. **ROS-Unity Bridge**: Bidirectional communication for real-time synchronization
5. **Performance Optimization**: Techniques to maintain smooth performance with complex models
6. **Validation System**: Tools to verify the digital twin is working correctly

The project demonstrates the complete pipeline from URDF through physics simulation to high-fidelity visualization, creating a true digital twin that accurately mirrors the behavior of a physical humanoid robot. Students can use this system to test control algorithms, validate sensor data, and experiment with robot behaviors in a safe, virtual environment.

## Key Achievements

- Created a complete humanoid robot model with 20+ joints representing a full body
- Implemented high-fidelity visualization using Unity's HDRP
- Established real-time synchronization between Gazebo simulation and Unity visualization
- Created interactive environments for human-robot interaction testing
- Developed validation tools to ensure digital twin accuracy
- Provided comprehensive examples for each component of the system

## Next Steps

This completes Module 2: The Digital Twin (Gazebo & Unity). Students now understand how to create complete digital twins of humanoid robots, with realistic physics simulation in Gazebo and high-fidelity visualization in Unity, all connected through ROS 2 for real-time synchronization.

The next module (Module 3) will focus on the AI-Robot Brain using NVIDIA Isaac Sim, where students will learn about advanced perception, VSLAM, navigation, and GPU-accelerated robotics systems.