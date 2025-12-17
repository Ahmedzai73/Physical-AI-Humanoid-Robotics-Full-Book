# Data Model: Physical AI & Humanoid Robotics — Full Book

**Date**: 2025-12-16
**Feature**: Physical AI & Humanoid Robotics — Full Book Plan

## Overview

This document defines the key data entities and their relationships for the Physical AI & Humanoid Robotics book project, including content structure, simulation data, and RAG system components.

## Content Entities

### Module
- **Fields**: id, title, description, learningObjectives, prerequisites, duration, difficulty
- **Relationships**: Contains many Chapters
- **Validation**: Title and description required, duration positive, difficulty in ['beginner', 'intermediate', 'advanced']

### Chapter
- **Fields**: id, moduleId, title, content, wordCount, learningObjectives, codeExamples, simulationSteps, diagrams
- **Relationships**: Belongs to one Module, contains many Exercises
- **Validation**: Title and content required, wordCount in range 800-1500

### Exercise
- **Fields**: id, chapterId, type, question, answer, difficulty, tags
- **Relationships**: Belongs to one Chapter
- **Validation**: Question and answer required, type in ['mcq', 'coding', 'simulation', 'essay']

### CodeExample
- **Fields**: id, chapterId, language, code, description, executionEnvironment, expectedOutput
- **Relationships**: Belongs to one Chapter
- **Validation**: Language and code required, executionEnvironment in ['ros2', 'gazebo', 'isaac', 'python', 'other']

### SimulationStep
- **Fields**: id, chapterId, description, commands, expectedResult, troubleshootingTips
- **Relationships**: Belongs to one Chapter
- **Validation**: Description and commands required

## Simulation Entities

### RobotModel
- **Fields**: id, name, urdfPath, sdfPath, description, jointCount, sensorCount
- **Relationships**: Used in many SimulationScenes
- **Validation**: Name and URDF path required, jointCount positive

### SimulationScene
- **Fields**: id, name, description, environmentType, robotModelId, worldFile, sensorConfig
- **Relationships**: Uses one RobotModel
- **Validation**: Name and environmentType required, environmentType in ['gazebo', 'isaac', 'unity']

### SensorData
- **Fields**: id, sceneId, sensorType, dataFormat, topicName, frequency, resolution
- **Relationships**: Belongs to one SimulationScene
- **Validation**: SensorType required, sensorType in ['lidar', 'camera', 'depth', 'imu', 'gps'], frequency positive

## RAG System Entities

### DocumentChunk
- **Fields**: id, chapterId, content, embedding, metadata, chunkOrder
- **Relationships**: Belongs to one Chapter
- **Validation**: Content and embedding required, chunkOrder non-negative

### VectorStore
- **Fields**: id, name, provider, dimensions, similarityFunction, indexConfig
- **Relationships**: Contains many DocumentChunks
- **Validation**: Name and provider required, provider in ['qdrant', 'pinecone', 'weaviate'], dimensions positive

### ChatSession
- **Fields**: id, userId, createdAt, messages, context, active
- **Relationships**: Contains many ChatMessages
- **Validation**: userId required, createdAt defaults to current time

### ChatMessage
- **Fields**: id, sessionId, role, content, timestamp, sources, groundingScore
- **Relationships**: Belongs to one ChatSession
- **Validation**: Role in ['user', 'assistant'], content required, groundingScore in range 0-1

## API Contract Entities

### APISpec
- **Fields**: id, name, version, description, endpoints, authentication
- **Validation**: Name and version required, endpoints is valid OpenAPI spec

### Endpoint
- **Fields**: id, apiId, path, method, parameters, responses, rateLimit
- **Validation**: Path and method required, method in ['GET', 'POST', 'PUT', 'DELETE']

## State Transitions

### Chapter States
- `draft` → `review` → `published` (content creation workflow)
- `published` → `deprecated` (content lifecycle)

### SimulationScene States
- `created` → `validated` → `published` (simulation validation workflow)
- `published` → `archived` (simulation lifecycle)

### ChatSession States
- `active` → `inactive` → `archived` (session lifecycle)
- `active` → `error` (error handling)

## Validation Rules

### Content Validation
- Each chapter must have 1+ learning objectives
- Each chapter must have 1+ code examples
- Each chapter must have 1+ simulation steps
- Each chapter must have 1+ diagrams (described in text)

### Simulation Validation
- Each robot model must have valid URDF/SDF files
- Each simulation scene must be testable in target environment
- Each sensor configuration must match actual robot capabilities

### RAG Validation
- Each document chunk must have a grounding score ≥ 0.7
- Each chat response must cite sources from book content
- Each vector embedding must be properly formatted

## Relationships Summary

```
Module (1) → (Many) Chapter (1) → (Many) Exercise
Module (1) → (Many) Chapter (1) → (Many) CodeExample
Module (1) → (Many) Chapter (1) → (Many) SimulationStep

RobotModel (1) → (Many) SimulationScene (1) → (Many) SensorData
Chapter (1) → (Many) DocumentChunk (1) → (Many) VectorStore

ChatSession (1) → (Many) ChatMessage
APISpec (1) → (Many) Endpoint
```