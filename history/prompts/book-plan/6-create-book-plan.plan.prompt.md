---
id: 6
title: "Create Book Plan"
stage: plan
date_iso: "2025-12-16"
surface: agent
model: claude-haiku-4-5-20251001
feature: book-plan
branch: book-plan
user: user
command: "/sp.plan"
labels: ["plan", "robotics", "book", "docusaurus", "ros2"]
files_yaml:
  - "specs/book-plan/plan.md"
  - "specs/book-plan/research.md"
  - "specs/book-plan/data-model.md"
  - "specs/book-plan/quickstart.md"
  - "specs/book-plan/contracts/rag-api.yaml"
tests_yaml: []
links:
  spec: null
  ticket: null
  adr: null
  pr: null
outcome: "Successfully created implementation plan for Physical AI & Humanoid Robotics book"
evaluation: "Plan includes all required components: technical context, constitution check, project structure, research, data model, quickstart guide, and API contracts"
---

# Prompt

```text
Physical AI & Humanoid Robotics — Full Book Plan

Objective:
Create a complete, multi-module textbook (Docusaurus MDX) covering the entire embodied-AI pipeline:
ROS 2 → Digital Twin Simulation → NVIDIA Isaac AI Brain → Vision-Language-Action Robotics.
All content will be structured for Spec-Kit Plus + Claude Code workflows and deployed to GitHub Pages.

Scope (4 Modules):
- Module 1: The Robotic Nervous System (ROS 2)
- Module 2: The Digital Twin (Gazebo & Unity)
- Module 3: The AI-Robot Brain (NVIDIA Isaac™)
- Module 4: Vision-Language-Action (VLA)

Deliverables:
- 50+ Docusaurus chapters across 4 modules
- Capstone: Autonomous Humanoid Agent receiving a voice command and performing a full action cycle
- GitHub Pages deployment
- RAG chatbot over the book (OpenAI Agents/ChatKit + FastAPI + Neon + Qdrant)

-------------------------------------------------------------------
Module 1 — The Robotic Nervous System (ROS 2)
-------------------------------------------------------------------
Purpose:
Teach ROS 2 as the "nervous system" of humanoid robots.

Chapters:
1. What is ROS 2 and Why It Matters
2. DDS & ROS Architecture
3. ROS 2 Nodes
4. Topics (Pub/Sub Messaging)
5. Services & Actions
6. Parameters & Configuration
7. Launch Files
8. rclpy Basics
9. Python Agents → ROS Control Bridge
10. URDF Introduction
11. Building a Humanoid URDF
12. Visualizing in RViz
13. Mini Project: Humanoid Joint Control
14. Summary + MCQs + Exercises

Outcomes:
- Students publish/subscribe topics
- Build URDF
- Control humanoid joints with Python

-------------------------------------------------------------------
Module 2 — The Digital Twin (Gazebo & Unity)
-------------------------------------------------------------------
Purpose:
Teach physics simulation, world-building, and perception simulation.

Chapters:
1. What is a Digital Twin?
2. Gazebo Overview
3. Importing URDF to Gazebo
4. Physics Simulation (gravity, collisions)
5. Sensor Simulation (LiDAR, Depth, IMU)
6. Subscribing to Sensor Data from ROS 2
7. Environment Building in Gazebo
8. Unity for Robotics
9. High-Fidelity Rendering in Unity HDRP
10. Human–Robot Interaction Simulation
11. ROS–Unity Communication
12. Mini Project: Full Digital Twin of Humanoid
13. Summary + MCQs + Exercises

Outcomes:
- Build full digital twin
- Simulate perception sensors
- Create Unity interaction scenes

-------------------------------------------------------------------
Module 3 — The AI-Robot Brain (NVIDIA Isaac™)
-------------------------------------------------------------------
Purpose:
Teach advanced perception, VSLAM, navigation, and GPU-accelerated robotics.

Chapters:
1. Overview of Isaac Sim
2. Installing Isaac Sim for Robotics
3. Importing Robots into Isaac
4. Photorealism & Materials
5. Synthetic Data Generation
6. Isaac ROS Overview
7. Isaac ROS VSLAM
8. Isaac ROS Perception Nodes
9. Overview of Nav2
10. Nav2 Architecture (planner, controller, recovery)
11. Integrating Isaac ROS + Nav2
12. Project: Humanoid Walks an Obstacle Course
13. Summary + MCQs + Exercises

Outcomes:
- Generate synthetic vision datasets
- Use Isaac ROS for VSLAM
- Run Nav2 for humanoid navigation

-------------------------------------------------------------------
Module 4 — Vision-Language-Action (VLA)
-------------------------------------------------------------------
Purpose:
Teach LLM-powered robotics: Voice → Understanding → Planning → Acting.

Chapters:
1. What is VLA?
2. End-to-End VLA Pipeline
3. Whisper Voice Commands
4. Natural Language → Structured Commands
5. LLM-Based Cognitive Planning
6. Safety Guardrails for LLM-Controlled Robots
7. Visual Perception Integration
8. Object Selection & Manipulation
9. Navigation via LLM Plans → Nav2
10. Manipulator Control
11. Full VLA Agent Loop
12. Capstone: Autonomous Humanoid
13. Testing & Failure Modes
14. Summary + MCQs + Tasks

Outcomes:
- Voice-to-Action pipeline working in simulation
- LLM translates "Clean the room" → ROS 2 actions
- Full autonomous humanoid

-------------------------------------------------------------------
Cross-Module Infrastructure
-------------------------------------------------------------------
- Docusaurus project setup
- GitHub Pages deployment
- Unified glossary and terminology style
- RAG System:
  - Book ingestion pipeline
  - Neon Postgres schema
  - Qdrant vector store
  - Agent configuration (OpenAI / ChatKit)

-------------------------------------------------------------------
Success Criteria:
-------------------------------------------------------------------
- 4 modules completed as independent but connected learning paths
- All MDX builds successfully in Docusaurus
- RAG chatbot achieves 90%+ grounding accuracy
- Capstone humanoid completes full Voice→Plan→Navigate→Perceive→Act cycle
- Clean coding standards and reproducible simulation instructions
```

# Response

Created a comprehensive implementation plan for the Physical AI & Humanoid Robotics book, including:

- Implementation plan document (plan.md) with technical context, constitution check, and project structure
- Research document (research.md) with technical decisions and architecture patterns
- Data model (data-model.md) defining key entities and relationships
- Quickstart guide (quickstart.md) for development setup and workflow
- API contract (contracts/rag-api.yaml) for the RAG system

The plan addresses all requirements from the original feature description, including the 4 modules, cross-module infrastructure, and success criteria. It follows the Spec-Kit Plus methodology and is structured for integration with Claude Code workflows.