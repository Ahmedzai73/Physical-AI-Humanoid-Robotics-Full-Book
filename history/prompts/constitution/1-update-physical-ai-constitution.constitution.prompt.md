---
id: 1
title: "Update Physical AI Constitution"
stage: constitution
date_iso: "2025-12-16"
surface: agent
model: claude-haiku-4-5-20251001
feature: none
branch: master
user: user
command: "/sp.constitution"
labels: ["constitution", "robotics", "physical-ai"]
files_yaml:
  - ".specify/memory/constitution.md"
tests_yaml: []
links:
  spec: null
  ticket: null
  adr: null
  pr: null
outcome: "Successfully updated constitution with specific principles for Physical AI & Humanoid Robotics project"
evaluation: "Constitution now properly reflects project requirements and constraints"
---

# Prompt

```text
Project: Physical AI & Humanoid Robotics — Full Book + Integrated RAG Chatbot

Core principles:
- Technical accuracy based on robotics, AI, and simulation frameworks (ROS 2, Gazebo, Unity, NVIDIA Isaac)
- Clarity for both intermediate and advanced learners in robotics and embodied AI
- Hands-on reproducibility through executable code, simulations, and project pipelines
- Cohesive book structure using Docusaurus + Spec-Kit Plus conventions
- Modular architecture enabling integration with Claude Code and structured specs

Key standards:
- All robotics and AI claims must reference official documentation (ROS 2, NVIDIA Isaac, Gazebo, Unity, OpenAI, etc.)
- Code samples must be validated in ROS 2 Humble + Gazebo Garden + Isaac Sim environments
- Explanations must follow "learn → simulate → deploy" pedagogy
- Book output must follow Docusaurus MDX structure compatible with GitHub Pages
- RAG Chatbot must follow OpenAI Agents/ChatKit SDK + FastAPI + Neon Postgres + Qdrant stack
- Terminology must stay consistent across modules (ROS graph, namespaces, URDF, digital twin, VLA models)

Constraints:
- Four modules, each 8–12 chapters (minimum 40 total chapters)
- Each chapter must include:
    - Learning objectives
    - Code examples
    - Simulation steps
    - At least one robotics diagram or flow explanation
- Capstone must detail the end-to-end pipeline:
    Voice → LLM Planning → ROS 2 Actions → Navigation → Perception → Manipulation
- All MDX must be exportable to GitHub Pages without breaking builds
- All specs must be reusable inside Claude Code for automated editing

Success criteria:
- Entire book compiles successfully in Docusaurus and deploys on GitHub Pages
- Each module forms a coherent learning path: ROS → Simulation → Isaac → VLA
- All code tested or validated via Spec-Kit Plus or Claude Code execution sandbox
- RAG Chatbot answers questions using book content with 90%+ grounding accuracy
- Final capstone robot completes a full Voice-to-Action task in simulation
```

# Response

Created a comprehensive constitution for the Physical AI & Humanoid Robotics project that incorporates all specified principles, standards, and constraints. The constitution includes six core principles covering technical accuracy, pedagogy, modularity, reproducibility, learning progression, and RAG integration. It also defines technical standards, development workflow, and governance procedures specific to the robotics education project.