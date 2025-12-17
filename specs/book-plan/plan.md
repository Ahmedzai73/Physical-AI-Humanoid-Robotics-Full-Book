# Implementation Plan: Physical AI & Humanoid Robotics — Full Book

**Branch**: `book-plan` | **Date**: 2025-12-16 | **Spec**: [link]
**Input**: Feature specification from `/specs/book-plan/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a complete, multi-module textbook (Docusaurus MDX) covering the entire embodied-AI pipeline: ROS 2 → Digital Twin Simulation → NVIDIA Isaac AI Brain → Vision-Language-Action Robotics. All content will be structured for Spec-Kit Plus + Claude Code workflows and deployed to GitHub Pages, including a RAG chatbot over the book content.

## Technical Context

**Language/Version**: Docusaurus MDX, Python 3.11, ROS 2 Humble, JavaScript/TypeScript
**Primary Dependencies**: Docusaurus, ROS 2, NVIDIA Isaac Sim, OpenAI API, FastAPI, Neon Postgres, Qdrant
**Storage**: Git repository for content, Neon Postgres for RAG metadata, Qdrant for vector storage
**Testing**: Spec-Kit Plus validation, simulation environment testing, Claude Code execution sandbox
**Target Platform**: GitHub Pages (web), ROS 2 Humble simulation environments
**Project Type**: Multi-module documentation + simulation + AI integration
**Performance Goals**: 90%+ grounding accuracy for RAG chatbot, fast Docusaurus build times
**Constraints**: All MDX builds successfully in Docusaurus, clean coding standards, reproducible simulation instructions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Technical Accuracy and Documentation Standards**: All content must reference official documentation (ROS 2, NVIDIA Isaac, Gazebo, Unity, OpenAI, etc.) and code samples must be validated in ROS 2 Humble + Gazebo Garden + Isaac Sim environments
- **Learn-Simulate-Deploy Pedagogy**: All explanations must follow the "learn → simulate → deploy" pedagogy with practical examples
- **Modular Architecture and Reusability**: Book structure must follow Docusaurus MDX conventions with modular architecture enabling integration with Claude Code
- **Reproducible Code and Simulation Environments**: All code examples must be validated in controlled simulation environments with clear setup instructions
- **Cohesive Learning Progression**: Curriculum must form coherent learning paths: ROS → Simulation → Isaac → VLA
- **RAG Integration and Content Grounding**: RAG chatbot must follow OpenAI Agents/ChatKit SDK + FastAPI + Neon + Qdrant stack with 90%+ grounding accuracy

## Project Structure

### Documentation (this feature)

```text
specs/book-plan/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
# Book Content Structure
docs/
├── module-1-ros/
│   ├── index.md
│   ├── chapter-1-what-is-ros.md
│   ├── chapter-2-architecture.md
│   └── [additional chapters...]
├── module-2-digital-twin/
│   ├── index.md
│   ├── chapter-1-digital-twin.md
│   └── [additional chapters...]
├── module-3-ai-brain/
│   ├── index.md
│   └── [additional chapters...]
├── module-4-vla/
│   ├── index.md
│   └── [additional chapters...]
└── capstone/
    └── autonomous-humanoid.md

# Simulation Code
simulation/
├── ros-examples/
│   ├── nodes/
│   ├── launch/
│   └── urdf/
├── gazebo-worlds/
│   ├── humanoid-models/
│   └── environments/
├── isaac-sim-scenes/
└── vla-pipeline/

# RAG System
rag-system/
├── ingestion/
│   ├── parser.py
│   └── chunker.py
├── api/
│   ├── main.py
│   └── models.py
├── vector-store/
│   └── qdrant-config.yaml
└── database/
    └── neon-schema.sql

# Docusaurus Configuration
docusaurus.config.js
package.json
README.md
```

**Structure Decision**: Multi-module documentation structure with separate directories for each module (ROS, Digital Twin, AI Brain, VLA) and dedicated folders for simulation code and RAG system. This allows for modular content creation while maintaining clear separation of concerns between documentation, simulation, and AI components.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |