<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.1.0
- Modified principles: Added 6 specific principles for Physical AI & Humanoid Robotics
- Added sections: Technical Standards, Development Workflow
- Removed sections: None
- Templates requiring updates: ✅ Updated
- Follow-up TODOs: None
-->

# Physical AI & Humanoid Robotics Constitution

## Core Principles

### I. Technical Accuracy and Documentation Standards
All robotics and AI claims must reference official documentation (ROS 2, NVIDIA Isaac, Gazebo, Unity, OpenAI, etc.) and code samples must be validated in ROS 2 Humble + Gazebo Garden + Isaac Sim environments. This ensures that all content is technically accurate and reproducible for learners at intermediate and advanced levels.

### II. Learn-Simulate-Deploy Pedagogy
All explanations must follow the "learn → simulate → deploy" pedagogy, where theoretical concepts are immediately followed by practical simulation examples and deployment considerations. This creates a hands-on learning experience that bridges theory with practice in robotics and embodied AI.

### III. Modular Architecture and Reusability
The book structure must follow Docusaurus MDX conventions with modular architecture enabling integration with Claude Code and structured specs. Each module and chapter should be independently consumable while contributing to the overall learning progression from ROS fundamentals through simulation to Isaac and VLA models.

### IV. Reproducible Code and Simulation Environments
All code examples must be validated in controlled simulation environments (ROS 2 Humble + Gazebo Garden + Isaac Sim) and include clear setup instructions. This ensures that readers can reproduce examples and gain hands-on experience with physical AI and humanoid robotics systems.

### V. Cohesive Learning Progression
The curriculum must form coherent learning paths: ROS → Simulation → Isaac → VLA, with each module building upon previous knowledge. Each chapter must include learning objectives, code examples, simulation steps, and at least one robotics diagram or flow explanation to support different learning styles.

### VI. RAG Integration and Content Grounding
The integrated RAG chatbot must follow OpenAI Agents/ChatKit SDK + FastAPI + Neon Postgres + Qdrant stack and achieve 90%+ grounding accuracy when answering questions using book content. This ensures the chatbot serves as an effective learning companion that accurately reflects the book's content.

## Technical Standards

All robotics and AI implementations must reference official documentation from ROS 2, NVIDIA Isaac, Gazebo, Unity, and OpenAI. Code samples must be validated in ROS 2 Humble + Gazebo Garden + Isaac Sim environments. Terminology must stay consistent across modules (ROS graph, namespaces, URDF, digital twin, VLA models). All MDX content must be exportable to GitHub Pages without breaking builds.

The project must support four modules with 8-12 chapters each (minimum 40 total chapters). The capstone must detail an end-to-end pipeline: Voice → LLM Planning → ROS 2 Actions → Navigation → Perception → Manipulation, demonstrating the complete physical AI workflow.

## Development Workflow

Each chapter must include: learning objectives, code examples, simulation steps, and at least one robotics diagram or flow explanation. All code must be tested or validated via Spec-Kit Plus or Claude Code execution sandbox. The book must compile successfully in Docusaurus and deploy on GitHub Pages. The final capstone robot must complete a full Voice-to-Action task in simulation.

Development follows the Spec-Kit Plus methodology with clear separation of concerns: specs define requirements, plans outline architecture decisions, and tasks break down implementation into testable units. All changes must reference code precisely and maintain backward compatibility where possible.

## Governance

This constitution supersedes all other development practices for the Physical AI & Humanoid Robotics project. All pull requests and reviews must verify compliance with these principles. Any changes to core principles require explicit documentation of rationale and impact assessment. The constitution must be referenced during architectural decision-making and code reviews.

All architectural decisions that meet significance criteria (long-term consequences, multiple viable options considered, cross-cutting influence) must be documented in Architecture Decision Records (ADRs) following the prescribed format.

**Version**: 1.1.0 | **Ratified**: 2025-12-16 | **Last Amended**: 2025-12-16
