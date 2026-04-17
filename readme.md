# VLM-PBRS: Hierarchical Hybrid Reinforcement Learning

Welcome to the **VLM-PBRS** project. This repository introduces a state-of-the-art **Hierarchical Hybrid Reward Architecture** combining Vision-Language Models (VLMs) and classical Reinforcement Learning (RL) for Embodied AI tasks.

## 🚀 The Architecture

Training RL agents purely on Vision-Language Models (VLMs) like LLaVA-7B suffers from severe spatial resolution limitations (VLMs cannot accurately regress continuous floating-point coordinates).

We solve this using a multi-tiered approach:
1. **Low-Level Micro Kinematics**: Physics-based dense rewards (`kinematic_reward`) handle high-frequency motor control.
2. **High-Level Macro VLM (PBRS)**: We use the robust Multiple Choice Question (MCQ) format to query the VLM for coarse spatial milestones (Left, Bottom, Right). The VLM acts as an intelligent supervisor.
3. **RBF Visual Smoothing**: Asymmetric semantic VLM labels are smoothed into a fully differentiable 3D continuous potential field using a Visual Prototype Cache.
4. **PBRS Integration**: VLM evaluations are integrated via Potential-Based Reward Shaping ($F = \gamma\Phi' - \Phi$), ensuring mathematical immunity to reward hacking.

## 🏃 Quick Start

Ensure you have your Ollama server running locally with `llava:7b` (or swap to a cloud API in the client).

```powershell
# Run the complete Hybrid training loop with periodic evaluations
python main.py --timesteps 300000 --run-eval
```

## 📂 Repository Structure

- `main.py`: The single entry point for training the Hybrid Architecture.
- `envs/visual_wrapper.py`: Contains the `AdaptiveVisualPBRS_Wrapper` with RBF smoothing and Hybrid Reward calculations.
- `vlm/llava_client.py`: The MCQ parser forcing strict categoric classification from the VLM.
- `archive_pure_vlm/`: Backups of the "pure visual" codebase configurations for future rigorous benchmarking (e.g., when massive API models replace local 7B models).
- `archive_docs_and_old_tests/`: Old proposals and redundant test files.

## 📜 Readings

For academic context, please review `PAPER_HYBRID_PROPOSAL.md` and `REVIEW_HYBRID_HIERARCHICAL_VLM.md`.
