# Pipeline

End-to-end flow from a tool point cloud to a deployed VLA policy.

```
Tool point cloud (Objaverse-XL)
        │
        ▼
[1] Uni3D frozen encoder ──► PCA-64 latent z          (79.5% variance retained)
        │                    build_tool_latents.py
        ▼
[2] K-means clustering ──► K=8 representatives         cluster_tools.py
        │
        ▼
[3] Cluster-supervised grasp spec  (annotate O(K))     grasp_transfer/
        │                          bbox-normalized + principal-axis aligned
        ▼
[4] Dr. Eureka reward generation                       mdp_rewards.py
        │
        ▼
[5] Tool-conditioned PPO  π(a|s,z)                     rl/train.py
        │  RSL-RL, 4096 parallel envs (Isaac Lab)      task: JarvisMultiTool-v0
        ▼
[6] 500-episode rollout ──► LeRobot v3 dataset         convert_raw_to_lerobot.py
        │                    HF: davekim0323/jarvis-screwdriver-v1
        ▼
[7] SmolVLA-450M fine-tune  (LoRA + 4-bit, Colab A100)
        │
        ▼
[8] Sim deploy  (deg→rad action wrapper)               vla/smolvla_wrapper.py
        │
        ▼
[9] (upside) sim-to-real on SO-101 lightbox
```

## Status

**Verified**
- Dr. Eureka RL → cube pick-and-place convergence
- Uni3D + PCA-64 + K-means tool latent space (8 clean representatives, 3 dropped)
- Single-tool (screwdriver) policy convergence
- New-tool warm-start adaptation in <100 iterations
- Policy rollout → LeRobot v3 dataset (499 eps / 29,940 frames) → HuggingFace upload
- 8-tool reach success

**Roadmap → CoRL 2026 workshop**
- 8-tool tool-conditioned policy with full grasp (relaxing the curriculum action constraint)
- Full RL → SmolVLA → sim-deploy loop closed end-to-end on the cube (Gate 1)
- Held-out tool generalization numbers for the results table (Gate 2)
- Real diversity CSVs through `eval/diversity_analysis.py`
- Optional sim-to-real lightbox transfer + Cosmos Transfer 2.5 domain randomization

## Notes

- **Action wrapper.** SmolVLA emits degree absolute targets; jarvis_env expects radian scaled RL actions. Conversion: deg→rad, arm `(target − default)/0.5`, gripper threshold @ 10°. See `vla/smolvla_wrapper.py`.
- **Diversity metric.** Per-trajectory L2 normalization is required for a valid Vendi score; across-set standardization inflates near-identical sets. See `eval/diversity_analysis.py`.
- **Environment.** Isaac Sim / Isaac Lab run Windows-native. See [docs/setup.md](docs/setup.md).
