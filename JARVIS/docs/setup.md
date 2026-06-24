# Setup

## Environment

| Component | Version / note |
|---|---|
| Isaac Sim | 5.1 (**Windows-native**) |
| Isaac Lab | 2.3.0 (`isaaclab.bat`, bundled Python 3.11) |
| RL | RSL-RL PPO + Dr. Eureka reward generation |
| Tool encoder | Uni3D (pretrained, frozen) → PCA-64 |
| VLA | SmolVLA-450M (LoRA + 4-bit) |
| Data | LeRobot v3 |

> **Isaac Sim / Isaac Lab must run Windows-native.** WSL2 is not viable for Isaac Sim on RTX 50-series (Blackwell) GPUs due to a driver issue. WSL2 is used only for SmolVLA inference and ROS2.

## Compute

| Resource | Use |
|---|---|
| RTX 5060 8GB (local, Windows) | env development, GUI verification |
| School GPU server | primary training, 4096 parallel envs |
| Colab A100 | SmolVLA fine-tuning |

## Install

```bash
# RL / sim deps (inside the Isaac Lab python env)
isaaclab.bat -p -m pip install rsl-rl-lib

# tool-latent + eval deps
pip install torch trimesh scikit-learn iisignature vendi-score lerobot
```

## Key paths & assets

- Working dir: `D:\Documents\HJ\YAICON-JARVIS\`
- Tool USD files: `C:\jarvis_tools\*.usd`
- PCA model: `models/pca_model_dim64.pkl` (`{pca, mean, std}`)
- HuggingFace dataset: `davekim0323/jarvis-screwdriver-v1` (cams: `observation.images.wrist`, `observation.images.external`)

## Useful tuning notes

- **PCA-64** is the operating point: 79.5% variance retained, latent stays smaller than the ~32-d observation.
- Screwdriver grasp offset (tool-local): `(0.002, 0.001, -0.016)`.
- K-means default threshold → ~11 raw clusters; the tight+pure 8 are kept as representatives.
