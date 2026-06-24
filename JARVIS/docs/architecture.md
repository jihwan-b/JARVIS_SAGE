# Architecture & Positioning

## Contributions

1. **Tool latent space as a manipulation prior.** A frozen tool-geometry latent conditions the policy, turning per-tool learning into generalization across a latent space.
2. **Cluster-supervised grasp specification.** Annotate grasps at the cluster level — `O(K)` annotations instead of `O(N)` per-tool.
3. **Synthetic tool augmentation.** Generate tool diversity beyond the dataset via the latent space.
4. **Tool-conditioned single policy `π(a | s, z)`.** One policy; the latent `z` carries it to unseen tools.
5. **End-to-end automated VLA adaptation.** Point cloud in → deployed policy out, with no human demonstrations.

The pretrained Uni3D encoder is used as-is — the contribution is the framework built around it.

## Positioning

JARVIS connects several lines of prior work through a tool point-cloud latent space:

| Work | Idea | Missing axis |
|---|---|---|
| RoboCat | multi-task generalist | tool conditioning |
| MimicGen | demo augmentation | tool generalization |
| AutoMate | automated assembly skills | tool latent |
| DrEureka | LLM reward design | tool generalization |
| RoLD | trajectory latent diffusion | tool latent |
| BHD / LIBERO | demo-free data / lifelong benchmark | held-out tool evaluation |

**Held-out tool generalization is the differentiating axis.** LIBERO (where BHD is strong) is a lifelong household pick-and-place benchmark with no tool-use generalization axis — so it operates on a different problem from JARVIS rather than competing on the same one.

## Hardware

SO-101 5-DoF arm + gripper, fixed to a desk, with a 3D-printed toolbox. Demo tools: Phillips screwdriver, hammer, Allen key, marker pen.

## Why tools

The team comes from a mechanical-engineering background — tool handling is the natural domain, and the project is named after Iron Man's JARVIS.
