"""
JARVIS env registration + tool-latent observation hook.

Registers `JarvisMultiTool-v0` and exposes the `tool_latent` observation term
that injects the current episode's tool latent z into the policy observation,
producing the tool-conditioned policy π(a | s, z).

Per-tool deployment overrides `TOOL_USD_PATH` in tool_env_cfg (hardcoded path,
no env-var support) and the matching grasp offset.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from tool_env_cfg import JarvisToolEnvCfg

# z is set per-env at reset from the sampled representative tool's latent.
# Shape: (num_envs, latent_dim). Registered as a managed buffer on the env.
_LATENT_DIM = 64


def tool_latent(env) -> torch.Tensor:
    """Observation term: current per-env tool latent z (num_envs, 64)."""
    buf = getattr(env, "_tool_latent", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, _LATENT_DIM, device=env.device)
        env._tool_latent = buf
    return buf


def set_tool_latent(env, env_ids: torch.Tensor, z: torch.Tensor) -> None:
    """Assign the latent for the tools sampled into `env_ids` at reset."""
    tool_latent(env)[env_ids] = z.to(env.device)


gym.register(
    id="JarvisMultiTool-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": JarvisToolEnvCfg},
    disable_env_checker=True,
)
