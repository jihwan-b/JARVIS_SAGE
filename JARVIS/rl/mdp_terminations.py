"""Termination terms for the JARVIS tool task."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def tool_dropped(env: ManagerBasedRLEnv, tool_cfg: SceneEntityCfg,
                 minimum_height: float = -0.05) -> torch.Tensor:
    """Tool fell off the table."""
    tool = env.scene[tool_cfg.name]
    return tool.data.root_pos_w[:, 2] < minimum_height


def reached_target(env: ManagerBasedRLEnv, tool_cfg: SceneEntityCfg,
                   command_name: str = "target_pose", threshold: float = 0.03) -> torch.Tensor:
    """Success: tool delivered within threshold of the commanded pose."""
    tool = env.scene[tool_cfg.name]
    target = env.command_manager.get_command(command_name)[:, :3]
    return torch.norm(tool.data.root_pos_w - target, dim=-1) < threshold
