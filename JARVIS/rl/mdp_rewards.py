"""
Reward terms for the JARVIS tool-conditioned pick-and-place task.

Dr. Eureka-generated dense rewards, shaped for physical stability. The grasp
point is expressed in the tool's local frame and rotated into world via
`quat_apply` so the target tracks the tool's orientation.

Stages encoded in the reward: reach the grasp point, close on it, lift, and
deliver to the target pose.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


def _tcp_pos(env, robot_cfg, tcp_offset):
    """Tool-center-point in world frame, anchored to the (fixed) gripper body —
    NOT the moving jaw, so the TCP does not shift while the gripper closes."""
    robot = env.scene[robot_cfg.name]
    body_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    body_quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0]]
    return body_pos + quat_apply(body_quat, tcp_offset.to(env.device))


def _grasp_pos(env, tool_cfg, grasp_offset):
    """Grasp point in world frame = tool pose ⊕ tool-local grasp offset."""
    tool = env.scene[tool_cfg.name]
    return tool.data.root_pos_w + quat_apply(tool.data.root_quat_w, grasp_offset.to(env.device))


def reaching_grasp_point(env: ManagerBasedRLEnv, std: float,
                         robot_cfg: SceneEntityCfg, tool_cfg: SceneEntityCfg,
                         tcp_offset, grasp_offset) -> torch.Tensor:
    tcp = _tcp_pos(env, robot_cfg, torch.tensor(tcp_offset))
    grasp = _grasp_pos(env, tool_cfg, torch.tensor(grasp_offset))
    d = torch.norm(tcp - grasp, dim=-1)
    return 1.0 - torch.tanh(d / std)


def grasping_tool(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg,
                  tool_cfg: SceneEntityCfg, tcp_offset, grasp_offset,
                  grasp_threshold: float = 0.015) -> torch.Tensor:
    """Reward closing the gripper only when the TCP is within threshold of the
    grasp point. Uses the gripper-close action sign so the policy cannot harvest
    reward without actually commanding a close."""
    tcp = _tcp_pos(env, robot_cfg, torch.tensor(tcp_offset))
    grasp = _grasp_pos(env, tool_cfg, torch.tensor(grasp_offset))
    near = (torch.norm(tcp - grasp, dim=-1) < grasp_threshold).float()
    close_cmd = torch.clamp(env.action_manager.action[:, -1], min=0.0)  # +ve = close
    return near * close_cmd


def lifting_tool(env: ManagerBasedRLEnv, tool_cfg: SceneEntityCfg,
                 minimal_height: float = 0.04) -> torch.Tensor:
    tool = env.scene[tool_cfg.name]
    return (tool.data.root_pos_w[:, 2] > minimal_height).float()


def tool_to_target(env: ManagerBasedRLEnv, std: float, tool_cfg: SceneEntityCfg,
                   command_name: str = "target_pose") -> torch.Tensor:
    tool = env.scene[tool_cfg.name]
    target = env.command_manager.get_command(command_name)[:, :3]
    d = torch.norm(tool.data.root_pos_w - target, dim=-1)
    return 1.0 - torch.tanh(d / std)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
