"""
Tool-conditioned RL environment config (Isaac Lab manager-based).

Observation (policy group), 33-d base:
    joint_pos (6) + joint_vel (6) + tool_pos (3) + tool_quat (4)
    + target_pose (7) + last_action (7)
The tool-latent vector z is concatenated to the policy observation for the
tool-conditioned variant, giving the single policy π(a | s, z) that generalizes
across tools by their geometry latent.

Action: 6-d (5 arm joints + gripper), relative scaled joint targets.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import mdp_rewards as R
import mdp_terminations as T

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TOOL_USD_PATH = "C:/jarvis_tools/screwdriver.usd"   # overridden per tool
TOOL_COLOR = (0.2, 0.4, 1.0)                         # avoid USD white override
ACTION_SCALE = 0.5
GRASP_OFFSET = (0.002, 0.001, -0.016)               # screwdriver, tool-local
TCP_OFFSET = (0.0, 0.0, 0.015)

ROBOT = SceneEntityCfg("robot", body_names=["gripper"])
TOOL = SceneEntityCfg("tool")


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #
@configclass
class JarvisSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING        # SO-101 articulation (set in __post_init__)
    tool: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tool",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TOOL_USD_PATH,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TOOL_COLOR),
        ),
    )


# --------------------------------------------------------------------------- #
# MDP
# --------------------------------------------------------------------------- #
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func="mdp.joint_pos_rel")
        joint_vel = ObservationTermCfg(func="mdp.joint_vel_rel")
        tool_pos = ObservationTermCfg(func="mdp.root_pos_w", params={"asset_cfg": TOOL})
        tool_quat = ObservationTermCfg(func="mdp.root_quat_w", params={"asset_cfg": TOOL})
        target_pose = ObservationTermCfg(func="mdp.generated_commands",
                                         params={"command_name": "target_pose"})
        last_action = ObservationTermCfg(func="mdp.last_action")
        tool_latent = ObservationTermCfg(func="mdp.tool_latent")   # z (concatenated)

        def __post_init__(self):
            self.enable_corruption = True      # match training-time obs noise
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    reaching = RewardTermCfg(func=R.reaching_grasp_point, weight=1.0,
                             params={"std": 0.1, "robot_cfg": ROBOT, "tool_cfg": TOOL,
                                     "tcp_offset": TCP_OFFSET, "grasp_offset": GRASP_OFFSET})
    grasping = RewardTermCfg(func=R.grasping_tool, weight=2.0,
                             params={"robot_cfg": ROBOT, "tool_cfg": TOOL,
                                     "tcp_offset": TCP_OFFSET, "grasp_offset": GRASP_OFFSET})
    lifting = RewardTermCfg(func=R.lifting_tool, weight=5.0, params={"tool_cfg": TOOL})
    to_target = RewardTermCfg(func=R.tool_to_target, weight=3.0,
                              params={"std": 0.1, "tool_cfg": TOOL})
    action_rate = RewardTermCfg(func=R.action_rate_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func="mdp.time_out", time_out=True)
    dropped = TerminationTermCfg(func=T.tool_dropped, params={"tool_cfg": TOOL})
    success = TerminationTermCfg(func=T.reached_target, params={"tool_cfg": TOOL})


@configclass
class EventsCfg:
    reset_tool_pose = EventTermCfg(func="mdp.reset_root_state_uniform", mode="reset",
                                   params={"asset_cfg": TOOL})  # position varies, orientation fixed


# --------------------------------------------------------------------------- #
# Env
# --------------------------------------------------------------------------- #
@configclass
class JarvisToolEnvCfg(ManagerBasedRLEnvCfg):
    scene: JarvisSceneCfg = JarvisSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.action_scale = ACTION_SCALE
