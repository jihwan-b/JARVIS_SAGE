"""
SmolVLA → jarvis_env action wrapper.

SmolVLA is trained/operated in **degree absolute joint targets** (the LeRobot
SO-101 convention), while the Isaac Lab `jarvis_env` PPO policy consumes
**radian, scaled relative RL actions**. This wrapper performs the 3-step
conversion between the two so a fine-tuned SmolVLA checkpoint can be deployed
directly inside the RL simulation environment.

Joint order (SO-101, 6 DoF):
    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

Conversion (per control step):
    1. degree → radian          : q_rad = deg2rad(q_deg)
    2. arm relative scaled action: a_arm = (q_rad - default_rad) / ACTION_SCALE
    3. gripper threshold         : a_grip = +1 (close) if grip_deg <= GRIP_CLOSE_DEG else -1 (open)
"""

from __future__ import annotations

import numpy as np

# SO-101 home pose in radians (matches rl3 default_joint_pos)
DEFAULT_JOINT_POS_RAD = np.array([0.0, 0.0, -0.0, 1.57, -0.0, 0.0], dtype=np.float32)

# RL action scale used at training time (raw action → joint delta)
ACTION_SCALE = 0.5

# Gripper decision threshold, in degrees. SmolVLA emits an absolute gripper
# angle; below this we treat the command as "close".
GRIP_CLOSE_DEG = 10.0

ARM_IDX = slice(0, 5)   # first 5 joints are the arm
GRIP_IDX = 5            # last joint is the gripper


def smolvla_deg_to_rl_action(
    smolvla_targets_deg: np.ndarray,
    default_joint_pos_rad: np.ndarray = DEFAULT_JOINT_POS_RAD,
    action_scale: float = ACTION_SCALE,
    grip_close_deg: float = GRIP_CLOSE_DEG,
) -> np.ndarray:
    """Convert a SmolVLA degree-target vector into a jarvis_env RL action.

    Args:
        smolvla_targets_deg: shape (6,) absolute joint targets in degrees.
        default_joint_pos_rad: shape (6,) home pose in radians.
        action_scale: training-time raw-action scale.
        grip_close_deg: gripper close threshold in degrees.

    Returns:
        rl_action: shape (6,) action consumable by jarvis_env.
    """
    q = np.asarray(smolvla_targets_deg, dtype=np.float32).reshape(-1)
    assert q.shape[0] == 6, f"expected 6 joint targets, got {q.shape[0]}"

    q_rad = np.deg2rad(q)

    rl_action = np.zeros(6, dtype=np.float32)
    # arm: relative-to-home, scaled
    rl_action[ARM_IDX] = (q_rad[ARM_IDX] - default_joint_pos_rad[ARM_IDX]) / action_scale
    # gripper: binary close/open from the degree command
    rl_action[GRIP_IDX] = 1.0 if q[GRIP_IDX] <= grip_close_deg else -1.0
    return rl_action


def batch_convert(targets_deg: np.ndarray, **kw) -> np.ndarray:
    """Vectorized conversion for a (T, 6) trajectory of degree targets."""
    targets_deg = np.asarray(targets_deg, dtype=np.float32)
    return np.stack([smolvla_deg_to_rl_action(t, **kw) for t in targets_deg], axis=0)


if __name__ == "__main__":
    # smoke test: home pose with open gripper → near-zero arm action, open grip
    home_deg = np.rad2deg(DEFAULT_JOINT_POS_RAD).copy()
    home_deg[GRIP_IDX] = 30.0  # open
    print("home →", smolvla_deg_to_rl_action(home_deg))

    # gripper closed
    home_deg[GRIP_IDX] = 5.0
    print("grasp →", smolvla_deg_to_rl_action(home_deg))
