"""feature_spec.py — single source of truth for the compensator feature layout.

Per the integration guide (§10 / §14 Task A): the converter (`sage_to_training.py`),
the trainer (`train_compensator.py`), AND `actuator_compensator._build_features` MUST all
import this module so that the training-time and deployment-time feature vectors are byte-for-byte
identical. Any layout change happens here and propagates to all three.

Units: lerobot `.pos` (arm DEGREES, gripper 0..100) — the DEPLOYMENT convention from
`so101_eval.py`, NOT SAGE radians. The converter is responsible for harmonizing SAGE rad/0..1/50Hz
into these units; this module assumes its inputs are already in `.pos` units.

Joint order (deployment keys, from so101_eval.py):
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

NOTE on the SAGE joint order: SAGE `joint_list.txt` uses
    Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw
which is the SAME physical ordering as the deployment keys above (pan->Rotation,
lift->Pitch, elbow->Elbow, wrist_flex->Wrist_Pitch, wrist_roll->Wrist_Roll, gripper->Jaw).
The converter maps SAGE columns -> deployment keys using DEPLOY_JOINT_ORDER / SAGE_JOINT_ORDER.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Joint ordering (the ONE canonical order used everywhere downstream)
# ----------------------------------------------------------------------------
DEPLOY_JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# SAGE joint_list.txt order, positionally aligned to DEPLOY_JOINT_ORDER above.
SAGE_JOINT_ORDER: tuple[str, ...] = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
)

N_JOINTS: int = len(DEPLOY_JOINT_ORDER)  # 6
GRIPPER_INDEX: int = DEPLOY_JOINT_ORDER.index("gripper")  # 5
ARM_INDICES: tuple[int, ...] = tuple(i for i in range(N_JOINTS) if i != GRIPPER_INDEX)

# ----------------------------------------------------------------------------
# qdot policy switch (guide §14 Task A: decide qdot inclusion, keep train==deploy)
# ----------------------------------------------------------------------------
# True  -> 36-dim layout: [a, q, qdot, q-a, dir, prev]   (guide's current layout; qdot from
#          Feetech Present_Velocity at deploy — recommended only if deploy reads velocity)
# False -> 30-dim layout: [a, q, q-a, dir, prev]          (qdot-free; safest MVP since the
#          so101_eval obs has no velocity channel)
#
# Whatever this is set to, the converter, trainer, and compensator all read it from here, so
# the two sides can never silently disagree.
USE_QDOT: bool = True

# Feature block order. Each entry is (name, width). Built dynamically from USE_QDOT so there is
# exactly one place that defines the layout.
def _blocks() -> list[tuple[str, int]]:
    blocks = [("a", N_JOINTS), ("q", N_JOINTS)]
    if USE_QDOT:
        blocks.append(("qdot", N_JOINTS))
    blocks += [("q_minus_a", N_JOINTS), ("dir", N_JOINTS), ("prev", N_JOINTS)]
    return blocks


FEATURE_BLOCKS: list[tuple[str, int]] = _blocks()
FEATURE_DIM: int = sum(w for _, w in FEATURE_BLOCKS)  # 36 if USE_QDOT else 30
TARGET_DIM: int = N_JOINTS  # residual per joint

# Human-readable feature names, useful for debugging / per-feature stats.
FEATURE_NAMES: list[str] = [
    f"{block}_{DEPLOY_JOINT_ORDER[j]}"
    for block, _ in FEATURE_BLOCKS
    for j in range(N_JOINTS)
]


def _as_vec(x, name: str) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.shape[0] != N_JOINTS:
        raise ValueError(f"{name} must have {N_JOINTS} elements, got {v.shape[0]}")
    return v


def build_features(
    a_desired,
    q_real,
    qdot_real=None,
    prev_cmd=None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Construct ONE feature row, identical at train and deploy time.

    Args:
        a_desired:  desired/commanded joint targets this step, `.pos` units, shape (6,)
        q_real:     measured joint positions this step, `.pos` units, shape (6,)
        qdot_real:  measured joint velocities, `.pos`/s, shape (6,). Required iff USE_QDOT.
                    Ignored when USE_QDOT is False.
        prev_cmd:   previously sent command, `.pos` units, shape (6,). None -> falls back to
                    a_desired (i.e. assume "no change yet" at the first step).
        eps:        threshold under which a movement direction is treated as 0 (no reversal).

    Returns:
        np.ndarray shape (FEATURE_DIM,), dtype float32.
    """
    a = _as_vec(a_desired, "a_desired")
    q = _as_vec(q_real, "q_real")
    prev = a.copy() if prev_cmd is None else _as_vec(prev_cmd, "prev_cmd")

    # direction sign of the *intended* motion this step (a - prev). Backlash on the SO-101
    # hobby servos is direction-dependent, so the sign of the commanded move is a key feature.
    delta = a - prev
    direction = np.sign(delta)
    direction[np.abs(delta) < eps] = 0.0

    q_minus_a = q - a

    parts = [a, q]
    if USE_QDOT:
        if qdot_real is None:
            raise ValueError("USE_QDOT is True but qdot_real was not provided")
        parts.append(_as_vec(qdot_real, "qdot_real"))
    parts += [q_minus_a, direction, prev]

    feat = np.concatenate(parts).astype(np.float32)
    assert feat.shape[0] == FEATURE_DIM, (feat.shape, FEATURE_DIM)
    return feat


def metadata() -> dict:
    """Layout metadata, embedded into train.npz so the trainer/deploy can assert agreement."""
    return {
        "feature_dim": FEATURE_DIM,
        "target_dim": TARGET_DIM,
        "use_qdot": USE_QDOT,
        "deploy_joint_order": list(DEPLOY_JOINT_ORDER),
        "sage_joint_order": list(SAGE_JOINT_ORDER),
        "feature_blocks": [list(b) for b in FEATURE_BLOCKS],
        "units": "lerobot .pos (arm degrees, gripper 0..100)",
    }


if __name__ == "__main__":
    # quick self-check
    import json

    a = np.zeros(6)
    q = np.full(6, 1.5)
    qd = np.zeros(6) if USE_QDOT else None
    f = build_features(a, q, qdot_real=qd, prev_cmd=np.full(6, -0.5))
    print(json.dumps(metadata(), indent=2))
    print("feature_dim:", f.shape[0], "names[:8]:", FEATURE_NAMES[:8])
