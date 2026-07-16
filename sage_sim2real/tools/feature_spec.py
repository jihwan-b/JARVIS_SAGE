"""feature_spec.py — single source of truth for the compensator feature layout (LEAKAGE-FIXED).

Per the integration guide (§10 / §14 Task A): the converter (`sage_to_training.py`), the trainer
(`train_compensator.py`), AND `actuator_compensator._build_features` MUST all import this module so
the training-time and deployment-time feature vectors are byte-for-byte identical.

WHY THIS REPLACES THE ORIGINAL `phase 3 from other one/tools/feature_spec.py`:
    The original layout was [a, q, qdot, q-a, dir, prev] with target Y = q - a. But `q-a` (and `q`
    itself) is the realized tracking error, which IS the target — so the MLP trivially copied it
    (verified: max|X[q_minus_a] - Y| = 0.0) and reported a fake ~99% RMSE improvement. Worse, at
    deploy `q` is the PRE-command position (the realized outcome doesn't exist yet), so the feature
    means something different than at train time and the corrector would emit u = 2a - q_now.

THE FIX (command-side features only):
    Predict the gap from information available BEFORE the command is sent — the command itself, a
    1-step command history, and the commanded motion direction/step. No realized measured state.
        features = [a, prev_cmd, delta, dir]   (24-dim)
            a        : command being sent this step
            prev_cmd : previous command (1-step history)
            delta    : a - prev_cmd  (commanded step / tick; velocity proxy at fixed rate -> lag cue)
            dir      : sign(delta)    (direction-dependent backlash cue)
        target   = g = q_realized - a   (unchanged; built by the converter, NEVER a feature)
    Because the control rate is fixed (~30 Hz) at train and deploy, `delta` is proportional to the
    commanded velocity, so we don't need a measured-velocity channel — which also dissolves the §12
    Risk-1 deploy-qdot problem (no Present_Velocity read, no finite-diff-of-measured noise).

Units: lerobot `.pos` (arm DEGREES, gripper 0..100) — the DEPLOYMENT convention from `so101_eval.py`.
The converter harmonizes SAGE rad/0..1/50Hz into these units before calling build_features.

Joint order (deployment keys): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
SAGE joint_list.txt order (positionally identical): Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw
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
# Feature layout — command-side only (single place that defines it)
# ----------------------------------------------------------------------------
FEATURE_BLOCKS: list[tuple[str, int]] = [
    ("a", N_JOINTS),      # command this step
    ("prev", N_JOINTS),   # previous command
    ("delta", N_JOINTS),  # a - prev  (commanded step / velocity proxy)
    ("dir", N_JOINTS),    # sign(delta)  (backlash direction cue)
]
FEATURE_DIM: int = sum(w for _, w in FEATURE_BLOCKS)  # 24
TARGET_DIM: int = N_JOINTS  # residual gap per joint

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


def build_features(a_desired, prev_cmd=None, eps: float = 1e-6) -> np.ndarray:
    """Construct ONE feature row, identical at train and deploy time.

    Args:
        a_desired: command targets this step, `.pos` units (arm deg, gripper 0..100), shape (6,).
        prev_cmd:  previous command, `.pos` units, shape (6,). None -> falls back to a_desired
                   (i.e. "no motion yet" at the first step, so delta/dir are 0).
        eps:       below this |delta| the direction is treated as 0 (no reversal).

    Returns:
        np.ndarray shape (FEATURE_DIM,) == (24,), dtype float32.

    NOTE: there is deliberately NO measured-position / measured-velocity argument. The realized
    state is the thing we predict; feeding it in is the leak this module was rewritten to remove.
    """
    a = _as_vec(a_desired, "a_desired")
    prev = a.copy() if prev_cmd is None else _as_vec(prev_cmd, "prev_cmd")

    delta = a - prev
    direction = np.sign(delta)
    direction[np.abs(delta) < eps] = 0.0

    feat = np.concatenate([a, prev, delta, direction]).astype(np.float32)
    assert feat.shape[0] == FEATURE_DIM, (feat.shape, FEATURE_DIM)
    return feat


def metadata() -> dict:
    """Layout metadata, embedded into train.npz so the trainer/deploy can assert agreement."""
    return {
        "feature_dim": FEATURE_DIM,
        "target_dim": TARGET_DIM,
        "deploy_joint_order": list(DEPLOY_JOINT_ORDER),
        "sage_joint_order": list(SAGE_JOINT_ORDER),
        "feature_blocks": [list(b) for b in FEATURE_BLOCKS],
        "units": "lerobot .pos (arm degrees, gripper 0..100)",
        "feature_side": "command-only (no realized measured state; leakage-free)",
    }


if __name__ == "__main__":
    import json

    a = np.array([10.0, -5.0, 12.0, -3.0, 2.0, 60.0])
    prev = a - np.array([1.0, 1.0, -1.0, 0.0, 0.5, 5.0])
    f = build_features(a, prev_cmd=prev)
    print(json.dumps(metadata(), indent=2))
    print("feature_dim:", f.shape[0], "names[:8]:", FEATURE_NAMES[:8])
