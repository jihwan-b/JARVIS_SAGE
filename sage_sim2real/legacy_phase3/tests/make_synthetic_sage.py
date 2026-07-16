"""make_synthetic_sage.py — generate SAGE-format CSVs with a KNOWN injected actuator gap.

This is a TEST FIXTURE, not part of the deliverable. It lets us validate the Phase 3 pipeline
end-to-end before real SO-101 paired data exists: we inject a known static bias + direction-
dependent backlash, then check the trainer recovers a model that reduces the residual RMSE.

Writes SAGE schema:  output/real/so101/custom/<motion>/{control.csv,state_motor.csv,joint_list.txt}
- control.csv:    type,timestamp,positions      (radians, timestamp microseconds)
- state_motor.csv: type,timestamp,positions,velocities,torques
- joint_list.txt:  Rotation,Pitch,Elbow,Wrist_Pitch,Wrist_Roll,Jaw
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

SAGE_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
DEG2RAD = np.pi / 180.0

# Known gap to inject, in .pos units (deg for arm, 0..100 for gripper) then converted to SAGE units.
STATIC_BIAS_DEG = np.array([0.8, -1.2, 1.5, -0.6, 0.4, 2.0])      # constant offset
BACKLASH_DEG    = np.array([1.0,  1.4, 0.9,  0.5, 0.3, 0.0])      # extra error when reversing dir


def gen_motion(n: int, rate_hz: float, seed: int):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate_hz
    # commanded arm trajectory in DEGREES (sinusoids w/ reversals), gripper in 0..100
    cmd_deg = np.zeros((n, 6))
    for j in range(5):
        cmd_deg[:, j] = 25.0 * np.sin(2 * np.pi * (0.2 + 0.1 * j) * t + j)
    cmd_deg[:, 5] = 50.0 + 50.0 * np.sin(2 * np.pi * 0.15 * t)  # gripper 0..100

    # measured = command + static_bias + backlash*(direction reversed) + small noise
    dcmd = np.diff(cmd_deg, axis=0, prepend=cmd_deg[:1])
    direction = np.sign(dcmd)
    reversed_dir = np.zeros_like(direction)
    reversed_dir[1:] = (direction[1:] * direction[:-1] < 0).astype(float)
    meas_deg = (cmd_deg
                + STATIC_BIAS_DEG
                + BACKLASH_DEG * reversed_dir * np.sign(dcmd)
                + rng.normal(0, 0.05, size=cmd_deg.shape))
    vel_deg_s = np.gradient(meas_deg, axis=0) * rate_hz
    return t, cmd_deg, meas_deg, vel_deg_s


def pos_deg_to_sage(arr_deg: np.ndarray) -> np.ndarray:
    """deg arm + 0..100 gripper  ->  rad arm + 0..1 gripper (SAGE units)."""
    out = arr_deg.astype(float).copy()
    out[:, :5] *= DEG2RAD
    out[:, 5] /= 100.0
    return out


def vel_deg_to_sage(arr_deg_s: np.ndarray) -> np.ndarray:
    out = arr_deg_s.astype(float).copy()
    out[:, :5] *= DEG2RAD
    out[:, 5] /= 100.0
    return out


def write_list_cell(row) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in row) + "]"


def write_motion(root: Path, motion: str, n: int, rate_hz: float, seed: int):
    t, cmd_deg, meas_deg, vel_deg_s = gen_motion(n, rate_hz, seed)
    cmd_sage = pos_deg_to_sage(cmd_deg)
    meas_sage = pos_deg_to_sage(meas_deg)
    vel_sage = vel_deg_to_sage(vel_deg_s)
    torque = np.zeros((n, 6))
    ts_us = (t * 1e6).astype(np.int64)  # microseconds

    d = root / "real" / "so101" / "custom" / motion
    d.mkdir(parents=True, exist_ok=True)
    (d / "joint_list.txt").write_text(",".join(SAGE_JOINTS) + "\n")

    with open(d / "control.csv", "w") as f:
        f.write("type,timestamp,positions\n")
        for i in range(n):
            f.write(f"CONTROL,{ts_us[i]},\"{write_list_cell(cmd_sage[i])}\"\n")

    with open(d / "state_motor.csv", "w") as f:
        f.write("type,timestamp,positions,velocities,torques\n")
        for i in range(n):
            f.write(f"STATE,{ts_us[i]},\"{write_list_cell(meas_sage[i])}\","
                    f"\"{write_list_cell(vel_sage[i])}\",\"{write_list_cell(torque[i])}\"\n")
    print(f"wrote {motion}: {n} rows -> {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("output"))
    ap.add_argument("--motions", nargs="+", default=["pick_place", "oscillation_low_freq"])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--rate", type=float, default=50.0)  # SAGE collects at 50 Hz
    args = ap.parse_args()
    for k, m in enumerate(args.motions):
        write_motion(args.root, m, args.n, args.rate, seed=k)
    print("injected STATIC_BIAS_DEG =", STATIC_BIAS_DEG.tolist())
    print("injected BACKLASH_DEG    =", BACKLASH_DEG.tolist())


if __name__ == "__main__":
    main()
