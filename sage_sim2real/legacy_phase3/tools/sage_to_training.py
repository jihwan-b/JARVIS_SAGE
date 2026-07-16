"""sage_to_training.py — Phase 3 Task B: SAGE paired CSV -> training tensors.

Reads SAGE `output/{sim,real}/so101/custom/{motion}/` and emits `train.npz`/`val.npz` whose
feature rows are built by `feature_spec.build_features` (so train==deploy by construction).

Pipeline per motion:
  1. parse control.csv (commands) + state_motor.csv (measured), both for real (and sim if present)
  2. unit-convert SAGE -> deployment `.pos`:  arm rad->deg, gripper 0..1->0..100, vel rad/s->deg/s
  3. time-align: real timestamps us->s, start-zero, linear-interp resample to --target-rate (~30Hz),
     reusing SAGE's np.interp approach. Actuator lag is NOT erased (it is part of the signal we model).
  4. build (feature, residual) pairs from REAL data.
       residual target (MVP, guide §14 Task B):  g = real_actual - command
       applied at deploy as  u = a_desired - g_hat  (so the model predicts the gap to subtract).
  5. concat across motions, split train/val, compute normalization stats, save npz + metadata.

Run (env C):
  python tools/sage_to_training.py --sage-output output --motions pick_place oscillation_low_freq \
      --target-rate 30 --val-frac 0.2 --out-dir outputs

This script is plain numpy/pandas (no Isaac Sim) and runs in env C.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np

import feature_spec as fs

# SAGE unit constants -------------------------------------------------------
RAD2DEG = 180.0 / np.pi
# gripper: SAGE reports 0..1, deployment uses 0..100
GRIPPER_SCALE = 100.0


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def _parse_positions_cell(cell) -> list[float]:
    """SAGE stores `positions`/`velocities`/`torques` as stringified lists, e.g. "[0.1, -0.2, ...]".

    Be liberal: also accept space-separated or comma-separated bare numbers.
    """
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [float(x) for x in cell]
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return []
    try:
        val = ast.literal_eval(s)
        return [float(x) for x in val]
    except (ValueError, SyntaxError):
        s = s.strip("[]()")
        sep = "," if "," in s else None
        return [float(t) for t in s.split(sep) if t.strip() != ""]


def _read_csv_simple(path: Path) -> dict[str, list]:
    """Minimal CSV reader returning column-name -> list. Avoids a hard pandas dependency for the
    parse step (pandas is still imported above for resample convenience if available)."""
    import csv

    cols: dict[str, list] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for h in header:
            cols[h] = []
        for row in reader:
            for h, v in zip(header, row):
                cols[h].append(v)
    return cols


def _load_stream(path: Path, value_col: str):
    """Load a SAGE csv (control or state_motor); return (timestamps[s], values[N,6])."""
    cols = _read_csv_simple(path)
    ts_raw = np.array([float(t) for t in cols["timestamp"]], dtype=np.float64)
    rows = [_parse_positions_cell(c) for c in cols[value_col]]
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        raise ValueError(f"No parseable {value_col} in {path}")
    arr = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return ts_raw, arr


def _to_seconds_startzero(ts: np.ndarray, is_real: bool) -> np.ndarray:
    """real timestamps are microseconds, sim are seconds; both start-zeroed (guide §9 / analysis.py)."""
    t = ts.astype(np.float64)
    if is_real:
        t = t / 1e6  # us -> s
    return t - t[0]


# ---------------------------------------------------------------------------
# Unit conversion: SAGE (rad, gripper 0..1, rad/s) -> deployment .pos (deg, 0..100, deg/s)
# ---------------------------------------------------------------------------
def _sage_to_pos(arr: np.ndarray, is_velocity: bool = False) -> np.ndarray:
    """Convert an [N,6] SAGE array (joint order == SAGE_JOINT_ORDER, positionally == deploy order)
    into deployment `.pos` units. Arm joints rad->deg (or rad/s->deg/s); gripper 0..1 -> 0..100
    (positions only; velocity gripper channel is left in scaled units consistently)."""
    out = arr.astype(np.float64).copy()
    # arm channels
    for j in fs.ARM_INDICES:
        out[:, j] = out[:, j] * RAD2DEG
    # gripper channel
    g = fs.GRIPPER_INDEX
    if is_velocity:
        # gripper velocity: scale by the same 0..1 -> 0..100 factor so d(gripper)/dt is consistent
        out[:, g] = out[:, g] * GRIPPER_SCALE
    else:
        out[:, g] = out[:, g] * GRIPPER_SCALE
    return out


# ---------------------------------------------------------------------------
# Resampling onto a common grid at the deployment rate
# ---------------------------------------------------------------------------
def _resample(ts: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Per-channel linear interpolation onto `grid` (analysis.py np.interp style)."""
    out = np.empty((grid.shape[0], vals.shape[1]), dtype=np.float64)
    for j in range(vals.shape[1]):
        col = vals[:, j]
        good = ~np.isnan(col)
        if good.sum() < 2:
            out[:, j] = np.nan
            continue
        out[:, j] = np.interp(grid, ts[good], col[good])
    return out


def _common_grid(*ts_arrays: np.ndarray, rate_hz: float) -> np.ndarray:
    t_start = max(float(t[0]) for t in ts_arrays)
    t_end = min(float(t[-1]) for t in ts_arrays)
    if t_end <= t_start:
        raise ValueError("No overlapping time window across streams")
    n = int(np.floor((t_end - t_start) * rate_hz)) + 1
    return t_start + np.arange(n) / rate_hz


# ---------------------------------------------------------------------------
# Per-motion processing
# ---------------------------------------------------------------------------
def _finite_difference(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """qdot fallback if velocities are unavailable: dq/dt on the resampled grid (deg/s).
    Used only when USE_QDOT and the state_motor velocities column is missing."""
    dt = np.gradient(grid)
    out = np.empty_like(vals)
    for j in range(vals.shape[1]):
        out[:, j] = np.gradient(vals[:, j]) / dt
    return out


def process_motion(motion_dir: Path, rate_hz: float) -> dict[str, np.ndarray]:
    """Return resampled, unit-converted command + measured streams for ONE motion (real side).

    Returns dict with keys: t, a_cmd[N,6], q_real[N,6], qdot_real[N,6]|None
    """
    ctrl_path = motion_dir / "control.csv"
    state_path = motion_dir / "state_motor.csv"
    if not ctrl_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Missing control/state CSV in {motion_dir}")

    # commands (control.csv -> positions)
    ts_c, cmd = _load_stream(ctrl_path, "positions")
    ts_c = _to_seconds_startzero(ts_c, is_real=True)
    cmd = _sage_to_pos(cmd, is_velocity=False)

    # measured (state_motor.csv -> positions, velocities)
    ts_s, meas_pos = _load_stream(state_path, "positions")
    ts_s = _to_seconds_startzero(ts_s, is_real=True)
    meas_pos = _sage_to_pos(meas_pos, is_velocity=False)

    meas_vel = None
    if fs.USE_QDOT:
        try:
            ts_v, vel = _load_stream(state_path, "velocities")
            ts_v = _to_seconds_startzero(ts_v, is_real=True)
            vel = _sage_to_pos(vel, is_velocity=True)
            meas_vel = (ts_v, vel)
        except (KeyError, ValueError):
            meas_vel = None  # fall back to finite difference below

    # common grid across command + measured streams at the deployment rate
    grids = [ts_c, ts_s]
    if meas_vel is not None:
        grids.append(meas_vel[0])
    grid = _common_grid(*grids, rate_hz=rate_hz)

    a_cmd = _resample(ts_c, cmd, grid)
    q_real = _resample(ts_s, meas_pos, grid)

    qdot_real = None
    if fs.USE_QDOT:
        if meas_vel is not None:
            qdot_real = _resample(meas_vel[0], meas_vel[1], grid)
        else:
            qdot_real = _finite_difference(q_real, grid)

    return {"t": grid, "a_cmd": a_cmd, "q_real": q_real, "qdot_real": qdot_real}


def build_pairs(stream: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Turn one motion's resampled streams into (X feature rows, Y residual targets).

    residual target (MVP): g = real_actual - command, i.e. how far the robot ended up from what we
    asked. Deploy applies u = a_desired - g_hat.  prev_cmd is the previous step's command.
    """
    a = stream["a_cmd"]
    q = stream["q_real"]
    qd = stream["qdot_real"]
    n = a.shape[0]

    X = np.empty((n, fs.FEATURE_DIM), dtype=np.float32)
    Y = np.empty((n, fs.TARGET_DIM), dtype=np.float32)

    prev = a[0].copy()
    for i in range(n):
        qdot_i = qd[i] if (fs.USE_QDOT and qd is not None) else None
        X[i] = fs.build_features(a[i], q[i], qdot_real=qdot_i, prev_cmd=prev)
        Y[i] = (q[i] - a[i]).astype(np.float32)  # measured gap g
        prev = a[i]
    return X, Y


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="SAGE paired CSV -> training npz (Phase 3 Task B)")
    p.add_argument("--sage-output", type=Path, required=True,
                   help="SAGE output root containing real/so101/custom/<motion>/")
    p.add_argument("--robot-name", default="so101")
    p.add_argument("--side", default="real", choices=["real", "sim"],
                   help="which side provides the measured stream for residual targets (real)")
    p.add_argument("--motions", nargs="+", required=True,
                   help="motion names under custom/, e.g. pick_place oscillation_low_freq")
    p.add_argument("--target-rate", type=float, default=30.0, help="deployment loop rate (Hz)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    base = args.sage_output / args.side / args.robot_name / "custom"
    all_X, all_Y, per_motion = [], [], {}
    for m in args.motions:
        mdir = base / m
        stream = process_motion(mdir, rate_hz=args.target_rate)
        X, Y = build_pairs(stream)
        all_X.append(X)
        all_Y.append(Y)
        per_motion[m] = int(X.shape[0])
        print(f"[{m}] {X.shape[0]} samples @ {args.target_rate}Hz  feat={X.shape[1]}")

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    # shuffle + split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(X.shape[0])
    X, Y = X[idx], Y[idx]
    n_val = int(round(args.val_frac * X.shape[0]))
    Xv, Yv = X[:n_val], Y[:n_val]
    Xt, Yt = X[n_val:], Y[n_val:]

    # normalization stats from TRAIN only
    x_mean = Xt.mean(axis=0)
    x_std = Xt.std(axis=0) + 1e-6
    y_mean = Yt.mean(axis=0)
    y_std = Yt.std(axis=0) + 1e-6

    meta = fs.metadata()
    meta.update({
        "target_rate_hz": args.target_rate,
        "side": args.side,
        "motions": list(args.motions),
        "per_motion_samples": per_motion,
        "n_train": int(Xt.shape[0]),
        "n_val": int(Xv.shape[0]),
        "residual_target": "g = real_actual - command; deploy applies u = a_desired - g_hat",
    })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(meta)
    np.savez(args.out_dir / "train.npz", X=Xt, Y=Yt,
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
             feature_names=np.array(fs.FEATURE_NAMES), meta=meta_json)
    np.savez(args.out_dir / "val.npz", X=Xv, Y=Yv,
             x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
             feature_names=np.array(fs.FEATURE_NAMES), meta=meta_json)

    with open(args.out_dir / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved -> {args.out_dir}/train.npz ({Xt.shape}), val.npz ({Xv.shape})")
    print(f"feature_dim={fs.FEATURE_DIM} use_qdot={fs.USE_QDOT}")


if __name__ == "__main__":
    main()
