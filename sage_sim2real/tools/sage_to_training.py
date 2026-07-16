"""sage_to_training.py — Phase 3 Task B: SAGE paired CSV -> training tensors (LEAKAGE-FIXED).

Reads SAGE `output/{real}/so101/custom/{motion}/` and emits `train.npz`/`val.npz` whose feature rows
are built by the corrected `feature_spec.build_features` (command-side only, so train==deploy and the
target can't leak into the input — see feature_spec.py for the why).

Pipeline per motion:
  1. parse control.csv (commands) + state_motor.csv (measured positions)
  2. unit-convert SAGE -> deployment `.pos`:  arm rad->deg, gripper 0..1->0..100
  3. time-align: real timestamps us->s, start-zero, linear-interp resample to --target-rate (~30Hz),
     reusing SAGE's np.interp approach. Actuator lag is NOT erased (it is part of the signal we model).
  4. build (feature, residual) pairs:
       features = build_features(a[i], prev_cmd=a[i-1])   <- command-side only
       target   = g[i] = q_measured[i] - a[i]             <- the gap; NEVER a feature
     Deploy applies u = a_desired - g_hat.
  5. concat across motions, split train/val, compute normalization stats, save npz + metadata.

We no longer read the velocities column or do measured-qdot finite-difference: the command step
`delta = a - prev` (a fixed-rate velocity proxy) carries the dynamic cue instead.

Run (env C):
  python tools/sage_to_training.py --sage-output output_bong \
      --motions pick_place oscillation_low_freq random_waypoints actuator_bandwidth backlash_detection \
      --target-rate 30 --val-frac 0.2 --out-dir runs/bong_v2
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

import feature_spec as fs

RAD2DEG = 180.0 / np.pi
GRIPPER_SCALE = 100.0  # SAGE gripper 0..1 -> deployment 0..100


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def _parse_positions_cell(cell) -> list[float]:
    """SAGE stores `positions` as a stringified list, e.g. "[0.1, -0.2, ...]"."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [float(x) for x in cell]
    s = str(cell).strip()
    if s == "" or s.lower() == "nan":
        return []
    try:
        return [float(x) for x in ast.literal_eval(s)]
    except (ValueError, SyntaxError):
        s = s.strip("[]()")
        sep = "," if "," in s else None
        return [float(t) for t in s.split(sep) if t.strip() != ""]


def _read_csv_simple(path: Path) -> dict[str, list]:
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


def _load_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a SAGE csv; return (timestamps[s, raw], values[N,6])."""
    cols = _read_csv_simple(path)
    ts_raw = np.array([float(t) for t in cols["timestamp"]], dtype=np.float64)
    rows = [_parse_positions_cell(c) for c in cols["positions"]]
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        raise ValueError(f"No parseable positions in {path}")
    arr = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    return ts_raw, arr


def _to_seconds_startzero(ts: np.ndarray) -> np.ndarray:
    """real timestamps are microseconds (guide §9); us->s and start-zeroed."""
    t = ts.astype(np.float64) / 1e6
    return t - t[0]


def _sage_pos_to_deploy(arr: np.ndarray) -> np.ndarray:
    """[N,6] SAGE positions -> deployment `.pos`: arm rad->deg, gripper 0..1->0..100."""
    out = arr.astype(np.float64).copy()
    for j in fs.ARM_INDICES:
        out[:, j] = out[:, j] * RAD2DEG
    out[:, fs.GRIPPER_INDEX] = out[:, fs.GRIPPER_INDEX] * GRIPPER_SCALE
    return out


# ---------------------------------------------------------------------------
# Resampling onto a common grid at the deployment rate
# ---------------------------------------------------------------------------
def _resample(ts: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.empty((grid.shape[0], vals.shape[1]), dtype=np.float64)
    for j in range(vals.shape[1]):
        col = vals[:, j]
        good = ~np.isnan(col)
        out[:, j] = np.interp(grid, ts[good], col[good]) if good.sum() >= 2 else np.nan
    return out


def _common_grid(ts_a: np.ndarray, ts_b: np.ndarray, rate_hz: float) -> np.ndarray:
    t_start = max(float(ts_a[0]), float(ts_b[0]))
    t_end = min(float(ts_a[-1]), float(ts_b[-1]))
    if t_end <= t_start:
        raise ValueError("No overlapping time window across command/measured streams")
    n = int(np.floor((t_end - t_start) * rate_hz)) + 1
    return t_start + np.arange(n) / rate_hz


def process_motion(motion_dir: Path, rate_hz: float) -> dict[str, np.ndarray]:
    """Return resampled, unit-converted command + measured streams for ONE motion (real side)."""
    ctrl_path = motion_dir / "control.csv"
    state_path = motion_dir / "state_motor.csv"
    if not ctrl_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Missing control/state CSV in {motion_dir}")

    ts_c, cmd = _load_positions(ctrl_path)
    ts_c = _to_seconds_startzero(ts_c)
    cmd = _sage_pos_to_deploy(cmd)

    ts_s, meas = _load_positions(state_path)
    ts_s = _to_seconds_startzero(ts_s)
    meas = _sage_pos_to_deploy(meas)

    grid = _common_grid(ts_c, ts_s, rate_hz=rate_hz)
    a_cmd = _resample(ts_c, cmd, grid)
    q_real = _resample(ts_s, meas, grid)
    return {"t": grid, "a_cmd": a_cmd, "q_real": q_real}


def build_pairs(stream: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Resampled streams -> (X command-side features, Y residual gap targets).

    target g[i] = q_measured[i] - a[i].  features use ONLY a[i] and a[i-1] (prev). q never enters X.
    """
    a = stream["a_cmd"]
    q = stream["q_real"]
    n = a.shape[0]

    X = np.empty((n, fs.FEATURE_DIM), dtype=np.float32)
    Y = np.empty((n, fs.TARGET_DIM), dtype=np.float32)

    prev = a[0].copy()
    for i in range(n):
        X[i] = fs.build_features(a[i], prev_cmd=prev)
        Y[i] = (q[i] - a[i]).astype(np.float32)  # measured gap g  (target only, not a feature)
        prev = a[i]
    return X, Y


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="SAGE paired CSV -> training npz (Phase 3 Task B, fixed)")
    p.add_argument("--sage-output", type=Path, required=True,
                   help="SAGE output root containing real/so101/custom/<motion>/")
    p.add_argument("--robot-name", default="so101")
    p.add_argument("--side", default="real", choices=["real", "sim"])
    p.add_argument("--motions", nargs="+", required=True)
    p.add_argument("--target-rate", type=float, default=30.0, help="deployment loop rate (Hz)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    base = args.sage_output / args.side / args.robot_name / "custom"
    # Build features per motion AND split each motion CHRONOLOGICALLY (first 1-val_frac -> train,
    # last val_frac -> val). SAGE is collected at 50Hz and resampled to ~30Hz, so adjacent frames
    # are nearly identical; a random shuffle-then-split scatters neighbours of every val frame into
    # train, which the model memorizes -> inflated val score (temporal leakage). Holding out the
    # TAIL of each motion keeps train/val frames temporally disjoint, so val is genuinely unseen.
    all_X, all_Y = [], []
    Xt_parts, Yt_parts, Xv_parts, Yv_parts = [], [], [], []
    per_motion = {}
    for m in args.motions:
        stream = process_motion(base / m, rate_hz=args.target_rate)
        X, Y = build_pairs(stream)
        all_X.append(X)
        all_Y.append(Y)
        per_motion[m] = int(X.shape[0])
        n = X.shape[0]
        k = int(round((1.0 - args.val_frac) * n))   # chronological cut for THIS motion
        Xt_parts.append(X[:k]); Yt_parts.append(Y[:k])
        Xv_parts.append(X[k:]); Yv_parts.append(Y[k:])
        print(f"[{m}] {n} samples @ {args.target_rate}Hz  feat={X.shape[1]}  "
              f"(train {k} / val {n - k}, chronological)")

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    # --- leak check 1 (TARGET leakage): no feature block may equal the target g. ---
    # Every block is width N_JOINTS, so reshape to (samples, n_blocks, N_JOINTS) and compare to Y.
    blocks = X.reshape(X.shape[0], len(fs.FEATURE_BLOCKS), fs.N_JOINTS)
    block_gap = np.abs(blocks - Y[:, None, :]).mean(axis=(0, 2))  # mean |block - Y| per block
    closest = int(np.argmin(block_gap))
    assert block_gap.min() > 1e-2, (
        f"feature block '{fs.FEATURE_BLOCKS[closest][0]}' matches target g (target leakage): "
        f"{dict(zip([b for b, _ in fs.FEATURE_BLOCKS], block_gap.round(4)))}"
    )
    print(f"leak check 1 (target) OK — min mean|block-Y| = {block_gap.min():.3f} "
          f"(block '{fs.FEATURE_BLOCKS[closest][0]}')")

    # assemble chronological split (shuffle ONLY within train, never across the train/val boundary)
    Xt = np.concatenate(Xt_parts, axis=0)
    Yt = np.concatenate(Yt_parts, axis=0)
    Xv = np.concatenate(Xv_parts, axis=0)
    Yv = np.concatenate(Yv_parts, axis=0)
    rng = np.random.default_rng(args.seed)
    tr_perm = rng.permutation(Xt.shape[0])
    Xt, Yt = Xt[tr_perm], Yt[tr_perm]

    # --- leak check 2 (TEMPORAL leakage guard): warn if any val row has a near-duplicate in train.
    # With a chronological split this should be ~0; a random split would light this up. Cheap check
    # on a random subsample of val rows (nearest train row by max-abs feature distance).
    if Xv.shape[0] > 0:
        rs = rng.choice(Xv.shape[0], size=min(200, Xv.shape[0]), replace=False)
        dmin = np.empty(rs.shape[0])
        for i, vi in enumerate(rs):
            dmin[i] = np.abs(Xt - Xv[vi]).max(axis=1).min()
        n_dup = int((dmin < 1e-3).sum())
        print(f"leak check 2 (temporal) — {n_dup}/{rs.shape[0]} val rows have a near-duplicate "
              f"in train (want ~0; chronological split keeps this low)")

    x_mean, x_std = Xt.mean(axis=0), Xt.std(axis=0) + 1e-6
    y_mean, y_std = Yt.mean(axis=0), Yt.std(axis=0) + 1e-6

    meta = fs.metadata()
    meta.update({
        "target_rate_hz": args.target_rate,
        "side": args.side,
        "motions": list(args.motions),
        "per_motion_samples": per_motion,
        "n_train": int(Xt.shape[0]),
        "n_val": int(Xv.shape[0]),
        "residual_target": "g = real_actual - command; deploy applies u = a_desired - g_hat",
        "split": "per-motion chronological (tail %.0f%% of each motion -> val; train shuffled within)"
                 % (args.val_frac * 100),
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
    print(f"saved -> {args.out_dir}/train.npz ({Xt.shape}), val.npz ({Xv.shape})  feature_dim={fs.FEATURE_DIM}")


if __name__ == "__main__":
    main()
